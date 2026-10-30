"""End-to-end experiment runner for GeoCompass.

The runner is deterministic, CPU-only, and produces a single JSON artifact plus
CSV tables.  Every number in the generated report comes from this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time

from . import stats
from .bandit import build_policy, reward_from_utility
from .conformal import ConformalGate, fixed_threshold_gate, negative_transfer
from .features import FEATURE_NAMES, extract_features
from .memory import ExperienceMemory
from .planner import Retriever, clear_caches, oracle_run, run_skill
from .raster import generate_scene
from .retrieval import BM25Retriever, HybridRetriever
from .selector import (
    CURATED_FEATURES,
    STRUCTURAL_FEATURES,
    CalibratedSelector,
    FixedSelector,
    LogisticSelector,
)
from .tasks import FAMILIES, build_library, build_task_specs, generic_pipeline, is_success
from .transfer import diagnose, gate_scores
from .types import Outcome, Scene, Skill, TaskSpec

CONFORMAL_ALPHAS = (0.10, 0.20, 0.30)
SENSITIVITY_K = (4, 6, 8, 12)

DEFAULT_COUNTS = {
    "source_train": 10,
    "source_val": 4,
    "target_dev": 8,
    "target_test": 6,
    "stream": 8,
}


@dataclass
class Config:
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    counts: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_COUNTS))
    k: int = 8
    budget: int = 3
    memory_mode: str = "full"
    output: str = "experiments/geocompass"
    include_ablations: bool = True
    retriever: str = "hashed"
    dataset: str | None = None

    def to_dict(self) -> dict:
        return {
            "seeds": list(self.seeds),
            "counts": dict(self.counts),
            "k": self.k,
            "budget": self.budget,
            "memory_mode": self.memory_mode,
            "include_ablations": self.include_ablations,
            "retriever": self.retriever,
            "dataset": self.dataset,
        }


def build_candidate_source(kind: str, library: list[Skill]) -> object:
    """First stage used for the main pipeline; ablations build their own."""
    if kind == "bm25":
        return BM25Retriever(library)
    if kind == "hybrid":
        return HybridRetriever(library)
    if kind == "hashed":
        return Retriever(library)
    raise ValueError(f"unknown retriever: {kind}")


# --------------------------------------------------------------------------- data


def _make_data(seed: int, config: Config) -> tuple[dict[str, list[TaskSpec]], dict[str, Scene], list[Skill]]:
    if config.dataset:
        from .dataset import load_dataset

        return load_dataset(config.dataset, seed)
    splits = build_task_specs(seed, config.counts)
    scenes = {}
    for tasks in splits.values():
        for task in tasks:
            scenes[task.task_id] = generate_scene(task)
    library = build_library(splits["source_train"])
    return splits, scenes, library


def _outcome(task: TaskSpec, skill: Skill, result, step: int) -> Outcome:
    return Outcome(
        task_id=task.task_id,
        task_family=task.family,
        target=task.target,
        skill_id=skill.skill_id,
        kind="pipeline",
        success=bool(result.success),
        metric=float(result.score),
        baseline_metric=float(result.baseline_score),
        utility=float(result.utility),
        step=step,
        extra={"signature": skill.signature, "fallback": bool(result.fallback)},
    )


def _candidate_records(task: TaskSpec, scene: Scene, retriever: Retriever,
                       k: int) -> list[dict]:
    records = []
    for rank, (skill, semantic) in enumerate(retriever.candidates(task, k=k)):
        result = run_skill(task, scene, skill)
        records.append({
            "task": task,
            "skill": skill,
            "semantic": semantic,
            "rank": rank,
            "result": result,
            "label": int(result.success),
            "score": result.score,
            "utility": result.utility,
        })
    return records


def _all_records(tasks: list[TaskSpec], scenes: dict[str, Scene],
                 retriever: Retriever, k: int) -> list[dict]:
    records = []
    for task in tasks:
        records.extend(_candidate_records(task, scenes[task.task_id], retriever, k))
    return records


def _rows_from_records(records: list[dict], memory: ExperienceMemory,
                       candidate_count: int = 8, ingest: bool = True,
                       step_start: int = 0) -> list[dict]:
    rows = []
    step = step_start
    for record in records:
        task, skill = record["task"], record["skill"]
        stats_row = memory.stats(skill.skill_id, task.family)
        features = extract_features(
            task=task,
            skill=skill,
            stage="pipeline",
            semantic=record["semantic"],
            semantic_rank=record["rank"],
            candidate_count=candidate_count,
            memory=stats_row,
            step=step,
            library_size=20,
        )
        rows.append({
            "features": features,
            "label": int(record["label"]),
            "task_id": task.task_id,
            "family": task.family,
            "skill_id": skill.skill_id,
            "success": bool(record["result"].success),
            "utility": float(record["result"].utility),
            "score": float(record["result"].score),
            "step": step,
        })
        if ingest:
            memory.ingest(_outcome(task, skill, record["result"], step))
            step += 1
    return rows


def _rows_for_tasks(tasks: list[TaskSpec], scenes: dict[str, Scene], retriever: Retriever,
                    memory: ExperienceMemory, k: int, ingest: bool = False,
                    step_start: int = 0) -> list[dict]:
    rows = []
    step = step_start
    for task in tasks:
        records = _candidate_records(task, scenes[task.task_id], retriever, k)
        count = len(records)
        for record in records:
            skill = record["skill"]
            stats_row = memory.stats(skill.skill_id, task.family)
            features = extract_features(
                task=task,
                skill=skill,
                stage="pipeline",
                semantic=record["semantic"],
                semantic_rank=record["rank"],
                candidate_count=count,
                memory=stats_row,
                step=step,
                library_size=len(retriever.library),
            )
            rows.append({
                "features": features,
                "label": int(record["label"]),
                "task_id": task.task_id,
                "family": task.family,
                "skill_id": skill.skill_id,
                "success": bool(record["result"].success),
                "utility": float(record["result"].utility),
                "score": float(record["result"].score),
                "step": step,
            })
            if ingest:
                memory.ingest(_outcome(task, skill, record["result"], step))
                step += 1
    return rows


# ---------------------------------------------------------------------- selectors


def _fit_selector(train_rows: list[dict], calib_rows: list[dict],
                  features: tuple[str, ...] | None = None,
                  calibrate: str = "isotonic") -> CalibratedSelector:
    base = LogisticSelector(l2=1.5, features=features)
    base.fit([row["features"] for row in train_rows],
             [row["label"] for row in train_rows])
    calibrated = CalibratedSelector(base, calibrate)
    if calib_rows:
        calibrated.fit_calibrator(
            [row["features"] for row in calib_rows],
            [row["label"] for row in calib_rows],
        )
    return calibrated


def _selector_metrics(rows: list[dict], model, memory: ExperienceMemory,
                      gate: dict[str, float] | None = None) -> dict:
    scores: list[float] = []
    labels: list[int] = []
    top1: list[float] = []
    ndcg: list[float] = []
    by_task: dict[str, list[dict]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], []).append(row)
    for task_id, task_rows in by_task.items():
        scored = []
        for row in task_rows:
            score = model.predict_proba(row["features"])
            if gate:
                score *= gate.get(row["skill_id"], 1.0)
            scored.append((score, row))
            scores.append(score)
            labels.append(row["label"])
        scored.sort(key=lambda pair: pair[0], reverse=True)
        top1.append(float(scored[0][1]["label"]))
        dcg = 0.0
        for rank, (_, row) in enumerate(scored[:5]):
            dcg += row["label"] / (1.0 if rank == 0 else math.log2(rank + 1))
        ideal = sorted((row["label"] for row in task_rows), reverse=True)[:5]
        idcg = sum(label / (1.0 if rank == 0 else math.log2(rank + 1))
                   for rank, label in enumerate(ideal))
        ndcg.append(dcg / idcg if idcg else 0.0)
    return {
        "auroc": stats.auroc(scores, labels),
        "auprc": stats.auprc(scores, labels),
        "brier": stats.brier(scores, labels),
        "nll": stats.nll(scores, labels),
        "ece": stats.ece(scores, labels, bins=10),
        "p@1": stats.mean(top1),
        "ndcg@5": stats.mean(ndcg),
        "candidates": len(rows),
        "positive_rate": stats.mean([float(label) for label in labels]),
        "reliability": stats.reliability(scores, labels, bins=10),
    }


def _evaluate_records(records: list[dict], model, memory: ExperienceMemory, *,
                      k: int, budget: int, gate: dict[str, float] | None = None,
                      task: TaskSpec) -> dict:
    features = []
    for record in records:
        stats_row = memory.stats(record["skill"].skill_id, task.family)
        feat = extract_features(
            task=task,
            skill=record["skill"],
            stage="pipeline",
            semantic=record["semantic"],
            semantic_rank=record["rank"],
            candidate_count=len(records),
            memory=stats_row,
            step=0,
            library_size=20,
        )
        score = model.predict_proba(feat)
        if gate:
            score *= gate.get(record["skill"].skill_id, 1.0)
        features.append((score, record))
    features.sort(key=lambda pair: pair[0], reverse=True)
    if not features:
        raise RuntimeError("no candidates after gating")
    tried = features[: max(1, budget)]
    best = max((record["result"] for _, record in tried), key=lambda result: result.score)
    any_success = any(record["result"].success for _, record in tried)
    negative = any(record["result"].fallback or record["result"].utility < -0.02
                   for _, record in tried)
    chosen_score, chosen = features[0]
    return {
        "task_id": task.task_id,
        "family": task.family,
        "cloud": task.cloud,
        "success": 1 if any_success else 0,
        "score": best.score,
        "utility": best.utility,
        "baseline_score": best.baseline_score,
        "attempts": len(tried),
        "skill_id": best.skill_id,
        "score_model": chosen_score,
        "semantic": chosen["semantic"],
        "fallback": bool(best.fallback),
        "negative_transfer": 1 if negative else 0,
        "tried": [record["skill"].skill_id for _, record in tried],
        "attempted_id": tried[0][1]["skill"].skill_id,
    }


def _evaluate_method(tasks: list[TaskSpec], scenes: dict[str, Scene], retriever: Retriever,
                     model, memory: ExperienceMemory, k: int, budget: int,
                     gates: dict[str, dict[str, float]] | None = None) -> dict:
    results = []
    for task in tasks:
        records = _candidate_records(task, scenes[task.task_id], retriever, k)
        gate = gates.get(task.family) if gates else None
        if gate:
            allowed = [record for record in records if gate.get(record["skill"].skill_id, 1.0) > 0.0]
            if allowed:
                records = allowed
        results.append(_evaluate_records(records, model, memory, k=k, budget=budget,
                                         gate=gate, task=task))
    return {
        "tasks": results,
        "summary": _summarise(results),
    }


def _summarise(results: list[dict]) -> dict:
    if not results:
        return {}
    success = [float(row["success"]) for row in results]
    scores = [float(row["score"]) for row in results]
    utility = [float(row["utility"]) for row in results]
    negative = [float(row["negative_transfer"]) for row in results]
    attempts = [float(row["attempts"]) for row in results]
    low, high = stats.bootstrap_ci(success)
    slow, shigh = stats.bootstrap_ci(scores)
    return {
        "n": len(results),
        "success": stats.mean(success),
        "success_ci": [low, high],
        "score": stats.mean(scores),
        "score_ci": [slow, shigh],
        "utility": stats.mean(utility),
        "negative_transfer": stats.mean(negative),
        "attempts": stats.mean(attempts),
        "by_family": _group_summary(results, "family"),
    }


def _group_summary(results: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in results:
        groups.setdefault(str(row[key]), []).append(row)
    return {
        name: {
            "n": len(rows),
            "success": stats.mean([float(row["success"]) for row in rows]),
            "score": stats.mean([float(row["score"]) for row in rows]),
        }
        for name, rows in sorted(groups.items())
    }


# ------------------------------------------------------------------------- probes


def _build_verdicts(library: list[Skill], dev_tasks: list[TaskSpec],
                    scenes: dict[str, Scene], retriever: Retriever) -> tuple[dict, dict]:
    verdicts: dict[str, dict] = {family: {} for family in FAMILIES}
    matrix: dict[str, dict] = {}
    for family in FAMILIES:
        family_tasks = [task for task in dev_tasks if task.family == family]
        if not family_tasks:
            continue
        for skill in library:
            outcomes = [run_skill(task, scenes[task.task_id], skill).utility
                        for task in family_tasks]
            verdict = diagnose(skill, family, family_tasks, outcomes)
            verdicts[family][skill.skill_id] = verdict
            matrix.setdefault(skill.source_family, {}).setdefault(family, []).append({
                "skill_id": skill.skill_id,
                "utility": stats.mean(outcomes),
                "compatibility": verdict.compatibility,
                "status": verdict.status,
                "score": verdict.score,
            })
    return verdicts, matrix


def _gate_for_family(verdicts: dict, family: str, strategy: str) -> dict[str, float]:
    rows = list(verdicts.get(family, {}).values())
    return gate_scores(rows, strategy) if rows else {}


# ------------------------------------------------------------------------ stream


def _seed_memory(mode: str, source_records: list[dict]) -> ExperienceMemory:
    memory = ExperienceMemory(mode=mode)
    step = 0
    for record in source_records:
        memory.ingest(_outcome(record["task"], record["skill"], record["result"], step))
        step += 1
    return memory


def _stream_experiment(stream_tasks: list[TaskSpec], scenes: dict[str, Scene],
                       retriever: Retriever, model, source_records: list[dict],
                       k: int, budget: int, seed: int) -> dict:
    outputs = {}
    for mode in ("none", "flat", "tiered", "tiered_dedup", "full"):
        memory = _seed_memory(mode, source_records)
        results = []
        confidence = []
        outcomes = []
        for step, task in enumerate(stream_tasks):
            records = _candidate_records(task, scenes[task.task_id], retriever, k)
            result = _evaluate_records(records, model, memory, k=k, budget=budget, task=task)
            results.append(result)
            confidence.append(result["score_model"])
            outcomes.append(result["success"])
            chosen = next((record for record in records
                           if record["skill"].skill_id == result.get("attempted_id")), records[0])
            memory.ingest(_outcome(task, chosen["skill"], chosen["result"], step))
        blocks = []
        block_size = max(1, len(stream_tasks) // 6)
        for start in range(0, len(stream_tasks), block_size):
            block = results[start:start + block_size]
            if block:
                blocks.append({
                    "block": start // block_size + 1,
                    "success": stats.mean([row["success"] for row in block]),
                    "score": stats.mean([row["score"] for row in block]),
                })
        outputs[mode] = {
            "tasks": results,
            "summary": _summarise(results),
            "memory": memory.snapshot(),
            "blocks": blocks,
            "ece": stats.ece(confidence, outcomes, bins=5),
        }
    return outputs


# ------------------------------------------------------------------ transfer pairs


TRANSFER_PAIRS = (
    ("landcover", "change"),
    ("change", "landcover"),
    ("water", "flood"),
    ("flood", "water"),
    ("crop", "landcover"),
    ("landcover", "crop"),
    ("change", "urban"),
    ("urban", "change"),
    ("urban", "burn"),
    ("burn", "urban"),
)


def _generic_record(task: TaskSpec, scene: Scene) -> dict:
    skill = generic_pipeline(task)
    result = run_skill(task, scene, skill)
    return {
        "task": task,
        "skill": skill,
        "semantic": 0.0,
        "rank": 0,
        "result": result,
        "label": int(result.success),
        "score": result.score,
        "utility": result.utility,
    }


def _pair_experiment(test_tasks: list[TaskSpec], scenes: dict[str, Scene],
                     library: list[Skill], verdicts: dict, model,
                     memory: ExperienceMemory, k: int, budget: int) -> dict:
    outputs = {}
    for source_family, target_family in TRANSFER_PAIRS:
        subset = [skill for skill in library if skill.source_family == source_family]
        target_tasks = [task for task in test_tasks if task.family == target_family]
        if not subset or not target_tasks:
            continue
        retriever = Retriever(subset)
        pair_key = f"{source_family}->{target_family}"
        outputs[pair_key] = {}
        for strategy in ("none", "semantic", "outcome", "proposed", "oracle"):
            results = []
            for task in target_tasks:
                scene = scenes[task.task_id]
                if strategy == "oracle":
                    result = oracle_run(task, scene, retriever, k=len(subset)).to_dict()
                    results.append({
                        **result,
                        "success": 1 if result["success"] else 0,
                        "negative_transfer": 0,
                        "attempts": result["attempts"],
                    })
                    continue
                records = _candidate_records(task, scene, retriever, k=len(subset))
                gate = None
                if strategy != "none":
                    gate = _gate_for_family(verdicts, target_family, strategy)
                    allowed = [record for record in records
                               if gate.get(record["skill"].skill_id, 1.0) > 0.0]
                    records = allowed if allowed else [_generic_record(task, scene)]
                results.append(_evaluate_records(records, model, memory, k=len(subset),
                                                 budget=budget, gate=gate, task=task))
            blocked = 0
            if strategy in {"semantic", "outcome", "proposed"}:
                gate = _gate_for_family(verdicts, target_family, strategy)
                blocked = sum(1 for skill in subset if gate.get(skill.skill_id, 1.0) == 0.0)
            outputs[pair_key][strategy] = {
                "summary": _summarise(results),
                "tasks": results,
                "blocked": blocked,
                "library": len(subset),
            }
    return outputs


# --------------------------------------------------------------------- ablation


def _feature_ablation(train_rows: list[dict], calib_rows: list[dict],
                      test_records: list[dict], tasks: list[TaskSpec],
                      scenes: dict[str, Scene], retriever: Retriever,
                      memory: ExperienceMemory, k: int, budget: int) -> dict:
    subsets = {
        "semantic": ("bias", "semantic", "semantic_rank"),
        "structural": STRUCTURAL_FEATURES,
        "history": (
            "bias", "semantic", "semantic_rank", "target_match", "family_match",
            "input_match", "mechanism_match", "history_success", "history_utility",
            "history_std", "history_strength", "history_recency", "history_coverage",
            "history_duplicates", "novelty", "history_recent",
            "history_family_success", "history_family_strength", "success_x_target",
        ),
        "curated": CURATED_FEATURES,
        "full": FEATURE_NAMES,
    }
    outputs = {}
    for name, features in subsets.items():
        features = tuple(dict.fromkeys(features))
        model = _fit_selector(train_rows, calib_rows, features=features, calibrate="isotonic")
        metrics = {"ranking": _selector_metrics(
            [row for task in tasks for row in _rows_for_tasks([task], scenes, retriever, memory, k)],
            model, memory,
        )}
        results = []
        for task in tasks:
            records = _candidate_records(task, scenes[task.task_id], retriever, k)
            results.append(_evaluate_records(records, model, memory, k=k, budget=1, task=task))
        metrics["summary"] = _summarise(results)
        metrics["tasks"] = results
        metrics["coefficients"] = model.base.coefficients()
        outputs[name] = metrics
    return outputs


# -------------------------------------------------------------------------- main


# ------------------------------------------------------------------ retrieval


def _top1_outcomes(rows: list[dict], model) -> list[dict]:
    """Best candidate per task under a scorer, plus the hindsight optimum."""
    by_task: dict[str, list[tuple[float, dict]]] = {}
    for row in rows:
        score = float(model.predict_proba(row["features"]))
        by_task.setdefault(row["task_id"], []).append((score, row))
    outcomes = []
    for task_id, pairs in by_task.items():
        score, row = max(pairs, key=lambda pair: pair[0])
        outcomes.append({
            "task_id": task_id,
            "family": row["family"],
            "score": score,
            "utility": float(row["utility"]),
            "success": bool(row["success"]),
            "best_utility": max(other["utility"] for _, other in pairs),
        })
    return outcomes


def _retrieval_ablation(library: list[Skill], train_tasks: list[TaskSpec],
                        val_tasks: list[TaskSpec], test_tasks: list[TaskSpec],
                        scenes: dict[str, Scene], k: int) -> dict:
    """Same pipeline, three first stages.

    Each stage trains its own reranker.  Sharing one model across stages would
    be cheaper but wrong: the ``semantic`` feature is on a different scale for a
    BM25 score than for a cosine similarity, so a model fitted on one stage
    cannot be applied to another without mis-scaling that feature.
    """
    output = {}
    for kind in ("hashed", "bm25", "hybrid"):
        source = build_candidate_source(kind, library)
        memory = ExperienceMemory(mode="full")
        train_records = _all_records(train_tasks, scenes, source, k)
        train_rows = _rows_from_records(train_records, memory, candidate_count=k, ingest=True)
        val_rows = _rows_for_tasks(val_tasks, scenes, source, memory, k)
        model = _fit_selector(train_rows, val_rows)
        test_rows = _rows_for_tasks(test_tasks, scenes, source, memory, k)
        outcomes = _top1_outcomes(test_rows, model)
        output[kind] = {
            "top1_success": stats.mean([1.0 if row["success"] else 0.0 for row in outcomes]),
            "top1_utility": stats.mean([row["utility"] for row in outcomes]),
            "candidate_ceiling": stats.mean(
                [1.0 if row["best_utility"] > -0.02 else 0.0 for row in outcomes]),
            "candidates": len(test_rows) / max(1, len(outcomes)),
            "tasks": outcomes,
        }
    return output


# ------------------------------------------------------------------ gating


def _fallback_row(task: TaskSpec, baseline_score: float) -> dict:
    return {
        "task_id": task.task_id,
        "family": task.family,
        "cloud": task.cloud,
        "success": 1 if is_success(task, baseline_score) else 0,
        "score": baseline_score,
        "utility": 0.0,
        "baseline_score": baseline_score,
        "attempts": 0,
        "skill_id": "fallback",
        "score_model": 0.0,
        "semantic": 0.0,
        "fallback": True,
        "negative_transfer": 0,
        "tried": [],
        "attempted_id": "fallback",
    }


def _apply_gate(gate: ConformalGate, test_tasks: list[TaskSpec],
                probe: dict[str, dict], results: dict[str, dict]) -> list[dict]:
    rows = []
    for task in test_tasks:
        result = results[task.task_id]
        if gate.commit(probe[task.task_id]["score"]):
            rows.append(result)
        else:
            rows.append(_fallback_row(task, result["baseline_score"]))
    return rows


def _conformal_experiment(dev_rows: list[dict], test_rows: list[dict],
                          test_tasks: list[TaskSpec], results: dict[str, dict],
                          model, alphas: tuple[float, ...] = CONFORMAL_ALPHAS,
                          delta: float = 0.1) -> dict:
    calibration = _top1_outcomes(dev_rows, model)
    probe_rows = _top1_outcomes(test_rows, model)
    probe = {row["task_id"]: row for row in probe_rows}
    calibration_scores = [row["score"] for row in calibration]
    calibration_losses = negative_transfer([row["utility"] for row in calibration])
    test_scores = [row["score"] for row in probe_rows]
    test_losses = negative_transfer([row["utility"] for row in probe_rows])
    output = {}
    for alpha in alphas:
        gate = ConformalGate(alpha=alpha, delta=delta)
        certificate = gate.fit(calibration_scores, calibration_losses)
        empirical = fixed_threshold_gate(calibration_scores, calibration_losses, alpha)
        rows = _apply_gate(gate, test_tasks, probe, results)
        empirical_rows = _apply_gate(empirical, test_tasks, probe, results)
        output[f"{alpha:.2f}"] = {
            "certificate": certificate.to_dict(),
            "test": gate.evaluate(test_scores, test_losses),
            "summary": _summarise(rows),
            "empirical": {
                "test": empirical.evaluate(test_scores, test_losses),
                "summary": _summarise(empirical_rows),
                "threshold": empirical.certificate.threshold,
            },
        }
    return output


# ------------------------------------------------------------------ bandit


def _bandit_experiment(stream_tasks: list[TaskSpec], scenes: dict[str, Scene],
                       retriever, policies: dict[str, object], k: int,
                       warm_rows: list[tuple[dict, float]] | None = None,
                       warm_names: tuple[str, ...] = ("linucb", "thompson", "epsilon_greedy")) -> dict:
    """Prequential selection with cumulative regret against the best candidate."""
    output = {}
    for name, policy in policies.items():
        policy.reset()
        if warm_rows and name in warm_names:
            policy.warm_start([row for row, _ in warm_rows], [reward for _, reward in warm_rows])
        cumulative = 0.0
        curve: list[float] = []
        chosen: list[dict] = []
        for step, task in enumerate(stream_tasks):
            records = _candidate_records(task, scenes[task.task_id], retriever, k)
            count = len(records)
            rows = []
            for record in records:
                features = extract_features(
                    task=task,
                    skill=record["skill"],
                    stage="pipeline",
                    semantic=record["semantic"],
                    semantic_rank=record["rank"],
                    candidate_count=count,
                    memory=None,
                    step=step,
                    library_size=len(retriever.library),
                )
                rows.append((features, record))
            if hasattr(policy, "begin"):
                policy.begin()
            scored = [(policy.score(features), features, record) for features, record in rows]
            _, features, record = max(scored, key=lambda item: item[0])
            reward = reward_from_utility(record["result"].utility)
            oracle = max(reward_from_utility(item[1]["result"].utility) for item in rows)
            policy.update(features, reward)
            cumulative += oracle - reward
            curve.append(cumulative)
            chosen.append({
                "step": step,
                "task_id": task.task_id,
                "family": task.family,
                "skill_id": record["skill"].skill_id,
                "utility": record["result"].utility,
                "success": 1 if record["result"].success else 0,
                "reward": reward,
                "oracle_reward": oracle,
                "regret": oracle - reward,
                "negative_transfer": 1 if record["result"].utility < -0.02 else 0,
            })
        output[name] = {
            "policy": name,
            "cumulative_regret": cumulative,
            "mean_regret": cumulative / max(1, len(stream_tasks)),
            "success": stats.mean([float(row["success"]) for row in chosen]),
            "utility": stats.mean([row["utility"] for row in chosen]),
            "negative_transfer": stats.mean([float(row["negative_transfer"]) for row in chosen]),
            "curve": curve,
            "chosen": chosen,
        }
    return output


# ------------------------------------------------------------- sensitivity


def _k_sensitivity(library: list[Skill], test_tasks: list[TaskSpec],
                   scenes: dict[str, Scene], memory: ExperienceMemory, model,
                   values: tuple[int, ...] = SENSITIVITY_K) -> dict:
    """How much of the result depends on the candidate-pool size?

    The pool size is a free parameter of every retrieval-plus-rerank pipeline
    and is rarely reported.  The same trained model is applied at several pool
    widths; the per-task scores move because ``semantic_rank`` and the feature
    normalisation depend on the pool.
    """
    source = build_candidate_source("hashed", library)
    output = {}
    for value in values:
        rows = []
        sizes = []
        for task in test_tasks:
            records = _candidate_records(task, scenes[task.task_id], source, value)
            sizes.append(len(records))
            rows.append(_evaluate_records(records, model, memory, k=value, budget=1,
                                          task=task))
        output[str(value)] = {
            "summary": _summarise(rows),
            "pool_mean": stats.mean(sizes),
        }
    return output


def run_seed(seed: int, config: Config) -> dict:
    started = time.time()
    clear_caches()
    splits, scenes, library = _make_data(seed, config)
    retriever = build_candidate_source(config.retriever, library)
    train_tasks = splits["source_train"]
    val_tasks = splits["source_val"]
    test_tasks = splits["target_test"]
    dev_tasks = splits["target_dev"]
    stream_tasks = splits["stream"]

    # Historical experience: every candidate skill tried on every source task.
    train_records = _all_records(train_tasks, scenes, retriever, config.k)
    source_memory = ExperienceMemory(mode="full")
    train_rows = _rows_from_records(train_records, source_memory,
                                    candidate_count=config.k, ingest=True)
    source_snapshot = source_memory.snapshot()
    val_rows = _rows_for_tasks(val_tasks, scenes, retriever, source_memory,
                               config.k, ingest=False)

    # Base model and calibrated variants.
    base = LogisticSelector(l2=1.5)
    base.fit([row["features"] for row in train_rows], [row["label"] for row in train_rows])
    temperature = CalibratedSelector(base, "temperature")
    temperature.fit_calibrator([row["features"] for row in val_rows],
                               [row["label"] for row in val_rows])
    isotonic = CalibratedSelector(base, "isotonic")
    isotonic.fit_calibrator([row["features"] for row in val_rows],
                            [row["label"] for row in val_rows])
    dev_rows = _rows_for_tasks(dev_tasks, scenes, retriever, source_memory, config.k)
    fewshot = CalibratedSelector(base, "isotonic")
    fewshot.fit_calibrator([row["features"] for row in dev_rows],
                           [row["label"] for row in dev_rows])
    memory_base = LogisticSelector(l2=1.5, features=FEATURE_NAMES)
    memory_base.fit([row["features"] for row in train_rows],
                    [row["label"] for row in train_rows])
    memory_model = CalibratedSelector(memory_base, "isotonic")
    memory_model.fit_calibrator([row["features"] for row in val_rows],
                                [row["label"] for row in val_rows])
    random_selector = FixedSelector("random", "random", seed=seed)
    semantic_selector = FixedSelector("semantic", "semantic")
    frequency_selector = FixedSelector("frequency", "frequency")

    # Transfer diagnosis on the target development split.
    verdicts, transfer_matrix = _build_verdicts(library, dev_tasks, scenes, retriever)
    proposed_gates = {
        family: _gate_for_family(verdicts, family, "proposed") for family in FAMILIES
    }
    semantic_gates = {
        family: _gate_for_family(verdicts, family, "semantic") for family in FAMILIES
    }
    outcome_gates = {
        family: _gate_for_family(verdicts, family, "outcome") for family in FAMILIES
    }

    # Candidate ranking quality on target test (all candidates executed once).
    test_candidate_rows = []
    for task in test_tasks:
        test_candidate_rows.extend(
            _rows_for_tasks([task], scenes, retriever, source_memory, config.k)
        )

    methods = {
        "random": random_selector,
        "semantic": semantic_selector,
        "frequency": frequency_selector,
        "logistic": base,
        "temperature": temperature,
        "isotonic": isotonic,
        "fewshot": fewshot,
    }
    method_outputs = {}
    for name, model in methods.items():
        results = []
        for task in test_tasks:
            records = _candidate_records(task, scenes[task.task_id], retriever, config.k)
            results.append(_evaluate_records(records, model, source_memory,
                                             k=config.k, budget=1, task=task))
        method_outputs[name] = {
            "tasks": results,
            "summary": _summarise(results),
            "ranking": _selector_metrics(test_candidate_rows, model, source_memory),
        }

    # Budgeted retry.
    budget_results = []
    for task in test_tasks:
        records = _candidate_records(task, scenes[task.task_id], retriever, config.k)
        budget_results.append(_evaluate_records(records, isotonic, source_memory,
                                                k=config.k, budget=config.budget, task=task))
    method_outputs["isotonic_budget"] = {
        "tasks": budget_results,
        "summary": _summarise(budget_results),
        "ranking": method_outputs["isotonic"]["ranking"],
    }

    # Oracle and transfer-gated variants.
    oracle_results = []
    for task in test_tasks:
        scene = scenes[task.task_id]
        oracle_results.append(oracle_run(task, scene, retriever, k=len(library)).to_dict())
    oracle_summary = _summarise([
        {
            "success": 1 if row["success"] else 0,
            "score": row["score"],
            "utility": row["utility"],
            "negative_transfer": 0,
            "attempts": row["attempts"],
            "family": row["family"],
        }
        for row in oracle_results
    ])

    gate_outputs = {
        "none": method_outputs["isotonic"],
    }
    for strategy, gates in (
        ("semantic", semantic_gates),
        ("outcome", outcome_gates),
        ("proposed", proposed_gates),
    ):
        results = []
        for task in test_tasks:
            records = _candidate_records(task, scenes[task.task_id], retriever, config.k)
            gate = gates[task.family]
            allowed = [record for record in records if gate.get(record["skill"].skill_id, 1.0) > 0.0]
            if allowed:
                records = allowed
            results.append(_evaluate_records(records, isotonic, source_memory,
                                             k=config.k, budget=1, gate=gates[task.family],
                                             task=task))
        gate_outputs[strategy] = {
            "tasks": results,
            "summary": _summarise(results),
            "blocked": sum(1 for family in gates.values() for value in family.values() if value == 0.0),
        }

    pairs = _pair_experiment(test_tasks, scenes, library, verdicts, isotonic,
                             source_memory, config.k, config.budget)

    stream = _stream_experiment(stream_tasks, scenes, retriever, memory_model,
                                train_records, config.k, config.budget, seed)

    retrieval = {}
    conformal = {}
    if config.include_ablations:
        retrieval = _retrieval_ablation(library, train_tasks, val_tasks, test_tasks,
                                        scenes, config.k)
        conformal = _conformal_experiment(
            dev_rows, test_candidate_rows, test_tasks,
            {row["task_id"]: row for row in method_outputs["isotonic"]["tasks"]},
            isotonic,
        )

    sensitivity = {}
    if config.include_ablations:
        sensitivity = _k_sensitivity(library, test_tasks, scenes, source_memory, isotonic)

    bandit_warm = []
    for record in train_records:
        features = extract_features(
            task=record["task"],
            skill=record["skill"],
            stage="pipeline",
            semantic=record["semantic"],
            semantic_rank=record["rank"],
            candidate_count=config.k,
            memory=None,
            step=0,
            library_size=len(library),
        )
        bandit_warm.append((features, reward_from_utility(record["result"].utility)))

    bandit = _bandit_experiment(
        stream_tasks, scenes, retriever,
        {
            "static": build_policy("static", model=isotonic),
            "linucb_cold": build_policy("linucb"),
            "linucb": build_policy("linucb"),
            "thompson": build_policy("thompson", seed=seed),
            "epsilon_greedy": build_policy("epsilon_greedy", seed=seed),
        },
        config.k,
        warm_rows=bandit_warm,
        warm_names=("linucb", "thompson", "epsilon_greedy"),
    )

    ablation = {}
    if config.include_ablations:
        ablation = _feature_ablation(train_rows, val_rows, test_candidate_rows, test_tasks,
                                     scenes, retriever, source_memory, config.k, config.budget)

    library_rows = []
    for skill in library:
        library_rows.append({
            "skill_id": skill.skill_id,
            "name": skill.name,
            "source_family": skill.source_family,
            "targets": list(skill.targets),
            "ops": [op.op for op in skill.ops],
            "signature": skill.signature,
            "source_tasks": list(skill.stats.source_tasks),
        })

    return {
        "seed": seed,
        "elapsed_seconds": round(time.time() - started, 2),
        "library": {
            "size": len(library),
            "skills": library_rows,
            "by_family": {
                family: sum(1 for skill in library if skill.source_family == family)
                for family in FAMILIES
            },
        },
        "memory": source_snapshot,
        "data": {
            "tasks": {name: len(tasks) for name, tasks in splits.items()},
            "candidate_records": len(train_records),
        },
        "selector": {
            "coefficients": base.coefficients(),
            "temperature": temperature.calibrator.temperature,
            "train_rows": len(train_rows),
            "calibration_rows": len(val_rows),
            "positive_rate": stats.mean([float(row["label"]) for row in train_rows]),
        },
        "methods": method_outputs,
        "oracle": {"tasks": oracle_results, "summary": oracle_summary},
        "selection": gate_outputs,
        "pairs": pairs,
        "transfer": {
            "matrix": transfer_matrix,
            "verdicts": {
                family: {skill_id: verdict.to_dict() for skill_id, verdict in rows.items()}
                for family, rows in verdicts.items()
            },
        },
        "stream": stream,
        "retrieval": retrieval,
        "conformal": conformal,
        "sensitivity": sensitivity,
        "bandit": bandit,
        "ablation": ablation,
    }


def aggregate(seed_results: list[dict]) -> dict:
    methods = {}
    names = list(seed_results[0]["methods"]) if seed_results else []
    for name in names:
        task_rows = [row for result in seed_results for row in result["methods"][name]["tasks"]]
        methods[name] = _summarise(task_rows)
        methods[name]["ranking"] = {
            key: stats.mean([result["methods"][name]["ranking"][key] for result in seed_results])
            for key in ("auroc", "auprc", "brier", "nll", "ece", "p@1", "ndcg@5")
        }
        bins: dict[int, dict[str, list[float]]] = {}
        for result in seed_results:
            for row in result["methods"][name]["ranking"].get("reliability", []):
                bucket = bins.setdefault(row["bin"], {"confidence": [], "accuracy": [], "count": []})
                bucket["confidence"].append(row["confidence"])
                bucket["accuracy"].append(row["accuracy"])
                bucket["count"].append(row["count"])
        methods[name]["ranking"]["reliability"] = [
            {
                "bin": index,
                "confidence": stats.mean(values["confidence"]),
                "accuracy": stats.mean(values["accuracy"]),
                "count": stats.mean(values["count"]),
            }
            for index, values in sorted(bins.items())
        ]
        methods[name]["per_seed_success"] = [
            result["methods"][name]["summary"]["success"] for result in seed_results
        ]
    gates = {}
    for name in ("none", "semantic", "outcome", "proposed"):
        task_rows = [row for result in seed_results for row in result["selection"][name]["tasks"]]
        gates[name] = _summarise(task_rows)
    oracle_rows = [row for result in seed_results for row in result["oracle"]["tasks"]]
    oracle = _summarise([
        {
            "success": 1 if row["success"] else 0,
            "score": row["score"],
            "utility": row["utility"],
            "negative_transfer": 0,
            "attempts": row["attempts"],
            "family": row["family"],
        }
        for row in oracle_rows
    ])
    # Aggregate transferability matrix (source family -> target family).
    transfer_matrix = {}
    for result in seed_results:
        for source, targets in result["transfer"]["matrix"].items():
            for target, rows in targets.items():
                bucket = transfer_matrix.setdefault(source, {}).setdefault(target, [])
                bucket.extend(rows)
    transfer_summary = {}
    for source, targets in transfer_matrix.items():
        transfer_summary[source] = {}
        for target, rows in targets.items():
            transfer_summary[source][target] = {
                "skills": len(rows),
                "utility": stats.mean([row["utility"] for row in rows]),
                "compatibility": stats.mean([row["compatibility"] for row in rows]),
                "transferable": sum(1 for row in rows if row["status"] == "transferable") / len(rows),
                "blocked": sum(1 for row in rows if row["status"] == "blocked") / len(rows),
            }

    pairs = {}
    pair_keys = list(seed_results[0].get("pairs", {})) if seed_results else []
    for pair in pair_keys:
        pairs[pair] = {}
        for strategy in ("none", "semantic", "outcome", "proposed", "oracle"):
            rows = [row for result in seed_results for row in result["pairs"][pair][strategy]["tasks"]]
            summary = _summarise(rows)
            summary["blocked"] = stats.mean([
                result["pairs"][pair][strategy]["blocked"] for result in seed_results
            ])
            summary["library"] = stats.mean([
                result["pairs"][pair][strategy]["library"] for result in seed_results
            ])
            pairs[pair][strategy] = summary

    stream = {}
    for mode in ("none", "flat", "tiered", "tiered_dedup", "full"):
        rows = [row for result in seed_results for row in result["stream"][mode]["tasks"]]
        stream[mode] = {
            "summary": _summarise(rows),
            "memory": seed_results[0]["stream"][mode]["memory"],
            "ece": stats.mean([result["stream"][mode]["ece"] for result in seed_results]),
        }
    paired = {}
    for left in ("isotonic", "proposed", "temperature"):
        for right in ("semantic", "random"):
            if left not in methods or right not in methods:
                continue
            successes_left, successes_right, scores_left, scores_right = [], [], [], []
            for result in seed_results:
                left_rows = {row["task_id"]: row for row in result["methods"][left]["tasks"]}
                right_rows = {row["task_id"]: row for row in result["methods"][right]["tasks"]}
                for task_id in left_rows:
                    successes_left.append(left_rows[task_id]["success"])
                    successes_right.append(right_rows[task_id]["success"])
                    scores_left.append(left_rows[task_id]["score"])
                    scores_right.append(right_rows[task_id]["score"])
            paired[f"{left}_vs_{right}"] = {
                "success_delta": stats.mean(successes_left) - stats.mean(successes_right),
                "success_mcnemar_p": stats.mcnemar(successes_left, successes_right)["p"],
                "score_delta": stats.mean(scores_left) - stats.mean(scores_right),
                "score_permutation_p": stats.sign_permutation(scores_left, scores_right),
            }
    if paired:
        keys = list(paired)
        mcnemar_adjusted = stats.holm([paired[key]["success_mcnemar_p"] for key in keys])
        permutation_adjusted = stats.holm([paired[key]["score_permutation_p"] for key in keys])
        for key, mcnemar_p, permutation_p in zip(keys, mcnemar_adjusted, permutation_adjusted):
            paired[key]["success_mcnemar_p_holm"] = mcnemar_p
            paired[key]["score_permutation_p_holm"] = permutation_p

    ablation = {}
    if seed_results and seed_results[0].get("ablation"):
        for name in seed_results[0]["ablation"]:
            ablation[name] = {
                "success": stats.mean([result["ablation"][name]["summary"]["success"]
                                       for result in seed_results]),
                "score": stats.mean([result["ablation"][name]["summary"]["score"]
                                     for result in seed_results]),
                "ece": stats.mean([result["ablation"][name]["ranking"]["ece"]
                                   for result in seed_results]),
                "auroc": stats.mean([result["ablation"][name]["ranking"]["auroc"]
                                     for result in seed_results]),
            }

    retrieval = {}
    if seed_results and seed_results[0].get("retrieval"):
        for name in seed_results[0]["retrieval"]:
            retrieval[name] = {
                key: stats.mean([result["retrieval"][name][key] for result in seed_results])
                for key in ("top1_success", "top1_utility", "candidate_ceiling", "candidates")
            }

    conformal = {}
    if seed_results and seed_results[0].get("conformal"):
        from .conformal import binom_upper

        def pooled(tests: list[dict], totals: list[int]) -> dict:
            """Pool committed decisions across seeds instead of averaging rates.

            Averaging per-seed coverage hides seeds where the gate abstained
            entirely; pooling keeps coverage, risk, and the confidence bound on
            the same denominator.
            """
            total = sum(totals)
            committed = sum(int(round(test["committed"])) for test in tests)
            failures = sum(int(round(test["risk"] * test["committed"])) for test in tests)
            return {
                "coverage": committed / total if total else 0.0,
                "risk": failures / committed if committed else 0.0,
                "upper": binom_upper(failures, committed, 0.1) if committed else 1.0,
                "committed": committed,
            }

        for alpha in seed_results[0]["conformal"]:
            blocks = [result["conformal"][alpha] for result in seed_results]
            totals = [block["summary"]["n"] for block in blocks]
            conformal[alpha] = {
                "certificate": {
                    key: stats.mean([block["certificate"][key] for block in blocks])
                    for key in ("risk", "upper_bound", "coverage", "threshold")
                },
                "test": pooled([block["test"] for block in blocks], totals),
                "success": stats.mean([block["summary"]["success"] for block in blocks]),
                "score": stats.mean([block["summary"]["score"] for block in blocks]),
                "negative_transfer": stats.mean(
                    [block["summary"]["negative_transfer"] for block in blocks]),
                "empirical": {
                    "threshold": stats.mean(
                        [block["empirical"]["threshold"] for block in blocks]),
                    "test": pooled([block["empirical"]["test"] for block in blocks], totals),
                    "success": stats.mean(
                        [block["empirical"]["summary"]["success"] for block in blocks]),
                    "score": stats.mean(
                        [block["empirical"]["summary"]["score"] for block in blocks]),
                    "negative_transfer": stats.mean(
                        [block["empirical"]["summary"]["negative_transfer"] for block in blocks]),
                },
            }

    sensitivity = {}
    if seed_results and seed_results[0].get("sensitivity"):
        for width in seed_results[0]["sensitivity"]:
            blocks = [result["sensitivity"][width] for result in seed_results]
            sensitivity[width] = {
                "success": stats.mean([block["summary"]["success"] for block in blocks]),
                "score": stats.mean([block["summary"]["score"] for block in blocks]),
                "negative_transfer": stats.mean(
                    [block["summary"]["negative_transfer"] for block in blocks]),
                "pool_mean": stats.mean([block["pool_mean"] for block in blocks]),
            }

    bandit = {}
    if seed_results and seed_results[0].get("bandit"):
        for name in seed_results[0]["bandit"]:
            blocks = [result["bandit"][name] for result in seed_results]
            length = min(len(block["curve"]) for block in blocks)
            bandit[name] = {
                "cumulative_regret": stats.mean([block["cumulative_regret"] for block in blocks]),
                "mean_regret": stats.mean([block["mean_regret"] for block in blocks]),
                "success": stats.mean([block["success"] for block in blocks]),
                "utility": stats.mean([block["utility"] for block in blocks]),
                "negative_transfer": stats.mean(
                    [block["negative_transfer"] for block in blocks]),
                "curve": [stats.mean([block["curve"][index] for block in blocks])
                          for index in range(length)],
            }

    return {
        "methods": methods,
        "selection": gates,
        "pairs": pairs,
        "transfer_matrix": transfer_summary,
        "oracle": oracle,
        "stream": stream,
        "retrieval": retrieval,
        "conformal": conformal,
        "sensitivity": sensitivity,
        "bandit": bandit,
        "ablation": ablation,
        "paired": paired,
        "library_size": stats.mean([result["library"]["size"] for result in seed_results]),
        "elapsed_seconds": sum(result["elapsed_seconds"] for result in seed_results),
    }


def run(config: Config | None = None) -> dict:
    config = config or Config()
    seed_results = [run_seed(seed, config) for seed in config.seeds]
    results = {
        "config": config.to_dict(),
        "seeds": seed_results,
        "aggregate": aggregate(seed_results),
    }
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_csv(results, output)
    return results


def _write_csv(results: dict, output: Path) -> None:
    tables = output / "tables"
    tables.mkdir(exist_ok=True)
    lines = ["method,success,success_low,success_high,score,score_low,score_high,negative_transfer,auroc,auprc,ece,brier,nll"]
    aggregate = results["aggregate"]
    for name, row in aggregate["methods"].items():
        ranking = row["ranking"]
        lines.append(
            f"{name},{row['success']:.4f},{row['success_ci'][0]:.4f},{row['success_ci'][1]:.4f},"
            f"{row['score']:.4f},{row['score_ci'][0]:.4f},{row['score_ci'][1]:.4f},"
            f"{row['negative_transfer']:.4f},{ranking['auroc']:.4f},{ranking['auprc']:.4f},"
            f"{ranking['ece']:.4f},{ranking['brier']:.4f},{ranking['nll']:.4f}"
        )
    (tables / "methods.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = ["strategy,success,score,negative_transfer"]
    for name, row in aggregate["selection"].items():
        lines.append(f"{name},{row['success']:.4f},{row['score']:.4f},{row['negative_transfer']:.4f}")
    (tables / "transfer_gate.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = ["mode,success,score,retained,characters,duplicate_ratio,ece"]
    for name, row in aggregate["stream"].items():
        memory = row["memory"]
        lines.append(
            f"{name},{row['summary']['success']:.4f},{row['summary']['score']:.4f},"
            f"{memory['retained']},{memory['characters']},{memory['duplicate_ratio']:.4f},{row['ece']:.4f}"
        )
    (tables / "memory_stream.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = ["pair,strategy,success,score,negative_transfer,blocked,library"]
    for pair, strategies in aggregate["pairs"].items():
        for strategy, row in strategies.items():
            lines.append(
                f"{pair},{strategy},{row['success']:.4f},{row['score']:.4f},"
                f"{row['negative_transfer']:.4f},{row['blocked']:.2f},{row['library']:.2f}"
            )
    (tables / "transfer_pairs.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = ["pair,success_delta,mcnemar_p,score_delta,permutation_p"]
    for name, row in aggregate["paired"].items():
        lines.append(
            f"{name},{row['success_delta']:.4f},{row['success_mcnemar_p']:.4f},"
            f"{row['score_delta']:.4f},{row['score_permutation_p']:.4f}"
        )
    (tables / "paired_tests.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    retrieval = aggregate.get("retrieval")
    if retrieval:
        lines = ["stage,top1_success,top1_utility,candidate_ceiling,candidates"]
        for name, row in retrieval.items():
            lines.append(
                f"{name},{row['top1_success']:.4f},{row['top1_utility']:.4f},"
                f"{row['candidate_ceiling']:.4f},{row['candidates']:.2f}"
            )
        (tables / "retrieval.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    conformal = aggregate.get("conformal")
    if conformal:
        lines = ["alpha,gate,threshold,coverage,risk,upper,success,negative_transfer"]
        for alpha, row in conformal.items():
            lines.append(
                f"{alpha},conformal,{row['certificate']['threshold']:.4f},"
                f"{row['test']['coverage']:.4f},{row['test']['risk']:.4f},"
                f"{row['test']['upper']:.4f},{row['success']:.4f},"
                f"{row['negative_transfer']:.4f}"
            )
            empirical = row["empirical"]
            lines.append(
                f"{alpha},empirical,{empirical['threshold']:.4f},"
                f"{empirical['test']['coverage']:.4f},{empirical['test']['risk']:.4f},"
                f"{empirical['test']['upper']:.4f},{empirical['success']:.4f},"
                f"{empirical['negative_transfer']:.4f}"
            )
        (tables / "conformal.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sensitivity = aggregate.get("sensitivity")
    if sensitivity:
        lines = ["pool_size,success,score,negative_transfer,pool_mean"]
        for width, row in sensitivity.items():
            lines.append(f"{width},{row['success']:.4f},{row['score']:.4f},"
                         f"{row['negative_transfer']:.4f},{row['pool_mean']:.2f}")
        (tables / "sensitivity_k.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    bandit = aggregate.get("bandit")
    if bandit:
        lines = ["policy,cumulative_regret,mean_regret,success,utility,negative_transfer"]
        for name, row in bandit.items():
            lines.append(
                f"{name},{row['cumulative_regret']:.4f},{row['mean_regret']:.4f},"
                f"{row['success']:.4f},{row['utility']:.4f},{row['negative_transfer']:.4f}"
            )
        (tables / "bandit.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run GeoCompass experiments")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--output", default="experiments/geocompass")
    parser.add_argument("--retriever", default="hashed", choices=("hashed", "bm25", "hybrid"))
    parser.add_argument("--stream", type=int, default=None,
                        help="stream tasks per family (default 8)")
    parser.add_argument("--dataset", default=None,
                        help="manifest.json (or its directory) describing real chips")
    parser.add_argument("--no-ablations", action="store_true")
    args = parser.parse_args(argv)
    counts = dict(DEFAULT_COUNTS)
    if args.stream is not None:
        if args.stream <= 0:
            parser.error("--stream must be positive")
        counts["stream"] = args.stream
    config = Config(
        seeds=[int(value) for value in args.seeds.split(",") if value],
        counts=counts,
        k=args.k,
        budget=args.budget,
        output=args.output,
        include_ablations=not args.no_ablations,
        retriever=args.retriever,
        dataset=args.dataset,
    )
    results = run(config)
    print(json.dumps(results["aggregate"]["methods"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
