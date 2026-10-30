"""Skill retrieval, execution, and task-level evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field

from .features import HashedEmbedding, cosine
from .ops import Context, Incompatible, execute_skill, grade
from .tasks import generic_pipeline, is_success
from .types import Scene, Skill, TaskSpec


class Retriever:
    """Semantic candidate generation, shared by every reranking method."""

    def __init__(self, library: list[Skill], embedding: HashedEmbedding | None = None):
        self.library = list(library)
        self.embedding = embedding or HashedEmbedding()
        self.vectors = {skill.skill_id: self.embedding.skill(skill) for skill in self.library}

    def candidates(self, task: TaskSpec, k: int = 8,
                   exclude: set[str] | None = None) -> list[tuple[Skill, float]]:
        query = self.embedding.task(task)
        scored = []
        for skill in self.library:
            if not skill.active:
                continue
            if exclude and skill.skill_id in exclude:
                continue
            scored.append((skill, cosine(query, self.vectors[skill.skill_id])))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]


@dataclass
class PipelineResult:
    task_id: str
    family: str
    skill_id: str
    score: float
    metric: dict[str, float]
    baseline_score: float
    utility: float
    success: bool
    fallback: bool = False
    error: str = ""
    skill: Skill | None = None
    context: Context | None = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "skill_id": self.skill_id,
            "score": round(self.score, 6),
            "metric": {key: round(value, 6) for key, value in self.metric.items()},
            "baseline_score": round(self.baseline_score, 6),
            "utility": round(self.utility, 6),
            "success": self.success,
            "fallback": self.fallback,
            "error": self.error,
        }


_BASELINE_CACHE: dict[str, tuple[float, dict[str, float]]] = {}
_PAIR_CACHE: dict[tuple[str, str], PipelineResult] = {}


def clear_caches() -> None:
    _BASELINE_CACHE.clear()
    _PAIR_CACHE.clear()


def baseline(task: TaskSpec, scene: Scene) -> tuple[float, dict[str, float]]:
    if task.task_id in _BASELINE_CACHE:
        return _BASELINE_CACHE[task.task_id]
    skill = generic_pipeline(task)
    try:
        ctx = execute_skill(skill, task, scene)
        metric = grade(scene, ctx.map)
    except Incompatible:
        metric = {"score": 0.0, "oa": 0.0, "f1": 0.0, "iou": 0.0}
    _BASELINE_CACHE[task.task_id] = (metric["score"], metric)
    return _BASELINE_CACHE[task.task_id]


def run_skill(task: TaskSpec, scene: Scene, skill: Skill) -> PipelineResult:
    key = (task.task_id, skill.skill_id)
    if key in _PAIR_CACHE:
        return _PAIR_CACHE[key]
    base_score, _ = baseline(task, scene)
    fallback = False
    error = ""
    used = skill
    try:
        ctx = execute_skill(skill, task, scene)
    except Incompatible as exc:
        fallback = True
        error = str(exc)
        used = generic_pipeline(task)
        ctx = execute_skill(used, task, scene)
    metric = grade(scene, ctx.map)
    score = metric["score"]
    result = PipelineResult(
        task_id=task.task_id,
        family=task.family,
        skill_id=used.skill_id,
        score=score,
        metric=metric,
        baseline_score=base_score,
        utility=score - base_score,
        success=is_success(task, score),
        fallback=fallback,
        error=error,
        skill=used,
        context=ctx,
    )
    _PAIR_CACHE[key] = result
    return result


@dataclass
class SelectionResult:
    task_id: str
    family: str
    success: bool
    score: float
    baseline_score: float
    utility: float
    attempts: int
    skill_id: str
    score_model: float
    semantic: float
    fallback: bool
    negative_transfer: bool
    tried: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "success": self.success,
            "score": round(self.score, 6),
            "baseline_score": round(self.baseline_score, 6),
            "utility": round(self.utility, 6),
            "attempts": self.attempts,
            "skill_id": self.skill_id,
            "score_model": round(self.score_model, 5),
            "semantic": round(self.semantic, 5),
            "fallback": self.fallback,
            "negative_transfer": self.negative_transfer,
            "tried": list(self.tried),
            "errors": list(self.errors),
        }


def select_and_run(
    task: TaskSpec,
    scene: Scene,
    retriever: Retriever,
    selector,
    memory,
    *,
    step: int = 0,
    budget: int = 1,
    k: int = 8,
    gate_scores: dict[str, float] | None = None,
) -> SelectionResult:
    from .selector import rank_candidates

    pool = retriever.candidates(task, k=k)
    if gate_scores:
        pool = [
            (skill, semantic) for skill, semantic in pool
            if gate_scores.get(skill.skill_id, 1.0) > 0.0
        ]
    ranked = rank_candidates(task, pool, selector, "pipeline", memory, step,
                             len(retriever.library), transfer_scores=gate_scores)
    if not ranked:
        from .selector import Ranked
        fallback_skill = generic_pipeline(task)
        ranked = [Ranked(fallback_skill, 0.0, 0.0, {})]
    base_score, _ = baseline(task, scene)
    best: PipelineResult | None = None
    tried: list[str] = []
    errors: list[str] = []
    negative = False
    for attempt in range(min(budget, len(ranked))):
        candidate = ranked[attempt]
        tried.append(candidate.skill.skill_id)
        result = run_skill(task, scene, candidate.skill)
        if result.error:
            errors.append(result.error)
        if result.fallback or result.utility < -0.02:
            negative = True
        if best is None or result.score > best.score:
            best = result
        if result.success:
            break
    assert best is not None
    chosen_rank = ranked[min(len(tried) - 1, len(ranked) - 1)]
    return SelectionResult(
        task_id=task.task_id,
        family=task.family,
        success=best.success,
        score=best.score,
        baseline_score=base_score,
        utility=best.utility,
        attempts=len(tried),
        skill_id=best.skill_id,
        score_model=chosen_rank.score,
        semantic=chosen_rank.semantic,
        fallback=best.fallback,
        negative_transfer=negative,
        tried=tried,
        errors=errors,
    )


def oracle_run(task: TaskSpec, scene: Scene, retriever: Retriever,
               k: int = 8) -> SelectionResult:
    """Upper bound that executes every retrieved candidate and keeps the best."""
    pool = retriever.candidates(task, k=k)
    base_score, _ = baseline(task, scene)
    best: PipelineResult | None = None
    for skill, _ in pool:
        result = run_skill(task, scene, skill)
        if best is None or result.score > best.score:
            best = result
    if best is None:
        best = run_skill(task, scene, generic_pipeline(task))
    return SelectionResult(
        task_id=task.task_id,
        family=task.family,
        success=best.success,
        score=best.score,
        baseline_score=base_score,
        utility=best.utility,
        attempts=len(pool),
        skill_id=best.skill_id,
        score_model=1.0,
        semantic=1.0,
        fallback=best.fallback,
        negative_transfer=False,
        tried=[best.skill_id],
    )
