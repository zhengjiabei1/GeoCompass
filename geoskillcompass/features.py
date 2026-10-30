"""Hashed embeddings and interpretable selector features."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re

from .types import BANDS, Skill, TaskSpec

TOKEN_RE = re.compile(r"[a-z0-9]+")

FEATURE_NAMES = (
    "bias",
    "semantic",
    "semantic_rank",
    "target_match",
    "family_match",
    "input_match",
    "mechanism_match",
    "band_coverage",
    "stage_refinement",
    "bitemporal_skill",
    "adaptive",
    "cost",
    "history_success",
    "history_utility",
    "history_std",
    "history_strength",
    "history_recency",
    "history_coverage",
    "history_duplicates",
    "novelty",
    "history_recent",
    "history_family_success",
    "history_family_strength",
    "semantic_x_target",
    "input_x_mechanism",
    "family_x_target",
    "success_x_target",
    "target_mismatch",
    "seasonal_fit",
    "resolution_fit",
    "library_coverage",
)


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class HashedEmbedding:
    """Signed feature hashing; deterministic and dependency-free."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vector(self, tokens: list[str]) -> list[float]:
        vector = [0.0] * self.dim
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def skill(self, skill: Skill) -> list[float]:
        # Surface-form embedding only.  The discriminator gets structured
        # metadata (target, input kind, mechanism) separately; this mirrors the
        # realistic case where a dense retriever sees skill descriptions but
        # not the task's hidden compatibility structure.
        parts = [skill.name, skill.description]
        for op in skill.ops:
            parts.append(op.op.replace("_", " "))
            for key, value in op.params:
                if isinstance(value, str):
                    parts.append(value.replace("_", " "))
                elif isinstance(value, list):
                    parts.extend(str(item).replace("_", " ") for item in value)
        return self._vector(_tokens(" ".join(parts)))

    def task(self, task: TaskSpec) -> list[float]:
        words = [task.family.replace("_", " "), task.target.replace("_", " "),
                 task.visible()["evidence"].replace("_", " "), task.season, task.region,
                 "single date" if not task.bitemporal else "bitemporal two dates",
                 "classification map" if not task.bitemporal else "change detection map"]
        return self._vector(_tokens(" ".join(words)))


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


@dataclass
class MemoryStats:
    """Aggregated historical evidence for one skill (provided by memory.py)."""

    uses: int = 0
    successes: int = 0
    utility_sum: float = 0.0
    utility_sq_sum: float = 0.0
    last_seen: int = -1
    duplicates: int = 0
    coverage: int = 0
    family_uses: int = 0
    family_successes: int = 0
    retained: int = 0
    total_seen: int = 0
    recent_uses: int = 0
    recent_successes: int = 0


def _family_mechanism_match(task: TaskSpec, skill: Skill) -> float:
    names = [op.op for op in skill.ops]
    index_names = {
        op.get("name") or op.get("index")
        for op in skill.ops
        if op.op in {"index", "temporal_diff", "temporal_ratio"}
    }
    if task.family in {"landcover"}:
        classify = [op for op in skill.ops if op.op == "classify"]
        if not classify:
            return 0.0
        features = classify[0].get("features", [])
        return 1.0 if len(features) >= 6 else 0.35
    if task.family == "crop":
        return 1.0 if "classify" in names and index_names else (0.5 if "classify" in names else 0.0)
    if task.family in {"water", "flood"}:
        return 1.0 if index_names & {"ndwi", "ndmi"} else 0.0
    if task.family == "change":
        if "spectral_change" in names:
            return 1.0
        return 1.0 if "temporal_diff" in names and index_names & {"ndvi", "ndmi"} else 0.0
    if task.family == "urban":
        if "spectral_change" in names:
            return 1.0
        return 1.0 if "temporal_diff" in names and index_names & {"ndbi", "bsi"} else 0.0
    if task.family == "burn":
        if "spectral_change" in names:
            return 1.0
        return 1.0 if "temporal_diff" in names and index_names & {"nbr", "ndvi"} else 0.0
    return 0.0


def _band_coverage(task: TaskSpec, skill: Skill) -> float:
    available = set(BANDS)
    return len(set(skill.required_bands) & available) / max(1, len(skill.required_bands))


def _seasonal_fit(task: TaskSpec, skill: Skill) -> float:
    # Skills induced in the same season are slightly more likely to transfer
    # because threshold behaviour follows phenology.
    source_seasons = skill.params.get("source_seasons", ())
    if not source_seasons:
        return 0.5
    return 1.0 if task.season in source_seasons else 0.4


def _resolution_fit(task: TaskSpec, skill: Skill) -> float:
    if "min_size" in (skill.params or {}):
        target = int(skill.params.get("min_size", 12))
        expected = max(4, int(round((task.resolution / 10.0) ** 2 * 12)))
        return 1.0 / (1.0 + abs(target - expected) / max(4.0, expected))
    return 0.7


def _library_coverage(task: TaskSpec, library_size: int) -> float:
    return min(1.0, library_size / 20.0)


def _history(stats: MemoryStats | None) -> dict[str, float]:
    if stats is None or stats.uses == 0:
        return {
            "history_success": 0.5,
            "history_utility": 0.0,
            "history_std": 0.25,
            "history_strength": 0.0,
            "history_recency": 0.0,
            "history_coverage": 0.0,
            "history_duplicates": 0.0,
            "novelty": 1.0,
            "history_recent": 0.5,
            "history_family_success": 0.5,
            "history_family_strength": 0.0,
        }
    mean = stats.utility_sum / stats.uses
    variance = max(0.0, stats.utility_sq_sum / stats.uses - mean * mean)
    return {
        "history_success": (stats.successes + 1.0) / (stats.uses + 2.0),
        "history_utility": mean,
        "history_std": math.sqrt(variance),
        "history_strength": stats.uses / (stats.uses + 4.0),
        "history_recency": math.exp(-max(0, stats.total_seen - stats.last_seen) / 40.0)
        if stats.last_seen >= 0 else 0.0,
        "history_coverage": min(1.0, stats.coverage / 5.0),
        "history_duplicates": stats.duplicates / max(1, stats.retained),
        "novelty": 1.0 / (1.0 + stats.uses),
        "history_recent": (stats.recent_successes + 1.0) / (stats.recent_uses + 2.0),
        "history_family_success": (stats.family_successes + 1.0) / (stats.family_uses + 2.0),
        "history_family_strength": stats.family_uses / (stats.family_uses + 2.0),
    }


def extract_features(
    *,
    task: TaskSpec,
    skill: Skill,
    stage: str,
    semantic: float,
    semantic_rank: int,
    candidate_count: int,
    memory: MemoryStats | None,
    step: int,
    library_size: int,
) -> dict[str, float]:
    target_match = 1.0 if task.target in skill.targets else 0.0
    family_match = 1.0 if task.family == skill.source_family else 0.0
    input_match = 1.0 if task.input_kind == skill.input_kind else 0.0
    mechanism = _family_mechanism_match(task, skill)
    history = _history(memory)
    rank_score = 1.0 - (semantic_rank / max(1, candidate_count - 1))
    features = {
        "bias": 1.0,
        "semantic": semantic,
        "semantic_rank": rank_score,
        "target_match": target_match,
        "family_match": family_match,
        "input_match": input_match,
        "mechanism_match": mechanism,
        "band_coverage": _band_coverage(task, skill),
        "stage_refinement": 1.0 if stage == "refinement" else 0.0,
        "bitemporal_skill": 1.0 if skill.input_kind == "bitemporal" else 0.0,
        "adaptive": 1.0 if skill.adaptive else 0.0,
        "cost": min(1.0, len(skill.ops) / 4.0),
        **history,
        "semantic_x_target": semantic * target_match,
        "input_x_mechanism": input_match * mechanism,
        "family_x_target": family_match * target_match,
        "success_x_target": history["history_success"] * target_match,
        "target_mismatch": 1.0 - target_match,
        "seasonal_fit": _seasonal_fit(task, skill),
        "resolution_fit": _resolution_fit(task, skill),
        "library_coverage": _library_coverage(task, library_size),
    }
    return {name: float(features[name]) for name in FEATURE_NAMES}
