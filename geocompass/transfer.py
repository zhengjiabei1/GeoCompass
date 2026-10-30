"""Cross-task transfer diagnosis with uncertainty-aware gating."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .features import _band_coverage, _family_mechanism_match
from .types import Skill, TaskSpec


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return h


def beta_cdf(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    log_term = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1 - x)
    )
    term = math.exp(log_term)
    if x < (a + 1) / (a + b + 2):
        return term * _betacf(a, b, x) / a
    return 1.0 - term * _betacf(b, a, 1 - x) / b


def beta_ppf(probability: float, a: float, b: float) -> float:
    """Bisection inverse CDF; adequate for the small posteriors used here."""
    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2
        if beta_cdf(middle, a, b) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


@dataclass
class TransferVerdict:
    skill_id: str
    source_family: str
    target_family: str
    probes: int
    positive: int
    success: int
    mean_utility: float
    compatibility: float
    score: float
    lower: float
    upper: float
    status: str  # transferable | uncertain | blocked
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "source_family": self.source_family,
            "target_family": self.target_family,
            "probes": self.probes,
            "positive": self.positive,
            "success": self.success,
            "mean_utility": round(self.mean_utility, 5),
            "compatibility": round(self.compatibility, 5),
            "score": round(self.score, 5),
            "lower": round(self.lower, 5),
            "upper": round(self.upper, 5),
            "status": self.status,
            "reason": self.reason,
        }


def compatibility(skill: Skill, task: TaskSpec) -> float:
    target = 1.0 if task.target in skill.targets else 0.0
    family = 1.0 if task.family == skill.source_family else 0.0
    input_match = 1.0 if task.input_kind == skill.input_kind else 0.0
    mechanism = _family_mechanism_match(task, skill)
    bands = _band_coverage(task, skill)
    return (
        0.30 * target
        + 0.15 * family
        + 0.15 * input_match
        + 0.30 * mechanism
        + 0.10 * bands
    )


def diagnose(
    skill: Skill,
    task_family: str,
    probe_tasks: list[TaskSpec],
    probe_outcomes: list[float],
    *,
    min_probes: int = 2,
    block_upper: float = 0.45,
    block_score: float = 0.32,
    uncertain_lower: float = 0.38,
) -> TransferVerdict:
    """Estimate whether a skill transfers to a target family.

    ``probe_outcomes`` are utility deltas measured on a small held-out probe
    split of the target family.  Priors are set by structural compatibility, so
    a skill with no probes can still be treated as uncertain rather than
    silently accepted.
    """
    target_tasks = [task for task in probe_tasks if task.family == task_family] or probe_tasks
    compat = sum(compatibility(skill, task) for task in target_tasks) / max(1, len(target_tasks))
    positive = sum(1 for value in probe_outcomes if value > 0.005)
    successes = sum(1 for value in probe_outcomes if value >= 0.02)
    n = len(probe_outcomes)
    alpha = 1.0 + 2.0 * compat + positive
    beta = 1.0 + 2.0 * (1.0 - compat) + (n - positive)
    mean = alpha / (alpha + beta)
    lower = beta_ppf(0.10, alpha, beta)
    upper = beta_ppf(0.90, alpha, beta)
    mean_utility = sum(probe_outcomes) / n if n else 0.0
    positive_rate = positive / n if n else 0.0
    normalised_utility = max(0.0, min(1.0, (mean_utility + 0.10) / 0.60))
    score = 0.60 * compat + 0.40 * normalised_utility
    if n < min_probes:
        if compat >= 0.60:
            status, reason = "uncertain", "insufficient probes but strong structural prior"
        else:
            status, reason = "blocked", "insufficient probes and weak structural prior"
    elif compat < 0.35 and mean_utility < 0.25:
        status, reason = "blocked", "low compatibility and weak probe utility"
    elif mean_utility >= 0.25 and positive_rate >= 0.75:
        status, reason = "transferable", "strong empirical transfer overrides family prior"
    elif compat >= 0.55 and mean_utility > 0.0:
        status, reason = "transferable", "structural and empirical evidence agree"
    elif compat >= 0.35 and mean_utility > 0.10:
        status, reason = "transferable", "partial structural compatibility with positive transfer"
    else:
        status, reason = "uncertain", "mixed structural and empirical evidence"
    return TransferVerdict(
        skill_id=skill.skill_id,
        source_family=skill.source_family,
        target_family=task_family,
        probes=n,
        positive=positive,
        success=successes,
        mean_utility=mean_utility,
        compatibility=compat,
        score=score,
        lower=lower,
        upper=upper,
        status=status,
        reason=reason,
    )


def gate_scores(verdicts: list[TransferVerdict], strategy: str) -> dict[str, float]:
    """Return multiplicative score factors for a gating strategy."""
    scores: dict[str, float] = {}
    for verdict in verdicts:
        if strategy == "none":
            scores[verdict.skill_id] = 1.0
        elif strategy == "semantic":
            scores[verdict.skill_id] = 1.0 if verdict.compatibility >= 0.55 else 0.05
        elif strategy == "outcome":
            scores[verdict.skill_id] = 1.0 if verdict.mean_utility > 0.0 else 0.05
        elif strategy == "proposed":
            # Hard-block skills with both weak structure and weak probe
            # evidence; strongly evidenced cross-family skills can survive.
            if verdict.status == "blocked":
                scores[verdict.skill_id] = 0.0
            elif verdict.status == "uncertain":
                scores[verdict.skill_id] = 0.15
            else:
                scores[verdict.skill_id] = max(0.5, verdict.score)
        else:
            raise ValueError(f"unknown gate strategy: {strategy}")
    return scores


def diagnose_matrix(
    skills: list[Skill],
    target_families: list[str],
    probes: dict[tuple[str, str], list[float]],
    probe_tasks: dict[str, list[TaskSpec]],
) -> dict[str, dict[str, TransferVerdict]]:
    matrix: dict[str, dict[str, TransferVerdict]] = {}
    for skill in skills:
        matrix[skill.skill_id] = {}
        for family in target_families:
            outcomes = probes.get((skill.skill_id, family), [])
            matrix[skill.skill_id][family] = diagnose(
                skill, family, probe_tasks.get(family, []), outcomes
            )
    return matrix
