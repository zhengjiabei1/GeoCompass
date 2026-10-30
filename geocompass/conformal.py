"""Risk-controlled reuse gating.

The agent has three options for a retrieved skill: commit to it, fall back to
the generic pipeline, or (with a budget) try another candidate.  Committing is
useful only while the probability of *negative transfer* stays below a level the
user is willing to accept.  Picking that operating point by eyeballing a
validation curve gives no guarantee on unseen tasks.

This module implements split conformal risk control for the commit/abstain
decision.  The gate is a one-sided threshold policy ``commit iff score >= t``;
the loss of a task is the negative-transfer indicator of the executed skill, and
abstaining is defined to have zero loss because the fallback is the baseline
itself.  Because the family of policies is nested in ``t``, the empirical risk
is monotone and the calibration is a single sweep.  A Hoeffding bound with a
union-bound correction over the threshold grid gives a finite-sample guarantee
on the negative-transfer rate at confidence ``1 - delta``.

The procedure and its references are recorded in ``docs/RELATED_WORK.md``
and ``docs/BENCHMARK.md``; the implementation is deliberately dependency-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from .transfer import beta_ppf

NEGATIVE_TRANSFER_MARGIN = -0.02


def binom_upper(failures: int, trials: int, delta: float) -> float:
    """Clopper-Pearson upper confidence bound for a binomial rate.

    A Bernoulli loss is bounded, so the exact binomial tail is admissible and
    much tighter than a Hoeffding bound at the sample sizes this benchmark can
    afford.  ``beta_ppf`` is the inverse Beta quantile already used by the
    transfer diagnosis.
    """
    if trials <= 0:
        return 1.0
    if failures >= trials:
        return 1.0
    return beta_ppf(1.0 - delta, failures + 1, trials - failures)


def required_calibration_size(alpha: float, delta: float,
                              grid_size: int = 20) -> int:
    """Smallest calibration set on which a zero-loss threshold can be certified.

    The gate needs ``binom_upper(0, n, delta / grid) <= alpha``.  Reporting this
    number keeps the guarantee honest instead of silently returning an
    infeasible gate: a 5% risk target at 10% confidence over a 20-point grid
    needs about 103 held-out tasks with no observed negative transfer.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    per_candidate = max(1e-12, delta / max(1, grid_size))
    return int(math.ceil(math.log(per_candidate) / math.log(1.0 - alpha)))


@dataclass
class Certificate:
    """Outcome of one calibration run."""

    alpha: float
    delta: float
    threshold: float
    calibration_size: int
    risk: float
    upper_bound: float
    coverage: float
    feasible: bool = True
    grid_size: int = 0

    def to_dict(self) -> dict:
        return {
            "alpha": round(self.alpha, 4),
            "delta": round(self.delta, 4),
            "threshold": round(self.threshold, 5),
            "calibration_size": self.calibration_size,
            "risk": round(self.risk, 5),
            "upper_bound": round(self.upper_bound, 5),
            "coverage": round(self.coverage, 5),
            "feasible": self.feasible,
            "grid_size": self.grid_size,
        }


@dataclass
class ConformalGate:
    """Threshold policy calibrated with a finite-sample risk guarantee."""

    alpha: float = 0.05
    delta: float = 0.10
    grid_size: int = 40
    certificate: Certificate | None = None
    _threshold: float = 1.0
    _grid: list[float] = field(default_factory=list)

    def _candidates(self, scores: list[float]) -> list[float]:
        if not scores:
            return [1.0]
        ordered = sorted(scores)
        size = max(2, min(self.grid_size, len(ordered)))
        grid = []
        for position in range(size):
            index = round(position * (len(ordered) - 1) / (size - 1))
            grid.append(ordered[index])
        grid.append(ordered[-1] + 1e-12)
        return sorted(set(grid))

    def fit(self, scores: list[float], losses: list[float]) -> Certificate:
        """Choose the most permissive threshold whose upper bound meets ``alpha``."""
        if len(scores) != len(losses):
            raise ValueError("scores and losses must have equal length")
        if not scores:
            self.certificate = Certificate(self.alpha, self.delta, 1.0, 0, 0.0, 1.0, 0.0, False, 0)
            self._threshold = 1.0
            return self.certificate
        grid = self._candidates(scores)
        self._grid = grid
        count = len(scores)
        per_candidate_delta = self.delta / max(1, len(grid))
        best: Certificate | None = None
        for threshold in grid:
            committed = [
                loss for score, loss in zip(scores, losses) if score >= threshold
            ]
            if not committed:
                continue
            failures = int(round(sum(committed)))
            risk = failures / len(committed)
            coverage = len(committed) / count
            bound = binom_upper(failures, len(committed), per_candidate_delta)
            feasible = bound <= self.alpha
            if best is None:
                best = Certificate(self.alpha, self.delta, threshold, count, risk, bound,
                                   coverage, feasible, len(grid))
                continue
            better = (feasible, coverage) > (best.feasible, best.coverage)
            if better:
                best = Certificate(self.alpha, self.delta, threshold, count, risk, bound,
                                   coverage, feasible, len(grid))
        if best is None or not best.feasible:
            best = Certificate(self.alpha, self.delta, 1.0, count, 0.0, 1.0, 0.0, False,
                               len(grid))
            self._threshold = math.inf
        else:
            self._threshold = best.threshold
        self.certificate = best
        return best

    def commit(self, score: float) -> bool:
        return score >= self._threshold

    def evaluate(self, scores: list[float], losses: list[float],
                 delta: float | None = None) -> dict:
        """Held-out risk and coverage, with a Clopper-Pearson upper bound."""
        delta = 0.1 if delta is None else delta
        if not scores:
            return {"coverage": 0.0, "risk": 0.0, "upper": 1.0, "committed": 0}
        committed = [loss for score, loss in zip(scores, losses) if self.commit(score)]
        if not committed:
            return {"coverage": 0.0, "risk": 0.0, "upper": 1.0, "committed": 0}
        risk = sum(committed) / len(committed)
        failures = int(round(sum(committed)))
        upper = binom_upper(failures, len(committed), delta)
        return {
            "coverage": len(committed) / len(scores),
            "risk": risk,
            "upper": upper,
            "committed": len(committed),
        }


def fixed_threshold_gate(scores: list[float], losses: list[float], alpha: float) -> ConformalGate:
    """Baseline: pick the threshold whose *empirical* risk is exactly ``alpha``.

    This is what a validation-curve sweep does: it scans every prefix of the
    score-ordered calibration set and keeps the widest prefix whose observed
    risk is still acceptable.  No confidence correction is applied, so the
    threshold is optimistic relative to the conformal gate, which is the point
    of reporting both.
    """
    gate = ConformalGate(alpha=alpha, delta=1.0, grid_size=200)
    ordered = sorted(zip(scores, losses), key=lambda pair: pair[0], reverse=True)
    total = len(ordered)
    failures = 0
    best: tuple[float, float, float] | None = None
    for index, (score, loss) in enumerate(ordered, start=1):
        failures += int(loss)
        risk = failures / index
        if risk <= alpha:
            best = (score, risk, index / total)
    if best is None:
        gate._threshold = math.inf
        gate.certificate = Certificate(
            alpha=alpha, delta=1.0, threshold=1.0, calibration_size=total,
            risk=0.0, upper_bound=math.nan, coverage=0.0, feasible=True, grid_size=total)
        return gate
    threshold, risk, coverage = best
    gate._threshold = threshold
    gate.certificate = Certificate(
        alpha=alpha, delta=1.0, threshold=threshold, calibration_size=total,
        risk=risk, upper_bound=math.nan, coverage=coverage, feasible=True, grid_size=total)
    return gate


def negative_transfer(losses_from_utility: list[float], margin: float = NEGATIVE_TRANSFER_MARGIN) -> list[float]:
    """Indicator that reusing a skill was worse than the baseline by ``margin``."""
    return [1.0 if utility < margin else 0.0 for utility in losses_from_utility]
