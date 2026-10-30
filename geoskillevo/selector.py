"""Calibrated discriminators that rerank historical skills."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random

from .features import FEATURE_NAMES

# Structural feature set used by the default discriminator.  These features
# describe what a task requires and what a skill provides; they are available
# without executing the skill.
STRUCTURAL_FEATURES = (
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
    "semantic_x_target",
    "input_x_mechanism",
    "family_x_target",
    "target_mismatch",
    "seasonal_fit",
    "resolution_fit",
)

# Curated feature set used by the memory-enabled variant.  Constant or
# near-collinear features are omitted to reduce variance on the small
# historical corpus.
CURATED_FEATURES = (
    "bias",
    "semantic",
    "semantic_rank",
    "target_match",
    "family_match",
    "input_match",
    "mechanism_match",
    "band_coverage",
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
    "success_x_target",
)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-min(60.0, value))
        return 1.0 / (1.0 + z)
    z = math.exp(max(-60.0, value))
    return z / (1.0 + z)


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(vector)
    rows = [list(matrix[i]) + [vector[i]] for i in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) < 1e-12:
            continue
        rows[column], rows[pivot] = rows[pivot], rows[column]
        factor = rows[column][column]
        rows[column] = [value / factor for value in rows[column]]
        for row in range(n):
            if row == column:
                continue
            multiplier = rows[row][column]
            if multiplier:
                rows[row] = [
                    a - multiplier * b for a, b in zip(rows[row], rows[column])
                ]
    return [rows[i][n] for i in range(n)]


class LogisticSelector:
    """L2-regularised logistic regression fitted with IRLS.

    The model is intentionally small: a few dozen interpretable features and a
    handful of parameters.  It is a discriminator, not a fine-tuned LLM.
    """

    def __init__(self, l2: float = 1.5, max_iter: float = 40, name: str = "logistic",
                 features: list[str] | tuple[str, ...] | None = None):
        self.l2 = l2
        self.max_iter = int(max_iter)
        self.name = name
        self.feature_names = tuple(features) if features else STRUCTURAL_FEATURES
        self.weights: list[float] = []
        self.bias = 0.0
        self.means: list[float] = []
        self.stds: list[float] = []
        self.fitted = False

    def _matrix(self, rows: list[dict[str, float]]) -> list[list[float]]:
        return [[float(row[name]) for name in self.feature_names] for row in rows]

    def fit(self, rows: list[dict[str, float]], labels: list[int],
            sample_weight: list[float] | None = None) -> "LogisticSelector":
        if not rows:
            raise ValueError("cannot fit on an empty dataset")
        if len(rows) != len(labels):
            raise ValueError("rows and labels differ in length")
        matrix = self._matrix(rows)
        n = len(matrix)
        d = len(self.feature_names)
        self.means = [sum(row[j] for row in matrix) / n for j in range(d)]
        self.stds = []
        for j in range(d):
            variance = sum((row[j] - self.means[j]) ** 2 for row in matrix) / n
            self.stds.append(max(1e-6, math.sqrt(variance)))
        # The first feature is the bias; keep it unscaled.
        self.means[0], self.stds[0] = 0.0, 1.0
        scaled = [
            [(row[j] - self.means[j]) / self.stds[j] for j in range(d)]
            for row in matrix
        ]
        weights = sample_weight or [1.0] * n
        positives = sum(1 for label in labels if label == 1)
        negatives = n - positives
        balanced = []
        for label, weight in zip(labels, weights):
            factor = (n / (2.0 * max(1, positives))) if label == 1 else (n / (2.0 * max(1, negatives)))
            balanced.append(weight * factor)
        beta = [0.0] * d
        for _ in range(self.max_iter):
            probabilities = [_sigmoid(sum(b * x for b, x in zip(beta, row))) for row in scaled]
            gradient = [0.0] * d
            hessian = [[0.0] * d for _ in range(d)]
            for row, label, probability, weight in zip(scaled, labels, probabilities, balanced):
                residual = (probability - label) * weight
                variance = max(1e-6, probability * (1.0 - probability)) * weight
                for j in range(d):
                    gradient[j] += residual * row[j]
                    if row[j] == 0.0:
                        continue
                    for k in range(j, d):
                        value = variance * row[j] * row[k]
                        hessian[j][k] += value
                        if j != k:
                            hessian[k][j] += value
            for j in range(d):
                gradient[j] += self.l2 * beta[j]
                hessian[j][j] += self.l2 if j != 0 else 1e-8
            step = _solve(hessian, gradient)
            beta = [value - delta for value, delta in zip(beta, step)]
            if max(abs(delta) for delta in step) < 1e-7:
                break
        self.weights = beta
        self.bias = 0.0  # absorbed by the bias feature
        self.fitted = True
        return self

    def decision(self, row: dict[str, float]) -> float:
        vector = [float(row[name]) for name in self.feature_names]
        scaled = [
            (vector[j] - self.means[j]) / self.stds[j] if self.fitted else vector[j]
            for j in range(len(vector))
        ]
        return sum(weight * value for weight, value in zip(self.weights, scaled))

    def predict_proba(self, row: dict[str, float]) -> float:
        return _sigmoid(self.decision(row))

    def predict_many(self, rows: list[dict[str, float]]) -> list[float]:
        return [self.predict_proba(row) for row in rows]

    def coefficients(self) -> dict[str, float]:
        return {name: round(value, 5) for name, value in zip(self.feature_names, self.weights)}


class TemperatureCalibrator:
    """Single-parameter temperature scaling on the logit."""

    def __init__(self):
        self.temperature = 1.0
        self.fitted = False

    @staticmethod
    def _nll(logits: list[float], labels: list[int], temperature: float) -> float:
        total = 0.0
        for logit, label in zip(logits, labels):
            probability = _sigmoid(logit / temperature)
            probability = min(1 - 1e-9, max(1e-9, probability))
            total -= math.log(probability if label else 1 - probability)
        return total / max(1, len(labels))

    def fit(self, probabilities: list[float], labels: list[int]) -> "TemperatureCalibrator":
        logits = [math.log(max(1e-9, p) / max(1e-9, 1 - p)) for p in probabilities]
        low, high = 0.05, 10.0
        golden = (math.sqrt(5) - 1) / 2
        left, right = low, high
        c, d = right - golden * (right - left), left + golden * (right - left)
        for _ in range(80):
            if self._nll(logits, labels, c) < self._nll(logits, labels, d):
                right, d = d, c
                c = right - golden * (right - left)
            else:
                left, c = c, d
                d = left + golden * (right - left)
        self.temperature = max(0.05, (left + right) / 2)
        self.fitted = True
        return self

    def transform(self, probability: float) -> float:
        logit = math.log(max(1e-9, probability) / max(1e-9, 1 - probability))
        return _sigmoid(logit / self.temperature)


class IsotonicCalibrator:
    """Pool-adjacent-violators calibration with linear interpolation."""

    def __init__(self):
        self.x: list[float] = []
        self.y: list[float] = []
        self.fitted = False

    def fit(self, probabilities: list[float], labels: list[int]) -> "IsotonicCalibrator":
        pairs = sorted(zip(probabilities, labels), key=lambda pair: pair[0])
        blocks: list[list[float]] = []  # [sum_y, count, min_x, max_x]
        for probability, label in pairs:
            blocks.append([float(label), 1.0, probability, probability])
            while len(blocks) > 1 and (
                blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]
            ):
                b = blocks.pop()
                a = blocks.pop()
                blocks.append([a[0] + b[0], a[1] + b[1], a[2], b[3]])
        self.x = [block[2] for block in blocks]
        self.y = [block[0] / block[1] for block in blocks]
        self.fitted = True
        return self

    def transform(self, probability: float) -> float:
        if not self.fitted or not self.x:
            return probability
        if probability <= self.x[0]:
            return self.y[0]
        if probability >= self.x[-1]:
            return self.y[-1]
        for index in range(1, len(self.x)):
            if probability <= self.x[index]:
                left, right = self.x[index - 1], self.x[index]
                ratio = (probability - left) / max(1e-12, right - left)
                return self.y[index - 1] * (1 - ratio) + self.y[index] * ratio
        return self.y[-1]


class CalibratedSelector:
    """Base discriminator plus a post-hoc calibrator."""

    def __init__(self, base: LogisticSelector, method: str = "temperature"):
        self.base = base
        self.method = method
        self.calibrator = TemperatureCalibrator() if method == "temperature" else IsotonicCalibrator()
        self.name = f"{base.name}+{method}"

    def fit_calibrator(self, rows: list[dict[str, float]], labels: list[int]) -> "CalibratedSelector":
        probabilities = self.base.predict_many(rows)
        self.calibrator.fit(probabilities, labels)
        return self

    def predict_proba(self, row: dict[str, float]) -> float:
        probability = self.base.predict_proba(row)
        if isinstance(self.calibrator, TemperatureCalibrator):
            return self.calibrator.transform(probability)
        return self.calibrator.transform(probability)


class FixedSelector:
    """Baselines that do not use the learned discriminator."""

    def __init__(self, name: str, source: str = "semantic", prior: float = 0.5, seed: int = 0):
        self.name = name
        self.source = source
        self.prior = prior
        self.random = random.Random(seed)

    def predict_proba(self, row: dict[str, float]) -> float:
        if self.source == "semantic":
            return float(row["semantic"])
        if self.source == "frequency":
            return float(row["history_success"])
        if self.source == "random":
            return self.random.random()
        return self.prior


@dataclass
class Ranked:
    skill: object
    score: float
    semantic: float
    features: dict[str, float]


def rank_candidates(task, candidates, model, stage, memory, step, library_size,
                    transfer_scores: dict[str, float] | None = None) -> list[Ranked]:
    from .features import extract_features

    ranked: list[Ranked] = []
    count = len(candidates)
    for index, (skill, semantic) in enumerate(candidates):
        stats = memory.stats(skill.skill_id, task.family) if memory is not None else None
        features = extract_features(
            task=task,
            skill=skill,
            stage=stage,
            semantic=semantic,
            semantic_rank=index,
            candidate_count=count,
            memory=stats,
            step=step,
            library_size=library_size,
        )
        score = model.predict_proba(features)
        if transfer_scores is not None:
            score *= transfer_scores.get(skill.skill_id, 1.0)
        ranked.append(Ranked(skill=skill, score=score, semantic=semantic, features=features))
    ranked.sort(key=lambda item: (item.score, item.semantic), reverse=True)
    return ranked
