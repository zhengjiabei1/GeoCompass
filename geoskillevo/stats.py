"""Dependency-free statistics for the SkillEvo experiments."""
from __future__ import annotations

import math
import random


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    centre = mean(values)
    return math.sqrt(sum((value - centre) ** 2 for value in values) / (len(values) - 1))


def bootstrap_ci(values: list[float], statistic=mean, samples: int = 2000,
                 confidence: float = 0.95, seed: int = 17) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    estimates = []
    n = len(values)
    for _ in range(samples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(statistic(draw))
    estimates.sort()
    low = int((1 - confidence) / 2 * samples)
    high = int((1 + confidence) / 2 * samples) - 1
    return (estimates[max(0, low)], estimates[min(samples - 1, high)])


def paired_bootstrap(left: list[float], right: list[float], samples: int = 2000,
                     seed: int = 19) -> tuple[float, float, float]:
    """Bootstrap CI for the paired difference ``left - right``."""
    if len(left) != len(right):
        raise ValueError("paired samples must have equal length")
    rng = random.Random(seed)
    n = len(left)
    diffs = [a - b for a, b in zip(left, right)]
    draws = []
    for _ in range(samples):
        draws.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    low = draws[int(0.025 * samples)]
    high = draws[int(0.975 * samples) - 1]
    return (mean(diffs), low, high)


def auroc(scores: list[float], labels: list[int]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[index]]:
            end += 1
        rank = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[order[position]] = rank
        index = end + 1
    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def auprc(scores: list[float], labels: list[int]) -> float:
    """Area under the precision-recall curve with tie-aware thresholds."""
    positives = sum(labels)
    if positives == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    tp = 0
    fp = 0
    previous_recall = 0.0
    area = 0.0
    index = 0
    while index < len(order):
        score = scores[order[index]]
        while index < len(order) and scores[order[index]] == score:
            if labels[order[index]]:
                tp += 1
            else:
                fp += 1
            index += 1
        recall = tp / positives
        precision = tp / max(1, tp + fp)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def brier(scores: list[float], labels: list[int]) -> float:
    return mean([(score - label) ** 2 for score, label in zip(scores, labels)])


def nll(scores: list[float], labels: list[int]) -> float:
    return -mean([
        math.log(min(1 - 1e-9, max(1e-9, score if label else 1 - score)))
        for score, label in zip(scores, labels)
    ])


def reliability(scores: list[float], labels: list[int], bins: int = 10) -> list[dict[str, float]]:
    rows = []
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        members = [
            (score, label) for score, label in zip(scores, labels)
            if low <= score < high or (index == bins - 1 and score == 1.0)
        ]
        if not members:
            continue
        confidence = mean([score for score, _ in members])
        accuracy = mean([float(label) for _, label in members])
        rows.append({
            "bin": index,
            "count": len(members),
            "confidence": confidence,
            "accuracy": accuracy,
            "gap": abs(confidence - accuracy),
        })
    return rows


def ece(scores: list[float], labels: list[int], bins: int = 10) -> float:
    rows = reliability(scores, labels, bins)
    total = sum(row["count"] for row in rows)
    return sum(row["count"] / total * row["gap"] for row in rows) if total else 0.0


def mcnemar(left: list[int], right: list[int]) -> dict[str, float]:
    """Exact two-sided McNemar test for paired binary outcomes."""
    if len(left) != len(right):
        raise ValueError("paired outcomes must have equal length")
    b = sum(1 for a, r in zip(left, right) if a == 1 and r == 0)
    c = sum(1 for a, r in zip(left, right) if a == 0 and r == 1)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0}
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return {"b": float(b), "c": float(c), "p": min(1.0, 2 * tail)}


def sign_permutation(left: list[float], right: list[float], samples: int = 5000,
                     seed: int = 23) -> float:
    """Two-sided paired sign-flip permutation test for a mean difference."""
    if len(left) != len(right):
        raise ValueError("paired samples must have equal length")
    diffs = [a - b for a, b in zip(left, right)]
    observed = abs(mean(diffs))
    rng = random.Random(seed)
    count = 0
    for _ in range(samples):
        draw = [diff if rng.random() < 0.5 else -diff for diff in diffs]
        if abs(mean(draw)) >= observed:
            count += 1
    return (count + 1) / (samples + 1)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]
