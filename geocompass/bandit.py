"""Online skill selection as a contextual bandit.

Reusing a skill is not only a prediction problem.  The agent decides whether to
commit to a candidate or to spend an execution on a less certain one, and it
learns from the outcome either way.  That is a contextual bandit: the context is
the task-skill feature vector, the arms are the retrieved candidates, and the
reward is the utility of the executed skill relative to the generic baseline.

The benchmark previously evaluated the selector offline and only replayed a
fixed policy online.  This module adds the online policies that make the
explore/commit trade-off explicit, together with cumulative regret against the
best candidate in hindsight, so that a myopic reranker can be compared with
policies that deliberately pay for information.

All policies share one linear reward model over the structural feature set and
differ only in how they turn the posterior into an action.  Sharing the model
across arms keeps the parameter count independent of the library size, which
matters when the stream is short.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

from .selector import STRUCTURAL_FEATURES, _solve

FEATURES = tuple(STRUCTURAL_FEATURES)


def vector(row: dict[str, float]) -> list[float]:
    """Fixed-order feature vector, so every policy sees the same context."""
    return [float(row.get(name, 0.0)) for name in FEATURES]


def reward_from_utility(utility: float) -> float:
    """Map a utility in [-1, 1] to a bounded reward in [0, 1]."""
    return max(0.0, min(1.0, 0.5 * (utility + 1.0)))


@dataclass
class LinearPosterior:
    """Bayesian linear regression over a shared reward model."""

    dimension: int = len(FEATURES)
    ridge: float = 1.0
    noise: float = 0.25
    _a: list[list[float]] = field(default_factory=list)
    _b: list[float] = field(default_factory=list)
    _mean: list[float] = field(default_factory=list)
    _observations: int = 0

    def __post_init__(self) -> None:
        if not self._a:
            self.reset()

    def reset(self) -> None:
        self._a = [[self.ridge if i == j else 0.0 for j in range(self.dimension)]
                   for i in range(self.dimension)]
        self._b = [0.0] * self.dimension
        self._mean = [0.0] * self.dimension
        self._observations = 0

    @property
    def observations(self) -> int:
        return self._observations

    @property
    def mean(self) -> list[float]:
        return list(self._mean)

    def _refresh(self) -> None:
        self._mean = _solve([row[:] for row in self._a], list(self._b))

    def expected(self, x: list[float]) -> float:
        return sum(weight * value for weight, value in zip(self._mean, x))

    def leverage(self, x: list[float]) -> float:
        """``x^T A^{-1} x``, the posterior variance up to the noise scale."""
        solution = _solve([row[:] for row in self._a], list(x))
        return sum(a * b for a, b in zip(x, solution))

    def sample(self, rng: random.Random, scale: float = 1.0) -> list[float]:
        """Draw a coefficient vector from the Gaussian posterior.

        A full matrix square root is unnecessary here: the policies only need a
        consistent random projection, so the draw uses the Cholesky factor of
        ``A^{-1}`` computed once per update.
        """
        inverse = _inverse([row[:] for row in self._a])
        factor = _cholesky(inverse)
        noise = [rng.gauss(0.0, 1.0) for _ in range(self.dimension)]
        return [
            self._mean[i] + scale * math.sqrt(self.noise) * sum(factor[i][j] * noise[j]
                                                               for j in range(i + 1))
            for i in range(self.dimension)
        ]

    def update(self, x: list[float], reward: float) -> None:
        for i in range(self.dimension):
            if x[i] == 0.0:
                continue
            for j in range(self.dimension):
                if x[j] != 0.0:
                    self._a[i][j] += x[i] * x[j] / (self.noise ** 2)
            self._b[i] += x[i] * reward / (self.noise ** 2)
        self._observations += 1
        self._refresh()

    def fit(self, contexts: list[list[float]], rewards: list[float]) -> None:
        """Closed-form ridge fit, used to warm-start the online posterior.

        Cold-start bandits pay for information that the source split has already
        bought.  Seeding the posterior with a ridge solution on offline outcomes
        keeps the exploration bonus honest while removing that duplicated cost.
        The update uses the same likelihood scale as the online steps, so the
        prior is not silently outvoted by the first few stream observations.
        """
        scale = 1.0 / (self.noise ** 2)
        for context, reward in zip(contexts, rewards):
            for i in range(self.dimension):
                if context[i] == 0.0:
                    continue
                for j in range(self.dimension):
                    if context[j] != 0.0:
                        self._a[i][j] += scale * context[i] * context[j]
                self._b[i] += scale * context[i] * reward
        self._refresh()


def _inverse(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    augmented = [row[:] + [1.0 if i == j else 0.0 for j in range(size)]
                 for i, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            augmented[pivot][column] = 1e-12
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [a - factor * b for a, b in zip(augmented[row], augmented[column])]
    return [row[size:] for row in augmented]


def _cholesky(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            total = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                lower[i][j] = math.sqrt(max(total, 1e-12))
            else:
                lower[i][j] = total / lower[j][j]
    return lower


class Policy:
    """Common interface: score every candidate, then learn from the outcome."""

    name = "policy"

    def reset(self) -> None:
        raise NotImplementedError

    def warm_start(self, rows: list[dict[str, float]],
                   rewards: list[float]) -> None:
        """Optional offline prior; the default policy ignores it."""
        return None

    def score(self, row: dict[str, float]) -> float:
        raise NotImplementedError

    def update(self, row: dict[str, float], reward: float) -> None:
        raise NotImplementedError


@dataclass
class LinUCB(Policy):
    alpha: float = 0.6
    posterior: LinearPosterior = field(default_factory=LinearPosterior)
    name: str = "linucb"

    def reset(self) -> None:
        self.posterior.reset()

    def warm_start(self, rows: list[dict[str, float]],
                   rewards: list[float]) -> None:
        self.posterior.fit([vector(row) for row in rows], rewards)

    def score(self, row: dict[str, float]) -> float:
        x = vector(row)
        return self.posterior.expected(x) + self.alpha * math.sqrt(max(0.0, self.posterior.leverage(x)))

    def update(self, row: dict[str, float], reward: float) -> None:
        self.posterior.update(vector(row), reward)


@dataclass
class ThompsonSampling(Policy):
    seed: int = 0
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    posterior: LinearPosterior = field(default_factory=LinearPosterior)
    name: str = "thompson"
    _draw: list[float] = field(default_factory=list)

    def reset(self) -> None:
        self.posterior.reset()
        self.rng = random.Random(self.seed)
        self._draw = []

    def warm_start(self, rows: list[dict[str, float]],
                   rewards: list[float]) -> None:
        self.posterior.fit([vector(row) for row in rows], rewards)
        self._draw = []

    def begin(self) -> None:
        """Draw one coefficient vector per decision, shared by all candidates."""
        self._draw = self.posterior.sample(self.rng)

    def score(self, row: dict[str, float]) -> float:
        if not self._draw:
            self.begin()
        x = vector(row)
        return sum(weight * value for weight, value in zip(self._draw, x))

    def update(self, row: dict[str, float], reward: float) -> None:
        self.posterior.update(vector(row), reward)


@dataclass
class EpsilonGreedy(Policy):
    seed: int = 0
    epsilon: float = 0.1
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    posterior: LinearPosterior = field(default_factory=LinearPosterior)
    name: str = "epsilon_greedy"

    def reset(self) -> None:
        self.posterior.reset()
        self.rng = random.Random(self.seed)

    def warm_start(self, rows: list[dict[str, float]],
                   rewards: list[float]) -> None:
        self.posterior.fit([vector(row) for row in rows], rewards)

    def score(self, row: dict[str, float]) -> float:
        base = self.posterior.expected(vector(row))
        return base + (self.epsilon if self.rng.random() < self.epsilon else 0.0)

    def update(self, row: dict[str, float], reward: float) -> None:
        self.posterior.update(vector(row), reward)


@dataclass
class StaticPolicy(Policy):
    """No exploration: rank by a fitted offline score and never update."""

    model: object | None = None
    name: str = "static"

    def reset(self) -> None:
        return None

    def score(self, row: dict[str, float]) -> float:
        if self.model is None:
            return 0.0
        return float(self.model.predict_proba(row))

    def update(self, row: dict[str, float], reward: float) -> None:
        return None


def build_policy(kind: str, model=None, seed: int = 0) -> Policy:
    if kind == "linucb":
        return LinUCB()
    if kind == "thompson":
        return ThompsonSampling(seed=seed)
    if kind == "epsilon_greedy":
        return EpsilonGreedy(seed=seed)
    if kind == "static":
        return StaticPolicy(model=model)
    raise ValueError(f"unknown policy: {kind}")
