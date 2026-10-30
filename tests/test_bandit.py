"""Tests for the contextual-bandit selection policies."""
from __future__ import annotations

import math
import random
import unittest

from geocompass.bandit import (
    EpsilonGreedy,
    LinUCB,
    LinearPosterior,
    StaticPolicy,
    ThompsonSampling,
    build_policy,
    reward_from_utility,
    vector,
)
from geocompass.selector import STRUCTURAL_FEATURES


def _row(bias: float, semantic: float) -> dict[str, float]:
    row = {name: 0.0 for name in STRUCTURAL_FEATURES}
    row["bias"] = 1.0
    row["semantic"] = semantic
    row["target_match"] = bias
    return row


class RewardTests(unittest.TestCase):
    def test_bounds(self) -> None:
        self.assertEqual(reward_from_utility(-2.0), 0.0)
        self.assertEqual(reward_from_utility(2.0), 1.0)
        self.assertAlmostEqual(reward_from_utility(0.0), 0.5)


class VectorTests(unittest.TestCase):
    def test_fixed_order(self) -> None:
        row = _row(1.0, 0.5)
        values = vector(row)
        self.assertEqual(len(values), len(STRUCTURAL_FEATURES))
        self.assertEqual(values[STRUCTURAL_FEATURES.index("semantic")], 0.5)

    def test_missing_features_default_to_zero(self) -> None:
        self.assertTrue(all(value == 0.0 for value in vector({})))


class PosteriorTests(unittest.TestCase):
    def test_learns_positive_association(self) -> None:
        posterior = LinearPosterior()
        for _ in range(60):
            posterior.update(vector(_row(1.0, 0.9)), 1.0)
            posterior.update(vector(_row(0.0, 0.1)), 0.0)
        good = posterior.expected(vector(_row(1.0, 0.9)))
        bad = posterior.expected(vector(_row(0.0, 0.1)))
        self.assertGreater(good, bad)
        self.assertEqual(posterior.observations, 120)

    def test_leverage_decreases_with_evidence(self) -> None:
        posterior = LinearPosterior()
        row = vector(_row(1.0, 0.5))
        before = posterior.leverage(row)
        for _ in range(10):
            posterior.update(row, 1.0)
        self.assertLess(posterior.leverage(row), before)


class PolicyTests(unittest.TestCase):
    def test_policies_are_deterministic_given_seed(self) -> None:
        for policy in (ThompsonSampling(seed=7),
                       EpsilonGreedy(seed=7)):
            policy.reset()
            first = [policy.score(_row(1.0, 0.5)) for _ in range(3)]
            policy.reset()
            second = [policy.score(_row(1.0, 0.5)) for _ in range(3)]
            self.assertEqual(first, second)

    def test_linucb_optimism_on_unseen_context(self) -> None:
        policy = LinUCB(alpha=1.0)
        policy.reset()
        unseen = policy.score(_row(1.0, 0.9))
        self.assertTrue(math.isfinite(unseen))

    def test_update_moves_scores(self) -> None:
        policy = LinUCB(alpha=0.0)
        policy.reset()
        row = _row(1.0, 0.8)
        before = policy.score(row)
        for _ in range(20):
            policy.update(row, 1.0)
        self.assertGreater(policy.score(row), before)

    def test_warm_start_orders_contexts(self) -> None:
        policy = LinUCB(alpha=0.0)
        policy.reset()
        good = _row(1.0, 0.9)
        bad = _row(0.0, 0.1)
        rows = [good] * 20 + [bad] * 20
        rewards = [1.0] * 20 + [0.0] * 20
        policy.warm_start(rows, rewards)
        self.assertGreater(policy.score(good), policy.score(bad))

    def test_warm_start_reduces_optimism(self) -> None:
        cold = LinUCB(alpha=1.0)
        cold.reset()
        warm = LinUCB(alpha=1.0)
        warm.reset()
        row = _row(1.0, 0.5)
        warm.warm_start([row] * 40, [1.0] * 40)
        self.assertLess(warm.posterior.leverage(vector(row)),
                        cold.posterior.leverage(vector(row)))

    def test_static_policy_is_constant(self) -> None:
        policy = StaticPolicy()
        policy.reset()
        self.assertEqual(policy.score(_row(1.0, 0.5)), 0.0)
        policy.update(_row(1.0, 0.5), 1.0)
        self.assertEqual(policy.score(_row(1.0, 0.5)), 0.0)

    def test_factory(self) -> None:
        self.assertIsInstance(build_policy("linucb"), LinUCB)
        self.assertIsInstance(build_policy("thompson", seed=3), ThompsonSampling)
        with self.assertRaises(ValueError):
            build_policy("nope")


if __name__ == "__main__":
    unittest.main()
