"""Tests for conformal risk control and the empirical baseline gate."""
from __future__ import annotations

import math
import unittest

from geocompass.conformal import (
    ConformalGate,
    fixed_threshold_gate,
    negative_transfer,
)


class NegativeTransferTests(unittest.TestCase):
    def test_margin(self) -> None:
        utilities = [0.2, 0.0, -0.02, -0.5]
        self.assertEqual(negative_transfer(utilities), [0.0, 0.0, 0.0, 1.0])


class ConformalGateTests(unittest.TestCase):
    def setUp(self) -> None:
        # Scores increase with reliability, with noise at the boundary.
        self.scores = [index / 100 for index in range(100)]
        self.losses = [1.0 if score < 0.4 and index % 3 == 0 else 0.0
                       for index, score in enumerate(self.scores)]

    def test_threshold_is_monotone_choice(self) -> None:
        gate = ConformalGate(alpha=0.05, delta=0.1)
        certificate = gate.fit(self.scores, self.losses)
        self.assertTrue(math.isfinite(certificate.threshold))
        self.assertLessEqual(certificate.risk, certificate.upper_bound)

    def test_abstains_when_infeasible(self) -> None:
        scores = [0.9, 0.8, 0.7, 0.6]
        losses = [1.0, 1.0, 1.0, 1.0]
        gate = ConformalGate(alpha=0.05, delta=0.1)
        certificate = gate.fit(scores, losses)
        self.assertFalse(certificate.feasible)
        self.assertFalse(any(gate.commit(score) for score in scores))

    def test_commits_on_confident_scores(self) -> None:
        scores = [0.9 + index / 1000 for index in range(200)] + [0.2, 0.1]
        losses = [0.0] * 200 + [1.0, 1.0]
        gate = ConformalGate(alpha=0.3, delta=0.2)
        certificate = gate.fit(scores, losses)
        self.assertTrue(certificate.feasible)
        self.assertTrue(gate.commit(0.99))
        self.assertFalse(gate.commit(0.0))

    def test_evaluate_reports_coverage_and_risk(self) -> None:
        gate = ConformalGate(alpha=0.2, delta=0.2)
        gate.fit(self.scores, self.losses)
        report = gate.evaluate(self.scores, self.losses)
        self.assertIn("coverage", report)
        self.assertIn("upper", report)
        self.assertGreaterEqual(report["risk"], 0.0)
        self.assertLessEqual(report["risk"], 1.0)

    def test_empty_calibration(self) -> None:
        gate = ConformalGate()
        certificate = gate.fit([], [])
        self.assertFalse(certificate.feasible)
        self.assertFalse(gate.commit(0.9))

    def test_length_mismatch(self) -> None:
        gate = ConformalGate()
        with self.assertRaises(ValueError):
            gate.fit([0.1, 0.2], [0.0])

    def test_empirical_gate_is_more_permissive(self) -> None:
        conformal = ConformalGate(alpha=0.1, delta=0.1).fit(self.scores, self.losses)
        empirical = fixed_threshold_gate(self.scores, self.losses, 0.1)
        self.assertLessEqual(conformal.coverage, empirical.certificate.coverage + 1e-9)


if __name__ == "__main__":
    unittest.main()
