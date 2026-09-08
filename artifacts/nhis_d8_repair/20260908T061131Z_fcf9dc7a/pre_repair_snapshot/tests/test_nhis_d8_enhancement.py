"""Unit tests for D8 Accuracy Enhancement Study runner."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

from nhis_fairbias.d8_enhancement_runner import (
    D8EnhancementRunner,
    compute_group_fairness_gaps,
)


class TestD8EnhancementRunner(unittest.TestCase):
    """Test group fairness gap metrics and smoke test runner execution."""

    def test_compute_group_fairness_gaps(self):
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0, 1, 1])
        prot = np.array([0, 0, 0, 1, 1, 1])

        gaps = compute_group_fairness_gaps(y_true, y_pred, prot)
        self.assertIn("demographic_parity_difference", gaps)
        self.assertIn("equal_opportunity_difference", gaps)
        self.assertGreaterEqual(gaps["demographic_parity_difference"], 0.0)
        self.assertGreaterEqual(gaps["equal_opportunity_difference"], 0.0)

    def test_runner_smoke_test_arm1(self):
        runner = D8EnhancementRunner(smoke_test=True)
        res = runner.run_arm("D6_ARM_001")
        self.assertEqual(res["arm_id"], "D6_ARM_001")
        self.assertIn("conditions", res)
        self.assertIn("baseline", res["conditions"])
        self.assertIn("canonical_fairbias", res["conditions"])
        self.assertIn("posthoc_enhancement", res["conditions"])
        self.assertIn("joint_enhancement", res["conditions"])


if __name__ == "__main__":
    unittest.main()
