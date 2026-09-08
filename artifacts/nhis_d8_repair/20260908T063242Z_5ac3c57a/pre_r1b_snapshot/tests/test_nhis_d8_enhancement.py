"""Unit tests for D8 Accuracy Enhancement Study runner with synthetic data."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from nhis_fairbias.d8_enhancement_runner import (
    D8EnhancementRunner,
    compute_group_fairness_gaps,
)


class FakePreprocessor:
    def get_feature_family_lists(self, feature_set: str):
        return (["cat1"], ["num1", "num2"])


class FakeNHISStudyAdapter:
    """Synthetic in-memory adapter that generates mock NHIS cohorts without reading parquet files."""

    def __init__(self, n_rows: int = 150):
        self.n_rows = n_rows
        self.preprocessor = FakePreprocessor()

    def get_cohort(self, year: int, outcome: str, protected_attribute: str, feature_set: str, disability_arm: bool):
        np.random.seed(year)
        n = self.n_rows
        x_num1 = np.random.randn(n) * 2.0 + 5.0
        x_num2 = np.random.randn(n) + 10.0
        x_cat1 = np.random.choice([0, 1, 2, 3], size=n)
        o_prot = np.random.choice([0, 1], size=n)
        prob = 1.0 / (1.0 + np.exp(-(0.3 * x_num1 + (x_cat1 == 0).astype(float) * 1.2 - 2.0)))
        y = (np.random.rand(n) < prob).astype(int)

        X = pd.DataFrame({"num1": x_num1, "num2": x_num2, "cat1": x_cat1})
        y_s = pd.Series(y, name=outcome)
        o_s = pd.Series(o_prot, name=protected_attribute)
        strata = pd.Series(np.random.randint(100, 110, size=n), name="STRATA")
        psu = pd.Series(np.random.randint(1, 5, size=n), name="PSU")
        return X, y_s, o_s, strata, psu


class TestD8EnhancementRunner(unittest.TestCase):
    """Test group fairness gap metrics and synthetic smoke test runner execution."""

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
        fake_adapter = FakeNHISStudyAdapter(n_rows=100)
        runner = D8EnhancementRunner(
            adapter=fake_adapter,
            canonical_provider=lambda arm_id: {},
            smoke_test=True,
            random_seed=42,
        )
        res = runner.run_arm("D6_ARM_001")
        self.assertEqual(res["arm_id"], "D6_ARM_001")
        self.assertIn("conditions", res)
        self.assertIn("baseline", res["conditions"])
        self.assertIn("canonical_fairbias", res["conditions"])
        self.assertIn("posthoc_enhancement", res["conditions"])
        self.assertIn("joint_enhancement", res["conditions"])


if __name__ == "__main__":
    unittest.main()
