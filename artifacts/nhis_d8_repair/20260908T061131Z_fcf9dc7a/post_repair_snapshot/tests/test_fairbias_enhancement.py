"""Unit and integration tests for FairAccuracyEnhancement module."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fairbias.config import FairBiasConfig
from fairbias.evaluator import FairEvaluator
from fairbias.enhancement import FairAccuracyEnhancement
from fairbias.transform import FairTransform


class TestFairAccuracyEnhancement(unittest.TestCase):
    """Verifies monotonic candidate progression, utility verification, and bounded fairness."""

    def setUp(self):
        np.random.seed(42)
        n = 200
        # Synthetic dataset with 1 protected, 1 numeric, 1 categorical
        x_num = np.random.randn(n) * 2.0 + 5.0
        x_cat = np.random.choice(["A", "B", "C", "D"], size=n)
        o_prot = np.random.choice([0, 1], size=n)
        # Target correlated with x_num and x_cat
        prob = 1.0 / (1.0 + np.exp(-(0.5 * x_num + (x_cat == "A").astype(float) * 1.5 - 2.0)))
        y = (np.random.rand(n) < prob).astype(int)

        self.df = pd.DataFrame({
            "num_feat": x_num,
            "cat_feat": x_cat,
        })
        self.Y = pd.Series(y, name="target")
        self.O = pd.DataFrame({"protected": o_prot})

        self.config = FairBiasConfig.compas_default(
            random_seed=42,
            classifier="LR",
        )
        self.evaluator = FairEvaluator(
            config=self.config,
            label_O=["protected"],
            label_Y="target",
            cate_attrs=["cat_feat"],
            num_attrs=["num_feat"],
        )
        self.transformer = FairTransform(
            n_bins=5,
            log_epsilon=1e-5,
            x_max=100.0,
        )
        self.ae = FairAccuracyEnhancement(
            evaluator=self.evaluator,
            transformer=self.transformer,
            label_Y="target",
            cate_attrs=["cat_feat"],
            num_attrs=["num_feat"],
            max_fairness_degradation=0.05,
            poly_exponents=(1 / 3, 3.0),
        )

    def test_ranking_identifies_top_feature(self):
        attr = self.ae.find_target_correlated_attribute(self.df, self.Y, {})
        self.assertIn(attr, ["num_feat", "cat_feat"])
        self.assertNotIn(attr, self.ae.skip_attr_list)

    def test_monotonic_exponent_tracking_no_oscillation(self):
        # Force num_feat to be evaluated
        changed_dict = {}
        # Initial step
        df_step1, changed1, attr1 = self.ae.enhance_step(
            X_train=self.df,
            Y_train=self.Y,
            changed_dict=changed_dict,
        )
        # Whether accepted or skipped, num_feat's evaluated powers must be recorded
        tried = self.ae.tried_exponents["num_feat"]
        self.assertTrue(len(tried) > 0)
        # Power 1.0 (base) should be recorded
        self.assertIn(1.0, tried)

        # Run multiple steps; should never oscillate or hang
        current_changed = changed1
        for _ in range(5):
            df_next, next_changed, next_attr = self.ae.enhance_step(
                X_train=self.df,
                Y_train=self.Y,
                changed_dict=current_changed,
            )
            current_changed = next_changed
            if next_attr is None:
                break

        # Must terminate finite without loop
        self.assertTrue(len(self.ae.skip_attr_list) > 0 or len(current_changed) > 0)

    def test_fairness_degradation_bound_rejection(self):
        # If candidate fairness exceeds threshold + max_fairness_degradation, reject
        ae_strict = FairAccuracyEnhancement(
            evaluator=self.evaluator,
            transformer=self.transformer,
            label_Y="target",
            cate_attrs=["cat_feat"],
            num_attrs=["num_feat"],
            max_fairness_degradation=0.0000001,  # strictly zero tolerance
        )
        # Mock calculate_epsilon to return a huge violation for candidate
        with unittest.mock.patch.object(
            self.evaluator, "calculate_epsilon",
            return_value={"protected": {"num_feat": 999.0, "cat_feat": 999.0}}
        ):
            df_res, changed_res, attr_res = ae_strict.enhance_step(
                X_train=self.df,
                Y_train=self.Y,
                changed_dict={},
                O_train=self.O,
                epsilon_threshold=0.01,
            )
            # All candidates should be rejected due to fairness degradation
            self.assertEqual(changed_res, {})
            self.assertIsNone(attr_res)
            self.assertEqual(set(self.df.columns), ae_strict.skip_attr_list)

    def test_utility_rejection_when_score_drops(self):
        # Mock _evaluate_utility to always report 0.0 for candidates, rejecting them
        with unittest.mock.patch.object(self.ae, "_evaluate_utility", side_effect=[0.9, 0.1, 0.1, 0.1, 0.1]):
            df_res, changed_res, attr_res = self.ae.enhance_step(
                X_train=self.df,
                Y_train=self.Y,
                changed_dict={},
            )
            self.assertEqual(changed_res, {})
            self.assertIsNone(attr_res)

    def test_successful_enhancement_improves_utility(self):
        # When candidate utility strictly improves, transformation is accepted
        with unittest.mock.patch.object(self.ae, "_evaluate_utility", side_effect=[0.5, 0.8]):
            df_res, changed_res, attr_res = self.ae.enhance_step(
                X_train=self.df,
                Y_train=self.Y,
                changed_dict={},
            )
            self.assertIsNotNone(attr_res)
            self.assertIn(attr_res, changed_res)
            self.assertTrue(len(changed_res) > 0)


if __name__ == "__main__":
    unittest.main()
