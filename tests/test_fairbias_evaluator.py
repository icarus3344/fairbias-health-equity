"""Unit tests for FairEvaluator train-only scaling, metric calculations, and epsilon properties."""

import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from fairbias.config import FairBiasConfig
from fairbias.evaluator import FairEvaluator


class TestFairEvaluator(unittest.TestCase):
    """Tests leakage-free scaling, metric accuracy, and epsilon computation in FairEvaluator."""

    def setUp(self):
        np.random.seed(42)
        n_samples = 200
        self.X = pd.DataFrame({
            "feat_num1": np.random.uniform(10, 100, n_samples),
            "feat_num2": np.random.uniform(0, 1, n_samples),
            "feat_cat": np.random.choice([0, 1, 2], n_samples),
        })
        self.O = pd.DataFrame({
            "group": np.random.choice([0, 1], n_samples),
        })
        # Synthetic target correlated with num1 and group
        p = 1 / (1 + np.exp(-(0.05 * self.X["feat_num1"] + 0.5 * self.O["group"] - 3)))
        self.Y = pd.Series((np.random.uniform(0, 1, n_samples) < p).astype(int), name="target")

    def test_scaler_fitted_strictly_on_train(self):
        config = FairBiasConfig(eval_norm="min-max", random_seed=42)
        evaluator = FairEvaluator(config=config, label_O=["group"], label_Y="target")

        X_train, X_test, Y_train, Y_test, O_train, O_test = train_test_split(
            self.X, self.Y, self.O, test_size=0.3, random_state=42
        )

        # Ensure train and test index do not overlap
        self.assertEqual(len(set(X_train.index) & set(X_test.index)), 0)

        # Run fit_and_predict
        y_pred, y_prob = evaluator.fit_and_predict(X_train, Y_train, X_test)

        self.assertIsNotNone(evaluator.scaler)
        # Scaler data_min_ must match X_train min, NOT X_test min
        np.testing.assert_allclose(evaluator.scaler.data_min_, X_train.min(axis=0).values)

    def test_metric_calculation_structure(self):
        config = FairBiasConfig(random_seed=42)
        evaluator = FairEvaluator(config=config, label_O=["group"], label_Y="target")

        X_train, X_test, Y_train, Y_test, O_train, O_test = train_test_split(
            self.X, self.Y, self.O, test_size=0.3, random_state=42
        )

        metrics = evaluator.evaluate(X_train, Y_train, O_train, X_test, Y_test, O_test)

        self.assertIn("ACC", metrics)
        self.assertIn("F1", metrics)
        self.assertIn("SP", metrics)
        self.assertIn("EO", metrics)
        self.assertIn("EOpp", metrics)
        self.assertIn("group", metrics["SP"])
        self.assertIn("group", metrics["EO"])

        # Check values are in [0, 1]
        self.assertGreaterEqual(metrics["ACC"], 0.0)
        self.assertLessEqual(metrics["ACC"], 1.0)
        self.assertGreaterEqual(metrics["SP"]["group"], 0.0)
        self.assertLessEqual(metrics["SP"]["group"], 1.0)

    def test_epsilon_synthetic_properties(self):
        # Feature identical to protected attribute should have high d_phi
        # Feature completely independent should have low d_phi
        n = 500
        group = pd.Series(np.random.choice([0, 1], n), name="group")
        X_test_df = pd.DataFrame({
            "copy_of_group": group.copy(),  # perfectly correlated
            "random_cat": np.random.choice([0, 1], n),  # independent
        })
        O_test_df = pd.DataFrame({"group": group})

        evaluator = FairEvaluator(config=FairBiasConfig(random_seed=42), label_O=["group"])
        eps_dict = evaluator.calculate_epsilon(
            X_test_df, O_test_df, cate_attrs=["copy_of_group", "random_cat"]
        )

        eps_copy = eps_dict["group"]["copy_of_group"]
        eps_rand = eps_dict["group"]["random_cat"]

        # d_phi of the proxy feature must dominate the independent one
        self.assertGreater(eps_copy, eps_rand)
        self.assertGreater(eps_copy, 0.5)
        self.assertLess(eps_rand, 0.5)

    def test_round_up_125_rule(self):
        from fairbias.evaluator import round_up_125

        self.assertAlmostEqual(round_up_125(0.008), 0.01)
        self.assertAlmostEqual(round_up_125(0.015), 0.02)
        self.assertAlmostEqual(round_up_125(0.035), 0.05)
        self.assertAlmostEqual(round_up_125(0.075), 0.10)
        self.assertAlmostEqual(round_up_125(1.5), 2.0)
        self.assertAlmostEqual(round_up_125(3.8), 5.0)
        self.assertAlmostEqual(round_up_125(7.2), 10.0)

    def test_compute_adaptive_threshold_kmeans_125(self):
        from fairbias.evaluator import compute_adaptive_threshold_kmeans_125

        # Bimodal distances: low-bias group around 0.02, high-bias group around 0.8
        distances = [0.01, 0.02, 0.03, 0.025, 0.75, 0.85, 0.90]
        threshold = compute_adaptive_threshold_kmeans_125(distances)

        # Low cluster mean ~ 0.02125, rounded up via 1-2-5 rule -> 0.05
        self.assertEqual(threshold, 0.05)

    def test_get_subsets_low_order(self):
        from fairbias.evaluator import get_subsets

        feats = ["f1", "f2", "f3", "f4", "f5"]
        # H=1 should produce exactly 5 singleton subsets
        subsets_h1 = get_subsets(feats, h_order=1)
        self.assertEqual(len(subsets_h1), 5)
        self.assertTrue(all(len(s) == 1 for s in subsets_h1))

        # H=2 should produce 5 (size 1) + 10 (size 2) = 15 subsets
        subsets_h2 = get_subsets(feats, h_order=2)
        self.assertEqual(len(subsets_h2), 15)


if __name__ == "__main__":
    unittest.main()
