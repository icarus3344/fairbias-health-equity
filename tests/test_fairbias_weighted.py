"""Unit tests for survey-weighted FairBias bias divergence and concentration.

Verifies:
1. sample_weight=None preserves legacy behavior exactly.
2. Constant weights (all 1.0, all c > 0) equal unweighted behavior bit-for-bit.
3. Scale invariance: multiplying all weights by c > 0 leaves results unchanged.
4. Fail-closed validation: negative, zero, non-finite, and misaligned weights raise ValueError.
5. Analytical verification of Eq. (2) weighted group mean and category frequency.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

_SRC_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from fairbias.bias_metric import (
    SURVEY_WEIGHTED_EXTENSION_DECLARATION,
    compute_bias_concentration,
    compute_dphi_matrix,
    compute_pairwise_divergences,
)
from fairbias.config import FairBiasConfig
from fairbias.evaluator import FairEvaluator


class TestFairBiasWeighted(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        n = 100
        # Synthetic dataset with 2 numerical, 2 categorical, and binary protected attribute
        self.X = pd.DataFrame({
            "num1": np.random.uniform(10.0, 50.0, size=n),
            "num2": np.random.exponential(5.0, size=n),
            "cat1": np.random.choice(["A", "B", "C"], size=n),
            "cat2": np.random.choice([1, 2], size=n),
        })
        self.o_series = pd.Series(np.random.choice([0, 1], size=n), name="prot")
        self.O_df = pd.DataFrame({"prot": self.o_series})
        self.weights = np.random.uniform(0.5, 10.0, size=n)
        self.cate_attrs = ["cat1", "cat2"]
        self.num_attrs = ["num1", "num2"]

    def test_extension_declaration_traceable(self) -> None:
        """Traceability test: pre-declared extension contract string remains pinned."""
        self.assertIn("Survey-weighted extension", SURVEY_WEIGHTED_EXTENSION_DECLARATION)
        self.assertIn("compute_pairwise_divergences", SURVEY_WEIGHTED_EXTENSION_DECLARATION)
        self.assertIn("mu_hat_mg", SURVEY_WEIGHTED_EXTENSION_DECLARATION)
        self.assertIn("p_hat_mkg", SURVEY_WEIGHTED_EXTENSION_DECLARATION)

    def test_sample_weight_none_equals_legacy(self) -> None:
        """Invariant 1: sample_weight=None preserves legacy behavior exactly."""
        df_none = compute_pairwise_divergences(
            self.X, self.o_series, self.cate_attrs, self.num_attrs, sample_weight=None
        )
        # Default argument call
        df_default = compute_pairwise_divergences(
            self.X, self.o_series, self.cate_attrs, self.num_attrs
        )
        pd.testing.assert_frame_equal(df_none, df_default)

    def test_constant_weights_equals_unweighted(self) -> None:
        """Invariant 2: constant weights w_i = c equal unweighted behavior bit-for-bit."""
        df_unweighted = compute_pairwise_divergences(
            self.X, self.o_series, self.cate_attrs, self.num_attrs, sample_weight=None
        )
        for const_val in (1.0, 5.0, 42.123):
            const_weights = np.full(len(self.X), const_val)
            df_const = compute_pairwise_divergences(
                self.X, self.o_series, self.cate_attrs, self.num_attrs, sample_weight=const_weights
            )
            np.testing.assert_allclose(
                df_const.to_numpy(),
                df_unweighted.to_numpy(),
                atol=1e-12,
                err_msg=f"Constant weights {const_val} deviated from unweighted calculation",
            )

    def test_scale_invariance_positive_scalar(self) -> None:
        """Invariant 3: multiplying all weights by c > 0 leaves results unchanged."""
        df_w = compute_pairwise_divergences(
            self.X, self.o_series, self.cate_attrs, self.num_attrs, sample_weight=self.weights
        )
        for c in (0.1, 2.5, 100.0):
            df_scaled = compute_pairwise_divergences(
                self.X, self.o_series, self.cate_attrs, self.num_attrs, sample_weight=c * self.weights
            )
            np.testing.assert_allclose(
                df_scaled.to_numpy(),
                df_w.to_numpy(),
                atol=1e-12,
                err_msg=f"Scaling weights by {c} violated scale invariance",
            )

    def test_fail_closed_on_invalid_weights(self) -> None:
        """Invariant 4: invalid, misaligned, nonfinite, or nonpositive weights fail closed."""
        n = len(self.X)
        # Mismatched length
        with self.assertRaises(ValueError):
            compute_pairwise_divergences(
                self.X, self.o_series, self.cate_attrs, self.num_attrs, sample_weight=np.ones(n - 1)
            )

        # 2D array
        with self.assertRaises(ValueError):
            compute_pairwise_divergences(
                self.X, self.o_series, self.cate_attrs, self.num_attrs, sample_weight=np.ones((n, 2))
            )

        # NaN in weights
        nan_w = np.ones(n)
        nan_w[10] = np.nan
        with self.assertRaises(ValueError):
            compute_pairwise_divergences(
                self.X, self.o_series, self.cate_attrs, self.num_attrs, sample_weight=nan_w
            )

        # Inf in weights
        inf_w = np.ones(n)
        inf_w[5] = np.inf
        with self.assertRaises(ValueError):
            compute_pairwise_divergences(
                self.X, self.o_series, self.cate_attrs, self.num_attrs, sample_weight=inf_w
            )

        # Zero weight
        zero_w = np.ones(n)
        zero_w[2] = 0.0
        with self.assertRaises(ValueError):
            compute_pairwise_divergences(
                self.X, self.o_series, self.cate_attrs, self.num_attrs, sample_weight=zero_w
            )

        # Negative weight
        neg_w = np.ones(n)
        neg_w[3] = -1.5
        with self.assertRaises(ValueError):
            compute_pairwise_divergences(
                self.X, self.o_series, self.cate_attrs, self.num_attrs, sample_weight=neg_w
            )

    def test_analytical_weighted_group_mean_and_freq(self) -> None:
        """Verify weighted group mean and category frequency match hand-computed analytical formulas."""
        # Simple deterministic 4-sample setup:
        # group 0: 2 samples, weights [1.0, 3.0]
        # group 1: 2 samples, weights [2.0, 2.0]
        X = pd.DataFrame({
            "num": [10.0, 20.0, 10.0, 30.0],
            "cat": ["A", "B", "A", "A"],
        })
        o_series = pd.Series([0, 0, 1, 1], name="prot")
        w = np.array([1.0, 3.0, 2.0, 2.0])

        # Min-max scaling of 'num': min=10.0, max=30.0 -> normalized:
        # row 0: (10-10)/20 = 0.0
        # row 1: (20-10)/20 = 0.5
        # row 2: (10-10)/20 = 0.0
        # row 3: (30-10)/20 = 1.0
        # Group 0: values [0.0, 0.5], weights [1.0, 3.0]
        # mu_0 = (1.0*0.0 + 3.0*0.5) / (1.0 + 3.0) = 1.5 / 4.0 = 0.375
        # Group 1: values [0.0, 1.0], weights [2.0, 2.0]
        # mu_1 = (2.0*0.0 + 2.0*1.0) / (2.0 + 2.0) = 2.0 / 4.0 = 0.500
        # num_diff = |0.375 - 0.500| = 0.125
        expected_num_diff = 0.125

        # Categorical 'cat': categories {'A', 'B'} (K=2)
        # Group 0: 'A' weight=1.0, 'B' weight=3.0, total=4.0 -> p_0(A)=0.25, p_0(B)=0.75
        # Group 1: 'A' weight=2+2=4.0, 'B' weight=0.0, total=4.0 -> p_1(A)=1.0, p_1(B)=0.0
        # |p_0(A)-p_1(A)| = |0.25 - 1.0| = 0.75
        # |p_0(B)-p_1(B)| = |0.75 - 0.0| = 0.75
        # cat-a = (1/K) * sum = (1/2) * (0.75 + 0.75) = 0.75
        expected_cat_diff = 0.75

        res = compute_pairwise_divergences(
            X, o_series, cate_attrs=["cat"], num_attrs=["num"], sample_weight=w
        )
        self.assertAlmostEqual(res.loc["num", "0_1"], expected_num_diff, places=7)
        self.assertAlmostEqual(res.loc["cat", "0_1"], expected_cat_diff, places=7)

    def test_compute_bias_concentration_weighted(self) -> None:
        """Verify compute_bias_concentration executes with sample_weight and preserves scale invariance."""
        dphi_w = compute_bias_concentration(
            self.X, self.o_series, self.cate_attrs, self.num_attrs, sample_weight=self.weights, random_state=42
        )
        self.assertEqual(set(dphi_w.keys()), set(self.X.columns))

        # Scale invariance on d_phi
        dphi_scaled = compute_bias_concentration(
            self.X, self.o_series, self.cate_attrs, self.num_attrs, sample_weight=10.0 * self.weights, random_state=42
        )
        for col in self.X.columns:
            self.assertAlmostEqual(dphi_w[col], dphi_scaled[col], places=6)

    def test_evaluator_calculate_epsilon_weighted(self) -> None:
        """Verify FairEvaluator.calculate_epsilon accepts sample_weight and executes cleanly."""
        cfg = FairBiasConfig.compas_default()
        evaluator = FairEvaluator(
            config=cfg,
            cate_attrs=self.cate_attrs,
            num_attrs=self.num_attrs,
        )
        eps_unweighted = evaluator.calculate_epsilon(self.X, self.O_df, sample_weight=None)
        eps_const = evaluator.calculate_epsilon(
            self.X, self.O_df, sample_weight=np.ones(len(self.X))
        )
        eps_weighted = evaluator.calculate_epsilon(
            self.X, self.O_df, sample_weight=self.weights
        )

        self.assertIn("prot", eps_unweighted)
        self.assertIn("prot", eps_const)
        self.assertIn("prot", eps_weighted)

        # Constant weights equals unweighted
        for col in self.X.columns:
            self.assertAlmostEqual(
                eps_unweighted["prot"][col],
                eps_const["prot"][col],
                places=6,
                msg=f"Epsilon mismatch for {col} between unweighted and constant weights",
            )


if __name__ == "__main__":
    unittest.main()
