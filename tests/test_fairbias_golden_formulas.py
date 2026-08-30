"""Hand-computed golden tests for the paper formulas (Tang, Lu & Li 2024,
Intell. Comput. 2024;3:Article 0083, "Metric-Independent Mitigation of
Unpredefined Bias in Machine Classification").

Every expected value below is derived by hand from the printed equations:

- Eq. (1): w_max(X*) = sqrt( (g_1^2 + ... + g_|X*|^2) / |X*| )  (alpha = 2 RMS norm)
- Eq. (2): per-attribute divergences g_m (numerical centroid distance after
  min-max normalization; categorical mean absolute frequency gap over K categories)
- Eq. (3)/(4): pair distance as the unweighted mean of sub-distances over
  level-0..H exclusion contexts (near-full companion sets)
- Eq. (5): origin distance over contexts of X \\ {x_m} excluding up to h attributes
- Eq. (9): ΔEO = sum_{y in {1,0}} |Pr(y_hat=1|o=+,Y=y) - Pr(y_hat=1|o=-,Y=y)|,
  range [0, 2]

Reference hand computation (4 attributes, g = (1, 2, 4, 8), H = 1):

    w_max values (Eq. 1):  w({f1})=1, w({f2})=2, w({f3})=4, w({f4})=8,
    w({f3,f4}) = sqrt((16+64)/2) = sqrt(40), etc.

    d(f1, f2) (Eq. 3/4): contexts S in {{f3}, {f4}, {f3,f4}} (level-0 and
    level-1 exclusion of the remaining set {f3, f4}):
        |w({f1,f3}) - w({f2,f3})| = |sqrt(8.5)  - sqrt(10)  | = 0.2468017127457291
        |w({f1,f4}) - w({f2,f4})| = |sqrt(32.5) - sqrt(34)  | = 0.1300747693496112
        |w({f1,f3,f4}) - w({f2,f3,f4})| = |sqrt(27) - sqrt(28)| = 0.0953501994225486
        mean = 0.1574088938392963

    d(f1, origin) (Eq. 5): contexts S over {f2,f3,f4} with |S| >= 2:
        |w({f1,f2,f3}) - w({f2,f3})|     = |sqrt(7)    - sqrt(10)  | = 0.5165263491037888
        |w({f1,f2,f4}) - w({f2,f4})|     = |sqrt(23)   - sqrt(34)  | = 1.0351203715325817
        |w({f1,f3,f4}) - w({f3,f4})|     = |sqrt(27)   - sqrt(40)  | = 1.1284028976301270
        |w({f1,f2,f3,f4}) - w({f2,f3,f4})| = |sqrt(21.25) - sqrt(28)| = 0.6817303934827379
        mean = 0.8404450029373088
"""

import unittest
import unittest.mock

import numpy as np
import pandas as pd

from fairbias.bias_metric import (
    compute_pairwise_divergences,
    compute_shapley_distance_matrix,
    get_subsets,
    w_max,
)
from fairbias.config import FairBiasConfig
from fairbias.evaluator import FairEvaluator


GOLDEN_DF_S = pd.DataFrame(
    {"0_1": [1.0, 2.0, 4.0, 8.0]},
    index=["f1", "f2", "f3", "f4"],
)
GOLDEN_FEATURES = ["f1", "f2", "f3", "f4"]


class TestEquation1WMax(unittest.TestCase):
    """Eq. (1): w_max is the RMS norm of the individual contributions."""

    def test_rms_normalization(self):
        self.assertAlmostEqual(w_max([1.0, 2.0]), np.sqrt(2.5), places=12)
        self.assertAlmostEqual(w_max([1.0, 2.0, 4.0, 8.0]), np.sqrt(85.0 / 4.0), places=12)

    def test_singleton_and_empty(self):
        self.assertAlmostEqual(w_max([3.0]), 3.0, places=12)
        self.assertAlmostEqual(w_max([]), 0.0, places=12)


class TestEquation2Divergences(unittest.TestCase):
    """Eq. (2): per-attribute group-pair divergences g_m."""

    def test_categorical_frequency_gap(self):
        # Group 0: cat 0 x80, cat 1 x20; Group 1: cat 0 x30, cat 1 x70
        # g = (1/K) * (|0.8 - 0.3| + |0.2 - 0.7|) with K = 2 -> 0.5
        X = pd.DataFrame({"f": [0] * 80 + [1] * 20 + [0] * 30 + [1] * 70})
        o = pd.Series([0] * 100 + [1] * 100, name="o")
        df_s = compute_pairwise_divergences(X, o, cate_attrs=["f"], num_attrs=[])
        self.assertAlmostEqual(float(df_s.loc["f", "0_1"]), 0.5, places=12)

    def test_numerical_centroid_distance(self):
        # Group 0 values {0, 2}, group 1 values {1, 3}; within-pair min-max
        # normalization maps to {0, 2/3} and {1/3, 1}; centroid gap = 1/3.
        X = pd.DataFrame({"f": [0.0, 2.0, 1.0, 3.0]})
        o = pd.Series([0, 0, 1, 1], name="o")
        df_s = compute_pairwise_divergences(X, o, cate_attrs=[], num_attrs=["f"])
        self.assertAlmostEqual(float(df_s.loc["f", "0_1"]), 1.0 / 3.0, places=12)

    def test_no_family_scaling_is_applied(self):
        # Paper applies Eq. (1) to the RAW g_m values: no per-family mean
        # rescaling.  Two numerical attributes with divergences 0.1 and 0.9
        # must stay 0.1 and 0.9 (the old family-mean scaling would produce
        # 0.2 and 1.8).
        X = pd.DataFrame({
            "f_small": [0.0, 0.9, 0.1, 1.0],   # group means 0.45 vs 0.55 -> gap 0.1
            "f_large": [0.0, 0.1, 0.9, 1.0],   # group means 0.05 vs 0.95 -> gap 0.9
        })
        o = pd.Series([0, 0, 1, 1], name="o")
        df_s = compute_pairwise_divergences(X, o, cate_attrs=[], num_attrs=["f_small", "f_large"])
        self.assertAlmostEqual(float(df_s.loc["f_small", "0_1"]), 0.1, places=6)
        self.assertAlmostEqual(float(df_s.loc["f_large", "0_1"]), 0.9, places=6)


class TestEquations3to5DistanceMatrix(unittest.TestCase):
    """Eq. (3)/(4)/(5): Shapley-style distance with level-H EXCLUSION contexts."""

    def test_eq4_pair_distance_golden(self):
        # Hand value: d(f1, f2) = 0.1574088938392963 with H = 1
        dist, nodes = compute_shapley_distance_matrix(GOLDEN_DF_S, GOLDEN_FEATURES, h_order=1)
        i, j = nodes.index("f1"), nodes.index("f2")
        self.assertAlmostEqual(dist[i, j], 0.1574088938392963, places=9)

    def test_eq5_origin_distance_golden(self):
        # Hand value with H = 1: contexts S over {f2,f3,f4} with |S| >= 2
        #   |w({f1,f2,f3})   - w({f2,f3})|   = |sqrt(7)    - sqrt(10)|
        #   |w({f1,f2,f4})   - w({f2,f4})|   = |sqrt(23)   - sqrt(34)|
        #   |w({f1,f3,f4})   - w({f3,f4})|   = |sqrt(27)   - sqrt(40)|
        #   |w({f1,f2,f3,f4})- w({f2,f3,f4})|= |sqrt(21.25)- sqrt(28)|
        import math
        expected = (
            abs(math.sqrt(7.0) - math.sqrt(10.0))
            + abs(math.sqrt(23.0) - math.sqrt(34.0))
            + abs(math.sqrt(27.0) - math.sqrt(40.0))
            + abs(math.sqrt(21.25) - math.sqrt(28.0))
        ) / 4.0
        dist, nodes = compute_shapley_distance_matrix(GOLDEN_DF_S, GOLDEN_FEATURES, h_order=1)
        i, o = nodes.index("f1"), nodes.index("origin")
        self.assertAlmostEqual(dist[i, o], expected, places=12)
        self.assertAlmostEqual(dist[i, o], 0.8404450029373088, places=9)

    def test_eq4_full_shapley_when_h_ge_available(self):
        # H >= |remaining| degenerates to the full enumeration including the
        # empty context: d(f1, f2) = (|1-2| + 0.2468017 + 0.1300748 + 0.0953502)/4
        dist, nodes = compute_shapley_distance_matrix(GOLDEN_DF_S, GOLDEN_FEATURES, h_order=2)
        i, j = nodes.index("f1"), nodes.index("f2")
        expected = (1.0 + 0.2468017127457291 + 0.1300747693496112 + 0.0953501994225486) / 4.0
        self.assertAlmostEqual(dist[i, j], expected, places=9)

    def test_level_h_enumerates_near_full_contexts(self):
        # Eq. (4): at level h, Xc excludes h attributes besides xa/xb, so
        # |Xc| = |X| - 2 - h.  H = 1 over 5 features must keep subsets of
        # size >= 3 (6 contexts), NOT small subsets of size <= 1.
        feats = ["f1", "f2", "f3", "f4", "f5"]
        subsets_h1 = get_subsets(feats, h_order=1)
        self.assertEqual(len(subsets_h1), 6)  # 5 x size-4 + 1 x size-5
        self.assertTrue(all(len(s) >= 4 for s in subsets_h1))

        subsets_h2 = get_subsets(feats, h_order=2)
        self.assertEqual(len(subsets_h2), 16)  # C(5,3)+C(5,4)+C(5,5) = 10+5+1
        self.assertTrue(all(len(s) >= 3 for s in subsets_h2))

    def test_distance_matrix_is_symmetric_with_zero_diagonal(self):
        dist, nodes = compute_shapley_distance_matrix(GOLDEN_DF_S, GOLDEN_FEATURES, h_order=1)
        np.testing.assert_allclose(dist, dist.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-12)


class TestEquation9EO(unittest.TestCase):
    """Eq. (9): ΔEO is the SUM of the TPR and FPR conditional gaps, range [0, 2]."""

    def _metrics_for_hand_case(self):
        # Group 0: Y=[1,1,0,0], pred=[1,1,0,0] -> TPR=1, FPR=0
        # Group 1: Y=[1,1,0,0], pred=[0,0,1,1] -> TPR=0, FPR=1
        # ΔEO = |1-0| + |0-1| = 2.0 (the old max-based implementation returned 1.0)
        y_true = pd.Series([1, 1, 0, 0] * 2, name="y")
        y_pred = np.array([1, 1, 0, 0, 0, 0, 1, 1])
        O_df = pd.DataFrame({"g": [0] * 4 + [1] * 4})
        evaluator = FairEvaluator(
            config=FairBiasConfig(random_seed=0), label_O=["g"], label_Y="y"
        )
        return evaluator.compute_metrics(y_true, y_pred, O_df)

    def test_eo_is_tpr_gap_plus_fpr_gap(self):
        metrics = self._metrics_for_hand_case()
        self.assertAlmostEqual(metrics["EO"]["g"], 2.0, places=12)

    def test_eo_half_gap_case(self):
        # Group 0: Y=[1,0], pred=[1,0] -> TPR=1, FPR=0
        # Group 1: Y=[1,0], pred=[0,1] -> TPR=0, FPR=1 -> same as above per pair
        # Smaller case: TPR gap 0.5, FPR gap 0.5 -> EO = 1.0 (max semantics would say 0.5)
        y_true = pd.Series([1, 1, 0, 0, 1, 1, 0, 0], name="y")
        # Group 0: TPR = 1, FPR = 0.5 ; Group 1: TPR = 0.5, FPR = 0
        y_pred = np.array([1, 1, 1, 0, 0, 1, 0, 0])
        O_df = pd.DataFrame({"g": [0] * 4 + [1] * 4})
        evaluator = FairEvaluator(
            config=FairBiasConfig(random_seed=0), label_O=["g"], label_Y="y"
        )
        metrics = evaluator.compute_metrics(y_true, y_pred, O_df)
        self.assertAlmostEqual(metrics["EO"]["g"], 1.0, places=12)
        self.assertAlmostEqual(metrics["EOpp"]["g"], 0.5, places=12)  # TPR gap
        self.assertAlmostEqual(metrics["FPRB"]["g"], 0.5, places=12)  # FPR gap

    def test_eo_upper_bound_is_two(self):
        metrics = self._metrics_for_hand_case()
        self.assertLessEqual(metrics["EO"]["g"], 2.0)


class TestMDSFailureIsNotMasked(unittest.TestCase):
    """A broken MDS input must raise, never silently return zero bias."""

    def test_nan_distance_matrix_raises(self):
        from fairbias.bias_metric import compute_bias_concentration

        df_s = pd.DataFrame({"0_1": [np.nan, 1.0]}, index=["f1", "f2"])
        with unittest.mock.patch(
            "fairbias.bias_metric.compute_pairwise_divergences", return_value=df_s
        ):
            X = pd.DataFrame({"f1": [0.0, 1.0], "f2": [1.0, 0.0]})
            o = pd.Series([0, 1])
            with self.assertRaises((ValueError, RuntimeError)):
                compute_bias_concentration(X, o, cate_attrs=[], num_attrs=["f1", "f2"])

    def test_mds_exception_propagates(self):
        from fairbias.bias_metric import compute_bias_concentration

        X = pd.DataFrame({"f1": [0.0, 1.0, 2.0], "f2": [1.0, 0.0, 2.0]})
        o = pd.Series([0, 1, 0])
        with unittest.mock.patch(
            "fairbias.bias_metric.MDS", side_effect=RuntimeError("boom")
        ):
            with self.assertRaises(RuntimeError):
                compute_bias_concentration(X, o, cate_attrs=[], num_attrs=["f1", "f2"])


if __name__ == "__main__":
    unittest.main()
