"""Unit tests for simultaneous mapping, anti-collapse checks, and numerical scaling in FairTransform."""

import unittest
import pandas as pd
import numpy as np

from fairbias.transform import FairTransform, calculate_nmi_dict


class TestFairTransform(unittest.TestCase):
    """Tests simultaneous mapping logic and anti-collapse guards in FairTransform."""

    def setUp(self):
        self.transformer = FairTransform()

    def test_simultaneous_mapping_prevents_cascading(self):
        # Data with values 0, 1, 2
        df = pd.DataFrame({"cat_feat": [0, 1, 2, 1, 0, 2]})
        # Rebin mapping: 1 -> 0, 0 -> 2
        # In cascading sequential mapping: 1 -> 0 -> 2 (all 1s become 2s, which is wrong!)
        # In simultaneous mapping: 0 becomes 2, 1 becomes 0, 2 stays 2
        mapping = {1: 0, 0: 2}

        # Validate transform is valid
        is_valid = self.transformer.check_transform_validity(
            df, "cat_feat", mapping, cate_attrs=["cat_feat"]
        )
        self.assertTrue(is_valid)

        transformed = self.transformer.transform_data(
            df, {"cat_feat": mapping}, cate_attrs=["cat_feat"]
        )
        # Expected transformed array: [2, 0, 2, 0, 2, 2]
        expected = [2, 0, 2, 0, 2, 2]
        self.assertListEqual(list(transformed["cat_feat"]), expected)

    def test_anti_collapse_rejects_single_constant_category(self):
        # Binary feature
        df = pd.DataFrame({"bin_feat": [0, 1, 0, 1, 1, 0]})
        # Rebin mapping merging 0 -> 1 (collapsing everything to 1)
        collapsing_mapping = {0: 1}

        # Validity check should REJECT collapsing binary feature into a constant
        is_valid = self.transformer.check_transform_validity(
            df, "bin_feat", collapsing_mapping, cate_attrs=["bin_feat"]
        )
        self.assertFalse(is_valid, "Collapsing a binary feature to a single category must be rejected")

    def test_explicit_dropped_state_is_valid_and_recorded(self):
        # Paper: merging the two categories of a binary attribute is
        # equivalent to excluding the attribute.  Such an exclusion must be
        # representable as an explicit, recorded "dropped" state -- unlike
        # an accidental collapse through a raw mapping, which stays rejected
        # (test_anti_collapse_rejects_single_constant_category).
        df = pd.DataFrame({"bin_feat": [0, 1, 0, 1, 1, 0], "other": [1, 2, 3, 4, 5, 6]})

        is_valid = self.transformer.check_transform_validity(
            df, "bin_feat", "dropped", cate_attrs=["bin_feat"]
        )
        self.assertTrue(is_valid, "The explicit 'dropped' state must be accepted")

        transformed = self.transformer.transform_data(
            df, {"bin_feat": "dropped"}, cate_attrs=["bin_feat"]
        )
        self.assertNotIn("bin_feat", transformed.columns)
        self.assertIn("other", transformed.columns)

    def test_polynomial_power_transform(self):
        df = pd.DataFrame({"num_feat": [-2.0, 0.0, 3.0, 4.0]})
        change = {"power": 2.0}

        is_valid = self.transformer.check_transform_validity(
            df, "num_feat", change, num_attrs=["num_feat"]
        )
        self.assertTrue(is_valid)

        transformed = self.transformer.transform_data(
            df, {"num_feat": change}, num_attrs=["num_feat"]
        )
        # sign(x) * |x|^2 is sign-preserving and nonlinear
        expected = [-4.0, 0.0, 9.0, 16.0]
        np.testing.assert_allclose(transformed["num_feat"].values, expected)

    def test_fractional_power_keeps_sign(self):
        df = pd.DataFrame({"num_feat": [-8.0, 1.0, 8.0]})
        transformed = self.transformer.transform_data(
            df, {"num_feat": {"power": 1 / 3}}, num_attrs=["num_feat"]
        )
        np.testing.assert_allclose(
            transformed["num_feat"].values, [-2.0, 1.0, 2.0], atol=1e-9
        )

    def test_power_overflow_rejected(self):
        # Strict paper mode (x_max=None, the default): the ONLY magnitude
        # bound is the paper's numpy.float32 overflow rule (≈3.4e38).
        # (1e5)^5 = 1e25 is far below float32 max, so it must be ACCEPTED —
        # the old 1e9 engineering guard used to reject it.
        df = pd.DataFrame({"num_feat": [1e5, 2e5]})
        change = {"power": 5.0}
        is_valid = self.transformer.check_transform_validity(
            df, "num_feat", change, num_attrs=["num_feat"]
        )
        self.assertTrue(is_valid)

        # Genuine float32 overflow is still detected by the paper rule
        from fairbias.transform import power_transform_overflows
        df_huge = pd.DataFrame({"num_feat": [1e30, 2.0]})
        self.assertTrue(power_transform_overflows(df_huge["num_feat"], 5.0))
        self.assertFalse(power_transform_overflows(df["num_feat"], 5.0))

    def test_x_max_engineering_guard_is_opt_in(self):
        # A numeric x_max re-enables the NON-PAPER engineering guard and
        # must be opt-in (explicit constructor argument), never a default.
        from fairbias.transform import FairTransform
        guarded = FairTransform(x_max=1e9)
        df = pd.DataFrame({"num_feat": [1e5, 2e5]})
        change = {"power": 5.0}  # (1e5)^5 = 1e25 > 1e9
        self.assertFalse(
            guarded.check_transform_validity(
                df, "num_feat", change, num_attrs=["num_feat"]
            )
        )
        # The default constructor must not install the guard
        self.assertIsNone(FairTransform().x_max)

    def test_chained_category_composition(self):
        from fairbias.transform import compose_category_mapping

        # {3 -> 1} followed by merging 1 into 0 must drag 3 along to 0
        composed = compose_category_mapping({3: 1}, {1: 0})
        self.assertEqual(composed, {3: 0, 1: 0})

        df = pd.DataFrame({"cat_feat": [0, 1, 2, 3]})
        transformed = self.transformer.transform_data(
            df, {"cat_feat": composed}, cate_attrs=["cat_feat"]
        )
        self.assertListEqual(list(transformed["cat_feat"]), [0, 0, 2, 0])

    def test_nmi_calculation_positive_and_stable(self):
        X = pd.DataFrame({
            "f1": [0, 1, 0, 1, 0, 1],
            "f2": [10.5, 20.1, 15.2, 25.4, 11.2, 30.0],
        })
        Y = pd.Series([0, 1, 0, 1, 0, 1])

        nmi_dict = calculate_nmi_dict(X, Y)
        self.assertIn("f1", nmi_dict)
        self.assertIn("f2", nmi_dict)
        self.assertGreater(nmi_dict["f1"], 0.0)
        self.assertGreater(nmi_dict["f2"], 0.0)


if __name__ == "__main__":
    unittest.main()
