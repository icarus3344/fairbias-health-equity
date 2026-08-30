"""Unit tests for FairBiasMitigation denominator correctness, 0-based zorder,
epsilon-ball acceptance (paper semantics), and explicit recorded drops."""

import unittest
import pandas as pd
import numpy as np

from fairbias.config import FairBiasConfig
from fairbias.evaluator import FairEvaluator
from fairbias.mitigation import FairBiasMitigation
from fairbias.transform import FairTransform, calculate_nmi_dict


class TestFairBiasMitigation(unittest.TestCase):
    """Tests rebin proportion math, candidate sorting, and fallback logic in FairBiasMitigation."""

    def setUp(self):
        self.evaluator = FairEvaluator(label_O=["group"], label_Y="target")
        self.transformer = FairTransform()
        self.bm = FairBiasMitigation(
            evaluator=self.evaluator,
            transformer=self.transformer,
            label_O=["group"],
            cate_attrs=["charge_deg", "score_cat"],
            num_attrs=["age", "priors"],
        )

    def test_compute_r1_rebin_denominator_correctness(self):
        # Construct group distribution where proportions differ
        # Group 0: 80 items in cat A (code 0), 20 in cat B (code 1) -> total 100
        # Group 1: 30 items in cat A (code 0), 70 in cat B (code 1) -> total 100
        feat_vals = [0] * 80 + [1] * 20 + [0] * 30 + [1] * 70
        prot_vals = [0] * 100 + [1] * 100

        s_feat = pd.Series(feat_vals)
        s_prot = pd.Series(prot_vals)

        # 0-based zorder=0 should return top candidate
        rebin_0 = self.bm.compute_r1_rebin(s_feat, s_prot, zorder=0)
        self.assertIsNotNone(rebin_0)
        self.assertIsInstance(rebin_0, dict)

    def test_0_based_zorder_indexing(self):
        # 3 categories
        # Group 0: [0]*50, [1]*30, [2]*20
        # Group 1: [0]*10, [1]*40, [2]*50
        feat_vals = [0]*50 + [1]*30 + [2]*20 + [0]*10 + [1]*40 + [2]*50
        prot_vals = [0]*100 + [1]*100
        s_feat = pd.Series(feat_vals)
        s_prot = pd.Series(prot_vals)

        rebin_0 = self.bm.compute_r1_rebin(s_feat, s_prot, zorder=0)
        rebin_1 = self.bm.compute_r1_rebin(s_feat, s_prot, zorder=1)

        self.assertIsNotNone(rebin_0)
        self.assertIsNotNone(rebin_1)

    def test_attributes_inside_epsilon_ball_are_untouched(self):
        # Paper: mitigation only acts when max d_phi exceeds epsilon.  With a
        # huge threshold every attribute is inside the ball -> no-op.
        n = 100
        X = pd.DataFrame({
            "charge_deg": [0, 1, 2, 0, 1, 2] * 16 + [0, 1, 2, 0],
            "score_cat": [0, 1, 2, 3] * 25,
            "age": np.linspace(20, 60, n),
        })
        Y = pd.Series([0, 1] * 50, name="target")
        O = pd.DataFrame({"group": [0] * 50 + [1] * 50})

        nmi_org = calculate_nmi_dict(X, Y)
        current_epsilon = self.evaluator.calculate_epsilon(
            X, O, cate_attrs=["charge_deg", "score_cat"], num_attrs=["age"]
        )

        transformed_df, changed_dict, sel_o, sel_attr = self.bm.mitigate_step(
            X=X,
            Y=Y,
            O=O,
            nmi_org=nmi_org,
            changed_dict={},
            current_epsilon=current_epsilon,
            epsilon_threshold=1e9,
        )

        self.assertIsNone(sel_attr, "No transform may be accepted inside the epsilon ball")
        self.assertEqual(changed_dict, {})
        self.assertEqual(list(transformed_df.columns), list(X.columns))
        for col in transformed_df.columns:
            self.assertGreater(transformed_df[col].nunique(), 1, f"Column {col} was improperly collapsed")

    def test_binary_attribute_merge_records_explicit_drop(self):
        # Paper: for binary attributes, merging the two categories into one
        # is equivalent to excluding the attribute.  A strongly biased binary
        # categorical attribute under a small epsilon must end up either
        # below epsilon or explicitly recorded as "dropped" -- never as a
        # silent constant column.
        n = 200
        rng = np.random.default_rng(3)
        group = rng.integers(0, 2, n)
        biased_bin = group.copy()  # perfectly separated binary attribute
        X = pd.DataFrame({
            "biased_bin": biased_bin,
            "other_num": rng.normal(size=n),
        })
        Y = pd.Series(rng.integers(0, 2, n), name="target")
        O = pd.DataFrame({"group": group})

        evaluator = FairEvaluator(
            config=FairBiasConfig(random_seed=0),
            label_O=["group"], label_Y="target",
            cate_attrs=["biased_bin"], num_attrs=["other_num"],
        )
        bm = FairBiasMitigation(
            evaluator=evaluator,
            transformer=FairTransform(),
            label_O=["group"],
            cate_attrs=["biased_bin"],
            num_attrs=["other_num"],
        )

        nmi_org = calculate_nmi_dict(X, Y)
        eps = evaluator.calculate_epsilon(X, O, cate_attrs=["biased_bin"], num_attrs=["other_num"])

        transformed_df, changed_dict, sel_o, sel_attr = bm.mitigate_step(
            X=X, Y=Y, O=O, nmi_org=nmi_org,
            changed_dict={}, current_epsilon=eps,
            epsilon_threshold=1e-9,  # unreachable -> terminal merge -> drop
        )

        self.assertEqual(sel_attr, "biased_bin")
        self.assertEqual(changed_dict.get("biased_bin"), "dropped")
        self.assertNotIn("biased_bin", transformed_df.columns)

    def _unreachable_setup(self):
        # Two numerical attributes, both with non-zero d_phi and an epsilon
        # threshold of 1e-9 that no power transform can reach (no float32
        # overflow either) -> the transform search necessarily fails.
        n = 120
        rng = np.random.default_rng(7)
        group = rng.integers(0, 2, n)
        X = pd.DataFrame({
            "num_a": rng.normal(loc=0.0, size=n) + 0.5 * group,
            "num_b": rng.normal(loc=0.0, size=n) + 0.3 * group,
        })
        Y = pd.Series(rng.integers(0, 2, n), name="target")
        O = pd.DataFrame({"group": group})
        evaluator = FairEvaluator(
            config=FairBiasConfig(random_seed=0),
            label_O=["group"], label_Y="target",
            cate_attrs=[], num_attrs=["num_a", "num_b"],
        )
        nmi_org = calculate_nmi_dict(X, Y)
        eps = evaluator.calculate_epsilon(X, O, cate_attrs=[], num_attrs=["num_a", "num_b"])
        return evaluator, X, Y, O, nmi_org, eps

    def test_stop_mode_records_non_convergence_on_highest_attribute(self):
        # Strict paper semantics (default "stop"): when the CURRENT highest
        # d_phi attribute's exhaustive search cannot reach the epsilon ball,
        # the engine records non-convergence and applies NO transform — it
        # must NOT silently move on to a lower-ranked attribute.
        evaluator, X, Y, O, nmi_org, eps = self._unreachable_setup()
        bm = FairBiasMitigation(
            evaluator=evaluator, transformer=FairTransform(),
            label_O=["group"], cate_attrs=[], num_attrs=["num_a", "num_b"],
            failed_attribute_mode="stop",
        )
        transformed_df, changed_dict, sel_o, sel_attr = bm.mitigate_step(
            X=X, Y=Y, O=O, nmi_org=nmi_org, changed_dict={},
            current_epsilon=eps, epsilon_threshold=1e-9,
        )
        self.assertIsNone(sel_attr, "stop mode must not accept any transform on failure")
        self.assertEqual(changed_dict, {})
        self.assertIsNotNone(bm.non_convergence, "failure must be recorded as non-convergence")
        # The recorded attribute is the CURRENT HIGHEST d_phi attribute
        ranked = bm.find_ranked_epsilon_attributes(eps)
        self.assertEqual(bm.non_convergence["attribute"], ranked[0][2])
        self.assertEqual(bm.non_convergence["label_O"], ranked[0][1])
        self.assertGreater(bm.non_convergence["d_phi"], 0.0)

    def test_next_mode_is_named_engineering_fallback(self):
        # "next" mode: failures are recorded keyed by (protected, feature)
        # and the next-ranked attribute is tried; non_convergence stays None
        # unless every candidate fails, in which case it is the LAST failure.
        evaluator, X, Y, O, nmi_org, eps = self._unreachable_setup()
        bm = FairBiasMitigation(
            evaluator=evaluator, transformer=FairTransform(),
            label_O=["group"], cate_attrs=[], num_attrs=["num_a", "num_b"],
            failed_attribute_mode="next",
        )
        transformed_df, changed_dict, sel_o, sel_attr = bm.mitigate_step(
            X=X, Y=Y, O=O, nmi_org=nmi_org, changed_dict={},
            current_epsilon=eps, epsilon_threshold=1e-9,
        )
        self.assertIsNone(sel_attr)
        # Both numerical attributes were tried and recorded as failed
        self.assertEqual(
            bm.failed_attribute_keys,
            {("group", "num_a"), ("group", "num_b")},
        )
        # Neither failure may permanently mask the other group's ranking
        self.assertEqual(bm.find_ranked_epsilon_attributes(eps), [])

    def test_next_mode_failure_does_not_mask_other_protected_group(self):
        # A failure recorded for feature f under protected group g1 must NOT
        # hide f under protected group g2 (the old name-only set did).
        bm = FairBiasMitigation(
            evaluator=self.evaluator, transformer=self.transformer,
            label_O=["g1", "g2"], cate_attrs=[], num_attrs=["f"],
            failed_attribute_mode="next",
        )
        bm.failed_attribute_keys.add(("g1", "f"))
        eps = {"g1": {"f": 0.9}, "g2": {"f": 0.8}}
        ranked = bm.find_ranked_epsilon_attributes(eps)
        self.assertEqual(ranked, [(0.8, "g2", "f")])

    def test_invalid_failed_attribute_mode_rejected(self):
        with self.assertRaises(ValueError):
            FairBiasMitigation(
                evaluator=self.evaluator, transformer=self.transformer,
                label_O=["group"], cate_attrs=[], num_attrs=[],
                failed_attribute_mode="skip",
            )


if __name__ == "__main__":
    unittest.main()
