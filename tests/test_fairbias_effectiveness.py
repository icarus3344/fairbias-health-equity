"""Effectiveness regression tests for the reworked FairBias algorithm.

These tests guard against the regression class found in the audit: an algorithm
that passes contract tests while producing zero bias reduction. They verify that
mitigation actually lowers d_phi, that polynomial powers change d_phi, that the
NMI information-loss gate works, and that public type annotations are resolvable.
"""

import typing
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from fairbias.config import FairBiasConfig
from fairbias.evaluator import FairEvaluator
from fairbias.mitigation import FairBiasMitigation
from fairbias.transform import FairTransform, calculate_nmi_dict


def _make_biased_frame(n: int = 600, seed: int = 7):
    """Synthetic dataset with injected bias against the protected group."""
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 2, n)

    # Categorical feature whose distribution strongly depends on the group
    cat_bias = np.where(
        group == 0,
        rng.choice([0, 1, 2, 3], n, p=[0.55, 0.25, 0.12, 0.08]),
        rng.choice([0, 1, 2, 3], n, p=[0.10, 0.15, 0.30, 0.45]),
    )
    # Numerical feature whose mean depends on the group
    num_bias = rng.normal(loc=group * 2.0, scale=1.0, size=n)
    # Second group-independent numerical feature (gives the mean-scaling family
    # a relative ratio so power transforms can move the scaled divergence)
    num_indep = rng.normal(loc=0.0, scale=1.5, size=n) + group * 0.05
    # Unbiased noise feature
    noise_cat = rng.integers(0, 2, n)

    y_prob = 1 / (1 + np.exp(-(0.6 * (cat_bias >= 2) + 0.2 * num_bias - 1.0)))
    y = (rng.uniform(0, 1, n) < y_prob).astype(int)

    X = pd.DataFrame({
        "biased_cat": cat_bias,
        "biased_num": num_bias,
        "num_indep": num_indep,
        "noise_cat": noise_cat,
    })
    O = pd.DataFrame({"group": group})
    Y = pd.Series(y, name="target")
    return X, Y, O


class TestMitigationEffectiveness(unittest.TestCase):
    """Verifies that bias mitigation produces measurable d_phi reductions."""

    def setUp(self):
        self.config = FairBiasConfig(random_seed=0)
        self.evaluator = FairEvaluator(
            config=self.config,
            label_O=["group"],
            label_Y="target",
            cate_attrs=["biased_cat", "noise_cat"],
            num_attrs=["biased_num", "num_indep"],
        )
        self.transformer = FairTransform()
        self.bm = FairBiasMitigation(
            evaluator=self.evaluator,
            transformer=self.transformer,
            label_O=["group"],
            cate_attrs=["biased_cat", "noise_cat"],
            num_attrs=["biased_num", "num_indep"],
        )

    def test_mitigation_strictly_reduces_dphi(self):
        X, Y, O = _make_biased_frame()
        nmi_org = calculate_nmi_dict(X, Y)
        eps0 = self.evaluator.calculate_epsilon(X, O)
        top_attr = max(eps0["group"], key=eps0["group"].get)
        dphi_before = eps0["group"][top_attr]

        transformed_df, changed, sel_o, sel_attr = self.bm.mitigate_step(
            X=X, Y=Y, O=O, nmi_org=nmi_org,
            changed_dict={}, current_epsilon=eps0, X_search=X,
        )

        self.assertIsNotNone(sel_attr, "Mitigation accepted no candidate on clearly biased data")
        eps1 = self.evaluator.calculate_epsilon(transformed_df, O)
        self.assertLess(
            eps1[sel_o][sel_attr], dphi_before if sel_attr == top_attr else eps0["group"][sel_attr],
            "Accepted candidate must strictly lower the selected feature's d_phi",
        )
        self.assertGreater(len(changed), 0)

    def test_iterated_mitigation_lowers_max_dphi(self):
        X, Y, O = _make_biased_frame()
        nmi_org = calculate_nmi_dict(X, Y)
        eps = self.evaluator.calculate_epsilon(X, O)
        max_before = max(eps["group"].values())

        changed = {}
        current = X
        for _ in range(3):
            current, changed, sel_o, sel_attr = self.bm.mitigate_step(
                X=X, Y=Y, O=O, nmi_org=nmi_org,
                changed_dict=changed, current_epsilon=eps, X_search=current,
            )
            if sel_attr is None:
                break
            eps = self.evaluator.calculate_epsilon(current, O)

        max_after = max(eps["group"].values())
        self.assertLess(max_after, max_before, "Iterated mitigation must lower max d_phi")

    def test_polynomial_power_changes_dphi(self):
        X, Y, O = _make_biased_frame()
        eps0 = self.evaluator.calculate_epsilon(X, O)["group"]["biased_num"]

        transformed = self.transformer.transform_data(
            X, {"biased_num": {"power": 1 / 3}}, self.bm.num_attrs, self.bm.cate_attrs
        )
        eps1 = self.evaluator.calculate_epsilon(transformed, O)["group"]["biased_num"]

        # A genuine nonlinear power transform must move d_phi (scale-only
        # transforms left it bit-invariant in the audited regression).
        self.assertGreater(abs(eps1 - eps0), 1e-9)

    def test_nmi_gate_rejects_information_destroying_candidate(self):
        X, Y, O = _make_biased_frame()
        nmi_org = calculate_nmi_dict(X, Y)
        eps = self.evaluator.calculate_epsilon(X, O)

        strict_bm = FairBiasMitigation(
            evaluator=self.evaluator,
            transformer=self.transformer,
            label_O=["group"],
            cate_attrs=["biased_cat", "noise_cat"],
            num_attrs=["biased_num", "num_indep"],
            phi_threshold=0.0,  # reject any candidate with non-negative NMI loss ratio > 0
        )

        # Force a positive phi (information loss) for every candidate
        with mock.patch(
            "fairbias.mitigation.calculate_nmi_dict",
            return_value={"biased_cat": 0.0, "biased_num": 0.0, "num_indep": 0.0, "noise_cat": 0.0},
        ):
            _, changed, sel_o, sel_attr = strict_bm.mitigate_step(
                X=X, Y=Y, O=O, nmi_org=nmi_org,
                changed_dict={}, current_epsilon=eps, X_search=X,
            )
        self.assertIsNone(sel_attr, "NMI gate must reject all candidates under strict threshold")
        self.assertEqual(changed, {})

        # Same data, lenient gate: acceptance possible again
        with mock.patch(
            "fairbias.mitigation.calculate_nmi_dict",
            return_value=dict(nmi_org),
        ):
            _, _, lenient_o, lenient_attr = self.bm.mitigate_step(
                X=X, Y=Y, O=O, nmi_org=nmi_org,
                changed_dict={}, current_epsilon=eps, X_search=X,
            )
        self.assertIsNotNone(lenient_attr)


class TestAnnotationIntegrity(unittest.TestCase):
    """Guards the P2 regression where unresolved annotations broke get_type_hints."""

    def test_get_type_hints_resolves_public_functions(self):
        import fairbias.evaluator as evaluator_mod
        import fairbias.bias_metric as bias_metric_mod

        typing.get_type_hints(evaluator_mod.compute_adaptive_threshold_kmeans_125)
        typing.get_type_hints(evaluator_mod.round_up_125)
        typing.get_type_hints(bias_metric_mod.get_subsets)
        typing.get_type_hints(bias_metric_mod.compute_bias_concentration)


if __name__ == "__main__":
    unittest.main()
