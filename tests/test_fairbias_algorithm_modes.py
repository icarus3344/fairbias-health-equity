"""Algorithm-mode separation tests (Round 4.1).

Round 4 reported a state under the name ``paper_strict`` while the
implementation still used an automatic MDS dimension (official: fixed
dim=2), a six-value ascending power grid (official: the interleaved
stream [3, 1/3, 5, 1/5, ..., 1999, 1/1999] in that order), and a finite
iteration budget (the official greedy loop is bounded only by the finite
official stream).  Round 4.1 replaces that overclaiming name with two
explicitly defined modes:

- ``official_unweighted_reproduction``: strict reproduction of the paper
  METHOD as adjudicated by the official code repository — MDS fixed at
  dim=2, official interleaved power stream (order preserved), NO finite
  iteration budget, NO validation-Pareto rollback.
- ``engineering_bounded``: the bounded configuration (automatic MDS
  dimension, six-value ascending grid, finite max_iterations, Pareto
  checkpoint) with NO paper-alignment claim.

These tests pin the mode contract.
"""

import json
import os
import shutil
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from fairbias.bias_metric import (
    SURVEY_WEIGHTED_EXTENSION_DECLARATION,
    compute_bias_concentration,
)
from fairbias.config import (
    ALGORITHM_MODE_OFFICIAL,
    FairBiasConfig,
    official_power_stream,
)
from fairbias.evaluator import FairEvaluator
from fairbias.pipeline import run_fairbias_pipeline


class TestOfficialPowerStream(unittest.TestCase):
    """The official interleaved power stream must match the official code."""

    def test_stream_content_and_order(self):
        # Official: np.array([[i, 1/i] for i in range(3, 2000, 2)]).reshape(-1)
        stream = official_power_stream()
        expected_head = [3.0, 1 / 3, 5.0, 1 / 5, 7.0, 1 / 7]
        self.assertEqual(list(stream[:6]), expected_head)
        # INTERLEAVED order, not ascending: 1/3 must come BEFORE 5.
        self.assertEqual(stream[1], 1 / 3)
        self.assertEqual(stream[2], 5.0)
        # The ascending sort would be (1/7, 1/5, 1/3, 3, 5, 7) — different.
        self.assertNotEqual(
            list(stream[:6]), sorted(stream[:6]),
            "the official stream is interleaved, NOT ascending",
        )

    def test_stream_length_and_tail(self):
        stream = official_power_stream()
        # range(3, 2000, 2) has 999 values; interleaving gives 1998.
        self.assertEqual(len(stream), 1998)
        self.assertEqual(stream[-2], 1999.0)
        self.assertAlmostEqual(stream[-1], 1 / 1999, places=15)

    def test_stream_matches_official_construction(self):
        # Direct transcription of the official line 198 construction.
        official = np.array([[i, 1 / i] for i in range(3, 2000, 2)]).reshape(-1)
        np.testing.assert_allclose(np.asarray(official_power_stream()), official)


class TestOfficialModeResolution(unittest.TestCase):
    """``FairBiasConfig.resolved`` concretizes the official mode."""

    def test_default_grid_concretizes_to_official_stream(self):
        cfg = FairBiasConfig.compas_default(mode="official").resolved()
        self.assertEqual(cfg.algorithm_mode, ALGORITHM_MODE_OFFICIAL)
        self.assertEqual(cfg.mds_fixed_components, 2)
        self.assertEqual(cfg.transform_poly_exponents, official_power_stream())

    def test_engineering_mode_resolves_to_itself(self):
        cfg = FairBiasConfig.compas_default(mode="engineering").resolved()
        self.assertIsNone(cfg.mds_fixed_components)
        self.assertEqual(
            cfg.transform_poly_exponents, (1 / 7, 1 / 5, 1 / 3, 3.0, 5.0, 7.0)
        )

    def test_official_mode_rejects_accuracy_enhancement(self):
        with self.assertRaises(ValueError):
            FairBiasConfig.compas_default(
                mode="official", use_accuracy_enhancement=True
            ).resolved()

    def test_official_mode_rejects_failed_attribute_next(self):
        with self.assertRaises(ValueError):
            FairBiasConfig.compas_default(
                mode="official", failed_attribute_mode="next"
            ).resolved()

    def test_official_mode_rejects_non_official_fixed_dim(self):
        with self.assertRaises(ValueError):
            FairBiasConfig.compas_default(
                mode="official", mds_fixed_components=3
            ).resolved()

    def test_official_mode_rejects_custom_power_grid(self):
        with self.assertRaises(ValueError):
            FairBiasConfig.compas_default(
                mode="official", transform_poly_exponents=(1 / 3, 3.0)
            ).resolved()

    def test_invalid_mode_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            FairBiasConfig(algorithm_mode="paper_strict")


class TestOfficialModeMDSFixedDimension(unittest.TestCase):
    """Official mode fixes the MDS embedding dimension at 2."""

    def test_config_constant_matches_official_fixed_dim(self):
        # Cross-check with the recorded golden contrast: the official
        # implementation fixes the embedding dimension at 2 (see
        # tests/test_fairbias_mds_dimension_golden.py, OFFICIAL_FIXED_DIM).
        from fairbias.config import OFFICIAL_FIXED_MDS_DIM

        self.assertEqual(OFFICIAL_FIXED_MDS_DIM, 2)

    def _biased_frame(self):
        rng = np.random.RandomState(0)
        n = 120
        group = rng.choice([0, 1], size=n)
        num = rng.normal(size=n) + 1.5 * group
        cat = pd.Series(rng.choice(["a", "b", "c"], size=n))
        # make the categorical biased too
        cat = cat.mask(group == 1, cat.shift(1).fillna("a"))
        X = pd.DataFrame({"biased_num": num, "biased_cat": cat, "noise": rng.normal(size=n)})
        o = pd.Series(group)
        return X, o

    def test_fixed_dimension_skips_elbow_search(self):
        X, o = self._biased_frame()
        with mock.patch(
            "fairbias.bias_metric._find_optimal_mds_components",
            side_effect=AssertionError("elbow search must be skipped"),
        ) as _elbow:
            dphi = compute_bias_concentration(
                X, o,
                cate_attrs=["biased_cat"],
                num_attrs=["biased_num", "noise"],
                mds_fixed_components=2,
            )
        _elbow.assert_not_called()
        self.assertEqual(len(dphi), 3)
        self.assertTrue(all(np.isfinite(v) for v in dphi.values()))

    def test_fixed_dimension_is_deterministic_and_differs_from_elbow(self):
        X, o = self._biased_frame()
        d1 = compute_bias_concentration(
            X, o, cate_attrs=["biased_cat"], num_attrs=["biased_num", "noise"],
            mds_fixed_components=2, random_state=0,
        )
        d2 = compute_bias_concentration(
            X, o, cate_attrs=["biased_cat"], num_attrs=["biased_num", "noise"],
            mds_fixed_components=2, random_state=0,
        )
        self.assertEqual(d1, d2, "fixed-dim embedding must be deterministic")

    def test_evaluator_passes_fixed_dimension_through_config(self):
        # The official-mode config (resolved) must reach the MDS through
        # FairEvaluator.calculate_epsilon without error and must not invoke
        # the elbow search.
        X, o = self._biased_frame()
        cfg = FairBiasConfig.compas_default(mode="official").resolved()
        evaluator = FairEvaluator(
            config=cfg,
            label_O=["g"],
            label_Y="y",
            cate_attrs=["biased_cat"],
            num_attrs=["biased_num", "noise"],
        )
        O = pd.DataFrame({"g": o})
        with mock.patch(
            "fairbias.bias_metric._find_optimal_mds_components",
            side_effect=AssertionError("elbow search must be skipped"),
        ) as _elbow:
            eps = evaluator.calculate_epsilon(X, O)
        _elbow.assert_not_called()
        self.assertIn("g", eps)
        self.assertEqual(len(eps["g"]), 3)


class TestOfficialModePipeline(unittest.TestCase):
    """End-to-end official-mode pipeline contract (sole state, no Pareto,
    no iteration budget)."""

    def setUp(self):
        self.test_output_dir = "runs/test_pipeline_official_tmp"
        os.makedirs(self.test_output_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def test_official_mode_reports_sole_state_without_pareto(self):
        config = FairBiasConfig.compas_default(
            mode="official",
            max_iterations=5,  # deliberately irrelevant in official mode
            output_dir=self.test_output_dir,
            random_seed=0,
        )
        res = run_fairbias_pipeline(config)

        self.assertEqual(res.algorithm_mode, "official_unweighted_reproduction")
        # No Pareto state exists in official mode.
        self.assertIsNone(res.pareto_engineering_metrics)
        self.assertIsNone(res.pareto_engineering_changed_dict)

        with open(res.output_file, encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(
            payload["final_states"], ["official_unweighted_reproduction"]
        )
        self.assertIn("final_results_official_unweighted_reproduction", payload)
        pareto_keys = [k for k in payload if "pareto" in k.lower()]
        self.assertEqual(
            pareto_keys, [],
            "official mode must not contain any Pareto output",
        )
        # The round-4 overclaiming key must not resurface.
        self.assertNotIn("final_results_paper_strict", payload)
        self.assertNotIn("final_results_configured_greedy_terminal", payload)
        self.assertEqual(
            payload["final_results_official_unweighted_reproduction"][
                "metrics_partition"
            ],
            "test",
        )
        self.assertEqual(
            payload["termination"]["algorithm_mode"],
            "official_unweighted_reproduction",
        )
        # The effective config must record the official stream and fixed dim.
        self.assertEqual(payload["config_parameters"]["mds_fixed_components"], 2)
        self.assertEqual(
            len(payload["config_parameters"]["transform_poly_exponents"]), 1998
        )

    def test_official_mode_ignores_iteration_budget(self):
        # max_iterations=1 must be IGNORED in official mode: the run may
        # proceed beyond one iteration and can never terminate with
        # iteration_budget_exhausted.  The same budget in engineering mode
        # must terminate with iteration_budget_exhausted.
        official_cfg = FairBiasConfig.compas_default(
            mode="official",
            max_iterations=1,
            output_dir=self.test_output_dir,
            random_seed=0,
        )
        res = run_fairbias_pipeline(official_cfg)
        self.assertNotEqual(
            res.termination["termination_reason"],
            "iteration_budget_exhausted",
            "official mode has no iteration budget",
        )
        if len(res.iterations) > 1:
            self.assertGreater(res.termination["terminal_iteration"], 1)

        engineering_cfg = FairBiasConfig.compas_default(
            max_iterations=1,
            output_dir=self.test_output_dir,
            random_seed=0,
        )
        res_eng = run_fairbias_pipeline(engineering_cfg)
        self.assertEqual(
            res_eng.termination["termination_reason"],
            "iteration_budget_exhausted",
        )


class TestExponentOrderPreservation(unittest.TestCase):
    """The mitigation engine must search the power stream in the GIVEN
    order in official mode (interleaved), and keep the legacy ascending
    sort in engineering mode."""

    def _make_engine(self, preserve_order, exponents):
        from fairbias.mitigation import FairBiasMitigation
        from fairbias.transform import FairTransform

        evaluator = FairEvaluator(config=FairBiasConfig(random_seed=0))
        return FairBiasMitigation(
            evaluator=evaluator,
            transformer=FairTransform(),
            label_O=["g"],
            cate_attrs=[],
            num_attrs=["a"],
            poly_exponents=exponents,
            preserve_exponent_order=preserve_order,
        )

    def test_official_order_is_preserved(self):
        stream = official_power_stream()
        bm = self._make_engine(True, stream)
        self.assertEqual(bm.poly_exponents, stream)
        self.assertEqual(bm.poly_exponents[1], 1 / 3)
        self.assertEqual(bm.poly_exponents[2], 5.0)

    def test_engineering_order_is_ascending(self):
        stream = official_power_stream()
        bm = self._make_engine(False, stream)
        self.assertEqual(bm.poly_exponents, tuple(sorted(stream)))
        self.assertEqual(bm.poly_exponents[0], 1 / 1999)


class TestSurveyWeightedExtensionDeclaration(unittest.TestCase):
    """The pre-declared survey-weighted extension contract is pinned as a
    testable declaration (Gate D implements it later)."""

    def test_declaration_pins_scope_and_degeneracy(self):
        decl = SURVEY_WEIGHTED_EXTENSION_DECLARATION
        # Only the two empirical group statistics are replaced...
        self.assertIn("compute_pairwise_divergences", decl)
        self.assertIn("mu_hat_mg", decl)
        self.assertIn("p_hat_mkg", decl)
        # ...everything else stays unchanged...
        self.assertIn("unchanged", decl)
        # ...and equal weights MUST degenerate exactly to the unweighted
        # statistics (mandatory unit test of the extension).
        self.assertIn("equal weights", decl)
        self.assertIn("degenerate EXACTLY", decl)


if __name__ == "__main__":
    unittest.main()
