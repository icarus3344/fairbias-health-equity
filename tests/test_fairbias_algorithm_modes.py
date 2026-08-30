"""Algorithm-mode separation tests (Round 4.1; mode renamed twice —
the second rename is the Round 4.1 REPAIR-2 verdict of 2026-08-30).

Round 4 reported a state under the name ``paper_strict`` while the
implementation still used an automatic MDS dimension (official code:
fixed dim=2), a six-value ascending power grid (official code: the
interleaved stream [3, 1/3, 5, 1/5, ..., 1999, 1/1999] in that order),
and a finite iteration budget.  Round 4.1 replaced that overclaiming
name with two explicitly defined modes; the official-derived mode was
renamed AGAIN by the REPAIR-2 verdict (the interim name
``official_code_unweighted_reimplementation`` overclaimed official-code
compatibility, because the monotone cursor deviates from the official
restart-from-head power search):

- ``official_code_derived_monotone_cursor_unweighted``: an
  OFFICIAL-CODE-DERIVED VARIANT WITH A TERMINATION-SAFETY EXTENSION —
  derived from the official code repository's behavior (MDS fixed
  dim=2, official interleaved power stream searched in order, NO finite
  iteration budget, NO validation-Pareto rollback) but NOT claimed to
  be behaviorally equivalent to it: the monotone per-attribute stream
  cursor is a DELIBERATE deviation from the official restart-from-head
  search (it permanently excludes powers the official code would retry
  after other attributes change).  It is also NOT a paper-text method
  reproduction: the paper text prescribes elbow-plot MDS dimension
  selection, which this mode does not implement.
- ``engineering_bounded``: the bounded configuration (automatic MDS
  dimension, six-value ascending grid, finite max_iterations, Pareto
  checkpoint) with NO paper-alignment claim.

These tests pin the mode contract, the (Round 4.1 REPAIR, Codex P0)
monotone power-stream cursor that guarantees termination of the
budget-free official-derived loop, and the (Round 4.1 REPAIR-2, Codex
P1) numeric-vs-categorical failure-scope recording.
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

        self.assertEqual(res.algorithm_mode, "official_code_derived_monotone_cursor_unweighted")
        # No Pareto state exists in official-code-derived mode.
        self.assertIsNone(res.pareto_engineering_metrics)
        self.assertIsNone(res.pareto_engineering_changed_dict)

        with open(res.output_file, encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(
            payload["final_states"],
            ["official_code_derived_monotone_cursor_unweighted"],
        )
        self.assertIn(
            "final_results_official_code_derived_monotone_cursor_unweighted",
            payload,
        )
        pareto_keys = [k for k in payload if "pareto" in k.lower()]
        self.assertEqual(
            pareto_keys, [],
            "official-code-derived mode must not contain any Pareto output",
        )
        # The round-4 overclaiming key must not resurface.
        self.assertNotIn("final_results_paper_strict", payload)
        self.assertNotIn("final_results_configured_greedy_terminal", payload)
        # The superseded round-4.1 names must not resurface either.
        self.assertNotIn("final_results_official_unweighted_reproduction", payload)
        self.assertNotIn(
            "final_results_official_code_unweighted_reimplementation", payload
        )
        self.assertEqual(
            payload["final_results_official_code_derived_monotone_cursor_unweighted"][
                "metrics_partition"
            ],
            "test",
        )
        self.assertEqual(
            payload["termination"]["algorithm_mode"],
            "official_code_derived_monotone_cursor_unweighted",
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


class TestOfficialModeMonotonePowerStreamCursor(unittest.TestCase):
    """Round 4.1 REPAIR (Codex P0): the official-code-derived mode must persist a
    MONOTONE per-attribute cursor into the power stream.

    Without the cursor, ``_search_numerical`` restarted from the head of
    the stream on every revisit of an attribute (skipping only the
    currently applied power), so a numeric attribute could oscillate
    between powers (supervisor diagnostic: ``3 -> 1/3 -> 3``) forever
    under the budget-free official loop.  These tests construct the
    revisit scenario directly and prove that powers are never reused and
    cannot oscillate.
    """

    def _make_engine(self, preserve_order, exponents):
        from fairbias.mitigation import FairBiasMitigation
        from fairbias.transform import FairTransform

        evaluator = FairEvaluator(config=FairBiasConfig(random_seed=0))
        return FairBiasMitigation(
            evaluator=evaluator,
            transformer=FairTransform(),
            label_O=["g"],
            cate_attrs=[],
            num_attrs=["num"],
            poly_exponents=exponents,
            failed_attribute_mode="stop",
            preserve_exponent_order=preserve_order,
        )

    def _frames(self):
        rng = np.random.default_rng(0)
        n = 40
        X = pd.DataFrame({"num": rng.normal(size=n)})
        Y = pd.Series(rng.integers(0, 2, n), name="y")
        O = pd.DataFrame({"g": rng.integers(0, 2, n)})
        return X, Y, O

    def _accept_all_powers(self, bm, X):
        """Patch ``_make_candidate`` so every power candidate is accepted.

        Returns a list that records each accepted power; "dropped"
        candidates are rejected so the search keeps consuming stream
        positions instead of terminating through the drop path.
        """
        accepted = []

        def fake_make_candidate(self, X_, changed_dict, attr, change, *a, **kw):
            if not isinstance(change, dict) or "power" not in change:
                return None  # reject the drop candidate
            accepted.append(float(change["power"]))
            new_changed = dict(changed_dict)
            new_changed[attr] = change
            return X_.copy(), new_changed, 0.0

        patcher = mock.patch.object(
            type(bm), "_make_candidate", autospec=True, side_effect=fake_make_candidate
        )
        return accepted, patcher

    def test_revisited_attribute_never_reuses_or_oscillates_powers(self):
        # Supervisor diagnostic reproduction: with the interleaved stream
        # (3, 1/3, 5, 1/5) and every candidate accepted, repeated revisits
        # of the SAME attribute must consume the stream strictly forward:
        # 3, 1/3, 5, 1/5, then failure.  The pre-repair code accepted 3
        # AGAIN on the third visit (3 -> 1/3 -> 3 -> ...) — an infinite
        # oscillation under the budget-free official loop.
        stream = (3.0, 1 / 3, 5.0, 1 / 5)
        bm = self._make_engine(True, stream)
        X, Y, O = self._frames()
        accepted, patcher = self._accept_all_powers(bm, X)

        current_epsilon = {"g": {"num": 0.9}}
        nmi_org = {"num": 1.0}
        changed_dict = {}
        with patcher:
            for _ in range(len(stream)):
                _, changed_dict, sel_o, sel_attr = bm.mitigate_step(
                    X=X, Y=Y, O=O, nmi_org=nmi_org,
                    changed_dict=changed_dict,
                    current_epsilon=current_epsilon,
                    epsilon_threshold=0.5,
                )
                self.assertEqual(sel_attr, "num")
            # Fifth visit: the stream is exhausted for this attribute.
            _, changed_dict, sel_o, sel_attr = bm.mitigate_step(
                X=X, Y=Y, O=O, nmi_org=nmi_org,
                changed_dict=changed_dict,
                current_epsilon=current_epsilon,
                epsilon_threshold=0.5,
            )

        self.assertEqual(
            accepted, [3.0, 1 / 3, 5.0, 1 / 5],
            "revisits must consume the interleaved stream strictly forward",
        )
        # No power may ever be reused (the pre-repair oscillation).
        self.assertNotIn(3.0, accepted[1:], "power 3.0 must not be reused")
        self.assertEqual(len(set(accepted)), len(accepted))
        # The fifth visit finds nothing: the failure is recorded with the
        # official stream scope (fail closed, no transform accepted).
        self.assertIsNone(sel_attr)
        self.assertIsNotNone(bm.non_convergence)
        self.assertEqual(
            bm.non_convergence["search_scope"], "official_power_stream"
        )
        self.assertEqual(
            bm.non_convergence["stream_positions_consumed"], len(stream)
        )
        # The cursor sits past the end of the stream: every revisit from
        # here on fails immediately (termination guarantee).
        self.assertEqual(bm._exponent_stream_cursors["num"], len(stream))

    def test_cursor_survives_rejected_candidates(self):
        # A power that is tried and REJECTED must also be consumed: the
        # cursor only ever moves forward, so a later revisit of the same
        # attribute resumes AFTER the rejected position.
        stream = (3.0, 1 / 3, 5.0, 1 / 5)
        bm = self._make_engine(True, stream)
        X, Y, O = self._frames()
        calls = []

        def fake_make_candidate(self, X_, changed_dict, attr, change, *a, **kw):
            if not isinstance(change, dict) or "power" not in change:
                return None
            power = float(change["power"])
            calls.append(power)
            if power == 3.0:
                return None  # first position always rejected
            new_changed = dict(changed_dict)
            new_changed[attr] = change
            return X_.copy(), new_changed, 0.0

        current_epsilon = {"g": {"num": 0.9}}
        nmi_org = {"num": 1.0}
        changed_dict = {}
        with mock.patch.object(
            type(bm), "_make_candidate", autospec=True,
            side_effect=fake_make_candidate,
        ):
            # Visit 1: 3.0 rejected, 1/3 accepted.
            _, changed_dict, _, sel_attr = bm.mitigate_step(
                X=X, Y=Y, O=O, nmi_org=nmi_org, changed_dict=changed_dict,
                current_epsilon=current_epsilon, epsilon_threshold=0.5,
            )
            self.assertEqual(sel_attr, "num")
            self.assertEqual(calls, [3.0, 1 / 3])
            # Visit 2: must resume AFTER 1/3 (positions 0 and 1 are
            # consumed); the rejected 3.0 must NOT be retried.
            _, changed_dict, _, sel_attr = bm.mitigate_step(
                X=X, Y=Y, O=O, nmi_org=nmi_org, changed_dict=changed_dict,
                current_epsilon=current_epsilon, epsilon_threshold=0.5,
            )
            self.assertEqual(sel_attr, "num")
            self.assertEqual(calls, [3.0, 1 / 3, 5.0])

        self.assertEqual(bm._exponent_stream_cursors["num"], 3)

    def test_engineering_mode_keeps_legacy_restart_and_no_cursor(self):
        # The engineering mode keeps the legacy restart-from-head search
        # (bounded there by max_iterations) and must NOT consume stream
        # positions via the cursor.
        stream = (3.0, 1 / 3, 5.0, 1 / 5)
        bm = self._make_engine(False, stream)
        self.assertEqual(
            bm.poly_exponents, tuple(sorted(stream)),
            "engineering mode sorts ascending (legacy behaviour)",
        )
        X, Y, O = self._frames()
        accepted, patcher = self._accept_all_powers(bm, X)

        current_epsilon = {"g": {"num": 0.9}}
        nmi_org = {"num": 1.0}
        changed_dict = {}
        with patcher:
            for _ in range(3):
                _, changed_dict, _, sel_attr = bm.mitigate_step(
                    X=X, Y=Y, O=O, nmi_org=nmi_org,
                    changed_dict=changed_dict,
                    current_epsilon=current_epsilon,
                    epsilon_threshold=0.5,
                )
                self.assertEqual(sel_attr, "num")

        # Ascending order consumed: 1/5, 1/3, then 1/5 AGAIN — the
        # legacy restart-from-head search CAN reuse powers (this is why
        # the monotone cursor is REQUIRED for the budget-free official
        # loop; in engineering mode the reuse is bounded by
        # max_iterations).
        self.assertEqual(accepted, [1 / 5, 1 / 3, 1 / 5])
        # No cursor state may be created in engineering mode.
        self.assertEqual(bm._exponent_stream_cursors, {})


class TestCategoricalFailureScopeRecording(unittest.TestCase):
    """Round 4.1 REPAIR-2 (Codex P1): in the official-code-derived mode a
    CATEGORICAL attribute failure must be recorded as
    ``search_scope="categorical_merge_chain"`` — NOT as
    ``official_power_stream`` exhaustion.  The pre-repair code recorded
    the power-stream scope (and a zero ``stream_positions_consumed``)
    unconditionally in this mode, mis-reporting categorical failures as
    power-stream exhaustion.  The supervisor reproduced this defect with
    a constructed categorical failure; these tests pin the corrected
    numeric-vs-categorical scope split.
    """

    def _make_engine(self, preserve_order, cate_attrs, num_attrs, exponents):
        from fairbias.mitigation import FairBiasMitigation
        from fairbias.transform import FairTransform

        evaluator = FairEvaluator(config=FairBiasConfig(random_seed=0))
        return FairBiasMitigation(
            evaluator=evaluator,
            transformer=FairTransform(),
            label_O=["g"],
            cate_attrs=cate_attrs,
            num_attrs=num_attrs,
            poly_exponents=exponents,
            failed_attribute_mode="stop",
            preserve_exponent_order=preserve_order,
        )

    def _frames(self):
        rng = np.random.default_rng(1)
        n = 40
        X = pd.DataFrame({
            "cat": pd.Series(rng.choice(["a", "b", "c"], size=n)),
            "num": rng.normal(size=n),
        })
        Y = pd.Series(rng.integers(0, 2, n), name="y")
        O = pd.DataFrame({"g": rng.integers(0, 2, n)})
        return X, Y, O

    def test_categorical_failure_records_merge_chain_scope(self):
        # Every categorical candidate is rejected: the merge chain is
        # exhausted and the failure must carry the CATEGORICAL scope.
        stream = (3.0, 1 / 3, 5.0, 1 / 5)
        bm = self._make_engine(True, cate_attrs=["cat"], num_attrs=[],
                               exponents=stream)
        X, Y, O = self._frames()

        with mock.patch.object(
            type(bm), "_make_candidate", autospec=True,
            side_effect=lambda self, *a, **kw: None,
        ):
            _, changed, sel_o, sel_attr = bm.mitigate_step(
                X=X, Y=Y, O=O, nmi_org={"cat": 1.0}, changed_dict={},
                current_epsilon={"g": {"cat": 0.9}}, epsilon_threshold=0.5,
            )

        self.assertIsNone(sel_attr, "no transform may be accepted on failure")
        self.assertIsNotNone(bm.non_convergence)
        self.assertEqual(bm.non_convergence["attribute"], "cat")
        self.assertEqual(
            bm.non_convergence["search_scope"], "categorical_merge_chain"
        )
        # Categorical failures must NOT carry the power-stream audit
        # fields: no cursor positions exist for a categorical attribute.
        self.assertNotIn("stream_positions_consumed", bm.non_convergence)
        self.assertIn(
            "categorical merge chain", bm.non_convergence["reason"]
        )
        self.assertNotIn("power stream", bm.non_convergence["reason"])
        # The power-stream cursor state is untouched by categorical search.
        self.assertEqual(bm._exponent_stream_cursors, {})

    def test_numeric_failure_still_records_power_stream_scope(self):
        # Contrast: a NUMERIC failure keeps the power-stream scope with
        # the consumed-position audit field.
        stream = (3.0, 1 / 3, 5.0, 1 / 5)
        bm = self._make_engine(True, cate_attrs=[], num_attrs=["num"],
                               exponents=stream)
        X, Y, O = self._frames()

        with mock.patch.object(
            type(bm), "_make_candidate", autospec=True,
            side_effect=lambda self, *a, **kw: None,
        ):
            _, changed, sel_o, sel_attr = bm.mitigate_step(
                X=X, Y=Y, O=O, nmi_org={"num": 1.0}, changed_dict={},
                current_epsilon={"g": {"num": 0.9}}, epsilon_threshold=0.5,
            )

        self.assertIsNone(sel_attr)
        self.assertIsNotNone(bm.non_convergence)
        self.assertEqual(
            bm.non_convergence["search_scope"], "official_power_stream"
        )
        self.assertEqual(
            bm.non_convergence["stream_positions_consumed"], len(stream)
        )
        self.assertNotIn("categorical", bm.non_convergence["reason"])


class TestTerminationNoteFollowsRecordedScope(unittest.TestCase):
    """Round 4.1 REPAIR-2 (Codex P1): the pipeline termination note must
    be generated from the REAL recorded ``search_scope`` — a categorical
    failure must not be described as power-stream exhaustion."""

    def test_categorical_scope_note_mentions_merge_chain(self):
        from fairbias.pipeline import _termination_note

        note = _termination_note(
            "candidate_grid_exhausted", True,
            {"search_scope": "categorical_merge_chain"},
        )
        self.assertIn("categorical merge chain", note)
        self.assertNotIn("power stream", note)

    def test_numeric_scope_note_mentions_power_stream(self):
        from fairbias.pipeline import _termination_note

        note = _termination_note(
            "candidate_grid_exhausted", True,
            {"search_scope": "official_power_stream"},
        )
        self.assertIn("power stream", note)
        self.assertIn("monotone", note)
        self.assertNotIn("categorical merge chain", note)

    def test_official_non_exhausted_note_unchanged(self):
        from fairbias.pipeline import _termination_note

        note = _termination_note("epsilon_reached", True, None)
        self.assertIn("no finite iteration budget", note)

    def test_engineering_exhausted_note_unchanged(self):
        from fairbias.pipeline import _termination_note

        note = _termination_note(
            "candidate_grid_exhausted", False,
            {"search_scope": "configured_grid"},
        )
        self.assertIn("CONFIGURED transform grid", note)
        self.assertNotIn("official", note)


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
