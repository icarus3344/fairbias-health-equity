"""Regression tests for audit findings from Codex Supervisor review.

These tests ensure that all verified risks and supervisor probes
have permanent executable regression guards.
"""

from __future__ import annotations

import unittest
from unittest import mock
import dataclasses
import json
import pathlib
import tempfile
import numpy as np
import pandas as pd

from fairbias.bias_metric import (
    compute_pairwise_divergences,
    compute_shapley_distance_matrix,
    compute_bias_concentration,
    compute_dphi_matrix,
)
from fairbias.config import FairBiasConfig
from fairbias.evaluator import FairEvaluator
from fairbias.transform import FairTransform
from fairbias.enhancement import FairAccuracyEnhancement, DEFAULT_POLY_GRID
from fairbias.enhancement_contracts import (
    EvaluationPartition,
    CandidateEvaluationResult,
    EnhancementStatus,
    FairnessEvaluationResult,
)
from fairbias.enhancement_state import (
    StatefulCandidateTracker,
    changed_dict_hash,
    hash_transform_state,
    transform_states_are_equivalent,
)
from nhis_fairbias.adapter import NHISStudyAdapter
from nhis_fairbias.preprocessing import NHISPreprocessor, NHISLeakageError
from nhis_fairbias.survey import (
    weighted_binary_proportion,
    weighted_category_proportion,
    positive_weight_mask,
    survey_design_audit,
)
from nhis_fairbias.d8_enhancement_runner import (
    D8EnhancementRunner,
    compute_group_fairness_gaps,
)
from scripts.run_nhis_d8_r4_substantive import R4_FROZEN_CONFIG


class TestAuditRemediationProbes(unittest.TestCase):
    """Targeted regression checks matching supervisor independent probes."""

    def setUp(self) -> None:
        self.cfg = FairBiasConfig(random_seed=0, classifier="LR", eval_norm="min-max", mds_fixed_components=1)
        self.ev = FairEvaluator(config=self.cfg, label_O=["o"], label_Y="y", num_attrs=["x"], cate_attrs=[])
        self.tf = FairTransform()

    def test_p1_01_multigroup_author_vs_extension_behavior(self) -> None:
        """P1-01: Multi-group Shapley distance matrix computation."""
        X = pd.DataFrame({"a": [0.0, 0.0, 1.0], "b": [0.0, 1.0, 0.0]})
        o_3 = pd.Series([0, 1, 2], name="grp")

        df_s_3 = compute_pairwise_divergences(X, o_3, cate_attrs=[], num_attrs=["a", "b"])

        dist_mean, nodes = compute_shapley_distance_matrix(
            df_s_3, ["a", "b"], h_order=0, multigroup_aggregation="mean_pair"
        )
        dist_author, _ = compute_shapley_distance_matrix(
            df_s_3, ["a", "b"], h_order=0, multigroup_aggregation="author_max_pair"
        )

        idx_a = nodes.index("a")
        idx_b = nodes.index("b")
        self.assertAlmostEqual(dist_mean[idx_a, idx_b], 2.0 / 3.0, places=5)
        self.assertAlmostEqual(dist_author[idx_a, idx_b], 0.0, places=5)

        # Binary check: must be exactly equal
        X_bin = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0], "b": [1.0, 0.0, 1.0, 0.0]})
        o_bin = pd.Series([0, 0, 1, 1], name="grp")
        df_s_bin = compute_pairwise_divergences(X_bin, o_bin, cate_attrs=[], num_attrs=["a", "b"])

        dist_bin_mean, _ = compute_shapley_distance_matrix(
            df_s_bin, ["a", "b"], h_order=1, multigroup_aggregation="mean_pair"
        )
        dist_bin_auth, _ = compute_shapley_distance_matrix(
            df_s_bin, ["a", "b"], h_order=1, multigroup_aggregation="author_max_pair"
        )
        np.testing.assert_allclose(dist_bin_mean, dist_bin_auth)

    def test_p1_02_all_missing_numeric_group_fail_closed(self) -> None:
        """R2 / Probe 1: Group with all NaN in numeric feature must fail closed."""
        xx = pd.DataFrame({"x": [np.nan, np.nan, 2.0, 3.0]})
        O = pd.DataFrame({"o": [0, 0, 1, 1]})
        with self.assertRaises(ValueError):
            compute_dphi_matrix(xx, O, [], ["x"], mds_fixed_components=1)

    def test_p1_02_all_missing_categorical_group_fail_closed(self) -> None:
        """R2 / Probe 2: Group with all None in categorical feature must fail closed."""
        xx = pd.DataFrame({"x": [None, None, 0, 1]})
        O = pd.DataFrame({"o": [0, 0, 1, 1]})
        with self.assertRaises(ValueError):
            compute_dphi_matrix(xx, O, ["x"], [], mds_fixed_components=1)

    def test_p1_02_undeclared_feature_fail_closed(self) -> None:
        """R2 / Probe 3: Feature present in X but not declared in num/cat must fail closed."""
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        O = pd.DataFrame({"o": [0, 0, 1, 1]})
        with self.assertRaises(ValueError):
            compute_dphi_matrix(X, O, [], [], mds_fixed_components=1)

    def test_p1_05_single_group_matrix_fail_closed(self) -> None:
        """R2 / Probe 4: Single protected group must raise ValueError."""
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        O = pd.DataFrame({"o": [0, 0, 0, 0]})
        with self.assertRaises(ValueError):
            compute_dphi_matrix(X, O, [], ["x"])

    def test_p1_02_ae_rejects_missing_group_geometry(self) -> None:
        """R2 / Probe 5: AE._is_fairness_acceptable must reject missing group geometry."""
        ae = FairAccuracyEnhancement(self.ev, self.tf, "y", [], ["x"], max_fairness_degradation=0)
        O = pd.DataFrame({"o": [0, 0, 1, 1]})
        fair = ae._is_fairness_acceptable(pd.DataFrame({"x": [np.nan, np.nan, 2.0, 3.0]}), O, 0.0005, None)
        self.assertFalse(fair.is_acceptable)
        self.assertFalse(fair.strictly_feasible)
        self.assertFalse(fair.relaxed_feasible)

    def test_p1_05_single_group_dp_not_estimable(self) -> None:
        """R2 / Probe 6: Single-group DP is unestimable and must return None."""
        gaps = compute_group_fairness_gaps(np.array([1, 0]), np.array([1, 0]), np.array([0, 0]))
        self.assertIsNone(gaps["demographic_parity_difference"])
        self.assertEqual(gaps["dp_unestimable_reason"], "INSUFFICIENT_GROUPS")

    def test_p1_05_undefined_tpr_returns_null(self) -> None:
        """R2 / Probe 7: Group with 0 positives leaves TPR undefined -> returns None."""
        gaps2 = compute_group_fairness_gaps(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, 1]))
        self.assertIsNone(gaps2["equal_opportunity_difference"])
        self.assertEqual(gaps2["eo_unestimable_reason"], "MISSING_POSITIVE_SAMPLES")

    def test_p1_03_infinite_weights_not_inference_eligible(self) -> None:
        """R4 / Probe 8: Infinite weight must disqualify survey inference eligibility."""
        design = survey_design_audit(
            pd.DataFrame({"WTFA_A": [np.inf, 1.0, 1.0, 1.0], "PSTRAT": [1, 1, 2, 2], "PPSU": [1, 2, 1, 2]}),
            year=2024,
        )
        self.assertFalse(design["survey_inference_eligible"])
        self.assertEqual(design["status"], "FAIL")
        self.assertIsNone(design["wtfa_a_sum"])

    def test_p2_02_valid_zero_event_is_zero(self) -> None:
        """R4 / Probe 9: Valid denominator with 0 occurrences returns 0.0."""
        zero = weighted_binary_proportion(pd.Series([2, 2]), pd.Series([1.0, 1.0]), code=1)
        self.assertEqual(zero, 0.0)

    def test_p1_06_temporal_missing_metadata_rejected(self) -> None:
        """R5 / Probe 10: Temporal regime without year/role provenance fails closed."""
        pre = NHISPreprocessor()
        frame = pd.DataFrame({
            f: [pre.specs[f].get("substantive_codes", [1])[0]] * 4
            if f in pre.categorical_features
            else [0.0, 1.0, 2.0, 3.0]
            for f in pre.expanded_features
        })
        with self.assertRaises(NHISLeakageError):
            pre.fit(frame)

    def test_p1_06_identical_fit_selection_rejected(self) -> None:
        """R5 / Probe 11: Identical fit and selection partitions must be rejected."""
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        y = pd.Series([0, 0, 1, 1])
        O = pd.DataFrame({"o": [0, 0, 1, 1]})
        with self.assertRaises(ValueError):
            EvaluationPartition(X, y, X, y, protected_fit=O)

    def test_p1_06_allow_real_data_public_setter_blocked(self) -> None:
        """R5 / Probe 12: allow_real_data property has no setter and cannot be mutated."""
        runner = D8EnhancementRunner(allow_real_data=False)
        with self.assertRaises(AttributeError):
            runner.allow_real_data = True

    def test_p2_05_manifest_grid_matches_engine(self) -> None:
        """R3 / Probe 14: Manifest declared grid matches engine active parameters."""
        actual = FairAccuracyEnhancement(self.ev, self.tf, "y", [], ["x"])
        declared = R4_FROZEN_CONFIG["enhancement_parameters"]
        self.assertEqual(tuple(declared["polynomial_exponent_grid"]), actual.poly_exponents)

    def test_p2_05_manifest_min_gain_matches_engine(self) -> None:
        """R3 / Probe 15: Manifest declared min gain matches engine active parameters."""
        actual = FairAccuracyEnhancement(self.ev, self.tf, "y", [], ["x"])
        declared = R4_FROZEN_CONFIG["enhancement_parameters"]
        self.assertEqual(declared["minimum_utility_gain"], actual.min_utility_gain)

    def test_p2_01_cache_key_uses_content_fingerprint(self) -> None:
        """R1 / Probe 16: Fingerprint method call discriminates partitions differing in truncation zone."""
        train = pd.DataFrame({"x": np.linspace(0.0, 1.0, 200)})
        label = pd.Series(([0] * 100) + ([1] * 100))
        prot = pd.DataFrame({"o": np.tile([0, 1], 100)})
        valid = train.copy()
        valid.index = np.arange(1000, 1200)
        vlabel = pd.Series(label.to_numpy(), index=valid.index)
        vlabel2 = vlabel.copy()
        vlabel2.iloc[90] = 1

        p1 = EvaluationPartition(train, label, valid, vlabel, protected_fit=prot)
        p2 = EvaluationPartition(train, label, valid, vlabel2, protected_fit=prot)

        k1 = f"sel={p1.selection_fingerprint()};eps=0.005"
        k2 = f"sel={p2.selection_fingerprint()};eps=0.005"
        self.assertNotEqual(k1, k2)
        self.assertNotIn("bound method", k1)

    def test_p2_01_cache_rechecks_hidden_selection_change(self) -> None:
        """R1 / Probe 17: Engine re-evaluates candidate when selection partition changes in hidden zone."""
        train = pd.DataFrame({"x": np.linspace(0.0, 1.0, 200)})
        label = pd.Series(([0] * 100) + ([1] * 100))
        prot = pd.DataFrame({"o": np.tile([0, 1], 100)})
        valid = train.copy()
        valid.index = np.arange(1000, 1200)
        vlabel = pd.Series(label.to_numpy(), index=valid.index)
        vlabel2 = vlabel.copy()
        vlabel2.iloc[90] = 1

        p1 = EvaluationPartition(train, label, valid, vlabel, protected_fit=prot)
        p2 = EvaluationPartition(train, label, valid, vlabel2, protected_fit=prot)

        engine = FairAccuracyEnhancement(self.ev, self.tf, "y", [], ["x"], poly_exponents=[3.0])
        calls = []

        def fake_utility(*args, **kwargs):
            p = kwargs["partition"]
            changed = bool(kwargs["changed_dict"])
            calls.append({"second_partition": p is p2, "candidate": changed})
            score = 0.6 if not changed else (0.9 if p is p2 else 0.5)
            return CandidateEvaluationResult(EnhancementStatus.VALID, score, model_fit_count=0)

        with mock.patch("fairbias.enhancement.evaluate_candidate_utility", side_effect=fake_utility), \
             mock.patch.object(self.ev, "calculate_epsilon", return_value={"o": {"x": 0.001}}):
            r1 = engine.enhance_step(train, label, {}, partition=p1, epsilon_threshold=0.005)
            r2 = engine.enhance_step(train, label, {}, partition=p2, epsilon_threshold=0.005)

        self.assertEqual(r2[2], "x")

    def test_p1_02_strict_relaxed_flags_correct(self) -> None:
        """R1 / Probe 18: Strict and relaxed feasibility flags are correctly distinguished."""
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        O = pd.DataFrame({"o": [0, 0, 1, 1]})
        strictae = FairAccuracyEnhancement(self.ev, self.tf, "y", [], ["x"], max_fairness_degradation=0.02)
        with mock.patch.object(self.ev, "calculate_epsilon", return_value={"o": {"x": 0.01}}):
            f = strictae._is_fairness_acceptable(X, O, 0.005, None)
        self.assertFalse(f.strictly_feasible)
        self.assertTrue(f.relaxed_feasible)
        self.assertTrue(f.is_acceptable)

    def test_probe_geometry_config_change_rechecks_engine_candidate(self) -> None:
        """Probe 1: Cache context must differ for a materially different geometry configuration."""
        X = pd.DataFrame({"x": np.linspace(0.05, 0.95, 20)})
        y = pd.Series([0] * 10 + [1] * 10)
        O = pd.DataFrame({"o": np.tile([0, 1], 10)})
        vX = X.copy()
        vX.index = pd.RangeIndex(100, 120)
        vy = pd.Series(y.to_numpy(), index=vX.index)
        p = EvaluationPartition(X, y, vX, vy, protected_fit=O)

        cfg0 = FairBiasConfig(random_seed=0, classifier="LR", h_order=0, mds_fixed_components=1)
        ev = FairEvaluator(config=cfg0, label_O=["o"], label_Y="y", num_attrs=["x"], cate_attrs=[])
        ae = FairAccuracyEnhancement(ev, FairTransform(), "y", [], ["x"], poly_exponents=[3.0], max_fairness_degradation=0)

        key0 = ae._build_candidate_cache_context_fingerprint(p, 0.005, None)

        def utility(**kw):
            return CandidateEvaluationResult(EnhancementStatus.VALID, 0.9 if kw["changed_dict"] else 0.6)

        def geometry(*args, **kwargs):
            return {"o": {"x": 0.01 if ev.config.h_order == 0 else 0.001}}

        with mock.patch("fairbias.enhancement.evaluate_candidate_utility", side_effect=utility), \
             mock.patch.object(ev, "calculate_epsilon", side_effect=geometry):
            first = ae.enhance_step(X, y, {}, partition=p, epsilon_threshold=0.005)
            # Change geometry config to h_order=1
            ev.config = dataclasses.replace(ev.config, h_order=1)
            key1 = ae._build_candidate_cache_context_fingerprint(p, 0.005, None)
            self.assertNotEqual(key0, key1)
            second = ae.enhance_step(X, y, {}, partition=p, epsilon_threshold=0.005)
            fresh = FairAccuracyEnhancement(ev, FairTransform(), "y", [], ["x"], poly_exponents=[3.0], max_fairness_degradation=0)
            control = fresh.enhance_step(X, y, {}, partition=p, epsilon_threshold=0.005)

        self.assertEqual(second[2], "x")
        self.assertEqual(second[2], control[2])

    def test_probe_lossless_power_candidate_signatures(self) -> None:
        """Probe 2: Distinct power-grid values must not be aliased by float formatting."""
        X = pd.DataFrame({"x": np.linspace(0.05, 0.95, 20)})
        y = pd.Series([0] * 10 + [1] * 10)
        O = pd.DataFrame({"o": np.tile([0, 1], 10)})
        vX = X.copy()
        vX.index = pd.RangeIndex(100, 120)
        vy = pd.Series(y.to_numpy(), index=vX.index)
        p = EvaluationPartition(X, y, vX, vy, protected_fit=O)

        cfg = FairBiasConfig(random_seed=0, classifier="LR", h_order=0, mds_fixed_components=1)
        ev = FairEvaluator(config=cfg, label_O=["o"], label_Y="y", num_attrs=["x"], cate_attrs=[])
        ae = FairAccuracyEnhancement(ev, FairTransform(), "y", [], ["x"], poly_exponents=[3.00001, 3.00002], max_fairness_degradation=0)

        powers = []
        def power_utility(**kw):
            power = kw["changed_dict"].get("x", {}).get("power")
            if power is not None:
                powers.append(power)
            score = 0.6 if power is None else (0.5 if power == 3.00001 else 0.9)
            return CandidateEvaluationResult(EnhancementStatus.VALID, score)

        with mock.patch("fairbias.enhancement.evaluate_candidate_utility", side_effect=power_utility), \
             mock.patch.object(ev, "calculate_epsilon", return_value={"o": {"x": 0.001}}):
            got = ae.enhance_step(X, y, {}, partition=p, epsilon_threshold=0.005)
            control = FairAccuracyEnhancement(ev, FairTransform(), "y", [], ["x"], poly_exponents=[3.00002], max_fairness_degradation=0).enhance_step(X, y, {}, partition=p, epsilon_threshold=0.005)

        self.assertIn(3.00001, powers)
        self.assertIn(3.00002, powers)
        self.assertEqual(got[1], control[1])
        self.assertEqual(got[1]["x"]["power"], 3.00002)

    def test_probe_cycle_history_scoped_to_evaluation_context(self) -> None:
        """Probe 3: Cycle history must not contaminate a new evaluation context."""
        X = pd.DataFrame({"x": np.linspace(0.05, 0.95, 20)})
        y = pd.Series([0] * 10 + [1] * 10)
        O = pd.DataFrame({"o": np.tile([0, 1], 10)})
        vX = X.copy()
        vX.index = pd.RangeIndex(100, 120)
        vy = pd.Series(y.to_numpy(), index=vX.index)
        p = EvaluationPartition(X, y, vX, vy, protected_fit=O)

        vy2 = vy.copy()
        vy2.iloc[2] = 1
        p2 = EvaluationPartition(X, y, vX, vy2, protected_fit=O)

        cfg = FairBiasConfig(random_seed=0, classifier="LR", h_order=0, mds_fixed_components=1)
        ev = FairEvaluator(config=cfg, label_O=["o"], label_Y="y", num_attrs=["x"], cate_attrs=[])
        ae = FairAccuracyEnhancement(ev, FairTransform(), "y", [], ["x"], poly_exponents=[3.0], max_fairness_degradation=0)

        def utility(**kw):
            return CandidateEvaluationResult(EnhancementStatus.VALID, 0.9 if kw["changed_dict"] else 0.6)

        with mock.patch("fairbias.enhancement.evaluate_candidate_utility", side_effect=utility), \
             mock.patch.object(ev, "calculate_epsilon", return_value={"o": {"x": 0.001}}):
            first = ae.enhance_step(X, y, {}, partition=p, epsilon_threshold=0.005)
            ae.enhance_step(X, y, first[1], partition=p, epsilon_threshold=0.005)
            second = ae.enhance_step(X, y, {}, partition=p2, epsilon_threshold=0.005)
            fresh = FairAccuracyEnhancement(ev, FairTransform(), "y", [], ["x"], poly_exponents=[3.0], max_fairness_degradation=0)
            control = fresh.enhance_step(X, y, {}, partition=p2, epsilon_threshold=0.005)

        self.assertEqual(second[2], "x")
        self.assertEqual(second[2], control[2])

    def test_probe_partial_overlap_rejected(self) -> None:
        """Probe 4, 5, 6: Partial overlap from same source rejected regardless of Index type."""
        source = pd.DataFrame({"x": np.arange(6, dtype=float)})
        labels = pd.Series([0, 1, 0, 1, 0, 1])

        # RangeIndex vs RangeIndex
        left_range = source.iloc[:4].copy()
        right_range = source.iloc[2:].copy()
        with self.assertRaises(ValueError):
            EvaluationPartition(left_range, labels.loc[left_range.index], right_range, labels.loc[right_range.index])

        # RangeIndex vs Int64Index
        right_int = source.iloc[2:].copy()
        right_int.index = pd.Index(right_int.index.to_numpy())
        with self.assertRaises(ValueError):
            EvaluationPartition(left_range, labels.loc[left_range.index], right_int, labels.loc[right_int.index])

        # Int64Index vs Int64Index
        left_int = source.iloc[:4].copy()
        left_int.index = pd.Index(left_int.index.to_numpy())
        with self.assertRaises(ValueError):
            EvaluationPartition(left_int, labels.loc[left_int.index], right_int, labels.loc[right_int.index])

        # Positive control: distinct source namespaces with overlapping numerical indices are accepted
        p_distinct = EvaluationPartition(
            left_int, labels.loc[left_int.index], right_int, labels.loc[right_int.index],
            fit_source="2022_train", selection_source="2023_val"
        )
        self.assertIsNotNone(p_distinct)

    def test_probe_adapter_rejects_unspecified_prefitted_preprocessor(self) -> None:
        """Probe 7: Adapter rejects preprocessor fitted with unspecified source."""
        pre = NHISPreprocessor()
        frame = pd.DataFrame({
            f: [pre.specs[f].get("substantive_codes", [1])[0]] * 4
            if f in pre.categorical_features
            else [float((pre.specs[f].get("substantive_codes") or [0])[0])] * 4
            for f in pre.expanded_features
        })
        pre.fit(frame, allow_unspecified_source=True)
        meta = pd.DataFrame({
            "survey_year": np.repeat([2022, 2023, 2024], [27651, 29522, 32629]),
            "study_role": np.repeat(["development_train", "development_validation", "frozen_test"], [27651, 29522, 32629]),
        })
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp_f:
            placeholder_path = pathlib.Path(tmp_f.name)
            with mock.patch("nhis_fairbias.adapter.pd.read_parquet", return_value=meta):
                with self.assertRaises(ValueError) as ctx:
                    NHISStudyAdapter(features_parquet_path=placeholder_path, preprocessor=pre)
                self.assertIn("invalid provenance", str(ctx.exception))

    def test_probe_tiebreak_retains_first_grid_occurrence(self) -> None:
        """Probe 8: Tie in utility retains first occurrence in grid, matching declared method."""
        cfg = FairBiasConfig(random_seed=0, classifier="LR", h_order=0, mds_fixed_components=1)
        ev = FairEvaluator(config=cfg, label_O=["o"], label_Y="y", num_attrs=["x"], cate_attrs=[])
        ae = FairAccuracyEnhancement(ev, FairTransform(), "y", [], ["x"], poly_exponents=[3.0, 5.0], max_fairness_degradation=0)

        # Confirm declared method matches substantive execution
        active_params = ae.get_active_parameters()
        self.assertEqual(
            active_params["candidate_ranking_method"],
            "strictly_highest_utility_gain_ties_retain_first_grid_occurrence",
        )

        def tie_utility(**kw):
            return CandidateEvaluationResult(EnhancementStatus.VALID, 0.9 if kw["changed_dict"] else 0.6)

        def tie_geometry(xx, *args, **kwargs):
            return {"o": {"x": 0.004 if xx.iloc[0, 0] > 1e-5 else 0.001}}

        X = pd.DataFrame({"x": np.linspace(0.05, 0.95, 20)})
        y = pd.Series([0] * 10 + [1] * 10)
        O = pd.DataFrame({"o": np.tile([0, 1], 10)})
        vX = X.copy()
        vX.index = pd.RangeIndex(100, 120)
        vy = pd.Series(y.to_numpy(), index=vX.index)
        p = EvaluationPartition(X, y, vX, vy, protected_fit=O)

        with mock.patch("fairbias.enhancement.evaluate_candidate_utility", side_effect=tie_utility), \
             mock.patch.object(ev, "calculate_epsilon", side_effect=tie_geometry):
            got = ae.enhance_step(X, y, {}, partition=p, epsilon_threshold=0.005)

        # Under first grid occurrence rule, p=3 is retained
        self.assertEqual(got[1]["x"]["power"], 3.0)

    def test_probe_candidate_audit_preserves_boolean_json_types(self) -> None:
        """Probe 9: JSON serialization preserves true/false bool types for audit fields."""
        cfg = FairBiasConfig(random_seed=0, classifier="LR", h_order=0, mds_fixed_components=1)
        ev = FairEvaluator(config=cfg, label_O=["o"], label_Y="y", num_attrs=["x"], cate_attrs=[])
        ae = FairAccuracyEnhancement(ev, FairTransform(), "y", [], ["x"], poly_exponents=[3.0], max_fairness_degradation=0.02)

        def utility(**kw):
            return CandidateEvaluationResult(EnhancementStatus.VALID, 0.9 if kw["changed_dict"] else 0.6)

        X = pd.DataFrame({"x": np.linspace(0.05, 0.95, 20)})
        y = pd.Series([0] * 10 + [1] * 10)
        O = pd.DataFrame({"o": np.tile([0, 1], 10)})
        vX = X.copy()
        vX.index = pd.RangeIndex(100, 120)
        vy = pd.Series(y.to_numpy(), index=vX.index)
        p = EvaluationPartition(X, y, vX, vy, protected_fit=O)

        with mock.patch("fairbias.enhancement.evaluate_candidate_utility", side_effect=utility), \
             mock.patch.object(ev, "calculate_epsilon", return_value={"o": {"x": 0.01}}):
            ae.enhance_step(X, y, {}, partition=p, epsilon_threshold=0.005)

        self.assertGreater(len(ae.audit_trail), 0)
        evt_dict = ae.audit_trail[0].to_dict()
        serialized = json.loads(json.dumps(evt_dict))

        for k in ["accepted", "strictly_feasible", "relaxed_feasible"]:
            self.assertIn(k, serialized)
            self.assertIs(type(serialized[k]), bool, f"{k} is {type(serialized[k])}, expected bool")

    def test_probe_survey_empty_and_overflow_not_eligible(self) -> None:
        """Probe 10, 11: Empty sample and overflowing sum must be disqualified."""
        # Empty sample
        f_empty = pd.DataFrame({"WTFA_A": [], "PSTRAT": [], "PPSU": []})
        got_empty = survey_design_audit(f_empty, year=2024)
        self.assertFalse(got_empty["survey_inference_eligible"])

        # Sum overflow
        f_over = pd.DataFrame({"WTFA_A": [1e308, 1e308], "PSTRAT": [1, 1], "PPSU": [0, 1]})
        with np.errstate(over="ignore"):
            got_over = survey_design_audit(f_over, year=2024)
        self.assertFalse(got_over["survey_inference_eligible"])

    def test_independent_namespaces_identical_values_allowed(self) -> None:
        """Independent cohorts sharing measured values and local indices must be accepted."""
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        y = pd.Series([0, 1, 0, 1])
        p = EvaluationPartition(X, y, X.copy(), y.copy(), fit_source="NHIS:2022", selection_source="NHIS:2023")
        self.assertIsNotNone(p)

    def test_mixed_year_partial_overlap_rejected(self) -> None:
        """Shared record across mixed-year cohorts (e.g. 2023:1) must be rejected."""
        left = pd.DataFrame({"x": [0.0, 1.0], "survey_year": [2022, 2023]}, index=[0, 1])
        right = pd.DataFrame({"x": [1.0, 2.0], "survey_year": [2023, 2024]}, index=[1, 2])
        with self.assertRaises(ValueError):
            EvaluationPartition(left, pd.Series([0, 1], index=left.index), right, pd.Series([1, 0], index=right.index))

    def test_duplicate_ids_within_partition_rejected(self) -> None:
        """Duplicate source IDs within fit partition must be rejected."""
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        y = pd.Series([0, 1, 0, 1])
        left = X.copy()
        left.index = [0, 0, 1, 2]
        right = X.copy()
        right.index = [3, 4, 5, 6]
        with self.assertRaises(ValueError):
            EvaluationPartition(
                left, pd.Series(y.to_numpy(), index=left.index),
                right, pd.Series(y.to_numpy(), index=right.index),
                fit_source="S", selection_source="S",
            )

    def test_audit_partition_fingerprints_bind_source(self) -> None:
        """Partition fingerprints must bind the source namespace."""
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        y = pd.Series([0, 1, 0, 1])
        right = X.copy()
        right.index = [10, 11, 12, 13]
        p1 = EvaluationPartition(X, y, right, pd.Series(y.to_numpy(), index=right.index), fit_source="A", selection_source="B")
        p2 = EvaluationPartition(X, y, right, pd.Series(y.to_numpy(), index=right.index), fit_source="C", selection_source="D")
        self.assertNotEqual(p1.fit_fingerprint(), p2.fit_fingerprint())
        self.assertNotEqual(p1.selection_fingerprint(), p2.selection_fingerprint())

    def test_runner_active_parameter_getter_callable_without_data(self) -> None:
        """D8EnhancementRunner.get_active_parameters must be callable without data access."""
        runner = D8EnhancementRunner(allow_real_data=False)
        params = runner.get_active_parameters()
        self.assertIn("enhancement_parameters", params)
        self.assertEqual(
            params["enhancement_parameters"]["candidate_ranking_method"],
            "strictly_highest_utility_gain_ties_retain_first_grid_occurrence",
        )

    def test_active_parameters_match_smoke_budgets(self) -> None:
        """Smoke test budgets in active_parameters must dynamically reflect smoke_test=True."""
        smoke_runner = D8EnhancementRunner(smoke_test=True, allow_real_data=False)
        params = smoke_runner.get_active_parameters()
        budgets = params["enhancement_parameters"]["search_budgets"]
        self.assertEqual(budgets["condition_3_max_steps"], 2)
        self.assertEqual(budgets["condition_4_max_iterations"], 3)

    def test_historical_state_barrier_all_arms(self) -> None:
        """Historical state barrier hashes match frozen reference under v1_legacy_8dec."""
        ref_hashes = {
            "D6_ARM_001": "40511e6c0d55b0ff",
            "D6_ARM_002": "4c0bbba5d40022d6",
            "D6_ARM_003": "38fa54a06a9c6427",
            "D6_ARM_004": "ff0a2fb81596b598",
        }
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        for arm, exp_hash in ref_hashes.items():
            state_file = repo_root / f"docs/releases/NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7/{arm}/frozen_changed_dict.json"
            state = json.loads(state_file.read_text())["changed_dict"]
            h_v1 = changed_dict_hash(state, version="v1_legacy_8dec")
            self.assertEqual(h_v1, exp_hash)
            if arm == "D6_ARM_003":
                h_v2 = hash_transform_state(state, version="v2_lossless")
                self.assertEqual(h_v2, "95ce9da442e66adb")


if __name__ == "__main__":
    unittest.main()
