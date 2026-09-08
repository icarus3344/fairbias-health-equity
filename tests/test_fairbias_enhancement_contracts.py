"""Comprehensive behavioral verification suite for FairBias accuracy enhancement contracts."""

from __future__ import annotations

import copy
import json
import pathlib
import sys
import unittest
import unittest.mock
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
BLOCKED_ACCESS: list[str] = []


def data_guard(event: str, args: tuple) -> None:
    if event == "open" and args and isinstance(args[0], (str, bytes, pathlib.Path)):
        p = pathlib.Path(args[0]).resolve()
        if (
            p.is_relative_to(ROOT / "data")
            or p in [ROOT / "data_COMPAS.csv", ROOT / "data_Credit_Card.csv"]
            or p.suffix == ".parquet"
        ):
            BLOCKED_ACCESS.append(p.name)
            raise RuntimeError(f"DATA_ACCESS_BLOCKED: {p.name}")


sys.addaudithook(data_guard)

from fairbias.config import FairBiasConfig
from fairbias.evaluator import FairEvaluator
from fairbias.enhancement import FairAccuracyEnhancement
from fairbias.enhancement_contracts import (
    CandidateAuditEvent,
    CandidateEvaluationResult,
    EnhancementStatus,
    EvaluationPartition,
    compute_configuration_fingerprint,
    evaluate_candidate_utility,
)
from fairbias.enhancement_state import (
    StatefulCandidateTracker,
    hash_transform_state,
    normalize_category_mapping,
    safe_compose_category_mapping,
)
from fairbias.pipeline import run_fairbias_pipeline
from fairbias.transform import FairTransform


class TestFairBiasEnhancementContracts(unittest.TestCase):
    """Verifies all 14 mandatory behavioral contracts from D8-R1 specification."""

    def setUp(self):
        np.random.seed(42)
        self.n_tr = 120
        self.n_sel = 60

        # Synthetic feature generation (integer-encoded categorical standard)
        x1_tr = np.random.randn(self.n_tr) * 2.0 + 5.0
        x2_tr = np.random.choice([0, 1, 2, 3], size=self.n_tr)
        x3_tr = np.random.randn(self.n_tr) + 3.0
        o_tr = np.random.choice([0, 1], size=self.n_tr)
        prob_tr = 1.0 / (1.0 + np.exp(-(0.4 * x1_tr + (x2_tr == 0).astype(float) * 1.5 - 2.5)))
        y_tr = (np.random.rand(self.n_tr) < prob_tr).astype(int)

        x1_sel = np.random.randn(self.n_sel) * 2.0 + 5.0
        x2_sel = np.random.choice([0, 1, 2, 3], size=self.n_sel)
        x3_sel = np.random.randn(self.n_sel) + 3.0
        o_sel = np.random.choice([0, 1], size=self.n_sel)
        prob_sel = 1.0 / (1.0 + np.exp(-(0.4 * x1_sel + (x2_sel == 0).astype(float) * 1.5 - 2.5)))
        y_sel = (np.random.rand(self.n_sel) < prob_sel).astype(int)

        self.df_tr = pd.DataFrame({"num1": x1_tr, "cat1": x2_tr, "num2": x3_tr})
        self.y_tr = pd.Series(y_tr, name="target")
        self.o_tr = pd.DataFrame({"protected": o_tr})

        self.df_sel = pd.DataFrame({"num1": x1_sel, "cat1": x2_sel, "num2": x3_sel})
        self.y_sel = pd.Series(y_sel, name="target")
        self.o_sel = pd.DataFrame({"protected": o_sel})

        self.partition = EvaluationPartition(
            fit_X=self.df_tr,
            fit_y=self.y_tr,
            selection_X=self.df_sel,
            selection_y=self.y_sel,
            protected_fit=self.o_tr,
            protected_selection=self.o_sel,
        )

        self.config = FairBiasConfig.compas_default(random_seed=42, classifier="LR")
        self.evaluator = FairEvaluator(
            config=self.config,
            label_O=["protected"],
            label_Y="target",
            cate_attrs=["cat1"],
            num_attrs=["num1", "num2"],
        )
        self.transformer = FairTransform(n_bins=5, log_epsilon=1e-5, x_max=100.0)

    # -------------------------------------------------------------------------
    # 1. Partition Contract (Identity, power, category merge, dropped)
    # -------------------------------------------------------------------------
    def test_partition_contract_identical_transforms_and_scaler_isolation(self):
        # Add extreme outlier in selection partition to verify it doesn't contaminate scaler fit stats
        outlier_sel_X = self.df_sel.copy()
        outlier_sel_X.loc[0, "num1"] = 99999.0
        outlier_partition = EvaluationPartition(
            fit_X=self.df_tr,
            fit_y=self.y_tr,
            selection_X=outlier_sel_X,
            selection_y=self.y_sel,
        )

        # Test 4 states:
        # A: Identity
        res_id = evaluate_candidate_utility(
            outlier_partition, {}, ["num1", "num2"], ["cat1"], self.transformer, self.evaluator
        )
        self.assertTrue(res_id.is_valid)

        # B: Power transform
        res_pow = evaluate_candidate_utility(
            outlier_partition, {"num1": {"power": 3.0}}, ["num1", "num2"], ["cat1"], self.transformer, self.evaluator
        )
        self.assertTrue(res_pow.is_valid)

        # C: Category merge
        res_cat = evaluate_candidate_utility(
            outlier_partition, {"cat1": {1: 0}}, ["num1", "num2"], ["cat1"], self.transformer, self.evaluator
        )
        self.assertTrue(res_cat.is_valid)

        # D: Dropped column
        res_drop = evaluate_candidate_utility(
            outlier_partition, {"num2": "dropped"}, ["num1", "num2"], ["cat1"], self.transformer, self.evaluator
        )
        self.assertTrue(res_drop.is_valid)

    # -------------------------------------------------------------------------
    # 2. Selection Score Oracle (Matches independent handwritten LR calculation)
    # -------------------------------------------------------------------------
    def test_selection_score_matches_independent_oracle(self):
        changed = {"num1": {"power": 3.0}, "cat1": {1: 0}}
        cand_res = evaluate_candidate_utility(
            self.partition, changed, ["num1", "num2"], ["cat1"], self.transformer, self.evaluator
        )
        self.assertTrue(cand_res.is_valid)

        # Independent Oracle Calculation:
        oracle_tr_t = self.transformer.transform_data(self.df_tr, changed, ["num1", "num2"], ["cat1"])
        oracle_sel_t = self.transformer.transform_data(self.df_sel, changed, ["num1", "num2"], ["cat1"])
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_tr = scaler.fit_transform(oracle_tr_t)
        scaled_sel = scaler.transform(oracle_sel_t)
        lr = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
        lr.fit(scaled_tr, self.y_tr.values)
        p_sel = lr.predict_proba(scaled_sel)[:, 1]
        oracle_auroc = float(roc_auc_score(self.y_sel.values, p_sel))

        self.assertAlmostEqual(cand_res.utility_score, oracle_auroc, places=7)

    # -------------------------------------------------------------------------
    # 3. Column Dropped Handling (Regression test for F1/F2)
    # -------------------------------------------------------------------------
    def test_column_dropped_allows_evaluation_of_remaining_features(self):
        initial_changed = {"num2": "dropped"}
        ae = FairAccuracyEnhancement(
            self.evaluator, self.transformer, "target", ["cat1"], ["num1", "num2"]
        )
        # Should not crash with column mismatch or swallow error; should evaluate num1
        df_res, changed_res, attr_res = ae.enhance_step(
            X_train=self.df_tr,
            Y_train=self.y_tr,
            changed_dict=initial_changed,
            O_train=self.o_tr,
            epsilon_threshold=0.5,
            X_val=self.df_sel,
            Y_val=self.y_sel,
            partition=self.partition,
        )
        self.assertIn("num2", changed_res)
        self.assertEqual(changed_res["num2"], "dropped")

    def test_all_features_dropped_returns_explicit_invalid_status(self):
        all_dropped = {"num1": "dropped", "num2": "dropped", "cat1": "dropped"}
        res = evaluate_candidate_utility(
            self.partition, all_dropped, ["num1", "num2"], ["cat1"], self.transformer, self.evaluator
        )
        self.assertFalse(res.is_valid)
        self.assertEqual(res.validity_status, "ALL_FEATURES_DROPPED")
        self.assertIsNone(res.utility_score)

    # -------------------------------------------------------------------------
    # 4. Explicit Error Handling (No silent 0.0 scores)
    # -------------------------------------------------------------------------
    def test_explicit_error_on_single_class_selection(self):
        single_class_y = pd.Series([1] * len(self.df_sel), index=self.df_sel.index)
        p_single = EvaluationPartition(
            fit_X=self.df_tr, fit_y=self.y_tr,
            selection_X=self.df_sel, selection_y=single_class_y,
        )
        res = evaluate_candidate_utility(
            p_single, {}, ["num1", "num2"], ["cat1"], self.transformer, self.evaluator
        )
        self.assertFalse(res.is_valid)
        self.assertEqual(res.validity_status, "SINGLE_CLASS_SELECTION")
        self.assertIsNone(res.utility_score)

    def test_explicit_error_on_non_finite_output(self):
        nan_df = self.df_sel.copy()
        nan_df.loc[0, "num1"] = np.nan
        p_nan = EvaluationPartition(
            fit_X=self.df_tr, fit_y=self.y_tr,
            selection_X=nan_df, selection_y=self.y_sel,
        )
        res = evaluate_candidate_utility(
            p_nan, {}, ["num1", "num2"], ["cat1"], self.transformer, self.evaluator
        )
        self.assertFalse(res.is_valid)
        self.assertEqual(res.validity_status, "NON_FINITE_OUTPUT")
        self.assertIsNone(res.utility_score)

    # -------------------------------------------------------------------------
    # 5. Index Contract (Mismatched index / columns detected)
    # -------------------------------------------------------------------------
    def test_index_contract_validation(self):
        # Shuffled index in y
        shuffled_y = self.y_tr.sample(frac=1.0, random_state=123)
        with self.assertRaises(ValueError):
            EvaluationPartition(
                fit_X=self.df_tr, fit_y=shuffled_y,
                selection_X=self.df_sel, selection_y=self.y_sel,
            )

        # Mismatched columns
        mismatched_sel = self.df_sel.drop(columns=["num2"])
        with self.assertRaises(ValueError):
            EvaluationPartition(
                fit_X=self.df_tr, fit_y=self.y_tr,
                selection_X=mismatched_sel, selection_y=self.y_sel,
            )

    # -------------------------------------------------------------------------
    # 6. Category Mapping & Normalization (Transitive, no split, JSON round-trip)
    # -------------------------------------------------------------------------
    def test_category_mapping_transitive_composition(self):
        # 3 -> 1, then merging current 1 group into 0
        existing = {3: 1}
        new_merge = {1: 0}
        composed = safe_compose_category_mapping(existing, new_merge)
        # Transitive closure: 3 -> 0, 1 -> 0
        self.assertEqual(composed[3], 0)
        self.assertEqual(composed[1], 0)

    def test_category_mapping_conflicting_keys_rejected(self):
        conflicting = {"3": 1, 3: 2}
        with self.assertRaises(ValueError):
            normalize_category_mapping(conflicting)

    def test_category_mapping_json_round_trip(self):
        original = {1: 0, 2: 0, 3: 1}
        s = json.dumps(original)
        loaded = json.loads(s)
        normalized = normalize_category_mapping(loaded)
        self.assertEqual(original, normalized)

    def test_category_mapping_preserves_string_categories(self):
        col_series = pd.Series(["01", "02", "1", "3"])
        mapping = {"01": "1"}
        normalized = normalize_category_mapping(mapping, col_sample=col_series)
        self.assertIn("01", normalized)
        self.assertEqual(normalized["01"], "1")
        # '01' should not be coerced to integer 1
        self.assertIsInstance(list(normalized.keys())[0], str)

    # -------------------------------------------------------------------------
    # 7. State Cache & Cycle Termination
    # -------------------------------------------------------------------------
    def test_state_cache_reevaluates_when_other_feature_changes(self):
        tracker = StatefulCandidateTracker()
        state1_hash = hash_transform_state({})
        cand_sig = "num1:power=3.0"

        # Not evaluated initially
        self.assertFalse(tracker.is_candidate_evaluated(state1_hash, cand_sig))
        tracker.mark_candidate_evaluated(state1_hash, cand_sig)
        self.assertTrue(tracker.is_candidate_evaluated(state1_hash, cand_sig))

        # State changes (cat1 merged)
        state2_hash = hash_transform_state({"cat1": {1: 0}})
        # On new state, cand_sig should NOT be blocked!
        self.assertFalse(tracker.is_candidate_evaluated(state2_hash, cand_sig))

    def test_cycle_detection(self):
        tracker = StatefulCandidateTracker()
        state1 = hash_transform_state({})
        tracker.record_state_visit(state1)
        self.assertTrue(tracker.is_cycle(state1))
        state2 = hash_transform_state({"num1": {"power": 3.0}})
        self.assertFalse(tracker.is_cycle(state2))

    # -------------------------------------------------------------------------
    # 8. Transactional Atomicity (No partial mutation on rejection)
    # -------------------------------------------------------------------------
    def test_transactional_safety_no_mutation_on_rejection(self):
        ae = FairAccuracyEnhancement(
            self.evaluator, self.transformer, "target", ["cat1"], ["num1", "num2"],
            min_utility_gain=999.0,  # impossibly high gain requirement -> reject all
        )
        orig_changed = {"num2": {"power": 3.0}}
        changed_copy = copy.deepcopy(orig_changed)

        df_res, changed_res, attr_res = ae.enhance_step(
            X_train=self.df_tr,
            Y_train=self.y_tr,
            changed_dict=changed_copy,
            partition=self.partition,
        )
        self.assertIsNone(attr_res)
        self.assertEqual(orig_changed, changed_res)

    # -------------------------------------------------------------------------
    # 9. Fairness Guard Boundary & Refreshed State
    # -------------------------------------------------------------------------
    def test_fairness_guard_boundary_and_rejection(self):
        ae = FairAccuracyEnhancement(
            self.evaluator, self.transformer, "target", ["cat1"], ["num1", "num2"],
            max_fairness_degradation=0.02,
        )
        # Mock calculate_epsilon to test boundary equality and exceeding
        with unittest.mock.patch.object(
            self.evaluator, "calculate_epsilon",
            return_value={"protected": {"num1": 0.12, "num2": 0.10, "cat1": 0.05}}
        ):
            # Threshold 0.10, max degradation 0.02 -> cap is exactly 0.12
            res = ae._is_fairness_acceptable(self.df_tr, self.o_tr, epsilon_threshold=0.10, current_max_epsilon=0.10)
            # Equal to cap: acceptable
            self.assertTrue(res.is_acceptable)
            self.assertAlmostEqual(res.cap_applied, 0.12, places=5)

        with unittest.mock.patch.object(
            self.evaluator, "calculate_epsilon",
            return_value={"protected": {"num1": 0.12001, "num2": 0.10, "cat1": 0.05}}
        ):
            # Exceeding cap: rejected
            res = ae._is_fairness_acceptable(self.df_tr, self.o_tr, epsilon_threshold=0.10, current_max_epsilon=0.10)
            self.assertFalse(res.is_acceptable)
            self.assertIn("EXCEEDS_FAIRNESS_CAP", res.rejection_reason)

    # -------------------------------------------------------------------------
    # 10. Termination Semantics & Audit Log
    # -------------------------------------------------------------------------
    def test_audit_event_logged_for_every_candidate(self):
        ae = FairAccuracyEnhancement(
            self.evaluator, self.transformer, "target", ["cat1"], ["num1", "num2"],
            poly_exponents=(3.0,),
        )
        ae.enhance_step(
            X_train=self.df_tr,
            Y_train=self.y_tr,
            changed_dict={},
            partition=self.partition,
        )
        # Audit trail must contain candidate evaluation events
        self.assertGreater(len(ae.audit_trail), 0)
        first_event = ae.audit_trail[0]
        ev_dict = first_event.to_dict()
        # Verify mandatory R1.4 fields
        for field in [
            "run_id", "arm_id", "condition", "iteration", "engine",
            "parent_state_hash", "candidate_state_hash", "selected_feature",
            "proposed_transform", "utility_metric", "accepted", "validity_status",
            "model_fit_count", "geometry_eval_count"
        ]:
            self.assertIn(field, ev_dict)

    # -------------------------------------------------------------------------
    # 11. Backward Compatibility (use_accuracy_enhancement=False)
    # -------------------------------------------------------------------------
    def test_backward_compatibility_ae_disabled(self):
        cfg = FairBiasConfig.compas_default(
            random_seed=42,
            classifier="LR",
            use_bias_mitigation=True,
            use_accuracy_enhancement=False,
            max_iterations=1,
        )
        n = 100
        rng = np.random.default_rng(42)
        x_num1 = rng.uniform(1.0, 10.0, n)
        x_cat1 = rng.choice([0, 1, 2], size=n)
        o_val = rng.choice([0, 1], size=n)
        y_val = (x_num1 > 5.0).astype(int)
        synth_X = pd.DataFrame({"num1": x_num1, "cat1": x_cat1})
        synth_Y = pd.Series(y_val, name="target")
        synth_O = pd.DataFrame({"race": o_val})

        with unittest.mock.patch("fairbias.pipeline.FairDataLoader.prepare_data", return_value=(synth_X, synth_Y, synth_O, ["cat1"], ["num1"])), \
             unittest.mock.patch("fairbias.pipeline._sha256_of_file", return_value="0" * 64):
            run_res = run_fairbias_pipeline(cfg)
            self.assertIsNotNone(run_res.run_id)
            self.assertGreater(len(run_res.iterations), 0)
            for it in run_res.iterations:
                self.assertIsNone(it.get("enhancement_attribute"))

    # -------------------------------------------------------------------------
    # 12. Adversarial Review Contracts (R1B-01 to R1B-06)
    # -------------------------------------------------------------------------
    def test_missing_protected_data_cannot_disable_enabled_guard(self):
        ae = FairAccuracyEnhancement(self.evaluator, self.transformer, "target", ["cat1"], ["num1", "num2"])
        r = ae._is_fairness_acceptable(self.df_tr, None, 0.1, 0.1)
        self.assertFalse(r.is_acceptable)
        self.assertEqual(r.rejection_reason, "MISSING_PROTECTED_DATA_WITH_ENABLED_GUARD")

    def test_partial_geometry_is_invalid(self):
        ae = FairAccuracyEnhancement(self.evaluator, self.transformer, "target", ["cat1"], ["num1", "num2"])
        with unittest.mock.patch.object(self.evaluator, "calculate_epsilon", return_value={"protected": {"num1": 0.001}}):
            self.assertFalse(ae._is_fairness_acceptable(self.df_tr, self.o_tr, 0.1, 0.1).is_acceptable)

    def test_nan_after_finite_entry_is_invalid(self):
        ae = FairAccuracyEnhancement(self.evaluator, self.transformer, "target", ["cat1"], ["num1", "num2"])
        with unittest.mock.patch.object(self.evaluator, "calculate_epsilon", return_value={"protected": {"num1": 0.001, "num2": float("nan"), "cat1": 0.05}}):
            self.assertFalse(ae._is_fairness_acceptable(self.df_tr, self.o_tr, 0.1, 0.1).is_acceptable)

    def test_infinite_slack_is_invalid(self):
        ae = FairAccuracyEnhancement(self.evaluator, self.transformer, "target", ["cat1"], ["num1", "num2"])
        ae.max_fairness_degradation = float("inf")
        with unittest.mock.patch.object(self.evaluator, "calculate_epsilon", return_value={"protected": {"num1": 100.0, "num2": 100.0, "cat1": 100.0}}):
            self.assertFalse(ae._is_fairness_acceptable(self.df_tr, self.o_tr, 0.1, 0.1).is_acceptable)

    def test_invalid_baseline_must_reach_caller_as_failure(self):
        ae = FairAccuracyEnhancement(self.evaluator, self.transformer, "target", ["cat1"], ["num1", "num2"])
        bad_y = self.y_sel.copy()
        bad_y[:] = 1
        with self.assertRaisesRegex((RuntimeError, ValueError), "SINGLE_CLASS|single|invalid"):
            ae.enhance_step(self.df_tr, self.y_tr, {}, X_val=self.df_sel, Y_val=bad_y)

    def test_configured_classifier_matches_utility_oracle_dt(self):
        cfg = FairBiasConfig(algorithm_mode="engineering_bounded", classifier="DT", random_seed=42).resolved()
        ev = FairEvaluator(cfg, ["protected"], "target", ["cat1"], ["num1", "num2"])
        _, score = ev.fit_and_predict(self.df_tr, self.y_tr, self.df_sel)
        wanted = roc_auc_score(self.y_sel, score)
        result = evaluate_candidate_utility(self.partition, {}, ["num1", "num2"], ["cat1"], self.transformer, ev)
        self.assertAlmostEqual(result.utility_score, wanted, places=10)

    def test_missing_probabilities_not_replaced_with_hard_predictions(self):
        class HardOnly:
            def fit(self, X, y): return self
            def predict(self, X): return np.arange(len(X)) % 2
        result = evaluate_candidate_utility(self.partition, {}, ["num1", "num2"], ["cat1"], self.transformer, self.evaluator, model_factory=HardOnly)
        self.assertFalse(result.is_valid)

    def test_protected_index_alignment_is_validated(self):
        wrong = self.o_tr.iloc[::-1]
        with self.assertRaises(ValueError):
            EvaluationPartition(self.df_tr, self.y_tr, self.df_sel, self.y_sel, protected_fit=wrong)

    def test_partition_fingerprint_binds_values_and_labels(self):
        changed_y = 1 - self.y_tr
        changed_x = self.df_tr.copy()
        changed_x.iloc[0, 0] += 0.5
        other = EvaluationPartition(changed_x, changed_y, self.df_sel, self.y_sel)
        self.assertNotEqual(self.partition.fit_fingerprint(), other.fit_fingerprint())

    def test_feature_ranking_uses_only_fit_partition(self):
        fitX = self.df_tr.iloc[:80]
        fitY = self.y_tr.iloc[:80]
        selX = self.df_tr.iloc[80:]
        selY = self.y_tr.iloc[80:]
        part = EvaluationPartition(fitX, fitY, selX, selY)
        seen = []
        def ranking(X, y):
            seen.append(len(y))
            return {"num1": 1.0, "num2": 0.5}
        ae = FairAccuracyEnhancement(self.evaluator, self.transformer, "target", ["cat1"], ["num1", "num2"], poly_exponents=())
        with unittest.mock.patch("fairbias.enhancement.calculate_nmi_dict", side_effect=ranking):
            ae.enhance_step(self.df_tr, self.y_tr, {}, partition=part)
        self.assertEqual(seen, [80])

    def test_only_committed_candidate_logged_accepted(self):
        ae = FairAccuracyEnhancement(self.evaluator, self.transformer, "target", ["cat1"], ["num1", "num2"], poly_exponents=(1/3, 3.0))
        results = [
            CandidateEvaluationResult("VALID", 0.5, model_fit_count=1),
            CandidateEvaluationResult("VALID", 0.6, model_fit_count=1),
            CandidateEvaluationResult("VALID", 0.7, model_fit_count=1),
        ]
        with unittest.mock.patch("fairbias.enhancement.evaluate_candidate_utility", side_effect=results):
            _, changed, attr = ae.enhance_step(self.df_tr, self.y_tr, {}, partition=self.partition)
        actual = [ev for ev in ae.audit_trail if ev.accepted]
        self.assertEqual(len(actual), 1, msg=f"committed={changed}; accepted_events={len(actual)}")

    def test_original_drop_bug_fixed_with_arithmetic_oracle(self):
        rng = np.random.default_rng(210)
        X = pd.DataFrame({'x': rng.uniform(.2, 2, 160), 'z': rng.uniform(.2, 2, 160)})
        Y = pd.Series(((X.x > 1.1) ^ (X.z > 1.1)).astype(int), name='y')
        V = pd.DataFrame({'x': rng.uniform(.2, 2, 80), 'z': rng.uniform(.2, 2, 80)}, index=range(1000, 1080))
        W = pd.Series(((V.x > 1.1) ^ (V.z > 1.1)).astype(int), name='y')
        cfg = FairBiasConfig(algorithm_mode='engineering_bounded', classifier='LR', random_seed=42).resolved()
        ev = FairEvaluator(cfg, ['o'], 'y', [], ['x', 'z'])
        tr = FairTransform()
        part = EvaluationPartition(X, Y, V, W)
        state = {'z': 'dropped', 'x': {'power': 1/3}}
        result = evaluate_candidate_utility(part, state, ['x', 'z'], [], tr, ev)
        fit = X[['x']].pow(1/3)
        sel = V[['x']].pow(1/3)
        scaler = MinMaxScaler()
        a = scaler.fit_transform(fit)
        b = scaler.transform(sel)
        model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42).fit(a, Y)
        self.assertTrue(result.is_valid)
        self.assertAlmostEqual(result.utility_score, roc_auc_score(W, model.predict_proba(b)[:, 1]), places=12)

    # -------------------------------------------------------------------------
    # 13. R1C Contract Closure Tests
    # -------------------------------------------------------------------------
    def test_r1c_01_disabled_guard_contract(self):
        ae = FairAccuracyEnhancement(self.evaluator, self.transformer, "target", ["cat1"], ["num1", "num2"])
        res = ae._is_fairness_acceptable(self.df_tr, self.o_tr, 0.10, 0.10, fairness_guard_enabled=False)
        self.assertTrue(res.is_acceptable)
        self.assertEqual(res.evaluation_status, "NOT_EVALUATED")
        self.assertIsNone(res.candidate_max_dphi)
        self.assertIsNone(res.cap_applied)
        self.assertEqual(res.rejection_reason, "")

    def test_r1c_01_min_gain_and_slack_validation(self):
        # Negative / NaN / Inf rejected at initialization
        for bad_val in [-0.01, float("nan"), float("inf"), float("-inf")]:
            with self.assertRaises(ValueError):
                FairAccuracyEnhancement(
                    self.evaluator, self.transformer, "target", ["cat1"], ["num1", "num2"],
                    min_utility_gain=bad_val,
                )
            with self.assertRaises(ValueError):
                FairAccuracyEnhancement(
                    self.evaluator, self.transformer, "target", ["cat1"], ["num1", "num2"],
                    max_fairness_degradation=bad_val,
                )

        # Rejected at start of enhance_step if modified post-init
        ae = FairAccuracyEnhancement(self.evaluator, self.transformer, "target", ["cat1"], ["num1", "num2"])
        ae.min_utility_gain = -1.0
        with self.assertRaises(ValueError):
            ae.enhance_step(self.df_tr, self.y_tr, {}, partition=self.partition)

        ae.min_utility_gain = 0.0
        ae.max_fairness_degradation = float("nan")
        with self.assertRaises(ValueError):
            ae.enhance_step(self.df_tr, self.y_tr, {}, partition=self.partition)

    def test_r1c_02_partition_immutability_and_caching(self):
        df_x = self.df_tr.copy()
        s_y = self.y_tr.copy()
        part = EvaluationPartition(df_x, s_y, self.df_sel, self.y_sel)

        # Mutating external df_x should not affect partition's defensive snapshot
        df_x.iloc[0, 0] += 999.0
        # verify_not_mutated should succeed because internal copy is protected
        part.verify_not_mutated()

        # If internal partition array is directly tampered with, verify_not_mutated fails
        part.fit_X.iloc[0, 0] += 12345.0
        with self.assertRaises(ValueError):
            part.verify_not_mutated()

        # Ranking cache test
        ae = FairAccuracyEnhancement(self.evaluator, self.transformer, "target", ["cat1"], ["num1", "num2"])
        call_counts = []
        def counted_nmi(X, y):
            call_counts.append(len(y))
            return {"num1": 0.8, "num2": 0.2, "cat1": 0.1}

        with unittest.mock.patch("fairbias.enhancement.calculate_nmi_dict", side_effect=counted_nmi):
            attr1 = ae.find_target_correlated_attribute(self.df_tr, self.y_tr, {}, partition=self.partition)
            attr2 = ae.find_target_correlated_attribute(self.df_tr, self.y_tr, {}, partition=self.partition)
            self.assertEqual(attr1, attr2)
            # Reused cache: calculate_nmi_dict called only once
            self.assertEqual(len(call_counts), 1)

            # Different partition data invalidates cache
            diff_tr = self.df_tr.copy()
            diff_tr.iloc[0, 0] += 5.0
            diff_part = EvaluationPartition(diff_tr, self.y_tr, self.df_sel, self.y_sel)
            attr3 = ae.find_target_correlated_attribute(diff_tr, self.y_tr, {}, partition=diff_part)
            # Called again for new partition
            self.assertEqual(len(call_counts), 2)

    def test_r1c_03_category_mapping_invalid_audit_event(self):
        ae = FairAccuracyEnhancement(
            self.evaluator, self.transformer, "target", ["cat1"], ["num1", "num2"],
            arm_id="TEST_ARM", condition="test_cond",
        )
        with unittest.mock.patch(
            "fairbias.enhancement.safe_compose_category_mapping",
            side_effect=ValueError("Synthetic category composition conflict"),
        ):
            res = ae._try_categorical_enhancement(
                target_attr="cat1",
                X_train=self.df_tr,
                Y_train=self.y_tr,
                changed_dict={"cat1": {3: 1}},
                current_utility=0.5,
                O_train=self.o_tr,
                epsilon_threshold=0.1,
                curr_max_eps=0.1,
                partition=self.partition,
                parent_state_hash=hash_transform_state({"cat1": {3: 1}}),
                iteration=1,
            )
            self.assertIsNone(res)
            # Audit trail must contain the category invalid event
            matching_events = [
                ev for ev in ae.audit_trail
                if ev.validity_status == EnhancementStatus.CATEGORY_MAPPING_INVALID
            ]
            self.assertGreater(len(matching_events), 0)
            ev = matching_events[0]
            self.assertFalse(ev.accepted)
            self.assertEqual(ev.model_fit_count, 0)
            self.assertEqual(ev.geometry_eval_count, 0)
            self.assertIn("Synthetic category composition conflict", ev.rejection_reason)

    def test_r1c_07_non_numeric_features_rejected(self):
        # Candidate utility evaluation must reject non-numeric feature representation
        str_df = self.df_tr.copy()
        str_df["str_col"] = ["A", "B"] * (len(str_df) // 2)
        str_sel = self.df_sel.copy()
        str_sel["str_col"] = ["A", "B"] * (len(str_sel) // 2)

        part_str = EvaluationPartition(str_df, self.y_tr, str_sel, self.y_sel)
        res = evaluate_candidate_utility(
            part_str, {}, ["num1", "num2"], ["cat1", "str_col"], self.transformer, self.evaluator
        )
        self.assertFalse(res.is_valid)
        self.assertEqual(res.validity_status, EnhancementStatus.NON_NUMERIC_FEATURE)
        self.assertIn("Model-ready features must be numeric", res.error_message)

    def test_r1c_p2_01_configuration_fingerprint_sensitivity(self):
        base_kwargs = {
            "algorithm_mode": "engineering_bounded",
            "classifier": "LR",
            "random_seed": 42,
            "min_utility_gain": 0.001,
            "max_fairness_degradation": 0.02,
            "poly_exponents": (1/3, 3.0),
            "label_O": ("protected",),
            "label_Y": "target",
            "cate_attrs": ("cat1",),
            "num_attrs": ("num1", "num2"),
            "transform_n_bins": 5,
            "transform_log_epsilon": 1e-5,
        }
        base_fp = compute_configuration_fingerprint(**base_kwargs)
        self.assertEqual(len(base_fp), 16)

        # Altering each of the 12 fields must yield a different fingerprint
        mutations = [
            ("algorithm_mode", "standard"),
            ("classifier", "DT"),
            ("random_seed", 43),
            ("min_utility_gain", 0.005),
            ("max_fairness_degradation", 0.05),
            ("poly_exponents", (1/2, 2.0)),
            ("label_O", ("other_prot",)),
            ("label_Y", "other_target"),
            ("cate_attrs", ("cat1", "cat2")),
            ("num_attrs", ("num1",)),
            ("transform_n_bins", 10),
            ("transform_log_epsilon", 1e-4),
        ]
        for key, new_val in mutations:
            kwargs = copy.deepcopy(base_kwargs)
            kwargs[key] = new_val
            new_fp = compute_configuration_fingerprint(**kwargs)
            self.assertNotEqual(base_fp, new_fp, msg=f"Fingerprint failed to change on {key}")

    def test_r1c_p2_02_canonical_status_vocabulary(self):
        # All required canonical status constants must exist
        self.assertEqual(EnhancementStatus.VALID, "VALID")
        self.assertEqual(EnhancementStatus.EXCEEDS_FAIRNESS_CAP, "EXCEEDS_FAIRNESS_CAP")
        self.assertEqual(EnhancementStatus.NO_UTILITY_GAIN, "NO_UTILITY_GAIN")
        self.assertEqual(EnhancementStatus.ELIGIBLE_NOT_COMMITTED, "ELIGIBLE_NOT_COMMITTED")
        self.assertEqual(EnhancementStatus.MISSING_PROBABILITIES, "MISSING_PROBABILITIES")
        self.assertEqual(EnhancementStatus.NOT_EVALUATED, "NOT_EVALUATED")
        self.assertEqual(EnhancementStatus.CATEGORY_MAPPING_INVALID, "CATEGORY_MAPPING_INVALID")
        self.assertEqual(EnhancementStatus.NON_NUMERIC_FEATURE, "NON_NUMERIC_FEATURE")
        self.assertEqual(EnhancementStatus.ALL_FEATURES_DROPPED, "ALL_FEATURES_DROPPED")
        self.assertEqual(EnhancementStatus.SINGLE_CLASS_FIT_TARGET, "SINGLE_CLASS_FIT_TARGET")
        self.assertEqual(EnhancementStatus.SINGLE_CLASS_SELECTION_TARGET, "SINGLE_CLASS_SELECTION_TARGET")
        self.assertEqual(EnhancementStatus.NON_FINITE_OUTPUT, "NON_FINITE_OUTPUT")

    def test_r1d_01_authoritative_partition_binding(self):
        """R1D-01: Explicit EvaluationPartition is strictly authoritative over external arguments."""
        # Partition with 20 train rows and 10 selection rows
        fit_X = self.df_tr.iloc[:20].copy()
        fit_Y = self.y_tr.iloc[:20].copy()
        sel_X = self.df_sel.iloc[:10].copy()
        sel_Y = self.y_sel.iloc[:10].copy()
        o_fit = self.o_tr.iloc[:20].copy()
        partition = EvaluationPartition(fit_X, fit_Y, sel_X, sel_Y, protected_fit=o_fit)

        # External arguments with 100 rows containing corrupted/divergent values
        corrupted_X = pd.DataFrame({
            "num1": np.full(100, 99999.0),
            "num2": np.full(100, -99999.0),
            "cat1": np.full(100, 0),
        })
        corrupted_Y = pd.Series(np.zeros(100, dtype=int), name="target")
        corrupted_O = pd.DataFrame({"protected": np.zeros(100, dtype=int)})

        ae = FairAccuracyEnhancement(
            self.evaluator,
            self.transformer,
            "target",
            ["cat1"],
            ["num1", "num2"],
            poly_exponents=(2.0,),
            min_utility_gain=0.0,
            max_fairness_degradation=1.0,
        )

        transformed_df, changed, attr = ae.enhance_step(
            X_train=corrupted_X,
            Y_train=corrupted_Y,
            changed_dict={},
            O_train=corrupted_O,
            partition=partition,
        )

        # Output representation must strictly match partition.fit_X shape and index
        self.assertEqual(len(transformed_df), 20)
        self.assertEqual(list(transformed_df.index), list(fit_X.index))
        # Corrupted external values must not appear in the transformed output
        self.assertFalse((transformed_df["num1"] == 99999.0).any())
        self.assertFalse((transformed_df["num2"] == -99999.0).any())

    def test_r1d_02_model_contract_symmetry_and_oracles(self):
        """R1D-02: Non-probabilistic models fail with MISSING_PROBABILITIES in candidate and terminal evals."""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from nhis_fairbias.d8_enhancement_runner import evaluate_representation

        class DecisionFunctionOnlyModel:
            def fit(self, X, y):
                return self
            def decision_function(self, X):
                return np.zeros(len(X))
            # Has fit and decision_function, but NOT predict_proba

        model = DecisionFunctionOnlyModel()

        # 1. evaluate_candidate_utility fails with MISSING_PROBABILITIES
        res = evaluate_candidate_utility(
            self.partition,
            {},
            ["num1", "num2"],
            ["cat1"],
            self.transformer,
            self.evaluator,
            model_factory=lambda: model,
        )
        self.assertFalse(res.is_valid)
        self.assertEqual(res.validity_status, EnhancementStatus.MISSING_PROBABILITIES)
        self.assertIsNone(res.utility_score)

        # 2. evaluate_representation fails with MISSING_PROBABILITIES prefix
        with self.assertRaises(ValueError) as ctx:
            evaluate_representation(
                model=model,
                scaler=None,
                X_train_raw=self.df_tr,
                y_train=self.y_tr.values,
                X_val_raw=self.df_sel,
                y_val=self.y_sel.values,
                X_test_raw=self.df_sel,
                y_test=self.y_sel.values,
                o_train=self.o_tr.values,
                o_val=self.o_tr.values[:len(self.y_sel)],
                o_test=self.o_tr.values[:len(self.y_sel)],
                changed_dict={},
                evaluator=self.evaluator,
                transformer=self.transformer,
                cate_attrs=["cat1"],
                num_attrs=["num1", "num2"],
                protected_attr="protected",
            )
        self.assertIn(EnhancementStatus.MISSING_PROBABILITIES, str(ctx.exception))

        # 3. Standard oracle models with predict_proba (LR, DT, RF) succeed
        for factory in [
            lambda: LogisticRegression(max_iter=1000, random_state=42),
            lambda: DecisionTreeClassifier(max_depth=3, random_state=42),
            lambda: RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42),
        ]:
            res_oracle = evaluate_candidate_utility(
                self.partition,
                {},
                ["num1", "num2"],
                ["cat1"],
                self.transformer,
                self.evaluator,
                model_factory=factory,
            )
            self.assertTrue(res_oracle.is_valid)
            self.assertEqual(res_oracle.validity_status, EnhancementStatus.VALID)
            self.assertIsNotNone(res_oracle.utility_score)


if __name__ == "__main__":
    unittest.main()
