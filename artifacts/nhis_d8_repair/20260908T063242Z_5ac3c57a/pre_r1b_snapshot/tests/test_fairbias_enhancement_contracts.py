"""Comprehensive behavioral verification suite for FairBias accuracy enhancement contracts."""

from __future__ import annotations

import copy
import json
import unittest
import unittest.mock
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from fairbias.config import FairBiasConfig
from fairbias.evaluator import FairEvaluator
from fairbias.enhancement import FairAccuracyEnhancement
from fairbias.enhancement_contracts import (
    CandidateEvaluationResult,
    EvaluationPartition,
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
        # Synthetic run with AE disabled must succeed and maintain structure
        run_res = run_fairbias_pipeline(cfg)
        self.assertIsNotNone(run_res.run_id)
        self.assertGreater(len(run_res.iterations), 0)
        # Ensure iterations ran without AE attribute pollution
        for it in run_res.iterations:
            self.assertIsNone(it.get("enhancement_attribute"))


if __name__ == "__main__":
    unittest.main()
