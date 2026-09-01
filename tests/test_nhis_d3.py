"""Comprehensive unit and integration test suite for NHIS Gate D3.

Tests:
1. Pooled 64/16/20 split counts (57473, 14368, 17961), mutual exclusivity, exhaustiveness.
2. Fixed seed reproducibility.
3. Survey year independence (all 3 years present in all 3 partitions).
4. Train-only preprocessing fit (parameters derived strictly from train partition).
5. Validation/test leakage guards (raises NHISLeakageError).
6. Tang-original mode rejects/ignores survey weights.
7. Semantic typing preserved regardless of pandas dtypes.
8. Forbidden predictors and protected attributes never enter X.
9. Power transformation follows declared author sequence.
10. Categorical frequency merging follows author rule.
11. Geometry recomputation after accepted mitigation step.
12. All 9 artifacts in artifacts/nhis/d3/ exist and pass validation.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_DIR = str(_REPO_ROOT / "src")
_SCRIPTS_DIR = str(_REPO_ROOT / "scripts")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from prepare_nhis_d3 import (
    build_paper_fidelity_manifest,
    require_pooled_split_audit_pass,
)
from fairbias.config import (
    ALGORITHM_MODE_ENGINEERING,
    ALGORITHM_MODE_OFFICIAL,
    ALGORITHM_MODE_PAPER_FAITHFUL,
    FairBiasConfig,
    official_power_stream,
)
from fairbias.evaluator import FairEvaluator
from fairbias.mitigation import FairBiasMitigation
from fairbias.pipeline import run_fairbias_pipeline
from fairbias.transform import FairTransform
from nhis_fairbias.adapter import DISABILITY_COMPONENTS
from nhis_fairbias.download import compute_sha256
from nhis_fairbias.features import DEFAULT_FEATURE_CONFIG, load_feature_registry
from nhis_fairbias.pooled import (
    DEFAULT_POOLED_SEED,
    EXPECTED_D0_MEDDL12M_COUNTS,
    EXPECTED_TEST_ROWS,
    EXPECTED_TOTAL_ROWS,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_VAL_ROWS,
    NHISPooledAdapter,
    audit_pooled_splits,
    generate_pooled_splits,
)
from nhis_fairbias.preprocessing import (
    SENTINEL_CATEGORICAL_MISSING,
    SENTINEL_EXPLICIT_MISSING,
    SENTINEL_STRUCTURAL_NOT_IN_UNIVERSE,
    NHISLeakageError,
    NHISPreprocessor,
)


class TestNHISGateD3(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = _REPO_ROOT
        cls.d3_dir = cls.repo_root / "artifacts" / "nhis" / "d3"
        cls.adapter = NHISPooledAdapter()
        cls.registry = load_feature_registry(DEFAULT_FEATURE_CONFIG)

    # ------------------------------------------------------------------
    # 1. Pooled 64/16/20 Split Counts, Exclusivity, Exhaustiveness
    # ------------------------------------------------------------------

    def test_pooled_split_counts_and_fractions(self) -> None:
        """Split counts follow exactly 64/16/20: 57473 train, 14368 val, 17961 test."""
        manifest = self.adapter.split_manifest
        self.assertEqual(len(manifest), EXPECTED_TOTAL_ROWS)

        counts = manifest["split_role"].value_counts().to_dict()
        self.assertEqual(counts["train"], EXPECTED_TRAIN_ROWS)
        self.assertEqual(counts["val"], EXPECTED_VAL_ROWS)
        self.assertEqual(counts["test"], EXPECTED_TEST_ROWS)

        self.assertAlmostEqual(counts["train"] / EXPECTED_TOTAL_ROWS, 0.64, delta=0.001)
        self.assertAlmostEqual(counts["val"] / EXPECTED_TOTAL_ROWS, 0.16, delta=0.001)
        self.assertAlmostEqual(counts["test"] / EXPECTED_TOTAL_ROWS, 0.20, delta=0.001)

    def test_splits_mutually_exclusive_and_exhaustive(self) -> None:
        """Partitions are mutually exclusive and their union equals all 89,802 records."""
        manifest = self.adapter.split_manifest
        train_idx = set(manifest[manifest["split_role"] == "train"]["orig_row_idx"])
        val_idx = set(manifest[manifest["split_role"] == "val"]["orig_row_idx"])
        test_idx = set(manifest[manifest["split_role"] == "test"]["orig_row_idx"])

        # Mutually exclusive
        self.assertEqual(len(train_idx & val_idx), 0)
        self.assertEqual(len(train_idx & test_idx), 0)
        self.assertEqual(len(val_idx & test_idx), 0)

        # Exhaustive
        all_indices = train_idx | val_idx | test_idx
        self.assertEqual(len(all_indices), EXPECTED_TOTAL_ROWS)
        self.assertEqual(all_indices, set(range(EXPECTED_TOTAL_ROWS)))

    # ------------------------------------------------------------------
    # 2. Seed Reproducibility
    # ------------------------------------------------------------------

    def test_seed_reproducibility(self) -> None:
        """Splits generated with the fixed seed reproduce identical assignments."""
        df_raw = self.adapter._raw_df
        split1 = generate_pooled_splits(df_raw, seed=DEFAULT_POOLED_SEED)
        split2 = generate_pooled_splits(df_raw, seed=DEFAULT_POOLED_SEED)

        pd.testing.assert_series_equal(split1["split_role"], split2["split_role"])
        pd.testing.assert_series_equal(split1["record_id"], split2["record_id"])

    # ------------------------------------------------------------------
    # 3. Survey Year Independence
    # ------------------------------------------------------------------

    def test_survey_year_independence(self) -> None:
        """All three years (2022, 2023, 2024) appear in train, val, and test partitions."""
        audit_res = audit_pooled_splits(self.adapter.split_manifest)
        self.assertEqual(audit_res["status"], "PASS")
        self.assertTrue(audit_res["all_years_present_in_all_splits"])

        by_year = audit_res["year_breakdown_by_split"]
        for role in ("train", "val", "test"):
            self.assertEqual(set(by_year[role].keys()), {2022, 2023, 2024})
            for yr, count in by_year[role].items():
                self.assertGreater(count, 0)

    # ------------------------------------------------------------------
    # 4. Train-Only Preprocessing Fit
    # ------------------------------------------------------------------

    def test_train_only_preprocessing_fit(self) -> None:
        """Preprocessing statistics are fitted strictly on pooled train (57,473 rows)."""
        pre = self.adapter.preprocessor
        fit_rec = pre.fitted_record

        self.assertEqual(fit_rec.row_count, EXPECTED_TRAIN_ROWS)
        self.assertEqual(fit_rec.fit_study_role, "pooled_train")

        # Pooled train agep_a median is 55.0 (differs from 2022-only median of 54.0)
        self.assertEqual(fit_rec.numerical_medians["agep_a"], 55.0)

        # 4-state employment sum equals train row count
        self.assertEqual(sum(fit_rec.empwrkft_distribution.values()), EXPECTED_TRAIN_ROWS)

    # ------------------------------------------------------------------
    # 5. Validation/Test Leakage Guards
    # ------------------------------------------------------------------

    def test_leakage_guard_rejects_fit_on_val_or_test(self) -> None:
        """Attempting to fit preprocessing on validation or test raises NHISLeakageError."""
        pre = NHISPreprocessor(feature_registry=self.registry)
        val_df = self.adapter.get_partition("val")
        test_df = self.adapter.get_partition("test")

        with self.assertRaises(NHISLeakageError):
            pre.fit(val_df, regime="pooled")

        with self.assertRaises(NHISLeakageError):
            pre.fit(test_df, regime="pooled")

    # ------------------------------------------------------------------
    # 6. Tang-Original Mode Rejects Survey Weights
    # ------------------------------------------------------------------

    def test_tang_original_mode_rejects_survey_weights(self) -> None:
        """tang2024_paper_faithful mode strictly rejects sample_weight."""
        cfg = FairBiasConfig.compas_default(mode="paper")
        self.assertEqual(cfg.algorithm_mode, ALGORITHM_MODE_PAPER_FAITHFUL)

        evaluator = FairEvaluator(
            config=cfg,
            label_O=["SEX_A"],
            label_Y="MEDDL12M_A",
            cate_attrs=["educp_a"],
            num_attrs=["agep_a"],
        )

        dummy_X = pd.DataFrame({"agep_a": [30.0, 40.0], "educp_a": [1, 2]})
        dummy_O = pd.DataFrame({"SEX_A": [1, 2]})
        weights = pd.Series([1.5, 2.0])

        with self.assertRaises(ValueError):
            evaluator.calculate_epsilon(
                dummy_X, dummy_O, sample_weight=weights
            )

        with self.assertRaises(NHISLeakageError):
            self.adapter.compute_train_epsilon(
                config=cfg, sample_weight=weights
            )

    # ------------------------------------------------------------------
    # 7. Semantic Typing & Forbidden/Protected Exclusions
    # ------------------------------------------------------------------

    def test_semantic_feature_typing_and_forbidden_exclusions(self) -> None:
        """Categorical features retain categorical role regardless of dtype; no protected in X."""
        cohorts = self.adapter.get_pooled_cohort(
            outcome="MEDDL12M_A", protected_attribute="SEX_A", feature_set="primary_core"
        )
        X_train, y_train, o_train, w_train, meta = cohorts["train"]

        # Exactly 21 features in PRIMARY_CORE
        self.assertEqual(len(X_train.columns), 21)

        # Protected attributes and design variables never in X
        forbidden = {
            "SEX_A", "HISPALLP_A", "DISAB3_A",
            "sex_a", "hispallp_a", "disab3_a",
            "MEDDL12M_A", "MEDNG12M_A", "meddl12m", "medng12m",
            "WTFA_A", "PSTRAT", "PPSU", "survey_year", "split_role",
        }
        self.assertTrue(set(X_train.columns).isdisjoint(forbidden))

        # Outcome and protected attribute series are clean
        self.assertTrue(y_train.notna().all())
        self.assertTrue(o_train.notna().all())
        self.assertTrue(set(y_train.unique()).issubset({0, 1}))
        self.assertEqual(sorted(o_train.unique()), [1, 2])

    # ------------------------------------------------------------------
    # 8. Disability Sensitivity Arms in Pooled Regime
    # ------------------------------------------------------------------

    def test_disability_sensitivity_arms_pooled(self) -> None:
        """PRIMARY_CORE has 21 (full) vs 15 (excluded components); components strictly excluded."""
        full_names = self.adapter.get_feature_names("primary_core", "full_feature")
        excl_names = self.adapter.get_feature_names("primary_core", "exclude_disability_components")

        self.assertEqual(len(full_names), 21)
        self.assertEqual(len(excl_names), 15)

        for comp in DISABILITY_COMPONENTS:
            self.assertIn(comp, full_names)
            self.assertNotIn(comp, excl_names)

    # ------------------------------------------------------------------
    # 9. Power Search Sequence & Categorical Frequency Merging
    # ------------------------------------------------------------------

    def test_power_search_sequence(self) -> None:
        """Official power stream follows [3, 1/3, 5, 1/5, 7, 1/7, ...]."""
        stream = official_power_stream(up_to=7)
        expected = (3.0, 1.0 / 3.0, 5.0, 1.0 / 5.0, 7.0, 1.0 / 7.0)
        self.assertEqual(stream, expected)

    def test_categorical_frequency_merging_rule(self) -> None:
        """Categorical merging identifies pair with highest protected frequency disparity."""
        cfg = FairBiasConfig.compas_default(mode="paper")
        evaluator = FairEvaluator(config=cfg)
        transformer = FairTransform()
        mitigator = FairBiasMitigation(
            evaluator=evaluator,
            transformer=transformer,
            label_O=["prot"],
            cate_attrs=["cat"],
            num_attrs=[],
        )

        # Synthetic feature with 3 categories where category A is concentrated in group 1
        feature_ser = pd.Series(["A", "A", "A", "B", "C", "C"])
        prot_ser = pd.Series([1, 1, 1, 0, 0, 0])
        rebin = mitigator.compute_r1_rebin(feature_ser, prot_ser)

        self.assertIsNotNone(rebin)
        self.assertIsInstance(rebin, dict)
        # Rebin merges extreme categories
        self.assertEqual(len(rebin), 1)

    # ------------------------------------------------------------------
    # 10. Geometry Recomputation After Mitigation Step
    # ------------------------------------------------------------------

    def test_geometry_recomputation_trace(self) -> None:
        """Mitigation engine records structured step trace with d_phi before and after."""
        cfg = FairBiasConfig.compas_default(mode="paper")
        evaluator = FairEvaluator(config=cfg)
        transformer = FairTransform()
        mitigator = FairBiasMitigation(
            evaluator=evaluator,
            transformer=transformer,
            label_O=["prot"],
            cate_attrs=["cat"],
            num_attrs=["num"],
        )

        np.random.seed(42)
        syn_df = pd.DataFrame({
            "num": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "cat": ["A", "B", "A", "B", "A", "B"],
        })
        syn_y = pd.Series([0, 1, 0, 1, 0, 1])
        syn_o = pd.DataFrame({"prot": [0, 0, 0, 1, 1, 1]})

        current_eps = {"prot": {"num": 0.05, "cat": 0.02}}
        epsilon_thresh = 0.01

        candidate_df, temp_changed, sel_o, sel_attr = mitigator.mitigate_step(
            X=syn_df,
            Y=syn_y,
            O=syn_o,
            nmi_org={"num": 0.5, "cat": 0.5},
            changed_dict={},
            current_epsilon=current_eps,
            epsilon_threshold=epsilon_thresh,
            iteration=1,
        )

        # Trace record is appended
        self.assertGreater(len(mitigator.step_traces), 0)
        trace_step = mitigator.step_traces[0]
        self.assertEqual(trace_step.iteration, 1)
        self.assertEqual(trace_step.selected_feature, "num")
        self.assertEqual(trace_step.d_phi_before, 0.05)

    # ------------------------------------------------------------------
    # 11. Stratified Split Balance Across 9 Strata
    # ------------------------------------------------------------------

    def test_pooled_split_strata_balance(self) -> None:
        """Verify all 9 strata (survey_year x meddl12m_state) are present and balanced."""
        manifest = self.adapter.split_manifest
        self.assertIn("meddl12m_state", manifest.columns)

        audit_res = audit_pooled_splits(manifest)
        self.assertEqual(audit_res["status"], "PASS")
        self.assertTrue(audit_res["stratum_balance_valid"])

        stratum_audit = audit_res["stratum_audit"]
        expected_strata = {
            "2022__0", "2022__1", "2022__missing_or_non_substantive",
            "2023__0", "2023__1", "2023__missing_or_non_substantive",
            "2024__0", "2024__1", "2024__missing_or_non_substantive",
        }
        self.assertEqual(set(stratum_audit.keys()), expected_strata)

        for st_name, st_info in stratum_audit.items():
            tr_frac = st_info["train_fraction"]
            va_frac = st_info["val_fraction"]
            te_frac = st_info["test_fraction"]
            self.assertAlmostEqual(tr_frac, 0.64, delta=0.02, msg=f"Train balance failed for {st_name}")
            self.assertAlmostEqual(va_frac, 0.16, delta=0.02, msg=f"Val balance failed for {st_name}")
            self.assertAlmostEqual(te_frac, 0.20, delta=0.02, msg=f"Test balance failed for {st_name}")

    # ------------------------------------------------------------------
    # 12. Single Master Split Reused Across Protected Attributes
    # ------------------------------------------------------------------

    def test_same_master_split_reused_across_protected_attributes(self) -> None:
        """Master split assignment is immutable and reused across all protected attributes."""
        manifest = self.adapter.split_manifest
        self.assertEqual(len(manifest), EXPECTED_TOTAL_ROWS)

        c_sex = self.adapter.get_pooled_cohort(outcome="MEDDL12M_A", protected_attribute="SEX_A")
        c_hisp = self.adapter.get_pooled_cohort(outcome="MEDDL12M_A", protected_attribute="HISPALLP_A")
        c_disab = self.adapter.get_pooled_cohort(outcome="MEDDL12M_A", protected_attribute="DISAB3_A")

        # Every partition in each cohort must be a strict subset of the master partition assignment
        for role in ("train", "val", "test"):
            master_role_indices = set(manifest[manifest["split_role"] == role]["orig_row_idx"])
            self.assertTrue(set(c_sex[role][0].index).issubset(master_role_indices))
            self.assertTrue(set(c_hisp[role][0].index).issubset(master_role_indices))
            self.assertTrue(set(c_disab[role][0].index).issubset(master_role_indices))

        # Intersection of eligible rows across attributes shares identical split roles
        common_train = set(c_sex["train"][0].index) & set(c_hisp["train"][0].index) & set(c_disab["train"][0].index)
        self.assertGreater(len(common_train), 50000)

    # ------------------------------------------------------------------
    # 13. Paper Faithful Mode Routing, Predicates & Power Policies
    # ------------------------------------------------------------------

    def test_paper_faithful_config_and_policies(self) -> None:
        """tang2024_paper_faithful resolves to author stream, restart revisit, and paper predicates."""
        cfg = FairBiasConfig.compas_default(mode="paper")
        self.assertEqual(cfg.algorithm_mode, ALGORITHM_MODE_PAPER_FAITHFUL)
        self.assertTrue(cfg.is_paper_reference)
        self.assertFalse(cfg.is_official_code_variant)
        self.assertFalse(cfg.is_engineering)

        resolved = cfg.resolved()
        self.assertEqual(resolved.power_sequence_policy, "official_stream")
        self.assertEqual(resolved.power_revisit_policy, "restart")
        self.assertEqual(resolved.transform_poly_exponents, official_power_stream())
        self.assertIsNone(resolved.mds_fixed_components)

    def test_monotone_cursor_remains_behaviorally_separate(self) -> None:
        """official_code_derived mode retains monotone cursor and fixed dim=2."""
        cfg = FairBiasConfig.compas_default(mode="official")
        self.assertEqual(cfg.algorithm_mode, ALGORITHM_MODE_OFFICIAL)
        self.assertTrue(cfg.is_official_code_variant)
        self.assertFalse(cfg.is_paper_reference)
        self.assertFalse(cfg.is_engineering)

        resolved = cfg.resolved()
        self.assertEqual(resolved.power_sequence_policy, "official_stream")
        self.assertEqual(resolved.power_revisit_policy, "monotone_cursor")
        self.assertEqual(resolved.mds_fixed_components, 2)

    def test_engineering_mode_remains_unchanged(self) -> None:
        """engineering_bounded mode retains sorted_grid, restart, and engineering predicates."""
        cfg = FairBiasConfig.compas_default(mode="engineering")
        self.assertEqual(cfg.algorithm_mode, ALGORITHM_MODE_ENGINEERING)
        self.assertTrue(cfg.is_engineering)
        self.assertFalse(cfg.is_paper_reference)
        self.assertFalse(cfg.is_official_code_variant)

        resolved = cfg.resolved()
        self.assertEqual(resolved.power_sequence_policy, "sorted_grid")
        self.assertEqual(resolved.power_revisit_policy, "restart")

    # ------------------------------------------------------------------
    # 14. Paper Faithful Pipeline Execution & Manifest Labeling
    # ------------------------------------------------------------------

    def test_paper_faithful_no_iteration_budget_and_no_pareto(self) -> None:
        """Paper-faithful execution has no 5-iteration budget, no validation Pareto, and clean manifest."""
        with tempfile.TemporaryDirectory() as tmp_out:
            cfg = FairBiasConfig.compas_default(mode="paper", output_dir=tmp_out)
            res = run_fairbias_pipeline(cfg)

            # Paper mode ignores iteration_budget
            self.assertNotEqual(res.termination["termination_reason"], "iteration_budget_exhausted")

            # Validation Pareto is bypassed
            self.assertIsNone(res.pareto_engineering_metrics)
            self.assertIsNone(res.pareto_engineering_changed_dict)
            self.assertIn("N/A", res.best_selection_reason)

            # Greedy terminal metrics is the sole reported state
            self.assertIsNotNone(res.greedy_terminal_metrics)

            # Manifest labels itself accurately
            with open(res.output_file, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.assertEqual(payload["algorithm_mode"], "tang2024_paper_faithful")
            self.assertEqual(payload["final_states"], ["tang2024_paper_faithful"])
            self.assertIn("final_results_tang2024_paper_faithful", payload)
            self.assertNotIn("final_results_pareto_engineering", payload)
            self.assertNotIn("final_results_configured_greedy_terminal", payload)
            self.assertIn("tang2024_paper_faithful", payload["algorithm_mode_definition"])
            self.assertNotIn("engineering_bounded", payload["algorithm_mode_definition"])

    # ------------------------------------------------------------------
    # 15. Fail-Closed State Cycle Detection
    # ------------------------------------------------------------------

    def test_fail_closed_state_cycle_detection(self) -> None:
        """Fail-closed cycle detector stops when candidate reproduces visited state."""
        cfg = FairBiasConfig.compas_default(mode="paper")
        evaluator = FairEvaluator(config=cfg)
        transformer = FairTransform()
        mitigator = FairBiasMitigation(
            evaluator=evaluator,
            transformer=transformer,
            label_O=["prot"],
            cate_attrs=[],
            num_attrs=["num"],
            power_sequence_policy="official_stream",
            power_revisit_policy="restart",
        )
        mitigator.visited_states.add(json.dumps({"num": {"power": 3.0}}, sort_keys=True))

        syn_df = pd.DataFrame({"num": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        syn_y = pd.Series([0, 1, 0, 1, 0, 1])
        syn_o = pd.DataFrame({"prot": [0, 0, 0, 1, 1, 1]})

        current_eps = {"prot": {"num": 0.05}}
        epsilon_thresh = 0.01

        cand_transform = {"num": {"power": 3.0}}
        with patch.object(mitigator, "_search_numerical", return_value=(syn_df, cand_transform)):
            cand_df, temp_changed, sel_o, sel_attr = mitigator.mitigate_step(
                X=syn_df,
                Y=syn_y,
                O=syn_o,
                nmi_org={"num": 0.5},
                changed_dict={},
                current_epsilon=current_eps,
                epsilon_threshold=epsilon_thresh,
                iteration=1,
            )
        self.assertIsNotNone(mitigator.non_convergence)
        self.assertEqual(mitigator.non_convergence["search_scope"], "state_cycle_detected")
        self.assertIn("state cycle detected", mitigator.non_convergence["reason"])

    # ------------------------------------------------------------------
    # 16. Clean-Checkout Reproducibility
    # ------------------------------------------------------------------

    def test_clean_checkout_reproducibility(self) -> None:
        """Artifact generation runs cleanly from scratch in a fresh isolated directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd = [
                sys.executable,
                str(self.repo_root / "scripts" / "prepare_nhis_d3.py"),
                "--output-dir", tmp_dir,
                "--force",
            ]
            env = dict(os.environ, PYTHONPATH=str(self.repo_root / "src"))
            res = subprocess.run(cmd, capture_output=True, text=True, env=env)
            self.assertEqual(res.returncode, 0, f"prepare_nhis_d3 failed: {res.stderr}")

            required = [
                "d3_manifest.json",
                "pooled_split_manifest.csv",
                "pooled_split_audit.json",
                "paper_fidelity_manifest.json",
                "paper_algorithm_contract.json",
                "pooled_preprocessing_fit.json",
                "experiment_arms.csv",
                "fairbias_transform_trace_schema.json",
                "numerical_transform_audit_schema.json",
            ]
            tmp_path = pathlib.Path(tmp_dir)
            for r in required:
                self.assertTrue((tmp_path / r).is_file(), f"Missing {r} in clean checkout")

            with (tmp_path / "d3_manifest.json").open("r", encoding="utf-8") as f:
                d3_man = json.load(f)
            self.assertEqual(d3_man["status"], "PASS")

    # ------------------------------------------------------------------
    # 17. Artifact Integrity and Manifest Verification
    # ------------------------------------------------------------------

    def test_all_9_artifacts_exist_and_pass(self) -> None:
        """Verify all 9 required Gate D3 artifacts exist and pass all audits."""
        required_artifacts = (
            "d3_manifest.json",
            "pooled_split_manifest.csv",
            "pooled_split_audit.json",
            "paper_fidelity_manifest.json",
            "paper_algorithm_contract.json",
            "pooled_preprocessing_fit.json",
            "experiment_arms.csv",
            "fairbias_transform_trace_schema.json",
            "numerical_transform_audit_schema.json",
        )
        # If pre-existing artifacts are not present, generate them on the fly
        if not (self.d3_dir / "d3_manifest.json").is_file():
            cmd = [
                sys.executable,
                str(self.repo_root / "scripts" / "prepare_nhis_d3.py"),
                "--output-dir", str(self.d3_dir),
                "--force",
            ]
            env = dict(os.environ, PYTHONPATH=str(self.repo_root / "src"))
            subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)

        for fname in required_artifacts:
            fpath = self.d3_dir / fname
            self.assertTrue(fpath.is_file(), f"Missing required D3 artifact: {fname}")

        # Check d3_manifest.json
        manifest_path = self.d3_dir / "d3_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["status"], "PASS")
        self.assertEqual(manifest["gate"], "D3")
        self.assertEqual(manifest["algorithm_mode"], ALGORITHM_MODE_PAPER_FAITHFUL)

        # Verify output artifact hashes
        for art_name, meta in manifest["output_artifacts"].items():
            actual_sha = compute_sha256(self.repo_root / meta["path"])
            self.assertEqual(actual_sha, meta["sha256"], f"SHA256 mismatch for {art_name}")

        # Check pooled_split_audit.json
        with (self.d3_dir / "pooled_split_audit.json").open("r", encoding="utf-8") as f:
            split_audit = json.load(f)
        self.assertEqual(split_audit["status"], "PASS")
        self.assertTrue(split_audit["mutually_exclusive_and_exhaustive"])
        self.assertTrue(split_audit["has_exact_counts"])
        self.assertTrue(split_audit["all_years_present_in_all_splits"])
        self.assertTrue(split_audit["stratum_balance_valid"])
        self.assertTrue(split_audit["d0_outcome_totals_match"])

        # Check paper_fidelity_manifest.json
        with (self.d3_dir / "paper_fidelity_manifest.json").open("r", encoding="utf-8") as f:
            paper_fid = json.load(f)
        paper_cfg = FairBiasConfig(algorithm_mode=ALGORITHM_MODE_PAPER_FAITHFUL).resolved()
        mds_fid = paper_fid["fidelity_analysis"]["mds_embedding_dimensionality"]
        self.assertEqual(
            mds_fid["resolved_repo_parameters"]["mds_max_components"],
            paper_cfg.mds_max_components,
        )
        self.assertEqual(
            mds_fid["resolved_repo_parameters"]["mds_slope_threshold"],
            paper_cfg.mds_slope_threshold,
        )
        self.assertEqual(
            mds_fid["resolved_repo_parameters"]["mds_fixed_components"],
            paper_cfg.mds_fixed_components,
        )

        # Check experiment_arms.csv
        arms_df = pd.read_csv(self.d3_dir / "experiment_arms.csv")
        self.assertEqual(len(arms_df), 4)
        self.assertTrue((arms_df["status"] == "PASS").all())

    # ------------------------------------------------------------------
    # 18. Dynamic MDS Fidelity Manifest Parameters (Gate D3.2 Repair 1)
    # ------------------------------------------------------------------

    def test_mds_fidelity_manifest_derives_dynamically_from_resolved_config(self) -> None:
        """Fidelity manifest dynamically derives MDS operational parameters from resolved FairBiasConfig."""
        paper_cfg = FairBiasConfig(algorithm_mode=ALGORITHM_MODE_PAPER_FAITHFUL).resolved()
        manifest_data = build_paper_fidelity_manifest(paper_cfg)
        mds_entry = manifest_data["fidelity_analysis"]["mds_embedding_dimensionality"]

        # Machine-readable parameters must match the resolved config dynamically
        resolved_params = mds_entry["resolved_repo_parameters"]
        self.assertEqual(resolved_params["mds_max_components"], paper_cfg.mds_max_components)
        self.assertEqual(resolved_params["mds_slope_threshold"], paper_cfg.mds_slope_threshold)
        self.assertEqual(resolved_params["mds_fixed_components"], paper_cfg.mds_fixed_components)

        # Human-readable prose must be generated from the resolved config values, not stale hardcoded literals
        repo_op_text = mds_entry["repo_operationalization"]
        self.assertIn(f"resolved mds_max_components = {paper_cfg.mds_max_components}", repo_op_text)
        self.assertIn(f"resolved mds_slope_threshold = {paper_cfg.mds_slope_threshold}", repo_op_text)
        self.assertIn("resolved mds_fixed_components = None (stress-elbow dimension selection)", repo_op_text)

        # Regression check: ensure custom configuration overrides propagate automatically (no hardcoding)
        custom_cfg = FairBiasConfig(
            algorithm_mode=ALGORITHM_MODE_PAPER_FAITHFUL,
            mds_max_components=20,
            mds_slope_threshold=0.02,
        ).resolved()
        custom_manifest = build_paper_fidelity_manifest(custom_cfg)
        custom_mds = custom_manifest["fidelity_analysis"]["mds_embedding_dimensionality"]
        self.assertEqual(custom_mds["resolved_repo_parameters"]["mds_max_components"], 20)
        self.assertEqual(custom_mds["resolved_repo_parameters"]["mds_slope_threshold"], 0.02)
        self.assertIn("resolved mds_max_components = 20", custom_mds["repo_operationalization"])
        self.assertIn("resolved mds_slope_threshold = 0.02", custom_mds["repo_operationalization"])

        # P2: Check integer mds_fixed_components is described as explicit fixed-dimension override, not stress-elbow
        fixed_cfg = FairBiasConfig(
            algorithm_mode=ALGORITHM_MODE_PAPER_FAITHFUL,
            mds_fixed_components=4,
        ).resolved()
        fixed_manifest = build_paper_fidelity_manifest(fixed_cfg)
        fixed_mds = fixed_manifest["fidelity_analysis"]["mds_embedding_dimensionality"]
        self.assertEqual(fixed_mds["resolved_repo_parameters"]["mds_fixed_components"], 4)
        self.assertIn("resolved mds_fixed_components = 4", fixed_mds["repo_operationalization"])
        self.assertIn("explicit fixed-dimension override, stress-elbow search bypassed", fixed_mds["repo_operationalization"])
        self.assertNotIn("stress-elbow dimension selection", fixed_mds["repo_operationalization"])

        # P2: Explicit non-paper-faithful config raises ValueError
        eng_cfg = FairBiasConfig(algorithm_mode=ALGORITHM_MODE_ENGINEERING)
        with self.assertRaises(ValueError):
            build_paper_fidelity_manifest(eng_cfg)

    # ------------------------------------------------------------------
    # 19. Exact Gate D0 Outcome Totals Invariant (Gate D3.2 Repair 2 - Test A)
    # ------------------------------------------------------------------

    def test_pooled_split_d0_exact_outcome_totals_invariant(self) -> None:
        """Master split manifest observed totals for every survey_year x meddl12m_state equal frozen D0 counts."""
        manifest = self.adapter.split_manifest
        audit = audit_pooled_splits(manifest)

        # Audit flags
        self.assertTrue(audit["d0_outcome_totals_match"])
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(len(audit["d0_outcome_totals_mismatches"]), 0)

        # Explicitly verify all 9 cells across all 3 years and 3 outcome states
        observed = audit["d0_outcome_totals_observed"]
        expected = audit["d0_outcome_totals_expected"]

        for year in (2022, 2023, 2024):
            year_sum_observed = 0
            year_sum_expected = 0
            for state in ("1", "0", "missing_or_non_substantive"):
                exp_cnt = EXPECTED_D0_MEDDL12M_COUNTS[year][state]
                obs_audit = observed[year][state]
                self.assertEqual(
                    obs_audit,
                    exp_cnt,
                    f"D0 count mismatch in audit for year {year}, state {state}: {obs_audit} != {exp_cnt}",
                )
                self.assertEqual(
                    expected[year][state],
                    exp_cnt,
                    f"Expected count in audit mismatch for year {year}, state {state}",
                )

                # Also verify directly from manifest dataframe
                obs_direct = int(
                    ((manifest["survey_year"] == year) & (manifest["meddl12m_state"] == state)).sum()
                )
                self.assertEqual(
                    obs_direct,
                    exp_cnt,
                    f"Direct manifest count mismatch for year {year}, state {state}: {obs_direct} != {exp_cnt}",
                )

                year_sum_observed += obs_direct
                year_sum_expected += exp_cnt

            # Verify yearly exact totals
            if year == 2022:
                self.assertEqual(year_sum_observed, 1770 + 25683 + 198)
            elif year == 2023:
                self.assertEqual(year_sum_observed, 1931 + 27352 + 239)
            elif year == 2024:
                self.assertEqual(year_sum_observed, 2564 + 29791 + 274)
            self.assertEqual(year_sum_observed, year_sum_expected)

        # Global sum
        total_observed = sum(
            sum(observed[yr].values()) for yr in (2022, 2023, 2024)
        )
        self.assertEqual(total_observed, EXPECTED_TOTAL_ROWS)

    # ------------------------------------------------------------------
    # 20. Negative Mutation Test for D0 Invariant (Gate D3.2 Repair 2 - Test B)
    # ------------------------------------------------------------------

    def test_pooled_split_d0_invariant_negative_mutation(self) -> None:
        """Mutating a single row's meddl12m_state fails the D0 invariant and sets overall status to FAIL."""
        manifest_copy = self.adapter.split_manifest.copy(deep=True)

        # Pick a 2022 row with meddl12m_state == "0" and mutate it to "1"
        target_idx = manifest_copy[
            (manifest_copy["survey_year"] == 2022) & (manifest_copy["meddl12m_state"] == "0")
        ].index[0]
        manifest_copy.loc[target_idx, "meddl12m_state"] = "1"

        # Verify row counts and split allocations remain unchanged
        self.assertEqual(len(manifest_copy), EXPECTED_TOTAL_ROWS)
        counts = manifest_copy["split_role"].value_counts().to_dict()
        self.assertEqual(counts["train"], EXPECTED_TRAIN_ROWS)
        self.assertEqual(counts["val"], EXPECTED_VAL_ROWS)
        self.assertEqual(counts["test"], EXPECTED_TEST_ROWS)

        # Run audit on mutated manifest
        mutated_audit = audit_pooled_splits(manifest_copy)

        # Assert invariant failure and audit FAIL
        self.assertFalse(
            mutated_audit["d0_outcome_totals_match"],
            "d0_outcome_totals_match must be False under single-row state mutation",
        )
        self.assertEqual(
            mutated_audit["status"],
            "FAIL",
            "Overall audit status must be FAIL when D0 outcome totals invariant fails",
        )

        # Check machine-readable mismatch reporting
        mismatches = mutated_audit["d0_outcome_totals_mismatches"]
        self.assertTrue(len(mismatches) >= 2, f"Expected at least 2 mismatched cells, got {mismatches}")
        mismatched_keys = {(m["survey_year"], m["meddl12m_state"]) for m in mismatches if "survey_year" in m}
        self.assertIn((2022, "0"), mismatched_keys)
        self.assertIn((2022, "1"), mismatched_keys)

        # Ensure disk artifact was NOT mutated
        clean_manifest = pd.read_csv(self.d3_dir / "pooled_split_manifest.csv")
        clean_obs_2022_0 = int(
            ((clean_manifest["survey_year"] == 2022) & (clean_manifest["meddl12m_state"] == "0")).sum()
        )
        self.assertEqual(clean_obs_2022_0, 25683)

    # ------------------------------------------------------------------
    # 21. Fail-Closed Split Audit Guard in Gate Preparation (Gate D3.2.1 P1)
    # ------------------------------------------------------------------

    def test_require_pooled_split_audit_pass_guard_success_and_failure(self) -> None:
        """require_pooled_split_audit_pass allows PASS audit and raises RuntimeError on FAIL."""
        pass_audit = {
            "status": "PASS",
            "d0_outcome_totals_match": True,
            "has_exact_counts": True,
        }
        # Must return cleanly without raising
        require_pooled_split_audit_pass(pass_audit)

        # Failure with mismatch diagnostics must raise RuntimeError
        failed_audit = {
            "status": "FAIL",
            "d0_outcome_totals_match": False,
            "d0_outcome_totals_mismatches": [
                {
                    "survey_year": 2022,
                    "meddl12m_state": "0",
                    "expected": 25683,
                    "observed": 25682,
                    "difference": -1,
                },
                {
                    "survey_year": 2022,
                    "meddl12m_state": "1",
                    "expected": 1770,
                    "observed": 1771,
                    "difference": 1,
                },
            ],
        }
        with self.assertRaises(RuntimeError) as ctx:
            require_pooled_split_audit_pass(failed_audit)

        err_msg = str(ctx.exception)
        self.assertIn("status='FAIL'", err_msg)
        self.assertIn("D0 outcome total mismatches", err_msg)
        self.assertIn("Cannot proceed with Gate D3 artifact generation", err_msg)

        # Any non-PASS status (e.g. None or UNKNOWN) must fail closed
        with self.assertRaises(RuntimeError):
            require_pooled_split_audit_pass({"status": "UNKNOWN"})

    def test_prepare_nhis_d3_fails_closed_when_split_audit_fails(self) -> None:
        """Gate preparation production path terminates fail-closed without writing d3_manifest on audit failure."""
        import prepare_nhis_d3

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)

            failed_audit_payload = {
                "status": "FAIL",
                "d0_outcome_totals_match": False,
                "d0_outcome_totals_mismatches": [
                    {"survey_year": 2022, "meddl12m_state": "0", "expected": 25683, "observed": 25682, "difference": -1}
                ],
            }

            with patch("prepare_nhis_d3.audit_pooled_splits", return_value=failed_audit_payload):
                with self.assertRaises(RuntimeError) as ctx:
                    prepare_nhis_d3.main(["--output-dir", str(tmp_path), "--force"])

                self.assertIn("status='FAIL'", str(ctx.exception))

            # Verify audit json was written (forensic diagnostics preserved)
            self.assertTrue((tmp_path / "pooled_split_audit.json").is_file())
            with (tmp_path / "pooled_split_audit.json").open("r", encoding="utf-8") as f:
                saved_audit = json.load(f)
            self.assertEqual(saved_audit["status"], "FAIL")

            # Verify fail-fast: downstream artifacts and top-level PASS manifest MUST NOT exist
            self.assertFalse(
                (tmp_path / "d3_manifest.json").is_file(),
                "d3_manifest.json must NOT be generated when pooled split audit fails",
            )
            self.assertFalse(
                (tmp_path / "paper_fidelity_manifest.json").is_file(),
                "Downstream artifacts must not be generated when pooled split audit fails",
            )
            self.assertFalse(
                (tmp_path / "pooled_preprocessing_fit.json").is_file(),
                "Preprocessing fit must not execute when pooled split audit fails",
            )


if __name__ == "__main__":
    unittest.main()
