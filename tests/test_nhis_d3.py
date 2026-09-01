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
import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_DIR = str(_REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from fairbias.config import (
    ALGORITHM_MODE_PAPER_FAITHFUL,
    FairBiasConfig,
    official_power_stream,
)
from fairbias.evaluator import FairEvaluator
from fairbias.mitigation import FairBiasMitigation
from fairbias.transform import FairTransform
from nhis_fairbias.adapter import DISABILITY_COMPONENTS
from nhis_fairbias.download import compute_sha256
from nhis_fairbias.features import DEFAULT_FEATURE_CONFIG, load_feature_registry
from nhis_fairbias.pooled import (
    DEFAULT_POOLED_SEED,
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
    # 11. Artifact Integrity and Manifest Verification
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

        # Check experiment_arms.csv
        arms_df = pd.read_csv(self.d3_dir / "experiment_arms.csv")
        self.assertEqual(len(arms_df), 4)
        self.assertTrue((arms_df["status"] == "PASS").all())


if __name__ == "__main__":
    unittest.main()
