"""Unit and integration tests for NHIS Gate D2 frozen temporal interface and preprocessing."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_DIR = str(_REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from nhis_fairbias.adapter import DISABILITY_COMPONENTS, NHISStudyAdapter
from nhis_fairbias.features import DEFAULT_FEATURE_CONFIG, load_feature_registry
from nhis_fairbias.preprocessing import (
    SENTINEL_CATEGORICAL_MISSING,
    SENTINEL_EXPLICIT_MISSING,
    SENTINEL_STRUCTURAL_NOT_IN_UNIVERSE,
    NHISLeakageError,
    NHISPreprocessingError,
    NHISPreprocessor,
    construct_empwrkft_series,
)


class TestNHISGateD2(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = _REPO_ROOT
        cls.adapter = NHISStudyAdapter()
        cls.registry = load_feature_registry(DEFAULT_FEATURE_CONFIG)

    # ------------------------------------------------------------------
    # 1. Temporal Contract & Partition Integrity
    # ------------------------------------------------------------------

    def test_temporal_row_counts_and_roles(self) -> None:
        """Explicit partition row counts and study roles: 2022=27651, 2023=29522, 2024=32629."""
        df_2022 = self.adapter.get_raw_partition(2022)
        df_2023 = self.adapter.get_raw_partition(2023)
        df_2024 = self.adapter.get_raw_partition(2024)

        self.assertEqual(len(df_2022), 27651)
        self.assertEqual(len(df_2023), 29522)
        self.assertEqual(len(df_2024), 32629)

        self.assertEqual(set(df_2022["study_role"].unique()), {"development_train"})
        self.assertEqual(set(df_2023["study_role"].unique()), {"development_validation"})
        self.assertEqual(set(df_2024["study_role"].unique()), {"frozen_test"})

    def test_random_splitting_forbidden(self) -> None:
        """Random splitting (e.g. 64/16/20) is strictly rejected."""
        with self.assertRaises(ValueError):
            self.adapter.random_split()

    def test_leakage_guard_rejects_fitting_on_2023_or_2024(self) -> None:
        """Attempting to fit preprocessing on 2023 or 2024 raises NHISLeakageError."""
        pre = NHISPreprocessor(feature_registry=self.registry)
        df_2023 = self.adapter.get_raw_partition(2023)
        df_2024 = self.adapter.get_raw_partition(2024)

        with self.assertRaises(NHISLeakageError):
            pre.fit(df_2023)

        with self.assertRaises(NHISLeakageError):
            pre.fit(df_2024)

    def test_leakage_guard_rejects_epsilon_on_2023_or_2024(self) -> None:
        """Attempting to compute epsilon threshold on 2023 or 2024 raises NHISLeakageError."""
        with self.assertRaises(NHISLeakageError):
            self.adapter.compute_epsilon(year=2023)

        with self.assertRaises(NHISLeakageError):
            self.adapter.compute_epsilon(year=2024)

    # ------------------------------------------------------------------
    # 2. Predictor Semantics & Families
    # ------------------------------------------------------------------

    def test_predictor_families_primary_and_expanded(self) -> None:
        """PRIMARY_CORE has exactly 21 features (18 cat, 3 num); EXPANDED has 24 (20 cat, 4 num)."""
        p_cats, p_nums = self.adapter.preprocessor.get_feature_family_lists("primary_core")
        self.assertEqual(len(p_cats), 18)
        self.assertEqual(len(p_nums), 3)
        self.assertEqual(len(p_cats) + len(p_nums), 21)

        e_cats, e_nums = self.adapter.preprocessor.get_feature_family_lists("expanded")
        self.assertEqual(len(e_cats), 20)
        self.assertEqual(len(e_nums), 4)
        self.assertEqual(len(e_cats) + len(e_nums), 24)

        # Every feature belongs to exactly one family
        self.assertEqual(len(set(p_cats).intersection(set(p_nums))), 0)
        self.assertEqual(len(set(e_cats).intersection(set(e_nums))), 0)

    # ------------------------------------------------------------------
    # 3. Outcomes: Never Imputed, Substantive Retained
    # ------------------------------------------------------------------

    def test_outcomes_never_imputed_and_substantive_counts(self) -> None:
        """Outcomes are never imputed; substantive counts match D0 audit."""
        # Check MEDDL12M_A
        X_22, y_22, _, _, _ = self.adapter.get_cohort(2022, outcome="MEDDL12M_A")
        X_23, y_23, _, _, _ = self.adapter.get_cohort(2023, outcome="MEDDL12M_A")
        X_24, y_24, _, _, _ = self.adapter.get_cohort(2024, outcome="MEDDL12M_A")

        # In SEX_A analysis (missing SEX_A: 2022=3, 2023=6, 2024=5)
        # Verify y contains no NAs and only {0, 1}
        for y_ser in (y_22, y_23, y_24):
            self.assertTrue(y_ser.notna().all())
            self.assertTrue(set(y_ser.unique()).issubset({0, 1}))

        # Total substantive records for MEDDL12M_A across years
        raw_all = self.adapter._raw_df
        valid_meddl = raw_all["meddl12m"].dropna()
        self.assertEqual(len(valid_meddl), 89091)
        self.assertEqual(int(((raw_all["survey_year"] == 2022) & raw_all["meddl12m"].notna()).sum()), 27453)
        self.assertEqual(int(((raw_all["survey_year"] == 2023) & raw_all["meddl12m"].notna()).sum()), 29283)
        self.assertEqual(int(((raw_all["survey_year"] == 2024) & raw_all["meddl12m"].notna()).sum()), 32355)

        # Check MEDNG12M_A
        valid_medng = raw_all["medng12m"].dropna()
        self.assertEqual(len(valid_medng), 89074)
        self.assertEqual(int(((raw_all["survey_year"] == 2022) & raw_all["medng12m"].notna()).sum()), 27444)
        self.assertEqual(int(((raw_all["survey_year"] == 2023) & raw_all["medng12m"].notna()).sum()), 29276)
        self.assertEqual(int(((raw_all["survey_year"] == 2024) & raw_all["medng12m"].notna()).sum()), 32354)

    # ------------------------------------------------------------------
    # 4. Protected Attributes: Never Imputed, Multicategorical Preserved
    # ------------------------------------------------------------------

    def test_protected_attributes_never_imputed_and_hispallp_multicategorical(self) -> None:
        """HISPALLP_A remains 7-class multicategorical (never binarized), never imputed."""
        _, _, o_hisp, _, _ = self.adapter.get_cohort(2022, protected_attribute="HISPALLP_A")
        self.assertTrue(o_hisp.notna().all())
        # All 7 substantive categories present
        self.assertEqual(sorted(o_hisp.unique()), [1, 2, 3, 4, 5, 6, 7])

        # SEX_A has {1, 2}
        _, _, o_sex, _, _ = self.adapter.get_cohort(2022, protected_attribute="SEX_A")
        self.assertTrue(o_sex.notna().all())
        self.assertEqual(sorted(o_sex.unique()), [1, 2])

        # DISAB3_A has {1, 2}
        _, _, o_disab, _, _ = self.adapter.get_cohort(2022, protected_attribute="DISAB3_A")
        self.assertTrue(o_disab.notna().all())
        self.assertEqual(sorted(o_disab.unique()), [1, 2])

    # ------------------------------------------------------------------
    # 5. EMPWRKFT1_A Structural Missingness
    # ------------------------------------------------------------------

    def test_empwrkft_structural_missingness_synthetic(self) -> None:
        """Verify synthetic combinations of EMPWRKLSW1_A and EMPWRKFT1_A."""
        # 1. Full-time: lsw=1, ft=1 -> 1
        # 2. Part-time: lsw=1, ft=2 -> 2
        # 3. In-universe nonresponse: lsw=1, ft in {7,8,9,NA} -> -2
        # 4. Not working last week: lsw=2 -> -1
        # 5. LSW nonresponse: lsw in {7,8,9,NA} -> -2
        lsw = pd.Series([1, 1, 1, 1, 1, 1, 2, 2, 7, 8, 9, pd.NA])
        ft = pd.Series([1, 2, 7, 8, 9, pd.NA, pd.NA, 1, 1, 2, pd.NA, pd.NA])
        expected = [1, 2, -2, -2, -2, -2, -1, -1, -2, -2, -2, -2]

        res = construct_empwrkft_series(lsw, ft)
        self.assertEqual(res.tolist(), expected)

    def test_empwrkft_structural_missingness_real_distribution(self) -> None:
        """Verify real NHIS distribution of constructed employment states."""
        pre = self.adapter.preprocessor
        rec = pre.fitted_record
        dist = rec.empwrkft_distribution

        self.assertEqual(dist["full-time"], 12381)
        self.assertEqual(dist["part-time"], 2957)
        self.assertEqual(dist["structural_not_in_universe"], 11104)
        self.assertEqual(dist["explicit_missing"], 1209)
        self.assertEqual(sum(dist.values()), 27651)

    # ------------------------------------------------------------------
    # 6. Other Missing Predictors
    # ------------------------------------------------------------------

    def test_numerical_median_fitted_on_2022_frozen_applied(self) -> None:
        """Numerical predictors use 2022-train-fitted median; applied frozen to 2023/2024."""
        pre = self.adapter.preprocessor
        med_2022_age = pre.fitted_record.numerical_medians["agep_a"]
        self.assertEqual(med_2022_age, 54.0)

        # Raw median in 2023 is 55.0, but preprocessed 2023 missing values must use 54.0
        df_2023 = self.adapter.get_raw_partition(2023)
        missing_idx = df_2023.index[df_2023["agep_a"].isna()]
        self.assertGreater(len(missing_idx), 0)

        transformed_2023 = pre.transform(df_2023, feature_set="primary_core")
        for idx in missing_idx:
            self.assertEqual(transformed_2023.loc[idx, "agep_a"], 54.0)

    def test_categorical_missing_sentinel(self) -> None:
        """Categorical missing values receive explicit sentinel -1."""
        pre = self.adapter.preprocessor
        df_2022 = self.adapter.get_raw_partition(2022)
        # EDUCP_A has 149 missing in 2022
        missing_idx = df_2022.index[df_2022["educp_a"].isna()]
        self.assertEqual(len(missing_idx), 149)

        transformed_2022 = pre.transform(df_2022, feature_set="primary_core")
        for idx in missing_idx:
            self.assertEqual(transformed_2022.loc[idx, "educp_a"], SENTINEL_CATEGORICAL_MISSING)

    # ------------------------------------------------------------------
    # 7. Disability Sensitivity Arms
    # ------------------------------------------------------------------

    def test_disability_sensitivity_arms(self) -> None:
        """Disability component exclusion: PRIMARY (21 -> 15), EXPANDED (24 -> 18)."""
        self.assertEqual(len(DISABILITY_COMPONENTS), 6)

        p_full = self.adapter.get_feature_names("primary_core", "full_feature")
        p_excl = self.adapter.get_feature_names("primary_core", "exclude_disability_components")
        self.assertEqual(len(p_full), 21)
        self.assertEqual(len(p_excl), 15)
        for comp in DISABILITY_COMPONENTS:
            self.assertIn(comp, p_full)
            self.assertNotIn(comp, p_excl)

        e_full = self.adapter.get_feature_names("expanded", "full_feature")
        e_excl = self.adapter.get_feature_names("expanded", "exclude_disability_components")
        self.assertEqual(len(e_full), 24)
        self.assertEqual(len(e_excl), 18)
        for comp in DISABILITY_COMPONENTS:
            self.assertIn(comp, e_full)
            self.assertNotIn(comp, e_excl)

    # ------------------------------------------------------------------
    # 8. Epsilon Determination on 2022 Development Train
    # ------------------------------------------------------------------

    def test_epsilon_unweighted_and_weighted_2022(self) -> None:
        """Compute unweighted and survey-weighted epsilon on 2022 development_train."""
        res_unw = self.adapter.compute_epsilon(
            year=2022, outcome="MEDDL12M_A", protected_attribute="SEX_A", weighted=False
        )
        self.assertEqual(res_unw["year"], 2022)
        self.assertEqual(res_unw["feature_count"], 21)
        self.assertGreater(res_unw["epsilon_threshold"], 0.0)

        res_w = self.adapter.compute_epsilon(
            year=2022, outcome="MEDDL12M_A", protected_attribute="SEX_A", weighted=True
        )
        self.assertEqual(res_w["year"], 2022)
        self.assertEqual(res_w["feature_count"], 21)
        self.assertGreater(res_w["epsilon_threshold"], 0.0)

    # ------------------------------------------------------------------
    # 9. Gate D2 Artifact Verification
    # ------------------------------------------------------------------

    def test_all_10_artifacts_exist_and_pass(self) -> None:
        """Verify all 10 required Gate D2 artifacts exist and pass all audits."""
        import json
        from nhis_fairbias.download import compute_sha256

        d2_dir = self.repo_root / "artifacts" / "nhis" / "d2"
        self.assertTrue(d2_dir.is_dir(), "artifacts/nhis/d2 directory missing")

        required_artifacts = (
            "d2_manifest.json",
            "temporal_partition_audit.csv",
            "preprocessing_contract.json",
            "preprocessing_fit_2022.json",
            "feature_transform_roles.csv",
            "missingness_strategy.csv",
            "analysis_arm_manifest.csv",
            "disability_sensitivity_manifest.csv",
            "survey_weight_contract.json",
            "weighted_fairbias_equivalence_audit.json",
        )
        for fname in required_artifacts:
            fpath = d2_dir / fname
            self.assertTrue(fpath.is_file(), f"Missing required D2 artifact: {fname}")

        # Check d2_manifest.json
        manifest_path = d2_dir / "d2_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["status"], "PASS")
        self.assertEqual(manifest["gate"], "D2")

        # Verify output artifact hashes
        for art_name, meta in manifest["output_artifacts"].items():
            actual_sha = compute_sha256(self.repo_root / meta["path"])
            self.assertEqual(actual_sha, meta["sha256"], f"SHA256 mismatch for {art_name}")

        # Check temporal_partition_audit.csv
        temporal_df = pd.read_csv(d2_dir / "temporal_partition_audit.csv")
        self.assertEqual(len(temporal_df), 3)
        self.assertTrue((temporal_df["status"] == "PASS").all())

        # Check weighted_fairbias_equivalence_audit.json
        with (d2_dir / "weighted_fairbias_equivalence_audit.json").open("r", encoding="utf-8") as f:
            audit = json.load(f)
        self.assertEqual(audit["status"], "PASS")
        invars = audit["weight_invariants_verification"]
        self.assertEqual(invars["constant_weights_status"], "PASS")
        self.assertEqual(invars["scale_invariance_status"], "PASS")
        for check, res in invars["fail_closed_checks"].items():
            self.assertEqual(res, "PASS", f"Fail closed check {check} failed")

        # Check analysis_arm_manifest.csv
        arm_df = pd.read_csv(d2_dir / "analysis_arm_manifest.csv")
        self.assertEqual(len(arm_df), 32)
        self.assertTrue((arm_df["status"] == "PASS").all())

        # Check feature_transform_roles.csv
        roles_df = pd.read_csv(d2_dir / "feature_transform_roles.csv")
        self.assertEqual(len(roles_df), 24)

        # Check missingness_strategy.csv
        miss_df = pd.read_csv(d2_dir / "missingness_strategy.csv")
        self.assertEqual(len(miss_df), 24)
        self.assertTrue((miss_df["status"] == "PASS").all())


if __name__ == "__main__":
    unittest.main()

