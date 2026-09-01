"""Unit and integration tests for Gate D1 NHIS feature engineering and leakage guards."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

_SRC_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from nhis_fairbias.features import (
    DEFAULT_FEATURE_CONFIG,
    NHISFeatureError,
    VALID_SEMANTIC_TYPES,
    build_leakage_guard_audit_table,
    harmonize_features_year,
    load_feature_registry,
    recode_predictor_series,
)
from nhis_fairbias.schema import DEFAULT_STUDY_CONFIG, exact_primary_frequency, load_study_config


class TestNHISFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = pathlib.Path(__file__).resolve().parents[1]
        cls.study_config = load_study_config(DEFAULT_STUDY_CONFIG)
        cls.feature_registry = load_feature_registry(DEFAULT_FEATURE_CONFIG)
        cls.features_parquet_path = (
            cls.repo_root / cls.study_config["outputs"]["features_parquet"]
        )

    def test_a_every_primary_feature_has_explicit_semantic_type(self) -> None:
        """A. Every primary feature has an explicit valid semantic type."""
        primary_core = self.feature_registry["primary_core"]
        self.assertGreater(len(primary_core), 0)
        for name, spec in primary_core.items():
            sem_type = spec.get("semantic_type")
            self.assertIn(
                sem_type,
                VALID_SEMANTIC_TYPES,
                f"Feature {name} has invalid semantic_type: {sem_type}",
            )

    def test_b_no_automatic_dtype_based_feature_typing(self) -> None:
        """B. No automatic dtype-based feature typing: integer columns are categorized."""
        # For example, EDUCP_A is stored as integer in CSV, but must be explicitly ordinal, NOT continuous
        educ_spec = self.feature_registry["primary_core"]["EDUCP_A"]
        self.assertEqual(educ_spec["semantic_type"], "ordinal")

        # REGION is stored as integer, but must be categorical_nominal
        region_spec = self.feature_registry["primary_core"]["REGION"]
        self.assertEqual(region_spec["semantic_type"], "categorical_nominal")

        # HYPEV_A is stored as integer, but must be categorical_binary
        hypev_spec = self.feature_registry["primary_core"]["HYPEV_A"]
        self.assertEqual(hypev_spec["semantic_type"], "categorical_binary")

        # Check that categorical feature lists do not simply include all integer columns
        cat_features = self.feature_registry["feature_lists"]["primary_categorical_features"]
        num_features = self.feature_registry["feature_lists"]["primary_numerical_features"]
        self.assertIn("region", cat_features)
        self.assertNotIn("region", num_features)
        self.assertIn("agep_a", num_features)

    def test_c_all_forbidden_features_excluded(self) -> None:
        """C. All forbidden outcome-proximal features are strictly excluded."""
        forbidden_dict = self.feature_registry["forbidden_outcome_proximal"]
        primary_list = self.feature_registry["feature_lists"]["primary_core_features"]
        expanded_list = self.feature_registry["feature_lists"]["expanded_utilization_features"]

        for var_name in forbidden_dict:
            self.assertNotIn(var_name, primary_list)
            self.assertNotIn(var_name.lower(), primary_list)
            self.assertNotIn(var_name, expanded_list)
            self.assertNotIn(var_name.lower(), expanded_list)

    def test_d_protected_variables_excluded_from_primary_x(self) -> None:
        """D. Protected variables (SEX_A, HISPALLP_A, DISAB3_A) are excluded from primary X."""
        primary_list = self.feature_registry["feature_lists"]["primary_core_features"]
        for prot in ("SEX_A", "HISPALLP_A", "DISAB3_A", "sex_a", "hispallp_a", "disab3_a"):
            self.assertNotIn(prot, primary_list)

    def test_e_outcome_variables_excluded_from_x(self) -> None:
        """E. Outcome variables (MEDDL12M_A, MEDNG12M_A) are excluded from predictor matrices."""
        primary_list = self.feature_registry["feature_lists"]["primary_core_features"]
        expanded_list = self.feature_registry["feature_lists"]["expanded_utilization_features"]
        for out in ("MEDDL12M_A", "MEDNG12M_A", "meddl12m", "meddly12m", "medng12m"):
            self.assertNotIn(out, primary_list)
            self.assertNotIn(out, expanded_list)

    def test_f_raw_columns_preserved(self) -> None:
        """F. Raw source columns are preserved alongside derived columns."""
        if self.features_parquet_path.is_file():
            df = pd.read_parquet(self.features_parquet_path)
            # Check preservation of outcomes
            self.assertIn("MEDDL12M_A_raw", df.columns)
            self.assertIn("MEDDL12M_A", df.columns)
            self.assertIn("meddl12m", df.columns)
            self.assertIn("meddly12m", df.columns)

            # Check preservation of protected
            self.assertIn("SEX_A", df.columns)
            self.assertIn("sex_a", df.columns)
            self.assertIn("HISPALLP_A", df.columns)
            self.assertIn("hispallp_a", df.columns)
            self.assertIn("DISAB3_A", df.columns)
            self.assertIn("disab3_a", df.columns)

            # Check preservation of primary predictors
            for raw_col in ("AGEP_A", "EDUCP_A", "REGION", "RATCAT_A", "PHSTAT_A", "HYPEV_A"):
                self.assertIn(raw_col, df.columns)
                harm_col = self.feature_registry["primary_core"][raw_col]["harmonized_name"]
                self.assertIn(harm_col, df.columns)

    def test_g_invalid_code_to_na_behavior_on_synthetic_examples(self) -> None:
        """G. Invalid codes (7/8/9/97/99) are mapped to NA on synthetic examples."""
        # Example 1: PHSTAT_A valid 1..5, missing 7, 8, 9
        s = pd.Series([1, 2, 5, 7, 8, 9, 3], name="PHSTAT_A")
        rec = recode_predictor_series(
            s,
            substantive_codes=[1, 2, 3, 4, 5],
            missing_codes=[7, 8, 9],
            semantic_type="ordinal",
        )
        self.assertEqual(rec.tolist(), [1, 2, 5, pd.NA, pd.NA, pd.NA, 3])

        # Example 2: Unexpected code outside substantive+missing raises NHISFeatureError
        bad_s = pd.Series([1, 2, 99], name="PHSTAT_A")
        with self.assertRaises(NHISFeatureError):
            recode_predictor_series(
                bad_s,
                substantive_codes=[1, 2, 3, 4, 5],
                missing_codes=[7, 8, 9],
                semantic_type="ordinal",
            )

    def test_h_cross_year_required_feature_presence(self) -> None:
        """H. Every PRIMARY_CORE feature is verified present across 2022, 2023, 2024."""
        for yr in (2022, 2023, 2024):
            csv_path = self.repo_root / f"data/raw/nhis/{yr}/adult{str(yr)[2:]}.csv"
            header = pd.read_csv(csv_path, nrows=0).columns
            for raw_name in self.feature_registry["primary_core"]:
                self.assertIn(
                    raw_name,
                    header,
                    f"PRIMARY_CORE feature {raw_name} is missing in NHIS {yr} header!",
                )

    def test_i_deterministic_feature_list_order(self) -> None:
        """I. Feature lists have deterministic fixed ordering."""
        fl = self.feature_registry["feature_lists"]
        primary = fl["primary_core_features"]
        self.assertEqual(len(primary), 21)
        self.assertEqual(primary[0], "agep_a")
        self.assertEqual(primary[1], "educp_a")
        self.assertEqual(primary[2], "region")
        self.assertEqual(primary[-1], "usualpl_a")

    def test_j_hispallp_a_code_7_official_semantic_label(self) -> None:
        """J. HISPALLP_A code 7 has the exact official label: 'Other single and multiple races'."""
        prot_hisp = self.feature_registry["protected_attributes"]["HISPALLP_A"]
        self.assertEqual(
            prot_hisp["code_labels"]["7"],
            "Other single and multiple races",
            "HISPALLP_A code 7 must be labeled 'Other single and multiple races'.",
        )

    def test_k_gate_d0_exact_meddl12m_counts_remain_unchanged(self) -> None:
        """K. Gate D0 exact MEDDL12M_A counts across 2022-2024 remain unchanged."""
        if self.features_parquet_path.is_file():
            df = pd.read_parquet(self.features_parquet_path)
            for year, expected in (
                (2022, {1: 1770, 2: 25683, 7: 3, 8: 192, 9: 3}),
                (2023, {1: 1931, 2: 27352, 7: 6, 8: 222, 9: 11}),
                (2024, {1: 2564, 2: 29791, 7: 10, 8: 255, 9: 9}),
            ):
                sub = df[df["survey_year"] == year]
                counts = sub["MEDDL12M_A_raw"].value_counts().to_dict()
                for code, exp_cnt in expected.items():
                    self.assertEqual(
                        counts.get(code, 0),
                        exp_cnt,
                        f"MEDDL12M_A count mismatch for code {code} in {year}",
                    )

    def test_l_2024_role_remains_frozen_test(self) -> None:
        """L. 2024 study role remains strictly 'frozen_test'."""
        if self.features_parquet_path.is_file():
            df = pd.read_parquet(self.features_parquet_path)
            sub_2024 = df[df["survey_year"] == 2024]
            roles = set(sub_2024["study_role"].unique())
            self.assertEqual(roles, {"frozen_test"})
            self.assertTrue(sub_2024["WTFA_DEV"].isna().all())

    def test_row_counts_total_and_per_year(self) -> None:
        """Verify row counts match D0 contract: 27,651 (2022), 29,522 (2023), 32,629 (2024), 89,802 total."""
        if self.features_parquet_path.is_file():
            df = pd.read_parquet(self.features_parquet_path)
            self.assertEqual(len(df), 89802)
            self.assertEqual(len(df[df["survey_year"] == 2022]), 27651)
            self.assertEqual(len(df[df["survey_year"] == 2023]), 29522)
            self.assertEqual(len(df[df["survey_year"] == 2024]), 32629)


if __name__ == "__main__":
    unittest.main()
