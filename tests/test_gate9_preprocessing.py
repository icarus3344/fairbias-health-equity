"""Tests for Gate 9: Train-Only Preprocessing, Missing-Value Contracts & Design Quality Checks."""

from __future__ import annotations

import pathlib
import sys
import unittest

_SRC_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import pandas as pd

from meps_fairness.data.cohort import (
    extract_meps_cohort,
    resolve_prior_round_inheritance,
)
from meps_fairness.data.preprocess import (
    MEPSPreprocessor,
    check_survey_design_quality,
)
from meps_fairness.data.split import split_panel26_duid_grouped


class TestGate9Preprocessing(unittest.TestCase):
    """Unit and integration tests for Gate 9 preprocessor and survey design checks."""

    def setUp(self) -> None:
        self.repo_root = pathlib.Path(__file__).resolve().parents[1]
        self.dta_path = self.repo_root / "data/interim/meps/h244/h244.dta"

    def test_train_only_fit_and_zero_test_leakage(self) -> None:
        """Verify preprocessor fits only on train and transforms test without leaking stats."""
        # Train data: Feature 1 (cont) has mean 10, std 2; Cat 1 has values [1, 2]
        train_df = pd.DataFrame({
            "AGEY1X": [20.0, 30.0, 40.0, 50.0],
            "POVCATY1": [1, 2, 1, 2],
        })

        # Test data has completely different scale and an unseen category (3)
        test_df = pd.DataFrame({
            "AGEY1X": [100.0, 200.0],
            "POVCATY1": [2, 3],  # 3 is unseen
        })

        preprocessor = MEPSPreprocessor(
            continuous_features=["AGEY1X"],
            categorical_features=["POVCATY1"],
        )
        train_trans = preprocessor.fit_transform(train_df)

        # Fitted mean and std should match train_df only
        stats = preprocessor.fitted_continuous_["AGEY1X"]
        self.assertAlmostEqual(stats.mean, 35.0)

        # Transform test data using frozen train stats
        test_trans = preprocessor.transform(test_df)

        # Check that test AGEY1X was normalized by train stats (35.0, std)
        expected_test_val_0 = (100.0 - stats.mean) / stats.std
        self.assertAlmostEqual(test_trans["AGEY1X"].iloc[0], expected_test_val_0)

        # Check unseen category 3 in test: POVCATY1__1 and POVCATY1__2 should be 0 for row 1
        self.assertEqual(test_trans["POVCATY1__1"].iloc[1], 0.0)
        self.assertEqual(test_trans["POVCATY1__2"].iloc[1], 0.0)

    def test_missing_code_imputation_and_indicator(self) -> None:
        """Verify negative codes in continuous features are imputed with train median and flagged."""
        train_df = pd.DataFrame({
            "TOTEXPY1": [100.0, 200.0, 300.0, -1.0],  # -1 is inapplicable
        })
        preprocessor = MEPSPreprocessor(
            continuous_features=["TOTEXPY1"],
            categorical_features=[],
        )
        train_trans = preprocessor.fit_transform(train_df)

        # Training median of valid [100, 200, 300] is 200.0
        stats = preprocessor.fitted_continuous_["TOTEXPY1"]
        self.assertEqual(stats.impute_value, 200.0)
        self.assertTrue(stats.has_missing)
        self.assertIn("TOTEXPY1__missing", train_trans.columns)
        self.assertEqual(train_trans["TOTEXPY1__missing"].iloc[3], 1.0)
        self.assertEqual(train_trans["TOTEXPY1__missing"].iloc[0], 0.0)

    def test_legitimate_negative_income_preservation(self) -> None:
        """Verify TTLPY1X, FAMINCY1, POVLEVY1 preserve legitimate negative values and only impute sentinel -1."""
        train_df = pd.DataFrame({
            "TTLPY1X": [-300.0, 10000.0, 30000.0, -1.0],  # -300 is valid loss, -1 is missing
            "POVLEVY1": [-2.13, 100.0, 200.0, -1.0],      # -2.13 is valid loss, -1 is missing
        })
        preprocessor = MEPSPreprocessor(
            continuous_features=["TTLPY1X", "POVLEVY1"],
            categorical_features=[],
        )
        train_trans = preprocessor.fit_transform(train_df)

        # Valid values for TTLPY1X are [-300.0, 10000.0, 30000.0]; median is 10000.0
        stats_ttl = preprocessor.fitted_continuous_["TTLPY1X"]
        self.assertEqual(stats_ttl.impute_value, 10000.0)
        self.assertTrue(stats_ttl.is_negative_allowed)
        # Row 0 (-300.0) must NOT be flagged as missing
        self.assertEqual(train_trans["TTLPY1X__missing"].iloc[0], 0.0)
        # Row 3 (-1.0) MUST be flagged as missing
        self.assertEqual(train_trans["TTLPY1X__missing"].iloc[3], 1.0)

        # Valid values for POVLEVY1 are [-2.13, 100.0, 200.0]; median is 100.0
        stats_pov = preprocessor.fitted_continuous_["POVLEVY1"]
        self.assertEqual(stats_pov.impute_value, 100.0)
        self.assertEqual(train_trans["POVLEVY1__missing"].iloc[0], 0.0)
        self.assertEqual(train_trans["POVLEVY1__missing"].iloc[3], 1.0)

    def test_structural_zero_employment_handling(self) -> None:
        """Verify -1 INAPPLICABLE in employment characteristics is recoded to structural zero (0.0)."""
        train_df = pd.DataFrame({
            "HOUR1": [40.0, 35.0, -1.0, -7.0],       # -1 is unemployed (0 hrs), -7 is Refused (missing)
            "NUMEMP1": [50.0, 100.0, -1.0, -8.0],    # -1 is unemployed (0 emp), -8 is DK (missing)
            "WAGEPY1X": [45000.0, 0.0, -1.0, -15.0], # -1 is inapplicable (0 wage), -15 is missing
        })
        preprocessor = MEPSPreprocessor(
            continuous_features=["HOUR1", "NUMEMP1", "WAGEPY1X"],
            categorical_features=[],
        )
        train_trans = preprocessor.fit_transform(train_df)

        stats_hr = preprocessor.fitted_continuous_["HOUR1"]
        self.assertTrue(stats_hr.is_structural_zero)
        # Row 2 (-1.0) is structural zero, so NOT missing
        self.assertEqual(train_trans["HOUR1__missing"].iloc[2], 0.0)
        # Row 3 (-7.0) is item non-response, so missing
        self.assertEqual(train_trans["HOUR1__missing"].iloc[3], 1.0)

        # Valid values for HOUR1 are [40.0, 35.0, 0.0]; median is 35.0
        self.assertEqual(stats_hr.impute_value, 35.0)

    def test_prior_round_inheritance(self) -> None:
        """Verify HOUR2, NUMEMP2, CHOIC2, SELFCM2, UNION2 inherit Round 1 values when coded -2."""
        raw_df = pd.DataFrame({
            "HOUR1": [40.0, 20.0],
            "HOUR2": [-2.0, 30.0],  # Row 0 should inherit 40.0
            "NUMEMP1": [100.0, 50.0],
            "NUMEMP2": [-2.0, 25.0],  # Row 0 should inherit 100.0
            "CHOIC1": [1, 2],
            "CHOIC2": [-2, 1],  # Row 0 should inherit 1
            "SELFCM1": [2, 1],
            "SELFCM2": [-2, 2],  # Row 0 should inherit 2
            "UNION1": [1, 2],
            "UNION2": [-2, 2],  # Row 0 should inherit 1
        })
        resolved = resolve_prior_round_inheritance(raw_df)
        self.assertEqual(resolved["HOUR2"].iloc[0], 40.0)
        self.assertEqual(resolved["NUMEMP2"].iloc[0], 100.0)
        self.assertEqual(resolved["CHOIC2"].iloc[0], 1)
        self.assertEqual(resolved["SELFCM2"].iloc[0], 2)
        self.assertEqual(resolved["UNION2"].iloc[0], 1)

        # Preprocessor should also handle inheritance transparently
        preprocessor = MEPSPreprocessor(
            continuous_features=["HOUR1", "HOUR2"],
            categorical_features=["CHOIC1", "CHOIC2"],
        )
        trans = preprocessor.fit_transform(raw_df)
        self.assertIn("HOUR2", trans.columns)
        self.assertIn("CHOIC2__1", trans.columns)
        self.assertEqual(trans["CHOIC2__1"].iloc[0], 1.0)

    def test_categorical_structural_inapplicable_preservation(self) -> None:
        """Verify categorical variables preserve -1 INAPPLICABLE as a category dummy and drop -7/-8/-15."""
        train_df = pd.DataFrame({
            "LOCATN2": [1, 2, -1, -8],  # 1=Office, 2=Hospital, -1=No USC, -8=DK (missing)
        })
        preprocessor = MEPSPreprocessor(
            continuous_features=[],
            categorical_features=["LOCATN2"],
        )
        train_trans = preprocessor.fit_transform(train_df)

        # -1 INAPPLICABLE dummy must exist
        self.assertIn("LOCATN2__-1", train_trans.columns)
        self.assertIn("LOCATN2__1", train_trans.columns)
        self.assertIn("LOCATN2__2", train_trans.columns)
        # -8 Don't Know dummy must NOT exist
        self.assertNotIn("LOCATN2__-8", train_trans.columns)

        # Row 2 (-1) should have LOCATN2__-1 == 1.0
        self.assertEqual(train_trans["LOCATN2__-1"].iloc[2], 1.0)
        # Row 3 (-8) should have all category dummies == 0.0
        self.assertEqual(train_trans["LOCATN2__-1"].iloc[3], 0.0)
        self.assertEqual(train_trans["LOCATN2__1"].iloc[3], 0.0)
        self.assertEqual(train_trans["LOCATN2__2"].iloc[3], 0.0)

    def test_survey_design_quality_checks_on_panel26(self) -> None:
        """Integration test: Verify survey design quality checks on real Panel 26 cohort."""
        if not self.dta_path.is_file():
            self.skipTest(f"Missing Panel 26 Stata file: {self.dta_path}")

        df = pd.read_stata(self.dta_path, convert_categoricals=False)
        cohort = extract_meps_cohort(df, panel_number=26, allow_target=True)
        split_result = split_panel26_duid_grouped(cohort, seed=20260828)

        quality_report = check_survey_design_quality(cohort.design)
        self.assertEqual(quality_report["status"], "PASSED")
        self.assertGreater(quality_report["kish_effective_n"], 500)
        self.assertGreater(quality_report["unique_strata_count"], 50)
        self.assertGreater(quality_report["unique_psu_clusters_count"], 100)

        # Also test on training partition
        train_quality = check_survey_design_quality(split_result.train.design)
        self.assertEqual(train_quality["status"], "PASSED")


if __name__ == "__main__":
    unittest.main()

