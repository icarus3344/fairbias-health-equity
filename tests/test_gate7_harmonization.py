"""Tests for Gate 7: Codebook Verification, Cohort Derivation, and Variable Harmonization."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest

_SRC_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import pandas as pd

from meps_fairness.data.cohort import (
    ALL_BASELINE_PREDICTOR_COLUMNS,
    ALL_CATEGORICAL_PREDICTORS,
    ALL_CONTINUOUS_PREDICTORS,
    AUDIT_COLUMNS,
    BASELINE_INSURANCE_MONTHS,
    DESIGN_COLUMNS,
    FOLLOWUP_INSURANCE_MONTHS,
    CohortData,
    derive_age_band,
    derive_composite_disability,
    extract_meps_cohort,
    is_prohibited_temporal_feature,
    resolve_prior_round_inheritance,
)


class TestGate7Harmonization(unittest.TestCase):
    """Unit and integration tests for Gate 7 harmonization and cohort derivation."""

    def setUp(self) -> None:
        self.repo_root = pathlib.Path(__file__).resolve().parents[1]
        self.config_path = self.repo_root / "configs/cohort_and_variables.json"
        self.snapshot_path = self.repo_root / "docs/data/MEPS_SCHEMA_SNAPSHOT.json"

    def test_cohort_and_variables_json_validity(self) -> None:
        """Verify configs/cohort_and_variables.json exists, is valid JSON, and has all expected top-level sections."""
        self.assertTrue(self.config_path.is_file(), f"Missing config: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("source_codebooks", data)
        self.assertIn("survey_design", data)
        self.assertIn("cohort_eligibility", data)
        self.assertIn("target_outcomes", data)
        self.assertIn("audit_dimensions", data)
        self.assertIn("baseline_predictors", data)
        self.assertIn("variable_contracts", data)
        self.assertIn("temporal_contract", data)

        # Verify exactly 74 baseline predictors (19 continuous, 55 categorical)
        cont_count = sum(len(g["continuous"]) for g in data["baseline_predictors"].values())
        cat_count = sum(len(g["categorical"]) for g in data["baseline_predictors"].values())
        self.assertEqual(cont_count, 19)
        self.assertEqual(cat_count, 55)
        self.assertEqual(cont_count + cat_count, 74)

    def test_all_candidate_variables_in_shared_schema(self) -> None:
        """Verify that all retained candidate variables exist in the shared schema of both panels."""
        self.assertTrue(self.snapshot_path.is_file(), f"Missing snapshot: {self.snapshot_path}")
        with open(self.snapshot_path, "r", encoding="utf-8") as f:
            snap = json.load(f)

        shared_columns = set(snap["cross_panel_comparison"]["shared_columns"])

        for col in DESIGN_COLUMNS:
            self.assertIn(col, shared_columns, f"Design column {col} missing from shared schema")

        for col in BASELINE_INSURANCE_MONTHS:
            self.assertIn(col, shared_columns, f"Baseline insurance column {col} missing from shared schema")

        for col in FOLLOWUP_INSURANCE_MONTHS:
            self.assertIn(col, shared_columns, f"Followup insurance column {col} missing from shared schema")

        for col in AUDIT_COLUMNS:
            self.assertIn(col, shared_columns, f"Audit column {col} missing from shared schema")

        for col in ALL_BASELINE_PREDICTOR_COLUMNS:
            self.assertIn(col, shared_columns, f"Predictor column {col} missing from shared schema")

    def test_synthetic_cohort_filtering_and_target_derivation(self) -> None:
        """Verify cohort filtering logic, target construction, and audit variables on synthetic data."""
        n_samples = 10
        data = {
            "DUPERSID": [f"ID{i}" for i in range(n_samples)],
            "DUID": [100 + i // 2 for i in range(n_samples)],
            "PID": [101 + i % 2 for i in range(n_samples)],
            "PANEL": [26] * n_samples,
            "YEARIND": [1] * n_samples,
            "ALL5RDS": [1] * n_samples,
            "LONGWT": [1000.0] * n_samples,
            "VARSTR": [2001] * n_samples,
            "VARPSU": [1] * n_samples,
            "AGEY1X": [20, 30, 40, 50, 60, 17, 65, 35, 45, 55],  # 17 (<18) and 65 (>=65) should be excluded
            "RACETHX": [1, 2, 3, 4, 5, 1, 2, 1, 2, 3],
            "SEX": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "POVCATY1": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "POVLEVY1": [50.0, 110.0, 150.0, 250.0, 450.0, 50.0, 110.0, 150.0, 250.0, 450.0],
            "EDUCYR": [12] * n_samples,
            "FAMSZEY1": [3] * n_samples,
            "HIDEG": [1] * n_samples,
            "MARRY1X": [1] * n_samples,
            "MARRY2X": [1] * n_samples,
            "MARRYY1X": [1] * n_samples,
            "REGION1": [1] * n_samples,
            "REGION2": [1] * n_samples,
            "REGIONY1": [1] * n_samples,
            "FAMINCY1": [50000.0] * n_samples,
            "TTLPY1X": [40000.0] * n_samples,
            "WAGEPY1X": [35000.0] * n_samples,
            "HOUR1": [40] * n_samples,
            "HOUR2": [40] * n_samples,
            "NUMEMP1": [100] * n_samples,
            "NUMEMP2": [100] * n_samples,
            "EMPST1": [1] * n_samples,
            "EMPST2": [1] * n_samples,
            "OFFER1X": [1] * n_samples,
            "OFFER2X": [1] * n_samples,
            "HELD1X": [1] * n_samples,
            "HELD2X": [1] * n_samples,
            "CHOIC1": [1] * n_samples,
            "CHOIC2": [1] * n_samples,
            "SELFCM1": [2] * n_samples,
            "SELFCM2": [2] * n_samples,
            "UNION1": [2] * n_samples,
            "UNION2": [2] * n_samples,
            "RTHLTH1": [1] * n_samples,
            "RTHLTH2": [1] * n_samples,
            "MNHLTH1": [1] * n_samples,
            "MNHLTH2": [1] * n_samples,
            "HIBPDXY1": [2] * n_samples,
            "DIABDXY1_M18": [2] * n_samples,
            "ASTHDXY1": [2] * n_samples,
            "CHDDXY1": [2] * n_samples,
            "ANGIDXY1": [2] * n_samples,
            "MIDXY1": [2] * n_samples,
            "OHRTDXY1": [2] * n_samples,
            "STRKDXY1": [2] * n_samples,
            "EMPHDXY1": [2] * n_samples,
            "CHBRON1": [2] * n_samples,
            "CHOLDXY1": [2] * n_samples,
            "CANCERY1": [2] * n_samples,
            "ARTHDXY1": [2] * n_samples,
            "JTPAIN1_M18": [2] * n_samples,
            "ACTLIM1": [2] * n_samples,
            "ADLHLP1": [2] * n_samples,
            "IADLHP1": [2] * n_samples,
            "WLKLIM1": [2] * n_samples,
            "SOCLIM1": [2] * n_samples,
            "COGLIM1": [2] * n_samples,
            "OBTOTVY1": [0] * n_samples,
            "OBDRVY1": [0] * n_samples,
            "OPTOTVY1": [0] * n_samples,
            "ERTOTY1": [0] * n_samples,
            "IPDISY1": [0] * n_samples,
            "RXTOTY1": [0] * n_samples,
            "TOTEXPY1": [0.0] * n_samples,
            "TOTSLFY1": [0.0] * n_samples,
            "HAVEUS2": [1] * n_samples,
            "LOCATN2": [1] * n_samples,
            "PROVTY2_M18": [1] * n_samples,
            "INSCOVY1": [1] * n_samples,
            "PRIEUY1": [1] * n_samples,
            "PRINGY1": [2] * n_samples,
            "PUBY1X": [2] * n_samples,
            "MCAIDY1X": [2] * n_samples,
            "MCAREY1X": [2] * n_samples,
            "TRICRY1X": [2] * n_samples,
            "VAPROGY1": [2] * n_samples,
        }

        # Baseline months: All 1 except row 7 which has one month = 2 (uninsured in Jan Y1, should be excluded)
        for m in BASELINE_INSURANCE_MONTHS:
            data[m] = [1] * n_samples
        data["INSJAY1X"][7] = 2  # Row 7 excluded for baseline disruption

        # Follow-up months: Row 0 has INSJAY2X = 2 (interrupted), Row 1 has 3 months = 2, Row 2 has INSDEY2X = 2
        for m in FOLLOWUP_INSURANCE_MONTHS:
            data[m] = [1] * n_samples
        data["INSJAY2X"][0] = 2  # Row 0: primary positive
        data["INSJAY2X"][1] = 2  # Row 1: prolonged positive (3 months)
        data["INSFEY2X"][1] = 2
        data["INSMAY2X"][1] = 2
        data["INSDEY2X"][2] = 2  # Row 2: yearend positive

        df = pd.DataFrame(data)
        cohort = extract_meps_cohort(df, panel_number=26, allow_target=True)

        # 10 rows initially - 2 out of age range (indices 5, 6) - 1 baseline interrupted (index 7) = 7 eligible rows
        self.assertEqual(cohort.raw_record_count, 10)
        self.assertEqual(cohort.eligible_record_count, 7)
        self.assertEqual(len(cohort.X), 7)
        self.assertEqual(len(cohort.y), 7)

        # Primary outcome: row 0, 1, 2 should be 1.0; others 0.0
        self.assertEqual(cohort.y.iloc[0], 1.0)
        self.assertEqual(cohort.y.iloc[1], 1.0)
        self.assertEqual(cohort.y.iloc[2], 1.0)
        self.assertEqual(cohort.y.iloc[3], 0.0)

        # Prolonged outcome: row 1 should be 1.0, row 0 and 2 should be 0.0
        self.assertEqual(cohort.y_prolonged.iloc[1], 1.0)
        self.assertEqual(cohort.y_prolonged.iloc[0], 0.0)

        # Yearend outcome: row 2 should be 1.0, row 0 and 1 should be 0.0
        self.assertEqual(cohort.y_yearend.iloc[2], 1.0)
        self.assertEqual(cohort.y_yearend.iloc[0], 0.0)

        # Leakage validation
        cohort.validate_leakage_and_integrity()

    def test_strict_target_fails_closed_on_invalid_followup_codes(self) -> None:
        """Verify strict target construction fails closed (raises ValueError) if any row has non 1/2 follow-up code."""
        data = {
            "DUPERSID": ["ID1"],
            "DUID": [100],
            "PID": [101],
            "PANEL": [26],
            "YEARIND": [1],
            "ALL5RDS": [1],
            "LONGWT": [1000.0],
            "VARSTR": [2001],
            "VARPSU": [1],
            "AGEY1X": [30],
            "RACETHX": [1],
            "SEX": [1],
            "POVCATY1": [1],
            "POVLEVY1": [100.0],
        }
        for m in BASELINE_INSURANCE_MONTHS:
            data[m] = [1]
        for m in FOLLOWUP_INSURANCE_MONTHS:
            data[m] = [1]

        # Case 1: -1 in one follow-up month (e.g. INSJAY2X == -1)
        data_invalid_1 = dict(data)
        data_invalid_1["INSJAY2X"] = [-1]
        df_invalid_1 = pd.DataFrame(data_invalid_1)
        with self.assertRaises(ValueError) as ctx:
            extract_meps_cohort(df_invalid_1, panel_number=26, allow_target=True)
        self.assertIn("Target ambiguity stop condition", str(ctx.exception))

        # Case 2: -7 (Refused) in follow-up month
        data_invalid_2 = dict(data)
        data_invalid_2["INSDEY2X"] = [-7]
        df_invalid_2 = pd.DataFrame(data_invalid_2)
        with self.assertRaises(ValueError) as ctx:
            extract_meps_cohort(df_invalid_2, panel_number=26, allow_target=True)
        self.assertIn("Target ambiguity stop condition", str(ctx.exception))

    def test_temporal_contract_and_leakage_rejection(self) -> None:
        """Verify that temporal contract rejects all Round 3, 4, 5 and Year 2 variables, while allowing Round 1, 2, Y1."""
        df = pd.DataFrame({
            "DUPERSID": ["ID1"],
            "DUID": [100],
            "PID": [101],
            "PANEL": [26],
            "YEARIND": [1],
            "ALL5RDS": [1],
            "LONGWT": [1000.0],
            "VARSTR": [2001],
            "VARPSU": [1],
            "AGEY1X": [30],
            "RACETHX": [1],
            "SEX": [1],
            "POVCATY1": [1],
            "POVLEVY1": [100.0],
        })
        for m in BASELINE_INSURANCE_MONTHS:
            df[m] = [1]
        for m in FOLLOWUP_INSURANCE_MONTHS:
            df[m] = [1]

        cohort = extract_meps_cohort(df, panel_number=26, allow_target=True)

        # 1. Prohibited Round 3 predictors must raise ValueError
        round3_vars = [
            "ACTLIM3", "ADLHLP3", "IADLHP3", "WLKLIM3", "SOCLIM3", "COGLIM3",
            "CHBRON3", "JTPAIN3_M18", "MARRY3X", "REGION3", "EMPST3", "HOUR3",
            "NUMEMP3", "OFFER3X", "HELD3X", "CHOIC3", "SELFCM3", "UNION3",
            "RTHLTH3", "MNHLTH3"
        ]
        for r3_var in round3_vars:
            cohort.X[r3_var] = [1]
            with self.assertRaises(ValueError, msg=f"Should reject Round 3 variable {r3_var}"):
                cohort.validate_leakage_and_integrity()
            del cohort.X[r3_var]

        # 2. Prohibited post-baseline Round 4 and 5 variables
        for r_var in ["RTHLTH4", "RTHLTH5", "HOUR4", "HOUR5"]:
            cohort.X[r_var] = [1]
            with self.assertRaises(ValueError, msg=f"Should reject post-baseline variable {r_var}"):
                cohort.validate_leakage_and_integrity()
            del cohort.X[r_var]

        # 3. Prohibited Year 2 variables
        for y2_var in ["POVCATY2", "TOTEXPY2", "INSJAY2X", "MARRYY2X"]:
            cohort.X[y2_var] = [1]
            with self.assertRaises(ValueError, msg=f"Should reject Year 2 variable {y2_var}"):
                cohort.validate_leakage_and_integrity()
            del cohort.X[y2_var]

        # 4. Prohibited identifiers, design columns, and protected attributes
        for col_name, err_sub in [
            ("DUPERSID", "identifier"),
            ("VARSTR", "survey design"),
            ("SEX", "Protected attribute"),
            ("RACETHX", "Protected attribute"),
        ]:
            cohort.X[col_name] = [1]
            with self.assertRaises(ValueError):
                cohort.validate_leakage_and_integrity()
            del cohort.X[col_name]

    def test_real_panel26_cohort_extraction_and_power(self) -> None:
        """Integration test on real extracted HC-244 Stata file: verify cohort size, 136 positive events, and underpowering."""
        dta_path = self.repo_root / "data/interim/meps/h244/h244.dta"
        if not dta_path.is_file():
            self.skipTest(f"Extracted Stata file not found: {dta_path}")

        df = pd.read_stata(dta_path, convert_categoricals=False)
        cohort = extract_meps_cohort(df, panel_number=26, allow_target=True)

        self.assertEqual(cohort.panel, 26)
        self.assertEqual(cohort.raw_record_count, 6741)
        self.assertGreater(cohort.eligible_record_count, 2000)
        self.assertEqual(cohort.eligible_record_count, 2882)

        positives = int((cohort.y == 1.0).sum())
        self.assertEqual(
            positives,
            136,
            f"Expected exactly 136 positive events in Panel 26, got {positives}"
        )
        # Verify that underpowering flag (< 200) is correctly detectable for pre-unlock gating
        is_adequately_powered = (positives >= 200)
        self.assertFalse(is_adequately_powered, "Panel 26 positive cases (136) must be correctly flagged as underpowered (< 200)")
        self.assertFalse(cohort.X.isna().all().any())
        self.assertEqual(len(cohort.design), cohort.eligible_record_count)
        self.assertEqual(len(cohort.audit), cohort.eligible_record_count)


if __name__ == "__main__":
    unittest.main()

