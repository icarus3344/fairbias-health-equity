"""Integration tests for the audited NHIS 2022-2024 D0 data foundation."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest

import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_DIR = str(_REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from nhis_fairbias.harmonize import read_harmonized_parquet, validate_harmonized_roles
from nhis_fairbias.schema import load_study_config, load_variable_registry, validate_harmonized_columns


PARQUET_PATH = _REPO_ROOT / "data" / "processed" / "nhis" / "nhis_2022_2024_core.parquet"
MANIFEST_PATH = _REPO_ROOT / "artifacts" / "nhis" / "data" / "data_manifest.json"
SCHEMA_AUDIT_PATH = _REPO_ROOT / "artifacts" / "nhis" / "data" / "schema_audit.csv"
OUTCOME_AUDIT_PATH = _REPO_ROOT / "artifacts" / "nhis" / "data" / "outcome_audit.csv"
PROTECTED_AUDIT_PATH = _REPO_ROOT / "artifacts" / "nhis" / "data" / "protected_attribute_audit.csv"
WEIGHT_AUDIT_PATH = _REPO_ROOT / "artifacts" / "nhis" / "data" / "weight_audit.csv"


@unittest.skipUnless(PARQUET_PATH.is_file(), "NHIS prepared parquet not present; skipping real-data integration tests.")
class TestNHISIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = read_harmonized_parquet(PARQUET_PATH)
        self.study_config = load_study_config()
        self.registry = load_variable_registry()

    def test_total_and_per_year_row_counts(self) -> None:
        self.assertEqual(len(self.frame), 89802)
        counts = self.frame["survey_year"].value_counts().to_dict()
        self.assertEqual(counts[2022], 27651)
        self.assertEqual(counts[2023], 29522)
        self.assertEqual(counts[2024], 32629)

    def test_contract_columns_and_roles(self) -> None:
        validate_harmonized_columns(self.frame)
        validate_harmonized_roles(self.frame, [2022, 2023, 2024])
        self.assertIn("meddl12m", self.frame.columns)
        self.assertIn("medng12m", self.frame.columns)

    def test_exact_primary_outcome_frequencies(self) -> None:
        expected = {
            2022: {1: 1770, 2: 25683, 7: 3, 8: 192, 9: 3},
            2023: {1: 1931, 2: 27352, 7: 6, 8: 222, 9: 11},
            2024: {1: 2564, 2: 29791, 7: 10, 8: 255, 9: 9},
        }
        for year, counts in expected.items():
            subset = self.frame.loc[self.frame["survey_year"].eq(year)]
            raw_counts = subset["MEDDL12M_A_raw"].value_counts().to_dict()
            for code, expected_count in counts.items():
                self.assertEqual(raw_counts.get(code, 0), expected_count, f"Mismatch for {year} code {code}")
            self.assertEqual(sum(raw_counts.values()), len(subset))

    def test_development_weights_lock(self) -> None:
        # 2022 and 2023: WTFA_DEV == WTFA_A / 2
        for year in (2022, 2023):
            subset = self.frame.loc[self.frame["survey_year"].eq(year)]
            self.assertTrue(subset["WTFA_DEV"].notna().all())
            diff = (subset["WTFA_DEV"] - subset["WTFA_A"] / 2.0).abs()
            self.assertTrue((diff < 1e-9).all())
        # 2024: WTFA_DEV must be completely missing
        subset_2024 = self.frame.loc[self.frame["survey_year"].eq(2024)]
        self.assertTrue(subset_2024["WTFA_DEV"].isna().all())

    def test_audit_artifacts_exist_and_pass(self) -> None:
        for path in (MANIFEST_PATH, SCHEMA_AUDIT_PATH, OUTCOME_AUDIT_PATH, PROTECTED_AUDIT_PATH, WEIGHT_AUDIT_PATH):
            self.assertTrue(path.is_file(), f"Missing audit artifact: {path}")

        # Check schema audit
        schema_audit = pd.read_csv(SCHEMA_AUDIT_PATH)
        self.assertTrue((schema_audit["status"] == "PASS").all())

        # Check weight audit
        weight_audit = pd.read_csv(WEIGHT_AUDIT_PATH)
        self.assertTrue((weight_audit["status"] == "PASS").all())
        self.assertEqual(set(weight_audit["year"]), {2022, 2023, 2024})

        # Check manifest
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["audit_status"], "PASS")
        self.assertEqual(manifest["processed"]["row_count"], 89802)
        for year_str in ("2022", "2023", "2024"):
            self.assertEqual(manifest["years"][year_str]["validation"]["row_count"], "PASS")
            self.assertEqual(manifest["years"][year_str]["validation"]["meddl12m_frequency"], "PASS")
