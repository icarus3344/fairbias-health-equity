"""Offline tests for NHIS D0 recoding, roles, and source preservation."""

from __future__ import annotations

import pathlib
import sys
import unittest

import pandas as pd

_SRC_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from nhis_fairbias.harmonize import (
    combine_harmonized_years,
    harmonize_year,
    validate_harmonized_roles,
)
from nhis_fairbias.schema import load_variable_registry, REQUIRED_COLUMNS


def _frame(year: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SRVY_YR": pd.Series([year, year, year, year], dtype="Int64"),
            "WTFA_A": [100.0, 200.0, 300.0, 400.0],
            "PSTRAT": pd.Series([1, 1, 2, 2], dtype="Int64"),
            "PPSU": pd.Series([1, 2, 1, 2], dtype="Int64"),
            "MEDDL12M_A": pd.Series([1, 2, 7, 9], dtype="Int64"),
            "MEDNG12M_A": pd.Series([2, 1, 8, 9], dtype="Int64"),
            "SEX_A": pd.Series([1, 2, 7, 1], dtype="Int64"),
            "HISPALLP_A": pd.Series([1, 7, 8, 2], dtype="Int64"),
            "DISAB3_A": pd.Series([1, 2, 9, 1], dtype="Int64"),
        }
    )


class TestNHISHarmonize(unittest.TestCase):
    def test_outcomes_and_protected_codes_are_derived_separately(self) -> None:
        registry = load_variable_registry()
        source = _frame(2022)
        output = harmonize_year(
            source,
            year=2022,
            study_role="development_train",
            registry=registry,
        )
        self.assertEqual(output["meddl12m"].tolist(), [1, 0, pd.NA, pd.NA])
        self.assertEqual(output["meddly12m"].tolist(), [1, 0, pd.NA, pd.NA])
        self.assertEqual(output["medng12m"].tolist(), [0, 1, pd.NA, pd.NA])
        self.assertEqual(output["hispallp_a"].tolist(), [1, 7, pd.NA, 2])
        self.assertEqual(output["sex_a"].tolist(), [1, 2, pd.NA, 1])
        self.assertEqual(output["disab3_a"].tolist(), [1, 2, pd.NA, 1])
        self.assertEqual(output["WTFA_DEV"].tolist(), [50.0, 100.0, 150.0, 200.0])

    def test_raw_source_columns_are_not_overwritten(self) -> None:
        registry = load_variable_registry()
        source = _frame(2022)
        output = harmonize_year(
            source,
            year=2022,
            study_role="development_train",
            registry=registry,
        )
        for column in REQUIRED_COLUMNS:
            pd.testing.assert_series_equal(output[column], source[column], check_names=False)
        pd.testing.assert_series_equal(output["MEDDL12M_A_raw"], source["MEDDL12M_A"], check_names=False)
        pd.testing.assert_series_equal(output["MEDNG12M_A_raw"], source["MEDNG12M_A"], check_names=False)

    def test_fixed_roles_and_2024_frozen_weight_lock(self) -> None:
        registry = load_variable_registry()
        frames = {
            2022: harmonize_year(_frame(2022), year=2022, study_role="development_train", registry=registry),
            2023: harmonize_year(_frame(2023), year=2023, study_role="development_validation", registry=registry),
            2024: harmonize_year(_frame(2024), year=2024, study_role="frozen_test", registry=registry),
        }
        combined = combine_harmonized_years(frames, [2022, 2023, 2024])
        validate_harmonized_roles(combined, [2022, 2023, 2024])
        self.assertTrue(combined.loc[combined["survey_year"].eq(2024), "WTFA_DEV"].isna().all())
        self.assertEqual(set(combined.loc[combined["survey_year"].eq(2024), "study_role"]), {"frozen_test"})
