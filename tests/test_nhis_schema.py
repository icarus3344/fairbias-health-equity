"""Offline tests for the explicit NHIS D0 registry and source checks."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import pandas as pd

_SRC_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from nhis_fairbias.schema import (
    NHISSchemaError,
    REQUIRED_COLUMNS,
    exact_primary_frequency,
    load_study_config,
    load_variable_registry,
    read_nhis_csv,
    validate_required_columns,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SRVY_YR": [2022, 2022, 2022, 2022, 2022],
            "WTFA_A": [10.0, 20.0, 30.0, 40.0, 50.0],
            "PSTRAT": [1, 1, 2, 2, 3],
            "PPSU": [1, 2, 1, 2, 1],
            "MEDDL12M_A": [1, 2, 7, 8, 9],
            "MEDNG12M_A": [1, 2, 7, 8, 9],
            "SEX_A": [1, 2, 1, 2, 7],
            "HISPALLP_A": [1, 2, 3, 7, 8],
            "DISAB3_A": [1, 2, 1, 2, 9],
        }
    )


class TestNHISSchema(unittest.TestCase):
    def test_registry_has_exact_required_columns_and_explicit_roles(self) -> None:
        registry = load_variable_registry()
        self.assertEqual(tuple(registry["required_variables"]), REQUIRED_COLUMNS)
        self.assertEqual(registry["variables"]["HISPALLP_A"]["valid_codes"], [1, 2, 3, 4, 5, 6, 7])
        self.assertTrue(registry["variables"]["HISPALLP_A"]["do_not_merge_categories"])
        self.assertEqual(
            registry["variables"]["HISPALLP_A"]["code_labels"]["7"],
            "Other single and multiple races",
        )

    def test_required_columns_fail_closed(self) -> None:
        with self.assertRaises(NHISSchemaError):
            validate_required_columns(["SRVY_YR"])

    def test_exact_primary_frequency_checks_all_codes(self) -> None:
        observed = exact_primary_frequency(
            _frame(),
            year=2022,
            expected_counts={"1": 1, "2": 1, "7": 1, "8": 1, "9": 1},
        )
        self.assertEqual(observed, {1: 1, 2: 1, 7: 1, 8: 1, 9: 1})
        with self.assertRaises(NHISSchemaError):
            exact_primary_frequency(
                _frame(), year=2022, expected_counts={"1": 2, "2": 1, "7": 1, "8": 1, "9": 1}
            )

    def test_csv_reader_uses_declared_types_not_dtype_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = pathlib.Path(tmp) / "adult22.csv"
            _frame().to_csv(csv_path, index=False)
            frame = read_nhis_csv(csv_path, registry=load_variable_registry())
            self.assertEqual(tuple(frame.columns), REQUIRED_COLUMNS)
            self.assertEqual(str(frame["SRVY_YR"].dtype), "Int64")
            self.assertEqual(str(frame["WTFA_A"].dtype), "float64")
