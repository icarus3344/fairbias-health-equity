"""Offline tests for NHIS D0 survey-design diagnostics."""

from __future__ import annotations

import pathlib
import sys
import unittest

import pandas as pd

_SRC_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from nhis_fairbias.survey import (
    survey_design_audit,
    weighted_binary_proportion,
    weighted_category_proportion,
)


class TestNHISSurvey(unittest.TestCase):
    def test_weight_diagnostics_are_explicit_and_non_iid(self) -> None:
        frame = pd.DataFrame(
            {
                "WTFA_A": ["10", "20", "30", "40"],
                "PSTRAT": [1, 1, 2, 2],
                "PPSU": [1, 2, 1, 2],
            }
        )
        result = survey_design_audit(frame, year=2022)
        self.assertTrue(result["wtfa_a_exists"])
        self.assertTrue(result["wtfa_a_numeric"])
        self.assertEqual(result["wtfa_a_nonpositive_n"], 0)
        self.assertEqual(result["pstrat_nunique"], 2)
        self.assertEqual(result["ppsu_nunique"], 2)
        self.assertEqual(result["wtfa_a_sum"], 100.0)
        self.assertEqual(result["wtfa_a_median"], 25.0)

    def test_weighted_outcome_proportion_uses_substantive_denominator(self) -> None:
        raw = pd.Series([1, 2, 7], dtype="Int64")
        weights = pd.Series([10.0, 20.0, 30.0])
        self.assertAlmostEqual(weighted_binary_proportion(raw, weights, code=1) or 0.0, 1 / 3)
        self.assertAlmostEqual(weighted_binary_proportion(raw, weights, code=2) or 0.0, 2 / 3)

    def test_protected_category_audit_keeps_categories_and_missing_bucket(self) -> None:
        raw = pd.Series([1, 7, 8], dtype="Int64")
        weights = pd.Series([10.0, 20.0, 30.0])
        self.assertAlmostEqual(
            weighted_category_proportion(raw, weights, code=1, valid_codes=(1, 2, 3, 4, 5, 6, 7)) or 0.0,
            1 / 6,
        )
        self.assertAlmostEqual(
            weighted_category_proportion(raw, weights, code=None, valid_codes=(1, 2, 3, 4, 5, 6, 7)) or 0.0,
            1 / 2,
        )
