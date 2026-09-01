"""Descriptive survey-design diagnostics for NHIS D0 artifacts."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .schema import canonical_code_series


def explicit_numeric(series: pd.Series) -> tuple[pd.Series, int]:
    """Parse a declared numeric field and return parsed values plus bad-value count."""

    parsed = pd.to_numeric(series, errors="coerce")
    non_numeric = series.notna() & parsed.isna()
    return parsed, int(non_numeric.sum())


def _finite_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def survey_design_audit(frame: pd.DataFrame, *, year: int) -> dict[str, Any]:
    """Compute the requested weight/stratum/PSU diagnostics without iid assumptions."""

    weight_exists = "WTFA_A" in frame.columns
    if not weight_exists:
        return {
            "year": int(year),
            "wtfa_a_exists": False,
            "wtfa_a_numeric": False,
            "wtfa_a_non_numeric_n": None,
            "wtfa_a_missing_n": None,
            "wtfa_a_nonpositive_n": None,
            "pstrat_missing_n": None,
            "ppsu_missing_n": None,
            "pstrat_nunique": None,
            "ppsu_nunique": None,
            "wtfa_a_sum": None,
            "wtfa_a_min": None,
            "wtfa_a_median": None,
            "wtfa_a_max": None,
            "status": "FAIL",
        }

    weights, non_numeric_n = explicit_numeric(frame["WTFA_A"])
    weight_missing = int(weights.isna().sum())
    nonpositive = int(weights.le(0).fillna(False).sum())
    pstrat = frame["PSTRAT"] if "PSTRAT" in frame.columns else pd.Series(pd.NA, index=frame.index)
    ppsu = frame["PPSU"] if "PPSU" in frame.columns else pd.Series(pd.NA, index=frame.index)
    status = "PASS" if non_numeric_n == 0 else "FAIL"
    return {
        "year": int(year),
        "wtfa_a_exists": True,
        "wtfa_a_numeric": non_numeric_n == 0,
        "wtfa_a_non_numeric_n": non_numeric_n,
        "wtfa_a_missing_n": weight_missing,
        "wtfa_a_nonpositive_n": nonpositive,
        "missing_wtfa_a": weight_missing,
        "count_wtfa_a_nonpositive": nonpositive,
        "missing_pstrat": int(pstrat.isna().sum()),
        "missing_ppsu": int(ppsu.isna().sum()),
        "unique_pstrat": int(pstrat.nunique(dropna=True)),
        "unique_ppsu": int(ppsu.nunique(dropna=True)),
        "pstrat_missing_n": int(pstrat.isna().sum()),
        "ppsu_missing_n": int(ppsu.isna().sum()),
        "pstrat_nunique": int(pstrat.nunique(dropna=True)),
        "ppsu_nunique": int(ppsu.nunique(dropna=True)),
        "wtfa_a_sum": _finite_or_none(weights.sum(min_count=1)),
        "wtfa_a_min": _finite_or_none(weights.min()),
        "wtfa_a_median": _finite_or_none(weights.median()),
        "wtfa_a_max": _finite_or_none(weights.max()),
        "sum_wtfa_a": _finite_or_none(weights.sum(min_count=1)),
        "min_wtfa_a": _finite_or_none(weights.min()),
        "median_wtfa_a": _finite_or_none(weights.median()),
        "max_wtfa_a": _finite_or_none(weights.max()),
        "status": status,
    }


def positive_weight_mask(weights: pd.Series) -> pd.Series:
    """Return rows eligible for descriptive weighted proportions."""

    numeric = pd.to_numeric(weights, errors="coerce")
    return numeric.notna() & numeric.gt(0)


def weighted_binary_proportion(
    raw_codes: pd.Series,
    weights: pd.Series,
    *,
    code: int,
    valid_codes: tuple[int, ...] = (1, 2),
) -> float | None:
    """Compute a weighted proportion among substantive outcome codes only."""

    codes = canonical_code_series(raw_codes)
    numeric_weights = pd.to_numeric(weights, errors="coerce")
    eligible = positive_weight_mask(numeric_weights) & codes.isin(list(valid_codes)).fillna(False)
    denominator = numeric_weights.loc[eligible].sum(min_count=1)
    if pd.isna(denominator) or float(denominator) <= 0:
        return None
    numerator = numeric_weights.loc[eligible & codes.eq(int(code))].sum(min_count=1)
    return float(numerator / denominator)


def weighted_category_proportion(
    raw_codes: pd.Series,
    weights: pd.Series,
    *,
    code: int | None,
    valid_codes: tuple[int, ...],
) -> float | None:
    """Compute a category share using all positive-weight records as denominator."""

    codes = canonical_code_series(raw_codes)
    numeric_weights = pd.to_numeric(weights, errors="coerce")
    eligible = positive_weight_mask(numeric_weights)
    denominator = numeric_weights.loc[eligible].sum(min_count=1)
    if pd.isna(denominator) or float(denominator) <= 0:
        return None
    if code is None:
        category = ~codes.isin(list(valid_codes)).fillna(False)
    else:
        category = codes.eq(int(code)).fillna(False)
    numerator = numeric_weights.loc[eligible & category].sum(min_count=1)
    return float(numerator / denominator)
