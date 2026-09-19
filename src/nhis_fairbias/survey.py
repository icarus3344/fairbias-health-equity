"""Descriptive survey-design diagnostics for NHIS D0 artifacts."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
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
    non_finite_n = int((weights.notna() & ~np.isfinite(weights)).sum())
    nonpositive = int(weights.le(0).fillna(False).sum())
    pstrat = frame["PSTRAT"] if "PSTRAT" in frame.columns else pd.Series(pd.NA, index=frame.index)
    ppsu = frame["PPSU"] if "PPSU" in frame.columns else pd.Series(pd.NA, index=frame.index)
    status = "PASS" if (non_numeric_n == 0 and non_finite_n == 0) else "FAIL"
    pstrat_missing = int(pstrat.isna().sum())
    ppsu_missing = int(ppsu.isna().sum())
    wtfa_sum = _finite_or_none(weights.sum(min_count=1))
    # Boundary notice: `survey_inference_eligible` certifies only data cleanliness prerequisites
    # (PSTRAT/PPSU present and complete, weights positive, finite, and numeric). It does NOT
    # certify that complex design variance estimation (e.g., Taylor series linearization,
    # jackknife, degrees-of-freedom, or singleton PSU handling) is fully estimable or identified;
    # complex survey sampling variance inference remains explicitly deferred.
    survey_inference_eligible = bool(
        non_numeric_n == 0
        and non_finite_n == 0
        and weight_missing == 0
        and nonpositive == 0
        and pstrat_missing == 0
        and ppsu_missing == 0
        and wtfa_sum is not None
        and wtfa_sum > 0
        and len(frame) > 0
    )
    return {
        "year": int(year),
        "wtfa_a_exists": True,
        "wtfa_a_numeric": non_numeric_n == 0,
        "wtfa_a_non_numeric_n": non_numeric_n,
        "wtfa_a_non_finite_n": non_finite_n,
        "wtfa_a_missing_n": weight_missing,
        "wtfa_a_nonpositive_n": nonpositive,
        "missing_wtfa_a": weight_missing,
        "count_wtfa_a_nonpositive": nonpositive,
        "missing_pstrat": pstrat_missing,
        "missing_ppsu": ppsu_missing,
        "unique_pstrat": int(pstrat.nunique(dropna=True)),
        "unique_ppsu": int(ppsu.nunique(dropna=True)),
        "pstrat_missing_n": pstrat_missing,
        "ppsu_missing_n": ppsu_missing,
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
        "survey_inference_eligible": survey_inference_eligible,
    }


def positive_weight_mask(weights: pd.Series) -> pd.Series:
    """Return rows eligible for descriptive weighted proportions."""

    numeric = pd.to_numeric(weights, errors="coerce")
    return numeric.notna() & np.isfinite(numeric) & numeric.gt(0)


def _validate_proportion_inputs(
    raw_codes: pd.Series,
    weights: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Validate alignment, uniqueness, and shape for direct proportion functions.

    Ensures 1D Series, identical lengths, unique indices, and exact index alignment.
    Raises ValueError on mismatches, duplicate indices, or dimensionality issues.
    """
    if not isinstance(raw_codes, pd.Series):
        if hasattr(raw_codes, "ndim") and raw_codes.ndim != 1:
            raise ValueError(f"raw_codes must be 1-dimensional, got ndim={raw_codes.ndim}")
        raw_codes = pd.Series(raw_codes)
    if not isinstance(weights, pd.Series):
        if hasattr(weights, "ndim") and weights.ndim != 1:
            raise ValueError(f"weights must be 1-dimensional, got ndim={weights.ndim}")
        weights = pd.Series(weights)

    if raw_codes.ndim != 1:
        raise ValueError(f"raw_codes must be 1-dimensional, got ndim={raw_codes.ndim}")
    if weights.ndim != 1:
        raise ValueError(f"weights must be 1-dimensional, got ndim={weights.ndim}")
    if len(raw_codes) != len(weights):
        raise ValueError(
            f"Input length mismatch: raw_codes has length {len(raw_codes)}, "
            f"weights has length {len(weights)}"
        )
    if not raw_codes.index.is_unique:
        raise ValueError("raw_codes index must be unique without duplicate labels")
    if not weights.index.is_unique:
        raise ValueError("weights index must be unique without duplicate labels")
    if not raw_codes.index.equals(weights.index):
        raise ValueError(
            "Index alignment mismatch: raw_codes index does not match weights index in values or order"
        )
    return raw_codes, weights


def weighted_binary_proportion(
    raw_codes: pd.Series,
    weights: pd.Series,
    *,
    code: int,
    valid_codes: tuple[int, ...] = (1, 2),
) -> float | None:
    """Compute a weighted proportion among substantive outcome codes only."""

    raw_codes, weights = _validate_proportion_inputs(raw_codes, weights)
    if len(raw_codes) == 0:
        return None

    codes = canonical_code_series(raw_codes)
    numeric_weights = pd.to_numeric(weights, errors="coerce")
    eligible = positive_weight_mask(numeric_weights) & codes.isin(list(valid_codes)).fillna(False)
    if not eligible.any():
        return None

    elig_w = numeric_weights.loc[eligible]
    max_w = float(elig_w.max())
    if max_w <= 0 or not np.isfinite(max_w):
        return None

    # Rescale by max_w to prevent floating point sum overflow
    scaled_w = elig_w / max_w
    denominator = float(scaled_w.sum(min_count=1))
    if pd.isna(denominator) or denominator <= 0 or not np.isfinite(denominator):
        return None

    match_mask = eligible & codes.eq(int(code))
    if not match_mask.any():
        return 0.0

    numerator = float((numeric_weights.loc[match_mask] / max_w).sum(min_count=1))
    if pd.isna(numerator) or not np.isfinite(numerator):
        return 0.0
    return float(numerator / denominator)


def weighted_category_proportion(
    raw_codes: pd.Series,
    weights: pd.Series,
    *,
    code: int | None,
    valid_codes: tuple[int, ...],
) -> float | None:
    """Compute a category share using all positive-weight records as denominator."""

    raw_codes, weights = _validate_proportion_inputs(raw_codes, weights)
    if len(raw_codes) == 0:
        return None

    codes = canonical_code_series(raw_codes)
    numeric_weights = pd.to_numeric(weights, errors="coerce")
    eligible = positive_weight_mask(numeric_weights)
    if not eligible.any():
        return None

    elig_w = numeric_weights.loc[eligible]
    max_w = float(elig_w.max())
    if max_w <= 0 or not np.isfinite(max_w):
        return None

    scaled_w = elig_w / max_w
    denominator = float(scaled_w.sum(min_count=1))
    if pd.isna(denominator) or denominator <= 0 or not np.isfinite(denominator):
        return None

    if code is None:
        category = ~codes.isin(list(valid_codes)).fillna(False)
    else:
        category = codes.eq(int(code)).fillna(False)
    match_mask = eligible & category
    if not match_mask.any():
        return 0.0

    numerator = float((numeric_weights.loc[match_mask] / max_w).sum(min_count=1))
    if pd.isna(numerator) or not np.isfinite(numerator):
        return 0.0
    return float(numerator / denominator)


def validate_survey_weights(
    weights: pd.Series | np.ndarray,
    expected_length: int | None = None,
    expected_index: pd.Index | None = None,
) -> np.ndarray:
    """Validate survey weights for the survey-weighted FairBias geometry extension.

    Requirements:
    - Finite (no NaN, inf, or -inf)
    - Non-negative (w >= 0; individual 0s permitted if group sum > 0)
    - Strictly positive total sum (not all-zero)
    - Finite total sum without floating-point overflow (overflow to inf is rejected)
    - Exact length match if expected_length provided
    - Exact index alignment if weights is a Series and expected_index provided
    """
    if isinstance(weights, pd.Series):
        if expected_index is not None and not weights.index.equals(expected_index):
            raise ValueError(
                "Index alignment mismatch: survey weights index does not match target index"
            )
        w_arr = np.asarray(weights.values, dtype=float)
    else:
        w_arr = np.asarray(weights, dtype=float)

    if w_arr.ndim != 1:
        raise ValueError(f"survey weights must be 1-dimensional, got ndim={w_arr.ndim}")
    if expected_length is not None and len(w_arr) != int(expected_length):
        raise ValueError(
            f"survey weights length ({len(w_arr)}) does not match expected length ({expected_length})"
        )
    if not np.all(np.isfinite(w_arr)):
        raise ValueError("survey weights contain NaN, Inf, or non-finite values")
    if np.any(w_arr < 0):
        raise ValueError("survey weights must be non-negative (w >= 0)")
    total_w = float(np.sum(w_arr))
    if not np.isfinite(total_w):
        raise ValueError("survey weights total sum overflows to infinity; non-finite sum rejected")
    if np.all(w_arr == 0) or total_w <= 0:
        raise ValueError("survey weights cannot be all-zero; total weight must be strictly positive")
    return w_arr


def get_survey_weighted_geometry_provenance(
    survey_weight_variable: str = "WTFA_A",
) -> dict[str, Any]:
    """Standardized provenance metadata dictionary for survey-weighted geometry extension (Gate D5)."""
    return {
        "base_algorithm": "tang2024_paper_faithful",
        "extension": "survey_weighted_geometry",
        "release_protocol": "survey_weighted_geometry_extension",
        "survey_weight_variable": str(survey_weight_variable),
        "classifier_weighted": False,
        "evaluation_weighted": False,
        "complex_survey_inference": False,
        "PSTRAT_used_for_variance": False,
        "PPSU_used_for_variance": False,
    }
