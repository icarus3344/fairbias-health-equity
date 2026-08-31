"""Train-only preprocessing, variable missing-code contracts, and survey design quality checks."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any, Sequence

import numpy as np
import pandas as pd

from meps_fairness.data.cohort import (
    ALL_CATEGORICAL_PREDICTORS,
    ALL_CONTINUOUS_PREDICTORS,
    resolve_prior_round_inheritance,
)

# Standard item non-response sentinels across MEPS
ITEM_NONRESPONSE_CODES: tuple[int, ...] = (-7, -8, -15)
ITEM_NONRESPONSE_FLOATS: tuple[float, ...] = (-7.0, -8.0, -15.0)

# Legitimate continuous income/poverty variables that can have valid negative values
LEGITIMATE_NEGATIVE_CONTINUOUS: tuple[str, ...] = ("TTLPY1X", "FAMINCY1", "POVLEVY1")

# Continuous employment variables where -1 INAPPLICABLE represents structural zero
STRUCTURAL_ZERO_CONTINUOUS: tuple[str, ...] = ("HOUR1", "HOUR2", "NUMEMP1", "NUMEMP2", "WAGEPY1X")


@dataclasses.dataclass(frozen=True)
class ContinuousFeatureStats:
    """Fitted statistics for a single continuous feature."""

    name: str
    impute_value: float
    mean: float
    std: float
    has_missing: bool
    is_structural_zero: bool
    is_negative_allowed: bool


@dataclasses.dataclass(frozen=True)
class CategoricalFeatureStats:
    """Fitted categories for a single categorical feature."""

    name: str
    categories: tuple[Any, ...]


class MEPSPreprocessor:
    """Strict train-only preprocessor for MEPS longitudinal data.

    Enforces:
    - Zero out-of-sample data leakage: fits strictly on training data.
    - Variable-specific missing contracts:
      1. Legitimate negative income/poverty values (TTLPY1X, FAMINCY1, POVLEVY1) are preserved as valid.
         Only exact codebook sentinel -1 (and non-response -7, -8, -15) is treated as missing.
      2. Inapplicable employment characteristics (HOUR1, HOUR2, NUMEMP1, NUMEMP2, WAGEPY1X == -1)
         are recoded to structural zero (0.0), not imputed with median worker values.
      3. Prior-round inheritance (-2: DETERMINED IN PREVIOUS ROUND) for HOUR2, NUMEMP2, CHOIC2,
         SELFCM2, UNION2 is resolved to Round 1 values before fitting/transforming.
      4. Categorical variables preserve codebook-defined inapplicable category (-1) as a distinct
         structural category, while treating item non-response (-7, -8, -15) as missing.
    - Continuous standardization: zero-mean unit-variance scaling based on training statistics.
    """

    def __init__(
        self,
        continuous_features: Sequence[str] | None = None,
        categorical_features: Sequence[str] | None = None,
        variable_contracts: dict[str, Any] | None = None,
    ) -> None:
        self.continuous_features = tuple(continuous_features or ALL_CONTINUOUS_PREDICTORS)
        self.categorical_features = tuple(categorical_features or ALL_CATEGORICAL_PREDICTORS)
        self.variable_contracts = variable_contracts
        self.fitted_continuous_: dict[str, ContinuousFeatureStats] = {}
        self.fitted_categorical_: dict[str, CategoricalFeatureStats] = {}
        self.feature_names_out_: list[str] = []
        self.is_fitted_: bool = False

    def _prepare_continuous_series(self, series: pd.Series, col: str) -> tuple[pd.Series, pd.Series, bool, bool]:
        """Prepare continuous series by handling structural zeros and identifying missing mask.

        Returns (adjusted_series, missing_mask, is_structural_zero, is_negative_allowed).
        """
        s = series.astype(float).copy()
        is_negative_allowed = col in LEGITIMATE_NEGATIVE_CONTINUOUS
        is_structural_zero = col in STRUCTURAL_ZERO_CONTINUOUS

        if is_structural_zero:
            # -1 INAPPLICABLE represents structural zero (e.g. unemployed hours/coworkers = 0)
            inapp_mask = s.isin([-1.0, -1])
            s[inapp_mask] = 0.0
            # Item non-response sentinels are true missing
            missing_mask = s.isin(ITEM_NONRESPONSE_FLOATS + ITEM_NONRESPONSE_CODES) | s.isna()
        elif is_negative_allowed:
            # -1 is missing/inapplicable sentinel; legitimate negative values (e.g. -300) are valid
            missing_mask = s.isin([-1.0, -1] + list(ITEM_NONRESPONSE_FLOATS + ITEM_NONRESPONSE_CODES)) | s.isna()
        else:
            # Standard continuous: negative values are missing sentinels (-1, -7, -8, -15)
            missing_mask = s.isin([-1.0, -1] + list(ITEM_NONRESPONSE_FLOATS + ITEM_NONRESPONSE_CODES)) | (s < 0) | s.isna()

        return s, missing_mask, is_structural_zero, is_negative_allowed

    def fit(self, X: pd.DataFrame) -> MEPSPreprocessor:
        """Fit preprocessing transformations strictly on the training partition X."""
        self.fitted_continuous_.clear()
        self.fitted_categorical_.clear()
        self.feature_names_out_.clear()

        X_inh = resolve_prior_round_inheritance(X)

        # 1. Fit Continuous Features
        for col in self.continuous_features:
            if col not in X_inh.columns:
                continue

            s_adj, missing_mask, is_struct_zero, is_neg_allowed = self._prepare_continuous_series(
                X_inh[col], col
            )
            valid_vals = s_adj[~missing_mask]

            if len(valid_vals) > 0:
                med = float(valid_vals.median())
                imputed = s_adj.copy()
                imputed[missing_mask] = med
                mean_val = float(imputed.mean())
                std_val = float(imputed.std(ddof=0))
                if std_val < 1e-8:
                    std_val = 1.0
            else:
                med = 0.0
                mean_val = 0.0
                std_val = 1.0

            has_missing = bool(missing_mask.any())
            self.fitted_continuous_[col] = ContinuousFeatureStats(
                name=col,
                impute_value=med,
                mean=mean_val,
                std=std_val,
                has_missing=has_missing,
                is_structural_zero=is_struct_zero,
                is_negative_allowed=is_neg_allowed,
            )
            self.feature_names_out_.append(col)
            if has_missing:
                self.feature_names_out_.append(f"{col}__missing")

        # 2. Fit Categorical Features
        for col in self.categorical_features:
            if col not in X_inh.columns:
                continue
            series = X_inh[col]
            # Exclude item non-response codes (-7, -8, -15) and NaN from valid category list
            # Preserve -1 INAPPLICABLE as a distinct structural category
            valid_mask = (~series.isin(ITEM_NONRESPONSE_CODES + ITEM_NONRESPONSE_FLOATS)) & series.notna()
            unique_cats = sorted(series[valid_mask].unique())

            self.fitted_categorical_[col] = CategoricalFeatureStats(
                name=col,
                categories=tuple(unique_cats),
            )
            for cat in unique_cats:
                self.feature_names_out_.append(f"{col}__{cat}")

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted transformations out-of-sample to any partition."""
        if not self.is_fitted_:
            raise ValueError("MEPSPreprocessor must be fitted before transform")

        X_inh = resolve_prior_round_inheritance(X)
        out_dict: dict[str, np.ndarray] = {}
        n_rows = len(X)

        # 1. Transform Continuous Features
        for col, stats in self.fitted_continuous_.items():
            if col in X_inh.columns:
                s_adj, missing_mask, _, _ = self._prepare_continuous_series(X_inh[col], col)
                imputed = s_adj.copy()
                imputed[missing_mask] = stats.impute_value
                scaled = (imputed.values - stats.mean) / stats.std
                out_dict[col] = scaled

                if stats.has_missing:
                    missing_flag = missing_mask.astype(float).values
                    out_dict[f"{col}__missing"] = missing_flag
            else:
                out_dict[col] = np.zeros(n_rows, dtype=float)
                if stats.has_missing:
                    out_dict[f"{col}__missing"] = np.zeros(n_rows, dtype=float)

        # 2. Transform Categorical Features
        for col, stats in self.fitted_categorical_.items():
            if col in X_inh.columns:
                series = X_inh[col]
                for cat in stats.categories:
                    match_flag = (series == cat).astype(float).values
                    out_dict[f"{col}__{cat}"] = match_flag
            else:
                for cat in stats.categories:
                    out_dict[f"{col}__{cat}"] = np.zeros(n_rows, dtype=float)

        # Return DataFrame with deterministic column ordering
        out_df = pd.DataFrame(out_dict, columns=self.feature_names_out_, index=X.index)
        return out_df

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform training data."""
        return self.fit(X).transform(X)


def check_survey_design_quality(design_df: pd.DataFrame) -> dict[str, Any]:
    """Execute structural survey design quality checks.

    Validates:
    - Positive finite LONGWT
    - Valid non-missing VARSTR and VARPSU
    - Kish effective sample size and weight summary statistics
    """
    n_records = len(design_df)
    if n_records == 0:
        return {"status": "FAILED", "error": "Empty design DataFrame"}

    w = design_df["LONGWT"].values
    if np.isnan(w).any() or np.isinf(w).any() or (w <= 0).any():
        return {
            "status": "FAILED",
            "error": "Non-positive, missing, or non-finite LONGWT values detected",
        }

    sum_w = float(np.sum(w))
    sum_w_sq = float(np.sum(w ** 2))
    kish_neff = float((sum_w ** 2) / sum_w_sq)

    strata = design_df["VARSTR"].values
    psus = design_df["VARPSU"].values

    if np.isnan(strata).any() or np.isnan(psus).any():
        return {
            "status": "FAILED",
            "error": "Missing VARSTR or VARPSU detected",
        }

    n_strata = int(len(np.unique(strata)))
    n_psus = int(len(design_df[["VARSTR", "VARPSU"]].drop_duplicates()))

    # Check minimum PSUs per stratum
    psu_counts_per_stratum = design_df.groupby("VARSTR")["VARPSU"].nunique()
    strata_with_single_psu = int((psu_counts_per_stratum < 2).sum())

    return {
        "status": "PASSED",
        "record_count": n_records,
        "sum_weights": sum_w,
        "min_weight": float(np.min(w)),
        "max_weight": float(np.max(w)),
        "mean_weight": float(np.mean(w)),
        "kish_effective_n": kish_neff,
        "unique_strata_count": n_strata,
        "unique_psu_clusters_count": n_psus,
        "strata_with_single_psu_count": strata_with_single_psu,
    }
