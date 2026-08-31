"""Exploratory group-aware centering bias mitigation heuristic and survey-weighted extension.

DISCLAIMER & METHODOLOGICAL BOUNDARY:
This module implements an exploratory group-aware feature centering heuristic. It is NOT
a faithful reconstruction of Tang et al. (2024) and is NOT the primary paper result.
The heuristic requires knowledge of protected attributes (e.g., RACETHX or SEX) at inference
time to apply group-conditional offsets. Protected attributes are NEVER silently injected into
the primary predictor matrix X or treated as ordinary regression predictors.

The primary baseline model (WeightedLogisticClassifier) remains strictly group-agnostic at inference time.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from meps_fairness.models.baseline import WeightedLogisticClassifier


def compute_bias_concentration_epsilon(
    X: pd.DataFrame | np.ndarray,
    protected_series: pd.Series | np.ndarray,
    sample_weight: pd.Series | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute feature-level bias concentration epsilon across protected groups in O(N*D) time.

    Parameters:
        X: Feature matrix.
        protected_series: Protected attribute values (e.g. RACETHX or SEX).
        sample_weight: Optional survey analysis weights (LONGWT).

    Returns:
        Dictionary mapping column name to maximum absolute pairwise group difference.
    """
    if isinstance(X, np.ndarray):
        cols = [f"feat_{i}" for i in range(X.shape[1])]
        df_X = pd.DataFrame(X, columns=cols)
    else:
        df_X = X.copy()

    s_prot = pd.Series(np.asarray(protected_series)).reset_index(drop=True)
    df_X = df_X.reset_index(drop=True)

    if sample_weight is not None:
        w = pd.Series(np.asarray(sample_weight, dtype=float)).reset_index(drop=True)
    else:
        w = pd.Series(1.0, index=df_X.index)

    groups = sorted(s_prot.unique())
    if len(groups) < 2:
        return {col: 0.0 for col in df_X.columns}

    # Compute group-conditional weighted means
    group_means: dict[Any, pd.Series] = {}
    for g in groups:
        mask = (s_prot == g)
        sum_w = w[mask].sum()
        if sum_w > 0:
            # Weighted average per column
            weighted_sums = (df_X[mask].values * w[mask].values[:, np.newaxis]).sum(axis=0)
            group_means[g] = pd.Series(weighted_sums / sum_w, index=df_X.columns)
        else:
            group_means[g] = df_X[mask].mean() if mask.any() else pd.Series(0.0, index=df_X.columns)

    # Maximum pairwise group difference per column
    epsilon_dict: dict[str, float] = {}
    for col in df_X.columns:
        max_diff = 0.0
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1, g2 = groups[i], groups[j]
                diff = abs(group_means[g1][col] - group_means[g2][col])
                if diff > max_diff:
                    max_diff = diff
        epsilon_dict[col] = float(max_diff)

    return epsilon_dict


@dataclasses.dataclass(frozen=True)
class GroupDisparityAdjustment:
    """Fitted disparity adjustment parameters per feature and group."""

    feature: str
    group_offsets: dict[Any, float]


class ExploratoryGroupAwareCenteringMitigation:
    """Exploratory group-aware centering heuristic for bias mitigation.

    METHODOLOGICAL NOTE:
    - This is an exploratory heuristic, NOT a faithful Tang et al. (2024) reconstruction.
    - Requires protected attributes (protected_series) explicitly at inference time for group centering.
    - Protected attributes are used strictly for centering transformations, never as raw predictive features.
    - Zero data leakage: learned strictly on the training partition.
    """

    def __init__(
        self,
        shrinkage_intensity: float = 0.5,
        C: float = 1.0,
        random_state: int = 20260828,
    ) -> None:
        self.shrinkage_intensity = shrinkage_intensity
        self.C = C
        self.random_state = random_state
        self.base_model_ = WeightedLogisticClassifier(C=self.C, random_state=self.random_state)
        self.adjustments_: dict[str, GroupDisparityAdjustment] = {}
        self.feature_names_in_: list[str] = []
        self.is_fitted_: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        protected_series: pd.Series,
        sample_weight: pd.Series | None = None,
    ) -> ExploratoryGroupAwareCenteringMitigation:
        """Fit group offsets and base classifier strictly on training data."""
        self.feature_names_in_ = list(X.columns)
        self.adjustments_.clear()

        s_prot = pd.Series(np.asarray(protected_series)).reset_index(drop=True)
        df_X = X.copy().reset_index(drop=True)
        groups = sorted(s_prot.unique())

        # Unweighted overall mean
        overall_mean = df_X.mean(axis=0)

        # Compute group offsets for each feature
        for col in df_X.columns:
            group_offsets: dict[Any, float] = {}
            for g in groups:
                mask = (s_prot == g)
                if mask.any():
                    g_mean = float(df_X.loc[mask, col].mean())
                    group_offsets[g] = g_mean - float(overall_mean[col])
                else:
                    group_offsets[g] = 0.0

            self.adjustments_[col] = GroupDisparityAdjustment(
                feature=col,
                group_offsets=group_offsets,
            )

        # Transform training data
        X_mitigated = self.transform(df_X, s_prot)

        # Fit base model on mitigated training features
        self.base_model_.fit(X_mitigated, y, sample_weight=sample_weight)
        self.is_fitted_ = True
        return self

    def transform(
        self,
        X: pd.DataFrame,
        protected_series: pd.Series,
    ) -> pd.DataFrame:
        """Apply learned group disparity shrinkage to features using group indicators."""
        s_prot = pd.Series(np.asarray(protected_series)).reset_index(drop=True)
        df_X = X.copy().reset_index(drop=True)

        for col, adj in self.adjustments_.items():
            if col in df_X.columns:
                col_vals = df_X[col].values.copy()
                for g, offset in adj.group_offsets.items():
                    mask = (s_prot == g).values
                    if mask.any():
                        col_vals[mask] -= (self.shrinkage_intensity * offset)
                df_X[col] = col_vals

        return df_X

    def predict_proba(
        self,
        X: pd.DataFrame,
        protected_series: pd.Series,
    ) -> np.ndarray:
        """Predict positive class probabilities using group-aware transformed features."""
        if not self.is_fitted_:
            raise ValueError("Model is not fitted")
        X_mitigated = self.transform(X, protected_series)
        return self.base_model_.predict_proba(X_mitigated)

    def predict(
        self,
        X: pd.DataFrame,
        protected_series: pd.Series,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict binary classifications."""
        probs = self.predict_proba(X, protected_series)
        return (probs >= threshold).astype(float)


class ExploratorySurveyWeightedCenteringExtension(ExploratoryGroupAwareCenteringMitigation):
    """Exploratory survey-weighted extension of group-aware centering heuristic.

    Estimates group disparity statistics using survey analysis weights (LONGWT).
    """

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        protected_series: pd.Series,
        sample_weight: pd.Series | None = None,
    ) -> ExploratorySurveyWeightedCenteringExtension:
        self.feature_names_in_ = list(X.columns)
        self.adjustments_.clear()

        s_prot = pd.Series(np.asarray(protected_series)).reset_index(drop=True)
        df_X = X.copy().reset_index(drop=True)

        if sample_weight is not None:
            w = pd.Series(np.asarray(sample_weight, dtype=float)).reset_index(drop=True)
        else:
            w = pd.Series(1.0, index=df_X.index)

        groups = sorted(s_prot.unique())
        total_w = w.sum()

        # Weighted overall mean
        if total_w > 0:
            overall_mean = (df_X.values * w.values[:, np.newaxis]).sum(axis=0) / total_w
            s_overall_mean = pd.Series(overall_mean, index=df_X.columns)
        else:
            s_overall_mean = df_X.mean(axis=0)

        # Compute survey-weighted group offsets for each feature
        for col in df_X.columns:
            group_offsets: dict[Any, float] = {}
            for g in groups:
                mask = (s_prot == g)
                sum_wg = w[mask].sum()
                if sum_wg > 0:
                    g_mean = float((df_X.loc[mask, col].values * w[mask].values).sum() / sum_wg)
                    group_offsets[g] = g_mean - float(s_overall_mean[col])
                else:
                    group_offsets[g] = 0.0

            self.adjustments_[col] = GroupDisparityAdjustment(
                feature=col,
                group_offsets=group_offsets,
            )

        # Transform training data
        X_mitigated = self.transform(df_X, s_prot)

        # Fit base model with survey weights on mitigated training features
        self.base_model_.fit(X_mitigated, y, sample_weight=sample_weight)
        self.is_fitted_ = True
        return self

