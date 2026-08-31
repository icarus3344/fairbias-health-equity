"""Survey-weighted Platt scaling calibration and capacity-aware threshold freezing."""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def find_weighted_capacity_threshold(
    probs: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    target_capacity: float = 0.10,
) -> float:
    """Find the continuous probability threshold corresponding to top K% cumulative weighted population.

    Parameters:
        probs: Predicted probability array.
        weights: Survey analysis weights (LONGWT).
        target_capacity: Target fraction of weighted population (e.g. 0.10 for 10%).

    Returns:
        Continuous probability threshold float.
    """
    p_arr = np.asarray(probs, dtype=float)
    w_arr = np.asarray(weights, dtype=float)

    if len(p_arr) == 0:
        return 0.5

    # Sort descending by probability
    sort_idx = np.argsort(-p_arr)
    p_sorted = p_arr[sort_idx]
    w_sorted = w_arr[sort_idx]

    cum_w = np.cumsum(w_sorted)
    tot_w = cum_w[-1]
    if tot_w <= 0:
        return float(np.percentile(p_arr, 100 * (1 - target_capacity)))

    target_w = target_capacity * tot_w
    idx = np.searchsorted(cum_w, target_w)
    idx = min(idx, len(p_sorted) - 1)
    return float(p_sorted[idx])


class SurveyWeightedPlattCalibrator:
    """Survey-weighted Platt scaling logistic calibrator fit on calibration set."""

    def __init__(self, random_state: int = 20260828) -> None:
        self.random_state = random_state
        self.calibrator_ = LogisticRegression(
            penalty="l2",
            C=10.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state,
        )
        self.frozen_threshold_10pct_: float | None = None
        self.is_fitted_: bool = False

    def _to_log_odds(self, probs: np.ndarray) -> np.ndarray:
        eps = 1e-7
        p_clipped = np.clip(probs, eps, 1.0 - eps)
        return np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)

    def fit(
        self,
        raw_probs: Sequence[float] | np.ndarray,
        y: Sequence[float] | np.ndarray,
        sample_weight: Sequence[float] | np.ndarray | None = None,
        freeze_capacity_fraction: float = 0.10,
    ) -> SurveyWeightedPlattCalibrator:
        """Fit univariate logistic regression on raw predicted log-odds with survey weights."""
        p_arr = np.asarray(raw_probs, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        w_arr = np.asarray(sample_weight, dtype=float).ravel() if sample_weight is not None else None

        # Check that both classes exist in calibration partition
        unique_y = np.unique(y_arr)
        if len(unique_y) < 2:
            # Fallback if partition lacks variation (rare/synthetic)
            self.is_fitted_ = True
            self.frozen_threshold_10pct_ = find_weighted_capacity_threshold(
                p_arr,
                w_arr if w_arr is not None else np.ones_like(p_arr),
                target_capacity=freeze_capacity_fraction,
            )
            return self

        X_logit = self._to_log_odds(p_arr)
        self.calibrator_.fit(X_logit, y_arr, sample_weight=w_arr)
        self.is_fitted_ = True

        # Predict calibrated probabilities on calibration set to freeze 10% threshold
        cal_probs = self.predict_proba(p_arr)
        self.frozen_threshold_10pct_ = find_weighted_capacity_threshold(
            cal_probs,
            w_arr if w_arr is not None else np.ones_like(cal_probs),
            target_capacity=freeze_capacity_fraction,
        )
        return self

    def predict_proba(self, raw_probs: Sequence[float] | np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities."""
        if not self.is_fitted_:
            raise ValueError("Calibrator is not fitted")
        p_arr = np.asarray(raw_probs, dtype=float)
        if len(self.calibrator_.classes_) < 2:
            return p_arr
        X_logit = self._to_log_odds(p_arr)
        probs = self.calibrator_.predict_proba(X_logit)
        if probs.shape[1] >= 2:
            return probs[:, 1]
        return probs[:, 0]
