"""Predictive classification models supporting complex survey analysis weights."""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class WeightedLogisticClassifier:
    """Survey-weighted Logistic Regression with L2 regularization."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 10000,
        random_state: int = 20260828,
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.model_ = LogisticRegression(
            penalty="l2",
            C=self.C,
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.is_fitted_: bool = False

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        sample_weight: pd.Series | np.ndarray | None = None,
    ) -> WeightedLogisticClassifier:
        """Fit model with optional sample weights."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        w_arr = np.asarray(sample_weight, dtype=float).ravel() if sample_weight is not None else None

        self.model_.fit(X_arr, y_arr, sample_weight=w_arr)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict positive class probabilities (1D array)."""
        if not self.is_fitted_:
            raise ValueError("Model is not fitted")
        X_arr = np.asarray(X, dtype=float)
        probs = self.model_.predict_proba(X_arr)
        if probs.shape[1] >= 2:
            return probs[:, 1]
        return probs[:, 0]

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary classifications at specified threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(float)


class WeightedRandomForestClassifier:
    """Survey-weighted Random Forest Classifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        min_samples_leaf: int = 10,
        random_state: int = 20260828,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.is_fitted_: bool = False

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        sample_weight: pd.Series | np.ndarray | None = None,
    ) -> WeightedRandomForestClassifier:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        w_arr = np.asarray(sample_weight, dtype=float).ravel() if sample_weight is not None else None

        self.model_.fit(X_arr, y_arr, sample_weight=w_arr)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("Model is not fitted")
        X_arr = np.asarray(X, dtype=float)
        probs = self.model_.predict_proba(X_arr)
        if probs.shape[1] >= 2:
            return probs[:, 1]
        return probs[:, 0]

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(float)


class WeightedGradientBoostingClassifier:
    """Survey-weighted Histogram Gradient Boosting Classifier."""

    def __init__(
        self,
        max_iter: int = 100,
        max_depth: int = 4,
        min_samples_leaf: int = 20,
        random_state: int = 20260828,
    ) -> None:
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model_ = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.is_fitted_: bool = False

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        sample_weight: pd.Series | np.ndarray | None = None,
    ) -> WeightedGradientBoostingClassifier:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        w_arr = np.asarray(sample_weight, dtype=float).ravel() if sample_weight is not None else None

        self.model_.fit(X_arr, y_arr, sample_weight=w_arr)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("Model is not fitted")
        X_arr = np.asarray(X, dtype=float)
        probs = self.model_.predict_proba(X_arr)
        if probs.shape[1] >= 2:
            return probs[:, 1]
        return probs[:, 0]

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(float)
