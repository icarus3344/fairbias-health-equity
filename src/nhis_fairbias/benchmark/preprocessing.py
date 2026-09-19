"""3-tier semantic, geometric, and one-hot preprocessing engine for NHIS Benchmark V1.

Implements:
1. Tier 1 (Semantic Layer):
   - Missing / non-substantive codes mapped to explicit token 'MISSING'.
2. Tier 2 (Geometric Layer):
   - Continuous features (agep_a, pcnt18uptc, pcntlt18tc) imputed with medians fit on F
     and scaled via MinMaxScaler fit on F.
3. Tier 3 (Prediction / One-Hot Layer):
   - Categories frozen on Partition F with explicit reserved tokens:
     V_feature = Substantive_Levels(F) union {'MISSING', 'UNKNOWN'}
   - In C, S, and T, unseen levels are mapped to 'UNKNOWN'.
   - Frozen OneHotEncoder applied across all partitions.
4. Strict anti-leakage:
   - Fitting is permitted ONLY on Partition F.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

NUMERICAL_FEATURES = ("agep_a", "pcnt18uptc", "pcntlt18tc")


class BenchmarkPreprocessor:
    """Strict 3-tier preprocessor enforcing zero leakage and reserved unknown categories."""

    def __init__(
        self,
        feature_names: Sequence[str],
        numerical_features: Optional[Sequence[str]] = None,
    ):
        self.feature_names = tuple(feature_names)
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must not contain duplicates")
        declared_num = (
            tuple(f for f in NUMERICAL_FEATURES if f in self.feature_names)
            if numerical_features is None
            else tuple(numerical_features)
        )
        if len(set(declared_num)) != len(declared_num):
            raise ValueError("numerical_features must not contain duplicates")
        unknown_num = set(declared_num) - set(self.feature_names)
        if unknown_num:
            raise ValueError(f"numerical_features not present in feature_names: {unknown_num}")
        self.numerical_features = tuple(declared_num)
        self._declared_num_features = tuple(f for f in self.feature_names if f in self.numerical_features)
        self.num_features = self._declared_num_features
        self.cat_features = tuple(f for f in self.feature_names if f not in self.numerical_features)

        self.fitted_: bool = False
        self.num_medians_: Dict[str, float] = {}
        self.dropped_features_: List[str] = []
        self.scaler_: Optional[MinMaxScaler] = None
        self.cat_vocabularies_: Dict[str, List[str]] = {}
        self.one_hot_encoder_: Optional[OneHotEncoder] = None
        self.transformed_feature_names_: List[str] = []

    def _reset_fit_state(self) -> None:
        self.fitted_ = False
        self.num_medians_ = {}
        self.dropped_features_ = []
        self.scaler_ = None
        self.cat_vocabularies_ = {}
        self.one_hot_encoder_ = None
        self.transformed_feature_names_ = []

    def _validate_columns(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("BenchmarkPreprocessor expects a pandas DataFrame")
        if bool(X.columns.duplicated().any()):
            duplicated = X.columns[X.columns.duplicated()].tolist()
            raise ValueError(f"X contains duplicate feature columns: {duplicated}")
        missing_cols = [col for col in self.feature_names if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required features in X: {missing_cols}")

    @staticmethod
    def _categorical_strings(series: pd.Series) -> pd.Series:
        """Map actual missing cells to MISSING without stringifying them first."""
        return series.map(lambda value: "MISSING" if pd.isna(value) else str(value))

    @staticmethod
    def _finite_numeric(series: pd.Series, col: str) -> pd.Series:
        vals = pd.to_numeric(series, errors="coerce")
        original_nonmissing = ~series.isna()
        if np.any(~np.isfinite(vals[original_nonmissing].to_numpy(dtype=float))):
            raise ValueError(f"Numerical feature {col!r} contains non-finite values")
        return vals

    def fit(self, X: pd.DataFrame) -> BenchmarkPreprocessor:
        """Fit preprocessor statistics exclusively on Partition F."""
        self._reset_fit_state()
        self._validate_columns(X)

        # Tier 2: Numerical statistics
        X_num = pd.DataFrame(index=X.index)
        active_num_features = []
        for col in self._declared_num_features:
            vals = self._finite_numeric(X[col], col)
            if vals.notna().sum() == 0:
                self.dropped_features_.append(col)
                continue
            med = float(vals.median())
            self.num_medians_[col] = med
            X_num[col] = vals.fillna(med)
            active_num_features.append(col)
        self.num_features = tuple(active_num_features)

        self.scaler_ = MinMaxScaler() if self.num_features else None
        if len(self.num_features) > 0:
            self.scaler_.fit(X_num[list(self.num_features)])

        # Tier 3: Categorical vocabulary with reserved 'MISSING' and 'UNKNOWN' tokens
        self.cat_vocabularies_ = {}
        categories_for_ohe = []
        X_cat_str = pd.DataFrame(index=X.index)

        for col in self.cat_features:
            raw_vals = self._categorical_strings(X[col])
            # Find unique substantive levels observed on F
            unique_levels = sorted({v for v in raw_vals.unique() if v not in ("MISSING", "UNKNOWN")})
            # Full vocabulary includes explicit 'MISSING' and reserved 'UNKNOWN'
            vocab = unique_levels + ["MISSING", "UNKNOWN"]
            self.cat_vocabularies_[col] = vocab
            categories_for_ohe.append(vocab)

            X_cat_str[col] = raw_vals

        if len(self.cat_features) > 0:
            self.one_hot_encoder_ = OneHotEncoder(
                categories=categories_for_ohe,
                handle_unknown="ignore",
                sparse_output=False,
            )
            self.one_hot_encoder_.fit(X_cat_str[list(self.cat_features)])

        # Record transformed column names
        feat_names = list(self.num_features)
        if self.one_hot_encoder_ is not None and len(self.cat_features) > 0:
            feat_names.extend(self.one_hot_encoder_.get_feature_names_out(list(self.cat_features)))
        self.transformed_feature_names_ = feat_names

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using frozen statistics learned on Partition F."""
        if not self.fitted_:
            raise RuntimeError("BenchmarkPreprocessor must be fitted before transforming.")
        self._validate_columns(X)

        # Numerical Tier
        X_num = pd.DataFrame(index=X.index)
        for col in self.num_features:
            vals = self._finite_numeric(X[col], col)
            med = self.num_medians_[col]
            X_num[col] = vals.fillna(med)

        if len(self.num_features) > 0 and self.scaler_ is not None:
            num_scaled = self.scaler_.transform(X_num[list(self.num_features)])
        else:
            num_scaled = np.empty((len(X), 0))

        # Categorical Tier: Map unobserved levels to 'UNKNOWN'
        if len(self.cat_features) > 0 and self.one_hot_encoder_ is not None:
            X_cat_str = pd.DataFrame(index=X.index)
            for col in self.cat_features:
                raw_vals = self._categorical_strings(X[col])
                vocab_set = set(self.cat_vocabularies_[col])
                # If level is in vocab, retain it; otherwise map to 'UNKNOWN'
                X_cat_str[col] = raw_vals.apply(lambda v: v if v in vocab_set else "UNKNOWN")

            cat_ohe = self.one_hot_encoder_.transform(X_cat_str[list(self.cat_features)])
        else:
            cat_ohe = np.empty((len(X), 0))

        return np.hstack([num_scaled, cat_ohe])

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)
