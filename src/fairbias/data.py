"""Data loading, explicit schema management, and leakage-free categorical encoding.

Encoding contract (leakage prevention): ``prepare_data`` returns RAW,
unencoded frames.  The pipeline first splits the raw data into
train/validation/test partitions, then ``fit_encoders`` is called on the
TRAINING partition only, and ``transform_partition`` applies the fitted
encoders to every partition.  Categories unseen in training receive new,
deterministic codes appended after the training codes; median
binarizations (continuous targets, high-cardinality protected columns)
use the training median.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from fairbias.config import FairBiasConfig


# Explicit schema definitions for standard benchmark datasets
DATASET_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "compas": {
        "file_name": "data_COMPAS.csv",
        "target": "two_year_recid",
        "protected": ["sex"],
        "categorical": ["sex", "race", "c_charge_degree", "age_cat", "score_text"],
        "numerical": [
            "age",
            "priors_count",
            "juv_fel_count",
            "juv_misd_count",
            "juv_other_count",
            "decile_score",
            "days_b_screening_arrest",
        ],
        "excluded": [
            "id",
            "name",
            "first",
            "last",
            "compas_screening_date",
            "dob",
            "c_jail_in",
            "c_jail_out",
            "c_offense_date",
            "screening_date",
            "v_screening_date",
            "in_custody",
            "out_custody",
            "c_days_from_compas",
            "type_of_assessment",
            "v_type_of_assessment",
            "v_decile_score",
            "v_score_text",
        ],
    },
    "credit": {
        "file_name": "data_Credit_Card.csv",
        "target": "default payment next month",
        "protected": ["SEX"],
        "categorical": [
            "SEX",
            "EDUCATION",
            "MARRIAGE",
            "PAY_0",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
        ],
        "numerical": [
            "LIMIT_BAL",
            "AGE",
            "BILL_AMT1",
            "BILL_AMT2",
            "BILL_AMT3",
            "BILL_AMT4",
            "BILL_AMT5",
            "BILL_AMT6",
            "PAY_AMT1",
            "PAY_AMT2",
            "PAY_AMT3",
            "PAY_AMT4",
            "PAY_AMT5",
            "PAY_AMT6",
        ],
        "excluded": ["ID"],
    },
}


class FairDataLoader:
    """Robust data loader with explicit column semantics and train-only fitted encoders."""

    def __init__(self, config: Optional[FairBiasConfig] = None):
        self.config = config or FairBiasConfig.compas_default()
        self.raw_df: Optional[pd.DataFrame] = None
        self.categorical_columns: List[str] = []
        self.numerical_columns: List[str] = []
        self.excluded_columns: List[str] = []
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.encoding_mappings: Dict[str, Dict[str, int]] = {}
        # Train-fitted state (populated by fit_encoders)
        self._y_median: Optional[float] = None
        self._o_medians: Dict[str, float] = {}
        self._fitted: bool = False

    def load_raw(self, path: Optional[str] = None) -> pd.DataFrame:
        """Load raw dataset from filesystem."""
        data_path = path or self.config.dataset_path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at path: {data_path}")

        if data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
        elif data_path.endswith(".parquet"):
            df = pd.read_parquet(data_path)
        elif data_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(data_path)
        elif data_path.endswith(".json"):
            df = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        self.raw_df = df.copy()
        return df

    def get_schema(self) -> Dict[str, Any]:
        """Retrieve explicit or inferred schema."""
        ds_name = self.config.dataset_name.lower()
        if ds_name in DATASET_SCHEMAS:
            return DATASET_SCHEMAS[ds_name]

        # Inferred schema if not in standard registry
        assert self.raw_df is not None, "Load raw data before inferring schema"
        all_cols = list(self.raw_df.columns)
        target = self.config.label_Y
        protected = list(self.config.label_O)

        cat_cols = []
        num_cols = []
        for col in all_cols:
            if col == target or col in protected:
                continue
            s_col = self.raw_df[col]
            if (
                pd.api.types.is_numeric_dtype(s_col)
                and not pd.api.types.is_bool_dtype(s_col)
                and (pd.api.types.is_float_dtype(s_col) or s_col.nunique() > 5 or (len(s_col) <= 10 and s_col.nunique() > 2))
            ):
                num_cols.append(col)
            else:
                cat_cols.append(col)
        return {
            "target": target,
            "protected": protected,
            "categorical": cat_cols,
            "numerical": num_cols,
            "excluded": [],
        }

    def prepare_data(
        self, df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str], List[str]]:
        """
        Process the dataset into RAW (unencoded) feature matrix, target, and
        protected attributes.  Encoding is deliberately deferred so that it
        can be fitted on the training partition only (see ``fit_encoders``).

        Returns:
            X: Raw feature matrix (excluding target Y and protected attributes O).
            Y: Target Series (numeric binary passthrough; other encodings deferred).
            O: Raw protected-attribute DataFrame (encoding/binarization deferred).
            categorical_cols: List of categorical feature names in X.
            numerical_cols: List of numerical feature names in X.
        """
        if df is None:
            if self.raw_df is None:
                self.load_raw()
            df = self.raw_df.copy()
        else:
            df = df.copy()
            self.raw_df = df

        schema = self.get_schema()
        target_col = schema["target"]
        protected_cols = schema["protected"]
        excluded_cols = [col for col in schema.get("excluded", []) if col in df.columns]

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        for p_col in protected_cols:
            if p_col not in df.columns:
                raise ValueError(f"Protected column '{p_col}' not found in dataset")

        # Drop explicit excluded columns
        df = df.drop(columns=excluded_cols, errors="ignore")

        # Target Y: numeric {0,1} passes through; anything else is deferred to
        # the train-fitted encoders (no full-data statistics here).
        y_raw = df[target_col]
        if pd.api.types.is_numeric_dtype(y_raw):
            unique_vals = set(pd.unique(y_raw.dropna()))
            if unique_vals.issubset({0, 1}):
                Y = pd.Series(pd.to_numeric(y_raw, errors="coerce").fillna(0).astype(int), index=df.index, name=target_col)
            else:
                # Continuous target: median binarization deferred to fit_encoders
                Y = pd.Series(pd.to_numeric(y_raw, errors="coerce"), index=df.index, name=target_col)
        else:
            Y = y_raw.copy()
            Y.name = target_col

        # Protected attributes O: raw values; encoding/binarization deferred
        O = df[protected_cols].copy()

        # Feature matrix X: excludes target Y and protected attributes O
        cols_to_drop = [target_col] + [c for c in protected_cols if c in df.columns]
        X = df.drop(columns=cols_to_drop, errors="ignore").copy()

        # Identify categorical and numerical columns in X
        cat_schema = set(schema.get("categorical", []))
        num_schema = set(schema.get("numerical", []))

        categorical_cols: List[str] = []
        numerical_cols: List[str] = []

        for col in X.columns:
            if col in cat_schema:
                categorical_cols.append(col)
            elif col in num_schema:
                numerical_cols.append(col)
            elif (
                pd.api.types.is_numeric_dtype(X[col])
                and not pd.api.types.is_bool_dtype(X[col])
                and (pd.api.types.is_float_dtype(X[col]) or X[col].nunique() > 5 or (len(X[col]) <= 10 and X[col].nunique() > 2))
            ):
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)

        self.categorical_columns = categorical_cols
        self.numerical_columns = numerical_cols
        self.excluded_columns = excluded_cols

        return X, Y, O, categorical_cols, numerical_cols

    # ------------------------------------------------------------------
    # Train-only fitted encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _is_binary_numeric(s: pd.Series) -> bool:
        return (
            pd.api.types.is_numeric_dtype(s)
            and set(pd.unique(s.dropna())).issubset({0, 1})
        )

    def fit_encoders(
        self,
        X_train: pd.DataFrame,
        Y_train: pd.Series,
        O_train: pd.DataFrame,
    ) -> None:
        """
        Fit all encoders/binarizers STRICTLY on the training partition.

        - Non-numeric categorical X columns: LabelEncoder over the training
          categories.
        - String target Y: LabelEncoder over the training classes.
        - Continuous numeric target Y: training-median binarization threshold.
        - Non-numeric protected columns: LabelEncoder over training groups.
        - High-cardinality numeric protected columns (>5 values): training-
          median binarization threshold.
        """
        self.label_encoders = {}
        self.encoding_mappings = {}
        self._y_median = None
        self._o_medians = {}

        for col in self.categorical_columns:
            if col not in X_train.columns:
                continue
            if not pd.api.types.is_numeric_dtype(X_train[col]):
                le = LabelEncoder()
                le.fit(X_train[col].astype(str))
                self.label_encoders[col] = le
                self.encoding_mappings[col] = {
                    str(cls_): int(idx) for idx, cls_ in enumerate(le.classes_)
                }

        # Target Y
        if not pd.api.types.is_numeric_dtype(Y_train):
            le_y = LabelEncoder()
            le_y.fit(Y_train.astype(str))
            self.label_encoders[Y_train.name] = le_y
            self.encoding_mappings[Y_train.name] = {
                str(cls_): int(idx) for idx, cls_ in enumerate(le_y.classes_)
            }
        elif not self._is_binary_numeric(Y_train):
            self._y_median = float(pd.to_numeric(Y_train, errors="coerce").median())

        # Protected attributes O
        for p_col in O_train.columns:
            if not pd.api.types.is_numeric_dtype(O_train[p_col]):
                le_p = LabelEncoder()
                le_p.fit(O_train[p_col].astype(str))
                self.label_encoders[p_col] = le_p
                self.encoding_mappings[p_col] = {
                    str(cls_): int(idx) for idx, cls_ in enumerate(le_p.classes_)
                }
            elif O_train[p_col].nunique() > 5:
                self._o_medians[p_col] = float(O_train[p_col].median())

        self._fitted = True

    def _encode_series(
        self,
        s: pd.Series,
        encoder: LabelEncoder,
        unseen_sentinel: Optional[int] = None,
    ) -> pd.Series:
        """Apply a train-fitted encoder; unseen categories get deterministic
        new codes appended after the training codes (or the sentinel)."""
        classes = list(encoder.classes_)
        code_map = {cls_: idx for idx, cls_ in enumerate(classes)}
        str_vals = s.astype(str)

        unseen = sorted(set(str_vals.dropna()) - set(classes))
        next_code = len(classes)
        for u in unseen:
            if unseen_sentinel is not None:
                code_map[u] = unseen_sentinel
            else:
                code_map[u] = next_code
                next_code += 1

        return str_vals.map(code_map).astype(int)

    def transform_partition(
        self,
        X: pd.DataFrame,
        Y: pd.Series,
        O: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Apply the train-fitted encoders to one partition (train, validation,
        or test).  Must be called after ``fit_encoders``.
        """
        if not self._fitted:
            raise RuntimeError("fit_encoders must be called on the training partition first")

        X_out = X.copy()
        for col in self.categorical_columns:
            if col not in X_out.columns:
                continue
            if col in self.label_encoders:
                X_out[col] = self._encode_series(X_out[col], self.label_encoders[col])
            else:
                # Numeric categorical column: pass codes through unchanged
                X_out[col] = pd.to_numeric(X_out[col], errors="coerce").fillna(0).astype(int)

        for col in self.numerical_columns:
            if col in X_out.columns:
                X_out[col] = pd.to_numeric(X_out[col], errors="coerce").fillna(0.0).astype(float)

        # Target Y
        if Y.name in self.label_encoders:
            Y_out = self._encode_series(Y, self.label_encoders[Y.name], unseen_sentinel=-1)
            Y_out.name = Y.name
        elif self._y_median is not None:
            Y_out = pd.Series(
                (pd.to_numeric(Y, errors="coerce") > self._y_median).astype(int),
                index=Y.index, name=Y.name,
            )
        else:
            Y_out = pd.Series(
                pd.to_numeric(Y, errors="coerce").fillna(0).astype(int),
                index=Y.index, name=Y.name,
            )

        # Protected attributes O
        O_out = O.copy()
        for p_col in O_out.columns:
            if p_col in self.label_encoders:
                O_out[p_col] = self._encode_series(O_out[p_col], self.label_encoders[p_col])
            elif p_col in self._o_medians:
                O_out[p_col] = (O_out[p_col] > self._o_medians[p_col]).astype(int)
            else:
                O_out[p_col] = pd.to_numeric(O_out[p_col], errors="coerce").fillna(0).astype(int)

        return X_out, Y_out, O_out
