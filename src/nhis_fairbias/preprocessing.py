"""Explicit semantic preprocessing and structural-missingness engine for NHIS Gate D2.

Enforces strict temporal split discipline:
- 2022 (development_train) is the ONLY partition permitted to fit preprocessing statistics.
- 2023 (development_validation) and 2024 (frozen_test) apply frozen 2022 statistics.
- Any attempt to fit preprocessing on 2023 or 2024 raises NHISLeakageError.

Predictor families:
- FairBias categorical: categorical_nominal, categorical_binary, ordinal
- FairBias numerical: continuous, count
PRIMARY_CORE: 21 predictors (18 categorical, 3 numerical)
EXPANDED: 24 predictors (20 categorical, 4 numerical)

Missingness handling:
- EMPWRKFT1_A: Structural categorical state using EMPWRKLSW1_A and EMPWRKFT1_A:
    1: 'full-time'
    2: 'part-time'
   -1: 'structural_not_in_universe' (respondents not working last week)
   -2: 'explicit_missing' (unresolved in-universe/nonresponse records)
  Stable sentinels outside substantive NHIS codes: -1 and -2.
- Other categorical/ordinal: explicit missing category sentinel -1 ('explicit_missing').
- Numerical/count: 2022-train-fitted median imputation, then frozen application.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import pathlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .features import (
    DEFAULT_FEATURE_CONFIG,
    NHISFeatureError,
    load_feature_registry,
)

# Explicit sentinels outside substantive NHIS codes (all substantive NHIS codes >= 1)
SENTINEL_STRUCTURAL_NOT_IN_UNIVERSE = -1
SENTINEL_EXPLICIT_MISSING = -2
SENTINEL_CATEGORICAL_MISSING = -1

EMPWRKFT_STATE_MAP = {
    1: "full-time",
    2: "part-time",
    SENTINEL_STRUCTURAL_NOT_IN_UNIVERSE: "structural_not_in_universe",
    SENTINEL_EXPLICIT_MISSING: "explicit_missing",
}

CATEGORICAL_SEMANTIC_TYPES = (
    "categorical_nominal",
    "categorical_binary",
    "ordinal",
)
NUMERICAL_SEMANTIC_TYPES = (
    "continuous",
    "count",
)


class NHISPreprocessingError(ValueError):
    """Raised when preprocessing contract is violated."""


class NHISLeakageError(NHISPreprocessingError):
    """Raised when temporal leakage or forbidden split usage is attempted."""


@dataclasses.dataclass
class PreprocessingFitRecord:
    """Immutable record of fitted preprocessing statistics."""

    fit_year: Union[int, str]
    fit_study_role: str
    row_count: int
    numerical_medians: Dict[str, float]
    numerical_stats: Dict[str, Dict[str, Any]]
    categorical_categories: Dict[str, List[int]]
    empwrkft_distribution: Dict[str, int]
    rules_manifest: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fit_year": self.fit_year,
            "fit_study_role": self.fit_study_role,
            "row_count": self.row_count,
            "numerical_medians": copy.deepcopy(self.numerical_medians),
            "numerical_stats": copy.deepcopy(self.numerical_stats),
            "categorical_categories": copy.deepcopy(self.categorical_categories),
            "empwrkft_distribution": copy.deepcopy(self.empwrkft_distribution),
            "rules_manifest": copy.deepcopy(self.rules_manifest),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PreprocessingFitRecord:
        raw_yr = data["fit_year"]
        yr: Union[int, str] = int(raw_yr) if str(raw_yr).isdigit() else str(raw_yr)
        return cls(
            fit_year=yr,
            fit_study_role=str(data["fit_study_role"]),
            row_count=int(data["row_count"]),
            numerical_medians={k: float(v) for k, v in data["numerical_medians"].items()},
            numerical_stats=copy.deepcopy(data["numerical_stats"]),
            categorical_categories={k: list(v) for k, v in data["categorical_categories"].items()},
            empwrkft_distribution={k: int(v) for k, v in data["empwrkft_distribution"].items()},
            rules_manifest=copy.deepcopy(data["rules_manifest"]),
        )


def construct_empwrkft_series(
    empwrklsw_series: pd.Series,
    empwrkft_series: pd.Series,
    *,
    as_string: bool = False,
) -> pd.Series:
    """
    Construct the 4-state explicit categorical employment intensity variable:
    - 1: 'full-time'
    - 2: 'part-time'
    - -1: 'structural_not_in_universe' (respondent did not work last week: EMPWRKLSW1_A == 2)
    - -2: 'explicit_missing' (unresolved in-universe/nonresponse: EMPWRKLSW1_A in {7,8,9,NA}
          or (EMPWRKLSW1_A == 1 and EMPWRKFT1_A in {7,8,9,NA}))
    """
    lsw = pd.to_numeric(empwrklsw_series, errors="coerce")
    ft = pd.to_numeric(empwrkft_series, errors="coerce")

    n = len(empwrklsw_series)
    codes = np.full(n, SENTINEL_EXPLICIT_MISSING, dtype=int)

    # In universe for working (EMPWRKLSW1_A == 1)
    work_mask = (lsw == 1).fillna(False).to_numpy(dtype=bool)
    not_work_mask = (lsw == 2).fillna(False).to_numpy(dtype=bool)

    # Substantive full-time (ft == 1)
    ft_mask = work_mask & (ft == 1).fillna(False).to_numpy(dtype=bool)
    # Substantive part-time (ft == 2)
    pt_mask = work_mask & (ft == 2).fillna(False).to_numpy(dtype=bool)
    # In-universe but missing intensity
    in_u_missing = work_mask & (~ft_mask) & (~pt_mask)

    codes[ft_mask] = 1
    codes[pt_mask] = 2
    codes[not_work_mask] = SENTINEL_STRUCTURAL_NOT_IN_UNIVERSE
    codes[in_u_missing] = SENTINEL_EXPLICIT_MISSING

    if as_string:
        str_vals = [EMPWRKFT_STATE_MAP[c] for c in codes]
        return pd.Series(str_vals, index=empwrklsw_series.index, name="empwrkft1_a")
    return pd.Series(codes, index=empwrklsw_series.index, dtype=int, name="empwrkft1_a")


class NHISPreprocessor:
    """Explicit semantic preprocessor for NHIS cohorts."""

    def __init__(
        self,
        feature_registry: Optional[Mapping[str, Any]] = None,
        feature_config_path: Optional[Union[str, pathlib.Path]] = None,
    ):
        if feature_registry is not None:
            self.registry = dict(feature_registry)
        else:
            cfg_path = feature_config_path or DEFAULT_FEATURE_CONFIG
            self.registry = load_feature_registry(cfg_path)

        self._validate_registry()
        self._fitted_record: Optional[PreprocessingFitRecord] = None

    def _validate_registry(self) -> None:
        """Validate that all features belong to exactly one family with unambiguous metadata."""
        fl = self.registry["feature_lists"]
        self.primary_core_features = list(fl["primary_core_features"])
        self.expanded_features = self.primary_core_features + list(fl["expanded_utilization_features"])

        # Collect specifications
        self.specs: Dict[str, Dict[str, Any]] = {}
        for k, v in self.registry["primary_core"].items():
            harm_name = v.get("harmonized_name", k.lower())
            self.specs[harm_name] = v
        for k, v in self.registry["expanded_utilization"].items():
            harm_name = v.get("harmonized_name", k.lower())
            self.specs[harm_name] = v

        # Feature family categorization
        self.categorical_features: List[str] = []
        self.numerical_features: List[str] = []

        for name in self.expanded_features:
            if name not in self.specs:
                raise NHISFeatureError(f"Missing feature specification for: {name}")
            sem_type = self.specs[name]["semantic_type"]
            if sem_type in CATEGORICAL_SEMANTIC_TYPES:
                self.categorical_features.append(name)
            elif sem_type in NUMERICAL_SEMANTIC_TYPES:
                self.numerical_features.append(name)
            else:
                raise NHISFeatureError(
                    f"Ambiguous or invalid semantic_type for {name}: {sem_type}"
                )

        # Invariant checks
        primary_cat = [f for f in self.primary_core_features if f in self.categorical_features]
        primary_num = [f for f in self.primary_core_features if f in self.numerical_features]
        if len(self.primary_core_features) != 21:
            raise NHISFeatureError(f"Expected 21 primary features, got {len(self.primary_core_features)}")
        if len(primary_cat) != 18:
            raise NHISFeatureError(f"Expected 18 primary categorical features, got {len(primary_cat)}")
        if len(primary_num) != 3:
            raise NHISFeatureError(f"Expected 3 primary numerical features, got {len(primary_num)}")

        expanded_cat = [f for f in self.expanded_features if f in self.categorical_features]
        expanded_num = [f for f in self.expanded_features if f in self.numerical_features]
        if len(self.expanded_features) != 24:
            raise NHISFeatureError(f"Expected 24 expanded features, got {len(self.expanded_features)}")
        if len(expanded_cat) != 20:
            raise NHISFeatureError(f"Expected 20 expanded categorical features, got {len(expanded_cat)}")
        if len(expanded_num) != 4:
            raise NHISFeatureError(f"Expected 4 expanded numerical features, got {len(expanded_num)}")

    @property
    def is_fitted(self) -> bool:
        return self._fitted_record is not None

    @property
    def fitted_record(self) -> PreprocessingFitRecord:
        if self._fitted_record is None:
            raise NHISPreprocessingError("Preprocessor is not fitted. Call fit() on 2022 train first.")
        return self._fitted_record

    def fit(self, df_train: pd.DataFrame, *, regime: str = "temporal") -> NHISPreprocessor:
        """
        Fit numerical medians and record categories strictly on the training partition.

        Regimes:
        - 'temporal' (Gate D2): strictly fits on 2022 development_train partition.
          Raises NHISLeakageError if called on 2023 or 2024 data.
        - 'pooled' (Gate D3): strictly fits on pooled train split (split_role == 'train').
          Raises NHISLeakageError if validation or test data is present.
        """
        if regime == "temporal":
            # Strict temporal guard
            if "survey_year" in df_train.columns:
                years = set(df_train["survey_year"].unique())
                if years != {2022}:
                    raise NHISLeakageError(
                        f"Preprocessing can ONLY be fit on 2022 development_train partition. "
                        f"Attempted to fit on survey years: {years}"
                    )
            if "study_role" in df_train.columns:
                roles = set(df_train["study_role"].unique())
                if roles != {"development_train"}:
                    raise NHISLeakageError(
                        f"Preprocessing can ONLY be fit on 'development_train' role. "
                        f"Attempted to fit on study roles: {roles}"
                    )
            fit_yr: Union[int, str] = 2022
            fit_role = "development_train"
        elif regime == "pooled":
            # Strict pooled split guard
            if "split_role" in df_train.columns:
                split_roles = set(df_train["split_role"].unique())
                if split_roles != {"train"}:
                    raise NHISLeakageError(
                        f"Pooled preprocessing can ONLY be fit on pooled 'train' split. "
                        f"Attempted to fit on split roles: {split_roles}"
                    )
            fit_yr = "2022-2024_pooled"
            fit_role = "pooled_train"
        else:
            raise ValueError(f"Unknown preprocessing fit regime: {regime!r}. Must be 'temporal' or 'pooled'.")

        # Fit numerical medians
        num_medians: Dict[str, float] = {}
        num_stats: Dict[str, Dict[str, Any]] = {}
        for col in self.numerical_features:
            if col not in df_train.columns:
                # Check uppercase fallback
                raw_col = col.upper()
                if raw_col in df_train.columns:
                    s = pd.to_numeric(df_train[raw_col], errors="coerce")
                else:
                    raise NHISPreprocessingError(f"Numerical feature {col} missing from training frame")
            else:
                s = pd.to_numeric(df_train[col], errors="coerce")

            valid_s = s.dropna()
            if len(valid_s) == 0:
                raise NHISPreprocessingError(f"Numerical feature {col} has no valid training records")
            med_val = float(valid_s.median())
            num_medians[col] = med_val
            num_stats[col] = {
                "median": med_val,
                "mean": float(valid_s.mean()),
                "min": float(valid_s.min()),
                "max": float(valid_s.max()),
                "count_valid": int(len(valid_s)),
                "count_missing": int(s.isna().sum()),
            }

        # Record observed substantive categorical categories
        cat_categories: Dict[str, List[int]] = {}
        for col in self.categorical_features:
            if col == "empwrkft1_a":
                # Will be constructed explicitly
                cat_categories[col] = [
                    1,
                    2,
                    SENTINEL_STRUCTURAL_NOT_IN_UNIVERSE,
                    SENTINEL_EXPLICIT_MISSING,
                ]
                continue

            src_col = col if col in df_train.columns else col.upper()
            if src_col not in df_train.columns:
                raise NHISPreprocessingError(f"Categorical feature {col} missing from training frame")
            s = pd.to_numeric(df_train[src_col], errors="coerce").dropna().astype(int)
            substantive_set = set(self.specs[col]["substantive_codes"])
            observed = sorted([int(x) for x in set(s[s.isin(substantive_set)].unique())])
            cat_categories[col] = observed

        # Structural employment state distribution in 2022
        lsw_col = "empwrklsw1_a" if "empwrklsw1_a" in df_train.columns else "EMPWRKLSW1_A"
        ft_col = "empwrkft1_a" if "empwrkft1_a" in df_train.columns else "EMPWRKFT1_A"
        emp_series = construct_empwrkft_series(df_train[lsw_col], df_train[ft_col])
        emp_dist = {
            EMPWRKFT_STATE_MAP[code]: int((emp_series == code).sum())
            for code in (1, 2, SENTINEL_STRUCTURAL_NOT_IN_UNIVERSE, SENTINEL_EXPLICIT_MISSING)
        }

        if regime == "temporal":
            rules_manifest = {
                "schema_version": "nhis-fairbias-d2-1.0",
                "regime": "temporal",
                "employment_rule": (
                    "Constructed 4-state employment intensity: 1=full-time, 2=part-time, "
                    f"{SENTINEL_STRUCTURAL_NOT_IN_UNIVERSE}=structural_not_in_universe (EMPWRKLSW1_A==2), "
                    f"{SENTINEL_EXPLICIT_MISSING}=explicit_missing (unresolved in-universe/nonresponse)."
                ),
                "categorical_missing_rule": (
                    f"Mapped to explicit missing sentinel {SENTINEL_CATEGORICAL_MISSING} ('explicit_missing')."
                ),
                "numerical_missing_rule": (
                    "Imputed with 2022 development_train median, frozen and applied to 2023 and 2024."
                ),
                "temporal_freeze_rule": (
                    "2024 is frozen_test, evaluation-only; no fitting, tuning, or threshold selection allowed."
                ),
            }
        else:
            rules_manifest = {
                "schema_version": "nhis-fairbias-d3-1.0",
                "regime": "pooled",
                "employment_rule": (
                    "Constructed 4-state employment intensity: 1=full-time, 2=part-time, "
                    f"{SENTINEL_STRUCTURAL_NOT_IN_UNIVERSE}=structural_not_in_universe (EMPWRKLSW1_A==2), "
                    f"{SENTINEL_EXPLICIT_MISSING}=explicit_missing (unresolved in-universe/nonresponse)."
                ),
                "categorical_missing_rule": (
                    f"Mapped to explicit missing sentinel {SENTINEL_CATEGORICAL_MISSING} ('explicit_missing')."
                ),
                "numerical_missing_rule": (
                    "Imputed with pooled train partition median, frozen and applied to validation and test partitions."
                ),
                "data_leakage_rule": (
                    "Validation and test partitions are apply/evaluate only; no fitting, tuning, or threshold selection allowed."
                ),
            }

        self._fitted_record = PreprocessingFitRecord(
            fit_year=fit_yr,
            fit_study_role=fit_role,
            row_count=len(df_train),
            numerical_medians=num_medians,
            numerical_stats=num_stats,
            categorical_categories=cat_categories,
            empwrkft_distribution=emp_dist,
            rules_manifest=rules_manifest,
        )
        return self

    def transform(
        self,
        df: pd.DataFrame,
        *,
        feature_set: str = "primary_core",
        preserve_metadata: bool = True,
    ) -> pd.DataFrame:
        """
        Transform any partition (2022 train, 2023 val, 2024 test) using 2022-fitted statistics.
        """
        if not self.is_fitted:
            raise NHISPreprocessingError("Cannot transform before fitting. Call fit() first.")

        if feature_set == "primary_core":
            active_features = list(self.primary_core_features)
        elif feature_set in ("expanded", "expanded_utilization"):
            active_features = list(self.expanded_features)
        else:
            raise NHISPreprocessingError(f"Unknown feature_set: {feature_set}")

        record = self.fitted_record
        out = pd.DataFrame(index=df.index)

        # 1. Transform employment structural missingness
        lsw_col = "empwrklsw1_a" if "empwrklsw1_a" in df.columns else "EMPWRKLSW1_A"
        ft_col = "empwrkft1_a" if "empwrkft1_a" in df.columns else "EMPWRKFT1_A"
        if lsw_col in df.columns and ft_col in df.columns:
            emp_series = construct_empwrkft_series(df[lsw_col], df[ft_col])
        else:
            raise NHISPreprocessingError("Employment variables missing for structural state construction")

        for feat in active_features:
            if feat == "empwrkft1_a":
                out["empwrkft1_a"] = emp_series.copy()
                continue

            src_col = feat if feat in df.columns else feat.upper()
            if src_col not in df.columns:
                raise NHISPreprocessingError(f"Required feature {feat} not in dataframe")

            s = df[src_col]
            sem_type = self.specs[feat]["semantic_type"]

            if sem_type in NUMERICAL_SEMANTIC_TYPES:
                # Numerical: convert to float, impute with 2022 train median
                s_num = pd.to_numeric(s, errors="coerce").astype(float)
                # Check substantive codes/bounds if defined
                substantive_codes = self.specs[feat].get("substantive_codes", [])
                if substantive_codes:
                    min_c, max_c = min(substantive_codes), max(substantive_codes)
                    invalid_mask = s_num.notna() & ((s_num < min_c) | (s_num > max_c))
                    s_num.loc[invalid_mask] = np.nan
                train_med = record.numerical_medians[feat]
                out[feat] = s_num.fillna(train_med).astype(float)

            elif sem_type in CATEGORICAL_SEMANTIC_TYPES:
                # Categorical/ordinal: map missing to explicit sentinel -1
                s_code = pd.to_numeric(s, errors="coerce")
                substantive_set = set(self.specs[feat]["substantive_codes"])
                # Non-substantive codes -> sentinel -1
                is_substantive = s_code.isin(substantive_set)
                res = pd.Series(SENTINEL_CATEGORICAL_MISSING, index=df.index, dtype=int)
                res.loc[is_substantive] = s_code[is_substantive].astype(int)
                out[feat] = res

        # Optional metadata preservation
        if preserve_metadata:
            meta_cols = ["survey_year", "study_role", "WTFA_A", "PSTRAT", "PPSU", "WTFA_DEV"]
            for mc in meta_cols:
                if mc in df.columns:
                    out[mc] = df[mc].copy()

        return out

    def get_feature_family_lists(
        self, feature_set: str = "primary_core"
    ) -> Tuple[List[str], List[str]]:
        """Return (categorical_columns, numerical_columns) for the chosen feature set."""
        if feature_set == "primary_core":
            feats = self.primary_core_features
        elif feature_set in ("expanded", "expanded_utilization"):
            feats = self.expanded_features
        else:
            raise NHISPreprocessingError(f"Unknown feature_set: {feature_set}")

        cats = [f for f in feats if f in self.categorical_features]
        nums = [f for f in feats if f in self.numerical_features]
        return cats, nums

    def export_fit_json(self, path: Union[str, pathlib.Path]) -> None:
        """Serialize 2022-fitted statistics to JSON."""
        path_obj = pathlib.Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("w", encoding="utf-8") as handle:
            json.dump(self.fitted_record.to_dict(), handle, indent=2)

    def load_fit_json(self, path: Union[str, pathlib.Path]) -> NHISPreprocessor:
        """Load fitted statistics from JSON."""
        path_obj = pathlib.Path(path)
        with path_obj.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        self._fitted_record = PreprocessingFitRecord.from_dict(data)
        return self
