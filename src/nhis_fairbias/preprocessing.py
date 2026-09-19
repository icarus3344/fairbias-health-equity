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
import hashlib
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
        row_cnt = data["row_count"]
        if isinstance(row_cnt, bool) or not isinstance(row_cnt, int) or row_cnt <= 0:
            raise ValueError(f"row_count must be positive integer, got {row_cnt!r} ({type(row_cnt).__name__})")
        return cls(
            fit_year=yr,
            fit_study_role=str(data["fit_study_role"]),
            row_count=row_cnt,
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
            self.registry = copy.deepcopy(dict(feature_registry))
        else:
            cfg_path = feature_config_path or DEFAULT_FEATURE_CONFIG
            self.registry = copy.deepcopy(load_feature_registry(cfg_path))

        self._validate_registry()
        self._fitted_record: Optional[PreprocessingFitRecord] = None
        self._frozen_specs: Optional[Dict[str, Any]] = None
        self._fitted_rules_hash: Optional[str] = None

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

        self._frozen_primary_core_features = list(self.primary_core_features)
        self._frozen_expanded_features = list(self.expanded_features)

    @property
    def is_fitted(self) -> bool:
        return self._fitted_record is not None

    @property
    def fitted_record(self) -> PreprocessingFitRecord:
        if self._fitted_record is None:
            raise NHISPreprocessingError("Preprocessor is not fitted. Call fit() on 2022 train first.")
        return copy.deepcopy(self._fitted_record)

    def fit(
        self,
        df_train: pd.DataFrame,
        *,
        regime: str = "temporal",
        allow_unspecified_source: bool = False,
    ) -> NHISPreprocessor:
        """
        Fit numerical medians and record categories strictly on the training partition.

        Regimes:
        - 'temporal' (Gate D2): strictly fits on 2022 development_train partition.
          Raises NHISLeakageError if called on 2023 or 2024 data, or if provenance metadata is missing.
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
                fit_yr: Union[int, str] = 2022
            elif allow_unspecified_source:
                fit_yr = "unspecified"
            else:
                raise NHISLeakageError(
                    "Preprocessing temporal regime requires 'survey_year' column with value 2022. "
                    "Missing provenance metadata in production training cohort."
                )

            if "study_role" in df_train.columns:
                roles = set(df_train["study_role"].unique())
                if roles != {"development_train"}:
                    raise NHISLeakageError(
                        f"Preprocessing can ONLY be fit on 'development_train' role. "
                        f"Attempted to fit on study roles: {roles}"
                    )
                fit_role = "development_train"
            elif allow_unspecified_source:
                fit_role = "unspecified"
            else:
                raise NHISLeakageError(
                    "Preprocessing temporal regime requires 'study_role' column with value 'development_train'. "
                    "Missing provenance metadata in production training cohort."
                )
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
            src_col = col if col in df_train.columns else col.upper()
            if src_col not in df_train.columns:
                raise NHISPreprocessingError(f"Numerical feature {col} missing from training frame")

            s_raw = pd.to_numeric(df_train[src_col], errors="coerce").astype(float)
            s_clean = s_raw.replace([np.inf, -np.inf], np.nan)
            substantive_codes = self.specs[col].get("substantive_codes", [])
            if substantive_codes:
                min_c, max_c = min(substantive_codes), max(substantive_codes)
                invalid_mask = s_clean.notna() & ((s_clean < min_c) | (s_clean > max_c))
                s_clean.loc[invalid_mask] = np.nan

            valid_s = s_clean.dropna()
            if len(valid_s) == 0:
                raise NHISPreprocessingError(f"Numerical feature {col} has no valid training records")
            med_val = float(valid_s.median())
            if np.isnan(med_val) or np.isinf(med_val):
                raise NHISPreprocessingError(f"Calculated non-finite median for {col}: {med_val}")
            if substantive_codes:
                min_c, max_c = min(substantive_codes), max(substantive_codes)
                if med_val < min_c or med_val > max_c:
                    raise NHISPreprocessingError(
                        f"Calculated median for {col} ({med_val}) is outside substantive range [{min_c}, {max_c}]"
                    )
            num_medians[col] = med_val
            num_stats[col] = {
                "median": med_val,
                "mean": float(valid_s.mean()),
                "min": float(valid_s.min()),
                "max": float(valid_s.max()),
                "count_valid": int(len(valid_s)),
                "count_missing": int(s_clean.isna().sum()),
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
            is_unverified = (fit_yr == "unspecified" or fit_role == "unspecified")
            if is_unverified:
                rules_manifest["regime"] = "unspecified"
                rules_manifest["schema_version"] = "nhis-fairbias-unspecified-1.0"
                rules_manifest["provenance_status"] = "unverified_missing_provenance"
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
            is_unverified = False

        self.is_legacy_unverified = is_unverified
        self._frozen_specs = copy.deepcopy(self.specs)
        self._fitted_rules_hash = json.dumps(self._frozen_specs, sort_keys=True, default=str)
        self._frozen_numerical_medians = copy.deepcopy(num_medians)
        self._frozen_primary_core_features = list(self.primary_core_features)
        self._frozen_expanded_features = list(self.expanded_features)
        self._frozen_categorical_features = list(self.categorical_features)
        self._frozen_numerical_features = list(self.numerical_features)

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

        current_hash = json.dumps(self.specs, sort_keys=True, default=str)
        if (
            self._frozen_specs is None
            or self._fitted_rules_hash is None
            or current_hash != self._fitted_rules_hash
            or self.specs != self._frozen_specs
        ):
            raise ValueError(
                "Registry rules were mutated after fit. Transforming with mutated rules is prohibited."
            )
        active_specs = self._frozen_specs

        if feature_set == "primary_core":
            active_features = list(getattr(self, "_frozen_primary_core_features", self.primary_core_features))
        elif feature_set in ("expanded", "expanded_utilization"):
            active_features = list(getattr(self, "_frozen_expanded_features", self.expanded_features))
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
            sem_type = active_specs[feat]["semantic_type"]

            if sem_type in NUMERICAL_SEMANTIC_TYPES:
                # Numerical: convert to float, impute with 2022 train median
                s_num = pd.to_numeric(s, errors="coerce").astype(float)
                # Check substantive codes/bounds if defined
                substantive_codes = active_specs[feat].get("substantive_codes", [])
                if substantive_codes:
                    min_c, max_c = min(substantive_codes), max(substantive_codes)
                    invalid_mask = s_num.notna() & ((s_num < min_c) | (s_num > max_c))
                    s_num.loc[invalid_mask] = np.nan
                frozen_meds = getattr(self, "_frozen_numerical_medians", None)
                train_med = frozen_meds[feat] if frozen_meds is not None else record.numerical_medians[feat]
                out[feat] = s_num.fillna(train_med).astype(float)

            elif sem_type in CATEGORICAL_SEMANTIC_TYPES:
                # Categorical/ordinal: map missing to explicit sentinel -1
                s_code = pd.to_numeric(s, errors="coerce")
                substantive_set = set(active_specs[feat]["substantive_codes"])
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
        if hasattr(self, "_frozen_primary_core_features") and self._frozen_primary_core_features:
            primary_feats = list(self._frozen_primary_core_features)
            expanded_feats = list(getattr(self, "_frozen_expanded_features", self.expanded_features))
            cat_feats = list(getattr(self, "_frozen_categorical_features", self.categorical_features))
            num_feats = list(getattr(self, "_frozen_numerical_features", self.numerical_features))
        else:
            primary_feats = list(self.primary_core_features)
            expanded_feats = list(self.expanded_features)
            cat_feats = list(self.categorical_features)
            num_feats = list(self.numerical_features)

        if feature_set == "primary_core":
            feats = primary_feats
        elif feature_set in ("expanded", "expanded_utilization"):
            feats = expanded_feats
        else:
            raise NHISPreprocessingError(f"Unknown feature_set: {feature_set}")

        cats = [f for f in feats if f in cat_feats]
        nums = [f for f in feats if f in num_feats]
        return list(cats), list(nums)

    def export_fit_json(self, path: Union[str, pathlib.Path]) -> None:
        """Serialize 2022-fitted statistics to JSON with immutable rule identity."""
        if not self.is_fitted:
            raise NHISPreprocessingError("Cannot export unfitted preprocessor.")
        if getattr(self, "is_legacy_unverified", False) or getattr(getattr(self, "_fitted_record", None), "fit_year", None) == "unspecified":
            raise ValueError("Cannot export unverified legacy or unspecified provenance fit record as verified artifact; refit with valid provenance required.")
        if hasattr(self, "_frozen_specs") and self.specs != self._frozen_specs:
            raise ValueError("Preprocessor specs have been mutated after fit. Exporting with mutated specs is prohibited.")
        if hasattr(self, "_frozen_primary_core_features") and self.primary_core_features != self._frozen_primary_core_features:
            raise ValueError("primary_core_features mutated after fit; export rejected.")

        path_obj = pathlib.Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        rec_dict = self.fitted_record.to_dict()
        specs_json = json.dumps(self._frozen_specs, sort_keys=True, default=str)
        specs_hash = hashlib.sha256(specs_json.encode("utf-8")).hexdigest()
        rec_dict["format_version"] = "nhis_fairbias_fit_v2"
        rec_dict["rule_identity"] = {
            "specs_hash": specs_hash,
            "primary_core_features": list(self._frozen_primary_core_features),
            "expanded_features": list(getattr(self, "_frozen_expanded_features", self.expanded_features)),
            "categorical_features": list(getattr(self, "_frozen_categorical_features", self.categorical_features)),
            "numerical_features": list(getattr(self, "_frozen_numerical_features", self.numerical_features)),
            "specs": copy.deepcopy(self._frozen_specs),
        }
        with path_obj.open("w", encoding="utf-8") as handle:
            json.dump(rec_dict, handle, indent=2)

    def load_fit_json(self, path: Union[str, pathlib.Path]) -> NHISPreprocessor:
        """Load fitted statistics from JSON with rule identity verification.

        Performs comprehensive local validation before committing any state to self.
        If validation fails, self remains in its exact prior state.
        """
        path_obj = pathlib.Path(path)
        with path_obj.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, dict):
            raise ValueError(f"Artifact must be a JSON dictionary, got {type(data).__name__}")

        fmt_ver = data.get("format_version")
        if fmt_ver is not None and fmt_ver not in ("nhis_fairbias_fit_v2", "nhis_fairbias_fit_v1", "nhis_fairbias_fit_v1_legacy"):
            raise ValueError(f"Unsupported format_version: {fmt_ver!r}")

        # Format v2 explicitly requires rule_identity; legacy format rejects v2 rule_identity
        if fmt_ver == "nhis_fairbias_fit_v2" and "rule_identity" not in data:
            raise ValueError("Format v2 artifact requires rule_identity")
        if fmt_ver in ("nhis_fairbias_fit_v1", "nhis_fairbias_fit_v1_legacy") and "rule_identity" in data:
            raise ValueError(
                f"Contradictory artifact: legacy format_version ({fmt_ver!r}) declared with v2 rule_identity."
            )

        specs_json = json.dumps(self.specs, sort_keys=True, default=str)
        current_specs_hash = hashlib.sha256(specs_json.encode("utf-8")).hexdigest()

        if "rule_identity" in data:
            rule_id = data["rule_identity"]
            if not isinstance(rule_id, dict):
                raise ValueError("rule_identity must be a dictionary")

            for req_k in ("specs_hash", "specs", "primary_core_features", "expanded_features", "categorical_features", "numerical_features"):
                if req_k not in rule_id:
                    raise ValueError(f"rule_identity missing required field: {req_k!r}")

            file_hash = rule_id.get("specs_hash")
            file_specs = rule_id.get("specs")
            if file_hash != current_specs_hash or file_specs != self.specs:
                raise ValueError(
                    f"rule identity mismatch: artifact specs hash {file_hash} does not match current preprocessor specs hash {current_specs_hash}"
                )

            # Verify feature ordering and family lists consistency
            file_core_feats = rule_id.get("primary_core_features")
            if list(file_core_feats) != list(self.primary_core_features):
                raise ValueError(
                    f"Column ordering mismatch in primary_core_features: expected {list(self.primary_core_features)}, got {file_core_feats}"
                )

            file_exp_feats = rule_id.get("expanded_features")
            if list(file_exp_feats) != list(self.expanded_features):
                raise ValueError(
                    f"Column ordering mismatch in expanded_features: expected {list(self.expanded_features)}, got {file_exp_feats}"
                )

            file_cat_feats = rule_id.get("categorical_features")
            if list(file_cat_feats) != list(self.categorical_features):
                raise ValueError(
                    f"Categorical features mismatch in rule_identity: expected {list(self.categorical_features)}, got {file_cat_feats}"
                )

            file_num_feats = rule_id.get("numerical_features")
            if list(file_num_feats) != list(self.numerical_features):
                raise ValueError(
                    f"Numerical features mismatch in rule_identity: expected {list(self.numerical_features)}, got {file_num_feats}"
                )

            temp_is_legacy_unverified = False
        else:
            temp_is_legacy_unverified = True

        # Validate required body fields in data
        req_body_keys = (
            "fit_year",
            "fit_study_role",
            "row_count",
            "numerical_medians",
            "numerical_stats",
            "categorical_categories",
            "empwrkft_distribution",
            "rules_manifest",
        )
        for req_k in req_body_keys:
            if req_k not in data:
                raise ValueError(f"Fitted artifact missing required field: {req_k!r}")

        # Validate row count: strict positive integer, rejecting bool and float truncation
        row_cnt = data["row_count"]
        if isinstance(row_cnt, bool) or not isinstance(row_cnt, int) or row_cnt <= 0:
            raise ValueError(f"row_count must be a positive integer, got {row_cnt!r} ({type(row_cnt).__name__})")

        # Validate rules_manifest and regime/year/role consistency
        rules_manifest = data["rules_manifest"]
        if not isinstance(rules_manifest, dict):
            raise ValueError("rules_manifest must be a dictionary")
        manifest_regime = rules_manifest.get("regime")
        schema_version = rules_manifest.get("schema_version")
        raw_yr = str(data["fit_year"])
        fit_role = str(data["fit_study_role"])

        if manifest_regime == "temporal":
            if raw_yr != "2022":
                raise ValueError(f"Invalid fit_year for temporal regime: {raw_yr!r}; expected '2022'")
            if fit_role != "development_train":
                raise ValueError(f"Invalid fit_study_role for temporal regime: {fit_role!r}; expected 'development_train'")
            if schema_version not in ("nhis-fairbias-d2-1.0", "nhis-fairbias-d2-0.1"):
                raise ValueError(f"Invalid schema_version for temporal regime: {schema_version!r}")
        elif manifest_regime == "pooled":
            if raw_yr != "2022-2024_pooled":
                raise ValueError(f"Invalid fit_year for pooled regime: {raw_yr!r}; expected '2022-2024_pooled'")
            if fit_role != "pooled_train":
                raise ValueError(f"Invalid fit_study_role for pooled regime: {fit_role!r}; expected 'pooled_train'")
            if schema_version not in ("nhis-fairbias-d3-1.0", "nhis-fairbias-d3-0.1"):
                raise ValueError(f"Invalid schema_version for pooled regime: {schema_version!r}")
        elif manifest_regime in ("unspecified", None) and (raw_yr == "unspecified" or fit_role == "unspecified"):
            temp_is_legacy_unverified = True
        else:
            raise ValueError(
                f"Inconsistent regime/year/role metadata: regime={manifest_regime!r}, year={raw_yr!r}, role={fit_role!r}"
            )

        # Validate numerical medians and stats consistency
        num_meds_raw = data["numerical_medians"]
        if not isinstance(num_meds_raw, dict):
            raise ValueError("numerical_medians must be a dictionary")
        expected_num_feats = set(self.numerical_features)
        actual_med_feats = set(num_meds_raw.keys())
        if actual_med_feats != expected_num_feats:
            missing_m = expected_num_feats - actual_med_feats
            extra_m = actual_med_feats - expected_num_feats
            raise ValueError(
                f"numerical_medians features mismatch: missing={sorted(missing_m)}, extra={sorted(extra_m)}"
            )

        num_stats_raw = data["numerical_stats"]
        if not isinstance(num_stats_raw, dict):
            raise ValueError("numerical_stats must be a dictionary")
        if set(num_stats_raw.keys()) != expected_num_feats:
            raise ValueError("numerical_stats feature keys mismatch")

        parsed_medians: Dict[str, float] = {}
        for feat_name in sorted(self.numerical_features):
            raw_val = num_meds_raw[feat_name]
            if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)):
                raise ValueError(f"Median for {feat_name} is not a valid float: {raw_val!r}")
            val_f = float(raw_val)
            if np.isnan(val_f) or np.isinf(val_f):
                raise ValueError(f"Non-finite median for {feat_name}: {val_f}")

            substantive_codes = self.specs.get(feat_name, {}).get("substantive_codes", [])
            if substantive_codes:
                min_bound = min(substantive_codes)
                max_bound = max(substantive_codes)
                if val_f < min_bound or val_f > max_bound:
                    raise ValueError(
                        f"Median for {feat_name} ({val_f}) is outside substantive range [{min_bound}, {max_bound}]"
                    )
            parsed_medians[feat_name] = val_f

            # Validate numerical stats body (R7-05)
            stats = num_stats_raw[feat_name]
            if not isinstance(stats, dict):
                raise ValueError(f"numerical_stats for {feat_name} must be a dictionary")
            for req_stat_k in ("median", "mean", "min", "max", "count_valid", "count_missing"):
                if req_stat_k not in stats:
                    raise ValueError(f"numerical_stats for {feat_name} missing required field: {req_stat_k!r}")

            for stat_k in ("median", "mean", "min", "max"):
                sv = stats[stat_k]
                if isinstance(sv, bool) or not isinstance(sv, (int, float)) or not np.isfinite(float(sv)):
                    raise ValueError(f"Non-finite {stat_k} in numerical_stats for {feat_name}: {sv}")

            s_med = float(stats["median"])
            s_mean = float(stats["mean"])
            s_min = float(stats["min"])
            s_max = float(stats["max"])

            if abs(s_med - val_f) > 1e-6:
                raise ValueError(
                    f"numerical_stats median mismatch for {feat_name}: {s_med} vs {val_f}"
                )
            if not (s_min <= s_med <= s_max):
                raise ValueError(
                    f"numerical_stats bounds invalid for {feat_name}: min {s_min} <= median {s_med} <= max {s_max} violated"
                )
            if not (s_min <= s_mean <= s_max):
                raise ValueError(
                    f"numerical_stats bounds invalid for {feat_name}: min {s_min} <= mean {s_mean} <= max {s_max} violated"
                )

            if substantive_codes:
                min_bound = min(substantive_codes)
                max_bound = max(substantive_codes)
                if s_min < min_bound or s_max > max_bound:
                    raise ValueError(
                        f"numerical_stats min/max for {feat_name} [{s_min}, {s_max}] is outside substantive range [{min_bound}, {max_bound}]"
                    )

            cnt_valid = stats["count_valid"]
            if isinstance(cnt_valid, bool) or not isinstance(cnt_valid, int) or cnt_valid <= 0 or cnt_valid > row_cnt:
                raise ValueError(
                    f"Invalid count_valid in numerical_stats for {feat_name}: {cnt_valid!r}; "
                    f"must be positive integer <= row_count ({row_cnt})"
                )

            cnt_missing = stats["count_missing"]
            if isinstance(cnt_missing, bool) or not isinstance(cnt_missing, int) or cnt_missing < 0 or cnt_missing > row_cnt:
                raise ValueError(
                    f"Invalid count_missing in numerical_stats for {feat_name}: {cnt_missing!r}; "
                    f"must be non-negative integer <= row_count ({row_cnt})"
                )

            if cnt_valid + cnt_missing != row_cnt:
                raise ValueError(
                    f"Count mismatch in numerical_stats for {feat_name}: "
                    f"count_valid ({cnt_valid}) + count_missing ({cnt_missing}) != row_count ({row_cnt})"
                )

        # Validate categorical categories
        cat_cats_raw = data["categorical_categories"]
        if not isinstance(cat_cats_raw, dict):
            raise ValueError("categorical_categories must be a dictionary")
        expected_cat_feats = set(self.categorical_features)
        if set(cat_cats_raw.keys()) != expected_cat_feats:
            raise ValueError("categorical_categories feature keys mismatch")

        for feat_name in sorted(self.categorical_features):
            cats = cat_cats_raw[feat_name]
            if not isinstance(cats, list):
                raise ValueError(f"categorical_categories for {feat_name} must be a list")
            if feat_name == "empwrkft1_a":
                allowed_emp = {1, 2, SENTINEL_STRUCTURAL_NOT_IN_UNIVERSE, SENTINEL_EXPLICIT_MISSING}
                if not set(cats).issubset(allowed_emp):
                    raise ValueError(f"Categorical categories for {feat_name} contains invalid employment states: {cats}")
            else:
                substantive_set = set(self.specs[feat_name].get("substantive_codes", []))
                if substantive_set and not set(cats).issubset(substantive_set):
                    invalid_c = set(cats) - substantive_set
                    raise ValueError(
                        f"Categorical categories for {feat_name} contains invalid codes outside schema: {sorted(invalid_c)}"
                    )

        # Validate empwrkft_distribution
        emp_dist = data["empwrkft_distribution"]
        if not isinstance(emp_dist, dict):
            raise ValueError("empwrkft_distribution must be a dictionary")
        expected_states = set(EMPWRKFT_STATE_MAP.values())
        if set(emp_dist.keys()) != expected_states:
            raise ValueError(
                f"empwrkft_distribution keys mismatch: expected {sorted(expected_states)}, got {sorted(emp_dist.keys())}"
            )
        for state_name, cnt in emp_dist.items():
            if isinstance(cnt, bool) or not isinstance(cnt, int) or cnt < 0:
                raise ValueError(f"Invalid count in empwrkft_distribution for {state_name}: {cnt!r}")
        if sum(emp_dist.values()) != row_cnt:
            raise ValueError(
                f"empwrkft_distribution total count ({sum(emp_dist.values())}) does not match row_count ({row_cnt})"
            )

        # Parse record
        parsed_record = PreprocessingFitRecord.from_dict(data)

        # Atomic commit: all validations succeeded without exception
        self.is_legacy_unverified = temp_is_legacy_unverified
        self._fitted_record = parsed_record
        self._frozen_specs = copy.deepcopy(self.specs)
        self._fitted_rules_hash = json.dumps(self._frozen_specs, sort_keys=True, default=str)
        self._frozen_numerical_medians = copy.deepcopy(parsed_medians)
        self._frozen_primary_core_features = list(self.primary_core_features)
        self._frozen_expanded_features = list(self.expanded_features)
        self._frozen_categorical_features = list(self.categorical_features)
        self._frozen_numerical_features = list(self.numerical_features)
        return self
