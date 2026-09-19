"""Frozen temporal analysis interface and study adapter for NHIS Gate D2.

Guarantees:
1. Strict temporal partition contract:
   - 2022 = development_train
   - 2023 = development_validation
   - 2024 = frozen_test (evaluation-only, never participates in fitting/selection)
   - Random splitting is strictly forbidden and rejected.
2. Predictor semantics from configs/nhis/features.json exclusively:
   - PRIMARY_CORE: exactly 21 features (18 categorical, 3 numerical)
   - EXPANDED: exactly 24 features (20 categorical, 4 numerical)
3. Outcomes:
   - Never imputed.
   - Only substantive 1/2 records retained (recoded to 1/0).
4. Protected attributes:
   - Never imputed.
   - Official substantive categories preserved.
   - HISPALLP_A remains 7-class multicategorical (never median-binarized).
   - Protected dimensions analyzed separately.
5. Disability sensitivity arms:
   - Components: visiondf_a, hearingdf_a, diff_a, comdiff_a, uppslfcr_a, cogmemdff_a
   - Arms: full_feature vs exclude_disability_components
   - Sizes: PRIMARY (21 -> 15), EXPANDED (24 -> 18)
6. Epsilon determination:
   - Strictly computed on 2022 development_train.
   - 2023 and 2024 are rejected by the epsilon API.
"""

from __future__ import annotations

import copy
import pathlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from fairbias.bias_metric import compute_dphi_matrix
from fairbias.config import FairBiasConfig
from fairbias.evaluator import FairEvaluator

from .features import (
    DEFAULT_FEATURE_CONFIG,
    NHISFeatureError,
    load_feature_registry,
)
from .preprocessing import (
    NHISLeakageError,
    NHISPreprocessingError,
    NHISPreprocessor,
)
from .schema import (
    DEFAULT_STUDY_CONFIG,
    load_study_config,
)

DISABILITY_COMPONENTS: Tuple[str, ...] = (
    "visiondf_a",
    "hearingdf_a",
    "diff_a",
    "comdiff_a",
    "uppslfcr_a",
    "cogmemdff_a",
)

STUDY_YEAR_ROLES = {
    2022: "development_train",
    2023: "development_validation",
    2024: "frozen_test",
}

OUTCOME_MAP = {
    "MEDDL12M_A": "meddl12m",
    "meddl12m": "meddl12m",
    "MEDNG12M_A": "medng12m",
    "medng12m": "medng12m",
}

PROTECTED_MAP = {
    "SEX_A": "sex_a",
    "sex_a": "sex_a",
    "HISPALLP_A": "hispallp_a",
    "hispallp_a": "hispallp_a",
    "DISAB3_A": "disab3_a",
    "disab3_a": "disab3_a",
}


class NHISStudyAdapter:
    """Dedicated orchestrator and temporal interface for NHIS application study."""

    def __init__(
        self,
        features_parquet_path: Optional[Union[str, pathlib.Path]] = None,
        study_config_path: Optional[Union[str, pathlib.Path]] = None,
        feature_config_path: Optional[Union[str, pathlib.Path]] = None,
        preprocessor: Optional[NHISPreprocessor] = None,
    ):
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        study_cfg_file = study_config_path or DEFAULT_STUDY_CONFIG
        self.study_config = load_study_config(study_cfg_file)

        feat_cfg_file = feature_config_path or DEFAULT_FEATURE_CONFIG
        self.feature_registry = load_feature_registry(feat_cfg_file)

        pq_path = features_parquet_path or (repo_root / self.study_config["outputs"]["features_parquet"])
        self.features_parquet_path = pathlib.Path(pq_path).resolve()
        if not self.features_parquet_path.is_file():
            raise FileNotFoundError(f"Prepared features parquet not found: {self.features_parquet_path}")

        self._raw_df = pd.read_parquet(self.features_parquet_path)
        self._validate_partitions()

        # Preprocessor instance
        self.preprocessor = preprocessor or NHISPreprocessor(
            feature_registry=self.feature_registry
        )
        if not self.preprocessor.is_fitted:
            # Fit strictly on 2022 development_train partition
            df_2022 = self.get_raw_partition(2022)
            self.preprocessor.fit(df_2022)
        else:
            self._validate_preprocessor(self.preprocessor)

    def _validate_preprocessor(self, preprocessor: Any) -> None:
        """Validate pre-fitted preprocessor provenance, registry schema, feature lists, and ordering."""
        rec = getattr(preprocessor, "fitted_record", None)
        if rec is None or str(rec.fit_year) != "2022" or rec.fit_study_role != "development_train":
            raise ValueError(
                f"Production NHISStudyAdapter rejected pre-fitted preprocessor with invalid provenance: "
                f"fit_year={getattr(rec, 'fit_year', None)}, fit_study_role={getattr(rec, 'fit_study_role', None)}. "
                "Pre-fitted preprocessor must be fitted on 2022 development_train data."
            )
        if not getattr(preprocessor, "is_fitted", False):
            raise ValueError("Pre-fitted preprocessor must be fitted prior to adapter initialization")
        if getattr(preprocessor, "is_legacy_unverified", False):
            raise ValueError(
                "Production NHISStudyAdapter rejected unverified legacy preprocessor fit artifact; "
                "primary study requires verified rule identity."
            )
        manifest_regime = rec.rules_manifest.get("regime") if hasattr(rec, "rules_manifest") and isinstance(rec.rules_manifest, dict) else None
        if manifest_regime != "temporal":
            raise ValueError(
                f"Production NHISStudyAdapter rejected non-temporal preprocessor: regime={manifest_regime!r}; "
                "temporal adapter requires temporal regime preprocessor fitted on 2022 development_train."
            )

        # Validate feature registry compatibility
        prep_reg = getattr(preprocessor, "registry", None)
        if prep_reg is None:
            prep_reg = getattr(preprocessor, "feature_registry", None)
        if prep_reg is None or self.feature_registry is None:
            raise ValueError("Pre-fitted preprocessor is missing valid .registry attribute")

        # Validate feature_lists explicitly (both primary_core_features and expanded_utilization_features)
        # ensuring exact content and ordering match
        a_fl = self.feature_registry.get("feature_lists", {})
        p_fl = prep_reg.get("feature_lists", {})
        for fl_key in ("primary_core_features", "expanded_utilization_features"):
            a_list = list(a_fl.get(fl_key, []))
            p_list = list(p_fl.get(fl_key, []))
            if a_list != p_list:
                raise ValueError(
                    f"Pre-fitted preprocessor feature_lists '{fl_key}' mismatch: "
                    f"preprocessor has {p_list}, adapter expects {a_list}."
                )

        # Validate family structure and ordering
        for fam in ("primary_core", "expanded_utilization"):
            a_fam = list(self.feature_registry.get(fam, {}).keys())
            p_fam = list(prep_reg.get(fam, {}).keys())
            if a_fam != p_fam:
                raise ValueError(
                    f"Pre-fitted preprocessor family '{fam}' feature keys or ordering mismatch: "
                    f"preprocessor has {p_fam}, adapter expects {a_fam}."
                )

        adapter_specs = {}
        for fam in ("primary_core", "expanded_utilization"):
            for k, v in self.feature_registry.get(fam, {}).items():
                hname = v.get("harmonized_name", k.lower())
                adapter_specs[hname] = v

        prep_specs = {}
        for fam in ("primary_core", "expanded_utilization"):
            for k, v in prep_reg.get(fam, {}).items():
                hname = v.get("harmonized_name", k.lower())
                prep_specs[hname] = v

        if list(prep_specs.keys()) != list(adapter_specs.keys()):
            raise ValueError(
                f"Pre-fitted preprocessor feature set or ordering mismatch: "
                f"preprocessor has {list(prep_specs.keys())}, adapter expects {list(adapter_specs.keys())}."
            )

        critical_fields = (
            "semantic_type",
            "substantive_codes",
            "encoding",
            "categories",
            "missing_codes",
            "missing_rules",
            "valid_range",
        )
        for feat, a_spec in adapter_specs.items():
            p_spec = prep_specs.get(feat)
            if p_spec is None:
                raise ValueError(f"Pre-fitted preprocessor missing specification for feature {feat!r}")
            for field in critical_fields:
                a_val = a_spec.get(field)
                p_val = p_spec.get(field)
                if a_val != p_val:
                    raise ValueError(
                        f"Pre-fitted preprocessor feature registry mismatch on feature {feat!r} field {field!r}: "
                        f"preprocessor has {p_val!r}, adapter expects {a_val!r}."
                    )

    def _validate_partitions(self) -> None:
        """Validate exact temporal row counts and study roles."""
        years = set(self._raw_df["survey_year"].unique())
        if years != {2022, 2023, 2024}:
            raise NHISFeatureError(f"Unexpected survey years in parquet: {years}")

        expected_counts = {2022: 27651, 2023: 29522, 2024: 32629}
        for yr, exp_cnt in expected_counts.items():
            sub = self._raw_df[self._raw_df["survey_year"] == yr]
            if len(sub) != exp_cnt:
                raise NHISFeatureError(
                    f"Year {yr} row count mismatch: expected {exp_cnt}, got {len(sub)}"
                )
            role = set(sub["study_role"].unique())
            exp_role = STUDY_YEAR_ROLES[yr]
            if role != {exp_role}:
                raise NHISFeatureError(
                    f"Year {yr} study_role mismatch: expected {exp_role}, got {role}"
                )

    def get_raw_partition(self, year: int) -> pd.DataFrame:
        """Retrieve raw un-preprocessed partition for a specific survey year."""
        year_int = int(year)
        if year_int not in STUDY_YEAR_ROLES:
            raise ValueError(f"Invalid NHIS study year: {year_int}. Must be 2022, 2023, or 2024.")
        return self._raw_df[self._raw_df["survey_year"] == year_int].copy()

    def random_split(self, *args: Any, **kwargs: Any) -> Any:
        """Strictly forbidden for NHIS study: raises ValueError immediately."""
        raise ValueError(
            "Random splitting (e.g. 64/16/20) is strictly forbidden for the "
            "NHIS longitudinal/temporal application study. Use explicit survey_year partitions."
        )

    def get_feature_names(
        self,
        feature_set: str = "primary_core",
        disability_arm: str = "full_feature",
    ) -> List[str]:
        """Return ordered list of feature names for given feature_set and disability arm."""
        if hasattr(self.preprocessor, "_frozen_primary_core_features") and self.preprocessor._frozen_primary_core_features:
            primary_feats = list(self.preprocessor._frozen_primary_core_features)
            expanded_feats = list(getattr(self.preprocessor, "_frozen_expanded_features", self.preprocessor.expanded_features))
        else:
            primary_feats = list(self.preprocessor.primary_core_features)
            expanded_feats = list(self.preprocessor.expanded_features)

        if feature_set == "primary_core":
            feats = primary_feats
        elif feature_set in ("expanded", "expanded_utilization"):
            feats = expanded_feats
        else:
            raise ValueError(f"Unknown feature_set: {feature_set}")

        if disability_arm == "full_feature":
            return feats
        elif disability_arm == "exclude_disability_components":
            return [f for f in feats if f not in DISABILITY_COMPONENTS]
        else:
            raise ValueError(
                f"Unknown disability_arm: {disability_arm}. Must be 'full_feature' or 'exclude_disability_components'"
            )

    def get_cohort(
        self,
        year: int,
        *,
        outcome: str = "MEDDL12M_A",
        protected_attribute: str = "SEX_A",
        feature_set: str = "primary_core",
        disability_arm: str = "full_feature",
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
        """
        Produce preprocessed (X, y, o, w, metadata) cohort for a specific survey year.

        Filters:
        - Drops non-substantive records for outcome Y (retains substantive 1/2 recoded to 1/0).
        - Drops non-substantive records for protected attribute O (never imputes O).
        - HISPALLP_A remains 7-class multicategorical (never binarized).
        - X is preprocessed using 2022-train-fitted statistics.
        - w is WTFA_A (sampling weight).
        - metadata contains survey design variables (WTFA_A, PSTRAT, PPSU, survey_year, study_role).
        """
        raw_year_df = self.get_raw_partition(year)

        # Resolve column names strictly via supported harmonized outcome map
        harm_outcome = OUTCOME_MAP.get(outcome)
        if harm_outcome is None:
            raise ValueError(f"Unsupported outcome: {outcome}. Supported: MEDDL12M_A, MEDNG12M_A")

        if harm_outcome not in raw_year_df.columns:
            raise ValueError(
                f"Harmonized outcome column {harm_outcome!r} not found in raw data frame for {year}"
            )

        harm_prot = PROTECTED_MAP.get(protected_attribute)
        if harm_prot is None or harm_prot not in raw_year_df.columns:
            raise ValueError(
                f"Unsupported protected attribute: {protected_attribute}. Supported: SEX_A, HISPALLP_A, DISAB3_A"
            )

        # Filter substantive records
        y_raw = raw_year_df[harm_outcome]
        o_raw = raw_year_df[harm_prot]

        valid_mask = y_raw.notna() & o_raw.notna()
        filtered_df = raw_year_df[valid_mask].copy()

        # Check binary 0/1 contract on substantive records
        y_vals = filtered_df[harm_outcome].dropna()
        y_unique = set(y_vals.unique())
        if not y_unique.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"Outcome column {harm_outcome!r} contains non-binary values {y_unique}; "
                "harmonized outcome must be encoded strictly as binary {0, 1}."
            )

        # Transform features
        X_all = self.preprocessor.transform(
            filtered_df, feature_set=feature_set, preserve_metadata=False
        )
        active_feature_names = self.get_feature_names(
            feature_set=feature_set, disability_arm=disability_arm
        )
        X = X_all[active_feature_names].copy()

        y = filtered_df[harm_outcome].astype(int)
        y.name = outcome

        o = filtered_df[harm_prot].astype(int)
        o.name = protected_attribute

        # Survey weights
        w = filtered_df["WTFA_A"].astype(float)
        w.name = "WTFA_A"

        meta_cols = ["survey_year", "study_role", "WTFA_A", "PSTRAT", "PPSU"]
        if "WTFA_DEV" in filtered_df.columns:
            meta_cols.append("WTFA_DEV")
        metadata = filtered_df[meta_cols].copy()

        return X, y, o, w, metadata

    def compute_epsilon(
        self,
        *,
        year: int = 2022,
        outcome: str = "MEDDL12M_A",
        protected_attribute: str = "SEX_A",
        feature_set: str = "primary_core",
        disability_arm: str = "full_feature",
        weighted: bool = False,
        config: Optional[FairBiasConfig] = None,
    ) -> Dict[str, Any]:
        """
        Compute d_phi and epsilon threshold strictly on 2022 development_train.

        Raises NHISLeakageError if year != 2022.
        """
        if int(year) != 2022:
            raise NHISLeakageError(
                f"Epsilon determination is forbidden on test/validation years (attempted on {year}). "
                "Epsilon can ONLY be computed from 2022 development_train."
            )

        X, y, o, w, _ = self.get_cohort(
            year=2022,
            outcome=outcome,
            protected_attribute=protected_attribute,
            feature_set=feature_set,
            disability_arm=disability_arm,
        )

        all_cats, all_nums = self.preprocessor.get_feature_family_lists(feature_set)
        active_feats = set(X.columns)
        cate_attrs = [f for f in all_cats if f in active_feats]
        num_attrs = [f for f in all_nums if f in active_feats]

        cfg = config or FairBiasConfig.compas_default()
        sample_weight = w if weighted else None

        evaluator = FairEvaluator(
            config=cfg,
            label_O=[protected_attribute],
            label_Y=outcome,
            cate_attrs=cate_attrs,
            num_attrs=num_attrs,
        )

        O_df = pd.DataFrame({protected_attribute: o})
        dphi_dict = evaluator.calculate_epsilon(
            X, O_df, cate_attrs=cate_attrs, num_attrs=num_attrs, sample_weight=sample_weight
        )
        threshold = evaluator.compute_threshold(dphi_dict)

        return {
            "year": 2022,
            "outcome": outcome,
            "protected_attribute": protected_attribute,
            "feature_set": feature_set,
            "disability_arm": disability_arm,
            "weighted": weighted,
            "feature_count": len(X.columns),
            "d_phi": dphi_dict.get(protected_attribute, {}),
            "epsilon_threshold": float(threshold),
        }
