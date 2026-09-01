"""Pooled NHIS 2022-2024 analysis regime, immutable 64/16/20 split, and adapter for Gate D3.

Guarantees:
1. Pooled random split:
   - 64% train (57,473 rows)
   - 16% validation (14,368 rows)
   - 20% test (17,961 rows)
   - Total: exactly 89,802 observations.
   - Mutually exclusive, exhaustive, fixed documented random seed.
2. Independent metadata:
   - Survey year (2022, 2023, 2024) is strictly preserved as independent metadata.
   - All three years appear in all three partitions.
3. Strict data leakage guards:
   - Preprocessing and FairBias transform learning use TRAIN ONLY.
   - Any attempt to fit preprocessing or compute epsilon on validation/test raises NHISLeakageError.
4. Tang et al. (2024) paper-faithful baseline:
   - Purely unweighted; rejects/ignores survey weights in tang2024_paper_faithful mode.
   - Preserves semantic feature registry, explicit 4-state employment intensity, and disability arms.
"""

from __future__ import annotations

import copy
import pathlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from fairbias.bias_metric import compute_dphi_matrix
from fairbias.config import (
    ALGORITHM_MODE_PAPER_FAITHFUL,
    FairBiasConfig,
)
from fairbias.evaluator import FairEvaluator

from .adapter import (
    DISABILITY_COMPONENTS,
    OUTCOME_MAP,
    PROTECTED_MAP,
)
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

DEFAULT_POOLED_SEED = 2024
DEFAULT_SPLIT_FRACTIONS = (0.64, 0.16, 0.20)
EXPECTED_TOTAL_ROWS = 89802
EXPECTED_TRAIN_ROWS = 57473
EXPECTED_VAL_ROWS = 14368
EXPECTED_TEST_ROWS = 17961


def generate_pooled_splits(
    df: pd.DataFrame,
    seed: int = DEFAULT_POOLED_SEED,
    train_ratio: float = 0.64,
    val_ratio: float = 0.16,
    test_ratio: float = 0.20,
) -> pd.DataFrame:
    """
    Generate deterministic, mutually exclusive 64/16/20 split on pooled NHIS observations.

    Parameters
    ----------
    df : pd.DataFrame
        Pooled NHIS features dataframe (expected 89,802 rows).
    seed : int
        Fixed random seed for full reproducibility (default 2024).
    train_ratio, val_ratio, test_ratio : float
        Partition fractions summing to 1.0.

    Returns
    -------
    pd.DataFrame
        Manifest dataframe with columns:
        ['orig_row_idx', 'record_id', 'survey_year', 'study_role', 'split_role']
    """
    n_total = len(df)
    if n_total != EXPECTED_TOTAL_ROWS:
        raise NHISFeatureError(
            f"Expected {EXPECTED_TOTAL_ROWS} observations in pooled NHIS data, got {n_total}"
        )

    # Establish deterministic baseline index ordering
    df_sorted = df.reset_index(drop=True)
    rng = np.random.default_rng(seed=seed)
    permuted_indices = rng.permutation(n_total)

    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))
    n_test = n_total - n_train - n_val

    if (n_train, n_val, n_test) != (EXPECTED_TRAIN_ROWS, EXPECTED_VAL_ROWS, EXPECTED_TEST_ROWS):
        raise ValueError(
            f"Split count calculation mismatch: ({n_train}, {n_val}, {n_test}) != "
            f"({EXPECTED_TRAIN_ROWS}, {EXPECTED_VAL_ROWS}, EXPECTED_TEST_ROWS)"
        )

    train_idx = set(permuted_indices[:n_train])
    val_idx = set(permuted_indices[n_train : n_train + n_val])
    test_idx = set(permuted_indices[n_train + n_val :])

    # Assign split roles
    split_roles = []
    record_ids = []
    for i in range(n_total):
        yr = int(df_sorted.loc[i, "survey_year"])
        record_ids.append(f"NHIS_{yr}_{i:06d}")
        if i in train_idx:
            split_roles.append("train")
        elif i in val_idx:
            split_roles.append("val")
        elif i in test_idx:
            split_roles.append("test")
        else:
            raise RuntimeError(f"Unassigned row index: {i}")

    manifest_df = pd.DataFrame({
        "orig_row_idx": np.arange(n_total, dtype=int),
        "record_id": record_ids,
        "survey_year": df_sorted["survey_year"].astype(int).to_numpy(),
        "study_role": df_sorted["study_role"].astype(str).to_numpy(),
        "split_role": split_roles,
    })

    return manifest_df


def audit_pooled_splits(manifest_df: pd.DataFrame) -> Dict[str, Any]:
    """Audit split manifest for mutual exclusivity, completeness, and year independence."""
    total_rows = len(manifest_df)
    counts = manifest_df["split_role"].value_counts().to_dict()

    train_cnt = counts.get("train", 0)
    val_cnt = counts.get("val", 0)
    test_cnt = counts.get("test", 0)

    # Mutual exclusivity & exhaustiveness
    is_exhaustive = (train_cnt + val_cnt + test_cnt) == total_rows == EXPECTED_TOTAL_ROWS
    has_exact_counts = (
        train_cnt == EXPECTED_TRAIN_ROWS
        and val_cnt == EXPECTED_VAL_ROWS
        and test_cnt == EXPECTED_TEST_ROWS
    )

    # Year representation across all partitions
    year_breakdown: Dict[str, Dict[int, int]] = {}
    for role in ("train", "val", "test"):
        sub = manifest_df[manifest_df["split_role"] == role]
        year_breakdown[role] = sub["survey_year"].value_counts().to_dict()

    all_years_present = True
    for role, y_counts in year_breakdown.items():
        if set(y_counts.keys()) != {2022, 2023, 2024}:
            all_years_present = False

    status = "PASS" if (is_exhaustive and has_exact_counts and all_years_present) else "FAIL"

    return {
        "status": status,
        "total_observations": total_rows,
        "expected_observations": EXPECTED_TOTAL_ROWS,
        "split_counts": {
            "train": train_cnt,
            "val": val_cnt,
            "test": test_cnt,
        },
        "split_fractions": {
            "train": round(train_cnt / total_rows, 6),
            "val": round(val_cnt / total_rows, 6),
            "test": round(test_cnt / total_rows, 6),
        },
        "year_breakdown_by_split": year_breakdown,
        "mutually_exclusive_and_exhaustive": is_exhaustive,
        "has_exact_counts": has_exact_counts,
        "all_years_present_in_all_splits": all_years_present,
    }


class NHISPooledAdapter:
    """Adapter and orchestrator for the pooled NHIS 64/16/20 experiment regime."""

    def __init__(
        self,
        features_parquet_path: Optional[Union[str, pathlib.Path]] = None,
        split_manifest_path: Optional[Union[str, pathlib.Path]] = None,
        study_config_path: Optional[Union[str, pathlib.Path]] = None,
        feature_config_path: Optional[Union[str, pathlib.Path]] = None,
        preprocessor: Optional[NHISPreprocessor] = None,
        seed: int = DEFAULT_POOLED_SEED,
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
        if len(self._raw_df) != EXPECTED_TOTAL_ROWS:
            raise NHISFeatureError(
                f"Expected {EXPECTED_TOTAL_ROWS} rows in parquet, found {len(self._raw_df)}"
            )

        # Split manifest
        if split_manifest_path is not None and pathlib.Path(split_manifest_path).is_file():
            self.split_manifest = pd.read_csv(split_manifest_path)
        else:
            self.split_manifest = generate_pooled_splits(self._raw_df, seed=seed)

        # Merge split_role into raw dataframe
        self._raw_df["split_role"] = self.split_manifest["split_role"].to_numpy()
        self._raw_df["record_id"] = self.split_manifest["record_id"].to_numpy()

        # Preprocessor instance
        self.preprocessor = preprocessor or NHISPreprocessor(
            feature_registry=self.feature_registry
        )
        if not self.preprocessor.is_fitted:
            # Fit strictly on pooled train partition
            train_sub = self.get_partition("train")
            self.preprocessor.fit(train_sub, regime="pooled")

    def get_partition(self, split_role: str) -> pd.DataFrame:
        """Retrieve raw un-preprocessed slice for a specific split role ('train', 'val', 'test')."""
        role_str = str(split_role).strip().lower()
        if role_str not in ("train", "val", "test"):
            raise ValueError(f"Invalid split_role: {split_role!r}. Must be 'train', 'val', or 'test'.")
        return self._raw_df[self._raw_df["split_role"] == role_str].copy()

    def get_feature_names(
        self,
        feature_set: str = "primary_core",
        disability_arm: str = "full_feature",
    ) -> List[str]:
        """Return ordered list of feature names for given feature_set and disability arm."""
        if feature_set == "primary_core":
            feats = list(self.preprocessor.primary_core_features)
        elif feature_set in ("expanded", "expanded_utilization"):
            feats = list(self.preprocessor.expanded_features)
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

    def get_pooled_cohort(
        self,
        *,
        outcome: str = "MEDDL12M_A",
        protected_attribute: str = "SEX_A",
        feature_set: str = "primary_core",
        disability_arm: str = "full_feature",
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]]:
        """
        Produce preprocessed (X, y, o, w, metadata) cohorts across all 3 partitions ('train', 'val', 'test').

        Filters:
        - Drops non-substantive records for outcome Y (never imputes Y).
        - Drops non-substantive records for protected attribute O (never imputes O).
        - HISPALLP_A remains 7-class multicategorical (never binarized).
        - X is preprocessed using train-only fitted statistics.
        """
        harm_outcome = OUTCOME_MAP.get(outcome)
        if harm_outcome is None or harm_outcome not in self._raw_df.columns:
            raise ValueError(f"Unsupported outcome: {outcome}. Supported: MEDDL12M_A, MEDNG12M_A")

        harm_prot = PROTECTED_MAP.get(protected_attribute)
        if harm_prot is None or harm_prot not in self._raw_df.columns:
            raise ValueError(
                f"Unsupported protected attribute: {protected_attribute}. Supported: SEX_A, HISPALLP_A, DISAB3_A"
            )

        active_feature_names = self.get_feature_names(
            feature_set=feature_set, disability_arm=disability_arm
        )

        cohorts: Dict[str, Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]] = {}

        for role in ("train", "val", "test"):
            raw_sub = self.get_partition(role)
            y_raw = raw_sub[harm_outcome]
            o_raw = raw_sub[harm_prot]

            valid_mask = y_raw.notna() & o_raw.notna()
            filtered_df = raw_sub[valid_mask].copy()

            X_all = self.preprocessor.transform(
                filtered_df, feature_set=feature_set, preserve_metadata=False
            )
            X = X_all[active_feature_names].copy()

            y = filtered_df[harm_outcome].astype(int)
            y.name = outcome

            o = filtered_df[harm_prot].astype(int)
            o.name = protected_attribute

            w = filtered_df["WTFA_A"].astype(float)
            w.name = "WTFA_A"

            meta_cols = ["record_id", "survey_year", "study_role", "split_role", "WTFA_A", "PSTRAT", "PPSU"]
            metadata = filtered_df[meta_cols].copy()

            cohorts[role] = (X, y, o, w, metadata)

        return cohorts

    def compute_train_epsilon(
        self,
        *,
        outcome: str = "MEDDL12M_A",
        protected_attribute: str = "SEX_A",
        feature_set: str = "primary_core",
        disability_arm: str = "full_feature",
        config: Optional[FairBiasConfig] = None,
        sample_weight: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Compute d_phi and epsilon threshold strictly on the pooled train partition.

        Tang et al. (2024) baseline is strictly unweighted.
        """
        cfg = config or FairBiasConfig.compas_default(mode="paper")
        if cfg.algorithm_mode == ALGORITHM_MODE_PAPER_FAITHFUL and sample_weight is not None:
            raise NHISLeakageError(
                "tang2024_paper_faithful mode strictly forbids sample_weight. "
                "The Tang et al. (2024) baseline is unweighted."
            )

        cohorts = self.get_pooled_cohort(
            outcome=outcome,
            protected_attribute=protected_attribute,
            feature_set=feature_set,
            disability_arm=disability_arm,
        )
        X_train, y_train, o_train, _, _ = cohorts["train"]

        all_cats, all_nums = self.preprocessor.get_feature_family_lists(feature_set)
        active_feats = set(X_train.columns)
        cate_attrs = [f for f in all_cats if f in active_feats]
        num_attrs = [f for f in all_nums if f in active_feats]

        evaluator = FairEvaluator(
            config=cfg,
            label_O=[protected_attribute],
            label_Y=outcome,
            cate_attrs=cate_attrs,
            num_attrs=num_attrs,
        )

        O_df = pd.DataFrame({protected_attribute: o_train})
        dphi_dict = evaluator.calculate_epsilon(
            X_train, O_df, cate_attrs=cate_attrs, num_attrs=num_attrs, sample_weight=None
        )
        threshold = evaluator.compute_threshold(dphi_dict)

        return {
            "regime": "pooled_train",
            "outcome": outcome,
            "protected_attribute": protected_attribute,
            "feature_set": feature_set,
            "disability_arm": disability_arm,
            "algorithm_mode": cfg.algorithm_mode,
            "feature_count": len(X_train.columns),
            "d_phi": dphi_dict.get(protected_attribute, {}),
            "epsilon_threshold": float(threshold),
        }
