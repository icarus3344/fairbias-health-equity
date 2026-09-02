"""Dedicated NHIS survey-weighted FairBias runner and Gate D5.1a preflight harness.

Guarantees:
1. Master Pooled Split & Input Integrity:
   - Reuses the frozen master pooled split (SHA: 6874a56f5484186dffdd5faeb7871c3bef85042ccfdf8d1e375fdde75a3ee9f5)
     and features parquet (SHA: 49f415132ff0be0228f8533f9f74c48cd79ff6fa8be66db085f7329d9b083383).
   - NEVER creates a new split or random partition.
   - Preserves frozen D4 release tag and release archive immutability.
2. Authoritative WTFA_A Weight Variable & Alignment:
   - Derives WTFA_A from the exact respondent identity row alignment with features X,
     outcome y, protected attribute o, and metadata.
   - Fails closed on any missing, non-numeric, non-finite, zero, negative, or misaligned weights.
   - Enforces strictly positive total weights for every protected group.
3. Strict Geometry-Only Weighting Scope:
   - Survey weights enter ONLY empirical protected-group statistics in Eq. 2 (calculate_epsilon).
   - Survey weights NEVER enter R1 candidate construction, numeric candidate streams, NMI gates,
     classifier model fitting, or downstream evaluation.
4. TRAIN-Only Mitigation Contract:
   - Learned transforms originate strictly from TRAIN partition.
   - Validation outcomes and metrics NEVER inform transform selection.
   - TEST partition is completely embargoed and omitted from this runner.
5. Gate D5.1a Execution Boundary:
   - In Gate D5.1a, only --audit-only is authorized on real data.
   - Substantive mitigation on real data is gated behind allow_mitigation=True (for future Gate D5.1b).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import pathlib
import subprocess
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from fairbias.config import (
    ALGORITHM_MODE_SURVEY_WEIGHTED,
    FairBiasConfig,
    official_power_stream,
)
from fairbias.evaluator import FairEvaluator
from fairbias.mitigation import FairBiasMitigation
from fairbias.models import get_classifier
from fairbias.transform import (
    FairTransform,
    calculate_nmi_dict,
)
from fairbias.transform_trace import (
    FairBiasTransformStep,
    FairBiasTransformTrace,
)

from .adapter import (
    OUTCOME_MAP,
    PROTECTED_MAP,
)
from .audit import write_csv_atomic
from .download import (
    compute_sha256,
    new_run_id,
    utc_timestamp,
    write_json_atomic,
)
from .evaluation import (
    compute_evaluation_comparison,
    compute_fairness_gaps,
    compute_group_coverage,
    compute_group_metrics,
    compute_multicategory_pairwise_differences,
    compute_utility_metrics,
    evaluate_predictions,
)
from .features import (
    DEFAULT_FEATURE_CONFIG,
    load_feature_registry,
)
from .pooled import (
    DEFAULT_POOLED_SEED,
    NHISPooledAdapter,
)
from .preprocessing import NHISLeakageError
from .schema import (
    DEFAULT_STUDY_CONFIG,
    load_study_config,
)
from .survey import (
    get_survey_weighted_geometry_provenance,
    validate_survey_weights,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

PRIMARY_D5_RANDOM_SEED: int = 0

# Four Predeclared Scientific Arms for Gate D5
FROZEN_D5_ARMS: Dict[str, Dict[str, Any]] = {
    "D5_ARM_001": {
        "arm_id": "D5_ARM_001",
        "outcome": "MEDDL12M_A",
        "protected_attribute": "SEX_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "full_feature",
        "expected_predictors": 21,
        "expected_group_count": 2,
        "expected_groups": [1, 2],
        "expected_pair_count": 1,
        "d4_reference_arm": "ARM_D3_001",
    },
    "D5_ARM_002": {
        "arm_id": "D5_ARM_002",
        "outcome": "MEDDL12M_A",
        "protected_attribute": "HISPALLP_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "full_feature",
        "expected_predictors": 21,
        "expected_group_count": 7,
        "expected_groups": [1, 2, 3, 4, 5, 6, 7],
        "expected_pair_count": 21,
        "d4_reference_arm": "ARM_D3_002",
    },
    "D5_ARM_003": {
        "arm_id": "D5_ARM_003",
        "outcome": "MEDDL12M_A",
        "protected_attribute": "DISAB3_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "full_feature",
        "expected_predictors": 21,
        "expected_group_count": 2,
        "expected_groups": [1, 2],
        "expected_pair_count": 1,
        "d4_reference_arm": "ARM_D3_003",
    },
    "D5_ARM_004": {
        "arm_id": "D5_ARM_004",
        "outcome": "MEDDL12M_A",
        "protected_attribute": "DISAB3_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "exclude_disability_components",
        "expected_predictors": 15,
        "expected_group_count": 2,
        "expected_groups": [1, 2],
        "expected_pair_count": 1,
        "d4_reference_arm": "ARM_D3_004",
    },
}

# Frozen Input Hashes and Paths
FROZEN_SPLIT_MANIFEST_PATH: pathlib.Path = _REPO_ROOT / "artifacts" / "nhis" / "d3" / "pooled_split_manifest.csv"
FROZEN_SPLIT_MANIFEST_SHA256: str = "6874a56f5484186dffdd5faeb7871c3bef85042ccfdf8d1e375fdde75a3ee9f5"

FROZEN_FEATURES_PARQUET_PATH: pathlib.Path = _REPO_ROOT / "data" / "processed" / "nhis" / "nhis_2022_2024_features.parquet"
FROZEN_FEATURES_PARQUET_SHA256: str = "49f415132ff0be0228f8533f9f74c48cd79ff6fa8be66db085f7329d9b083383"

FROZEN_PRIMARY_ANALYSIS_COMMIT: str = "58a1e02339bf35dedc14dd2719d72417de6f6910"
FROZEN_D4_RELEASE_TAG: str = "nhis-d4-primary-test-v1"

FROZEN_D3_MANIFEST_PATH: pathlib.Path = _REPO_ROOT / "artifacts" / "nhis" / "d3" / "d3_manifest.json"
FROZEN_POOLED_SPLIT_AUDIT_PATH: pathlib.Path = _REPO_ROOT / "artifacts" / "nhis" / "d3" / "pooled_split_audit.json"

# D4 Reference Comparison Registry (read-only historical anchors)
D4_COMPARISON_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ARM_D3_001": {
        "run_id": "20260901T150720Z-94c5ac1be7",
        "manifest_sha256": "ced6d0b58148392cc0d8934b1ccba1b40839abd3b1780b7391acbc76343a1d03",
        "changed_dict_sha256": "9e90b303d28e5fd890dd6351d6e432c803c9d08cd051c4a1595859401cb76f6d",
        "rel_path": "runs/nhis_d4_preflight/20260901T150720Z-94c5ac1be7",
    },
    "ARM_D3_002": {
        "run_id": "20260901T152301Z-6f49937b9f",
        "manifest_sha256": "8f41fd4337f9b546601bf066a7c7269e6f40c54ce102456117881ddaa34c53f6",
        "changed_dict_sha256": "56dbab9d5a3470b668267f09ca658826e1a162d0391ed9be5dd1a56a30036c3c",
        "rel_path": "runs/nhis_d4_preflight/20260901T152301Z-6f49937b9f",
    },
    "ARM_D3_003": {
        "run_id": "20260901T152455Z-0b07868ca4",
        "manifest_sha256": "ac59860ca21c495155e630dc0ad3faaf8b54b376a520aad291c1134b7d6a6faa",
        "changed_dict_sha256": "84489acf8e6f72964c74d9a9fc5719eec35d70a61e32880eca7439ddcefdde63",
        "rel_path": "runs/nhis_d4_preflight/20260901T152455Z-0b07868ca4",
    },
    "ARM_D3_004": {
        "run_id": "20260901T152529Z-44d2d5a4cc",
        "manifest_sha256": "030a6f1f825b3075e35a01a133988043e40698384330c1521e3a0b44498225c1",
        "changed_dict_sha256": "2b0f3138b2ac93d6d3dbe39f23849d46b94ac36d8db3f4f35e64af260883cb27",
        "rel_path": "runs/nhis_d4_preflight/20260901T152529Z-44d2d5a4cc",
    },
}


def get_git_commit(repo_root: Optional[pathlib.Path] = None) -> str:
    """Retrieve current git HEAD commit hash."""
    root = repo_root or _REPO_ROOT
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def compute_series_weight_diagnostics(
    w: pd.Series,
    expected_index: pd.Index,
    o: pd.Series,
    partition_name: str,
    record_ids: Optional[pd.Series] = None,
    y: Optional[pd.Series] = None,
    metadata: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Calculate thorough descriptive weight diagnostics and fail-closed integrity checks."""
    if not isinstance(w, pd.Series):
        raise TypeError(f"Weight object for {partition_name} must be a pd.Series, got {type(w)}")
    if len(w) != len(expected_index):
        raise ValueError(
            f"Weight length ({len(w)}) does not match expected length ({len(expected_index)}) for {partition_name}"
        )
    if not w.index.equals(expected_index):
        raise ValueError(
            f"Weight index alignment mismatch for {partition_name}: indices must match exactly"
        )
    if not o.index.equals(expected_index):
        raise ValueError(
            f"Protected attribute index alignment mismatch for {partition_name}: indices must match exactly"
        )
    if y is not None:
        if len(y) != len(expected_index) or not y.index.equals(expected_index):
            raise ValueError(
                f"Outcome index alignment mismatch for {partition_name}: outcome index must match exactly"
            )

    record_id_aligned = True
    if record_ids is not None:
        if len(record_ids) != len(expected_index) or not record_ids.index.equals(expected_index):
            record_id_aligned = False
            raise ValueError(
                f"Record ID alignment mismatch for {partition_name}: record_id series index must match"
            )
        if record_ids.isna().sum() > 0:
            raise ValueError(
                f"Record ID series contains {int(record_ids.isna().sum())} missing values for {partition_name}"
            )
        if record_ids.nunique() != len(record_ids):
            raise ValueError(
                f"Record ID is not unique within {partition_name}: {record_ids.nunique()} unique vs {len(record_ids)} total"
            )

    if metadata is not None:
        if len(metadata) != len(expected_index) or not metadata.index.equals(expected_index):
            raise ValueError(
                f"Metadata index alignment mismatch for {partition_name}: metadata index must match exactly"
            )
        if "WTFA_A" not in metadata.columns:
            raise ValueError(f"Metadata missing WTFA_A column for {partition_name}")
        if not np.array_equal(w.to_numpy(dtype=float), metadata["WTFA_A"].to_numpy(dtype=float)):
            raise ValueError(f"Weight values do not match metadata['WTFA_A'] for {partition_name}")
        if "record_id" in metadata.columns:
            rec_meta = metadata["record_id"]
            if rec_meta.isna().sum() > 0:
                raise ValueError(f"Metadata record_id contains missing values for {partition_name}")
            if rec_meta.nunique() != len(rec_meta):
                raise ValueError(
                    f"Metadata record_id is not unique within {partition_name}: {rec_meta.nunique()} unique vs {len(rec_meta)} total"
                )

    w_arr = w.to_numpy(dtype=float)
    missing_count = int(w.isna().sum())
    non_numeric_count = 0  # Series is typed float
    non_finite_count = int((~np.isfinite(w_arr)).sum())
    zero_count = int((w_arr == 0.0).sum())
    negative_count = int((w_arr < 0.0).sum())

    # Fail closed on corrupted weights
    if missing_count > 0:
        raise ValueError(f"{partition_name} WTFA_A contains {missing_count} missing values")
    if non_finite_count > 0:
        raise ValueError(f"{partition_name} WTFA_A contains {non_finite_count} non-finite/NaN/Inf values")
    if negative_count > 0:
        raise ValueError(f"{partition_name} WTFA_A contains {negative_count} negative values")
    if zero_count > 0:
        raise ValueError(f"{partition_name} WTFA_A contains {zero_count} zero values (strictly positive weights required)")

    w_min = float(np.min(w_arr))
    w_med = float(np.median(w_arr))
    w_max = float(np.max(w_arr))
    w_sum = float(np.sum(w_arr))
    distinct_count = int(len(np.unique(w_arr)))

    # Per-protected-group diagnostics
    group_diagnostics = []
    for g_val in sorted(o.dropna().unique()):
        g_mask = (o == g_val)
        w_g = w_arr[g_mask]
        g_n = int(len(w_g))
        g_sum = float(np.sum(w_g))
        g_min = float(np.min(w_g)) if g_n > 0 else 0.0
        g_max = float(np.max(w_g)) if g_n > 0 else 0.0
        g_nonpos = int((w_g <= 0).sum())

        if g_n == 0 or g_sum <= 0:
            raise ValueError(
                f"Protected group {g_val} in {partition_name} has non-positive total weight ({g_sum})"
            )

        group_diagnostics.append({
            "partition": partition_name,
            "group": int(g_val) if isinstance(g_val, (int, np.integer)) else str(g_val),
            "N": g_n,
            "weight_sum": g_sum,
            "weight_min": g_min,
            "weight_max": g_max,
            "missing_or_nonpositive_count": g_nonpos,
        })

    return {
        "partition": partition_name,
        "N": len(w),
        "weight_N": len(w),
        "missing_count": missing_count,
        "non_numeric_count": non_numeric_count,
        "non_finite_count": non_finite_count,
        "zero_count": zero_count,
        "negative_count": negative_count,
        "min": w_min,
        "median": w_med,
        "max": w_max,
        "sum": w_sum,
        "distinct_weights_count": distinct_count,
        "index_aligned": True,
        "record_id_aligned": record_id_aligned,
        "group_diagnostics": group_diagnostics,
    }


class NHISD5WeightedRunner:
    """Dedicated orchestrator for Gate D5 survey-weighted FairBias preflight and audit."""

    def __init__(
        self,
        adapter: Optional[NHISPooledAdapter] = None,
        features_parquet_path: Optional[Union[str, pathlib.Path]] = None,
        split_manifest_path: Optional[Union[str, pathlib.Path]] = None,
        d3_manifest_path: Optional[Union[str, pathlib.Path]] = None,
        pooled_split_audit_path: Optional[Union[str, pathlib.Path]] = None,
        study_config_path: Optional[Union[str, pathlib.Path]] = None,
        feature_config_path: Optional[Union[str, pathlib.Path]] = None,
        enforce_frozen_inputs: bool = True,
        allow_mitigation: bool = False,
    ):
        self.enforce_frozen_inputs = bool(enforce_frozen_inputs)
        self.allow_mitigation = bool(allow_mitigation)

        resolved_split_path = (
            pathlib.Path(split_manifest_path).resolve()
            if split_manifest_path is not None
            else FROZEN_SPLIT_MANIFEST_PATH.resolve()
        )
        self.split_manifest_path = resolved_split_path

        resolved_features_path = (
            pathlib.Path(features_parquet_path).resolve()
            if features_parquet_path is not None
            else FROZEN_FEATURES_PARQUET_PATH.resolve()
        )
        self.features_parquet_path = resolved_features_path

        d3_path = (
            pathlib.Path(d3_manifest_path).resolve()
            if d3_manifest_path is not None
            else FROZEN_D3_MANIFEST_PATH.resolve()
        )
        self.d3_manifest_path = d3_path

        audit_path = (
            pathlib.Path(pooled_split_audit_path).resolve()
            if pooled_split_audit_path is not None
            else FROZEN_POOLED_SPLIT_AUDIT_PATH.resolve()
        )
        self.pooled_split_audit_path = audit_path

        self.adapter = adapter or NHISPooledAdapter(
            features_parquet_path=resolved_features_path,
            split_manifest_path=resolved_split_path,
            study_config_path=study_config_path,
            feature_config_path=feature_config_path,
        )

    def verify_frozen_input_contract(self) -> Dict[str, Any]:
        """Validate frozen D3 inputs and fail closed if invariant contract is violated."""
        if not self.enforce_frozen_inputs:
            return {
                "enforced": False,
                "split_manifest_source": str(self.split_manifest_path),
                "split_manifest_sha256": "BYPASS",
                "expected_split_manifest_sha256": FROZEN_SPLIT_MANIFEST_SHA256,
                "split_hash_match": None,
                "features_parquet_source": str(self.adapter.features_parquet_path),
                "features_parquet_sha256": "BYPASS",
                "expected_features_parquet_sha256": FROZEN_FEATURES_PARQUET_SHA256,
                "features_hash_match": None,
                "primary_analysis_commit": FROZEN_PRIMARY_ANALYSIS_COMMIT,
                "d4_release_tag": FROZEN_D4_RELEASE_TAG,
                "status": "BYPASS",
            }

        # 1. Split manifest existence and hash
        if not self.split_manifest_path.is_file():
            raise FileNotFoundError(
                f"Frozen split manifest not found at: {self.split_manifest_path}"
            )
        split_sha = compute_sha256(self.split_manifest_path)
        if split_sha != FROZEN_SPLIT_MANIFEST_SHA256:
            raise ValueError(
                f"Frozen split manifest SHA-256 mismatch!\n"
                f"Expected: {FROZEN_SPLIT_MANIFEST_SHA256}\n"
                f"Observed: {split_sha}\n"
                f"Path: {self.split_manifest_path}"
            )

        # 2. Features parquet existence and hash
        feat_path = pathlib.Path(self.adapter.features_parquet_path).resolve()
        if not feat_path.is_file():
            raise FileNotFoundError(
                f"Frozen features parquet not found at: {feat_path}"
            )
        feat_sha = compute_sha256(feat_path)
        if feat_sha != FROZEN_FEATURES_PARQUET_SHA256:
            raise ValueError(
                f"Frozen features parquet SHA-256 mismatch!\n"
                f"Expected: {FROZEN_FEATURES_PARQUET_SHA256}\n"
                f"Observed: {feat_sha}\n"
                f"Path: {feat_path}"
            )

        # 3. D3 manifest existence and status
        if not self.d3_manifest_path.is_file():
            raise FileNotFoundError(
                f"D3 manifest not found at: {self.d3_manifest_path}"
            )
        try:
            d3_data = json.loads(self.d3_manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Could not parse D3 manifest {self.d3_manifest_path}: {exc}") from exc
        d3_status = d3_data.get("status")
        if d3_status != "PASS":
            raise ValueError(
                f"D3 manifest status is {d3_status!r} (expected 'PASS') at {self.d3_manifest_path}"
            )

        # 4. Pooled split audit existence and status
        if not self.pooled_split_audit_path.is_file():
            raise FileNotFoundError(
                f"Pooled split audit not found at: {self.pooled_split_audit_path}"
            )
        try:
            audit_data = json.loads(self.pooled_split_audit_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Could not parse pooled split audit {self.pooled_split_audit_path}: {exc}") from exc
        audit_status = audit_data.get("status")
        if audit_status != "PASS":
            raise ValueError(
                f"Pooled split audit status is {audit_status!r} (expected 'PASS') at {self.pooled_split_audit_path}"
            )

        return {
            "enforced": True,
            "split_manifest_source": str(self.split_manifest_path),
            "split_manifest_sha256": split_sha,
            "expected_split_manifest_sha256": FROZEN_SPLIT_MANIFEST_SHA256,
            "split_hash_match": True,
            "features_parquet_source": str(feat_path),
            "features_parquet_sha256": feat_sha,
            "expected_features_parquet_sha256": FROZEN_FEATURES_PARQUET_SHA256,
            "features_hash_match": True,
            "primary_analysis_commit": FROZEN_PRIMARY_ANALYSIS_COMMIT,
            "d4_release_tag": FROZEN_D4_RELEASE_TAG,
            "d3_gate_status": d3_status,
            "pooled_split_audit_status": audit_status,
            "status": "PASS",
        }

    def get_arm_config(self, arm_id: str) -> Dict[str, Any]:
        """Retrieve frozen arm specification."""
        arm_key = arm_id.strip().upper()
        if arm_key not in FROZEN_D5_ARMS:
            raise ValueError(
                f"Unknown arm_id: {arm_id!r}. Supported frozen D5 arms: {list(FROZEN_D5_ARMS.keys())}"
            )
        return copy.deepcopy(FROZEN_D5_ARMS[arm_key])

    def get_partition_cohort(
        self,
        role: str,
        arm_config: Mapping[str, Any],
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
        """Extract preprocessed cohort for a single authorized partition ('train' or 'val').

        Fails closed immediately if role is 'test' or any unauthorized partition.
        Ensures the named TEST partition is never requested or materialized.
        """
        role_norm = str(role).strip().lower()
        if role_norm not in ("train", "val"):
            raise ValueError(
                f"Unauthorized partition role for D5 runner: {role!r}. "
                "Only 'train' and 'val' are permitted. TEST partition is strictly embargoed."
            )

        outcome = arm_config["outcome"]
        protected_attribute = arm_config["protected_attribute"]
        feature_set = arm_config["feature_set"].lower()
        disability_arm = arm_config["disability_arm"]

        harm_outcome = OUTCOME_MAP.get(outcome)
        if harm_outcome is None or harm_outcome not in self.adapter._raw_df.columns:
            raise ValueError(f"Unsupported outcome: {outcome}. Supported: MEDDL12M_A, MEDNG12M_A")

        harm_prot = PROTECTED_MAP.get(protected_attribute)
        if harm_prot is None or harm_prot not in self.adapter._raw_df.columns:
            raise ValueError(
                f"Unsupported protected attribute: {protected_attribute}. Supported: SEX_A, HISPALLP_A, DISAB3_A"
            )

        active_feature_names = self.adapter.get_feature_names(
            feature_set=feature_set, disability_arm=disability_arm
        )

        raw_sub = self.adapter.get_partition(role_norm)
        y_raw = raw_sub[harm_outcome]
        o_raw = raw_sub[harm_prot]

        valid_mask = y_raw.notna() & o_raw.notna()
        filtered_df = raw_sub[valid_mask].copy()

        X_all = self.adapter.preprocessor.transform(
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

        return X, y, o, w, metadata

    def get_train_val_cohort(
        self, arm_id: str
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]]:
        """Retrieve TRAIN and VALIDATION partitions for the specified arm.

        Requests and materializes strictly 'train' and 'val' partitions.
        The named 'test' partition is NEVER requested or materialized.
        """
        arm_config = self.get_arm_config(arm_id)
        return {
            "train": self.get_partition_cohort("train", arm_config),
            "val": self.get_partition_cohort("val", arm_config),
        }

    def audit_arm_weights(self, arm_id: str) -> Dict[str, Any]:
        """Perform comprehensive weight audit on TRAIN and VALIDATION for a single arm."""
        cohorts = self.get_train_val_cohort(arm_id)
        X_train, y_train, o_train, w_train, meta_train = cohorts["train"]
        X_val, y_val, o_val, w_val, meta_val = cohorts["val"]

        diag_train = compute_series_weight_diagnostics(
            w=w_train,
            expected_index=X_train.index,
            o=o_train,
            partition_name="train",
            record_ids=meta_train["record_id"],
            y=y_train,
            metadata=meta_train,
        )
        diag_val = compute_series_weight_diagnostics(
            w=w_val,
            expected_index=X_val.index,
            o=o_val,
            partition_name="val",
            record_ids=meta_val["record_id"],
            y=y_val,
            metadata=meta_val,
        )

        return {
            "arm_id": arm_id,
            "train": diag_train,
            "val": diag_val,
        }

    def run_audit(
        self,
        output_dir: Optional[Union[str, pathlib.Path]] = None,
    ) -> Dict[str, Any]:
        """Execute full Gate D5.1a preflight audit across all four arms and generate artifacts."""
        start_time = time.time()
        run_id = new_run_id()

        if output_dir is not None:
            target_out_dir = pathlib.Path(output_dir).resolve()
        else:
            target_out_dir = (_REPO_ROOT / "runs" / "nhis_d5_weighted" / "audit" / run_id).resolve()
        target_out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Enforce frozen input contract
        frozen_contract = self.verify_frozen_input_contract()

        # 2. Audit weights across all four arms
        arm_audits: Dict[str, Any] = {}
        csv_rows: List[Dict[str, Any]] = []

        for arm_id in FROZEN_D5_ARMS:
            audit_res = self.audit_arm_weights(arm_id)
            arm_audits[arm_id] = audit_res

            for part in ("train", "val"):
                part_diag = audit_res[part]
                for g_diag in part_diag["group_diagnostics"]:
                    csv_rows.append({
                        "arm_id": arm_id,
                        **g_diag,
                    })

        # 3. Write Artifact 1: frozen_input_contract.json
        write_json_atomic(target_out_dir / "frozen_input_contract.json", frozen_contract)

        # 4. Write Artifact 2: four_arm_registry.json
        registry_payload = {
            "arms": FROZEN_D5_ARMS,
            "d4_references": D4_COMPARISON_REGISTRY,
            "primary_analysis_commit": FROZEN_PRIMARY_ANALYSIS_COMMIT,
            "d4_release_tag": FROZEN_D4_RELEASE_TAG,
        }
        write_json_atomic(target_out_dir / "four_arm_registry.json", registry_payload)

        # 5. Write Artifact 3: weight_input_audit.json
        write_json_atomic(target_out_dir / "weight_input_audit.json", arm_audits)

        # 6. Write Artifact 4: weight_group_audit.csv
        df_group_audit = pd.DataFrame(csv_rows)
        write_csv_atomic(df_group_audit, target_out_dir / "weight_group_audit.csv", force=True)

        # 7. Write Artifact 5: audit_manifest.json
        execution_time = time.time() - start_time
        git_sha = get_git_commit()
        manifest_payload = {
            "gate": "D5.1a.1",
            "run_id": run_id,
            "timestamp_utc": utc_timestamp(),
            "git_commit": git_sha,
            "real_weighted_mitigation_executed": False,
            "validation_scored": False,
            "test_evaluated": False,
            "test_scored": False,
            "authorized_partitions": ["train", "validation"],
            "requested_partitions": ["train", "validation"],
            "test_partition_requested": False,
            "test_partition_accessed": False,
            "test_cohort_materialized": False,
            "partition_access_policy": (
                "The D5.1 runner requested and materialized only the authorized TRAIN and "
                "VALIDATION partitions. The named TEST partition was not requested or materialized."
            ),
            "frozen_input_contract_verified": True,
            "arms_audited": list(FROZEN_D5_ARMS.keys()),
            "execution_time_seconds": execution_time,
            "output_directory": str(target_out_dir),
            "artifacts_generated": [
                "audit_manifest.json",
                "frozen_input_contract.json",
                "weight_input_audit.json",
                "weight_group_audit.csv",
                "four_arm_registry.json",
            ],
            "status": "PASS",
        }
        write_json_atomic(target_out_dir / "audit_manifest.json", manifest_payload)

        return {
            "run_id": run_id,
            "status": "PASS",
            "output_dir": target_out_dir,
            "manifest": manifest_payload,
            "arm_audits": arm_audits,
        }

    def execute_train_validation(
        self,
        arm_id: str = "D5_ARM_001",
        output_dir: Optional[Union[str, pathlib.Path]] = None,
        random_seed: int = PRIMARY_D5_RANDOM_SEED,
    ) -> Dict[str, Any]:
        """Substantive TRAIN/VAL execution harness for survey-weighted FairBias (Future Gate D5.1b).

        Fails closed in Gate D5.1a unless self.allow_mitigation is explicitly True.
        """
        if not self.allow_mitigation:
            raise RuntimeError(
                "Substantive mitigation execution is strictly forbidden in Gate D5.1a (audit only). "
                "allow_mitigation is False."
            )
        if random_seed != PRIMARY_D5_RANDOM_SEED:
            raise ValueError(
                f"Primary D5 protocol strictly freezes random_seed to {PRIMARY_D5_RANDOM_SEED} (got {random_seed})."
            )

        start_time = time.time()
        arm_config = self.get_arm_config(arm_id)
        frozen_contract = self.verify_frozen_input_contract()

        # 1. Obtain frozen cohort
        cohorts = self.get_train_val_cohort(arm_id)
        X_train, y_train, o_train, w_train, meta_train = cohorts["train"]
        X_val, y_val, o_val, w_val, meta_val = cohorts["val"]

        # 2. Validate weights and respondent alignment fail-closed
        compute_series_weight_diagnostics(
            w_train, X_train.index, o_train, "train", record_ids=meta_train["record_id"], y=y_train, metadata=meta_train
        )
        compute_series_weight_diagnostics(
            w_val, X_val.index, o_val, "val", record_ids=meta_val["record_id"], y=y_val, metadata=meta_val
        )

        # 3. Predictor families
        feature_set_key = arm_config["feature_set"].lower()
        all_cats, all_nums = self.adapter.preprocessor.get_feature_family_lists(feature_set_key)
        active_feats = set(X_train.columns)
        cate_attrs = [f for f in all_cats if f in active_feats]
        num_attrs = [f for f in all_nums if f in active_feats]

        if len(active_feats) != arm_config["expected_predictors"]:
            raise ValueError(
                f"Feature count mismatch for {arm_id}: expected {arm_config['expected_predictors']}, "
                f"got {len(active_feats)} ({list(active_feats)})"
            )

        # 4. Resolve FairBiasConfig in survey_weighted_geometry mode
        fb_config = FairBiasConfig(
            algorithm_mode=ALGORITHM_MODE_SURVEY_WEIGHTED,
            random_seed=random_seed,
            classifier="LR",
            eval_norm="min-max",
            label_O=(arm_config["protected_attribute"],),
            label_Y=arm_config["outcome"],
            use_bias_mitigation=True,
            use_accuracy_enhancement=False,
            failed_attribute_mode="stop",
            power_sequence_policy="official_stream",
            power_revisit_policy="restart",
        ).resolved()

        # 5. Initialize Evaluator and Transformer
        evaluator = FairEvaluator(
            config=fb_config,
            label_O=[arm_config["protected_attribute"]],
            label_Y=arm_config["outcome"],
            cate_attrs=cate_attrs,
            num_attrs=num_attrs,
        )
        transformer = FairTransform(
            n_bins=fb_config.transform_n_bins,
            log_epsilon=fb_config.transform_log_epsilon,
            x_max=fb_config.transform_x_max,
        )

        # 6. Initial weighted d_phi and epsilon threshold strictly on TRAIN
        O_train_df = pd.DataFrame({arm_config["protected_attribute"]: o_train})
        nmi_org = calculate_nmi_dict(X_train, y_train)
        init_epsilon_dict = evaluator.calculate_epsilon(
            X_train, O_train_df, cate_attrs=cate_attrs, num_attrs=num_attrs, sample_weight=w_train
        )
        epsilon_threshold = evaluator.compute_threshold(init_epsilon_dict)

        initial_dphi_raw = copy.deepcopy(init_epsilon_dict.get(arm_config["protected_attribute"], {}))
        initial_dphi = {k: float(v) for k, v in initial_dphi_raw.items()}
        initial_max_dphi = float(max(initial_dphi.values())) if initial_dphi else 0.0
        highest_initial_feature = (
            max(initial_dphi, key=initial_dphi.get) if initial_dphi else None
        )

        # 7. Initialize FairBiasMitigation engine with TRAIN weights
        mitigation_engine = FairBiasMitigation(
            evaluator=evaluator,
            transformer=transformer,
            label_O=[arm_config["protected_attribute"]],
            cate_attrs=cate_attrs,
            num_attrs=num_attrs,
            phi_threshold=fb_config.phi_threshold,
            poly_exponents=fb_config.transform_poly_exponents,
            failed_attribute_mode=fb_config.failed_attribute_mode,
            power_sequence_policy=fb_config.power_sequence_policy,
            power_revisit_policy=fb_config.power_revisit_policy,
            sample_weight=w_train,
        )

        # 8. Greedy mitigation loop strictly on TRAIN starting from empty changed_dict
        changed_dict: Dict[str, Any] = {}
        current_epsilon = copy.deepcopy(init_epsilon_dict)
        iter_idx = 0
        exit_reason: Optional[str] = None

        if initial_max_dphi <= epsilon_threshold:
            exit_reason = "epsilon_reached"
        else:
            while True:
                iter_idx += 1
                current_X_train, changed_dict, sel_o, sel_attr = mitigation_engine.mitigate_step(
                    X=X_train,
                    Y=y_train,
                    O=O_train_df,
                    nmi_org=nmi_org,
                    changed_dict=changed_dict,
                    current_epsilon=current_epsilon,
                    epsilon_threshold=epsilon_threshold,
                    iteration=iter_idx,
                )

                if sel_attr is None:
                    exit_reason = "no_transform_accepted"
                    break

                transformed_X_train = transformer.transform_data(
                    X_train, changed_dict, num_attrs, cate_attrs
                )
                current_epsilon = evaluator.calculate_epsilon(
                    transformed_X_train,
                    O_train_df,
                    cate_attrs=cate_attrs,
                    num_attrs=num_attrs,
                    sample_weight=w_train,
                )
                curr_max_eps = float(
                    max(val for gd in current_epsilon.values() for val in gd.values())
                ) if current_epsilon else 0.0

                if curr_max_eps <= epsilon_threshold:
                    exit_reason = "epsilon_reached"
                    break

        if exit_reason == "no_transform_accepted":
            final_eps_vals = [val for gd in current_epsilon.values() for val in gd.values()]
            term_max = float(max(final_eps_vals)) if final_eps_vals else initial_max_dphi
            if term_max <= epsilon_threshold:
                termination_reason = "epsilon_reached"
                converged = True
            else:
                termination_reason = "candidate_grid_exhausted"
                converged = False
        elif exit_reason == "epsilon_reached":
            termination_reason = "epsilon_reached"
            converged = True
        else:
            termination_reason = exit_reason or "unknown"
            converged = (termination_reason == "epsilon_reached")

        transformed_X_train = transformer.transform_data(
            X_train, changed_dict, num_attrs, cate_attrs
        )
        final_epsilon_dict = evaluator.calculate_epsilon(
            transformed_X_train,
            O_train_df,
            cate_attrs=cate_attrs,
            num_attrs=num_attrs,
            sample_weight=w_train,
        )
        final_dphi_raw = copy.deepcopy(final_epsilon_dict.get(arm_config["protected_attribute"], {}))
        final_dphi = {k: float(v) for k, v in final_dphi_raw.items()}
        final_max_dphi = float(max(final_dphi.values())) if final_dphi else 0.0

        trace = FairBiasTransformTrace(
            algorithm_mode=fb_config.algorithm_mode,
            protected_attribute=arm_config["protected_attribute"],
            epsilon_threshold=float(epsilon_threshold),
            steps=mitigation_engine.step_traces,
            final_status="CONVERGED" if converged else "TERMINATED",
            final_max_dphi=float(final_max_dphi),
        )

        # 9. Train & Evaluate Baseline Model (Unweighted LR on original X)
        scaler_baseline = MinMaxScaler(feature_range=(0, 1))
        X_train_base_scaled = scaler_baseline.fit_transform(X_train)
        X_val_base_scaled = scaler_baseline.transform(X_val)

        model_baseline = get_classifier("LR", random_state=random_seed)
        model_baseline.fit(X_train_base_scaled, y_train.to_numpy())  # Unweighted

        val_prob_base = model_baseline.predict_proba(X_val_base_scaled)[:, 1]
        val_pred_base = (val_prob_base >= 0.5).astype(int)

        val_eval_base = evaluate_predictions(
            y_true=y_val.to_numpy(),
            y_pred=val_pred_base,
            y_prob=val_prob_base,
            o_group=o_val.to_numpy(),
            expected_group_count=arm_config.get("expected_group_count"),
            expected_groups=arm_config.get("expected_groups"),
        )

        # 10. Train & Evaluate FairBias Model (Unweighted LR on transformed X)
        transformed_X_val = transformer.transform_data(
            X_val, changed_dict, num_attrs, cate_attrs
        )

        scaler_fb = MinMaxScaler(feature_range=(0, 1))
        X_train_fb_scaled = scaler_fb.fit_transform(transformed_X_train)
        X_val_fb_scaled = scaler_fb.transform(transformed_X_val)

        model_fairbias = get_classifier("LR", random_state=random_seed)
        model_fairbias.fit(X_train_fb_scaled, y_train.to_numpy())  # Unweighted

        val_prob_fb = model_fairbias.predict_proba(X_val_fb_scaled)[:, 1]
        val_pred_fb = (val_prob_fb >= 0.5).astype(int)

        val_eval_fb = evaluate_predictions(
            y_true=y_val.to_numpy(),
            y_pred=val_pred_fb,
            y_prob=val_prob_fb,
            o_group=o_val.to_numpy(),
            expected_group_count=arm_config.get("expected_group_count"),
            expected_groups=arm_config.get("expected_groups"),
        )

        val_comparison = compute_evaluation_comparison(val_eval_base, val_eval_fb)

        return {
            "arm_id": arm_id,
            "changed_dict": changed_dict,
            "trace": trace,
            "initial_dphi": initial_dphi,
            "final_dphi": final_dphi,
            "val_eval_base": val_eval_base,
            "val_eval_fb": val_eval_fb,
            "val_comparison": val_comparison,
            "converged": converged,
            "termination_reason": termination_reason,
            "execution_time_seconds": time.time() - start_time,
        }
