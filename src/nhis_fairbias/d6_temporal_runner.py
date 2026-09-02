"""Dedicated NHIS temporal robustness TRAIN/VALIDATION runner and Gate D6.0a harness.

Guarantees & Scientific Invariants:
1. Repeated Cross-Sectional Temporal Robustness:
   - Evaluates whether preprocessing, FairBias representation learning, scaling,
     and classifier fitting learned strictly on 2022 transport forward in time to
     2023 (validation) and 2024 (future test) without relearning or tuning.
   - Repeated cross-sectional design; NOT longitudinal patient follow-up, prospective
     causal inference, or panel analysis.
   - repeated_cross_sectional = true, longitudinal = false, causal_analysis = false,
     temporal_robustness_analysis = true.
2. Temporal Partition Contract:
   - 2022 = development_train (preprocessing fit, FairBias epsilon/NMI/mitigation, scalers, LR)
   - 2023 = development_validation (transform only, evaluation only, diagnostic d_phi only)
   - 2024 = frozen_test (completely embargoed in D6.0; NEVER requested or evaluated)
   - Random splitting strictly forbidden; pooled adapter and split utilities are absent.
3. Strict Unweighted Semantics:
   - Core FairBias paper-faithful application; no survey weighting in geometry, classifier,
     or evaluation.
   - sample_weight is None for FairBias d_phi, LogisticRegression, MinMaxScaler, and evaluation.
4. Frozen Predictor & Outcome Semantics:
   - Primary outcome: MEDDL12M_A (substantive 1/2 recoded to 1/0, never imputed).
   - MEDNG12M_A strictly excluded from D6.
   - PRIMARY_CORE: exactly 21 features (18 categorical, 3 numerical).
   - Leakage blacklist enforced.
5. Exact Four Predeclared Arms:
   - D6_ARM_001: SEX_A (PRIMARY_CORE, full_feature, 21 preds, 2 groups, 1 pair)
   - D6_ARM_002: HISPALLP_A (PRIMARY_CORE, full_feature, 21 preds, 7 groups, 21 pairs)
   - D6_ARM_003: DISAB3_A (PRIMARY_CORE, full_feature, 21 preds, 2 groups, 1 pair)
   - D6_ARM_004: DISAB3_A (PRIMARY_CORE, exclude_disability_components, 15 preds, 2 groups, 1 pair)
6. Strict Order of Operations:
   - 2022 cohort requested -> preprocessing verified -> epsilon determined -> NMI computed ->
     FairBias mitigation executed -> changed_dict frozen -> final 2022 d_phi computed ->
     baseline scaler/LR fitted -> FairBias scaler/LR fitted.
   - ONLY AFTER all 2022 steps complete is the 2023 cohort requested.
   - 2023 receives frozen transform and evaluation only.
   - 2024 is NEVER requested.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import pathlib
import subprocess
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from fairbias.config import (
    ALGORITHM_MODE_PAPER_FAITHFUL,
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
    DISABILITY_COMPONENTS,
    OUTCOME_MAP,
    PROTECTED_MAP,
    STUDY_YEAR_ROLES,
    NHISStudyAdapter,
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
    NHISFeatureError,
    load_feature_registry,
)
from .preprocessing import (
    NHISLeakageError,
    NHISPreprocessingError,
    NHISPreprocessor,
    PreprocessingFitRecord,
)
from .schema import (
    DEFAULT_STUDY_CONFIG,
    load_study_config,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

PRIMARY_D6_RANDOM_SEED: int = 0
DEFAULT_PREDICTION_THRESHOLD: float = 0.5
CLASSIFIER_SOLVER: str = "lbfgs"
CLASSIFIER_MAX_ITER: int = 1000

# Input data contract
FROZEN_FEATURES_PARQUET_PATH: pathlib.Path = (
    _REPO_ROOT / "data" / "processed" / "nhis" / "nhis_2022_2024_features.parquet"
)
FROZEN_FEATURES_PARQUET_SHA256: str = (
    "49f415132ff0be0228f8533f9f74c48cd79ff6fa8be66db085f7329d9b083383"
)

# Expected row counts and substantive outcome numbers
EXPECTED_YEAR_ROW_COUNTS: Dict[int, int] = {
    2022: 27651,
    2023: 29522,
    2024: 32629,
}
EXPECTED_TOTAL_ROW_COUNT: int = 89802

EXPECTED_MEDDL12M_TOTALS: Dict[int, Dict[str, int]] = {
    2022: {"positive": 1770, "negative": 25683, "missing": 198},
    2023: {"positive": 1931, "negative": 27352, "missing": 239},
    2024: {"positive": 2564, "negative": 29791, "missing": 274},
}

# Temporal roles
TEMPORAL_TRAIN_YEAR: int = 2022
TEMPORAL_VALIDATION_YEAR: int = 2023
TEMPORAL_TEST_YEAR: int = 2024

TEMPORAL_STUDY_ROLES: Dict[int, str] = {
    2022: "development_train",
    2023: "development_validation",
    2024: "frozen_test",
}

# Prior release anchors
FROZEN_D4_TAG: str = "nhis-d4-primary-test-v1"
FROZEN_D4_TAG_OBJECT: str = "d74af83fb98a3805a7fb0767c5beeb3dbf1407e4"
FROZEN_D4_ARCHIVED_COMMIT: str = "bb1f0601bf3a079bb09882003740df8a78bc0374"

FROZEN_D5_TRAIN_VAL_TAG: str = "nhis-d5-weighted-train-val-v1"
FROZEN_D5_TRAIN_VAL_TAG_OBJECT: str = "4dcd6106f2d40de8d6bd991ebc9a61850b3cd70a"
FROZEN_D5_TRAIN_VAL_ARCHIVED_COMMIT: str = "14cc7aa629e20b326773621c164f69e3ef3acfdb"

FROZEN_D5_SECONDARY_TEST_TAG: str = "nhis-d5-weighted-secondary-test-v1"
FROZEN_D5_SECONDARY_TEST_TAG_OBJECT: str = "2be0d13ba2440eab09d970feca922c114b779a8d"
FROZEN_D5_SECONDARY_TEST_ARCHIVED_COMMIT: str = "f63f58918b89a276152b5948df9874bd9b2fc94e"

PRIOR_RELEASE_ARCHIVE_DIRS: Tuple[str, ...] = (
    "docs/releases/NHIS_D4_PRIMARY_TEST_RELEASE_V1_9920fc5a",
    "docs/releases/NHIS_D5_WEIGHTED_TRAIN_VAL_V1_bc6034e5",
    "docs/releases/NHIS_D5_WEIGHTED_SECONDARY_TEST_V1_14cc7aa6",
)

SCIENTIFIC_EXECUTION_BASE_COMMIT: str = "f63f58918b89a276152b5948df9874bd9b2fc94e"

FROZEN_SCIENTIFIC_PATHS_SPEC: Tuple[str, ...] = (
    "src/fairbias/**",
    "src/nhis_fairbias/adapter.py",
    "src/nhis_fairbias/preprocessing.py",
    "src/nhis_fairbias/evaluation.py",
    "src/nhis_fairbias/features.py",
    "src/nhis_fairbias/schema.py",
)

FROZEN_SCIENTIFIC_PATHS_DIFF: Tuple[str, ...] = (
    "src/fairbias",
    "src/nhis_fairbias/adapter.py",
    "src/nhis_fairbias/preprocessing.py",
    "src/nhis_fairbias/evaluation.py",
    "src/nhis_fairbias/features.py",
    "src/nhis_fairbias/schema.py",
)

# 2024 Disclosure Statement (Mandatory from Section 3)
TEMPORAL_2024_DISCLOSURE: str = (
    "D6 is a predeclared temporal robustness analysis using 2022 for development training, "
    "2023 for temporal validation, and 2024 for forward-time evaluation. However, 2024 respondents "
    "were previously included in the pooled D4/D5 analyses, so the 2024 partition is not a "
    "globally untouched primary holdout. D6-specific transformations and models are nevertheless "
    "frozen before D6-specific 2024 evaluation."
)

# Four Frozen Scientific Arms for D6
FROZEN_D6_ARMS: Dict[str, Dict[str, Any]] = {
    "D6_ARM_001": {
        "arm_id": "D6_ARM_001",
        "outcome": "MEDDL12M_A",
        "protected_attribute": "SEX_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "full_feature",
        "expected_predictors": 21,
        "expected_groups": [1, 2],
        "expected_group_count": 2,
        "expected_pair_count": 1,
    },
    "D6_ARM_002": {
        "arm_id": "D6_ARM_002",
        "outcome": "MEDDL12M_A",
        "protected_attribute": "HISPALLP_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "full_feature",
        "expected_predictors": 21,
        "expected_groups": [1, 2, 3, 4, 5, 6, 7],
        "expected_group_count": 7,
        "expected_pair_count": 21,
    },
    "D6_ARM_003": {
        "arm_id": "D6_ARM_003",
        "outcome": "MEDDL12M_A",
        "protected_attribute": "DISAB3_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "full_feature",
        "expected_predictors": 21,
        "expected_groups": [1, 2],
        "expected_group_count": 2,
        "expected_pair_count": 1,
    },
    "D6_ARM_004": {
        "arm_id": "D6_ARM_004",
        "outcome": "MEDDL12M_A",
        "protected_attribute": "DISAB3_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "exclude_disability_components",
        "expected_predictors": 15,
        "expected_groups": [1, 2],
        "expected_group_count": 2,
        "expected_pair_count": 1,
    },
}

REQUIRED_PER_ARM_ARTIFACTS: Tuple[str, ...] = (
    "arm_config.json",
    "input_provenance.json",
    "train_dphi_before_after.json",
    "train_fairbias_trace.json",
    "final_changed_dict.json",
    "validation_dphi_before_after.json",
    "validation_metrics_baseline.json",
    "validation_metrics_fairbias.json",
    "validation_group_metrics_baseline.csv",
    "validation_group_metrics_fairbias.csv",
    "validation_comparison.json",
)

EXCLUDED_DISABILITY_COMPONENTS: Tuple[str, ...] = (
    "visiondf_a",
    "hearingdf_a",
    "diff_a",
    "comdiff_a",
    "uppslfcr_a",
    "cogmemdff_a",
)

LEAKAGE_FORBIDDEN_PREDICTORS: Tuple[str, ...] = (
    "MEDDL12M_A",
    "MEDNG12M_A",
    "PAYBLL12M_A",
    "PAYNOBLLNW_A",
    "PAYWORRY_A",
    "DENDL12M_A",
    "DENNG12M_A",
    "RXSK12M_A",
    "RXLS12M_A",
    "RXDL12M_A",
    "RXDG12M_A",
)


def compute_cohort_source_row_digest(survey_year: int, index: pd.Index) -> str:
    """Compute deterministic SHA-256 digest of ordered (survey_year, original_index) sequence."""
    lines = [f"{survey_year}:{idx}" for idx in index]
    content = "\n".join(lines)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def verify_cohort_alignment_and_uniqueness(
    X: pd.DataFrame, y: pd.Series, o: pd.Series
) -> None:
    """Verify that X, y, and o have identical aligned indices and that indices are unique."""
    if not (X.index.equals(y.index) and X.index.equals(o.index)):
        raise ValueError("Cohort index alignment mismatch across X, y, and o")
    if not X.index.is_unique:
        raise ValueError("Cohort indices are not unique within partition")


def compute_canonical_json_sha256(data: Any) -> str:
    """Compute deterministic SHA-256 hash of a JSON-serializable structure.

    Canonical encoding: sort_keys=True, separators=(",", ":"), allow_nan=False.
    """
    canonical_bytes = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def extract_minmax_scaler_state(
    scaler: MinMaxScaler, feature_order: Sequence[str]
) -> Dict[str, Any]:
    """Extract deterministic JSON-serializable state dictionary from a fitted MinMaxScaler.

    Schema:
    - feature_order: List[str]
    - feature_range: List[float]
    - n_features_in_: int
    - data_min_: List[float]
    - data_max_: List[float]
    - data_range_: List[float]
    - scale_: List[float]
    - min_: List[float]
    """
    if not hasattr(scaler, "data_min_") or scaler.data_min_ is None:
        raise ValueError("MinMaxScaler is not fitted.")
    return {
        "feature_order": [str(f) for f in feature_order],
        "feature_range": [float(r) for r in scaler.feature_range],
        "n_features_in_": int(scaler.n_features_in_),
        "data_min_": [float(x) for x in scaler.data_min_],
        "data_max_": [float(x) for x in scaler.data_max_],
        "data_range_": [float(x) for x in scaler.data_range_],
        "scale_": [float(x) for x in scaler.scale_],
        "min_": [float(x) for x in scaler.min_],
    }


def extract_logistic_regression_state(
    model: LogisticRegression, feature_order: Sequence[str]
) -> Dict[str, Any]:
    """Extract deterministic JSON-serializable state dictionary from a fitted LogisticRegression.

    Schema:
    - feature_order: List[str]
    - classes_: List[int]
    - coef_: List[List[float]]
    - intercept_: List[float]
    - n_features_in_: int
    - n_iter_: List[int]
    - random_state: Optional[int]
    - solver: str
    - max_iter: int
    - penalty: Optional[str]
    - C: float
    - fit_intercept: bool
    """
    if not hasattr(model, "coef_") or model.coef_ is None:
        raise ValueError("LogisticRegression is not fitted.")
    return {
        "feature_order": [str(f) for f in feature_order],
        "classes_": [int(c) for c in model.classes_],
        "coef_": [[float(w) for w in row] for row in model.coef_],
        "intercept_": [float(b) for b in model.intercept_],
        "n_features_in_": int(model.n_features_in_),
        "n_iter_": [int(it) for it in model.n_iter_],
        "random_state": int(model.random_state) if model.random_state is not None else None,
        "solver": str(model.solver),
        "max_iter": int(model.max_iter),
        "penalty": str(model.penalty) if model.penalty is not None else None,
        "C": float(model.C),
        "fit_intercept": bool(model.fit_intercept),
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


def verify_scientific_code_boundary(
    repo_root: Optional[Union[str, pathlib.Path]] = None,
    base_commit: str = SCIENTIFIC_EXECUTION_BASE_COMMIT,
    target_commit: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify frozen scientific execution code boundary at release runtime.

    Enforces:
    1. Positive git repository and commit resolution.
    2. base_commit is an ancestor of target commit / current HEAD.
    3. Zero diff between base_commit and target commit for frozen scientific paths:
       - src/fairbias/**
       - src/nhis_fairbias/adapter.py
       - src/nhis_fairbias/preprocessing.py
       - src/nhis_fairbias/evaluation.py
       - src/nhis_fairbias/features.py
       - src/nhis_fairbias/schema.py
    """
    root = pathlib.Path(repo_root).resolve() if repo_root is not None else _REPO_ROOT

    # 1. Resolve target commit
    if target_commit is None:
        try:
            head_res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(root),
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to execute git rev-parse HEAD in {root}: {exc}"
            ) from exc

        if head_res.returncode != 0:
            raise RuntimeError(
                f"Git repository unavailable or git rev-parse HEAD failed "
                f"(exit {head_res.returncode}): {head_res.stderr.strip()}"
            )
        resolved_target = head_res.stdout.strip()
        if not resolved_target or resolved_target == "UNKNOWN":
            raise RuntimeError("Unable to resolve valid current git HEAD commit hash.")
    else:
        resolved_target = str(target_commit).strip()
        if not resolved_target or resolved_target == "UNKNOWN":
            raise RuntimeError(f"Invalid target commit: {target_commit!r}")

    # 2. Check base_commit is ancestor of resolved_target
    try:
        ancestor_res = subprocess.run(
            ["git", "merge-base", "--is-ancestor", base_commit, resolved_target],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to execute git merge-base for base {base_commit}: {exc}"
        ) from exc

    if ancestor_res.returncode == 1:
        raise RuntimeError(
            f"Provenance failure: scientific execution base commit {base_commit} is not an ancestor of "
            f"target commit {resolved_target}."
        )
    elif ancestor_res.returncode != 0:
        raise RuntimeError(
            f"git merge-base failed (exit {ancestor_res.returncode}): {ancestor_res.stderr.strip()}"
        )

    # 3. Check diff in frozen scientific execution paths
    try:
        diff_res = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                f"{base_commit}..{resolved_target}",
                "--",
                *FROZEN_SCIENTIFIC_PATHS_DIFF,
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to execute git diff for frozen scientific paths: {exc}"
        ) from exc

    if diff_res.returncode == 1:
        raise RuntimeError(
            f"Provenance failure: canonical D6 temporal analysis cannot run because reviewed scientific "
            f"code has changed between base {base_commit} and target {resolved_target}. "
            f"Frozen scientific execution paths ({', '.join(FROZEN_SCIENTIFIC_PATHS_SPEC)}) "
            f"must have zero diff."
        )
    elif diff_res.returncode != 0:
        raise RuntimeError(
            f"git diff failed (exit {diff_res.returncode}): {diff_res.stderr.strip()}"
        )

    return {
        "scientific_base_commit": base_commit,
        "current_git_commit": resolved_target,
        "scientific_base_is_ancestor": True,
        "scientific_code_diff_clean": True,
        "scientific_paths_checked": list(FROZEN_SCIENTIFIC_PATHS_SPEC),
    }


def verify_prior_release_tags(
    repo_root: Optional[Union[str, pathlib.Path]] = None,
) -> Dict[str, Any]:
    """Verify presence and exact commit resolution of prior D4 and D5 tags."""
    root = pathlib.Path(repo_root).resolve() if repo_root is not None else _REPO_ROOT

    # 1. D4 primary TEST tag
    tag_obj_d4 = subprocess.run(
        ["git", "rev-parse", FROZEN_D4_TAG],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if tag_obj_d4.returncode != 0 or tag_obj_d4.stdout.strip() != FROZEN_D4_TAG_OBJECT:
        raise RuntimeError(
            f"D4 tag {FROZEN_D4_TAG} object mismatch: expected {FROZEN_D4_TAG_OBJECT}, "
            f"got {tag_obj_d4.stdout.strip()}"
        )
    commit_d4 = subprocess.run(
        ["git", "rev-parse", f"{FROZEN_D4_TAG}^{{commit}}"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if commit_d4.returncode != 0 or commit_d4.stdout.strip() != FROZEN_D4_ARCHIVED_COMMIT:
        raise RuntimeError(
            f"D4 tag {FROZEN_D4_TAG} commit mismatch: expected {FROZEN_D4_ARCHIVED_COMMIT}, "
            f"got {commit_d4.stdout.strip()}"
        )

    # 2. D5 weighted TRAIN/VAL tag
    tag_obj_d5_tv = subprocess.run(
        ["git", "rev-parse", FROZEN_D5_TRAIN_VAL_TAG],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if tag_obj_d5_tv.returncode != 0 or tag_obj_d5_tv.stdout.strip() != FROZEN_D5_TRAIN_VAL_TAG_OBJECT:
        raise RuntimeError(
            f"D5 TRAIN/VAL tag {FROZEN_D5_TRAIN_VAL_TAG} object mismatch: "
            f"expected {FROZEN_D5_TRAIN_VAL_TAG_OBJECT}, got {tag_obj_d5_tv.stdout.strip()}"
        )
    commit_d5_tv = subprocess.run(
        ["git", "rev-parse", f"{FROZEN_D5_TRAIN_VAL_TAG}^{{commit}}"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if commit_d5_tv.returncode != 0 or commit_d5_tv.stdout.strip() != FROZEN_D5_TRAIN_VAL_ARCHIVED_COMMIT:
        raise RuntimeError(
            f"D5 TRAIN/VAL tag {FROZEN_D5_TRAIN_VAL_TAG} commit mismatch: "
            f"expected {FROZEN_D5_TRAIN_VAL_ARCHIVED_COMMIT}, got {commit_d5_tv.stdout.strip()}"
        )

    # 3. D5 secondary TEST tag
    tag_obj_d5_st = subprocess.run(
        ["git", "rev-parse", FROZEN_D5_SECONDARY_TEST_TAG],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if tag_obj_d5_st.returncode != 0 or tag_obj_d5_st.stdout.strip() != FROZEN_D5_SECONDARY_TEST_TAG_OBJECT:
        raise RuntimeError(
            f"D5 secondary TEST tag {FROZEN_D5_SECONDARY_TEST_TAG} object mismatch: "
            f"expected {FROZEN_D5_SECONDARY_TEST_TAG_OBJECT}, got {tag_obj_d5_st.stdout.strip()}"
        )
    commit_d5_st = subprocess.run(
        ["git", "rev-parse", f"{FROZEN_D5_SECONDARY_TEST_TAG}^{{commit}}"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if commit_d5_st.returncode != 0 or commit_d5_st.stdout.strip() != FROZEN_D5_SECONDARY_TEST_ARCHIVED_COMMIT:
        raise RuntimeError(
            f"D5 secondary TEST tag {FROZEN_D5_SECONDARY_TEST_TAG} commit mismatch: "
            f"expected {FROZEN_D5_SECONDARY_TEST_ARCHIVED_COMMIT}, got {commit_d5_st.stdout.strip()}"
        )

    return {
        "d4_primary_test_tag_verified": True,
        "d4_tag_commit": commit_d4.stdout.strip(),
        "d5_weighted_train_val_tag_verified": True,
        "d5_train_val_tag_commit": commit_d5_tv.stdout.strip(),
        "d5_weighted_secondary_test_tag_verified": True,
        "d5_secondary_test_tag_commit": commit_d5_st.stdout.strip(),
    }


def verify_frozen_features_parquet(
    features_parquet_path: Optional[Union[str, pathlib.Path]] = None,
) -> Dict[str, Any]:
    """Verify existence and immutable SHA-256 hash of prepared features parquet."""
    target_path = (
        pathlib.Path(features_parquet_path).resolve()
        if features_parquet_path is not None
        else FROZEN_FEATURES_PARQUET_PATH.resolve()
    )
    if not target_path.is_file():
        raise FileNotFoundError(f"Prepared features parquet not found: {target_path}")

    observed_sha = compute_sha256(target_path)
    if observed_sha != FROZEN_FEATURES_PARQUET_SHA256:
        raise ValueError(
            f"Frozen features parquet SHA-256 mismatch!\n"
            f"Expected: {FROZEN_FEATURES_PARQUET_SHA256}\n"
            f"Observed: {observed_sha}\n"
            f"Path: {target_path}"
        )

    return {
        "features_parquet_path": str(target_path),
        "features_parquet_sha256": observed_sha,
        "hash_verified": True,
    }


def verify_prior_release_archives(
    repo_root: Optional[Union[str, pathlib.Path]] = None,
) -> Dict[str, Any]:
    """Verify presence of prior historical release directories."""
    root = pathlib.Path(repo_root).resolve() if repo_root is not None else _REPO_ROOT
    checked: Dict[str, bool] = {}
    for arch_rel in PRIOR_RELEASE_ARCHIVE_DIRS:
        arch_path = root / arch_rel
        if not arch_path.is_dir():
            raise FileNotFoundError(
                f"Required prior release archive directory not found: {arch_path}."
            )
        checked[arch_rel] = True
    return {
        "prior_release_archives_verified": True,
        "archive_directories": list(checked.keys()),
    }


def verify_execution_preconditions(
    repo_root: Optional[Union[str, pathlib.Path]] = None,
    features_parquet_path: Optional[Union[str, pathlib.Path]] = None,
) -> Dict[str, Any]:
    """Verify all mandatory preconditions before substantive canonical release execution.

    Enforces:
    1. Frozen features parquet existence and exact SHA-256 hash.
    2. Frozen scientific execution boundary (base commit is ancestor + 0 diff across 6 paths).
    3. Prior immutable release tag anchors (D4, D5 train/val, D5 secondary test).
    4. Prior immutable release archive directories presence.
    5. Static D6 protocol invariants (4 arms, outcome, predictor counts, group/pair counts).

    Must fail closed BEFORE release directory creation or cohort loading.
    """
    root = pathlib.Path(repo_root).resolve() if repo_root is not None else _REPO_ROOT
    pq_path = (
        pathlib.Path(features_parquet_path).resolve()
        if features_parquet_path is not None
        else FROZEN_FEATURES_PARQUET_PATH.resolve()
    )

    # 1. Parquet hash verification
    pq_info = verify_frozen_features_parquet(pq_path)

    # 2. Scientific boundary verification
    boundary_info = verify_scientific_code_boundary(repo_root=root)

    # 3. Prior release tags verification
    tag_info = verify_prior_release_tags(repo_root=root)

    # 4. Prior release archive directories verification
    archive_info = verify_prior_release_archives(repo_root=root)

    # 5. Static D6 protocol invariants
    if set(FROZEN_D6_ARMS.keys()) != {"D6_ARM_001", "D6_ARM_002", "D6_ARM_003", "D6_ARM_004"}:
        raise ValueError(f"Arm registry mismatch: {list(FROZEN_D6_ARMS.keys())}")

    for arm_id, arm in FROZEN_D6_ARMS.items():
        if arm["outcome"] != "MEDDL12M_A":
            raise ValueError(f"Invalid outcome for {arm_id}: {arm['outcome']}")
        if arm["feature_set"] != "PRIMARY_CORE":
            raise ValueError(f"Invalid feature_set for {arm_id}: {arm['feature_set']}")
        if arm_id == "D6_ARM_004":
            if arm["disability_arm"] != "exclude_disability_components" or arm["expected_predictors"] != 15:
                raise ValueError(f"Invalid specification for {arm_id}")
        else:
            if arm["disability_arm"] != "full_feature" or arm["expected_predictors"] != 21:
                raise ValueError(f"Invalid specification for {arm_id}")
        if arm_id == "D6_ARM_002":
            if arm["expected_group_count"] != 7 or arm["expected_pair_count"] != 21:
                raise ValueError(f"HISPALLP_A coverage configuration mismatch in {arm_id}")

    return {
        "frozen_features_verified": bool(pq_info["hash_verified"]),
        "features_parquet_path": str(pq_info["features_parquet_path"]),
        "features_parquet_sha256": str(pq_info["features_parquet_sha256"]),
        "scientific_base_commit": str(boundary_info["scientific_base_commit"]),
        "current_git_commit": str(boundary_info["current_git_commit"]),
        "scientific_base_is_ancestor": bool(boundary_info["scientific_base_is_ancestor"]),
        "scientific_code_diff_clean": bool(boundary_info["scientific_code_diff_clean"]),
        "scientific_paths_checked": list(boundary_info["scientific_paths_checked"]),
        "prior_release_tags_verified": bool(
            tag_info["d4_primary_test_tag_verified"]
            and tag_info["d5_weighted_train_val_tag_verified"]
            and tag_info["d5_weighted_secondary_test_tag_verified"]
        ),
        "d4_tag_commit": str(tag_info["d4_tag_commit"]),
        "d5_train_val_tag_commit": str(tag_info["d5_train_val_tag_commit"]),
        "d5_secondary_test_tag_commit": str(tag_info["d5_secondary_test_tag_commit"]),
        "prior_release_archives_verified": bool(archive_info["prior_release_archives_verified"]),
        "prior_release_archive_dirs": list(archive_info["archive_directories"]),
        "predeclared_arms_verified": True,
        "predeclared_arms": list(FROZEN_D6_ARMS.keys()),
        "predeclared_arm_count": len(FROZEN_D6_ARMS),
        "survey_weighting": "NONE",
        "frozen_outcome": "MEDDL12M_A",
        "execution_mode": "AUDIT ONLY",
        "temporal_train_year": TEMPORAL_TRAIN_YEAR,
        "temporal_validation_year": TEMPORAL_VALIDATION_YEAR,
        "future_temporal_test_year": TEMPORAL_TEST_YEAR,
        "temporal_robustness_analysis": True,
        "repeated_cross_sectional": True,
        "longitudinal": False,
        "causal_analysis": False,
    }


class NHISD6TemporalRunner:
    """Dedicated orchestrator for Gate D6 temporal robustness TRAIN/VALIDATION analysis."""

    def __init__(
        self,
        adapter: Optional[NHISStudyAdapter] = None,
        features_parquet_path: Optional[Union[str, pathlib.Path]] = None,
        study_config_path: Optional[Union[str, pathlib.Path]] = None,
        feature_config_path: Optional[Union[str, pathlib.Path]] = None,
        enforce_frozen_inputs: bool = True,
        allow_execution: bool = False,
        repo_root: Optional[Union[str, pathlib.Path]] = None,
    ):
        self.enforce_frozen_inputs = bool(enforce_frozen_inputs)
        self.allow_execution = bool(allow_execution)
        self.repo_root = pathlib.Path(repo_root).resolve() if repo_root is not None else _REPO_ROOT

        self.features_parquet_path = (
            pathlib.Path(features_parquet_path).resolve()
            if features_parquet_path is not None
            else FROZEN_FEATURES_PARQUET_PATH.resolve()
        )
        self.study_config_path = study_config_path
        self.feature_config_path = feature_config_path

        # Lazy adapter construction to keep audit-only lightweight and free of cohort loads
        self._adapter = adapter

    @property
    def adapter(self) -> NHISStudyAdapter:
        """Lazily initialize the NHISStudyAdapter when first needed for execution."""
        if self._adapter is None:
            if self.enforce_frozen_inputs:
                verify_frozen_features_parquet(self.features_parquet_path)
            self._adapter = NHISStudyAdapter(
                features_parquet_path=self.features_parquet_path,
                study_config_path=self.study_config_path,
                feature_config_path=self.feature_config_path,
            )
        return self._adapter

    def run_audit_only(self) -> Dict[str, Any]:
        """Perform read-only audit of inputs, contracts, and schema without cohort loading or execution."""
        prec = verify_execution_preconditions(
            repo_root=self.repo_root,
            features_parquet_path=self.features_parquet_path,
        )

        return {
            "status": "PASS",
            "temporal_train_year": TEMPORAL_TRAIN_YEAR,
            "temporal_validation_year": TEMPORAL_VALIDATION_YEAR,
            "future_temporal_test_year": TEMPORAL_TEST_YEAR,
            "temporal_robustness_analysis": True,
            "repeated_cross_sectional": True,
            "longitudinal": False,
            "survey_weighted_geometry": False,
            "classifier_weighting": False,
            "evaluation_weighting": False,
            "frozen_features_verified": prec["frozen_features_verified"],
            "d4_tag_verified": True,
            "d5_train_val_tag_verified": True,
            "d5_secondary_test_tag_verified": True,
            "prior_release_archives_verified": prec["prior_release_archives_verified"],
            "real_temporal_training_executed": False,
            "validation_2023_cohort_requested": False,
            "validation_2023_evaluated": False,
            "test_2024_cohort_requested": False,
            "test_2024_evaluated": False,
            "predeclared_arms": prec["predeclared_arms"],
            "predeclared_arm_count": prec["predeclared_arm_count"],
            "survey_weighting": prec["survey_weighting"],
            "frozen_outcome": prec["frozen_outcome"],
            "execution_mode": prec["execution_mode"],
            "boundary_info": {
                "scientific_base_commit": prec["scientific_base_commit"],
                "current_git_commit": prec["current_git_commit"],
                "scientific_base_is_ancestor": prec["scientific_base_is_ancestor"],
                "scientific_code_diff_clean": prec["scientific_code_diff_clean"],
                "scientific_paths_checked": prec["scientific_paths_checked"],
            },
            "features_parquet_info": {
                "features_parquet_path": prec["features_parquet_path"],
                "features_parquet_sha256": prec["features_parquet_sha256"],
                "hash_verified": prec["frozen_features_verified"],
            },
            "tag_info": {
                "d4_tag_commit": prec["d4_tag_commit"],
                "d4_primary_test_tag_verified": True,
                "d5_train_val_tag_commit": prec["d5_train_val_tag_commit"],
                "d5_weighted_train_val_tag_verified": True,
                "d5_secondary_test_tag_commit": prec["d5_secondary_test_tag_commit"],
                "d5_weighted_secondary_test_tag_verified": True,
            },
            "preconditions": prec,
        }

    def execute_arm_train_validation(
        self,
        arm_id: str,
        event_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Execute strict 2022-train and 2023-validation lifecycle for a single predeclared arm."""
        if not self.allow_execution:
            raise RuntimeError(
                "Substantive temporal execution is disabled. In Gate D6.0a, only synthetic testing is authorized."
            )

        if arm_id not in FROZEN_D6_ARMS:
            raise ValueError(f"Unknown arm_id: {arm_id}. Predeclared: {list(FROZEN_D6_ARMS.keys())}")
        arm_config = copy.deepcopy(FROZEN_D6_ARMS[arm_id])

        def record_event(name: str) -> None:
            if event_callback is not None:
                event_callback(name)

        # ---------------------------------------------------------
        # Step A: Request 2022 Development Train Cohort
        # ---------------------------------------------------------
        record_event("request_2022")
        adapter = self.adapter
        X_train, y_train, o_train, w_train, meta_train = adapter.get_cohort(
            year=TEMPORAL_TRAIN_YEAR,
            outcome=arm_config["outcome"],
            protected_attribute=arm_config["protected_attribute"],
            feature_set=arm_config["feature_set"].lower(),
            disability_arm=arm_config["disability_arm"],
        )

        # Verify cohort index alignment and uniqueness
        verify_cohort_alignment_and_uniqueness(X_train, y_train, o_train)
        train_source_row_digest = compute_cohort_source_row_digest(
            TEMPORAL_TRAIN_YEAR, X_train.index
        )

        # Verify preprocessor provenance: fit_year must be 2022, fit_role must be development_train
        record_event("verify_2022_preprocessor")
        fit_record = adapter.preprocessor.fitted_record
        if fit_record.fit_year != TEMPORAL_TRAIN_YEAR:
            raise NHISLeakageError(
                f"Preprocessor fit year must be {TEMPORAL_TRAIN_YEAR}, got {fit_record.fit_year}"
            )
        if fit_record.fit_study_role != TEMPORAL_STUDY_ROLES[TEMPORAL_TRAIN_YEAR]:
            raise NHISLeakageError(
                f"Preprocessor fit role must be {TEMPORAL_STUDY_ROLES[TEMPORAL_TRAIN_YEAR]}, "
                f"got {fit_record.fit_study_role}"
            )

        # Verify expected predictor count
        expected_preds = arm_config["expected_predictors"]
        if len(X_train.columns) != expected_preds:
            raise ValueError(
                f"Predictor count mismatch for {arm_id}: expected {expected_preds}, "
                f"got {len(X_train.columns)} ({list(X_train.columns)})"
            )

        # Verify forbidden leakage predictors are strictly absent
        for bad_col in LEAKAGE_FORBIDDEN_PREDICTORS:
            if bad_col in X_train.columns or bad_col.lower() in X_train.columns:
                raise NHISLeakageError(f"Leakage variable {bad_col} found in feature matrix X")

        # ---------------------------------------------------------
        # Step B: Determine Predictor Families
        # ---------------------------------------------------------
        all_cats, all_nums = adapter.preprocessor.get_feature_family_lists(
            arm_config["feature_set"].lower()
        )
        active_cols = set(X_train.columns)
        cate_attrs = [c for c in all_cats if c in active_cols]
        num_attrs = [c for c in all_nums if c in active_cols]
        if len(cate_attrs) + len(num_attrs) != expected_preds:
            raise ValueError(
                f"Partitioned family feature count ({len(cate_attrs)} cat + {len(num_attrs)} num) "
                f"does not match expected total {expected_preds}"
            )

        # ---------------------------------------------------------
        # Step C: Initial d_phi on 2022 (Unweighted)
        # ---------------------------------------------------------
        fb_config = FairBiasConfig(
            algorithm_mode=ALGORITHM_MODE_PAPER_FAITHFUL,
            random_seed=PRIMARY_D6_RANDOM_SEED,
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

        record_event("compute_2022_initial_dphi")
        O_train_df = pd.DataFrame({arm_config["protected_attribute"]: o_train})
        init_epsilon_dict = evaluator.calculate_epsilon(
            X_train, O_train_df, cate_attrs=cate_attrs, num_attrs=num_attrs, sample_weight=None
        )

        # ---------------------------------------------------------
        # Step D: Determine 2022 Epsilon Threshold
        # ---------------------------------------------------------
        record_event("compute_2022_epsilon")
        epsilon_threshold = float(evaluator.compute_threshold(init_epsilon_dict))

        initial_dphi_raw = copy.deepcopy(init_epsilon_dict.get(arm_config["protected_attribute"], {}))
        initial_dphi = {k: float(v) for k, v in initial_dphi_raw.items()}
        initial_max_dphi = float(max(initial_dphi.values())) if initial_dphi else 0.0
        highest_initial_feature = (
            max(initial_dphi, key=initial_dphi.get) if initial_dphi else None
        )

        # ---------------------------------------------------------
        # Step E: Calculate 2022 NMI
        # ---------------------------------------------------------
        record_event("compute_2022_nmi")
        nmi_org = calculate_nmi_dict(X_train, y_train)

        # ---------------------------------------------------------
        # Step F: Run FairBias Mitigation on 2022
        # ---------------------------------------------------------
        record_event("learn_2022_changed_dict")
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
        )

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
                    sample_weight=None,
                )
                all_curr_eps = [val for gd in current_epsilon.values() for val in gd.values()]
                curr_max_eps = float(max(all_curr_eps)) if all_curr_eps else 0.0

                if curr_max_eps <= epsilon_threshold:
                    exit_reason = "epsilon_reached"
                    break

        if exit_reason == "no_transform_accepted":
            final_eps_vals = [val for gd in current_epsilon.values() for val in gd.values()]
            term_max = float(max(final_eps_vals)) if final_eps_vals else initial_max_dphi
            converged = (term_max <= epsilon_threshold)
            termination_reason = "epsilon_reached" if converged else "candidate_grid_exhausted"
        elif exit_reason == "epsilon_reached":
            termination_reason = "epsilon_reached"
            converged = True
        else:
            termination_reason = exit_reason or "unknown"
            converged = (termination_reason == "epsilon_reached")

        # ---------------------------------------------------------
        # Step G: Freeze Terminal changed_dict
        # ---------------------------------------------------------
        record_event("freeze_2022_changed_dict")
        frozen_changed_dict = copy.deepcopy(changed_dict)

        # ---------------------------------------------------------
        # Step H: Final 2022 d_phi on Transformed TRAIN
        # ---------------------------------------------------------
        record_event("compute_2022_final_dphi")
        transformed_X_train = transformer.transform_data(
            X_train, frozen_changed_dict, num_attrs, cate_attrs
        )
        final_epsilon_dict = evaluator.calculate_epsilon(
            transformed_X_train,
            O_train_df,
            cate_attrs=cate_attrs,
            num_attrs=num_attrs,
            sample_weight=None,
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

        # ---------------------------------------------------------
        # Step I: Fit Baseline Scaler on Original 2022 X
        # ---------------------------------------------------------
        record_event("fit_baseline_scaler_2022")
        scaler_baseline = MinMaxScaler(feature_range=(0, 1))
        X_train_base_scaled = scaler_baseline.fit_transform(X_train)

        # ---------------------------------------------------------
        # Step J: Fit Baseline LR on Original 2022 X (Unweighted)
        # ---------------------------------------------------------
        record_event("fit_baseline_lr_2022")
        model_baseline = get_classifier("LR", random_state=PRIMARY_D6_RANDOM_SEED)
        model_baseline.fit(X_train_base_scaled, y_train.to_numpy())

        # ---------------------------------------------------------
        # Step K: Apply Frozen changed_dict to 2022 (already computed transformed_X_train)
        # Step L: Fit FairBias Scaler on Transformed 2022 X
        # ---------------------------------------------------------
        record_event("fit_fairbias_scaler_2022")
        scaler_fb = MinMaxScaler(feature_range=(0, 1))
        X_train_fb_scaled = scaler_fb.fit_transform(transformed_X_train)

        # ---------------------------------------------------------
        # Step M: Fit FairBias LR on Transformed 2022 X (Unweighted)
        # ---------------------------------------------------------
        record_event("fit_fairbias_lr_2022")
        model_fairbias = get_classifier("LR", random_state=PRIMARY_D6_RANDOM_SEED)
        model_fairbias.fit(X_train_fb_scaled, y_train.to_numpy())

        # ---------------------------------------------------------
        # Step M.1: Freeze 2022 Deterministic Model & Scaler State Fingerprints
        # ---------------------------------------------------------
        record_event("freeze_2022_model_state")
        baseline_features = list(X_train.columns)
        fb_features = list(transformed_X_train.columns)

        baseline_scaler_state = extract_minmax_scaler_state(scaler_baseline, baseline_features)
        baseline_scaler_sha = compute_canonical_json_sha256(baseline_scaler_state)

        baseline_lr_state = extract_logistic_regression_state(model_baseline, baseline_features)
        baseline_lr_sha = compute_canonical_json_sha256(baseline_lr_state)

        fairbias_scaler_state = extract_minmax_scaler_state(scaler_fb, fb_features)
        fairbias_scaler_sha = compute_canonical_json_sha256(fairbias_scaler_state)

        fairbias_lr_state = extract_logistic_regression_state(model_fairbias, fb_features)
        fairbias_lr_sha = compute_canonical_json_sha256(fairbias_lr_state)

        changed_dict_sha = compute_canonical_json_sha256(frozen_changed_dict)

        frozen_2022_training_state = {
            "documentation": (
                "These frozen training-state fingerprints allow the future D6.1 temporal TEST harness "
                "to refit deterministically from the same frozen 2022 cohort and verify exact "
                "scaler/model/representation state reproduction before requesting the 2024 cohort."
            ),
            "baseline_scaler": {
                "sha256": baseline_scaler_sha,
                "state": baseline_scaler_state,
            },
            "baseline_logistic_regression": {
                "sha256": baseline_lr_sha,
                "state": baseline_lr_state,
            },
            "fairbias_scaler": {
                "sha256": fairbias_scaler_sha,
                "state": fairbias_scaler_state,
            },
            "fairbias_logistic_regression": {
                "sha256": fairbias_lr_sha,
                "state": fairbias_lr_state,
            },
            "changed_dict_sha256": changed_dict_sha,
        }

        # =========================================================
        # ONLY AFTER STEPS A-M HAVE COMPLETED:
        # Step N: Request 2023 Development Validation Cohort
        # =========================================================
        record_event("request_2023")
        X_val, y_val, o_val, w_val, meta_val = adapter.get_cohort(
            year=TEMPORAL_VALIDATION_YEAR,
            outcome=arm_config["outcome"],
            protected_attribute=arm_config["protected_attribute"],
            feature_set=arm_config["feature_set"].lower(),
            disability_arm=arm_config["disability_arm"],
        )

        # Verify validation cohort index alignment and uniqueness
        verify_cohort_alignment_and_uniqueness(X_val, y_val, o_val)
        val_source_row_digest = compute_cohort_source_row_digest(
            TEMPORAL_VALIDATION_YEAR, X_val.index
        )

        # ---------------------------------------------------------
        # Step O: Apply Frozen Preprocessing, changed_dict, Scalers to 2023
        # ---------------------------------------------------------
        record_event("transform_2023")
        X_val_base_scaled = scaler_baseline.transform(X_val)

        transformed_X_val = transformer.transform_data(
            X_val, frozen_changed_dict, num_attrs, cate_attrs
        )
        X_val_fb_scaled = scaler_fb.transform(transformed_X_val)

        # ---------------------------------------------------------
        # Step P: Score Baseline and FairBias Models on 2023
        # Step Q: Calculate 2023 Validation Metrics
        # ---------------------------------------------------------
        record_event("score_and_evaluate_2023")
        val_prob_base = model_baseline.predict_proba(X_val_base_scaled)[:, 1]
        val_pred_base = (val_prob_base >= DEFAULT_PREDICTION_THRESHOLD).astype(int)
        val_eval_base = evaluate_predictions(
            y_true=y_val.to_numpy(),
            y_pred=val_pred_base,
            y_prob=val_prob_base,
            o_group=o_val.to_numpy(),
            expected_group_count=arm_config["expected_group_count"],
            expected_groups=arm_config["expected_groups"],
        )

        val_prob_fb = model_fairbias.predict_proba(X_val_fb_scaled)[:, 1]
        val_pred_fb = (val_prob_fb >= DEFAULT_PREDICTION_THRESHOLD).astype(int)
        val_eval_fb = evaluate_predictions(
            y_true=y_val.to_numpy(),
            y_pred=val_pred_fb,
            y_prob=val_prob_fb,
            o_group=o_val.to_numpy(),
            expected_group_count=arm_config["expected_group_count"],
            expected_groups=arm_config["expected_groups"],
        )

        # Augment utility metrics with prediction and outcome volume context
        val_eval_base["utility"]["count_predicted_positive"] = int(np.sum(val_pred_base == 1))
        val_eval_base["utility"]["selection_rate"] = float(np.mean(val_pred_base == 1))
        val_eval_base["utility"]["count_outcome_positive"] = int(np.sum(y_val == 1))
        val_eval_base["utility"]["prevalence"] = float(np.mean(y_val == 1))

        val_eval_fb["utility"]["count_predicted_positive"] = int(np.sum(val_pred_fb == 1))
        val_eval_fb["utility"]["selection_rate"] = float(np.mean(val_pred_fb == 1))
        val_eval_fb["utility"]["count_outcome_positive"] = int(np.sum(y_val == 1))
        val_eval_fb["utility"]["prevalence"] = float(np.mean(y_val == 1))

        # Paired before-vs-after comparison (numbers only, no winner/selection labels)
        val_comparison = compute_evaluation_comparison(val_eval_base, val_eval_fb)
        val_comparison["predicted_positive_delta"] = int(
            val_eval_fb["utility"]["count_predicted_positive"]
            - val_eval_base["utility"]["count_predicted_positive"]
        )
        val_comparison["selection_rate_delta"] = float(
            val_eval_fb["utility"]["selection_rate"]
            - val_eval_base["utility"]["selection_rate"]
        )

        # ---------------------------------------------------------
        # Step R: Calculate 2023 Out-of-Time d_phi Diagnostic
        # ---------------------------------------------------------
        record_event("diagnostic_dphi_2023")
        O_val_df = pd.DataFrame({arm_config["protected_attribute"]: o_val})
        val_orig_dphi_raw = evaluator.calculate_epsilon(
            X_val, O_val_df, cate_attrs=cate_attrs, num_attrs=num_attrs, sample_weight=None
        ).get(arm_config["protected_attribute"], {})
        val_orig_dphi = {k: float(v) for k, v in val_orig_dphi_raw.items()}
        val_orig_max_dphi = float(max(val_orig_dphi.values())) if val_orig_dphi else 0.0

        val_trans_dphi_raw = evaluator.calculate_epsilon(
            transformed_X_val, O_val_df, cate_attrs=cate_attrs, num_attrs=num_attrs, sample_weight=None
        ).get(arm_config["protected_attribute"], {})
        val_trans_dphi = {k: float(v) for k, v in val_trans_dphi_raw.items()}
        val_trans_max_dphi = float(max(val_trans_dphi.values())) if val_trans_dphi else 0.0
        val_delta_max_dphi = float(val_trans_max_dphi - val_orig_max_dphi)
        val_max_dphi_below_train_epsilon = bool(val_trans_max_dphi <= epsilon_threshold)

        # ---------------------------------------------------------
        # Assemble Descriptive Drift Context for input_provenance
        # ---------------------------------------------------------
        train_grp_counts = {str(k): int(v) for k, v in o_train.value_counts().to_dict().items()}
        train_grp_pos = {
            str(g): int(((o_train == g) & (y_train == 1)).sum())
            for g in sorted(o_train.unique())
        }
        train_grp_prev = {
            str(g): (
                float(train_grp_pos[str(g)] / train_grp_counts[str(g)])
                if train_grp_counts.get(str(g), 0) > 0
                else 0.0
            )
            for g in sorted(o_train.unique())
        }

        val_grp_counts = {str(k): int(v) for k, v in o_val.value_counts().to_dict().items()}
        val_grp_pos = {
            str(g): int(((o_val == g) & (y_val == 1)).sum())
            for g in sorted(o_val.unique())
        }
        val_grp_prev = {
            str(g): (
                float(val_grp_pos[str(g)] / val_grp_counts[str(g)])
                if val_grp_counts.get(str(g), 0) > 0
                else 0.0
            )
            for g in sorted(o_val.unique())
        }

        # ---------------------------------------------------------
        # Build All 11 Required Per-Arm Artifact Payloads
        # ---------------------------------------------------------
        arm_config_payload = {
            **arm_config,
            "algorithm_mode": fb_config.algorithm_mode,
            "survey_weighting": "NONE",
            "train_year": TEMPORAL_TRAIN_YEAR,
            "validation_year": TEMPORAL_VALIDATION_YEAR,
            "future_test_year": TEMPORAL_TEST_YEAR,
            "classifier": {
                "type": "LR",
                "random_state": PRIMARY_D6_RANDOM_SEED,
                "solver": CLASSIFIER_SOLVER,
                "max_iter": CLASSIFIER_MAX_ITER,
                "sample_weight": None,
            },
            "prediction_threshold": DEFAULT_PREDICTION_THRESHOLD,
            "categorical_features": cate_attrs,
            "numerical_features": num_attrs,
            "repeated_cross_sectional": True,
            "longitudinal": False,
            "causal_analysis": False,
            "temporal_robustness_analysis": True,
        }

        pq_file = pathlib.Path(adapter.features_parquet_path).resolve()
        input_provenance_payload = {
            "arm_id": arm_config["arm_id"],
            "features_parquet_path": str(pq_file),
            "features_parquet_sha256": compute_sha256(pq_file) if pq_file.is_file() else "MOCK_SHA",
            "preprocessor_fit_year": int(fit_record.fit_year),
            "preprocessor_fit_role": str(fit_record.fit_study_role),
            "train_year": TEMPORAL_TRAIN_YEAR,
            "train_n": len(X_train),
            "train_source_row_digest": train_source_row_digest,
            "train_outcome_positive_count": int((y_train == 1).sum()),
            "train_prevalence": float((y_train == 1).mean()),
            "train_group_counts": train_grp_counts,
            "train_group_positive_counts": train_grp_pos,
            "train_group_prevalences": train_grp_prev,
            "validation_year": TEMPORAL_VALIDATION_YEAR,
            "validation_n": len(X_val),
            "validation_source_row_digest": val_source_row_digest,
            "validation_outcome_positive_count": int((y_val == 1).sum()),
            "validation_prevalence": float((y_val == 1).mean()),
            "validation_group_counts": val_grp_counts,
            "validation_group_positive_counts": val_grp_pos,
            "validation_group_prevalences": val_grp_prev,
            "predictor_names": list(X_train.columns),
            "protected_attribute": arm_config["protected_attribute"],
            "survey_weight_used_for_geometry": False,
            "survey_weight_used_for_classifier": False,
            "survey_weight_used_for_evaluation": False,
            "validation_used_for_selection": False,
            "test_year_requested": False,
            "test_year_evaluated": False,
            "frozen_2022_training_state": frozen_2022_training_state,
        }

        train_dphi_payload = {
            "arm_id": arm_config["arm_id"],
            "train_year": TEMPORAL_TRAIN_YEAR,
            "protected_attribute": arm_config["protected_attribute"],
            "epsilon_threshold": float(epsilon_threshold),
            "initial_dphi": initial_dphi,
            "initial_max_dphi": float(initial_max_dphi),
            "highest_initial_feature": highest_initial_feature,
            "final_dphi": final_dphi,
            "final_max_dphi": float(final_max_dphi),
            "delta_max_dphi": float(final_max_dphi - initial_max_dphi),
            "termination_reason": termination_reason,
            "converged": bool(converged),
            "total_mitigation_steps": len(mitigation_engine.step_traces),
        }

        validation_dphi_payload = {
            "arm_id": arm_config["arm_id"],
            "validation_year": TEMPORAL_VALIDATION_YEAR,
            "protected_attribute": arm_config["protected_attribute"],
            "frozen_train_epsilon_threshold": float(epsilon_threshold),
            "original_validation_dphi": val_orig_dphi,
            "original_validation_max_dphi": float(val_orig_max_dphi),
            "transformed_validation_dphi": val_trans_dphi,
            "transformed_validation_max_dphi": float(val_trans_max_dphi),
            "delta_max_dphi": float(val_delta_max_dphi),
            "max_dphi_below_train_epsilon": bool(val_max_dphi_below_train_epsilon),
            "dphi_transport_relearned": False,
            "diagnostic_only": True,
        }

        group_metrics_base_df = pd.DataFrame(val_eval_base["group_metrics"])
        group_metrics_fb_df = pd.DataFrame(val_eval_fb["group_metrics"])

        return {
            "arm_id": arm_config["arm_id"],
            "arm_config": arm_config_payload,
            "input_provenance": input_provenance_payload,
            "train_dphi_before_after": train_dphi_payload,
            "train_fairbias_trace": trace.to_dict(),
            "final_changed_dict": frozen_changed_dict,
            "validation_dphi_before_after": validation_dphi_payload,
            "validation_metrics_baseline": val_eval_base,
            "validation_metrics_fairbias": val_eval_fb,
            "validation_group_metrics_baseline": group_metrics_base_df,
            "validation_group_metrics_fairbias": group_metrics_fb_df,
            "validation_comparison": val_comparison,
        }


class NHISD6TemporalReleaseManager:
    """Release manager for Gate D6 temporal robustness TRAIN/VALIDATION harness."""

    def __init__(
        self,
        release_id: str,
        base_dir: Optional[Union[str, pathlib.Path]] = None,
        repo_root: Optional[Union[str, pathlib.Path]] = None,
        runner: Optional[NHISD6TemporalRunner] = None,
        allow_substantive_execution: bool = False,
    ):
        if not release_id or not str(release_id).strip():
            raise ValueError("A non-empty release_id is required.")
        self.release_id = str(release_id).strip()
        self.repo_root = pathlib.Path(repo_root).resolve() if repo_root is not None else _REPO_ROOT
        self.allow_substantive_execution = bool(allow_substantive_execution)

        default_base = self.repo_root / "runs" / "nhis_d6_temporal" / "releases"
        self.base_dir = pathlib.Path(base_dir).resolve() if base_dir is not None else default_base
        self.release_dir = self.base_dir / self.release_id

        self.runner = runner or NHISD6TemporalRunner(
            allow_execution=self.allow_substantive_execution,
            repo_root=self.repo_root,
        )

    def execute_release(self) -> Dict[str, Any]:
        """Execute substantive 4-arm release lifecycle with fail-closed state machine."""
        if not self.allow_substantive_execution:
            raise RuntimeError(
                "Substantive release execution is disabled. Explicit authorization is required."
            )

        # Release collisions fail closed
        if self.release_dir.exists():
            raise FileExistsError(
                f"Canonical release directory already exists: {self.release_dir}. "
                "Overwrites or restarts are strictly forbidden."
            )

        # Precondition verification must occur BEFORE release directory creation
        preconditions_info = verify_execution_preconditions(
            repo_root=self.repo_root,
            features_parquet_path=self.runner.features_parquet_path,
        )

        self.release_dir.mkdir(parents=True, exist_ok=False)
        created_at_ts = utc_timestamp()

        # Atomically record STARTED state
        state_path = self.release_dir / "release_state.json"
        write_json_atomic(
            state_path,
            {
                "release_id": self.release_id,
                "status": "STARTED",
                "created_at": created_at_ts,
                "updated_at": created_at_ts,
                "manifest_sha256": None,
                "error": None,
            },
        )

        try:
            arm_results: Dict[str, Dict[str, Any]] = {}
            for arm_id in sorted(FROZEN_D6_ARMS.keys()):
                arm_res = self.runner.execute_arm_train_validation(arm_id)
                arm_results[arm_id] = arm_res

                arm_dir = self.release_dir / arm_id
                arm_dir.mkdir(parents=True, exist_ok=True)

                # Persist exactly 11 artifacts per arm
                write_json_atomic(arm_dir / "arm_config.json", arm_res["arm_config"])
                write_json_atomic(arm_dir / "input_provenance.json", arm_res["input_provenance"])
                write_json_atomic(
                    arm_dir / "train_dphi_before_after.json", arm_res["train_dphi_before_after"]
                )
                write_json_atomic(
                    arm_dir / "train_fairbias_trace.json", arm_res["train_fairbias_trace"]
                )
                write_json_atomic(
                    arm_dir / "final_changed_dict.json", arm_res["final_changed_dict"]
                )
                write_json_atomic(
                    arm_dir / "validation_dphi_before_after.json",
                    arm_res["validation_dphi_before_after"],
                )
                write_json_atomic(
                    arm_dir / "validation_metrics_baseline.json",
                    arm_res["validation_metrics_baseline"],
                )
                write_json_atomic(
                    arm_dir / "validation_metrics_fairbias.json",
                    arm_res["validation_metrics_fairbias"],
                )
                write_csv_atomic(
                    arm_res["validation_group_metrics_baseline"],
                    arm_dir / "validation_group_metrics_baseline.csv",
                )
                write_csv_atomic(
                    arm_res["validation_group_metrics_fairbias"],
                    arm_dir / "validation_group_metrics_fairbias.csv",
                )
                write_json_atomic(
                    arm_dir / "validation_comparison.json", arm_res["validation_comparison"]
                )

            # Persist top-level preprocessing provenance
            prep_record = self.runner.adapter.preprocessor.fitted_record.to_dict()
            prep_prov_path = self.release_dir / "preprocessing_provenance.json"
            write_json_atomic(prep_prov_path, prep_record)

            # Manifest artifacts calculation (all 44 per-arm artifacts + top-level preprocessing)
            artifacts_dict: Dict[str, Dict[str, Any]] = {}
            for arm_id in sorted(FROZEN_D6_ARMS.keys()):
                for art_name in REQUIRED_PER_ARM_ARTIFACTS:
                    rel_p = f"{arm_id}/{art_name}"
                    art_file = self.release_dir / arm_id / art_name
                    if not art_file.is_file():
                        raise FileNotFoundError(f"Missing required artifact: {art_file}")
                    artifacts_dict[rel_p] = {
                        "sha256": compute_sha256(art_file),
                        "size_bytes": art_file.stat().st_size,
                    }

            artifacts_dict["preprocessing_provenance.json"] = {
                "sha256": compute_sha256(prep_prov_path),
                "size_bytes": prep_prov_path.stat().st_size,
            }

            manifest_payload = {
                "manifest_version": "nhis-fairbias-d6-1.0",
                "gate": "D6 temporal TRAIN/VALIDATION",
                "release_id": self.release_id,
                "status": "COMPLETE",
                "created_at": created_at_ts,
                "git_commit": get_git_commit(self.repo_root),
                "base_commit": SCIENTIFIC_EXECUTION_BASE_COMMIT,
                "temporal_robustness_analysis": True,
                "repeated_cross_sectional": True,
                "longitudinal": False,
                "causal_analysis": False,
                "primary_analysis": False,
                "replaces_d4_primary": False,
                "train_year": TEMPORAL_TRAIN_YEAR,
                "validation_year": TEMPORAL_VALIDATION_YEAR,
                "future_test_year": TEMPORAL_TEST_YEAR,
                "validation_used_for_selection": False,
                "test_year_requested": False,
                "test_year_evaluated": False,
                "survey_weighted_geometry": False,
                "classifier_weighted": False,
                "evaluation_weighted": False,
                "disclosure_2024": TEMPORAL_2024_DISCLOSURE,
                "execution_preconditions": {
                    "frozen_features_verified": bool(preconditions_info["frozen_features_verified"]),
                    "features_parquet_sha256": str(preconditions_info["features_parquet_sha256"]),
                    "scientific_base_commit": str(preconditions_info["scientific_base_commit"]),
                    "scientific_base_is_ancestor": bool(preconditions_info["scientific_base_is_ancestor"]),
                    "scientific_code_diff_clean": bool(preconditions_info["scientific_code_diff_clean"]),
                    "prior_release_tags_verified": bool(preconditions_info["prior_release_tags_verified"]),
                    "prior_release_archives_verified": bool(preconditions_info["prior_release_archives_verified"]),
                },
                "arms": {
                    arm_id: {
                        "arm_id": arm_id,
                        "outcome": FROZEN_D6_ARMS[arm_id]["outcome"],
                        "protected_attribute": FROZEN_D6_ARMS[arm_id]["protected_attribute"],
                        "feature_set": FROZEN_D6_ARMS[arm_id]["feature_set"],
                        "disability_arm": FROZEN_D6_ARMS[arm_id]["disability_arm"],
                        "expected_predictors": FROZEN_D6_ARMS[arm_id]["expected_predictors"],
                        "expected_group_count": FROZEN_D6_ARMS[arm_id]["expected_group_count"],
                        "expected_pair_count": FROZEN_D6_ARMS[arm_id]["expected_pair_count"],
                    }
                    for arm_id in sorted(FROZEN_D6_ARMS.keys())
                },
                "artifacts": artifacts_dict,
            }

            manifest_path = self.release_dir / "d6_temporal_train_val_manifest.json"
            write_json_atomic(manifest_path, manifest_payload)
            manifest_sha = compute_sha256(manifest_path)

            # Mark state as COMPLETE
            write_json_atomic(
                state_path,
                {
                    "release_id": self.release_id,
                    "status": "COMPLETE",
                    "created_at": created_at_ts,
                    "updated_at": utc_timestamp(),
                    "manifest_sha256": manifest_sha,
                    "error": None,
                },
            )
            return manifest_payload

        except Exception as exc:
            # Mark state as FAILED and fail closed
            write_json_atomic(
                state_path,
                {
                    "release_id": self.release_id,
                    "status": "FAILED",
                    "created_at": created_at_ts,
                    "updated_at": utc_timestamp(),
                    "manifest_sha256": None,
                    "error": str(exc),
                },
            )
            raise
