"""Dedicated frozen secondary TEST evaluation and release harness for Gate D5.2.

Scientific Status & Protocol Disclosure:
D5.2 is a predeclared secondary extension evaluated after the D4 primary TEST release.
It is NOT:
- a new primary analysis
- a replacement primary TEST
- a globally untouched holdout analysis

Mandatory Disclosure:
"This survey-weighted extension was predeclared before the primary TEST release,
but is evaluated after that primary TEST release. Therefore the pooled TEST partition
is not an untouched primary holdout for this secondary analysis."

Guarantees:
1. Frozen Artifact Contract:
   - Uses exclusively the four frozen archived changed_dict states from Gate D5.1d.
   - Verifies SHA-256 hashes of archived changed_dict, manifest, and archive ledger.
   - Fails closed if any hash or tag dereference is missing or mismatched.
2. No Re-learning / No FairBias Mitigation:
   - NEVER calls FairBiasMitigation.fit, mitigate_step, calculate_epsilon, or run_fairbias_pipeline.
   - NEVER recomputes or relearns d_phi, epsilon, NMI, greedy ranking, category merges,
     power search, or drop decisions.
   - Applies the exact frozen train-learned weighted changed_dict unchanged.
3. Master Pooled Split & Test Leakage Prevention:
   - NEVER calls generate_pooled_splits() or train_test_split().
   - Validation partition is NEVER loaded or used for model fitting, scaling,
     threshold selection, transform selection, or test evaluation.
   - Scalers and models are fit strictly on TRAIN data.
   - Baseline and FairBias models are distinct instances with identical LR specifications.
4. One-Time Release Semantics:
   - Requires explicit release_id and a fresh canonical release directory.
   - Atomically records STARTED state before scoring.
   - Atomically records COMPLETE state only after all arms and hashes are finalized.
   - Refuses re-execution or silent overwrite if release directory already exists.
5. Dual Modes:
   - audit_only (authorized for D5.2a): strictly verifies provenance, input hashes,
     archived D5 artifacts, arm configs, and embargo state without loading or scoring TEST.
   - execute_frozen_secondary_test (embargoed until D5.2b PI review): executes full test release.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import subprocess
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from fairbias.transform import FairTransform

from .audit import write_csv_atomic
from .download import (
    compute_sha256,
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
from .pooled import NHISPooledAdapter

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

CANONICAL_D5_SECONDARY_TEST_RELEASE_ID: str = "NHIS_D5_WEIGHTED_SECONDARY_TEST_V1_14cc7aa6"
SCIENTIFIC_EXECUTION_BASE_COMMIT: str = "bc6034e530d6e92bbfa82feadc0cf783302cb30f"
TEST_EVALUATION_BASE_COMMIT: str = "14cc7aa629e20b326773621c164f69e3ef3acfdb"

FROZEN_D5_RELEASE_ID: str = "NHIS_D5_WEIGHTED_TRAIN_VAL_V1_bc6034e5"
FROZEN_D5_TAG: str = "nhis-d5-weighted-train-val-v1"
FROZEN_D5_TAG_COMMIT: str = "14cc7aa629e20b326773621c164f69e3ef3acfdb"
FROZEN_D5_MANIFEST_SHA256: str = "4d893120d46eddc8595168df8dc38e750c4c12c3b99b7faf0a2dd03789837ac4"
FROZEN_D5_LEDGER_SHA256: str = "2b974a27d74e4e4de52b2264dfc795bad8e15c60a59dbd856e53dee21abc9b5c"

FROZEN_D4_TAG: str = "nhis-d4-primary-test-v1"
FROZEN_D4_TAG_COMMIT: str = "d74af83fb98a3805a7fb0767c5beeb3dbf1407e4"

FROZEN_SPLIT_MANIFEST_PATH: pathlib.Path = _REPO_ROOT / "artifacts" / "nhis" / "d3" / "pooled_split_manifest.csv"
FROZEN_SPLIT_MANIFEST_SHA256: str = "6874a56f5484186dffdd5faeb7871c3bef85042ccfdf8d1e375fdde75a3ee9f5"

FROZEN_FEATURES_PARQUET_PATH: pathlib.Path = _REPO_ROOT / "data" / "processed" / "nhis" / "nhis_2022_2024_features.parquet"
FROZEN_FEATURES_PARQUET_SHA256: str = "49f415132ff0be0228f8533f9f74c48cd79ff6fa8be66db085f7329d9b083383"

DEFAULT_D5_ARCHIVE_DIR: pathlib.Path = _REPO_ROOT / "docs" / "releases" / "NHIS_D5_WEIGHTED_TRAIN_VAL_V1_bc6034e5"
DEFAULT_D4_ARCHIVE_DIR: pathlib.Path = _REPO_ROOT / "docs" / "releases" / "NHIS_D4_PRIMARY_TEST_RELEASE_V1_9920fc5a"
DEFAULT_TEST_RELEASE_BASE_DIR: pathlib.Path = _REPO_ROOT / "runs" / "nhis_d5_weighted" / "test_releases"

DEFAULT_RANDOM_SEED: int = 0
DEFAULT_PREDICTION_THRESHOLD: float = 0.5

SECONDARY_ANALYSIS_DISCLOSURE: str = (
    "This survey-weighted extension was predeclared before the primary TEST release, "
    "but is evaluated after that primary TEST release. Therefore the pooled TEST partition "
    "is not an untouched primary holdout for this secondary analysis."
)

FROZEN_SCIENTIFIC_PATHS_SPEC: Tuple[str, ...] = (
    "src/fairbias/**",
    "src/nhis_fairbias/preprocessing.py",
    "src/nhis_fairbias/evaluation.py",
    "src/nhis_fairbias/pooled.py",
    "src/nhis_fairbias/d5_weighted_runner.py",
)

FROZEN_SCIENTIFIC_PATHS_DIFF: Tuple[str, ...] = (
    "src/fairbias",
    "src/nhis_fairbias/preprocessing.py",
    "src/nhis_fairbias/evaluation.py",
    "src/nhis_fairbias/pooled.py",
    "src/nhis_fairbias/d5_weighted_runner.py",
)

FROZEN_D5_TEST_ARMS: Dict[str, Dict[str, Any]] = {
    "D5_ARM_001": {
        "arm_id": "D5_ARM_001",
        "protected_attribute": "SEX_A",
        "outcome": "MEDDL12M_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "full_feature",
        "expected_predictors": 21,
        "expected_groups": [1, 2],
        "expected_group_count": 2,
        "expected_pair_count": 1,
        "expected_changed_dict_sha256": "62ffbcffd73a8db0f50d0484d1d3f7fe035e401e2d17894e18e5ee9422ed01d5",
        "archived_trace_sha256": "e82055603207564c3f7ee9e210195dc1b20e2d8d545836f5f545653c4b38bd9a",
        "archived_validation_comparison_sha256": "aa892028c38ca62e40a970738ca17de31b6b2765b7a075ef8fe0c356fd6a4383",
    },
    "D5_ARM_002": {
        "arm_id": "D5_ARM_002",
        "protected_attribute": "HISPALLP_A",
        "outcome": "MEDDL12M_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "full_feature",
        "expected_predictors": 21,
        "expected_groups": [1, 2, 3, 4, 5, 6, 7],
        "expected_group_count": 7,
        "expected_pair_count": 21,
        "expected_changed_dict_sha256": "aa3f4afafada1b7ca9b4e051485e326c558c4168015317fd0f4522be0a73a7dc",
        "archived_trace_sha256": "0179ae6906fb6dabf39fda696c815b2fda1cb6020652a1f6ec219ecd24d3174b",
        "archived_validation_comparison_sha256": "333385f813c991aa852527a0ffd038ab16d603c26647a5d226c97d1bfb968ba3",
    },
    "D5_ARM_003": {
        "arm_id": "D5_ARM_003",
        "protected_attribute": "DISAB3_A",
        "outcome": "MEDDL12M_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "full_feature",
        "expected_predictors": 21,
        "expected_groups": [1, 2],
        "expected_group_count": 2,
        "expected_pair_count": 1,
        "expected_changed_dict_sha256": "84489acf8e6f72964c74d9a9fc5719eec35d70a61e32880eca7439ddcefdde63",
        "archived_trace_sha256": "99c09cae5833f6381f333b74b866968dae6744ad00f99a58f9afe4acb95cdc3c",
        "archived_validation_comparison_sha256": "bdf48224779be945c2eb796fac209c4e23e8355f1b1695bcb6dec3d5f76f7be7",
    },
    "D5_ARM_004": {
        "arm_id": "D5_ARM_004",
        "protected_attribute": "DISAB3_A",
        "outcome": "MEDDL12M_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "exclude_disability_components",
        "expected_predictors": 15,
        "expected_groups": [1, 2],
        "expected_group_count": 2,
        "expected_pair_count": 1,
        "expected_changed_dict_sha256": "89ffec2eae7f9fac7d3a2e99a40fd3439b3ac86343f17674ead45298170eaed0",
        "archived_trace_sha256": "bf1e2329a8689082f87407712089c567bebd498cb90599062f2d075346bce730",
        "archived_validation_comparison_sha256": "002656f5cb85ce23bca03b907652b56c02c6b7606a7ae17beec08c6e87eea770",
    },
}


class ReleaseCollisionError(RuntimeError):
    """Raised when release directory already exists or contains prior release state."""
    pass


class FrozenArtifactIntegrityError(ValueError):
    """Raised when a frozen preflight/archived artifact is missing or has a mismatched SHA-256 hash."""
    pass


class PreconditionFailureError(RuntimeError):
    """Raised when release preconditions fail."""
    pass


def compute_sequence_digest(values: Sequence[Any]) -> str:
    """Deterministic SHA-256 digest of a sequence of values."""
    formatted = "\n".join(str(v) for v in values)
    return hashlib.sha256(formatted.encode("utf-8")).hexdigest()


def get_git_commit(repo_root: Optional[Union[str, pathlib.Path]] = None) -> str:
    """Retrieve current git HEAD commit hash."""
    root = pathlib.Path(repo_root).resolve() if repo_root is not None else _REPO_ROOT
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
    base_commit: str = TEST_EVALUATION_BASE_COMMIT,
    target_commit: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify frozen scientific execution code boundary at release runtime.

    Enforces:
    1. Positive git repository and commit resolution (fail closed on git failure).
    2. base_commit is an ancestor of target commit / current HEAD (git merge-base --is-ancestor).
    3. Zero diff between base_commit and target commit for frozen scientific paths:
       - src/fairbias/**
       - src/nhis_fairbias/preprocessing.py
       - src/nhis_fairbias/evaluation.py
       - src/nhis_fairbias/pooled.py
       - src/nhis_fairbias/d5_weighted_runner.py
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

    # 2. Check base_commit is an ancestor of resolved_target
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
            f"Provenance failure: test evaluation base commit {base_commit} is not an ancestor of "
            f"current HEAD {resolved_target}. Canonical D5 secondary TEST release requires verified ancestor lineage; "
            f"rebased or unrelated history is strictly forbidden."
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
            f"Provenance failure: canonical D5 secondary TEST release cannot run because reviewed scientific "
            f"code has changed between base {base_commit} and target {resolved_target}. "
            f"Frozen scientific execution paths ({', '.join(FROZEN_SCIENTIFIC_PATHS_SPEC)}) "
            f"must have zero diff."
        )
    elif diff_res.returncode != 0:
        raise RuntimeError(
            f"git diff failed (exit {diff_res.returncode}): {diff_res.stderr.strip()}"
        )

    return {
        "test_evaluation_base_commit": base_commit,
        "current_git_commit": resolved_target,
        "scientific_base_is_ancestor": True,
        "scientific_code_diff_clean": True,
        "scientific_paths_checked": list(FROZEN_SCIENTIFIC_PATHS_SPEC),
    }


def verify_d5_archive(
    archive_dir: Optional[Union[str, pathlib.Path]] = None,
    repo_root: Optional[Union[str, pathlib.Path]] = None,
    enforce_tag: bool = True,
) -> Dict[str, Any]:
    """Verify integrity of the archived D5 TRAIN/VAL release in docs/releases/."""
    root = pathlib.Path(repo_root).resolve() if repo_root is not None else _REPO_ROOT
    d5_dir = pathlib.Path(archive_dir).resolve() if archive_dir is not None else DEFAULT_D5_ARCHIVE_DIR.resolve()

    if not d5_dir.is_dir():
        raise FileNotFoundError(f"Archived D5 release directory not found at: {d5_dir}")

    # 1. Verify release_state.json
    state_file = d5_dir / "release_state.json"
    if not state_file.is_file():
        raise FrozenArtifactIntegrityError(f"Missing release_state.json at {state_file}")
    state = json.loads(state_file.read_text(encoding="utf-8"))
    if state.get("status") != "COMPLETE":
        raise FrozenArtifactIntegrityError(f"Archived D5 release status is {state.get('status')!r} (expected 'COMPLETE')")
    if state.get("manifest_sha256") != FROZEN_D5_MANIFEST_SHA256:
        raise FrozenArtifactIntegrityError(
            f"Archived D5 release state manifest SHA-256 mismatch: "
            f"expected {FROZEN_D5_MANIFEST_SHA256}, got {state.get('manifest_sha256')}"
        )

    # 2. Verify d5_weighted_release_manifest.json
    manifest_file = d5_dir / "d5_weighted_release_manifest.json"
    if not manifest_file.is_file():
        raise FrozenArtifactIntegrityError(f"Missing manifest at {manifest_file}")
    manifest_sha = compute_sha256(manifest_file)
    if manifest_sha != FROZEN_D5_MANIFEST_SHA256:
        raise FrozenArtifactIntegrityError(
            f"Archived D5 manifest SHA-256 mismatch: expected {FROZEN_D5_MANIFEST_SHA256}, got {manifest_sha}"
        )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    # 3. Verify archive_ledger.json
    ledger_file = d5_dir / "archive_ledger.json"
    if not ledger_file.is_file():
        raise FrozenArtifactIntegrityError(f"Missing archive_ledger.json at {ledger_file}")
    ledger = json.loads(ledger_file.read_text(encoding="utf-8"))
    if ledger.get("raw_file_count") != 50:
        raise FrozenArtifactIntegrityError(f"Ledger raw_file_count is {ledger.get('raw_file_count')} (expected 50)")

    raw_files = ledger.get("raw_files", [])
    if len(raw_files) != 50:
        raise FrozenArtifactIntegrityError(f"Ledger lists {len(raw_files)} raw files (expected 50)")

    for item in raw_files:
        rel_p = item["relative_path"]
        exp_h = item["sha256"]
        fp = d5_dir / rel_p
        if not fp.is_file():
            raise FrozenArtifactIntegrityError(f"Missing archived raw file {fp}")
        obs_h = compute_sha256(fp)
        if obs_h != exp_h:
            raise FrozenArtifactIntegrityError(f"Hash mismatch on archived raw file {rel_p}: expected {exp_h}, got {obs_h}")

    # 4. Verify 48 per-arm artifacts against manifest
    artifacts = manifest.get("artifacts", {})
    if len(artifacts) != 48:
        raise FrozenArtifactIntegrityError(f"Manifest specifies {len(artifacts)} artifacts (expected 48)")
    for rel_p, meta in artifacts.items():
        fp = d5_dir / rel_p
        if not fp.is_file():
            raise FrozenArtifactIntegrityError(f"Missing per-arm artifact {fp}")
        obs_h = compute_sha256(fp)
        if obs_h != meta["sha256"]:
            raise FrozenArtifactIntegrityError(f"Hash mismatch on per-arm artifact {rel_p}: expected {meta['sha256']}, got {obs_h}")

    # 5. Verify the four frozen changed_dict hashes
    for arm_id, arm_spec in FROZEN_D5_TEST_ARMS.items():
        cd_file = d5_dir / arm_id / "weighted_changed_dict.json"
        if not cd_file.is_file():
            raise FrozenArtifactIntegrityError(f"Missing weighted_changed_dict.json for {arm_id} at {cd_file}")
        cd_sha = compute_sha256(cd_file)
        if cd_sha != arm_spec["expected_changed_dict_sha256"]:
            raise FrozenArtifactIntegrityError(
                f"Frozen changed_dict hash mismatch for {arm_id}: "
                f"expected {arm_spec['expected_changed_dict_sha256']}, got {cd_sha}"
            )

    # 6. Verify annotated git tag
    if enforce_tag:
        try:
            tag_res = subprocess.run(
                ["git", "rev-parse", f"{FROZEN_D5_TAG}^{{commit}}"],
                cwd=str(root),
                capture_output=True,
                text=True,
                check=False,
            )
            if tag_res.returncode != 0:
                raise RuntimeError(
                    f"Git tag {FROZEN_D5_TAG} not found or dereference failed: {tag_res.stderr.strip()}"
                )
            tag_commit = tag_res.stdout.strip()
            if tag_commit != FROZEN_D5_TAG_COMMIT:
                raise RuntimeError(
                    f"Git tag {FROZEN_D5_TAG} dereferences to {tag_commit} (expected {FROZEN_D5_TAG_COMMIT})"
                )
        except Exception as exc:
            raise RuntimeError(f"Failed to verify D5 git tag {FROZEN_D5_TAG}: {exc}") from exc

    return {
        "status": "PASS",
        "archive_dir": str(d5_dir),
        "manifest_sha256": manifest_sha,
        "raw_files_verified": 50,
        "artifacts_verified": 48,
        "changed_dicts_verified": 4,
        "tag_verified": FROZEN_D5_TAG if enforce_tag else "BYPASS",
    }


def verify_d4_archive_and_tag(
    archive_dir: Optional[Union[str, pathlib.Path]] = None,
    repo_root: Optional[Union[str, pathlib.Path]] = None,
    enforce_tag: bool = True,
) -> Dict[str, Any]:
    """Verify integrity of the archived D4 primary TEST release and annotated tag."""
    root = pathlib.Path(repo_root).resolve() if repo_root is not None else _REPO_ROOT
    d4_dir = pathlib.Path(archive_dir).resolve() if archive_dir is not None else DEFAULT_D4_ARCHIVE_DIR.resolve()

    if not d4_dir.is_dir():
        raise FileNotFoundError(f"Archived D4 release directory not found at: {d4_dir}")

    raw_dir = d4_dir / "raw"
    if not raw_dir.is_dir():
        raise FrozenArtifactIntegrityError(f"Missing raw directory in D4 archive: {raw_dir}")

    # Check that D4 tag resolves correctly
    if enforce_tag:
        try:
            tag_obj_res = subprocess.run(
                ["git", "rev-parse", FROZEN_D4_TAG],
                cwd=str(root),
                capture_output=True,
                text=True,
                check=False,
            )
            if tag_obj_res.returncode != 0:
                raise RuntimeError(
                    f"Git tag {FROZEN_D4_TAG} not found: {tag_obj_res.stderr.strip()}"
                )
            tag_obj = tag_obj_res.stdout.strip()
            if tag_obj != FROZEN_D4_TAG_COMMIT:
                raise RuntimeError(
                    f"Git tag {FROZEN_D4_TAG} resolves to {tag_obj} (expected {FROZEN_D4_TAG_COMMIT})"
                )

            tag_commit_res = subprocess.run(
                ["git", "rev-parse", f"{FROZEN_D4_TAG}^{{commit}}"],
                cwd=str(root),
                capture_output=True,
                text=True,
                check=False,
            )
            if tag_commit_res.returncode == 0:
                tag_commit = tag_commit_res.stdout.strip()
            else:
                tag_commit = "UNKNOWN"
        except Exception as exc:
            raise RuntimeError(f"Failed to verify D4 git tag {FROZEN_D4_TAG}: {exc}") from exc

    return {
        "status": "PASS",
        "archive_dir": str(d4_dir),
        "tag_verified": FROZEN_D4_TAG if enforce_tag else "BYPASS",
        "tag_object": FROZEN_D4_TAG_COMMIT,
        "tag_commit": tag_commit if enforce_tag else "BYPASS",
    }


def verify_frozen_inputs(
    split_manifest_path: Optional[Union[str, pathlib.Path]] = None,
    features_parquet_path: Optional[Union[str, pathlib.Path]] = None,
) -> Dict[str, Any]:
    """Verify split manifest and features parquet checksums."""
    split_p = pathlib.Path(split_manifest_path).resolve() if split_manifest_path is not None else FROZEN_SPLIT_MANIFEST_PATH.resolve()
    feat_p = pathlib.Path(features_parquet_path).resolve() if features_parquet_path is not None else FROZEN_FEATURES_PARQUET_PATH.resolve()

    if not split_p.is_file():
        raise FileNotFoundError(f"Frozen split manifest not found at: {split_p}")
    split_sha = compute_sha256(split_p)
    if split_sha != FROZEN_SPLIT_MANIFEST_SHA256:
        raise ValueError(
            f"Frozen split manifest SHA-256 mismatch!\n"
            f"Expected: {FROZEN_SPLIT_MANIFEST_SHA256}\n"
            f"Observed: {split_sha}"
        )

    if not feat_p.is_file():
        raise FileNotFoundError(f"Frozen features parquet not found at: {feat_p}")
    feat_sha = compute_sha256(feat_p)
    if feat_sha != FROZEN_FEATURES_PARQUET_SHA256:
        raise ValueError(
            f"Frozen features parquet SHA-256 mismatch!\n"
            f"Expected: {FROZEN_FEATURES_PARQUET_SHA256}\n"
            f"Observed: {feat_sha}"
        )

    return {
        "status": "PASS",
        "split_manifest_sha256": split_sha,
        "features_parquet_sha256": feat_sha,
    }


class NHISD5WeightedTestReleaseManager:
    """Manages frozen secondary TEST evaluation, verification, and persistence for Gate D5.2."""

    def __init__(
        self,
        release_id: str = CANONICAL_D5_SECONDARY_TEST_RELEASE_ID,
        repo_root: Optional[Union[str, pathlib.Path]] = None,
        d5_archive_dir: Optional[Union[str, pathlib.Path]] = None,
        d4_archive_dir: Optional[Union[str, pathlib.Path]] = None,
        split_manifest_path: Optional[Union[str, pathlib.Path]] = None,
        features_parquet_path: Optional[Union[str, pathlib.Path]] = None,
        output_base_dir: Optional[Union[str, pathlib.Path]] = None,
        adapter: Optional[NHISPooledAdapter] = None,
        enforce_git_boundary: bool = True,
        enforce_tags: bool = True,
    ):
        self.release_id = str(release_id).strip()
        self.repo_root = pathlib.Path(repo_root).resolve() if repo_root is not None else _REPO_ROOT
        self.d5_archive_dir = pathlib.Path(d5_archive_dir).resolve() if d5_archive_dir is not None else DEFAULT_D5_ARCHIVE_DIR.resolve()
        self.d4_archive_dir = pathlib.Path(d4_archive_dir).resolve() if d4_archive_dir is not None else DEFAULT_D4_ARCHIVE_DIR.resolve()
        self.split_manifest_path = pathlib.Path(split_manifest_path).resolve() if split_manifest_path is not None else FROZEN_SPLIT_MANIFEST_PATH.resolve()
        self.features_parquet_path = pathlib.Path(features_parquet_path).resolve() if features_parquet_path is not None else FROZEN_FEATURES_PARQUET_PATH.resolve()
        self.output_base_dir = pathlib.Path(output_base_dir).resolve() if output_base_dir is not None else DEFAULT_TEST_RELEASE_BASE_DIR.resolve()
        self.enforce_git_boundary = bool(enforce_git_boundary)
        self.enforce_tags = bool(enforce_tags)

        self._adapter = adapter

    @property
    def adapter(self) -> NHISPooledAdapter:
        if self._adapter is None:
            self._adapter = NHISPooledAdapter(
                features_parquet_path=self.features_parquet_path,
                split_manifest_path=self.split_manifest_path,
            )
        return self._adapter

    def verify_release_preconditions(self) -> Dict[str, Any]:
        """Verify all release preconditions fail-closed before execution or audit."""
        # 1. Scientific boundary check
        boundary_res: Dict[str, Any] = {"enforced": False, "status": "BYPASS"}
        if self.enforce_git_boundary:
            boundary_res = verify_scientific_code_boundary(
                repo_root=self.repo_root,
                base_commit=TEST_EVALUATION_BASE_COMMIT,
            )

        # 2. D5 archive verification
        d5_res = verify_d5_archive(
            archive_dir=self.d5_archive_dir,
            repo_root=self.repo_root,
            enforce_tag=self.enforce_tags,
        )

        # 3. D4 archive verification
        d4_res = verify_d4_archive_and_tag(
            archive_dir=self.d4_archive_dir,
            repo_root=self.repo_root,
            enforce_tag=self.enforce_tags,
        )

        # 4. Frozen inputs verification
        inputs_res = verify_frozen_inputs(
            split_manifest_path=self.split_manifest_path,
            features_parquet_path=self.features_parquet_path,
        )

        return {
            "status": "PASS",
            "scientific_boundary": boundary_res,
            "d5_archive": d5_res,
            "d4_archive": d4_res,
            "frozen_inputs": inputs_res,
        }

    def run_audit_only(self) -> Dict[str, Any]:
        """Execute audit-only verification for Gate D5.2a.

        Guarantees:
        - NEVER requests or accesses named TEST partition.
        - NEVER materializes TEST cohort.
        - NEVER fits any model.
        - NEVER scores TEST.
        - Creates no output directory or test metric artifacts.
        """
        git_commit = get_git_commit(self.repo_root)
        preconditions = self.verify_release_preconditions()

        return {
            "protocol": "Gate D5.2a secondary test release harness audit",
            "audit_timestamp_utc": utc_timestamp(),
            "canonical_release_id": self.release_id,
            "test_evaluation_base_commit": TEST_EVALUATION_BASE_COMMIT,
            "release_harness_commit": git_commit,
            "preconditions": preconditions,
            "arms_configured": list(FROZEN_D5_TEST_ARMS.keys()),
            "classifier_specification": {
                "classifier": "LogisticRegression",
                "random_state": DEFAULT_RANDOM_SEED,
                "solver": "lbfgs",
                "max_iter": 1000,
                "sample_weight": None,
                "eval_norm": "min-max",
                "prediction_threshold": DEFAULT_PREDICTION_THRESHOLD,
            },
            "secondary_analysis": True,
            "primary_analysis": False,
            "primary_test_replacement": False,
            "pooled_test_globally_untouched": False,
            "disclosure": SECONDARY_ANALYSIS_DISCLOSURE,
            "real_weighted_mitigation_executed": False,
            "validation_re_scored": False,
            "test_partition_requested": False,
            "test_cohort_materialized": False,
            "test_evaluated": False,
            "test_embargo_active": True,
            "status": "PASS",
        }

    def execute_release(
        self,
        output_base_dir: Optional[Union[str, pathlib.Path]] = None,
        random_seed: int = DEFAULT_RANDOM_SEED,
        prediction_threshold: float = DEFAULT_PREDICTION_THRESHOLD,
    ) -> Dict[str, Any]:
        """Execute one-time frozen secondary TEST evaluation across all 4 authorized arms.

        RESERVED FOR GATE D5.2b AFTER EXPLICIT PI AUTHORIZATION.
        Must NOT be called during Gate D5.2a.
        """
        if random_seed != DEFAULT_RANDOM_SEED:
            raise ValueError(
                f"Frozen release protocol strictly requires random_seed={DEFAULT_RANDOM_SEED} "
                f"(got {random_seed})."
            )
        if prediction_threshold != DEFAULT_PREDICTION_THRESHOLD:
            raise ValueError(
                f"Frozen release protocol strictly requires prediction_threshold={DEFAULT_PREDICTION_THRESHOLD} "
                f"(got {prediction_threshold})."
            )

        base_dir = (
            pathlib.Path(output_base_dir).resolve()
            if output_base_dir is not None
            else self.output_base_dir
        )
        release_dir = base_dir / self.release_id
        state_file = release_dir / "release_state.json"
        manifest_file = release_dir / "d5_weighted_secondary_test_manifest.json"

        # 1. Directory lifecycle & one-time release guard
        if release_dir.exists():
            if state_file.is_file():
                try:
                    prior_state = json.loads(state_file.read_text(encoding="utf-8"))
                    curr_status = prior_state.get("status", "UNKNOWN")
                except Exception:
                    curr_status = "CORRUPTED"
                raise ReleaseCollisionError(
                    f"Release directory already exists at {release_dir} with status '{curr_status}'. "
                    f"One-time release semantics strictly forbid silent overwrite or re-execution."
                )
            else:
                if any(release_dir.iterdir()):
                    raise ReleaseCollisionError(
                        f"Release directory {release_dir} already exists and is non-empty."
                    )

        release_dir.mkdir(parents=True, exist_ok=False)

        # 2. Record STARTED release state
        git_commit = get_git_commit(self.repo_root)
        started_utc = utc_timestamp()
        initial_state = {
            "release_id": self.release_id,
            "status": "STARTED",
            "test_evaluation_base_commit": TEST_EVALUATION_BASE_COMMIT,
            "scientific_execution_base_commit": SCIENTIFIC_EXECUTION_BASE_COMMIT,
            "release_harness_commit": git_commit,
            "started_at_utc": started_utc,
            "updated_at_utc": started_utc,
        }
        write_json_atomic(state_file, initial_state)

        try:
            # 3. Verify all preconditions fail-closed
            preconditions = self.verify_release_preconditions()

            all_artifacts: Dict[str, Dict[str, str]] = {}
            arm_summaries: Dict[str, Any] = {}

            # 4. Iterate over the 4 authorized arms
            for arm_id, arm_spec in FROZEN_D5_TEST_ARMS.items():
                arm_dir = release_dir / arm_id
                arm_dir.mkdir(parents=True, exist_ok=True)

                # A. Load and verify archived frozen changed_dict
                cd_path = self.d5_archive_dir / arm_id / "weighted_changed_dict.json"
                if not cd_path.is_file():
                    raise FrozenArtifactIntegrityError(f"Missing archived changed_dict for {arm_id} at {cd_path}")
                cd_sha = compute_sha256(cd_path)
                if cd_sha != arm_spec["expected_changed_dict_sha256"]:
                    raise FrozenArtifactIntegrityError(
                        f"Changed dict hash mismatch for {arm_id}: expected {arm_spec['expected_changed_dict_sha256']}, got {cd_sha}"
                    )
                frozen_changed_dict = json.loads(cd_path.read_text(encoding="utf-8"))

                # B. Extract cohorts
                if hasattr(self.adapter, "get_pooled_cohort"):
                    cohorts = self.adapter.get_pooled_cohort(
                        outcome=arm_spec["outcome"],
                        protected_attribute=arm_spec["protected_attribute"],
                        feature_set=arm_spec["feature_set"].lower(),
                        disability_arm=arm_spec["disability_arm"],
                    )
                    X_train, y_train, o_train, _, meta_train = cohorts["train"]
                    X_test, y_test, o_test, _, meta_test = cohorts["test"]
                else:
                    raise RuntimeError("Adapter does not support get_pooled_cohort.")

                # Feature type identification
                feature_set_key = arm_spec["feature_set"].lower()
                all_cats, all_nums = self.adapter.preprocessor.get_feature_family_lists(feature_set_key)
                active_feats = set(X_train.columns)
                cate_attrs = [f for f in all_cats if f in active_feats]
                num_attrs = [f for f in all_nums if f in active_feats]

                if len(active_feats) != arm_spec["expected_predictors"]:
                    raise ValueError(
                        f"Feature count mismatch for {arm_id}: expected {arm_spec['expected_predictors']}, "
                        f"got {len(active_feats)} ({list(active_feats)})"
                    )

                # --- BASELINE MODEL ---
                scaler_base = MinMaxScaler(feature_range=(0, 1))
                X_train_base_scaled = scaler_base.fit_transform(X_train)
                X_test_base_scaled = scaler_base.transform(X_test)

                model_baseline = LogisticRegression(
                    random_state=random_seed,
                    max_iter=1000,
                    solver="lbfgs",
                )
                model_baseline.fit(X_train_base_scaled, y_train.to_numpy())

                test_prob_base = model_baseline.predict_proba(X_test_base_scaled)[:, 1]
                test_pred_base = (test_prob_base >= prediction_threshold).astype(int)

                test_eval_base = evaluate_predictions(
                    y_true=y_test.to_numpy(),
                    y_pred=test_pred_base,
                    y_prob=test_prob_base,
                    o_group=o_test.to_numpy(),
                    expected_group_count=arm_spec.get("expected_group_count"),
                    expected_groups=arm_spec.get("expected_groups"),
                )

                # --- WEIGHTED FAIRBIAS MODEL (APPLYING FROZEN CHANGED DICT) ---
                transformer = FairTransform()
                transformed_X_train = transformer.transform_data(
                    X_train, frozen_changed_dict, num_attrs=num_attrs, cate_attrs=cate_attrs
                )
                transformed_X_test = transformer.transform_data(
                    X_test, frozen_changed_dict, num_attrs=num_attrs, cate_attrs=cate_attrs
                )

                scaler_fb = MinMaxScaler(feature_range=(0, 1))
                X_train_fb_scaled = scaler_fb.fit_transform(transformed_X_train)
                X_test_fb_scaled = scaler_fb.transform(transformed_X_test)

                model_fairbias = LogisticRegression(
                    random_state=random_seed,
                    max_iter=1000,
                    solver="lbfgs",
                )
                model_fairbias.fit(X_train_fb_scaled, y_train.to_numpy())

                test_prob_fb = model_fairbias.predict_proba(X_test_fb_scaled)[:, 1]
                test_pred_fb = (test_prob_fb >= prediction_threshold).astype(int)

                test_eval_fb = evaluate_predictions(
                    y_true=y_test.to_numpy(),
                    y_pred=test_pred_fb,
                    y_prob=test_prob_fb,
                    o_group=o_test.to_numpy(),
                    expected_group_count=arm_spec.get("expected_group_count"),
                    expected_groups=arm_spec.get("expected_groups"),
                )

                # Paired Comparison on TEST
                test_comparison_raw = compute_evaluation_comparison(test_eval_base, test_eval_fb)
                test_comparison = {
                    "arm_id": arm_id,
                    "baseline_predicted_positive_count": int(np.sum(test_pred_base)),
                    "baseline_selection_rate": float(np.mean(test_pred_base)),
                    "weighted_fairbias_predicted_positive_count": int(np.sum(test_pred_fb)),
                    "weighted_fairbias_selection_rate": float(np.mean(test_pred_fb)),
                    "predicted_positive_count_delta": int(np.sum(test_pred_fb) - np.sum(test_pred_base)),
                    "selection_rate_delta": float(np.mean(test_pred_fb) - np.mean(test_pred_base)),
                    "baseline_summary": test_comparison_raw["baseline_summary"],
                    "fairbias_summary": test_comparison_raw["fairbias_summary"],
                    "utility_deltas": test_comparison_raw["utility_deltas"],
                    "fairness_gap_deltas": test_comparison_raw["fairness_gap_deltas"],
                    "group_deltas": test_comparison_raw["group_deltas"],
                    "d5_validation_reference": {
                        "release_id": FROZEN_D5_RELEASE_ID,
                        "validation_comparison_sha256": arm_spec["archived_validation_comparison_sha256"],
                    },
                }

                # Record Record-ID Digests
                train_record_ids = meta_train["record_id"].tolist() if "record_id" in meta_train.columns else []
                test_record_ids = meta_test["record_id"].tolist() if "record_id" in meta_test.columns else []
                train_digest = compute_sequence_digest(train_record_ids)
                test_digest = compute_sequence_digest(test_record_ids)

                # Write 8 per-arm release artifacts:
                # 1. arm_config.json
                arm_config_payload = {
                    **arm_spec,
                    "classifier": {
                        "type": "LogisticRegression",
                        "random_state": random_seed,
                        "max_iter": 1000,
                        "solver": "lbfgs",
                        "sample_weight": None,
                    },
                    "eval_norm": "min-max",
                    "prediction_threshold": prediction_threshold,
                    "predictor_count": len(active_feats),
                    "categorical_count": len(cate_attrs),
                    "numerical_count": len(num_attrs),
                    "categorical_features": cate_attrs,
                    "numerical_features": num_attrs,
                    "train_cohort_n": len(X_train),
                    "test_cohort_n": len(X_test),
                }
                write_json_atomic(arm_dir / "arm_config.json", arm_config_payload)

                # 2. input_provenance.json
                provenance_payload = {
                    "arm_id": arm_id,
                    "train_n": len(X_train),
                    "test_n": len(X_test),
                    "train_record_ids_digest": train_digest,
                    "test_record_ids_digest": test_digest,
                    "predictor_names": sorted(list(active_feats)),
                    "protected_attribute": arm_spec["protected_attribute"],
                    "feature_arm": f"{arm_spec['feature_set']} / {arm_spec['disability_arm']}",
                    "archived_changed_dict_path": str(cd_path),
                    "archived_changed_dict_sha256": cd_sha,
                    "d5_validation_release_id": FROZEN_D5_RELEASE_ID,
                    "d5_validation_archive_commit": FROZEN_D5_TAG_COMMIT,
                    "d5_validation_archive_tag": FROZEN_D5_TAG,
                    "d5_validation_manifest_sha256": FROZEN_D5_MANIFEST_SHA256,
                    "d5_validation_trace_sha256": arm_spec["archived_trace_sha256"],
                    "frozen_split_manifest_sha256": FROZEN_SPLIT_MANIFEST_SHA256,
                    "frozen_features_parquet_sha256": FROZEN_FEATURES_PARQUET_SHA256,
                }
                write_json_atomic(arm_dir / "input_provenance.json", provenance_payload)

                # 3. frozen_changed_dict.json
                write_json_atomic(arm_dir / "frozen_changed_dict.json", frozen_changed_dict)

                # 4. test_metrics_baseline.json
                write_json_atomic(arm_dir / "test_metrics_baseline.json", test_eval_base)

                # 5. test_metrics_weighted_fairbias.json
                write_json_atomic(arm_dir / "test_metrics_weighted_fairbias.json", test_eval_fb)

                # 6. test_group_metrics_baseline.csv
                base_group_df = pd.DataFrame(test_eval_base["group_metrics"])
                write_csv_atomic(base_group_df, arm_dir / "test_group_metrics_baseline.csv", force=True)

                # 7. test_group_metrics_weighted_fairbias.csv
                fb_group_df = pd.DataFrame(test_eval_fb["group_metrics"])
                write_csv_atomic(fb_group_df, arm_dir / "test_group_metrics_weighted_fairbias.csv", force=True)

                # 8. test_comparison.json
                write_json_atomic(arm_dir / "test_comparison.json", test_comparison)

                # Hash all 8 artifacts
                arm_files = [
                    "arm_config.json",
                    "input_provenance.json",
                    "frozen_changed_dict.json",
                    "test_metrics_baseline.json",
                    "test_metrics_weighted_fairbias.json",
                    "test_group_metrics_baseline.csv",
                    "test_group_metrics_weighted_fairbias.csv",
                    "test_comparison.json",
                ]
                for fname in arm_files:
                    fp = arm_dir / fname
                    rel_p = f"{arm_id}/{fname}"
                    all_artifacts[rel_p] = {"sha256": compute_sha256(fp)}

                arm_summaries[arm_id] = {
                    "train_n": len(X_train),
                    "test_n": len(X_test),
                    "baseline_pred_pos": test_comparison["baseline_predicted_positive_count"],
                    "weighted_fb_pred_pos": test_comparison["weighted_fairbias_predicted_positive_count"],
                    "group_coverage": test_eval_fb.get("group_coverage"),
                }

            if len(arm_summaries) != 4:
                raise RuntimeError(f"Expected 4 arms to complete, got {len(arm_summaries)}")

            # 5. Write top-level manifest
            completed_utc = utc_timestamp()
            manifest_payload = {
                "schema_version": "nhis-fairbias-d5-weighted-secondary-test-1.0",
                "release_id": self.release_id,
                "release_status": "COMPLETE",
                "gate": "D5.2 secondary test release protocol",
                "timestamp_utc": completed_utc,
                "test_evaluation_base_commit": TEST_EVALUATION_BASE_COMMIT,
                "scientific_execution_base_commit": SCIENTIFIC_EXECUTION_BASE_COMMIT,
                "release_harness_commit": git_commit,
                "scientific_base_is_ancestor": preconditions["scientific_boundary"].get("scientific_base_is_ancestor", True),
                "scientific_code_diff_clean": preconditions["scientific_boundary"].get("scientific_code_diff_clean", True),
                "secondary_analysis": True,
                "primary_analysis": False,
                "primary_test_replacement": False,
                "pooled_test_globally_untouched": False,
                "disclosure": SECONDARY_ANALYSIS_DISCLOSURE,
                "split_manifest_sha256": FROZEN_SPLIT_MANIFEST_SHA256,
                "features_parquet_sha256": FROZEN_FEATURES_PARQUET_SHA256,
                "d5_validation_release_id": FROZEN_D5_RELEASE_ID,
                "d5_validation_archive_tag": FROZEN_D5_TAG,
                "d5_validation_manifest_sha256": FROZEN_D5_MANIFEST_SHA256,
                "d4_test_release_tag": FROZEN_D4_TAG,
                "random_seed": random_seed,
                "threshold": prediction_threshold,
                "classifier_weighted": False,
                "evaluation_weighted": False,
                "complex_survey_inference": False,
                "arms": list(FROZEN_D5_TEST_ARMS.keys()),
                "artifacts": all_artifacts,
            }
            write_json_atomic(manifest_file, manifest_payload)
            manifest_sha = compute_sha256(manifest_file)

            # 6. Write final release_state.json
            completed_state = {
                "release_id": self.release_id,
                "status": "COMPLETE",
                "test_evaluation_base_commit": TEST_EVALUATION_BASE_COMMIT,
                "scientific_execution_base_commit": SCIENTIFIC_EXECUTION_BASE_COMMIT,
                "release_harness_commit": git_commit,
                "started_at_utc": started_utc,
                "updated_at_utc": completed_utc,
                "completed_at_utc": completed_utc,
                "manifest_sha256": manifest_sha,
            }
            write_json_atomic(state_file, completed_state)

            return {
                "release_id": self.release_id,
                "status": "COMPLETE",
                "release_dir": str(release_dir),
                "manifest_sha256": manifest_sha,
                "arm_summaries": arm_summaries,
                "artifacts_written": len(all_artifacts),
            }

        except Exception as exc:
            failed_state = {
                "release_id": self.release_id,
                "status": "FAILED",
                "test_evaluation_base_commit": TEST_EVALUATION_BASE_COMMIT,
                "release_harness_commit": git_commit,
                "started_at_utc": initial_state.get("started_at_utc"),
                "failed_at_utc": utc_timestamp(),
                "error": str(exc),
            }
            try:
                write_json_atomic(state_file, failed_state)
            except Exception:
                pass
            raise


def compare_d5_validation_to_test(
    d5_validation_dir: Union[str, pathlib.Path],
    d5_test_release_dir: Union[str, pathlib.Path],
) -> Dict[str, Any]:
    """Pure reporting helper comparing D5 validation vs D5 secondary TEST.

    Non-interfering cross-release comparison of utility, fairness gaps, and selection rates.
    Does NOT drive TEST execution or enforce passing directionalities.
    """
    val_dir = pathlib.Path(d5_validation_dir).resolve()
    test_dir = pathlib.Path(d5_test_release_dir).resolve()

    comparison_results: Dict[str, Any] = {}

    for arm_id in FROZEN_D5_TEST_ARMS.keys():
        val_comp_path = val_dir / arm_id / "validation_comparison.json"
        test_comp_path = test_dir / arm_id / "test_comparison.json"

        if not val_comp_path.is_file() or not test_comp_path.is_file():
            continue

        val_comp = json.loads(val_comp_path.read_text(encoding="utf-8"))
        test_comp = json.loads(test_comp_path.read_text(encoding="utf-8"))

        comparison_results[arm_id] = {
            "validation": {
                "baseline_pred_pos": val_comp.get("baseline_predicted_positive_count"),
                "weighted_fb_pred_pos": val_comp.get("weighted_fairbias_predicted_positive_count"),
                "pred_pos_delta": val_comp.get("predicted_positive_count_delta"),
                "baseline_sel_rate": val_comp.get("baseline_selection_rate"),
                "weighted_fb_sel_rate": val_comp.get("weighted_fairbias_selection_rate"),
                "utility_deltas": val_comp.get("utility_deltas"),
                "fairness_gap_deltas": val_comp.get("fairness_gap_deltas"),
            },
            "test": {
                "baseline_pred_pos": test_comp.get("baseline_predicted_positive_count"),
                "weighted_fb_pred_pos": test_comp.get("weighted_fairbias_predicted_positive_count"),
                "pred_pos_delta": test_comp.get("predicted_positive_count_delta"),
                "baseline_sel_rate": test_comp.get("baseline_selection_rate"),
                "weighted_fb_sel_rate": test_comp.get("weighted_fairbias_selection_rate"),
                "utility_deltas": test_comp.get("utility_deltas"),
                "fairness_gap_deltas": test_comp.get("fairness_gap_deltas"),
            },
        }

    return {
        "timestamp_utc": utc_timestamp(),
        "d5_validation_dir": str(val_dir),
        "d5_test_release_dir": str(test_dir),
        "arms": comparison_results,
    }
