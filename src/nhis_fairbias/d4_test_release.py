"""Dedicated NHIS one-time frozen TEST release harness and Gate D4.1 orchestrator.

Guarantees:
1. Frozen Artifact Contract:
   - Uses exclusively the four frozen preflight run states from Gate D4.0.3.
   - Verifies SHA-256 hashes of final_changed_dict.json, train_fairbias_trace.json,
     validation_metrics_baseline.json, validation_metrics_fairbias.json, and
     validation_comparison.json before execution.
   - Fails closed if any hash is missing or mismatched.
2. No Re-learning / No FairBias Mitigation:
   - NEVER calls FairBiasMitigation.mitigate_step, calculate_epsilon, compute_threshold,
     calculate_nmi_dict, or run_fairbias_pipeline.
   - NEVER recomputes or relearns d_phi, epsilon, NMI, greedy ranking, category merges,
     power search, or drop decisions.
   - Applies the exact frozen train-learned final_changed_dict unchanged.
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
   - Refuses re-execution or silent overwrite if release state already exists.
5. Dual Modes:
   - audit_only (authorized for D4.1a): strictly verifies provenance, input hashes,
     preflight artifacts, arm configs, and embargo state without loading or scoring TEST.
   - execute_frozen_test (embargoed until D4.1b PI review): executes full test release.
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from fairbias.models import get_classifier
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
from .features import (
    DEFAULT_FEATURE_CONFIG,
    load_feature_registry,
)
from .pooled import NHISPooledAdapter
from .preprocessing import NHISLeakageError

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

PRIMARY_D4_TEST_RELEASE_PROTOCOL_ID: str = "NHIS_D4_PRIMARY_TEST_RELEASE_V1"
REQUIRED_ANALYSIS_COMMIT: str = "58a1e02339bf35dedc14dd2719d72417de6f6910"
PRIMARY_D4_RANDOM_SEED: int = 0
PRIMARY_D4_PREDICTION_THRESHOLD: float = 0.5

FROZEN_SPLIT_MANIFEST_PATH: pathlib.Path = _REPO_ROOT / "artifacts" / "nhis" / "d3" / "pooled_split_manifest.csv"
FROZEN_SPLIT_MANIFEST_SHA256: str = "6874a56f5484186dffdd5faeb7871c3bef85042ccfdf8d1e375fdde75a3ee9f5"

FROZEN_FEATURES_PARQUET_PATH: pathlib.Path = _REPO_ROOT / "data" / "processed" / "nhis" / "nhis_2022_2024_features.parquet"
FROZEN_FEATURES_PARQUET_SHA256: str = "49f415132ff0be0228f8533f9f74c48cd79ff6fa8be66db085f7329d9b083383"

FROZEN_D3_MANIFEST_PATH: pathlib.Path = _REPO_ROOT / "artifacts" / "nhis" / "d3" / "d3_manifest.json"
FROZEN_POOLED_SPLIT_AUDIT_PATH: pathlib.Path = _REPO_ROOT / "artifacts" / "nhis" / "d3" / "pooled_split_audit.json"

DEFAULT_PREFLIGHT_RUNS_DIR: pathlib.Path = _REPO_ROOT / "runs" / "nhis_d4_preflight"
DEFAULT_TEST_RELEASE_BASE_DIR: pathlib.Path = _REPO_ROOT / "runs" / "nhis_d4_test_release"

FROZEN_FOUR_ARM_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ARM_D3_001": {
        "arm_id": "ARM_D3_001",
        "protected_attribute": "SEX_A",
        "outcome": "MEDDL12M_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "full_feature",
        "expected_predictors": 21,
        "expected_groups": [1, 2],
        "expected_group_count": 2,
        "expected_pair_count": 1,
        "preflight_run_id": "20260901T150720Z-94c5ac1be7",
        "frozen_artifact_hashes": {
            "final_changed_dict.json": "9e90b303d28e5fd890dd6351d6e432c803c9d08cd051c4a1595859401cb76f6d",
            "train_fairbias_trace.json": "bf5ebda4d88b2330dab9e1ab043295cb2dbef7a7c5d3bfce16e49fe199e893ee",
            "validation_metrics_baseline.json": "b1bd508f5307edefffc77513da9fa7d853253477183a875d51662ed96a8bcc2b",
            "validation_metrics_fairbias.json": "57d3eb5cf6493b895b0fa8e05b16ec5f95d843a2431b5eddb65600bcc960ef63",
            "validation_comparison.json": "508c6b94165c1dae0e41959473ed878b17177d5e0d620d57ec587b2b04181857",
        },
    },
    "ARM_D3_002": {
        "arm_id": "ARM_D3_002",
        "protected_attribute": "HISPALLP_A",
        "outcome": "MEDDL12M_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "full_feature",
        "expected_predictors": 21,
        "expected_groups": [1, 2, 3, 4, 5, 6, 7],
        "expected_group_count": 7,
        "expected_pair_count": 21,
        "preflight_run_id": "20260901T152301Z-6f49937b9f",
        "frozen_artifact_hashes": {
            "final_changed_dict.json": "56dbab9d5a3470b668267f09ca658826e1a162d0391ed9be5dd1a56a30036c3c",
            "train_fairbias_trace.json": "a30f73844cef273960e9258082d11cfdba1d3c9edfc8038098af2faea0eccd6d",
            "validation_metrics_baseline.json": "fcb8ff739178c96e83b8ce30b9d0788cb22ea911b631f65e39d016aaceb2c056",
            "validation_metrics_fairbias.json": "4e9efa8921fa7ecce49469f7297257e3a945faec38fd026a3b0f0b3596c587ba",
            "validation_comparison.json": "c90cfb02210922c0e23d058991714aa0ccad8f365c6ae9e8b98a0500b683d9b4",
        },
    },
    "ARM_D3_003": {
        "arm_id": "ARM_D3_003",
        "protected_attribute": "DISAB3_A",
        "outcome": "MEDDL12M_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "full_feature",
        "expected_predictors": 21,
        "expected_groups": [1, 2],
        "expected_group_count": 2,
        "expected_pair_count": 1,
        "preflight_run_id": "20260901T152455Z-0b07868ca4",
        "frozen_artifact_hashes": {
            "final_changed_dict.json": "84489acf8e6f72964c74d9a9fc5719eec35d70a61e32880eca7439ddcefdde63",
            "train_fairbias_trace.json": "5fd7cca0191be73acb34cd0b143a3a831f3f40dc8d4aba236cc09bdb559e470a",
            "validation_metrics_baseline.json": "675cc14de795a758c65e7e599aa76586272a2ea65ede2b7954d74936f19b5fd2",
            "validation_metrics_fairbias.json": "b20c17358b9ae9b8374b033a5eb85c60b87c871f0806835f663da49a317c486b",
            "validation_comparison.json": "7300873532c4406caa7b81e58033fc9060fb98a99021bd023dd66c83607a3105",
        },
    },
    "ARM_D3_004": {
        "arm_id": "ARM_D3_004",
        "protected_attribute": "DISAB3_A",
        "outcome": "MEDDL12M_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "exclude_disability_components",
        "expected_predictors": 15,
        "expected_groups": [1, 2],
        "expected_group_count": 2,
        "expected_pair_count": 1,
        "preflight_run_id": "20260901T152529Z-44d2d5a4cc",
        "frozen_artifact_hashes": {
            "final_changed_dict.json": "2b0f3138b2ac93d6d3dbe39f23849d46b94ac36d8db3f4f35e64af260883cb27",
            "train_fairbias_trace.json": "34ca358545b7b9b4b08dd6cccfc1f5d0d32927d5d2eb85afcd023514b21eb511",
            "validation_metrics_baseline.json": "4e66e549f4c5c2c74d3950cfe6601bc5523ff6e6d34b6f7b426b7395d3809d38",
            "validation_metrics_fairbias.json": "0bf4cc6000ee0b398c7182c5bc022f9bca55059eb0ab66c81f9813bd0e82f7fe",
            "validation_comparison.json": "737b4a931ec2b141e88093ff32210cf8736c17f7e7146a9064f9cf9dc6a8ab0c",
        },
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


class ReleaseCollisionError(RuntimeError):
    """Raised when release directory already exists or contains prior release state."""
    pass


class FrozenArtifactIntegrityError(ValueError):
    """Raised when a frozen preflight artifact is missing or has a mismatched SHA-256 hash."""
    pass


class NHISD4TestReleaseHarness:
    """Dedicated release harness for Gate D4.1 frozen TEST execution.

    In Gate D4.1a, only verify_audit() is permitted to execute on real data.
    The real TEST evaluation path (execute_release) is implemented with strict fail-closed
    guards and one-time release semantics, tested via synthetic data, and reserved for D4.1b.
    """

    def __init__(
        self,
        adapter: Optional[NHISPooledAdapter] = None,
        features_parquet_path: Optional[Union[str, pathlib.Path]] = None,
        split_manifest_path: Optional[Union[str, pathlib.Path]] = None,
        d3_manifest_path: Optional[Union[str, pathlib.Path]] = None,
        pooled_split_audit_path: Optional[Union[str, pathlib.Path]] = None,
        preflight_runs_dir: Optional[Union[str, pathlib.Path]] = None,
        enforce_frozen_inputs: bool = True,
        enforce_frozen_artifacts: bool = True,
    ):
        self.enforce_frozen_inputs = bool(enforce_frozen_inputs)
        self.enforce_frozen_artifacts = bool(enforce_frozen_artifacts)

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

        p_runs_dir = (
            pathlib.Path(preflight_runs_dir).resolve()
            if preflight_runs_dir is not None
            else DEFAULT_PREFLIGHT_RUNS_DIR.resolve()
        )
        self.preflight_runs_dir = p_runs_dir

        self._adapter = adapter

    @property
    def adapter(self) -> NHISPooledAdapter:
        if self._adapter is None:
            self._adapter = NHISPooledAdapter(
                features_parquet_path=self.features_parquet_path,
                split_manifest_path=self.split_manifest_path,
            )
        return self._adapter

    def verify_frozen_inputs(self) -> Dict[str, Any]:
        """Verify frozen split manifest, features parquet, and D3 audit artifacts."""
        if not self.enforce_frozen_inputs:
            return {
                "enforced": False,
                "status": "BYPASS",
            }

        # 1. Split manifest existence and SHA-256
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

        # 2. Features parquet existence and SHA-256
        feat_path = pathlib.Path(self.features_parquet_path).resolve()
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

        # 3. D3 manifest status
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

        # 4. Pooled split audit status
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
        d0_match = audit_data.get("d0_outcome_totals_match")
        if d0_match is not True:
            raise ValueError(
                f"Pooled split audit d0_outcome_totals_match is {d0_match!r} (expected True) at {self.pooled_split_audit_path}"
            )

        return {
            "enforced": True,
            "split_manifest_path": str(self.split_manifest_path),
            "split_manifest_sha256": split_sha,
            "features_parquet_path": str(feat_path),
            "features_parquet_sha256": feat_sha,
            "d3_gate_status": d3_status,
            "pooled_split_audit_status": audit_status,
            "d0_outcome_totals_match": True,
            "status": "PASS",
        }

    def verify_frozen_preflight_artifacts(self) -> Dict[str, Any]:
        """Verify existence and SHA-256 of all frozen artifacts across the four authorized arms."""
        if not self.enforce_frozen_artifacts:
            return {
                "enforced": False,
                "status": "BYPASS",
            }

        arm_results: Dict[str, Any] = {}
        for arm_id, arm_spec in FROZEN_FOUR_ARM_REGISTRY.items():
            run_id = arm_spec["preflight_run_id"]
            run_dir = self.preflight_runs_dir / run_id
            if not run_dir.is_dir():
                raise FileNotFoundError(
                    f"Frozen preflight run directory for {arm_id} not found: {run_dir}"
                )

            verified_hashes: Dict[str, str] = {}
            for fname, expected_sha in arm_spec["frozen_artifact_hashes"].items():
                fpath = run_dir / fname
                if not fpath.is_file():
                    raise FrozenArtifactIntegrityError(
                        f"Missing frozen preflight artifact for {arm_id}: {fpath}"
                    )
                observed_sha = compute_sha256(fpath)
                if observed_sha != expected_sha:
                    raise FrozenArtifactIntegrityError(
                        f"Frozen artifact SHA-256 mismatch for {arm_id} file {fname}!\n"
                        f"Expected: {expected_sha}\n"
                        f"Observed: {observed_sha}\n"
                        f"Path: {fpath}"
                    )
                verified_hashes[fname] = observed_sha

            arm_results[arm_id] = {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "verified_artifacts": verified_hashes,
                "status": "PASS",
            }

        return {
            "enforced": True,
            "preflight_runs_dir": str(self.preflight_runs_dir),
            "arms": arm_results,
            "status": "PASS",
        }

    def run_audit(self) -> Dict[str, Any]:
        """Execute comprehensive audit verification for Gate D4.1a.

        Guarantees:
        - NEVER accesses or evaluates TEST data.
        - Verifies git provenance, frozen input hashes, D3 statuses, four preflight run IDs,
          and all frozen artifact checksums.
        - Asserts real_test_evaluated == False.
        """
        git_head = get_git_commit(_REPO_ROOT)
        input_audit = self.verify_frozen_inputs()
        artifact_audit = self.verify_frozen_preflight_artifacts()

        return {
            "audit_timestamp": utc_timestamp(),
            "protocol_id": PRIMARY_D4_TEST_RELEASE_PROTOCOL_ID,
            "required_analysis_commit": REQUIRED_ANALYSIS_COMMIT,
            "current_git_commit": git_head,
            "input_audit": input_audit,
            "preflight_artifact_audit": artifact_audit,
            "four_arm_registry_count": len(FROZEN_FOUR_ARM_REGISTRY),
            "classifier_specification": {
                "classifier": "LR",
                "random_state": PRIMARY_D4_RANDOM_SEED,
                "max_iter": 1000,
                "solver": "lbfgs",
                "eval_norm": "min-max",
                "prediction_threshold": PRIMARY_D4_PREDICTION_THRESHOLD,
            },
            "real_test_evaluated": False,
            "test_embargo_active": True,
            "status": "PASS",
        }

    def execute_release(
        self,
        release_id: str,
        output_base_dir: Optional[Union[str, pathlib.Path]] = None,
        random_seed: int = PRIMARY_D4_RANDOM_SEED,
        prediction_threshold: float = PRIMARY_D4_PREDICTION_THRESHOLD,
    ) -> Dict[str, Any]:
        """Execute one-time frozen TEST evaluation across all four authorized arms.

        RESERVED FOR GATE D4.1b AFTER EXPLICIT PI AUTHORIZATION.
        Must NOT be called during Gate D4.1a.

        Single-execution semantics:
        - Creates a canonical release directory for release_id.
        - If release directory or state already exists, aborts fail-closed.
        - Writes STARTED release_state.json atomically prior to scoring.
        - Executes Baseline and FairBias evaluations using frozen changed_dict.
        - Writes COMPLETE release_state.json and top-level release manifest atomically.
        """
        if random_seed != PRIMARY_D4_RANDOM_SEED:
            raise ValueError(
                f"Frozen release protocol strictly requires random_seed={PRIMARY_D4_RANDOM_SEED} "
                f"(got {random_seed})."
            )
        if prediction_threshold != PRIMARY_D4_PREDICTION_THRESHOLD:
            raise ValueError(
                f"Frozen release protocol strictly requires prediction_threshold={PRIMARY_D4_PREDICTION_THRESHOLD} "
                f"(got {prediction_threshold})."
            )

        # 1. Directory lifecycle & one-time release guard
        base_dir = (
            pathlib.Path(output_base_dir).resolve()
            if output_base_dir is not None
            else DEFAULT_TEST_RELEASE_BASE_DIR.resolve()
        )
        release_dir = base_dir / release_id
        state_file = release_dir / "release_state.json"

        if release_dir.exists():
            if state_file.is_file():
                try:
                    prior_state = json.loads(state_file.read_text(encoding="utf-8"))
                    curr_status = prior_state.get("status", "UNKNOWN")
                except Exception:
                    curr_status = "CORRUPTED"
                raise ReleaseCollisionError(
                    f"Release directory already exists at {release_dir} with status '{curr_status}'. "
                    f"One-time release semantics strictly forbid silent overwrite or re-execution. "
                    f"Require explicit PI review before recovery."
                )
            else:
                # Directory exists with non-state contents
                if any(release_dir.iterdir()):
                    raise ReleaseCollisionError(
                        f"Release directory {release_dir} already exists and is non-empty. "
                        f"Must use a fresh canonical release directory."
                    )

        release_dir.mkdir(parents=True, exist_ok=False)

        # 2. Record STARTED release state
        start_time = time.time()
        git_head = get_git_commit(_REPO_ROOT)
        initial_state = {
            "protocol_id": PRIMARY_D4_TEST_RELEASE_PROTOCOL_ID,
            "release_id": release_id,
            "status": "STARTED",
            "started_at": utc_timestamp(),
            "git_commit": git_head,
            "required_analysis_commit": REQUIRED_ANALYSIS_COMMIT,
        }
        write_json_atomic(state_file, initial_state)

        try:
            # 3. Verify inputs and frozen preflight artifacts fail-closed
            input_audit = self.verify_frozen_inputs()
            preflight_audit = self.verify_frozen_preflight_artifacts()

            arms_manifest_data: Dict[str, Any] = {}
            arm_directories: List[pathlib.Path] = []

            # 4. Iterate over the four authorized arms
            for arm_id, arm_spec in FROZEN_FOUR_ARM_REGISTRY.items():
                arm_dir = release_dir / arm_id
                arm_dir.mkdir(parents=True, exist_ok=True)
                arm_directories.append(arm_dir)

                # A. Load frozen preflight changed_dict
                run_id = arm_spec["preflight_run_id"]
                preflight_dir = self.preflight_runs_dir / run_id
                changed_dict_path = preflight_dir / "final_changed_dict.json"
                if not changed_dict_path.is_file():
                    raise FrozenArtifactIntegrityError(
                        f"Missing frozen final_changed_dict.json for {arm_id} at {changed_dict_path}"
                    )
                cd_sha = compute_sha256(changed_dict_path)
                expected_cd_sha = arm_spec["frozen_artifact_hashes"]["final_changed_dict.json"]
                if cd_sha != expected_cd_sha:
                    raise FrozenArtifactIntegrityError(
                        f"final_changed_dict.json SHA-256 mismatch for {arm_id}: "
                        f"expected {expected_cd_sha}, got {cd_sha}"
                    )
                frozen_changed_dict = json.loads(changed_dict_path.read_text(encoding="utf-8"))

                # B. Extract TRAIN and TEST cohorts from adapter (NEVER use val)
                cohorts = self.adapter.get_pooled_cohort(
                    outcome=arm_spec["outcome"],
                    protected_attribute=arm_spec["protected_attribute"],
                    feature_set=arm_spec["feature_set"].lower(),
                    disability_arm=arm_spec["disability_arm"],
                )
                X_train, y_train, o_train, _, meta_train = cohorts["train"]
                X_test, y_test, o_test, _, meta_test = cohorts["test"]

                # Semantic feature types for transformer
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

                # --- FAIRBIAS MODEL ---
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

                # Paired Before-vs-After Comparison on TEST
                test_comparison = compute_evaluation_comparison(test_eval_base, test_eval_fb)

                # Write per-arm release artifacts
                # 1. arm_config.json
                arm_config_payload = {
                    **arm_spec,
                    "algorithm_mode": "tang2024_paper_faithful",
                    "classifier": {
                        "type": "LR",
                        "random_state": random_seed,
                        "max_iter": 1000,
                        "solver": "lbfgs",
                    },
                    "eval_norm": "min-max",
                    "prediction_threshold": prediction_threshold,
                    "active_feature_count": len(active_feats),
                    "categorical_count": len(cate_attrs),
                    "numerical_count": len(num_attrs),
                    "categorical_features": cate_attrs,
                    "numerical_features": num_attrs,
                    "train_cohort_n": len(X_train),
                    "test_cohort_n": len(X_test),
                }
                write_json_atomic(arm_dir / "arm_config.json", arm_config_payload)

                # 2. frozen_preflight_provenance.json
                provenance_payload = {
                    "arm_id": arm_id,
                    "preflight_run_id": run_id,
                    "frozen_artifact_hashes": arm_spec["frozen_artifact_hashes"],
                    "verified": True,
                    "verified_at": utc_timestamp(),
                }
                write_json_atomic(arm_dir / "frozen_preflight_provenance.json", provenance_payload)

                # 3. test_metrics_baseline.json
                write_json_atomic(arm_dir / "test_metrics_baseline.json", test_eval_base)

                # 4. test_metrics_fairbias.json
                write_json_atomic(arm_dir / "test_metrics_fairbias.json", test_eval_fb)

                # 5. test_group_metrics_baseline.csv
                base_group_df = pd.DataFrame(test_eval_base["group_metrics"])
                write_csv_atomic(base_group_df, arm_dir / "test_group_metrics_baseline.csv", force=True)

                # 6. test_group_metrics_fairbias.csv
                fb_group_df = pd.DataFrame(test_eval_fb["group_metrics"])
                write_csv_atomic(fb_group_df, arm_dir / "test_group_metrics_fairbias.csv", force=True)

                # 7. test_comparison.json
                write_json_atomic(arm_dir / "test_comparison.json", test_comparison)

                # Compute per-arm artifact checksums
                arm_artifact_hashes: Dict[str, Dict[str, Any]] = {}
                for fname in [
                    "arm_config.json",
                    "frozen_preflight_provenance.json",
                    "test_metrics_baseline.json",
                    "test_metrics_fairbias.json",
                    "test_group_metrics_baseline.csv",
                    "test_group_metrics_fairbias.csv",
                    "test_comparison.json",
                ]:
                    fp = arm_dir / fname
                    arm_artifact_hashes[fname] = {
                        "sha256": compute_sha256(fp),
                        "size_bytes": fp.stat().st_size,
                    }

                arms_manifest_data[arm_id] = {
                    "arm_id": arm_id,
                    "protected_attribute": arm_spec["protected_attribute"],
                    "feature_count": len(active_feats),
                    "train_cohort_n": len(X_train),
                    "test_cohort_n": len(X_test),
                    "group_coverage": test_eval_fb["group_coverage"],
                    "frozen_changed_dict_sha256": cd_sha,
                    "output_artifacts": arm_artifact_hashes,
                }

            # 5. Top-level release manifest
            manifest_payload = {
                "schema_version": "nhis-fairbias-d4-test-release-1.0",
                "release_id": release_id,
                "protocol_id": PRIMARY_D4_TEST_RELEASE_PROTOCOL_ID,
                "release_status": "COMPLETE",
                "release_harness_git_commit": git_head,
                "frozen_analysis_commit": REQUIRED_ANALYSIS_COMMIT,
                "timestamp": utc_timestamp(),
                "split_manifest_sha256": FROZEN_SPLIT_MANIFEST_SHA256,
                "features_parquet_sha256": FROZEN_FEATURES_PARQUET_SHA256,
                "frozen_preflight_runs": {
                    arm_id: {
                        "run_id": spec["preflight_run_id"],
                        "artifact_hashes": spec["frozen_artifact_hashes"],
                    }
                    for arm_id, spec in FROZEN_FOUR_ARM_REGISTRY.items()
                },
                "algorithm_mode": "tang2024_paper_faithful",
                "random_seed": random_seed,
                "classifier_specification": {
                    "classifier": "LR",
                    "random_state": random_seed,
                    "max_iter": 1000,
                    "solver": "lbfgs",
                    "eval_norm": "min-max",
                    "prediction_threshold": prediction_threshold,
                },
                "prediction_threshold": prediction_threshold,
                "arms": arms_manifest_data,
                "validation_used_for_test_execution": False,
                "fairbias_refit_or_relearned": False,
                "test_evaluated": True,
                "execution_time_seconds": round(time.time() - start_time, 3),
            }

            manifest_path = release_dir / "d4_test_release_manifest.json"
            write_json_atomic(manifest_path, manifest_payload)

            # 6. Update release state to COMPLETE
            final_state = {
                **initial_state,
                "status": "COMPLETE",
                "completed_at": utc_timestamp(),
                "manifest_sha256": compute_sha256(manifest_path),
                "execution_time_seconds": round(time.time() - start_time, 3),
            }
            write_json_atomic(state_file, final_state)

            return {
                "release_id": release_id,
                "release_dir": str(release_dir),
                "manifest_path": str(manifest_path),
                "manifest": manifest_payload,
                "state": final_state,
            }

        except Exception as exc:
            # On unexpected failure, update state to FAILED without overwriting prior files
            failed_state = {
                **initial_state,
                "status": "FAILED",
                "failed_at": utc_timestamp(),
                "error": str(exc),
            }
            try:
                write_json_atomic(state_file, failed_state)
            except Exception:
                pass
            raise
