"""NHIS Gate D5.1b: Frozen TRAIN/VALIDATION execution-release harness for survey-weighted geometry extension.

Guarantees:
1. Canonical Release Protocol:
   - Release ID: NHIS_D5_WEIGHTED_TRAIN_VAL_V1_bc6034e5
   - Exactly 4 frozen arms (D5_ARM_001 through D5_ARM_004).
   - Secondary TRAIN/VAL analysis (never primary TEST).
2. Fresh Directory & Fail-Closed State Machine:
   - Target root: runs/nhis_d5_weighted/releases/NHIS_D5_WEIGHTED_TRAIN_VAL_V1_bc6034e5/
   - Collision fails closed immediately.
   - States: STARTED -> COMPLETE (or FAILED on any error; no silent retries).
3. Embargo & Invariance:
   - TEST partition is strictly unrequested, unmaterialized, and unevaluated.
   - No --test CLI flag or test evaluation methods.
   - Classifier (LR) and evaluation remain unweighted.
   - Threshold = 0.5, Random Seed = 0.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .audit import write_csv_atomic
from .download import (
    compute_sha256,
    new_run_id,
    utc_timestamp,
    write_json_atomic,
)
from .d5_weighted_runner import (
    D4_COMPARISON_REGISTRY,
    FROZEN_D5_ARMS,
    FROZEN_D4_RELEASE_TAG,
    FROZEN_FEATURES_PARQUET_SHA256,
    FROZEN_PRIMARY_ANALYSIS_COMMIT,
    FROZEN_SPLIT_MANIFEST_SHA256,
    PRIMARY_D5_RANDOM_SEED,
    NHISD5WeightedRunner,
    compute_series_weight_diagnostics,
    get_git_commit,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

CANONICAL_D5_RELEASE_ID: str = "NHIS_D5_WEIGHTED_TRAIN_VAL_V1_bc6034e5"
SCIENTIFIC_EXECUTION_BASE_COMMIT: str = "bc6034e530d6e92bbfa82feadc0cf783302cb30f"

FROZEN_SCIENTIFIC_PATHS_SPEC: Tuple[str, ...] = (
    "src/fairbias/**",
    "src/nhis_fairbias/d5_weighted_runner.py",
)
FROZEN_SCIENTIFIC_PATHS_DIFF: Tuple[str, ...] = (
    "src/fairbias",
    "src/nhis_fairbias/d5_weighted_runner.py",
)

# Verified D4 Preflight Arm References (Provenance Only)
FROZEN_D4_PREFLIGHT_REFERENCES: Dict[str, Dict[str, Any]] = {
    "D5_ARM_001": {
        "reference_d4_arm": "ARM_D3_001",
        "run_id": "20260901T150720Z-94c5ac1be7",
        "manifest_sha256": "ced6d0b58148392cc0d8934b1ccba1b40839abd3b1780b7391acbc76343a1d03",
        "final_changed_dict_sha256": "9e90b303d28e5fd890dd6351d6e432c803c9d08cd051c4a1595859401cb76f6d",
        "train_fairbias_trace_sha256": "bf5ebda4d88b2330dab9e1ab043295cb2dbef7a7c5d3bfce16e49fe199e893ee",
        "validation_metrics_baseline_sha256": "b1bd508f5307edefffc77513da9fa7d853253477183a875d51662ed96a8bcc2b",
        "validation_metrics_fairbias_sha256": "57d3eb5cf6493b895b0fa8e05b16ec5f95d843a2431b5eddb65600bcc960ef63",
        "validation_comparison_sha256": "508c6b94165c1dae0e41959473ed878b17177d5e0d620d57ec587b2b04181857",
    },
    "D5_ARM_002": {
        "reference_d4_arm": "ARM_D3_002",
        "run_id": "20260901T152301Z-6f49937b9f",
        "manifest_sha256": "8f41fd4337f9b546601bf066a7c7269e6f40c54ce102456117881ddaa34c53f6",
        "final_changed_dict_sha256": "56dbab9d5a3470b668267f09ca658826e1a162d0391ed9be5dd1a56a30036c3c",
        "train_fairbias_trace_sha256": "a30f73844cef273960e9258082d11cfdba1d3c9edfc8038098af2faea0eccd6d",
        "validation_metrics_baseline_sha256": "fcb8ff739178c96e83b8ce30b9d0788cb22ea911b631f65e39d016aaceb2c056",
        "validation_metrics_fairbias_sha256": "4e9efa8921fa7ecce49469f7297257e3a945faec38fd026a3b0f0b3596c587ba",
        "validation_comparison_sha256": "c90cfb02210922c0e23d058991714aa0ccad8f365c6ae9e8b98a0500b683d9b4",
    },
    "D5_ARM_003": {
        "reference_d4_arm": "ARM_D3_003",
        "run_id": "20260901T152455Z-0b07868ca4",
        "manifest_sha256": "ac59860ca21c495155e630dc0ad3faaf8b54b376a520aad291c1134b7d6a6faa",
        "final_changed_dict_sha256": "84489acf8e6f72964c74d9a9fc5719eec35d70a61e32880eca7439ddcefdde63",
        "train_fairbias_trace_sha256": "5fd7cca0191be73acb34cd0b143a3a831f3f40dc8d4aba236cc09bdb559e470a",
        "validation_metrics_baseline_sha256": "675cc14de795a758c65e7e599aa76586272a2ea65ede2b7954d74936f19b5fd2",
        "validation_metrics_fairbias_sha256": "b20c17358b9ae9b8374b033a5eb85c60b87c871f0806835f663da49a317c486b",
        "validation_comparison_sha256": "7300873532c4406caa7b81e58033fc9060fb98a99021bd023dd66c83607a3105",
    },
    "D5_ARM_004": {
        "reference_d4_arm": "ARM_D3_004",
        "run_id": "20260901T152529Z-44d2d5a4cc",
        "manifest_sha256": "030a6f1f825b3075e35a01a133988043e40698384330c1521e3a0b44498225c1",
        "final_changed_dict_sha256": "2b0f3138b2ac93d6d3dbe39f23849d46b94ac36d8db3f4f35e64af260883cb27",
        "train_fairbias_trace_sha256": "34ca358545b7b9b4b08dd6cccfc1f5d0d32927d5d2eb85afcd023514b21eb511",
        "validation_metrics_baseline_sha256": "4e66e549f4c5c2c74d3950cfe6601bc5523ff6e6d34b6f7b426b7395d3809d38",
        "validation_metrics_fairbias_sha256": "0bf4cc6000ee0b398c7182c5bc022f9bca55059eb0ab66c81f9813bd0e82f7fe",
        "validation_comparison_sha256": "737b4a931ec2b141e88093ff32210cf8736c17f7e7146a9064f9cf9dc6a8ab0c",
    },
}

REQUIRED_PER_ARM_ARTIFACTS: Tuple[str, ...] = (
    "arm_config.json",
    "input_provenance.json",
    "weight_audit.json",
    "initial_weighted_dphi.json",
    "final_weighted_dphi.json",
    "weighted_changed_dict.json",
    "weighted_transform_trace.json",
    "validation_metrics_baseline.json",
    "validation_metrics_weighted_fairbias.json",
    "validation_group_metrics_baseline.csv",
    "validation_group_metrics_weighted_fairbias.csv",
    "validation_comparison.json",
)


def compute_sequence_digest(values: Sequence[Any]) -> str:
    """Deterministic SHA-256 digest of a sequence of values."""
    formatted = "\n".join(str(v) for v in values)
    return hashlib.sha256(formatted.encode("utf-8")).hexdigest()


def verify_scientific_code_boundary(
    repo_root: Optional[Union[str, pathlib.Path]] = None,
    base_commit: str = SCIENTIFIC_EXECUTION_BASE_COMMIT,
    target_commit: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify frozen scientific execution code boundary at release runtime.

    Enforces:
    1. Positive git repository and commit resolution (fail closed on git failure).
    2. base_commit is an ancestor of current HEAD (git merge-base --is-ancestor).
    3. Zero diff between base_commit and current HEAD for frozen scientific paths:
       - src/fairbias/**
       - src/nhis_fairbias/d5_weighted_runner.py
    """
    root = pathlib.Path(repo_root).resolve() if repo_root is not None else _REPO_ROOT

    # 1. Resolve target commit (defaults to current HEAD)
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
            f"Provenance failure: scientific execution base commit {base_commit} is not an ancestor of "
            f"current HEAD {resolved_target}. Canonical D5 release requires verified ancestor lineage; "
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
            f"Provenance failure: canonical D5 release cannot run because reviewed scientific execution "
            f"code has changed between base {base_commit} and current HEAD {resolved_target}. "
            f"Frozen scientific execution paths ({', '.join(FROZEN_SCIENTIFIC_PATHS_SPEC)}) "
            f"must have zero diff."
        )
    elif diff_res.returncode != 0:
        raise RuntimeError(
            f"git diff failed (exit {diff_res.returncode}): {diff_res.stderr.strip()}"
        )

    return {
        "scientific_execution_base_commit": base_commit,
        "current_git_commit": resolved_target,
        "scientific_base_is_ancestor": True,
        "scientific_code_diff_clean": True,
        "scientific_paths_checked": list(FROZEN_SCIENTIFIC_PATHS_SPEC),
    }


class NHISD5WeightedReleaseManager:
    """Orchestrates frozen TRAIN/VAL release execution, persistence, and verification for Gate D5."""

    def __init__(
        self,
        release_id: str = CANONICAL_D5_RELEASE_ID,
        releases_root: Optional[Union[str, pathlib.Path]] = None,
        runner: Optional[NHISD5WeightedRunner] = None,
        allow_substantive_execution: bool = False,
    ) -> None:
        self.release_id = str(release_id).strip()
        self.releases_root = (
            pathlib.Path(releases_root).resolve()
            if releases_root is not None
            else (_REPO_ROOT / "runs" / "nhis_d5_weighted" / "releases").resolve()
        )
        self.release_dir = self.releases_root / self.release_id
        self.allow_substantive_execution = bool(allow_substantive_execution)
        self.runner = runner or NHISD5WeightedRunner(
            enforce_frozen_inputs=True,
            allow_mitigation=self.allow_substantive_execution,
        )

    def verify_release_preconditions(self) -> Dict[str, Any]:
        """Verify canonical release ID, directory freshness, scientific boundary, and frozen input contract."""
        # 1. Canonical release ID validation
        if self.release_id != CANONICAL_D5_RELEASE_ID:
            raise ValueError(
                f"Invalid release ID: {self.release_id!r}. Expected canonical {CANONICAL_D5_RELEASE_ID}"
            )

        # 2. Fresh release directory requirement (Fail closed on duplicate/existing)
        if self.release_dir.exists():
            raise FileExistsError(
                f"Canonical release directory already exists: {self.release_dir}. "
                "Releases must be fresh; overwriting or merging is strictly forbidden."
            )

        # 3. Frozen input contract verification
        contract = self.runner.verify_frozen_input_contract()

        # 4. Scientific boundary verification: base commit must be ancestor of HEAD with zero diff in scientific paths
        boundary_info = verify_scientific_code_boundary(
            repo_root=_REPO_ROOT,
            base_commit=SCIENTIFIC_EXECUTION_BASE_COMMIT,
        )

        # 5. Weight contract verification across all four arms
        arm_audits = {}
        for arm_id in FROZEN_D5_ARMS:
            audit = self.runner.audit_arm_weights(arm_id)
            arm_audits[arm_id] = audit

        return {
            "release_id": self.release_id,
            "release_dir": str(self.release_dir),
            "current_git_commit": boundary_info["current_git_commit"],
            "scientific_execution_base_commit": SCIENTIFIC_EXECUTION_BASE_COMMIT,
            "scientific_base_is_ancestor": boundary_info["scientific_base_is_ancestor"],
            "scientific_code_diff_clean": boundary_info["scientific_code_diff_clean"],
            "scientific_paths_checked": boundary_info["scientific_paths_checked"],
            "contract": contract,
            "arm_audits_passed": True,
            "status": "PASS",
        }

    def write_release_state(
        self,
        status: str,
        *,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Atomically persist release_state.json."""
        self.release_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.release_dir / "release_state.json"

        # Load existing if available to preserve started_at
        existing_data = {}
        if state_path.is_file():
            try:
                existing_data = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                existing_data = {}

        payload: Dict[str, Any] = {
            "release_id": self.release_id,
            "status": status,
            "scientific_execution_base_commit": SCIENTIFIC_EXECUTION_BASE_COMMIT,
            "release_harness_commit": get_git_commit(),
            "started_at_utc": existing_data.get("started_at_utc") or (utc_timestamp() if status == "STARTED" else None),
            "updated_at_utc": utc_timestamp(),
        }

        if status == "COMPLETE":
            payload["completed_at_utc"] = utc_timestamp()
        elif status == "FAILED":
            payload["failed_at_utc"] = utc_timestamp()
            payload["error_type"] = error_type
            payload["error_message"] = error_message

        if extra:
            payload.update(extra)

        write_json_atomic(state_path, payload)
        return payload

    def run_audit_only(self) -> Dict[str, Any]:
        """Perform comprehensive read-only audit of inputs, contracts, and schema without mitigation."""
        pre = self.verify_release_preconditions()
        current_commit = pre["current_git_commit"]

        # Audit-only does not create the release directory or substantive artifacts
        return {
            "gate": "D5.1b",
            "mode": "AUDIT_ONLY",
            "canonical_release_id": CANONICAL_D5_RELEASE_ID,
            "release_dir_fresh": not self.release_dir.exists(),
            "scientific_execution_base_commit": SCIENTIFIC_EXECUTION_BASE_COMMIT,
            "release_harness_commit": current_commit,
            "scientific_base_is_ancestor": pre["scientific_base_is_ancestor"],
            "scientific_code_diff_clean": pre["scientific_code_diff_clean"],
            "scientific_paths_checked": pre["scientific_paths_checked"],
            "frozen_split_manifest_sha256": FROZEN_SPLIT_MANIFEST_SHA256,
            "frozen_features_parquet_sha256": FROZEN_FEATURES_PARQUET_SHA256,
            "four_arms": list(FROZEN_D5_ARMS.keys()),
            "preconditions": pre,
            "real_weighted_mitigation_executed": False,
            "validation_scored": False,
            "test_partition_requested": False,
            "test_cohort_materialized": False,
            "test_evaluated": False,
            "status": "PASS",
        }

    def execute_release(self) -> Dict[str, Any]:
        """Execute substantive four-arm release protocol and persist all artifacts (Future Gate D5.1c)."""
        if not self.allow_substantive_execution:
            raise RuntimeError(
                "Substantive release execution is forbidden in Gate D5.1b. "
                "allow_substantive_execution is False."
            )

        # 1. Preflight integrity checks
        pre_info = self.verify_release_preconditions()
        harness_commit = pre_info["current_git_commit"]

        # 2. Initialize fresh release directory and write STARTED
        self.release_dir.mkdir(parents=True, exist_ok=False)
        self.write_release_state("STARTED")

        arm_results: Dict[str, Any] = {}
        arm_artifact_hashes: Dict[str, Dict[str, str]] = {}
        all_artifact_hashes: Dict[str, str] = {}

        try:
            # 3. Execute all four arms sequentially
            for arm_id, arm_spec in FROZEN_D5_ARMS.items():
                arm_dir = self.release_dir / arm_id
                arm_dir.mkdir(parents=True, exist_ok=False)

                # Execute substantive train/validation run via runner
                exec_res = self.runner.execute_train_validation(
                    arm_id=arm_id,
                    random_seed=PRIMARY_D5_RANDOM_SEED,
                )
                arm_results[arm_id] = exec_res

                # Extract partition cohorts for provenance digests
                cohorts = self.runner.get_train_val_cohort(arm_id)
                X_train, y_train, o_train, w_train, meta_train = cohorts["train"]
                X_val, y_val, o_val, w_val, meta_val = cohorts["val"]

                feature_set_key = arm_spec["feature_set"].lower()
                all_cats, all_nums = self.runner.adapter.preprocessor.get_feature_family_lists(feature_set_key)
                active_cols = list(X_train.columns)
                active_cats = [c for c in all_cats if c in active_cols]
                active_nums = [c for c in all_nums if c in active_cols]

                # --- Artifact 1: arm_config.json ---
                d4_ref = FROZEN_D4_PREFLIGHT_REFERENCES.get(arm_id, {})
                arm_config_payload: Dict[str, Any] = {
                    "arm_id": arm_id,
                    "outcome": arm_spec["outcome"],
                    "protected_attribute": arm_spec["protected_attribute"],
                    "feature_set": arm_spec["feature_set"],
                    "disability_arm": arm_spec["disability_arm"],
                    "predictor_count": arm_spec["expected_predictors"],
                    "base_algorithm": "tang2024_paper_faithful",
                    "extension": "survey_weighted_geometry",
                    "survey_weight_variable": "WTFA_A",
                    "classifier": "LogisticRegression",
                    "classifier_weighted": False,
                    "evaluation_weighted": False,
                    "complex_survey_inference": False,
                    "random_seed": PRIMARY_D5_RANDOM_SEED,
                    "prediction_threshold": 0.5,
                    "PSTRAT_used_for_variance": False,
                    "PPSU_used_for_variance": False,
                    "d4_reference_arm": d4_ref.get("reference_d4_arm"),
                    "d4_reference_run_id": d4_ref.get("run_id"),
                    "d4_reference_hashes": d4_ref,
                }
                if arm_spec["protected_attribute"] == "HISPALLP_A":
                    arm_config_payload["group_count"] = 7
                    arm_config_payload["pair_count"] = 21
                    arm_config_payload["multicategory_status"] = "empirical multicategory pairwise extension"

                p1 = arm_dir / "arm_config.json"
                write_json_atomic(p1, arm_config_payload)

                # --- Artifact 2: input_provenance.json ---
                provenance_payload: Dict[str, Any] = {
                    "arm_id": arm_id,
                    "frozen_split_path": str(self.runner.split_manifest_path),
                    "frozen_split_sha256": FROZEN_SPLIT_MANIFEST_SHA256,
                    "frozen_features_parquet_path": str(self.runner.features_parquet_path),
                    "frozen_features_parquet_sha256": FROZEN_FEATURES_PARQUET_SHA256,
                    "train_n": len(X_train),
                    "validation_n": len(X_val),
                    "train_record_id_digest": compute_sequence_digest(meta_train["record_id"]),
                    "validation_record_id_digest": compute_sequence_digest(meta_val["record_id"]),
                    "train_wtfa_a_digest": compute_sequence_digest([f"{float(x):.6f}" for x in w_train]),
                    "validation_wtfa_a_digest": compute_sequence_digest([f"{float(x):.6f}" for x in w_val]),
                    "predictor_names": active_cols,
                    "categorical_predictor_names": active_cats,
                    "numerical_predictor_names": active_nums,
                    "scientific_execution_base_commit": SCIENTIFIC_EXECUTION_BASE_COMMIT,
                    "release_harness_commit": harness_commit,
                }
                p2 = arm_dir / "input_provenance.json"
                write_json_atomic(p2, provenance_payload)

                # --- Artifact 3: weight_audit.json ---
                weight_audit_payload = self.runner.audit_arm_weights(arm_id)
                p3 = arm_dir / "weight_audit.json"
                write_json_atomic(p3, weight_audit_payload)

                # --- Artifact 4: initial_weighted_dphi.json ---
                p4 = arm_dir / "initial_weighted_dphi.json"
                write_json_atomic(p4, exec_res["initial_dphi"])

                # --- Artifact 5: final_weighted_dphi.json ---
                p5 = arm_dir / "final_weighted_dphi.json"
                write_json_atomic(p5, exec_res["final_dphi"])

                # --- Artifact 6: weighted_changed_dict.json ---
                p6 = arm_dir / "weighted_changed_dict.json"
                write_json_atomic(p6, exec_res["changed_dict"])

                # --- Artifact 7: weighted_transform_trace.json ---
                trace_obj = exec_res["trace"]
                trace_dict = trace_obj.to_dict() if hasattr(trace_obj, "to_dict") else dict(trace_obj)
                steps = trace_dict.get("steps", [])
                accepted_steps = [
                    s for s in steps
                    if s.get("accepted_transformation") is not None and s.get("accepted_transformation") != "None"
                ]
                trace_dict["total_attempted_steps"] = len(steps)
                trace_dict["total_accepted_transforms"] = len(accepted_steps)
                trace_dict["selected_feature_sequence"] = [s.get("selected_feature") for s in steps]
                trace_dict["accepted_transformations"] = [
                    {
                        "iteration": s.get("iteration"),
                        "selected_feature": s.get("selected_feature"),
                        "accepted_transformation": s.get("accepted_transformation"),
                        "numerical_exponent": s.get("numerical_exponent"),
                        "categorical_merge_mapping": s.get("categorical_merge_mapping"),
                    }
                    for s in accepted_steps
                ]
                p7 = arm_dir / "weighted_transform_trace.json"
                write_json_atomic(p7, trace_dict)

                # --- Artifact 8: validation_metrics_baseline.json ---
                p8 = arm_dir / "validation_metrics_baseline.json"
                write_json_atomic(p8, exec_res["val_eval_base"])

                # --- Artifact 9: validation_metrics_weighted_fairbias.json ---
                p9 = arm_dir / "validation_metrics_weighted_fairbias.json"
                write_json_atomic(p9, exec_res["val_eval_fb"])

                # --- Artifact 10: validation_group_metrics_baseline.csv ---
                df_base_grp = pd.DataFrame(exec_res["val_eval_base"].get("group_metrics", []))
                p10 = arm_dir / "validation_group_metrics_baseline.csv"
                write_csv_atomic(df_base_grp, p10, force=True)

                # --- Artifact 11: validation_group_metrics_weighted_fairbias.csv ---
                df_fb_grp = pd.DataFrame(exec_res["val_eval_fb"].get("group_metrics", []))
                p11 = arm_dir / "validation_group_metrics_weighted_fairbias.csv"
                write_csv_atomic(df_fb_grp, p11, force=True)

                # --- Artifact 12: validation_comparison.json ---
                val_comp = copy.deepcopy(exec_res["val_comparison"])
                base_grp = exec_res["val_eval_base"].get("group_metrics", [])
                fb_grp = exec_res["val_eval_fb"].get("group_metrics", [])

                base_pred_pos = sum(g.get("predicted_positive_count", 0) for g in base_grp)
                fb_pred_pos = sum(g.get("predicted_positive_count", 0) for g in fb_grp)
                total_n = len(X_val)
                base_sel_rate = float(base_pred_pos / total_n) if total_n > 0 else None
                fb_sel_rate = float(fb_pred_pos / total_n) if total_n > 0 else None

                val_comp["baseline_predicted_positive_count"] = base_pred_pos
                val_comp["weighted_fairbias_predicted_positive_count"] = fb_pred_pos
                val_comp["predicted_positive_count_delta"] = int(fb_pred_pos - base_pred_pos)
                val_comp["baseline_selection_rate"] = base_sel_rate
                val_comp["weighted_fairbias_selection_rate"] = fb_sel_rate
                val_comp["selection_rate_delta"] = (
                    float(fb_sel_rate - base_sel_rate)
                    if (fb_sel_rate is not None and base_sel_rate is not None)
                    else None
                )
                val_comp["d4_reference"] = d4_ref

                p12 = arm_dir / "validation_comparison.json"
                write_json_atomic(p12, val_comp)

                # Verify all 12 artifacts exist and hash them
                arm_hashes: Dict[str, str] = {}
                for art_name in REQUIRED_PER_ARM_ARTIFACTS:
                    f_path = arm_dir / art_name
                    if not f_path.is_file():
                        raise FileNotFoundError(f"Missing required artifact {art_name} for arm {arm_id}")
                    h = compute_sha256(f_path)
                    arm_hashes[art_name] = h
                    rel_k = f"{arm_id}/{art_name}"
                    all_artifact_hashes[rel_k] = h

                arm_artifact_hashes[arm_id] = arm_hashes

        except Exception as exc:
            # Atomic transition to FAILED; record error and fail closed
            self.write_release_state(
                "FAILED",
                error_type=exc.__class__.__name__,
                error_message=str(exc),
            )
            raise

        # 4. Verify all four arms succeeded
        missing_arms = [a for a in FROZEN_D5_ARMS if a not in arm_results]
        if missing_arms:
            self.write_release_state(
                "FAILED",
                error_type="IncompleteReleaseError",
                error_message=f"Missing completed arms: {missing_arms}",
            )
            raise RuntimeError(f"Release incomplete: arms {missing_arms} were not completed.")

        # 5. Write top-level release manifest: d5_weighted_release_manifest.json
        manifest_payload: Dict[str, Any] = {
            "gate": "D5.1c execution protocol",
            "release_id": self.release_id,
            "status": "COMPLETE",
            "timestamp_utc": utc_timestamp(),
            "scientific_execution_base_commit": SCIENTIFIC_EXECUTION_BASE_COMMIT,
            "release_harness_commit": harness_commit,
            "scientific_base_is_ancestor": pre_info["scientific_base_is_ancestor"],
            "scientific_code_diff_clean": pre_info["scientific_code_diff_clean"],
            "secondary_analysis": True,
            "primary_test_replacement": False,
            "outcome": "MEDDL12M_A",
            "regime": "pooled_2022_2024_train_validation",
            "survey_weight_extension": "geometry_only",
            "classifier_weighted": False,
            "evaluation_weighted": False,
            "complex_survey_inference": False,
            "random_seed": PRIMARY_D5_RANDOM_SEED,
            "threshold": 0.5,
            "validation_used_for_mitigation": False,
            "test_partition_requested": False,
            "test_cohort_materialized": False,
            "test_evaluated": False,
            "arms": list(FROZEN_D5_ARMS.keys()),
            "artifacts": {
                rel_path: {"sha256": h} for rel_path, h in sorted(all_artifact_hashes.items())
            },
        }

        manifest_path = self.release_dir / "d5_weighted_release_manifest.json"
        write_json_atomic(manifest_path, manifest_payload)
        manifest_sha = compute_sha256(manifest_path)

        # 6. Update release state to COMPLETE
        self.write_release_state("COMPLETE", extra={"manifest_sha256": manifest_sha})

        return manifest_payload
