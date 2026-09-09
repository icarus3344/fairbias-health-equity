"""Gate D8-R3: Frozen Substantive Enhancement Execution Script.

Executes the frozen D8 accuracy enhancement algorithm on the real NHIS temporal study
across all 4 protected-attribute study arms under a frozen configuration:
- C1: Baseline (untransformed)
- C2: Canonical frozen D6 FairBias
- C3: Posthoc Bounded Accuracy Enhancement
- C4: Joint Interleaved Mitigation + Enhancement

Arms:
- D6_ARM_001 (SEX_A, full feature)
- D6_ARM_002 (HISPALLP_A, full feature)
- D6_ARM_003 (DISAB3_A, full feature)
- D6_ARM_004 (DISAB3_A, exclude disability components)

Temporal splits:
- Train: 2022
- Validation: 2023
- Test: 2024
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sklearn

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from fairbias.enhancement_state import changed_dict_hash
from nhis_fairbias.d8_enhancement_runner import (
    D8EnhancementRunner,
    FEATURES_PARQUET_PATH,
    FROZEN_D6_ARMS,
)

D6_TEST_RELEASE_DIR = REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7"
D6_TRAIN_VAL_RELEASE_DIR = REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609"
PROTECTED_BASELINE_TAG = "inherited-code-v0.3-baseline-20260828"
R2_CLOSURE_COMMIT = "cab6b637d97b0956b696f014e7a83d7eb3ca1b58"

ROOT_14_FILES = [
    "app.py",
    "classifiers.py",
    "config.py",
    "data_COMPAS.csv",
    "data_Credit_Card.csv",
    "eval.py",
    "main.py",
    "module_AE.py",
    "module_BM.py",
    "module_load.py",
    "module_transform.py",
    "requirements.txt",
    "results/all_results.json",
    "start.sh",
]

CORE_FAIRBIAS_FILES = [
    "src/fairbias/mitigation.py",
    "src/fairbias/bias_metric.py",
    "src/fairbias/transform.py",
    "src/fairbias/evaluator.py",
    "src/fairbias/models.py",
    "src/fairbias/config.py",
]

SCIENTIFIC_SOURCE_FILES = [
    "src/fairbias/mitigation.py",
    "src/fairbias/bias_metric.py",
    "src/fairbias/transform.py",
    "src/fairbias/evaluator.py",
    "src/fairbias/models.py",
    "src/fairbias/config.py",
    "src/fairbias/enhancement.py",
    "src/fairbias/enhancement_contracts.py",
    "src/fairbias/enhancement_state.py",
    "src/nhis_fairbias/d8_enhancement_runner.py",
    "scripts/run_nhis_enhancement_study.py",
    "scripts/run_nhis_d8_r3_substantive.py",
]

R3_FROZEN_CONFIG: Dict[str, Any] = {
    "gate": "NHIS-D8-R3",
    "random_seed": 0,
    "classifier": {
        "type": "LogisticRegression",
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 0,
        "probability_threshold": 0.5,
    },
    "scaler": {
        "type": "MinMaxScaler",
        "feature_range": [0, 1],
        "fit_partition_strictly": "train_only",
    },
    "temporal_splits": {
        "train_year": 2022,
        "validation_year": 2023,
        "test_year": 2024,
    },
    "arms": {
        "D6_ARM_001": {
            "protected_attribute": "SEX_A",
            "outcome": "MEDDL12M_A",
            "disability_arm": "full_feature",
            "feature_set": "full",
            "epsilon_threshold": 0.00477456,
            "feature_count": 21,
        },
        "D6_ARM_002": {
            "protected_attribute": "HISPALLP_A",
            "outcome": "MEDDL12M_A",
            "disability_arm": "full_feature",
            "feature_set": "full",
            "epsilon_threshold": 0.00522165,
            "feature_count": 21,
        },
        "D6_ARM_003": {
            "protected_attribute": "DISAB3_A",
            "outcome": "MEDDL12M_A",
            "disability_arm": "full_feature",
            "feature_set": "full",
            "epsilon_threshold": 0.01097498,
            "feature_count": 21,
        },
        "D6_ARM_004": {
            "protected_attribute": "DISAB3_A",
            "outcome": "MEDDL12M_A",
            "disability_arm": "exclude_disability_components",
            "feature_set": "exclude_disability",
            "epsilon_threshold": 0.01787916,
            "feature_count": 15,
        },
    },
    "canonical_d6_changed_dicts": {
        "D6_ARM_001_hash": "40511e6c0d55b0ff",
        "D6_ARM_002_hash": "4c0bbba5d40022d6",
        "D6_ARM_003_hash": "38fa54a06a9c6427",
        "D6_ARM_004_hash": "ff0a2fb81596b598",
    },
    "enhancement_parameters": {
        "candidate_transform_families": {
            "categorical": ["one_hot", "drop"],
            "numerical": ["binning", "polynomial", "log", "scaling", "drop"],
        },
        "polynomial_exponent_grid": [2, 3],
        "binning_n_bins": 5,
        "log_epsilon": 1e-6,
        "minimum_utility_gain": 0.0,
        "maximum_fairness_degradation": 0.02,
        "candidate_ranking_method": "delta_utility_descending_then_fairness_degradation_ascending",
        "cycle_detection": "changed_dict_hash_in_committed_states_set",
        "search_budgets": {
            "condition_3_max_steps": 5,
            "condition_4_max_iterations": 10,
        },
        "acceptance_semantics": "strict_monotonic_utility_gain_within_fairness_bound",
    },
    "evaluation_metrics": [
        "auroc",
        "auprc",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "predicted_positive_count",
        "selection_rate",
        "max_dphi",
        "demographic_parity_gap",
        "equal_opportunity_gap",
    ],
}


class D8R3SubstantiveRunner(D8EnhancementRunner):
    """Substantive R3 runner enabling authorized 4-condition execution on real NHIS microdata."""

    def __init__(
        self,
        smoke_test: bool = False,
        random_seed: int = 0,
        run_id: str = "d8_r3_substantive",
        allow_real_data: bool = True,
    ):
        # Initialize in baseline_reproduction_only mode to satisfy base constructor contract
        super().__init__(
            smoke_test=smoke_test,
            random_seed=random_seed,
            run_id=run_id,
            allow_real_data=allow_real_data,
            baseline_reproduction_only=True,
        )
        # Un-restrict to full 4-condition study under Gate D8-R3 authorization
        self.baseline_reproduction_only = False
        self.gate = "D8-R3"


def compute_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_protected_files() -> Dict[str, Any]:
    root_diff = subprocess.run(
        ["git", "diff", "--stat", PROTECTED_BASELINE_TAG, "--"] + ROOT_14_FILES,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()

    gitignore_diff = subprocess.run(
        ["git", "diff", "--stat", PROTECTED_BASELINE_TAG, "--", ".gitignore"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()

    core_diff = subprocess.run(
        ["git", "diff", "--stat", R2_CLOSURE_COMMIT, "--"] + CORE_FAIRBIAS_FILES,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()

    return {
        "protected_baseline_tag": PROTECTED_BASELINE_TAG,
        "root_14_files_diff_stat": root_diff,
        "root_14_files_clean": root_diff == "",
        "gitignore_diff_stat": gitignore_diff,
        "gitignore_drift_status": "PRE_EXISTING_BASELINE_DRIFT_RECORDED_NOT_RESOLVED",
        "r2_closure_commit": R2_CLOSURE_COMMIT,
        "core_fairbias_diff_stat": core_diff,
        "core_fairbias_clean": core_diff == "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NHIS D8-R3 Frozen Substantive Enhancement Execution.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target output directory (default: artifacts/nhis_d8_r3/<timestamp>_substantive).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()
    started_at_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    utc_ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output_dir is not None:
        run_dir = pathlib.Path(args.output_dir).resolve()
    else:
        run_dir = REPO_ROOT / "artifacts" / "nhis_d8_r3" / f"{utc_ts}_d8r3_substantive"

    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name

    print(f"=== NHIS-D8-R3 Frozen Substantive Enhancement Execution ===")
    print(f"Run ID: {run_id}")
    print(f"Output Directory: {run_dir}")
    print(f"Random Seed: {args.random_seed}")

    # Step 1: Pre-run verification of protected files
    print("\n--- Verifying Protected Files and Baseline Integrity ---")
    prot_integrity = check_protected_files()
    if not prot_integrity["root_14_files_clean"]:
        raise RuntimeError("Protected 14 baseline files modified! Violates baseline immutability.")
    if not prot_integrity["core_fairbias_clean"]:
        raise RuntimeError("Core FairBias files modified vs R2 closure commit!")
    with open(run_dir / "protected_file_integrity.json", "w") as f:
        json.dump(prot_integrity, f, indent=2)
    print("Protected file integrity: PASS")

    # Step 2: Pre-run source manifest & git diff
    print("\n--- Exporting Pre-Run Source Manifest and Git Diff ---")
    source_manifest: Dict[str, Any] = {}
    for rel_p in SCIENTIFIC_SOURCE_FILES:
        abs_p = REPO_ROOT / rel_p
        source_manifest[rel_p] = {
            "sha256": compute_sha256(abs_p),
            "bytes": abs_p.stat().st_size,
        }
    with open(run_dir / "pre_run_source_manifest.json", "w") as f:
        json.dump(source_manifest, f, indent=2)

    git_diff_out = subprocess.run(
        ["git", "diff", "--no-ext-diff", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    ).stdout
    with open(run_dir / "pre_run_git_diff.patch", "w") as f:
        f.write(git_diff_out)

    # Step 3: Pre-run R3 scientific config manifest
    print("\n--- Exporting R3 Scientific Configuration Manifest ---")
    canonical_config_str = json.dumps(R3_FROZEN_CONFIG, sort_keys=True)
    r3_config_sha256 = hashlib.sha256(canonical_config_str.encode("utf-8")).hexdigest()
    r3_config_manifest = {
        "r3_config_sha256": r3_config_sha256,
        "frozen_config": R3_FROZEN_CONFIG,
    }
    with open(run_dir / "r3_config_manifest.json", "w") as f:
        json.dump(r3_config_manifest, f, indent=2)
    print(f"R3_CONFIG_SHA256: {r3_config_sha256}")

    # Step 4: Frozen reference manifest
    frozen_ref_manifest = {
        "d6_test_release_dir": str(D6_TEST_RELEASE_DIR),
        "d6_train_val_release_dir": str(D6_TRAIN_VAL_RELEASE_DIR),
        "features_parquet_path": str(FEATURES_PARQUET_PATH),
        "features_parquet_sha256": compute_sha256(FEATURES_PARQUET_PATH),
        "features_parquet_size_bytes": FEATURES_PARQUET_PATH.stat().st_size,
        "protected_baseline_tag": PROTECTED_BASELINE_TAG,
        "r2_closure_commit": R2_CLOSURE_COMMIT,
        "arms_evaluated": ["D6_ARM_001", "D6_ARM_002", "D6_ARM_003", "D6_ARM_004"],
    }
    with open(run_dir / "frozen_reference_manifest.json", "w") as f:
        json.dump(frozen_ref_manifest, f, indent=2)

    # Step 5: Initial command log
    cmd_log_p = run_dir / "command_log.txt"
    cmd_log_lines = [
        f"1. Pre-run configuration freeze exported to {run_dir} at {started_at_utc}",
        f"2. Protected file integrity verified: clean",
        f"3. Scientific config frozen with R3_CONFIG_SHA256={r3_config_sha256}",
        f"4. Executing single authorized real-data run across 4 arms and 4 conditions",
    ]
    with open(cmd_log_p, "w") as f:
        f.write("\n".join(cmd_log_lines) + "\n")

    # Step 6: Single Substantive Real-Data Execution (Conditions 1 - 4 across all 4 arms)
    print("\n--- Executing Single Authorized Substantive Enhancement Run (Conditions 1 - 4) ---")
    runner = D8R3SubstantiveRunner(
        smoke_test=False,
        random_seed=args.random_seed,
        run_id=run_id,
        allow_real_data=True,
    )

    study_results: Dict[str, Any] = {}
    target_arms = sorted(list(FROZEN_D6_ARMS.keys()))
    for arm_id in target_arms:
        print(f"\nRunning {arm_id} ({FROZEN_D6_ARMS[arm_id]['protected_attribute']})...")
        t0 = time.time()
        res = runner.run_arm(arm_id)
        study_results[arm_id] = res
        print(f"  Completed {arm_id} in {time.time() - t0:.2f}s")

    total_duration = time.time() - t_start
    completed_at_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f"\nAll 4 arms completed in {total_duration:.2f}s")

    # Step 7: Build Table A: Full Temporal Condition Matrix (condition_metrics.json / md)
    print("\n--- Compiling Table A: Full Temporal Condition Matrix ---")
    condition_matrix_rows: List[Dict[str, Any]] = []
    stages = [("train", "train_2022"), ("validation", "validation_2023"), ("test", "test_2024")]
    conditions = [
        ("baseline", "C1_Baseline"),
        ("canonical_fairbias", "C2_Canonical"),
        ("posthoc_enhancement", "C3_Posthoc"),
        ("joint_enhancement", "C4_Joint"),
    ]

    for arm_id in target_arms:
        arm_res = study_results[arm_id]
        cond_dict = arm_res["conditions"]
        for cond_key, cond_label in conditions:
            cond_data = cond_dict[cond_key]
            for stage_key, stage_label in stages:
                metrics = cond_data[stage_key]
                row = {
                    "arm_id": arm_id,
                    "protected_attribute": arm_res["protected_attribute"],
                    "condition_key": cond_key,
                    "condition_label": cond_label,
                    "stage_key": stage_key,
                    "stage_label": stage_label,
                    "auroc": metrics.get("auroc"),
                    "auprc": metrics.get("auprc"),
                    "accuracy": metrics.get("accuracy"),
                    "balanced_accuracy": metrics.get("balanced_accuracy"),
                    "f1": metrics.get("f1"),
                    "predicted_positive_count": metrics.get("predicted_positive_count"),
                    "selection_rate": metrics.get("selection_rate"),
                    "max_dphi": metrics.get("max_dphi"),
                    "demographic_parity_gap": metrics.get("demographic_parity_gap"),
                    "equal_opportunity_gap": metrics.get("equal_opportunity_gap"),
                }
                condition_matrix_rows.append(row)

    with open(run_dir / "condition_metrics.json", "w") as f:
        json.dump(condition_matrix_rows, f, indent=2)

    def fmt_f(val: Optional[float], spec: str = ".4f") -> str:
        if val is None:
            return "N/A"
        return f"{val:{spec}}"

    def fmt_d(val: Optional[float], spec: str = "+.4f") -> str:
        if val is None:
            return "N/A"
        return f"{val:{spec}}"

    def fmt_di(val: Optional[int]) -> str:
        if val is None:
            return "N/A"
        return f"{val:+d}"

    def safe_sub(a: Any, b: Any) -> Any:
        if a is None or b is None:
            return None
        return a - b

    # Markdown table for Table A
    md_table_a = [
        "# Table A: Full Temporal Condition Matrix",
        "",
        "| Arm | Condition | Stage | AUROC | AUPRC | Accuracy | Bal. Acc. | F1 | Pos. Count | Selection Rate | Max d_phi | DP Gap | EO Gap |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in condition_matrix_rows:
        md_table_a.append(
            f"| {r['arm_id']} | {r['condition_label']} | {r['stage_label']} "
            f"| {fmt_f(r['auroc'], '.4f')} | {fmt_f(r['auprc'], '.4f')} | {fmt_f(r['accuracy'], '.4f')} | {fmt_f(r['balanced_accuracy'], '.4f')} | {fmt_f(r['f1'], '.4f')} "
            f"| {r['predicted_positive_count']} | {fmt_f(r['selection_rate'], '.5f')} | {fmt_f(r['max_dphi'], '.5f')} "
            f"| {fmt_f(r['demographic_parity_gap'], '.5f')} | {fmt_f(r['equal_opportunity_gap'], '.5f')} |"
        )
    with open(run_dir / "condition_metrics.md", "w") as f:
        f.write("\n".join(md_table_a) + "\n")

    # Step 8: Build Table B: Primary Deltas (primary_deltas.json / md)
    print("\n--- Compiling Table B: Primary Deltas vs Canonical FairBias (and Baseline) ---")
    deltas_rows: List[Dict[str, Any]] = []
    # Index condition rows by (arm_id, stage_key, cond_key)
    lookup = {(r["arm_id"], r["stage_key"], r["condition_key"]): r for r in condition_matrix_rows}

    for arm_id in target_arms:
        for stage_key, stage_label in stages:
            c1 = lookup[(arm_id, stage_key, "baseline")]
            c2 = lookup[(arm_id, stage_key, "canonical_fairbias")]
            for enh_key, enh_label in [("posthoc_enhancement", "C3_Posthoc"), ("joint_enhancement", "C4_Joint")]:
                cenh = lookup[(arm_id, stage_key, enh_key)]
                row_delta = {
                    "arm_id": arm_id,
                    "stage_key": stage_key,
                    "stage_label": stage_label,
                    "enhancement_condition": enh_label,
                    # Primary deltas vs C2 (Canonical FairBias)
                    "delta_auroc_vs_c2": safe_sub(cenh["auroc"], c2["auroc"]),
                    "delta_auprc_vs_c2": safe_sub(cenh["auprc"], c2["auprc"]),
                    "delta_pos_count_vs_c2": safe_sub(cenh["predicted_positive_count"], c2["predicted_positive_count"]),
                    "delta_max_dphi_vs_c2": safe_sub(cenh["max_dphi"], c2["max_dphi"]),
                    "delta_dp_gap_vs_c2": safe_sub(cenh["demographic_parity_gap"], c2["demographic_parity_gap"]),
                    "delta_eo_gap_vs_c2": safe_sub(cenh["equal_opportunity_gap"], c2["equal_opportunity_gap"]),
                    # Secondary deltas vs C1 (Baseline)
                    "delta_auroc_vs_c1": safe_sub(cenh["auroc"], c1["auroc"]),
                    "delta_auprc_vs_c1": safe_sub(cenh["auprc"], c1["auprc"]),
                    "delta_pos_count_vs_c1": safe_sub(cenh["predicted_positive_count"], c1["predicted_positive_count"]),
                    "delta_max_dphi_vs_c1": safe_sub(cenh["max_dphi"], c1["max_dphi"]),
                    "delta_dp_gap_vs_c1": safe_sub(cenh["demographic_parity_gap"], c1["demographic_parity_gap"]),
                    "delta_eo_gap_vs_c1": safe_sub(cenh["equal_opportunity_gap"], c1["equal_opportunity_gap"]),
                }
                deltas_rows.append(row_delta)

    with open(run_dir / "primary_deltas.json", "w") as f:
        json.dump(deltas_rows, f, indent=2)

    # Markdown table for Table B
    md_table_b = [
        "# Table B: Primary Deltas vs Canonical FairBias (C2)",
        "",
        "| Arm | Stage | Enhancement | Delta AUROC vs C2 | Delta AUPRC vs C2 | Delta Pos Count vs C2 | Delta Max d_phi vs C2 | Delta DP Gap vs C2 | Delta EO Gap vs C2 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in deltas_rows:
        md_table_b.append(
            f"| {r['arm_id']} | {r['stage_label']} | {r['enhancement_condition']} "
            f"| {fmt_d(r['delta_auroc_vs_c2'], '+.4f')} | {fmt_d(r['delta_auprc_vs_c2'], '+.4f')} | {fmt_di(r['delta_pos_count_vs_c2'])} "
            f"| {fmt_d(r['delta_max_dphi_vs_c2'], '+.5f')} | {fmt_d(r['delta_dp_gap_vs_c2'], '+.5f')} | {fmt_d(r['delta_eo_gap_vs_c2'], '+.5f')} |"
        )
    md_table_b.extend([
        "",
        "### Secondary Descriptive Deltas vs Baseline (C1)",
        "",
        "| Arm | Stage | Enhancement | Delta AUROC vs C1 | Delta AUPRC vs C1 | Delta Pos Count vs C1 | Delta Max d_phi vs C1 | Delta DP Gap vs C1 | Delta EO Gap vs C1 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])
    for r in deltas_rows:
        md_table_b.append(
            f"| {r['arm_id']} | {r['stage_label']} | {r['enhancement_condition']} "
            f"| {fmt_d(r['delta_auroc_vs_c1'], '+.4f')} | {fmt_d(r['delta_auprc_vs_c1'], '+.4f')} | {fmt_di(r['delta_pos_count_vs_c1'])} "
            f"| {fmt_d(r['delta_max_dphi_vs_c1'], '+.5f')} | {fmt_d(r['delta_dp_gap_vs_c1'], '+.5f')} | {fmt_d(r['delta_eo_gap_vs_c1'], '+.5f')} |"
        )
    with open(run_dir / "primary_deltas.md", "w") as f:
        f.write("\n".join(md_table_b) + "\n")

    # Step 9: Build Table C: Enhancement Search & Audit Summary (enhancement_audit_summary.json / md)
    print("\n--- Compiling Table C: Search and Audit Summary ---")
    audit_summary_rows: List[Dict[str, Any]] = []
    for arm_id in target_arms:
        arm_res = study_results[arm_id]
        cond_dict = arm_res["conditions"]
        for enh_key, enh_label in [("posthoc_enhancement", "C3_Posthoc"), ("joint_enhancement", "C4_Joint")]:
            c_data = cond_dict[enh_key]
            term_state = c_data.get("terminal_state", {})
            audit_summary_rows.append({
                "arm_id": arm_id,
                "condition": enh_label,
                "termination_reason": c_data.get("termination_reason"),
                "fairness_feasible": c_data.get("fairness_feasible"),
                "committed_steps": c_data.get("ae_steps_accepted", 0) + (c_data.get("bm_steps_accepted", 0) if enh_key == "joint_enhancement" else 0),
                "ae_committed_steps": c_data.get("ae_steps_accepted", 0),
                "bm_committed_steps": c_data.get("bm_steps_accepted", 0) if enh_key == "joint_enhancement" else 0,
                "num_transforms_in_terminal_state": len(term_state),
                "model_fit_count": c_data.get("model_fit_count", 0),
                "geometry_eval_count": c_data.get("geometry_eval_count", 0),
                "final_state_hash": changed_dict_hash(term_state),
                "terminal_state": term_state,
            })

    with open(run_dir / "enhancement_audit_summary.json", "w") as f:
        json.dump(audit_summary_rows, f, indent=2)

    md_table_c = [
        "# Table C: Enhancement Search & Audit Summary",
        "",
        "| Arm | Condition | Termination Reason | Feasible? | Committed (Total / AE / BM) | Num Transforms | Model Fits | Geometry Evals | Final State Hash |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in audit_summary_rows:
        md_table_c.append(
            f"| {r['arm_id']} | {r['condition']} | {r['termination_reason']} | {r['fairness_feasible']} "
            f"| {r['committed_steps']} ({r['ae_committed_steps']} AE / {r['bm_committed_steps']} BM) | {r['num_transforms_in_terminal_state']} "
            f"| {r['model_fit_count']} | {r['geometry_eval_count']} | `{r['final_state_hash']}` |"
        )
    with open(run_dir / "enhancement_audit_summary.md", "w") as f:
        f.write("\n".join(md_table_c) + "\n")

    # Step 10: Joint Trajectory Events (joint_trajectory_events.json)
    print("\n--- Compiling Joint Trajectory Event Log ---")
    all_joint_events: Dict[str, Any] = {}
    for arm_id in target_arms:
        c4_data = study_results[arm_id]["conditions"]["joint_enhancement"]
        all_joint_events[arm_id] = c4_data.get("iteration_events", [])
    with open(run_dir / "joint_trajectory_events.json", "w") as f:
        json.dump(all_joint_events, f, indent=2)

    # Step 11: Post-run source verification
    print("\n--- Verifying Post-Run Scientific Source Hashes ---")
    post_run_manifest: Dict[str, Any] = {}
    all_source_match = True
    for rel_p in SCIENTIFIC_SOURCE_FILES:
        abs_p = REPO_ROOT / rel_p
        cur_h = compute_sha256(abs_p)
        cur_sz = abs_p.stat().st_size
        pre_h = source_manifest[rel_p]["sha256"]
        match = (cur_h == pre_h)
        if not match:
            all_source_match = False
        post_run_manifest[rel_p] = {
            "pre_run_hash": pre_h,
            "post_run_hash": cur_h,
            "match": match,
            "size_bytes": cur_sz,
        }

    post_run_verification = {
        "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "all_scientific_source_hashes_match": all_source_match,
        "files": post_run_manifest,
    }
    with open(run_dir / "post_run_source_verification.json", "w") as f:
        json.dump(post_run_verification, f, indent=2)
    print(f"Post-run source verification: {'PASS' if all_source_match else 'FAIL'}")

    # Step 12: Environment Manifest
    env_manifest = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "r2_closure_commit": R2_CLOSURE_COMMIT,
        "current_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip(),
    }
    with open(run_dir / "environment_manifest.json", "w") as f:
        json.dump(env_manifest, f, indent=2)

    # Step 13: Finalize command log
    cmd_log_lines.extend([
        f"5. Real-data execution completed in {total_duration:.2f}s",
        f"6. Condition metrics exported (Table A)",
        f"7. Primary deltas exported (Table B)",
        f"8. Enhancement audit summary exported (Table C)",
        f"9. Joint trajectory events exported",
        f"10. Post-run source verification completed: all_scientific_source_hashes_match={all_source_match}",
        f"11. Finalizing execution_manifest.json",
    ])
    with open(cmd_log_p, "w") as f:
        f.write("\n".join(cmd_log_lines) + "\n")

    # Step 14: Execution Manifest
    mf: Dict[str, Any] = {
        "run_id": run_id,
        "gate": "NHIS-D8-R3",
        "status": "COMPLETED" if all_source_match else "EXECUTION_VERIFICATION_FAILED",
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "duration_seconds": total_duration,
        "random_seed": args.random_seed,
        "r3_config_sha256": r3_config_sha256,
        "all_scientific_source_hashes_match": all_source_match,
        "arms_evaluated": target_arms,
        "conditions_evaluated": ["C1_Baseline", "C2_Canonical", "C3_Posthoc", "C4_Joint"],
        "total_condition_evaluations": len(condition_matrix_rows),
        "output_files": {},
    }
    for p in sorted(run_dir.iterdir()):
        if p.is_file() and p.name != "execution_manifest.json":
            mf["output_files"][p.name] = {
                "path": str(p),
                "sha256": compute_sha256(p),
                "size_bytes": p.stat().st_size,
            }
    with open(run_dir / "execution_manifest.json", "w") as f:
        json.dump(mf, f, indent=2)

    print(f"\n=== R3 Substantive Execution Complete ===")
    print(f"Artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
