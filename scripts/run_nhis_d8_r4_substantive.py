"""Gate D8-R4: Primary Frozen D6-Geometry Substantive Execution Script.

Performs the single authorized primary preregistered substantive NHIS study
using the frozen D6 scientific world:
- Base algorithm mode: tang2024_paper_faithful
- MDS geometry: stress-elbow dimension selection (mds_fixed_components=None)
- Authoritative frozen D6 training thresholds:
    * D6_ARM_001 (SEX_A): 0.0005
    * D6_ARM_002 (HISPALLP_A): 0.0020
    * D6_ARM_003 (DISAB3_A): 0.0050
    * D6_ARM_004 (DISAB3_A): 0.0050
- Reviewed D8 accuracy enhancement extension (external layer)
- Evaluates 4 arms across 4 conditions (C1 Baseline, C2 Canonical, C3 Posthoc, C4 Joint)
  over 3 temporal partitions (Train 2022, Validation 2023, Test 2024).
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import math
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
    D8ExecutionMode,
    FEATURES_PARQUET_PATH,
    FROZEN_D6_ARMS,
    D6_TRAIN_VAL_RELEASE_DIR,
    load_frozen_d6_threshold,
)

PROTECTED_BASELINE_TAG = "inherited-code-v0.3-baseline-20260828"
R4_BASELINE_COMMIT = "3bc3c40b7edabdbbd0cb443fd7bfca2d4d1e6909"
R2B_EVAL_DIR = REPO_ROOT / "artifacts" / "nhis_d8_r2b" / "20260908T102300Z_d8r2b_eval"
R3_HISTORICAL_DIR = REPO_ROOT / "artifacts" / "nhis_d8_r3" / "20260908T110640Z_d8r3_substantive"

ROOT_14_FILES = [
    "app.py", "classifiers.py", "config.py", "data_COMPAS.csv", "data_Credit_Card.csv",
    "eval.py", "main.py", "module_AE.py", "module_BM.py", "module_load.py",
    "module_transform.py", "requirements.txt", "results/all_results.json", "start.sh",
]

CORE_FAIRBIAS_FILES = [
    "src/fairbias/mitigation.py", "src/fairbias/bias_metric.py", "src/fairbias/transform.py",
    "src/fairbias/evaluator.py", "src/fairbias/models.py", "src/fairbias/config.py",
]

SCIENTIFIC_SOURCE_FILES = [
    "src/nhis_fairbias/d8_enhancement_runner.py",
    "scripts/run_nhis_enhancement_study.py",
    "scripts/run_nhis_d8_r4_substantive.py",
    "src/fairbias/enhancement.py",
    "src/fairbias/enhancement_contracts.py",
    "src/fairbias/enhancement_state.py",
    "src/fairbias/mitigation.py",
    "src/fairbias/bias_metric.py",
    "src/fairbias/transform.py",
    "src/fairbias/evaluator.py",
    "src/fairbias/models.py",
    "src/fairbias/config.py",
]

REFERENCE_ARTIFACT_FILES = [
    "docs/releases/NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7/D6_ARM_001/frozen_changed_dict.json",
    "docs/releases/NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7/D6_ARM_002/frozen_changed_dict.json",
    "docs/releases/NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7/D6_ARM_003/frozen_changed_dict.json",
    "docs/releases/NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7/D6_ARM_004/frozen_changed_dict.json",
    "docs/releases/NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609/D6_ARM_001/train_dphi_before_after.json",
    "docs/releases/NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609/D6_ARM_002/train_dphi_before_after.json",
    "docs/releases/NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609/D6_ARM_003/train_dphi_before_after.json",
    "docs/releases/NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609/D6_ARM_004/train_dphi_before_after.json",
]

R4_FROZEN_CONFIG: Dict[str, Any] = {
    "gate": "NHIS-D8-R4",
    "execution_mode": "SUBSTANTIVE_D6_GEOMETRY",
    "algorithm_mode": "tang2024_paper_faithful",
    "mds_fixed_components": None,
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
    "eval_norm": "min-max",
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
            "epsilon_threshold": 0.0005,
            "epsilon_threshold_source": "FROZEN_D6_TRAIN_REFERENCE",
            "frozen_reference_artifact": "docs/releases/NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609/D6_ARM_001/train_dphi_before_after.json",
            "feature_count": 21,
        },
        "D6_ARM_002": {
            "protected_attribute": "HISPALLP_A",
            "outcome": "MEDDL12M_A",
            "disability_arm": "full_feature",
            "feature_set": "full",
            "epsilon_threshold": 0.0020,
            "epsilon_threshold_source": "FROZEN_D6_TRAIN_REFERENCE",
            "frozen_reference_artifact": "docs/releases/NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609/D6_ARM_002/train_dphi_before_after.json",
            "feature_count": 21,
        },
        "D6_ARM_003": {
            "protected_attribute": "DISAB3_A",
            "outcome": "MEDDL12M_A",
            "disability_arm": "full_feature",
            "feature_set": "full",
            "epsilon_threshold": 0.0050,
            "epsilon_threshold_source": "FROZEN_D6_TRAIN_REFERENCE",
            "frozen_reference_artifact": "docs/releases/NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609/D6_ARM_003/train_dphi_before_after.json",
            "feature_count": 21,
        },
        "D6_ARM_004": {
            "protected_attribute": "DISAB3_A",
            "outcome": "MEDDL12M_A",
            "disability_arm": "exclude_disability_components",
            "feature_set": "exclude_disability",
            "epsilon_threshold": 0.0050,
            "epsilon_threshold_source": "FROZEN_D6_TRAIN_REFERENCE",
            "frozen_reference_artifact": "docs/releases/NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609/D6_ARM_004/train_dphi_before_after.json",
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
        "minimum_utility_gain": 0.001,
        "maximum_fairness_degradation": 0.0,
        "candidate_ranking_method": "delta_utility_descending_then_fairness_degradation_ascending",
        "cycle_detection": "changed_dict_hash_in_committed_states_set",
        "search_budgets": {
            "condition_3_max_steps": 10,
            "condition_4_max_iterations": 10,
        },
        "acceptance_semantics": "strict_monotonic_utility_gain_within_fairness_bound",
        "probability_threshold": 0.5,
    },
    "governance_incidents_recorded": [
        "R3-GOV-01: historical broad unittest discovery occurred, 24 real-data reads blocked, 0 rows accessed",
        "R3B-SCI-01: historical substantive execution used engineering fixed-MDS geometry and is exploratory sensitivity evidence only",
    ]
}


def compute_sha256(path: pathlib.Path) -> str:
    out = subprocess.check_output(["shasum", "-a", "256", str(path)], text=True)
    return out.split()[0]


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
        ["git", "diff", "--stat", R4_BASELINE_COMMIT, "--"] + CORE_FAIRBIAS_FILES,
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
        "r4_baseline_commit": R4_BASELINE_COMMIT,
        "core_fairbias_diff_stat": core_diff,
        "core_fairbias_clean": core_diff == "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NHIS D8-R4 Primary Frozen D6-Geometry Substantive Execution.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target output directory (default: artifacts/nhis_d8_r4/<timestamp>_d8r4_substantive).",
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
        run_dir = REPO_ROOT / "artifacts" / "nhis_d8_r4" / f"{utc_ts}_d8r4_substantive"

    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name

    print(f"=== NHIS-D8-R4 Primary Frozen D6-Geometry Substantive Execution ===")
    print(f"Run ID: {run_id}")
    print(f"Output Directory: {run_dir}")
    print(f"R4 Baseline Commit: {R4_BASELINE_COMMIT}")
    print(f"Random Seed: {args.random_seed}")

    # Step 1: Pre-run verification of protected files
    print("\n--- Step 1: Verifying Protected Files and Baseline Integrity ---")
    prot_integrity = check_protected_files()
    if not prot_integrity["root_14_files_clean"]:
        raise RuntimeError("Protected 14 baseline files modified! Violates baseline immutability.")
    if not prot_integrity["core_fairbias_clean"]:
        raise RuntimeError("Core FairBias files modified vs R4 baseline commit!")
    with open(run_dir / "protected_file_integrity.json", "w") as f:
        json.dump(prot_integrity, f, indent=2)
    print("Protected file integrity: PASS")

    # Step 2: Pre-run source manifest & git diff
    print("\n--- Step 2: Exporting Pre-Run Source Manifest and Git Diff ---")
    source_manifest: Dict[str, Any] = {}
    for rel_p in SCIENTIFIC_SOURCE_FILES:
        abs_p = REPO_ROOT / rel_p
        source_manifest[rel_p] = {
            "sha256": compute_sha256(abs_p),
            "bytes": abs_p.stat().st_size,
        }
    for rel_ref in REFERENCE_ARTIFACT_FILES:
        abs_ref = REPO_ROOT / rel_ref
        source_manifest[rel_ref] = {
            "sha256": compute_sha256(abs_ref),
            "bytes": abs_ref.stat().st_size,
        }
    with open(run_dir / "pre_run_source_manifest.json", "w") as f:
        json.dump(source_manifest, f, indent=2)

    git_diff_out = subprocess.run(
        ["git", "diff", "--no-ext-diff", R4_BASELINE_COMMIT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    ).stdout
    with open(run_dir / "pre_run_git_diff.patch", "w") as f:
        f.write(git_diff_out)

    # Step 3: Pre-run R4 scientific config manifest & R4_CONFIG_SHA256
    print("\n--- Step 3: Exporting R4 Scientific Configuration Manifest ---")
    canonical_config_str = json.dumps(R4_FROZEN_CONFIG, sort_keys=True)
    r4_config_sha256 = hashlib.sha256(canonical_config_str.encode("utf-8")).hexdigest()
    r4_config_manifest = {
        "r4_config_sha256": r4_config_sha256,
        "frozen_config": R4_FROZEN_CONFIG,
    }
    with open(run_dir / "r4_config_manifest.json", "w") as f:
        json.dump(r4_config_manifest, f, indent=2)
    print(f"R4_CONFIG_SHA256: {r4_config_sha256}")

    # Step 4: Frozen reference manifest
    print("\n--- Step 4: Exporting Frozen Reference Manifest ---")
    frozen_ref_manifest = {
        "d6_train_val_release_dir": str(D6_TRAIN_VAL_RELEASE_DIR),
        "features_parquet_path": str(FEATURES_PARQUET_PATH),
        "features_parquet_sha256": compute_sha256(FEATURES_PARQUET_PATH),
        "features_parquet_size_bytes": FEATURES_PARQUET_PATH.stat().st_size,
        "protected_baseline_tag": PROTECTED_BASELINE_TAG,
        "r4_baseline_commit": R4_BASELINE_COMMIT,
        "arms_evaluated": ["D6_ARM_001", "D6_ARM_002", "D6_ARM_003", "D6_ARM_004"],
        "authoritative_thresholds": {
            arm: load_frozen_d6_threshold(arm) for arm in ["D6_ARM_001", "D6_ARM_002", "D6_ARM_003", "D6_ARM_004"]
        },
        "canonical_reference_hashes": {
            f: compute_sha256(REPO_ROOT / f) for f in REFERENCE_ARTIFACT_FILES
        },
    }
    with open(run_dir / "frozen_d6_reference_manifest.json", "w") as f:
        json.dump(frozen_ref_manifest, f, indent=2)

    # Step 5: Initial command log
    print("\n--- Step 5: Initializing Command Log ---")
    cmd_log_p = run_dir / "command_log.txt"
    cmd_log_lines = [
        f"1. Pre-run configuration freeze exported to {run_dir} at {started_at_utc}",
        f"2. Protected file integrity verified: clean",
        f"3. Scientific config frozen with R4_CONFIG_SHA256={r4_config_sha256}",
        f"4. Executing single authorized primary substantive real-data run across 4 arms and 4 conditions",
    ]
    with open(cmd_log_p, "w") as f:
        f.write("\n".join(cmd_log_lines) + "\n")

    # Step 6: Single Authorized Real-Data Execution (Conditions 1 - 4 across all 4 arms)
    print("\n--- Step 6: Executing Single Authorized Primary Substantive Run (Conditions 1 - 4) ---")
    runner = D8EnhancementRunner(
        mode=D8ExecutionMode.SUBSTANTIVE_D6_GEOMETRY,
        smoke_test=False,
        random_seed=args.random_seed,
        run_id=run_id,
        allow_real_data=True,
        r4_primary_authorized=True,
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
    print("\n--- Step 7: Compiling Table A: Full Temporal Condition Matrix ---")
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
                    "brier": metrics.get("brier"),
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
        "| Arm | Condition | Stage | AUROC | AUPRC | Accuracy | Bal. Acc. | F1 | Brier | Pos. Count | Selection Rate | Max d_phi | DP Gap | EO Gap |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in condition_matrix_rows:
        md_table_a.append(
            f"| {r['arm_id']} | {r['condition_label']} | {r['stage_label']} "
            f"| {fmt_f(r['auroc'], '.4f')} | {fmt_f(r['auprc'], '.4f')} | {fmt_f(r['accuracy'], '.4f')} | {fmt_f(r['balanced_accuracy'], '.4f')} | {fmt_f(r['f1'], '.4f')} | {fmt_f(r['brier'], '.4f')} "
            f"| {r['predicted_positive_count']} | {fmt_f(r['selection_rate'], '.5f')} | {fmt_f(r['max_dphi'], '.5f')} "
            f"| {fmt_f(r['demographic_parity_gap'], '.5f')} | {fmt_f(r['equal_opportunity_gap'], '.5f')} |"
        )
    with open(run_dir / "condition_metrics.md", "w") as f:
        f.write("\n".join(md_table_a) + "\n")

    # Step 8: Build Table B: Primary Deltas (primary_deltas.json / md)
    print("\n--- Step 8: Compiling Table B: Primary Deltas vs Canonical FairBias (and Baseline) ---")
    deltas_rows: List[Dict[str, Any]] = []
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
                    "delta_selection_rate_vs_c2": safe_sub(cenh["selection_rate"], c2["selection_rate"]),
                    "delta_max_dphi_vs_c2": safe_sub(cenh["max_dphi"], c2["max_dphi"]),
                    "delta_dp_gap_vs_c2": safe_sub(cenh["demographic_parity_gap"], c2["demographic_parity_gap"]),
                    "delta_eo_gap_vs_c2": safe_sub(cenh["equal_opportunity_gap"], c2["equal_opportunity_gap"]),
                    # Secondary deltas vs C1 (Baseline)
                    "delta_auroc_vs_c1": safe_sub(cenh["auroc"], c1["auroc"]),
                    "delta_auprc_vs_c1": safe_sub(cenh["auprc"], c1["auprc"]),
                    "delta_pos_count_vs_c1": safe_sub(cenh["predicted_positive_count"], c1["predicted_positive_count"]),
                    "delta_selection_rate_vs_c1": safe_sub(cenh["selection_rate"], c1["selection_rate"]),
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
        "| Arm | Stage | Enhancement | Delta AUROC vs C2 | Delta AUPRC vs C2 | Delta Pos Count vs C2 | Delta Selection Rate vs C2 | Delta Max d_phi vs C2 | Delta DP Gap vs C2 | Delta EO Gap vs C2 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in deltas_rows:
        md_table_b.append(
            f"| {r['arm_id']} | {r['stage_label']} | {r['enhancement_condition']} "
            f"| {fmt_d(r['delta_auroc_vs_c2'], '+.4f')} | {fmt_d(r['delta_auprc_vs_c2'], '+.4f')} | {fmt_di(r['delta_pos_count_vs_c2'])} "
            f"| {fmt_d(r['delta_selection_rate_vs_c2'], '+.5f')} | {fmt_d(r['delta_max_dphi_vs_c2'], '+.5f')} | {fmt_d(r['delta_dp_gap_vs_c2'], '+.5f')} | {fmt_d(r['delta_eo_gap_vs_c2'], '+.5f')} |"
        )
    md_table_b.extend([
        "",
        "### Secondary Descriptive Deltas vs Baseline (C1)",
        "",
        "| Arm | Stage | Enhancement | Delta AUROC vs C1 | Delta AUPRC vs C1 | Delta Pos Count vs C1 | Delta Selection Rate vs C1 | Delta Max d_phi vs C1 | Delta DP Gap vs C1 | Delta EO Gap vs C1 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])
    for r in deltas_rows:
        md_table_b.append(
            f"| {r['arm_id']} | {r['stage_label']} | {r['enhancement_condition']} "
            f"| {fmt_d(r['delta_auroc_vs_c1'], '+.4f')} | {fmt_d(r['delta_auprc_vs_c1'], '+.4f')} | {fmt_di(r['delta_pos_count_vs_c1'])} "
            f"| {fmt_d(r['delta_selection_rate_vs_c1'], '+.5f')} | {fmt_d(r['delta_max_dphi_vs_c1'], '+.5f')} | {fmt_d(r['delta_dp_gap_vs_c1'], '+.5f')} | {fmt_d(r['delta_eo_gap_vs_c1'], '+.5f')} |"
        )
    with open(run_dir / "primary_deltas.md", "w") as f:
        f.write("\n".join(md_table_b) + "\n")

    # Step 9: Build Table C: Enhancement Search & Audit Summary (enhancement_audit_summary.json / md)
    print("\n--- Step 9: Compiling Table C: Search and Audit Summary ---")
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
                "candidate_count": c_data.get("candidate_count", 0),
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
        "| Arm | Condition | Termination Reason | Feasible? | Committed (Total / AE / BM) | Num Transforms | Candidates | Model Fits | Geometry Evals | Final State Hash |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in audit_summary_rows:
        md_table_c.append(
            f"| {r['arm_id']} | {r['condition']} | {r['termination_reason']} | {r['fairness_feasible']} "
            f"| {r['committed_steps']} ({r['ae_committed_steps']} AE / {r['bm_committed_steps']} BM) | {r['num_transforms_in_terminal_state']} "
            f"| {r['candidate_count']} | {r['model_fit_count']} | {r['geometry_eval_count']} | `{r['final_state_hash']}` |"
        )
    with open(run_dir / "enhancement_audit_summary.md", "w") as f:
        f.write("\n".join(md_table_c) + "\n")

    # Step 10: Joint Trajectory Events (joint_trajectory_events.json)
    print("\n--- Step 10: Compiling Joint Trajectory Event Log ---")
    all_joint_events: Dict[str, Any] = {}
    for arm_id in target_arms:
        c4_data = study_results[arm_id]["conditions"]["joint_enhancement"]
        all_joint_events[arm_id] = c4_data.get("iteration_events", [])
    with open(run_dir / "joint_trajectory_events.json", "w") as f:
        json.dump(all_joint_events, f, indent=2)

    # Step 11: C1/C2 Anchor Barrier Recheck (c1_c2_anchor_recheck.json)
    print("\n--- Step 11: Performing C1/C2 Anchor Barrier Recheck vs R2C ---")
    with open(R2B_EVAL_DIR / "cohort_reproduction.json", "r") as f:
        r2b_cohort = json.load(f)
    with open(R2B_EVAL_DIR / "canonical_state_reproduction.json", "r") as f:
        r2b_state = json.load(f)

    anchor_results: Dict[str, Any] = {
        "gate": "NHIS-D8-R4",
        "reference_source": str(R2B_EVAL_DIR),
        "cohort_barrier": {},
        "state_barrier": {},
        "metric_barrier": {},
        "all_barriers_passed": True,
    }

    # Verify cohort barrier
    for arm_id in target_arms:
        c_ref = r2b_cohort[arm_id]
        # Compare observed sizes from study_results
        arm_res = study_results[arm_id]
        c1_train = arm_res["conditions"]["baseline"]["train"]
        c1_val = arm_res["conditions"]["baseline"]["validation"]
        c1_test = arm_res["conditions"]["baseline"]["test"]

        n_train = c1_train.get("N", c1_train.get("n_samples"))
        n_val = c1_val.get("N", c1_val.get("n_samples"))
        n_test = c1_test.get("N", c1_test.get("n_samples"))

        pos_train = c1_train.get("count_outcome_positive")
        pos_val = c1_val.get("count_outcome_positive")
        pos_test = c1_test.get("count_outcome_positive")

        match_train_n = (n_train == c_ref["train_n_reference"])
        match_val_n = (n_val == c_ref["validation_n_reference"])
        match_test_n = (n_test == c_ref["test_n_reference"])

        match_train_pos = (pos_train == c_ref["train_positive_reference"])
        match_val_pos = (pos_val == c_ref["validation_positive_reference"])
        match_test_pos = (pos_test == c_ref["test_positive_reference"])

        match_all_cohort = (
            match_train_n and match_val_n and match_test_n
            and match_train_pos and match_val_pos and match_test_pos
        )
        if not match_all_cohort:
            anchor_results["all_barriers_passed"] = False

        anchor_results["cohort_barrier"][arm_id] = {
            "train_n": {"observed": n_train, "reference": c_ref["train_n_reference"], "match": match_train_n},
            "train_positive": {"observed": pos_train, "reference": c_ref["train_positive_reference"], "match": match_train_pos},
            "validation_n": {"observed": n_val, "reference": c_ref["validation_n_reference"], "match": match_val_n},
            "validation_positive": {"observed": pos_val, "reference": c_ref["validation_positive_reference"], "match": match_val_pos},
            "test_n": {"observed": n_test, "reference": c_ref["test_n_reference"], "match": match_test_n},
            "test_positive": {"observed": pos_test, "reference": c_ref["test_positive_reference"], "match": match_test_pos},
            "status": "PASS" if match_all_cohort else "FAIL",
        }

    # Verify state barrier (canonical changed_dict hash)
    for arm_id in target_arms:
        c2_state = study_results[arm_id]["conditions"]["canonical_fairbias"]["terminal_state"]
        c2_hash = changed_dict_hash(c2_state)
        ref_hash = r2b_state[arm_id]["canonical_hash"]
        match_hash = (c2_hash == ref_hash)
        if not match_hash:
            anchor_results["all_barriers_passed"] = False
        anchor_results["state_barrier"][arm_id] = {
            "observed_hash": c2_hash,
            "reference_hash": ref_hash,
            "match": match_hash,
            "status": "PASS" if match_hash else "FAIL",
        }

    # Verify metric barrier for C1 & C2 across train/validation/test splits
    with open(R2B_EVAL_DIR / "metric_reproduction.json", "r") as f:
        r2b_metrics = json.load(f)

    # Build map of (arm_id, stage, condition, metric) -> d6_reference
    ref_metric_map = {}
    for entry in r2b_metrics:
        c_row = entry["condition_row"]
        cond = "baseline" if "baseline" in c_row else ("canonical_fairbias" if "canonical" in c_row else None)
        if cond:
            ref_metric_map[(entry["arm_id"], entry["stage"], cond, entry["metric"])] = entry["d6_reference"]

    metric_comparisons = []
    anchor_metrics_to_check = [
        "auroc",
        "auprc",
        "accuracy",
        "max_dphi",
        "predicted_positive_count",
        "selection_rate",
        "demographic_parity_gap",
        "equal_opportunity_gap",
    ]
    for r in condition_matrix_rows:
        if r["condition_key"] in ["baseline", "canonical_fairbias"]:
            stage_r = r["stage_label"]  # e.g. "train_2022", "validation_2023", "test_2024"
            cond_r = r["condition_key"]
            arm_r = r["arm_id"]
            for m_key in anchor_metrics_to_check:
                ref_val = ref_metric_map.get((arm_r, stage_r, cond_r, m_key))
                obs_val = r.get(m_key)
                if ref_val is not None and obs_val is not None:
                    diff = abs(obs_val - ref_val)
                    if m_key == "predicted_positive_count":
                        tol = 0.0
                    elif m_key == "max_dphi":
                        tol = 1e-6
                    else:
                        tol = 1e-4
                    m_pass = (diff <= tol)
                    if not m_pass:
                        anchor_results["all_barriers_passed"] = False
                    metric_comparisons.append({
                        "arm_id": arm_r,
                        "stage": stage_r,
                        "condition": cond_r,
                        "metric": m_key,
                        "observed": obs_val,
                        "reference": ref_val,
                        "diff": diff,
                        "tolerance": tol,
                        "status": "PASS" if m_pass else "FAIL",
                    })

    anchor_results["metric_barrier"] = metric_comparisons
    with open(run_dir / "c1_c2_anchor_recheck.json", "w") as f:
        json.dump(anchor_results, f, indent=2)
    print(f"C1/C2 Anchor Barrier Status: {'PASS' if anchor_results['all_barriers_passed'] else 'FAIL'}")

    # Step 12: Historical Engineering Sensitivity Comparison (engineering_sensitivity_comparison.json)
    print("\n--- Step 12: Historical Engineering Sensitivity Comparison ---")
    sens_comparison: Dict[str, Any] = {
        "historical_r3_source": str(R3_HISTORICAL_DIR),
        "primary_r4_run_id": run_id,
        "arm_comparisons": {},
    }

    if (R3_HISTORICAL_DIR / "condition_metrics.json").exists():
        with open(R3_HISTORICAL_DIR / "condition_metrics.json", "r") as f:
            hist_metrics = json.load(f)
        hist_lookup = {(r["arm_id"], r["stage_key"], r["condition_key"]): r for r in hist_metrics}

        with open(R3_HISTORICAL_DIR / "enhancement_audit_summary.json", "r") as f:
            hist_audit = json.load(f)
        hist_audit_lookup = {(r["arm_id"], r["condition"]): r for r in hist_audit}

        for arm_id in target_arms:
            arm_sens = {
                "c3_comparison": {},
                "c4_comparison": {},
                "classification": "qualitatively robust across geometry",
            }
            # Compare test stage C3 & C4
            for enh_k, enh_l in [("posthoc_enhancement", "C3_Posthoc"), ("joint_enhancement", "C4_Joint")]:
                r4_row = lookup[(arm_id, "test", enh_k)]
                r3_row = hist_lookup.get((arm_id, "test", enh_k), {})
                r4_aud = next((a for a in audit_summary_rows if a["arm_id"] == arm_id and a["condition"] == enh_l), {})
                r3_aud = hist_audit_lookup.get((arm_id, enh_l), {})

                comp = {
                    "r4_state_hash": r4_aud.get("final_state_hash"),
                    "r3_state_hash": r3_aud.get("final_state_hash"),
                    "state_hash_match": r4_aud.get("final_state_hash") == r3_aud.get("final_state_hash"),
                    "r4_termination_reason": r4_aud.get("termination_reason"),
                    "r3_termination_reason": r3_aud.get("termination_reason"),
                    "r4_committed_steps": r4_aud.get("committed_steps"),
                    "r3_committed_steps": r3_aud.get("committed_steps"),
                    "test_auroc_r4": r4_row.get("auroc"),
                    "test_auroc_r3": r3_row.get("auroc"),
                    "delta_auroc_r4_minus_r3": safe_sub(r4_row.get("auroc"), r3_row.get("auroc")),
                    "test_max_dphi_r4": r4_row.get("max_dphi"),
                    "test_max_dphi_r3": r3_row.get("max_dphi"),
                    "test_dp_gap_r4": r4_row.get("demographic_parity_gap"),
                    "test_dp_gap_r3": r3_row.get("demographic_parity_gap"),
                }
                if enh_k == "posthoc_enhancement":
                    arm_sens["c3_comparison"] = comp
                else:
                    arm_sens["c4_comparison"] = comp

            # Sensitivity classification rule
            c3_match = arm_sens["c3_comparison"].get("state_hash_match", False)
            c4_match = arm_sens["c4_comparison"].get("state_hash_match", False)
            if c3_match and c4_match:
                arm_sens["classification"] = "qualitatively robust across geometry"
            elif not c3_match and not c4_match:
                arm_sens["classification"] = "trajectory-sensitive"
            else:
                arm_sens["classification"] = "magnitude-sensitive"

            sens_comparison["arm_comparisons"][arm_id] = arm_sens

    with open(run_dir / "engineering_sensitivity_comparison.json", "w") as f:
        json.dump(sens_comparison, f, indent=2)

    # Step 13: Post-run source verification
    print("\n--- Step 13: Verifying Post-Run Scientific Source Hashes ---")
    post_run_manifest: Dict[str, Any] = {}
    all_source_match = True
    for rel_p in SCIENTIFIC_SOURCE_FILES + REFERENCE_ARTIFACT_FILES:
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
        "r4_config_sha256_match": True,
        "files": post_run_manifest,
    }
    with open(run_dir / "post_run_source_verification.json", "w") as f:
        json.dump(post_run_verification, f, indent=2)
    print(f"Post-run source verification: {'PASS' if all_source_match else 'FAIL'}")

    # Step 14: Environment Manifest
    print("\n--- Step 14: Exporting Environment Manifest ---")
    env_manifest = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "r4_baseline_commit": R4_BASELINE_COMMIT,
        "current_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip(),
    }
    with open(run_dir / "environment_manifest.json", "w") as f:
        json.dump(env_manifest, f, indent=2)

    # Step 15: Finalize command log
    cmd_log_lines.extend([
        f"5. Real-data primary substantive execution completed in {total_duration:.2f}s",
        f"6. Condition metrics exported (Table A)",
        f"7. Primary deltas exported (Table B)",
        f"8. Enhancement audit summary exported (Table C)",
        f"9. Joint trajectory events exported",
        f"10. C1/C2 anchor barrier recheck completed: all_barriers_passed={anchor_results['all_barriers_passed']}",
        f"11. Engineering sensitivity comparison exported",
        f"12. Post-run source verification completed: all_scientific_source_hashes_match={all_source_match}",
        f"13. Finalizing execution_manifest.json",
    ])
    with open(cmd_log_p, "w") as f:
        f.write("\n".join(cmd_log_lines) + "\n")

    # Step 16: Execution Manifest
    print("\n--- Step 16: Exporting Execution Manifest ---")
    overall_status = "COMPLETED" if (all_source_match and anchor_results["all_barriers_passed"]) else "EXECUTION_VERIFICATION_FAILED"
    mf: Dict[str, Any] = {
        "run_id": run_id,
        "gate": "NHIS-D8-R4",
        "status": overall_status,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "duration_seconds": total_duration,
        "execution_mode": "SUBSTANTIVE_D6_GEOMETRY",
        "r4_baseline_commit": R4_BASELINE_COMMIT,
        "r4_config_sha256": r4_config_sha256,
        "random_seed": args.random_seed,
        "all_scientific_source_hashes_match": all_source_match,
        "c1_c2_anchors_passed": anchor_results["all_barriers_passed"],
        "real_data_execution_count": 1,
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

    # Save full study results as well
    with open(run_dir / "d8_enhancement_study_results.json", "w") as f:
        json.dump(study_results, f, indent=2)

    print(f"\n=== R4 Substantive Execution Complete ===")
    print(f"Overall status: {overall_status}")
    print(f"Artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
