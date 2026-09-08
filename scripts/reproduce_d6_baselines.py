"""Gate D8-R2B: Baseline Reproduction and Controlled Evaluation Runner.

Performs controlled evidence-bounded baseline and canonical FairBias reproduction
across the 4 frozen NHIS arms:
- D6_ARM_001 (SEX_A)
- D6_ARM_002 (HISPALLP_A)
- D6_ARM_003 (DISAB3_A, full features)
- D6_ARM_004 (DISAB3_A, exclude disability components)

Compares observed cohorts, schemas, canonical transformation states, and metrics
against frozen D6 releases:
- docs/releases/NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7
- docs/releases/NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609
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
import secrets
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
R1_CLOSURE_COMMIT = "d29f7e8fa2e0ab56f21de7407b7d7d921b7b9955"

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

SHARED_6_FILES = [
    "src/fairbias/mitigation.py",
    "src/fairbias/bias_metric.py",
    "src/fairbias/transform.py",
    "src/fairbias/evaluator.py",
    "src/fairbias/models.py",
    "src/fairbias/config.py",
]

PRE_RUN_SOURCE_FILES = [
    "src/nhis_fairbias/d8_enhancement_runner.py",
    "scripts/run_nhis_enhancement_study.py",
    "scripts/reproduce_d6_baselines.py",
    "tests/test_nhis_d8_synthetic_contracts.py",
    "tests/test_fairbias_enhancement.py",
    "tests/test_fairbias_enhancement_contracts.py",
] + SHARED_6_FILES


def compute_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_schema_dict(
    feature_order: List[str],
    categorical_features: List[str],
    numerical_features: List[str],
    protected_attribute: str,
    outcome: str,
    disability_arm: str,
    feature_count: int,
) -> Dict[str, Any]:
    return {
        "disability_arm": str(disability_arm),
        "feature_count": int(feature_count),
        "feature_order": list(feature_order),
        "categorical_features": list(categorical_features),
        "numerical_features": list(numerical_features),
        "outcome": str(outcome),
        "protected_attribute": str(protected_attribute),
    }


def hash_canonical_schema(schema: Dict[str, Any]) -> str:
    canonical_bytes = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def compare_cohorts(
    obs_cohort: Dict[str, Any],
    tv_prov: Dict[str, Any],
    te_prov: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool]:
    """Compare observed cohort directly against reference without reference copying."""
    train_n_ref = int(tv_prov["train_n"])
    train_pos_ref = int(tv_prov["train_outcome_positive_count"])
    val_n_ref = int(tv_prov["validation_n"])
    val_pos_ref = int(tv_prov["validation_outcome_positive_count"])
    test_n_ref = int(te_prov["test_n"])
    test_pos_ref = int(te_prov["test_outcome_positive_count"])

    obs_train_n = int(obs_cohort["train_n"])
    obs_train_pos = int(obs_cohort["train_positives"])
    obs_val_n = int(obs_cohort["validation_n"])
    obs_val_pos = int(obs_cohort["validation_positives"])
    obs_test_n = int(obs_cohort["test_n"])
    obs_test_pos = int(obs_cohort["test_positives"])

    record = {
        "train_n_reference": train_n_ref,
        "train_n_observed": obs_train_n,
        "train_n_match": bool(obs_train_n == train_n_ref),

        "train_positive_reference": train_pos_ref,
        "train_positive_observed": obs_train_pos,
        "train_positive_match": bool(obs_train_pos == train_pos_ref),

        "validation_n_reference": val_n_ref,
        "validation_n_observed": obs_val_n,
        "validation_n_match": bool(obs_val_n == val_n_ref),

        "validation_positive_reference": val_pos_ref,
        "validation_positive_observed": obs_val_pos,
        "validation_positive_match": bool(obs_val_pos == val_pos_ref),

        "test_n_reference": test_n_ref,
        "test_n_observed": obs_test_n,
        "test_n_match": bool(obs_test_n == test_n_ref),

        "test_positive_reference": test_pos_ref,
        "test_positive_observed": obs_test_pos,
        "test_positive_match": bool(obs_test_pos == test_pos_ref),
    }

    match = bool(
        record["train_n_match"]
        and record["train_positive_match"]
        and record["validation_n_match"]
        and record["validation_positive_match"]
        and record["test_n_match"]
        and record["test_positive_match"]
    )
    return record, match


def compare_schemas(
    obs_schema: Dict[str, Any],
    d6_cfg: Dict[str, Any],
    tv_prov: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool]:
    """Compare observed schema against canonical reference schema and verify SHA-256 hashes."""
    ref_fo = tv_prov["frozen_2022_training_state"]["baseline_logistic_regression"]["state"]["feature_order"]
    ref_schema = canonical_schema_dict(
        feature_order=ref_fo,
        categorical_features=d6_cfg["categorical_features"],
        numerical_features=d6_cfg["numerical_features"],
        protected_attribute=d6_cfg["protected_attribute"],
        outcome=d6_cfg["outcome"],
        disability_arm=d6_cfg["disability_arm"],
        feature_count=d6_cfg["expected_predictors"],
    )

    ref_hash = hash_canonical_schema(ref_schema)
    obs_hash = hash_canonical_schema(obs_schema)

    fo_match = bool(ref_schema["feature_order"] == obs_schema["feature_order"])
    cat_match = bool(ref_schema["categorical_features"] == obs_schema["categorical_features"])
    num_match = bool(ref_schema["numerical_features"] == obs_schema["numerical_features"])
    prot_match = bool(ref_schema["protected_attribute"] == obs_schema["protected_attribute"])
    outcome_match = bool(ref_schema["outcome"] == obs_schema["outcome"])
    disab_match = bool(ref_schema["disability_arm"] == obs_schema["disability_arm"])
    count_match = bool(ref_schema["feature_count"] == obs_schema["feature_count"])
    hash_match = bool(ref_hash == obs_hash)

    all_match = bool(
        hash_match
        and fo_match
        and cat_match
        and num_match
        and prot_match
        and outcome_match
        and disab_match
        and count_match
    )

    record = {
        "reference_schema": ref_schema,
        "observed_schema": obs_schema,
        "reference_schema_hash": ref_hash,
        "observed_schema_hash": obs_hash,
        "schema_hashes_match": hash_match,
        "feature_order_match": fo_match,
        "categorical_features_match": cat_match,
        "numerical_features_match": num_match,
        "protected_attribute_match": prot_match,
        "outcome_match": outcome_match,
        "disability_arm_match": disab_match,
        "feature_count_match": count_match,
        "schema_match": all_match,
    }
    return record, all_match


def compare_temporal_metrics(
    arm_id: str,
    arm_label: str,
    base_res: Dict[str, Any],
    canon_res: Dict[str, Any],
    tv_prov: Dict[str, Any],
    d6_train_dphi: Dict[str, Any],
    d6_val_base: Dict[str, Any],
    d6_val_fb: Dict[str, Any],
    d6_val_dphi: Dict[str, Any],
    d6_test_base: Dict[str, Any],
    d6_test_fb: Dict[str, Any],
    d6_test_dphi: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Compare Train 2022, Validation 2023, and Test 2024 metrics."""
    rows: List[Dict[str, Any]] = []

    # 1. Train / 2022 metrics
    train_checks = [
        (f"{arm_label} train", "N", tv_prov["train_n"], base_res["train"]["N"], 0),
        (f"{arm_label} train baseline", "count_outcome_positive", tv_prov["train_outcome_positive_count"], base_res["train"]["count_outcome_positive"], 0),
        (f"{arm_label} train baseline", "max_dphi", d6_train_dphi["initial_max_dphi"], base_res["train"]["max_dphi"], 1e-10),
        (f"{arm_label} train canonical", "count_outcome_positive", tv_prov["train_outcome_positive_count"], canon_res["train"]["count_outcome_positive"], 0),
        (f"{arm_label} train canonical", "max_dphi", d6_train_dphi["final_max_dphi"], canon_res["train"]["max_dphi"], 1e-10),
    ]
    # Filter out None tolerance placeholder
    for row_name, m_name, ref_val, obs_val, tol in train_checks:
        if tol is None:
            continue
        diff = abs(float(obs_val) - float(ref_val))
        passes = bool(diff <= tol)
        rows.append({
            "stage": "train_2022",
            "condition_row": row_name,
            "arm_id": arm_id,
            "metric": m_name,
            "d6_reference": ref_val,
            "d8_reproduced": obs_val,
            "absolute_difference": diff,
            "tolerance": tol,
            "status": "PASS" if passes else "FAIL",
        })

    # 2. Validation / 2023 metrics
    val_specs = [
        (
            f"{arm_label} validation baseline",
            base_res["validation"],
            d6_val_base["utility"],
            d6_val_base["fairness_gaps"],
            d6_val_dphi["original_validation_max_dphi"],
        ),
        (
            f"{arm_label} validation canonical",
            canon_res["validation"],
            d6_val_fb["utility"],
            d6_val_fb["fairness_gaps"],
            d6_val_dphi["transformed_validation_max_dphi"],
        ),
    ]
    for row_name, d8_m, d6_util, d6_gaps, d6_dphi in val_specs:
        checks = [
            ("auroc", d6_util["auroc"], d8_m["auroc"], 1e-10),
            ("auprc", d6_util["auprc"], d8_m["auprc"], 1e-10),
            ("accuracy", d6_util["accuracy"], d8_m["accuracy"], 1e-10),
            ("predicted_positive_count", d6_util["count_predicted_positive"], d8_m["predicted_positive_count"], 0),
            ("selection_rate", d6_util["selection_rate"], d8_m["selection_rate"], 1e-10),
            ("max_dphi", d6_dphi, d8_m["max_dphi"], 1e-10),
            ("demographic_parity_gap", d6_gaps["demographic_parity_gap"], d8_m["demographic_parity_gap"], 1e-10),
            ("equal_opportunity_gap", d6_gaps["equal_opportunity_gap"], d8_m["equal_opportunity_gap"], 1e-10),
        ]
        for m_name, ref_val, obs_val, tol in checks:
            diff = abs(float(obs_val) - float(ref_val))
            passes = bool(diff <= tol)
            rows.append({
                "stage": "validation_2023",
                "condition_row": row_name,
                "arm_id": arm_id,
                "metric": m_name,
                "d6_reference": ref_val,
                "d8_reproduced": obs_val,
                "absolute_difference": diff,
                "tolerance": tol,
                "status": "PASS" if passes else "FAIL",
            })

    # 3. Test / 2024 metrics
    test_specs = [
        (
            f"{arm_label} test baseline",
            base_res["test"],
            d6_test_base["utility"],
            d6_test_base["fairness_gaps"],
            d6_test_dphi["original_test_max_dphi"],
        ),
        (
            f"{arm_label} test canonical",
            canon_res["test"],
            d6_test_fb["utility"],
            d6_test_fb["fairness_gaps"],
            d6_test_dphi["transformed_test_max_dphi"],
        ),
    ]
    for row_name, d8_m, d6_util, d6_gaps, d6_dphi in test_specs:
        checks = [
            ("auroc", d6_util["auroc"], d8_m["auroc"], 1e-10),
            ("auprc", d6_util["auprc"], d8_m["auprc"], 1e-10),
            ("accuracy", d6_util["accuracy"], d8_m["accuracy"], 1e-10),
            ("predicted_positive_count", d6_util["count_predicted_positive"], d8_m["predicted_positive_count"], 0),
            ("selection_rate", d6_util["selection_rate"], d8_m["selection_rate"], 1e-10),
            ("max_dphi", d6_dphi, d8_m["max_dphi"], 1e-10),
            ("demographic_parity_gap", d6_gaps["demographic_parity_gap"], d8_m["demographic_parity_gap"], 1e-10),
            ("equal_opportunity_gap", d6_gaps["equal_opportunity_gap"], d8_m["equal_opportunity_gap"], 1e-10),
        ]
        for m_name, ref_val, obs_val, tol in checks:
            diff = abs(float(obs_val) - float(ref_val))
            passes = bool(diff <= tol)
            rows.append({
                "stage": "test_2024",
                "condition_row": row_name,
                "arm_id": arm_id,
                "metric": m_name,
                "d6_reference": ref_val,
                "d8_reproduced": obs_val,
                "absolute_difference": diff,
                "tolerance": tol,
                "status": "PASS" if passes else "FAIL",
            })

    return rows


def verify_protected_files() -> Dict[str, Any]:
    integrity: Dict[str, Any] = {"root_14": {}, "gitignore": {}, "shared_6": {}}

    for f_rel in ROOT_14_FILES:
        p = REPO_ROOT / f_rel
        diff = subprocess.run(
            ["git", "diff", PROTECTED_BASELINE_TAG, "--", f_rel],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        ).stdout
        integrity["root_14"][f_rel] = {
            "sha256": compute_sha256(p),
            "bytes": p.stat().st_size,
            "diff_vs_baseline_tag": diff,
            "clean": len(diff.strip()) == 0,
        }

    git_p = REPO_ROOT / ".gitignore"
    diff_git = subprocess.run(
        ["git", "diff", PROTECTED_BASELINE_TAG, "--", ".gitignore"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    ).stdout
    integrity["gitignore"][".gitignore"] = {
        "sha256": compute_sha256(git_p),
        "bytes": git_p.stat().st_size,
        "diff_vs_baseline_tag": diff_git,
        "pre_existing_drift": "PRE_EXISTING_BASELINE_DRIFT_RECORDED_NOT_RESOLVED",
    }

    for f_rel in SHARED_6_FILES:
        p = REPO_ROOT / f_rel
        diff = subprocess.run(
            ["git", "diff", R1_CLOSURE_COMMIT, "--", f_rel],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        ).stdout
        integrity["shared_6"][f_rel] = {
            "sha256": compute_sha256(p),
            "bytes": p.stat().st_size,
            "diff_vs_head": diff,
            "clean": len(diff.strip()) == 0,
        }

    return integrity


def export_pre_run_freeze(output_dir: pathlib.Path) -> Dict[str, Any]:
    """Export pre-run source manifest and git diff patch, declaring source frozen."""
    source_manifest: Dict[str, Any] = {}
    for f_rel in PRE_RUN_SOURCE_FILES:
        p = REPO_ROOT / f_rel
        if p.exists():
            source_manifest[f_rel] = {
                "sha256": compute_sha256(p),
                "bytes": p.stat().st_size,
            }

    with open(output_dir / "pre_run_source_manifest.json", "w") as f:
        json.dump(source_manifest, f, indent=2)

    diff_bytes = subprocess.run(
        ["git", "diff", R1_CLOSURE_COMMIT],
        cwd=REPO_ROOT,
        capture_output=True,
    ).stdout
    with open(output_dir / "pre_run_git_diff.patch", "wb") as f:
        f.write(diff_bytes)

    return source_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="NHIS-D8-R2B Baseline Reproduction Gate")
    parser.add_argument("--output-dir", type=str, default=None, help="Explicit output directory")
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed (frozen at 0)")
    parser.add_argument("--command-log-source", type=str, default=None, help="Optional path to existing command log")
    args = parser.parse_args()

    t_start = time.time()
    started_at_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    if args.output_dir is not None:
        run_dir = pathlib.Path(args.output_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        utc_compact = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rand_suffix = secrets.token_hex(4)
        run_dir = REPO_ROOT / "artifacts" / "nhis_d8_r2b" / f"{utc_compact}_{rand_suffix}"
        run_dir.mkdir(parents=True, exist_ok=False)

    run_id = f"r2b_repro_{run_dir.name}"

    print(f"=== Starting NHIS-D8-R2B Evidence Integrity Closure Gate ===")
    print(f"Run ID: {run_id}")
    print(f"Deliverable Directory: {run_dir}")
    print(f"Features Parquet: {FEATURES_PARQUET_PATH} (SHA: {compute_sha256(FEATURES_PARQUET_PATH)})")

    # Step 1: Protected file integrity
    print("\n--- Phase B: Verifying Protected Baseline Files ---")
    prot_integrity = verify_protected_files()
    all_root_clean = all(item["clean"] for item in prot_integrity["root_14"].values())
    all_shared_clean = all(item["clean"] for item in prot_integrity["shared_6"].values())
    if not all_root_clean or not all_shared_clean:
        raise RuntimeError("Protected file integrity violation! Aborting real-data execution.")
    with open(run_dir / "protected_file_integrity.json", "w") as f:
        json.dump(prot_integrity, f, indent=2)
    print("Protected file integrity: PASS")

    # Step 2: Pre-run source freeze
    print("\n--- Freezing Source and Exporting Pre-Run Freeze Manifests ---")
    source_manifest = export_pre_run_freeze(run_dir)
    print("Pre-run source freeze exported: PASS")

    # Step 3: Run D8EnhancementRunner in baseline-reproduction-only mode
    print("\n--- Executing Single Authorized Real-Data Baseline Run (Conditions 1 & 2) ---")
    runner = D8EnhancementRunner(
        smoke_test=False,
        random_seed=args.random_seed,
        run_id=run_id,
        allow_real_data=True,
        baseline_reproduction_only=True,
    )

    study_results: Dict[str, Any] = {}
    target_arms = sorted(list(FROZEN_D6_ARMS.keys()))
    for arm_id in target_arms:
        print(f"Running {arm_id} ({FROZEN_D6_ARMS[arm_id]['protected_attribute']})...")
        t0 = time.time()
        res = runner.run_arm(arm_id)
        study_results[arm_id] = res
        print(f"  Completed in {time.time() - t0:.2f}s")

    total_duration = time.time() - t_start
    completed_at_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Step 4: Compare cohorts, schemas, canonical states, and metrics
    print("\n--- Performing Strict Multi-Barrier Reproduction Checks ---")
    cohort_repro: Dict[str, Any] = {}
    schema_repro: Dict[str, Any] = {}
    canonical_state_repro: Dict[str, Any] = {}
    all_metric_rows: List[Dict[str, Any]] = []

    arm_label_map = {
        "D6_ARM_001": "SEX",
        "D6_ARM_002": "HISP",
        "D6_ARM_003": "DISAB-full",
        "D6_ARM_004": "DISAB-exclude",
    }

    for arm_id in target_arms:
        arm_res = study_results[arm_id]
        conds = arm_res["conditions"]
        base_res = conds["baseline"]
        canon_res = conds["canonical_fairbias"]
        arm_label = arm_label_map[arm_id]

        arm_dir_test = D6_TEST_RELEASE_DIR / arm_id
        arm_dir_tv = D6_TRAIN_VAL_RELEASE_DIR / arm_id

        with open(arm_dir_test / "arm_config.json") as f:
            d6_cfg = json.load(f)
        with open(arm_dir_test / "frozen_changed_dict.json") as f:
            d6_changed_ref = json.load(f)
        with open(arm_dir_test / "test_metrics_baseline.json") as f:
            d6_test_base = json.load(f)
        with open(arm_dir_test / "test_metrics_fairbias.json") as f:
            d6_test_fb = json.load(f)
        with open(arm_dir_test / "test_dphi_before_after.json") as f:
            d6_test_dphi = json.load(f)
        with open(arm_dir_test / "input_provenance.json") as f:
            te_prov = json.load(f)

        with open(arm_dir_tv / "validation_metrics_baseline.json") as f:
            d6_val_base = json.load(f)
        with open(arm_dir_tv / "validation_metrics_fairbias.json") as f:
            d6_val_fb = json.load(f)
        with open(arm_dir_tv / "validation_dphi_before_after.json") as f:
            d6_val_dphi = json.load(f)
        with open(arm_dir_tv / "train_dphi_before_after.json") as f:
            d6_train_dphi = json.load(f)
        with open(arm_dir_tv / "input_provenance.json") as f:
            tv_prov = json.load(f)

        # R2B-01: Cohort reproduction
        c_record, c_pass = compare_cohorts(arm_res["observed_cohort"], tv_prov, te_prov)
        cohort_repro[arm_id] = c_record

        # R2B-02: Schema reproduction
        s_record, s_pass = compare_schemas(arm_res["observed_schema"], d6_cfg, tv_prov)
        schema_repro[arm_id] = s_record

        # Canonical transformation state reproduction
        d6_cd = d6_changed_ref["changed_dict"]
        d8_cd = canon_res["terminal_state"]
        d6_cd_hash = changed_dict_hash(d6_cd)
        d8_cd_hash = changed_dict_hash(d8_cd)
        canonical_state_repro[arm_id] = {
            "d6_transforms_count": len(d6_cd),
            "d8_transforms_count": len(d8_cd),
            "d6_hash": d6_cd_hash,
            "d8_hash": d8_cd_hash,
            "hashes_match": bool(d6_cd_hash == d8_cd_hash),
            "exact_content_match": bool(d6_cd == d8_cd),
        }

        # R2B-03: Complete temporal metrics
        m_rows = compare_temporal_metrics(
            arm_id=arm_id,
            arm_label=arm_label,
            base_res=base_res,
            canon_res=canon_res,
            tv_prov=tv_prov,
            d6_train_dphi=d6_train_dphi,
            d6_val_base=d6_val_base,
            d6_val_fb=d6_val_fb,
            d6_val_dphi=d6_val_dphi,
            d6_test_base=d6_test_base,
            d6_test_fb=d6_test_fb,
            d6_test_dphi=d6_test_dphi,
        )
        all_metric_rows.extend(m_rows)

    # Global barrier evaluation
    all_cohorts_pass = all(
        c["train_n_match"] and c["train_positive_match"]
        and c["validation_n_match"] and c["validation_positive_match"]
        and c["test_n_match"] and c["test_positive_match"]
        for c in cohort_repro.values()
    )
    all_schemas_pass = all(
        s["schema_match"] and s["schema_hashes_match"]
        for s in schema_repro.values()
    )
    all_states_pass = all(
        st["hashes_match"] and st["exact_content_match"]
        for st in canonical_state_repro.values()
    )
    all_train_metrics_pass = all(
        r["status"] == "PASS" for r in all_metric_rows if r["stage"] == "train_2022"
    )
    all_val_metrics_pass = all(
        r["status"] == "PASS" for r in all_metric_rows if r["stage"] == "validation_2023"
    )
    all_test_metrics_pass = all(
        r["status"] == "PASS" for r in all_metric_rows if r["stage"] == "test_2024"
    )
    all_metrics_pass = all_train_metrics_pass and all_val_metrics_pass and all_test_metrics_pass

    all_conditions_bypassed = all(
        arm_res["audit"]["baseline_reproduction_only"] is True
        and arm_res["audit"]["posthoc_enhancement_called"] is False
        and arm_res["audit"]["joint_enhancement_called"] is False
        and arm_res["audit"]["candidate_fits_performed"] == 0
        for arm_res in study_results.values()
    )

    all_reproduction_barriers_pass = bool(
        all_cohorts_pass
        and all_schemas_pass
        and all_states_pass
        and all_train_metrics_pass
        and all_val_metrics_pass
        and all_test_metrics_pass
        and all_conditions_bypassed
    )

    print(f"\nReproduction Barriers Summary:")
    print(f"  All Cohorts Pass:        {all_cohorts_pass}")
    print(f"  All Schemas Pass:        {all_schemas_pass}")
    print(f"  All Canonical States:    {all_states_pass}")
    print(f"  All Temporal Metrics:    {all_metrics_pass} ({len(all_metric_rows)} metrics evaluated)")
    print(f"  Conditions 3/4 Bypassed: {all_conditions_bypassed}")
    print(f"  ALL_REPRODUCTION_BARRIERS_PASS: {all_reproduction_barriers_pass}")

    # Step 5: Save reproduction artifacts
    with open(run_dir / "cohort_reproduction.json", "w") as f:
        json.dump(cohort_repro, f, indent=2)

    with open(run_dir / "schema_reproduction.json", "w") as f:
        json.dump(schema_repro, f, indent=2)

    with open(run_dir / "canonical_state_reproduction.json", "w") as f:
        json.dump(canonical_state_repro, f, indent=2)

    with open(run_dir / "metric_reproduction.json", "w") as f:
        json.dump(all_metric_rows, f, indent=2)

    # Generate Markdown Table
    md_lines = [
        "# NHIS-D8-R2B Complete Temporal Reproduction Table\n",
        f"**Run ID**: `{run_id}`  ",
        f"**Execution Timestamp (UTC)**: `{started_at_utc}`  ",
        f"**ALL_REPRODUCTION_BARRIERS_PASS**: `{'true' if all_reproduction_barriers_pass else 'false'}`\n",
        "| Stage | Condition Row | Metric | D6 Reference | D8 Reproduced | Absolute Difference | Tolerance | Verdict |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in all_metric_rows:
        d6_fmt = f"{r['d6_reference']:.8f}" if isinstance(r["d6_reference"], float) else str(r["d6_reference"])
        d8_fmt = f"{r['d8_reproduced']:.8f}" if isinstance(r["d8_reproduced"], float) else str(r["d8_reproduced"])
        diff_fmt = f"{r['absolute_difference']:.2e}" if isinstance(r["absolute_difference"], float) else str(r["absolute_difference"])
        tol_fmt = f"{r['tolerance']:.2e}" if isinstance(r["tolerance"], float) and r["tolerance"] > 0 else str(r["tolerance"])
        verdict = f"**{r['status']}**" if r["status"] == "PASS" else f"<span style='color:red'>**{r['status']}**</span>"
        md_lines.append(
            f"| {r['stage']} | {r['condition_row']} | `{r['metric']}` | {d6_fmt} | {d8_fmt} | {diff_fmt} | {tol_fmt} | {verdict} |"
        )
    with open(run_dir / "metric_reproduction.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    # Frozen reference manifest
    frozen_ref_manifest = {
        "d6_test_release_dir": str(D6_TEST_RELEASE_DIR),
        "d6_train_val_release_dir": str(D6_TRAIN_VAL_RELEASE_DIR),
        "features_parquet_path": str(FEATURES_PARQUET_PATH),
        "features_parquet_sha256": compute_sha256(FEATURES_PARQUET_PATH),
        "features_parquet_size_bytes": FEATURES_PARQUET_PATH.stat().st_size,
        "protected_baseline_tag": PROTECTED_BASELINE_TAG,
        "r1_closure_commit": R1_CLOSURE_COMMIT,
        "arms_evaluated": target_arms,
    }
    with open(run_dir / "frozen_reference_manifest.json", "w") as f:
        json.dump(frozen_ref_manifest, f, indent=2)

    # Environment manifest
    import sklearn
    env_manifest = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "r1_closure_commit": R1_CLOSURE_COMMIT,
        "current_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip(),
    }
    with open(run_dir / "environment_manifest.json", "w") as f:
        json.dump(env_manifest, f, indent=2)

    # Command log
    cmd_log_p = run_dir / "command_log.txt"
    if args.command_log_source and pathlib.Path(args.command_log_source).exists():
        cmd_log_p.write_bytes(pathlib.Path(args.command_log_source).read_bytes())
    elif not cmd_log_p.exists():
        cmd_log_p.write_text(f"Run {run_id} executed via reproduce_d6_baselines.py at {started_at_utc}\n")

    # Execution manifest
    mf: Dict[str, Any] = {
        "run_id": run_id,
        "gate": "NHIS-D8-R2B",
        "status": "COMPLETED" if all_reproduction_barriers_pass else "REPRODUCTION_FAILED",
        "all_reproduction_barriers_pass": all_reproduction_barriers_pass,
        "all_cohorts_pass": all_cohorts_pass,
        "all_schemas_pass": all_schemas_pass,
        "all_states_pass": all_states_pass,
        "all_train_barriers_pass": all_train_metrics_pass,
        "all_validation_barriers_pass": all_val_metrics_pass,
        "all_test_barriers_pass": all_test_metrics_pass,
        "all_metrics_pass": all_metrics_pass,
        "all_conditions_bypassed": all_conditions_bypassed,
        "total_metrics_evaluated": len(all_metric_rows),
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "duration_seconds": total_duration,
        "baseline_reproduction_only": True,
        "allow_real_data": True,
        "random_seed": args.random_seed,
        "target_arms": target_arms,
        "output_files": {},
    }
    for p in run_dir.iterdir():
        if p.is_file() and p.name != "execution_manifest.json":
            mf["output_files"][p.name] = {
                "path": str(p),
                "sha256": compute_sha256(p),
                "size_bytes": p.stat().st_size,
            }
    with open(run_dir / "execution_manifest.json", "w") as f:
        json.dump(mf, f, indent=2)

    print(f"\nExecution Manifest written: {run_dir / 'execution_manifest.json'}")


if __name__ == "__main__":
    main()
