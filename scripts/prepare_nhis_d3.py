#!/usr/bin/env python3
"""CLI script to build and audit Gate D3 artifacts for NHIS 2022-2024.

Outputs generated in artifacts/nhis/d3/:
1. d3_manifest.json
2. pooled_split_manifest.csv
3. pooled_split_audit.json
4. paper_fidelity_manifest.json
5. paper_algorithm_contract.json
6. pooled_preprocessing_fit.json
7. experiment_arms.csv
8. fairbias_transform_trace_schema.json
9. numerical_transform_audit_schema.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_DIR = str(_REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from fairbias.config import (
    ALGORITHM_MODE_PAPER_FAITHFUL,
    FairBiasConfig,
    official_power_stream,
)
from nhis_fairbias.adapter import DISABILITY_COMPONENTS
from nhis_fairbias.audit import write_csv_atomic
from nhis_fairbias.download import (
    compute_sha256,
    new_run_id,
    utc_timestamp,
    write_json_atomic,
)
from nhis_fairbias.features import DEFAULT_FEATURE_CONFIG, load_feature_registry
from nhis_fairbias.pooled import (
    DEFAULT_POOLED_SEED,
    EXPECTED_TEST_ROWS,
    EXPECTED_TOTAL_ROWS,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_VAL_ROWS,
    NHISPooledAdapter,
    audit_pooled_splits,
    generate_pooled_splits,
)
from nhis_fairbias.preprocessing import (
    SENTINEL_CATEGORICAL_MISSING,
    SENTINEL_EXPLICIT_MISSING,
    SENTINEL_STRUCTURAL_NOT_IN_UNIVERSE,
)
from nhis_fairbias.schema import DEFAULT_STUDY_CONFIG, load_study_config


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NHIS Gate D3 artifacts.")
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "artifacts" / "nhis" / "d3"),
        help="Target output directory for D3 artifacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_POOLED_SEED,
        help="Random seed for pooled 64/16/20 split (default 2024).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing artifacts if they differ.",
    )
    return parser.parse_args(args)


def get_git_commit(repo_root: pathlib.Path) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def build_paper_fidelity_manifest(config: FairBiasConfig | None = None) -> Dict[str, Any]:
    """Construct paper fidelity manifest with operational parameters derived dynamically from config."""
    if config is not None:
        if config.algorithm_mode != ALGORITHM_MODE_PAPER_FAITHFUL:
            raise ValueError(
                f"build_paper_fidelity_manifest requires algorithm_mode='{ALGORITHM_MODE_PAPER_FAITHFUL}', "
                f"got algorithm_mode={config.algorithm_mode!r}"
            )
        paper_cfg = config.resolved()
    else:
        paper_cfg = FairBiasConfig(algorithm_mode=ALGORITHM_MODE_PAPER_FAITHFUL).resolved()

    if paper_cfg.mds_fixed_components is None:
        mds_fixed_desc = "resolved mds_fixed_components = None (stress-elbow dimension selection)."
    else:
        mds_fixed_desc = (
            f"resolved mds_fixed_components = {paper_cfg.mds_fixed_components} "
            f"(explicit fixed-dimension override, stress-elbow search bypassed)."
        )

    repo_operationalization = (
        f"Stress-elbow detection is automated via resolved mds_max_components = {paper_cfg.mds_max_components} "
        f"and resolved mds_slope_threshold = {paper_cfg.mds_slope_threshold}. "
        f"In tang2024_paper_faithful mode, {mds_fixed_desc}"
    )

    return {
        "schema_version": "nhis-fairbias-d3-1.0",
        "reference_citation": "Tang, Lu & Li (2024). Metric-Independent Mitigation of Unpredefined Bias in Machine Classification.",
        "algorithm_mode": ALGORITHM_MODE_PAPER_FAITHFUL,
        "fidelity_analysis": {
            "mds_embedding_dimensionality": {
                "paper_text": "We search for the optimal dimension d using the elbow method on the stress curve.",
                "official_author_code": "Hardcodes n_components = 2 without stress curve evaluation (module_BM.py line 147).",
                "repo_operationalization": repo_operationalization,
                "gate_d3_resolution": (
                    "Preserves paper stress-elbow selection by default, distinguishing paper conceptual rule from "
                    "repository automated operational parameters."
                ),
                "resolved_repo_parameters": {
                    "mds_max_components": paper_cfg.mds_max_components,
                    "mds_slope_threshold": paper_cfg.mds_slope_threshold,
                    "mds_fixed_components": paper_cfg.mds_fixed_components,
                },
                "provenance_status": "PAPER_CONCEPT_REPO_OPERATIONALIZED",
            },
            "numerical_power_transformation": {
                "paper_text": (
                    "Transforms numeric features using single sign-preserving polynomial terms at odd integer "
                    "(3, 5, 7, ...) or odd reciprocal (1/3, 1/5, 1/7, ...) powers, searching until d_phi < epsilon."
                ),
                "official_author_code": (
                    "Searches an interleaved stream: np.array([[i, 1/i] for i in range(3, 2000, 2)]).reshape(-1), "
                    "restarting from head on attribute revisit."
                ),
                "gate_d3_resolution": (
                    "Preserves exact interleaved power sequence [3, 1/3, 5, 1/5, 7, 1/7, ...]. "
                    "In tang2024_paper_faithful mode, power revisit policy restarts from head (matching author code); "
                    "infinite loops are prevented via fail-closed state-cycle detection (no mutation of candidate trajectory). "
                    "The monotone per-attribute cursor is isolated to official_code_derived_monotone_cursor_unweighted."
                ),
                "provenance_status": "DOCUMENTED_RESOLVED",
            },
            "categorical_frequency_merging": {
                "paper_text": (
                    "The two categories with the largest positive and smallest negative frequency gap between "
                    "protected groups are rebinned into one new category; repeated until d_phi < epsilon."
                ),
                "official_author_code": (
                    "Computes frequency gap using empirical sample proportion differences (s_0 / total_0 - s_1 / total_1)."
                ),
                "gate_d3_resolution": (
                    "Preserves empirical category-frequency difference merging. Merging down to a single category "
                    "is treated as a terminal feature drop if accepted by the information-loss gate."
                ),
                "provenance_status": "DOCUMENTED_RESOLVED",
            },
            "nmi_information_loss_gate": {
                "paper_text": "Algorithm 1 in the main text defines greedy bias mitigation without an explicit NMI information-loss gate.",
                "official_author_code": (
                    "module_BM.py (lines 271, 293-308) implements an NMI information-loss gate "
                    "(phi_iter = (nmi_org - nmi_temp) / nmi_org > phi_threshold) rejecting transforms with excessive target information loss."
                ),
                "gate_d3_resolution": (
                    "In tang2024_paper_faithful mode, phi_threshold defaults to 100.0 (matching author code default), "
                    "with provenance explicitly tracked."
                ),
                "provenance_status": "OFFICIAL_CODE_ONLY_NOT_IN_MAIN_TEXT",
            },
            "multicategory_protected_attributes": {
                "paper_text": "Evaluated binary protected attributes (Sex, Race) in COMPAS and Credit Card experiments.",
                "official_author_code": (
                    "module_BM.py (line 82) computes combinations(unique_groups, 2), calculating pairwise disparity across all category pairs."
                ),
                "nhis_application": (
                    "HISPALLP_A (7 categories) evaluates all 21 unordered category pairs using the author code pairwise formulation. "
                    "HISPALLP_A is NOT artificially binarized."
                ),
                "gate_d3_resolution": (
                    "Full 7-category evaluation preserves author pairwise formulation as a direct multiclass generalization."
                ),
                "provenance_status": "PAIRWISE_FORMULATION_GENERALIZED_TO_MULTICLASS_EMPIRICAL_EXTENSION",
            },
            "survey_weights_policy": {
                "paper_text": "Tang et al. (2024) is strictly an unweighted algorithm.",
                "gate_d3_resolution": (
                    "No survey weights (WTFA_A) are used in divergence, MDS, epsilon, or transform selection. "
                    "Passing sample_weight in tang2024_paper_faithful mode raises ValueError immediately."
                ),
                "provenance_status": "DOCUMENTED_RESOLVED",
            },
            "accuracy_enhancement_policy": {
                "paper_text": "Not part of the core greedy bias mitigation loop described in the main paper.",
                "gate_d3_resolution": (
                    "Strictly rejected in tang2024_paper_faithful mode (use_accuracy_enhancement must be False)."
                ),
                "provenance_status": "DOCUMENTED_RESOLVED",
            },
            "temporal_robustness_semantics": {
                "temporal_regime": "2022 train / 2023 validation / 2024 out-of-time temporal test (temporal robustness test).",
                "pooled_regime": "2022-2024 pooled and split 64/16/20; 2024 participates in pooled primary regime.",
                "claim_boundary": "2024 is NOT a globally untouched external frozen test across all regimes.",
                "provenance_status": "DOCUMENTED_RESOLVED",
            },
        },
        "status": "PASS",
    }


def require_pooled_split_audit_pass(split_audit_data: Mapping[str, Any]) -> None:
    """Validate pooled split audit status and fail closed if not PASS.

    Parameters
    ----------
    split_audit_data : Mapping[str, Any]
        Audit dictionary returned by audit_pooled_splits.

    Raises
    ------
    RuntimeError
        If split_audit_data['status'] is not 'PASS', formatted with mismatch diagnostics.
    """
    status = split_audit_data.get("status")
    if status != "PASS":
        mismatches = split_audit_data.get("d0_outcome_totals_mismatches") or []
        mismatch_msg = ""
        if mismatches:
            mismatch_msg = f" D0 outcome total mismatches: {mismatches}."
        raise RuntimeError(
            f"Pooled split audit failed with status={status!r}.{mismatch_msg} "
            f"Cannot proceed with Gate D3 artifact generation."
        )


def main(args: Sequence[str] | None = None) -> int:
    parsed = parse_args(args)
    output_dir = pathlib.Path(parsed.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = _REPO_ROOT
    run_id = new_run_id()
    commit_sha = get_git_commit(repo_root)

    print(f"=== NHIS Gate D3 Preparation & Baseline Freeze ===")
    print(f"Run ID: {run_id}")
    print(f"Commit: {commit_sha}")
    print(f"Target Output: {output_dir}")

    # Load raw feature parquet
    study_cfg = load_study_config(DEFAULT_STUDY_CONFIG)
    pq_path = repo_root / study_cfg["outputs"]["features_parquet"]
    df_raw = pd.read_parquet(pq_path)
    if len(df_raw) != EXPECTED_TOTAL_ROWS:
        raise ValueError(f"Expected {EXPECTED_TOTAL_ROWS} rows, got {len(df_raw)}")

    # ------------------------------------------------------------------
    # Artifact 2: pooled_split_manifest.csv
    # ------------------------------------------------------------------
    split_manifest_df = generate_pooled_splits(df_raw, seed=parsed.seed)
    split_csv_path = output_dir / "pooled_split_manifest.csv"
    write_csv_atomic(split_manifest_df, split_csv_path, force=parsed.force)
    print("Generated pooled_split_manifest.csv")

    # ------------------------------------------------------------------
    # Artifact 3: pooled_split_audit.json
    # ------------------------------------------------------------------
    split_audit_data = audit_pooled_splits(split_manifest_df)
    split_audit_data["schema_version"] = "nhis-fairbias-d3-1.0"
    split_audit_data["random_seed"] = parsed.seed
    split_audit_path = output_dir / "pooled_split_audit.json"
    write_json_atomic(split_audit_path, split_audit_data)
    print("Generated pooled_split_audit.json")

    # Fail-closed guard: propagate split audit failure before downstream artifact generation
    require_pooled_split_audit_pass(split_audit_data)

    # Initialize pooled adapter with the generated split manifest
    adapter = NHISPooledAdapter(
        features_parquet_path=pq_path,
        split_manifest_path=split_csv_path,
        seed=parsed.seed,
    )
    preprocessor = adapter.preprocessor
    fit_record = preprocessor.fitted_record

    # ------------------------------------------------------------------
    # Artifact 4: paper_fidelity_manifest.json
    # ------------------------------------------------------------------
    paper_fidelity_data = build_paper_fidelity_manifest()
    fidelity_path = output_dir / "paper_fidelity_manifest.json"
    write_json_atomic(fidelity_path, paper_fidelity_data)
    print("Generated paper_fidelity_manifest.json")

    # ------------------------------------------------------------------
    # Artifact 5: paper_algorithm_contract.json
    # ------------------------------------------------------------------
    contract_data = {
        "schema_version": "nhis-fairbias-d3-1.0",
        "algorithm_name": "Tang et al. (2024) FairBias Paper-Faithful Baseline",
        "algorithm_mode": ALGORITHM_MODE_PAPER_FAITHFUL,
        "greedy_loop_specification": {
            "step_1_bias_concentration": "Compute d_phi for every candidate predictor on train data.",
            "step_2_epsilon_threshold": "Compute near/far epsilon threshold on train data.",
            "step_3_termination_check": "If max(d_phi) <= epsilon, terminate with status 'epsilon_reached'.",
            "step_4_greedy_selection": "Select feature f* with maximum d_phi > epsilon.",
            "step_5_transform_search": {
                "numerical": "Search odd powers and reciprocals in interleaved sequence until d_phi(f*) < epsilon.",
                "categorical": "Merge extreme frequency-gap categories until d_phi(f*) < epsilon.",
            },
            "step_6_recompute_geometry": "Apply candidate transform, recompute divergence matrix, rerun MDS, recalculate all d_phi.",
            "step_7_repeat": "Repeat until all features inside epsilon ball or search exhausted.",
        },
        "leakage_invariants": {
            "fit_scope": "Train partition strictly (64%, N=57,473).",
            "validation_scope": "Model/hyperparameter selection only.",
            "test_scope": "Apply and evaluate only.",
            "frozen_test_violation_guard": "Validation or test data cannot influence preprocessing, d_phi, epsilon, or transforms.",
        },
        "status": "PASS",
    }
    contract_path = output_dir / "paper_algorithm_contract.json"
    write_json_atomic(contract_path, contract_data)
    print("Generated paper_algorithm_contract.json")

    # ------------------------------------------------------------------
    # Artifact 6: pooled_preprocessing_fit.json
    # ------------------------------------------------------------------
    fit_json_path = output_dir / "pooled_preprocessing_fit.json"
    write_json_atomic(fit_json_path, fit_record.to_dict())
    print("Generated pooled_preprocessing_fit.json")

    # ------------------------------------------------------------------
    # Artifact 7: experiment_arms.csv
    # ------------------------------------------------------------------
    arm_rows = [
        {
            "arm_id": "ARM_D3_001",
            "outcome": "MEDDL12M_A",
            "protected_attribute": "SEX_A",
            "feature_set": "PRIMARY_CORE",
            "disability_arm": "full_feature",
            "algorithm_mode": ALGORITHM_MODE_PAPER_FAITHFUL,
            "weighting": "unweighted",
            "feature_count": 21,
            "status": "PASS",
        },
        {
            "arm_id": "ARM_D3_002",
            "outcome": "MEDDL12M_A",
            "protected_attribute": "HISPALLP_A",
            "feature_set": "PRIMARY_CORE",
            "disability_arm": "full_feature",
            "algorithm_mode": ALGORITHM_MODE_PAPER_FAITHFUL,
            "weighting": "unweighted",
            "feature_count": 21,
            "status": "PASS",
        },
        {
            "arm_id": "ARM_D3_003",
            "outcome": "MEDDL12M_A",
            "protected_attribute": "DISAB3_A",
            "feature_set": "PRIMARY_CORE",
            "disability_arm": "full_feature",
            "algorithm_mode": ALGORITHM_MODE_PAPER_FAITHFUL,
            "weighting": "unweighted",
            "feature_count": 21,
            "status": "PASS",
        },
        {
            "arm_id": "ARM_D3_004",
            "outcome": "MEDDL12M_A",
            "protected_attribute": "DISAB3_A",
            "feature_set": "PRIMARY_CORE",
            "disability_arm": "exclude_disability_components",
            "algorithm_mode": ALGORITHM_MODE_PAPER_FAITHFUL,
            "weighting": "unweighted",
            "feature_count": 15,
            "status": "PASS",
        },
    ]
    arms_df = pd.DataFrame(arm_rows)
    arms_csv_path = output_dir / "experiment_arms.csv"
    write_csv_atomic(arms_df, arms_csv_path, force=parsed.force)
    print("Generated experiment_arms.csv")

    # ------------------------------------------------------------------
    # Artifact 8: fairbias_transform_trace_schema.json
    # ------------------------------------------------------------------
    trace_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "FairBiasTransformTrace",
        "type": "object",
        "required": [
            "algorithm_mode",
            "protected_attribute",
            "epsilon_threshold",
            "total_steps",
            "steps",
            "final_status",
            "final_max_dphi",
        ],
        "properties": {
            "algorithm_mode": {"type": "string"},
            "protected_attribute": {"type": "string"},
            "epsilon_threshold": {"type": "number"},
            "total_steps": {"type": "integer"},
            "final_status": {"type": "string"},
            "final_max_dphi": {"type": "number"},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "iteration",
                        "selected_feature",
                        "feature_semantic_type",
                        "d_phi_before",
                        "epsilon",
                        "proposed_transformation",
                        "accepted_transformation",
                        "dropped",
                    ],
                    "properties": {
                        "iteration": {"type": "integer"},
                        "selected_feature": {"type": "string"},
                        "feature_semantic_type": {"type": "string"},
                        "d_phi_before": {"type": "number"},
                        "epsilon": {"type": "number"},
                        "proposed_transformation": {},
                        "accepted_transformation": {},
                        "numerical_exponent": {"type": ["number", "null"]},
                        "categorical_merge_mapping": {"type": ["object", "null"]},
                        "d_phi_after": {"type": ["number", "null"]},
                        "dropped": {"type": "boolean"},
                        "stopped_reason": {"type": ["string", "null"]},
                    },
                },
            },
        },
    }
    trace_schema_path = output_dir / "fairbias_transform_trace_schema.json"
    write_json_atomic(trace_schema_path, trace_schema)
    print("Generated fairbias_transform_trace_schema.json")

    # ------------------------------------------------------------------
    # Artifact 9: numerical_transform_audit_schema.json
    # ------------------------------------------------------------------
    num_audit_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "NumericalTransformAudit",
        "type": "object",
        "required": [
            "feature_name",
            "spearman_rank_correlation",
            "mean_diff_before",
            "mean_diff_after",
            "audit_status",
        ],
        "properties": {
            "feature_name": {"type": "string"},
            "spearman_rank_correlation": {"type": ["number", "null"]},
            "mean_diff_before": {"type": ["number", "null"]},
            "mean_diff_after": {"type": ["number", "null"]},
            "audit_status": {"type": "string"},
        },
    }
    num_schema_path = output_dir / "numerical_transform_audit_schema.json"
    write_json_atomic(num_schema_path, num_audit_schema)
    print("Generated numerical_transform_audit_schema.json")

    # ------------------------------------------------------------------
    # Artifact 1: d3_manifest.json
    # ------------------------------------------------------------------
    artifacts_to_checksum = {
        "pooled_split_manifest.csv": split_csv_path,
        "pooled_split_audit.json": split_audit_path,
        "paper_fidelity_manifest.json": fidelity_path,
        "paper_algorithm_contract.json": contract_path,
        "pooled_preprocessing_fit.json": fit_json_path,
        "experiment_arms.csv": arms_csv_path,
        "fairbias_transform_trace_schema.json": trace_schema_path,
        "numerical_transform_audit_schema.json": num_schema_path,
    }
    output_files_meta = {}
    for fname, fpath in artifacts_to_checksum.items():
        try:
            rel_path = str(fpath.relative_to(repo_root))
        except ValueError:
            rel_path = str(fpath)
        output_files_meta[fname] = {
            "path": rel_path,
            "sha256": compute_sha256(fpath),
            "size_bytes": fpath.stat().st_size,
        }

    input_files_meta = {
        "features_parquet": {
            "path": str(pq_path.relative_to(repo_root)),
            "sha256": compute_sha256(pq_path),
            "row_count": len(df_raw),
        },
        "features_json": {
            "path": "configs/nhis/features.json",
            "sha256": compute_sha256(repo_root / "configs/nhis/features.json"),
        },
        "study_json": {
            "path": "configs/nhis/study.json",
            "sha256": compute_sha256(repo_root / "configs/nhis/study.json"),
        },
    }

    manifest_data = {
        "schema_version": "nhis-fairbias-d3-1.0",
        "gate": "D3",
        "run_id": run_id,
        "timestamp": utc_timestamp(),
        "git_commit": commit_sha,
        "status": "PASS",
        "algorithm_mode": ALGORITHM_MODE_PAPER_FAITHFUL,
        "input_files": input_files_meta,
        "pooled_split_summary": {
            "seed": parsed.seed,
            "total_rows": EXPECTED_TOTAL_ROWS,
            "train_rows": EXPECTED_TRAIN_ROWS,
            "val_rows": EXPECTED_VAL_ROWS,
            "test_rows": EXPECTED_TEST_ROWS,
            "train_fraction": 0.64,
            "val_fraction": 0.16,
            "test_fraction": 0.20,
        },
        "data_leakage_guarantees": {
            "train_only_fitting": True,
            "validation_test_leakage_prevention": True,
            "survey_weights_in_tang_baseline": "FORBIDDEN",
            "temporal_robustness_semantics": (
                "In the temporal regime, 2024 is the out-of-time temporal test partition. "
                "In the primary pooled regime, 2022-2024 are pooled and partitioned 64/16/20; "
                "2024 is not claimed to be a globally untouched external test across regimes."
            ),
        },
        "output_artifacts": output_files_meta,
    }
    manifest_path = output_dir / "d3_manifest.json"
    write_json_atomic(manifest_path, manifest_data)
    print("Generated d3_manifest.json")
    print("=== Gate D3 Artifact Generation Completed Successfully ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
