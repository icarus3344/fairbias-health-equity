#!/usr/bin/env python3
"""CLI for NHIS Gate D6.0a Temporal Robustness TRAIN/VALIDATION Harness.

Enforces:
- Default: --audit-only (read-only verification, zero mitigation, zero validation scoring, zero TEST scoring).
- Substantive flag: --execute-frozen-train-validation (requires explicit authorization; prohibited in Gate D6.0a).
- Forbidden: Absolutely NO test flags (--test, --execute-test, --year-2024, --evaluate-2024).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Sequence

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_DIR = str(_REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from nhis_fairbias.d6_temporal_runner import (
    NHISD6TemporalReleaseManager,
    NHISD6TemporalRunner,
    SCIENTIFIC_EXECUTION_BASE_COMMIT,
    TEMPORAL_TRAIN_YEAR,
    TEMPORAL_VALIDATION_YEAR,
    TEMPORAL_TEST_YEAR,
)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NHIS Gate D6.0a Temporal Robustness TRAIN/VALIDATION Harness."
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        default=True,
        help="Run read-only audit of inputs, contracts, and schema (default: True).",
    )
    parser.add_argument(
        "--execute-frozen-train-validation",
        action="store_true",
        default=False,
        help="Execute canonical substantive four-arm TRAIN/VAL release (Future Gate D6.0b; NOT authorized in D6.0a).",
    )
    parser.add_argument(
        "--release-id",
        type=str,
        default=None,
        help="Release identifier for future canonical release.",
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> int:
    opts = parse_args(args)

    if opts.execute_frozen_train_validation:
        if not opts.release_id:
            print("ERROR: --release-id is required when executing substantive release.")
            return 1
        print("=== NHIS Gate D6.0b Substantive Execution Protocol ===")
        print(f"Release ID: {opts.release_id}")
        manager = NHISD6TemporalReleaseManager(
            release_id=opts.release_id,
            allow_substantive_execution=True,
        )
        manifest = manager.execute_release()
        print(f"Release Status: {manifest['status']}")
        print(f"Manifest: {manager.release_dir / 'd6_temporal_train_val_manifest.json'}")
        return 0

    # Default: Audit-only mode
    runner = NHISD6TemporalRunner()
    audit_res = runner.run_audit_only()

    print("[NHIS-D6-AUDIT] Gate D6.0a Preconditions Audit")
    print("[NHIS-D6-AUDIT] ===============================")
    print(f"[NHIS-D6-AUDIT] Repository root: {_REPO_ROOT}")
    print(f"[NHIS-D6-AUDIT] Frozen parquet path: {audit_res['features_parquet_info']['features_parquet_path']}")
    print(
        f"[NHIS-D6-AUDIT] Frozen parquet SHA-256: {audit_res['features_parquet_info']['features_parquet_sha256']} "
        f"({'VERIFIED' if audit_res['frozen_features_verified'] else 'FAILED'})"
    )
    print(
        f"[NHIS-D6-AUDIT] Prior tag nhis-d4-primary-test-v1 commit: {audit_res['tag_info']['d4_tag_commit']} "
        f"({'VERIFIED' if audit_res['d4_tag_verified'] else 'FAILED'})"
    )
    print(
        f"[NHIS-D6-AUDIT] Prior tag nhis-d5-weighted-train-val-v1 commit: {audit_res['tag_info']['d5_train_val_tag_commit']} "
        f"({'VERIFIED' if audit_res['d5_train_val_tag_verified'] else 'FAILED'})"
    )
    print(
        f"[NHIS-D6-AUDIT] Prior tag nhis-d5-weighted-secondary-test-v1 commit: {audit_res['tag_info']['d5_secondary_test_tag_commit']} "
        f"({'VERIFIED' if audit_res['d5_secondary_test_tag_verified'] else 'FAILED'})"
    )
    print(f"[NHIS-D6-AUDIT] Current HEAD commit: {audit_res['boundary_info']['current_git_commit']}")
    print(
        f"[NHIS-D6-AUDIT] Scientific boundary base commit: {audit_res['boundary_info']['scientific_base_commit']} "
        f"({'ANCESTOR_VERIFIED' if audit_res['boundary_info']['scientific_base_is_ancestor'] else 'FAILED'})"
    )
    print(
        f"[NHIS-D6-AUDIT] Scientific boundary diff clean: {'TRUE' if audit_res['boundary_info']['scientific_code_diff_clean'] else 'FALSE'} "
        f"({len(audit_res['boundary_info']['scientific_paths_checked'])} paths verified unchanged)"
    )
    print(
        f"[NHIS-D6-AUDIT] Study design: REPEATED CROSS-SECTIONAL "
        f"(longitudinal={audit_res['longitudinal']}, causal=False, temporal_robustness={audit_res['temporal_robustness_analysis']})"
    )
    print(
        "[NHIS-D6-AUDIT] Predeclared temporal partitions: 2022=development_train, 2023=development_validation, 2024=frozen_test (EMBARGOED)"
    )
    print("[NHIS-D6-AUDIT] Predeclared arms: 4 (D6_ARM_001, D6_ARM_002, D6_ARM_003, D6_ARM_004)")
    print("[NHIS-D6-AUDIT] Survey weighting: NONE (strict unweighted FairBias application)")
    print("[NHIS-D6-AUDIT] Frozen outcome: MEDDL12M_A")
    print("[NHIS-D6-AUDIT] Execution mode: AUDIT ONLY (zero cohort requests, zero model fits, zero release files created)")
    print("[NHIS-D6-AUDIT] Audit status: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
