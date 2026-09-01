#!/usr/bin/env python3
"""CLI script for Gate D4.1 frozen TEST release harness.

In Gate D4.1a:
  python scripts/run_nhis_d4_test_release.py --audit-only
This is the ONLY authorized mode during Gate D4.1a. It performs read-only verification
of inputs, provenance, preflight run hashes, and embargo constraints without accessing,
loading, scoring, or predicting on the real NHIS TEST partition.

In Gate D4.1b (future, after explicit PI authorization only):
  python scripts/run_nhis_d4_test_release.py --execute-frozen-test --release-id <RELEASE_ID>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Sequence

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_DIR = str(_REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from nhis_fairbias.d4_test_release import (
    FROZEN_FOUR_ARM_REGISTRY,
    NHISD4TestReleaseHarness,
    PRIMARY_D4_PREDICTION_THRESHOLD,
    PRIMARY_D4_RANDOM_SEED,
    PRIMARY_D4_TEST_RELEASE_PROTOCOL_ID,
    REQUIRED_ANALYSIS_COMMIT,
)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NHIS Gate D4.1 Frozen TEST Release Harness CLI"
    )
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--audit-only",
        action="store_true",
        default=True,
        help="Run audit-only verification without evaluating TEST (default safe mode for Gate D4.1a).",
    )
    mode_group.add_argument(
        "--execute-frozen-test",
        action="store_true",
        help="Execute one-time frozen TEST evaluation (strictly embargoed until Gate D4.1b PI review).",
    )
    parser.add_argument(
        "--release-id",
        type=str,
        default=None,
        help="Unique canonical release ID (required when --execute-frozen-test is specified).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional base directory for release outputs.",
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> int:
    opts = parse_args(args)

    harness = NHISD4TestReleaseHarness()

    if opts.execute_frozen_test:
        if not opts.release_id:
            sys.stderr.write(
                "ERROR: --release-id <ID> is required when --execute-frozen-test is specified.\n"
            )
            return 1

        print("=== NHIS Gate D4.1 Frozen TEST Execution ===")
        print(f"Protocol ID: {PRIMARY_D4_TEST_RELEASE_PROTOCOL_ID}")
        print(f"Release ID:  {opts.release_id}")
        print(f"Random Seed: {PRIMARY_D4_RANDOM_SEED}")
        print(f"Threshold:   {PRIMARY_D4_PREDICTION_THRESHOLD}")

        results = harness.execute_release(
            release_id=opts.release_id,
            output_base_dir=opts.output_dir,
            random_seed=PRIMARY_D4_RANDOM_SEED,
            prediction_threshold=PRIMARY_D4_PREDICTION_THRESHOLD,
        )

        manifest = results["manifest"]
        print(f"\nRelease Status: {manifest['release_status']}")
        print(f"Release Directory: {results['release_dir']}")
        print(f"Manifest Path: {results['manifest_path']}")
        print(f"Execution Time: {manifest['execution_time_seconds']}s")
        print("REAL TEST EVALUATED: TRUE")
        return 0

    # Default / explicit --audit-only mode
    print("=== NHIS Gate D4.1 Frozen TEST Release Audit ===")
    print(f"Protocol ID: {PRIMARY_D4_TEST_RELEASE_PROTOCOL_ID}")
    print(f"Required Analysis Commit: {REQUIRED_ANALYSIS_COMMIT}")

    audit_result = harness.run_audit()

    print(f"\nCurrent Git Commit: {audit_result['current_git_commit']}")
    print(f"Four-Arm Registry: {audit_result['four_arm_registry_count']} arms verified")

    input_audit = audit_result["input_audit"]
    print("\n--- Frozen Data Inputs ---")
    print(f"Split Manifest:   {input_audit.get('split_manifest_path')}")
    print(f"Split SHA-256:    {input_audit.get('split_manifest_sha256')}")
    print(f"Features Parquet: {input_audit.get('features_parquet_path')}")
    print(f"Features SHA-256: {input_audit.get('features_parquet_sha256')}")
    print(f"D3 Gate Status:   {input_audit.get('d3_gate_status')}")
    print(f"Pooled Audit:     {input_audit.get('pooled_split_audit_status')}")

    preflight_audit = audit_result["preflight_artifact_audit"]
    print("\n--- Frozen Preflight Run States (Gate D4.0.3) ---")
    for arm_id, arm_info in preflight_audit.get("arms", {}).items():
        spec = FROZEN_FOUR_ARM_REGISTRY[arm_id]
        print(f"[{arm_id}] Run ID: {arm_info['run_id']}")
        print(f"  Protected Attribute: {spec['protected_attribute']}")
        print(f"  Expected Features:   {spec['expected_predictors']}")
        for fname, sha in arm_info["verified_artifacts"].items():
            print(f"  - {fname}: {sha} (MATCH)")

    clf_spec = audit_result["classifier_specification"]
    print(f"\nClassifier Specification: {clf_spec['classifier']} (random_state={clf_spec['random_state']}, solver={clf_spec['solver']}, max_iter={clf_spec['max_iter']})")
    print(f"Prediction Threshold:    {clf_spec['prediction_threshold']}")
    print(f"Test Embargo Active:     {audit_result['test_embargo_active']}")
    print("\n========================================================")
    print("REAL TEST EVALUATED: FALSE")
    print("========================================================")
    print(f"Gate D4.1a Audit Status: {audit_result['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
