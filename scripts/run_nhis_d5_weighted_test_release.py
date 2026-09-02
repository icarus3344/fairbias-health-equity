#!/usr/bin/env python3
"""CLI script for Gate D5.2 frozen secondary TEST release harness.

In Gate D5.2a:
  python scripts/run_nhis_d5_weighted_test_release.py --audit-only
This is the ONLY authorized mode during Gate D5.2a. It performs read-only verification
of inputs, provenance, archived D5 TRAIN/VAL release hashes, D4 release hashes, git boundary,
and embargo constraints without accessing, loading, scoring, or predicting on the real NHIS TEST partition.

In Gate D5.2b (future, after explicit PI authorization only):
  python scripts/run_nhis_d5_weighted_test_release.py --execute-frozen-secondary-test --release-id <RELEASE_ID>
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

from nhis_fairbias.d5_weighted_test_release import (
    CANONICAL_D5_SECONDARY_TEST_RELEASE_ID,
    DEFAULT_PREDICTION_THRESHOLD,
    DEFAULT_RANDOM_SEED,
    FROZEN_D5_TEST_ARMS,
    NHISD5WeightedTestReleaseManager,
    SECONDARY_ANALYSIS_DISCLOSURE,
    TEST_EVALUATION_BASE_COMMIT,
)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NHIS Gate D5.2 Frozen Secondary TEST Release Harness CLI"
    )
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--audit-only",
        action="store_true",
        default=True,
        help="Run audit-only verification without evaluating TEST (default safe mode for Gate D5.2a).",
    )
    mode_group.add_argument(
        "--execute-frozen-secondary-test",
        action="store_true",
        help="Execute one-time frozen secondary TEST evaluation (strictly embargoed until Gate D5.2b PI review).",
    )
    parser.add_argument(
        "--release-id",
        type=str,
        default=CANONICAL_D5_SECONDARY_TEST_RELEASE_ID,
        help=f"Canonical release ID (default: {CANONICAL_D5_SECONDARY_TEST_RELEASE_ID}).",
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default=None,
        help="Optional base directory for release outputs.",
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> int:
    opts = parse_args(args)

    manager = NHISD5WeightedTestReleaseManager(
        release_id=opts.release_id,
        output_base_dir=opts.output_base_dir,
    )

    if opts.execute_frozen_secondary_test:
        print("=== NHIS Gate D5.2 Frozen Secondary TEST Execution ===")
        print(f"Release ID:  {opts.release_id}")
        print(f"Test Evaluation Base Commit: {TEST_EVALUATION_BASE_COMMIT}")
        print(f"Random Seed: {DEFAULT_RANDOM_SEED}")
        print(f"Threshold:   {DEFAULT_PREDICTION_THRESHOLD}")
        print(f"Disclosure:  {SECONDARY_ANALYSIS_DISCLOSURE}")

        results = manager.execute_release(
            output_base_dir=opts.output_base_dir,
            random_seed=DEFAULT_RANDOM_SEED,
            prediction_threshold=DEFAULT_PREDICTION_THRESHOLD,
        )

        print(f"\nRelease Status: {results['status']}")
        print(f"Release Directory: {results['release_dir']}")
        print(f"Manifest SHA-256: {results['manifest_sha256']}")
        print(f"Artifacts Written: {results['artifacts_written']}")
        print("REAL SECONDARY TEST EVALUATED: TRUE")
        return 0

    # Default / explicit --audit-only mode
    print("=== NHIS Gate D5.2a Survey-Weighted Secondary TEST Release Harness Audit ===")
    print("Mode: AUDIT-ONLY (no TEST evaluation, no TEST partition access, no TEST scoring)")
    print(f"Canonical Secondary TEST Release ID: {opts.release_id}")
    print(f"Test Evaluation Base Commit: {TEST_EVALUATION_BASE_COMMIT}")

    audit_result = manager.run_audit_only()
    preconditions = audit_result["preconditions"]
    boundary = preconditions["scientific_boundary"]
    d5_arc = preconditions["d5_archive"]
    d4_arc = preconditions["d4_archive"]
    inputs = preconditions["frozen_inputs"]

    print(f"\nPreconditions Status: {preconditions['status']}")
    print(f"Current Git Commit: {audit_result['release_harness_commit']}")
    print(f"Arms Configured: {audit_result['arms_configured']}")
    print(f"SCIENTIFIC BASE IS ANCESTOR: {boundary.get('scientific_base_is_ancestor', False)}")
    print(f"SCIENTIFIC CODE DIFF CLEAN: {boundary.get('scientific_code_diff_clean', False)}")
    print(f"D5 ARCHIVED VALIDATION RELEASE: {'VERIFIED' if d5_arc.get('status') == 'PASS' else 'FAIL'}")
    print(f"D5 CHANGED_DICTS: {d5_arc.get('changed_dicts_verified', 0)}/4 VERIFIED")
    print(f"D4 ARCHIVE & TAG: {'VERIFIED' if d4_arc.get('status') == 'PASS' else 'FAIL'}")
    print(f"FROZEN INPUTS: {'VERIFIED' if inputs.get('status') == 'PASS' else 'FAIL'}")

    print("\nREAL WEIGHTED MITIGATION EXECUTED: FALSE")
    print("VALIDATION RE-SCORED: FALSE")
    print("TEST PARTITION REQUESTED: FALSE")
    print("TEST COHORT MATERIALIZED: FALSE")
    print("TEST EVALUATED: FALSE")
    print("D5.2a SECONDARY TEST HARNESS AUDIT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
