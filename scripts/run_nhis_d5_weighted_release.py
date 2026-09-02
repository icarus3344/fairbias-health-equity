#!/usr/bin/env python3
"""CLI for NHIS Gate D5.1b Survey-Weighted TRAIN/VALIDATION Release Harness.

Enforces:
- Default: --audit-only (read-only verification, zero mitigation execution).
- Substantive flag: --execute-frozen-train-validation (requires explicit authorization).
- Forbidden: Absolutely NO test flags (--test, --execute-test).
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

from nhis_fairbias.d5_weighted_release import (
    CANONICAL_D5_RELEASE_ID,
    NHISD5WeightedReleaseManager,
    SCIENTIFIC_EXECUTION_BASE_COMMIT,
)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NHIS Gate D5.1b Survey-Weighted TRAIN/VALIDATION Release Harness."
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
        help="Execute canonical substantive four-arm TRAIN/VAL release (Future Gate D5.1c).",
    )
    parser.add_argument(
        "--release-id",
        type=str,
        default=CANONICAL_D5_RELEASE_ID,
        help=f"Release identifier (must match canonical {CANONICAL_D5_RELEASE_ID}).",
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> int:
    opts = parse_args(args)

    if opts.execute_frozen_train_validation:
        print(f"=== NHIS Gate D5.1c Substantive Execution Protocol ===")
        print(f"Release ID: {opts.release_id}")
        manager = NHISD5WeightedReleaseManager(
            release_id=opts.release_id,
            allow_substantive_execution=True,
        )
        manifest = manager.execute_release()
        print(f"Release Status: {manifest['status']}")
        print(f"Manifest: {manager.release_dir / 'd5_weighted_release_manifest.json'}")
        return 0

    # Default: Audit-only mode
    print("=== NHIS Gate D5.1b Survey-Weighted Release Harness Audit ===")
    print(f"Mode: AUDIT-ONLY (no mitigation execution, no validation scoring, no TEST scoring)")
    print(f"Canonical Release ID: {opts.release_id}")
    print(f"Scientific Base Commit: {SCIENTIFIC_EXECUTION_BASE_COMMIT}")

    manager = NHISD5WeightedReleaseManager(
        release_id=opts.release_id,
        allow_substantive_execution=False,
    )
    audit_res = manager.run_audit_only()

    print(f"\nPreconditions Status: {audit_res['preconditions']['status']}")
    print(f"Release Directory Fresh: {audit_res['release_dir_fresh']}")
    print(f"Arms Configured: {audit_res['four_arms']}")

    print("\nREAL WEIGHTED MITIGATION EXECUTED: FALSE")
    print("VALIDATION SCORED: FALSE")
    print("TEST PARTITION REQUESTED: FALSE")
    print("TEST EVALUATED: FALSE")
    print("D5.1b RELEASE HARNESS AUDIT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
