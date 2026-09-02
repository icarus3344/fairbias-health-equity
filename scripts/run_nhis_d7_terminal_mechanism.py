#!/usr/bin/env python3
"""Execution script for NHIS FairBias Gate D7 Terminal Mechanism Audit.

Usage:
    Audit-only (Default):
        python scripts/run_nhis_d7_terminal_mechanism.py --audit-only

    Future Substantive Execution (Authorized only after PI freeze):
        python scripts/run_nhis_d7_terminal_mechanism.py \
            --execute-terminal-mechanism \
            --release-id <PI_FROZEN_RELEASE_ID> \
            --expected-execution-head <PI_REVIEWED_SHA>
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from nhis_fairbias.d7_terminal_mechanism import (
    D7TerminalMechanismReleaseManager,
    NHISD7TerminalMechanismHarness,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gate D7: Terminal-State FairBias Mechanism Audit Harness"
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        default=True,
        help="Run read-only verification of frozen D6 archives and 20 state anchors.",
    )
    parser.add_argument(
        "--execute-terminal-mechanism",
        action="store_true",
        default=False,
        help="Execute future substantive terminal mechanism analysis (requires reviewed HEAD & release ID).",
    )
    parser.add_argument(
        "--release-id",
        type=str,
        default=None,
        help="Target release identifier for substantive execution.",
    )
    parser.add_argument(
        "--expected-execution-head",
        type=str,
        default=None,
        help="Reviewed git commit SHA expected for substantive execution.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Substantive execution route
    if args.execute_terminal_mechanism:
        if not args.release_id or not args.expected_execution_head:
            sys.stderr.write(
                "ERROR: Substantive execution requires --release-id and --expected-execution-head.\n"
            )
            return 1

        manager = D7TerminalMechanismReleaseManager(repo_root=_REPO_ROOT)
        try:
            res = manager.execute_release(
                release_id=args.release_id,
                expected_execution_head=args.expected_execution_head,
            )
            print(f"RELEASE COMPLETE: {res['release_id']}")
            print(f"RELEASE DIRECTORY: {res['release_dir']}")
            print(f"MANIFEST SHA-256: {res['manifest_sha256']}")
            return 0
        except Exception as exc:
            sys.stderr.write(f"EXECUTION FAILED: {exc}\n")
            return 2

    # Audit-only execution
    harness = NHISD7TerminalMechanismHarness(repo_root=_REPO_ROOT)
    try:
        results = harness.run_audit_only()
    except Exception as exc:
        sys.stderr.write(f"D7.1a AUDIT FAILED: {exc}\n")
        return 1

    print("D7 TERMINAL MECHANISM AUDIT\n")
    print(f"D6 TRAIN/VAL ARCHIVE:\n{results['d6_train_val_archive']}\n")
    print(f"D6 TEST ARCHIVE:\n{results['d6_test_archive']}\n")
    print(f"D6 FROZEN STATES:\n{results['d6_frozen_states_loaded']}\n")
    print(f"EXPECTED COHORT DIGESTS:\n{results['expected_cohort_digests_loaded']}\n")
    print(f"PREPROCESSING ANCHOR:\n{results['preprocessing_anchor']}\n")
    print(f"SCORING MODE:\n{results['scoring_mode']}\n")
    print(f"FAMILY-I DIAGNOSTICS:\n{results['family1_diagnostics']}\n")
    print(f"FAMILY-II GEOMETRY DIAGNOSTICS:\n{results['family2_geometry_diagnostics']}\n")
    print(f"SCORE-SEPARATION DIAGNOSTICS:\n{results['score_separation_diagnostics']}\n")
    print(f"LOGIT-CONTRIBUTION DIAGNOSTICS:\n{results['logit_contribution_diagnostics']}\n")
    print(f"REAL NHIS COHORT ACCESSED:\n{str(results['real_nhis_cohort_accessed']).upper()}\n")
    print(f"ESTIMATOR FIT COUNT:\n{results['estimator_fit_count']}\n")
    print(f"FAIRBIAS EXECUTION COUNT:\n{results['fairbias_execution_count']}\n")
    print(f"D7.1a AUDIT:\n{results['audit_status']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
