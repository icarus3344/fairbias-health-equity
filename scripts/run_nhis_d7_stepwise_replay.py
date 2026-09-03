#!/usr/bin/env python3
"""Execution script for NHIS FairBias Gate D7.2 Stepwise Representation Replay Audit.

Scientific Question:
"Along the actual accepted FairBias transformation path learned from NHIS 2022,
at which path states do predictive discrimination, class-conditional score separation,
and prediction volume change, and how do those path-dependent changes transport to
the unchanged 2023 and 2024 repeated cross-sections?"

Usage:
    Audit-only (Default):
        python scripts/run_nhis_d7_stepwise_replay.py --audit-only

    Future Substantive Execution (Authorized only after PI review & freeze):
        python scripts/run_nhis_d7_stepwise_replay.py \
            --execute-stepwise-replay \
            --release-id <FROZEN_AFTER_REVIEW> \
            --expected-execution-head <FROZEN_AFTER_REVIEW>
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

from nhis_fairbias.d7_stepwise_replay import (
    D7StepwiseReplayReleaseManager,
    NHISD7StepwiseReplayHarness,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gate D7.2: Path-Dependent Stepwise FairBias Representation Replay Audit Harness"
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        default=True,
        help="Run read-only verification of frozen D6 archives, trace inventory, and terminal barriers.",
    )
    parser.add_argument(
        "--execute-stepwise-replay",
        action="store_true",
        default=False,
        help="Execute future substantive stepwise replay analysis (requires reviewed HEAD & release ID).",
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

    # Substantive execution route (Guarded)
    if args.execute_stepwise_replay:
        if not args.release_id or not args.expected_execution_head:
            sys.stderr.write(
                "ERROR: Substantive execution requires --release-id and --expected-execution-head.\n"
            )
            return 1

        manager = D7StepwiseReplayReleaseManager(repo_root=_REPO_ROOT)
        try:
            res = manager.execute_release(
                release_id=args.release_id,
                expected_execution_head=args.expected_execution_head,
            )
            print(f"RELEASE COMPLETE: {res['release_id']}")
            return 0
        except Exception as exc:
            sys.stderr.write(f"EXECUTION FAILED: {exc}\n")
            return 2

    # Audit-only execution
    harness = NHISD7StepwiseReplayHarness(repo_root=_REPO_ROOT)
    try:
        results = harness.run_audit_only()
    except Exception as exc:
        sys.stderr.write(f"D7.2a AUDIT FAILED: {exc}\n")
        return 1

    print("D7.2 STEPWISE REPLAY AUDIT\n")
    print(f"SCIENTIFIC QUESTION:\n{results['scientific_question']}\n")
    print(f"MANDATORY DISCLOSURE:\n{results['mandatory_disclosure']}\n")
    print(f"PROVENANCE:\n{results['provenance']}\n")
    print(f"STEP COUNTS BY ARM:\n{results['step_counts_by_arm']}\n")
    print(f"TOTAL ACCEPTED STEPS:\n{results['total_accepted_steps']}\n")
    print(f"TOTAL REPRESENTATION STATES:\n{results['total_representation_states']}\n")
    print(f"TERMINAL REPLAY BARRIER:\n{results['terminal_replay_barrier']}\n")
    print(
        f"REAL PREPARED PARQUET OPENED FOR PREPROCESSING VERIFICATION: "
        f"{'YES' if results.get('real_prepared_parquet_opened_for_preprocessing_verification') else 'NO'}\n"
    )
    print(f"REAL NHIS GET_COHORT CALLS:\n{results.get('real_nhis_get_cohort_calls', 0)}\n")
    print(f"MODELS FIT COUNT:\n{results['models_fit_count']}\n")
    print(f"FAIRBIAS MITIGATION EXECUTIONS:\n{results['fairbias_mitigation_count']}\n")
    print(f"D7.2 SUBSTANTIVE EXECUTION:\n{results['d7_2_substantive_execution']}\n")
    print(f"AUDIT STATUS:\n{results['audit_status']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
