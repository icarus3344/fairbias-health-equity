#!/usr/bin/env python3
"""Gate D6.1a.1 CLI for the isolated NHIS temporal TEST harness.

The default and only authorized Gate D6.1a action is:

    python scripts/run_nhis_d6_temporal_test.py --audit-only

The production route is complete but must remain unused until an explicitly
authorized D6.1b one-time execution supplies both a release id and the reviewed
execution commit SHA.
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

from nhis_fairbias.d6_temporal_test_release import (  # noqa: E402
    NHISD6TemporalTestReleaseManager,
)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NHIS Gate D6.1a.1 isolated temporal TEST harness"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--audit-only",
        action="store_true",
        help="Verify the frozen D6 archive and barrier contract without cohort access (default).",
    )
    mode.add_argument(
        "--execute-frozen-temporal-test",
        action="store_true",
        help="Future D6.1b substantive route; requires release id and reviewed execution HEAD.",
    )
    parser.add_argument(
        "--release-id",
        help="PI-frozen D6.1b release identifier; required only for substantive execution.",
    )
    parser.add_argument(
        "--expected-execution-head",
        help="PI-reviewed 40-character Git SHA; required only for substantive execution.",
    )
    return parser.parse_args(args)


def _print_audit(result: dict) -> None:
    print("D6 TEMPORAL TEST HARNESS AUDIT")
    print()
    print("FROZEN D6 TRAIN/VAL TAG:")
    print(result["frozen_d6_train_val_tag"])
    print()
    print("FROZEN D6 TRAIN/VAL ARCHIVE:")
    print(result["frozen_d6_train_val_archive"])
    print()
    print("D6 TRAIN/VAL MANIFEST:")
    print(result["d6_train_val_manifest"])
    print()
    print("D6 TRAIN/VAL LEDGER:")
    print(result["d6_train_val_ledger"])
    print()
    print("EXPECTED TRAIN DIGESTS:")
    print(f"{result['expected_train_digests_loaded']} LOADED")
    print()
    print("EXPECTED TRAINING STATE ANCHORS:")
    print(f"{result['expected_training_state_anchors_loaded']} LOADED")
    print()
    print("PREPROCESSING EXPECTED STATE:")
    print("VERIFIED")
    print()
    print("TEMPORAL TRAIN YEAR:")
    print(result["temporal_train_year"])
    print()
    print("FROZEN VALIDATION YEAR:")
    print(result["frozen_validation_year"])
    print()
    print("FUTURE TEST YEAR:")
    print(result["future_test_year"])
    print()
    print("PRODUCTION 2022 REPRODUCER:")
    print(result["production_2022_reproducer"])
    print()
    print("PRODUCTION 2024 SCORER:")
    print(result["production_2024_scorer"])
    print()
    print("PRODUCTION ARTIFACT WRITER:")
    print(result["production_artifact_writer"])
    print()
    print("GLOBAL BARRIER:")
    print(result["global_training_state_reproduction_barrier"])
    print()
    print("REAL 2022 STATE REPRODUCTION EXECUTED:")
    print(str(result["real_2022_state_reproduction_executed"]).upper())
    print()
    print("2023 COHORT REQUESTED:")
    print(str(result["2023_cohort_requested"]).upper())
    print()
    print("2024 TEST COHORT REQUESTED:")
    print(str(result["2024_test_cohort_requested"]).upper())
    print()
    print("2024 TEST EVALUATED:")
    print(str(result["2024_test_evaluated"]).upper())
    print()
    print("D6.1a.1 AUDIT:")
    print(result["status"])


def main(args: Sequence[str] | None = None) -> int:
    options = parse_args(args)
    manager = NHISD6TemporalTestReleaseManager(repo_root=_REPO_ROOT)
    if options.execute_frozen_temporal_test:
        if not options.release_id or not options.expected_execution_head:
            print(
                "ERROR: substantive D6.1 execution requires both --release-id and "
                "--expected-execution-head. No cohort was accessed and no release was created.",
                file=sys.stderr,
            )
            return 2
        result = manager.execute_production_release(
            _REPO_ROOT / "runs" / "nhis_d6_temporal_test" / "releases",
            release_id=options.release_id,
            expected_execution_head=options.expected_execution_head,
        )
        print(f"D6 temporal TEST release status: {result['status']}")
        return 0 if result["status"] == "COMPLETE" else 1

    audit = manager.run_audit_only()
    _print_audit(audit)
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
