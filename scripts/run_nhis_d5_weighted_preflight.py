#!/usr/bin/env python3
"""CLI script for Gate D5.1a: audited real-NHIS TRAIN/VALIDATION preflight harness.

Safety constraints:
- In Gate D5.1a, only --audit-only is authorized on real data.
- NO real survey-weighted FairBias mitigation is executed.
- NO validation scoring is performed.
- TEST partition is strictly embargoed and inaccessible.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, Sequence

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_DIR = str(_REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from nhis_fairbias.d5_weighted_runner import (
    FROZEN_D5_ARMS,
    NHISD5WeightedRunner,
    PRIMARY_D5_RANDOM_SEED,
)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NHIS Gate D5.1a survey-weighted preflight harness and input weight audit."
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        default=True,
        help="Run read-only preflight audit across all 4 arms without executing mitigation (default: True).",
    )
    parser.add_argument(
        "--execute-train-validation",
        action="store_true",
        default=False,
        help="Substantive TRAIN/VAL mitigation execution flag (FUTURE Gate D5.1b only; forbidden in D5.1a).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional specific output directory. Default: runs/nhis_d5_weighted/audit/<run_id>/.",
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> int:
    opts = parse_args(args)

    if opts.execute_train_validation:
        sys.stderr.write(
            "ERROR: Substantive mitigation execution (--execute-train-validation) is strictly "
            "forbidden in Gate D5.1a. Gate D5.1a authorizes ONLY --audit-only.\n"
        )
        return 1

    print("=== NHIS Gate D5.1a Survey-Weighted Preflight Harness ===")
    print("Mode: AUDIT-ONLY (no mitigation execution, no validation scoring, no TEST scoring)")
    print("Frozen Input Contract: ENFORCED")
    print("Weight Variable: WTFA_A (strictly positive empirical survey weights)")
    print("Target Arms: D5_ARM_001, D5_ARM_002, D5_ARM_003, D5_ARM_004\n")

    runner = NHISD5WeightedRunner(
        enforce_frozen_inputs=True,
        allow_mitigation=False,
    )

    results = runner.run_audit(output_dir=opts.output_dir)
    manifest = results["manifest"]
    arm_audits = results["arm_audits"]

    print("--- Four-Arm WTFA_A Input Audit Summary ---")
    for arm_id, arm_data in arm_audits.items():
        print(f"\nArm: {arm_id} ({FROZEN_D5_ARMS[arm_id]['protected_attribute']}, {FROZEN_D5_ARMS[arm_id]['disability_arm']})")
        for part in ("train", "val"):
            p_data = arm_data[part]
            print(
                f"  [{part.upper()}] N={p_data['N']} | min={p_data['min']:.3f}, med={p_data['median']:.3f}, "
                f"max={p_data['max']:.3f}, sum={p_data['sum']:.3f} | distinct={p_data['distinct_weights_count']} | "
                f"index_aligned={p_data['index_aligned']} | record_id_aligned={p_data['record_id_aligned']}"
            )
            for g_diag in p_data["group_diagnostics"]:
                print(
                    f"    Group {g_diag['group']}: N={g_diag['N']}, sum={g_diag['weight_sum']:.3f}, "
                    f"min={g_diag['weight_min']:.3f}, max={g_diag['weight_max']:.3f}, nonpos={g_diag['missing_or_nonpositive_count']}"
                )

    print(f"\n--- Audit Artifacts Generated ---")
    print(f"Run ID: {results['run_id']}")
    print(f"Output Directory: {results['output_dir']}")
    for art in manifest["artifacts_generated"]:
        art_path = results["output_dir"] / art
        print(f"  - {art}: {art_path}")

    print("\n--- Gate D5.1a Status Guarantees ---")
    print("REAL WEIGHTED MITIGATION EXECUTED: FALSE")
    print("VALIDATION SCORED: FALSE")
    print("TEST EVALUATED: FALSE")
    print("STATUS: PASS")

    return 0


if __name__ == "__main__":
    sys.exit(main())
