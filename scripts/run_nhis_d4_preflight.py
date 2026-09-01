#!/usr/bin/env python3
"""CLI script to execute Gate D4.0 NHIS FairBias train/validation preflight.

Embargo Policy:
The test partition is NEVER evaluated, scored, or predicted.
Execution is restricted to TRAIN + VALIDATION partitions on authorized arms.
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

from nhis_fairbias.d4_runner import (
    FROZEN_D4_ARMS,
    NHISD4Runner,
    PRIMARY_D4_RANDOM_SEED,
)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NHIS Gate D4.0 FairBias preflight on TRAIN and VALIDATION only."
    )
    parser.add_argument(
        "--arm",
        default="ARM_D3_001",
        choices=list(FROZEN_D4_ARMS.keys()),
        help="Target experiment arm (default: ARM_D3_001 / SEX_A).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional specific output directory. If omitted, writes to runs/nhis_d4_preflight/<run_id>/.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=PRIMARY_D4_RANDOM_SEED,
        help=f"Algorithm and classifier random seed (strictly frozen to {PRIMARY_D4_RANDOM_SEED}; non-zero is rejected).",
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> int:
    opts = parse_args(args)

    if opts.seed != PRIMARY_D4_RANDOM_SEED:
        sys.stderr.write(
            f"ERROR: Primary D4 analysis protocol strictly freezes random_seed to "
            f"{PRIMARY_D4_RANDOM_SEED} (got {opts.seed}). Seed sensitivity analysis is not permitted.\n"
        )
        return 1

    print(f"=== NHIS Gate D4.0 FairBias Preflight ===")
    print(f"Arm: {opts.arm}")
    print(f"Seed: {opts.seed} (frozen primary protocol)")
    print(f"Test Partition Embargo: ACTIVE (TRAIN + VALIDATION only)")

    runner = NHISD4Runner(allow_test_evaluation=False)
    results = runner.run_preflight(
        arm_id=opts.arm,
        output_dir=opts.output_dir,
        random_seed=opts.seed,
    )

    manifest = results["manifest"]
    dphi_rec = results["dphi_record"]
    term = manifest["termination_semantics"]
    base_util = results["val_eval_baseline"]["utility"]
    fb_util = results["val_eval_fairbias"]["utility"]
    base_gaps = results["val_eval_baseline"]["fairness_gaps"]
    fb_gaps = results["val_eval_fairbias"]["fairness_gaps"]
    comp = results["val_comparison"]

    print(f"\n--- Run Execution Summary ---")
    print(f"Run ID: {results['run_id']}")
    print(f"Output Directory: {results['output_dir']}")
    print(f"Manifest: {results['manifest_path']}")
    print(f"Commit: {manifest['git_commit']}")
    print(f"Train Cohort N (valid Y/O): {manifest['train_cohort_n_valid']}")
    print(f"Validation Cohort N (valid Y/O): {manifest['validation_cohort_n_valid']}")
    print(f"Active Predictors: {manifest['feature_count']} ({manifest['categorical_feature_count']} categorical, {manifest['numerical_feature_count']} numerical)")
    print(f"Initial Epsilon Threshold: {term['initial_epsilon_threshold']:.6f}")
    print(f"Initial Max d_phi: {term['initial_max_dphi']:.6f} ({term['highest_initial_feature']})")
    print(f"Final Max d_phi: {term['final_max_dphi']:.6f}")
    print(f"Termination Reason: {term['termination_reason']}")
    print(f"Converged: {term['converged']}")
    print(f"Accepted Transforms Count: {dphi_rec['accepted_steps_count']}")
    print(f"Final Changed Dict: {json.dumps(dphi_rec['final_changed_dict'], indent=2)}")

    print(f"\n--- Validation Baseline Performance ---")
    print(f"AUROC: {base_util.get('auroc'):.4f}" if base_util.get('auroc') is not None else "AUROC: null")
    print(f"AUPRC: {base_util.get('auprc'):.4f}" if base_util.get('auprc') is not None else "AUPRC: null")
    print(f"Balanced Accuracy: {base_util.get('balanced_accuracy'):.4f}" if base_util.get('balanced_accuracy') is not None else "Balanced Accuracy: null")
    print(f"F1: {base_util.get('f1'):.4f}" if base_util.get('f1') is not None else "F1: null")
    print(f"Accuracy: {base_util.get('accuracy'):.4f}" if base_util.get('accuracy') is not None else "Accuracy: null")
    print(f"Demographic Parity Gap: {base_gaps.get('demographic_parity_gap'):.4f}" if base_gaps.get('demographic_parity_gap') is not None else "Demographic Parity Gap: null")
    print(f"Equal Opportunity Gap: {base_gaps.get('equal_opportunity_gap'):.4f}" if base_gaps.get('equal_opportunity_gap') is not None else "Equal Opportunity Gap: null")
    print(f"FPR Gap: {base_gaps.get('fpr_gap'):.4f}" if base_gaps.get('fpr_gap') is not None else "FPR Gap: null")
    print(f"Equalized Odds Max Gap: {base_gaps.get('equalized_odds_max_gap'):.4f}" if base_gaps.get('equalized_odds_max_gap') is not None else "Equalized Odds Max Gap: null")

    print(f"\n--- Validation FairBias Performance ---")
    print(f"AUROC: {fb_util.get('auroc'):.4f}" if fb_util.get('auroc') is not None else "AUROC: null")
    print(f"AUPRC: {fb_util.get('auprc'):.4f}" if fb_util.get('auprc') is not None else "AUPRC: null")
    print(f"Balanced Accuracy: {fb_util.get('balanced_accuracy'):.4f}" if fb_util.get('balanced_accuracy') is not None else "Balanced Accuracy: null")
    print(f"F1: {fb_util.get('f1'):.4f}" if fb_util.get('f1') is not None else "F1: null")
    print(f"Accuracy: {fb_util.get('accuracy'):.4f}" if fb_util.get('accuracy') is not None else "Accuracy: null")
    print(f"Demographic Parity Gap: {fb_gaps.get('demographic_parity_gap'):.4f}" if fb_gaps.get('demographic_parity_gap') is not None else "Demographic Parity Gap: null")
    print(f"Equal Opportunity Gap: {fb_gaps.get('equal_opportunity_gap'):.4f}" if fb_gaps.get('equal_opportunity_gap') is not None else "Equal Opportunity Gap: null")
    print(f"FPR Gap: {fb_gaps.get('fpr_gap'):.4f}" if fb_gaps.get('fpr_gap') is not None else "FPR Gap: null")
    print(f"Equalized Odds Max Gap: {fb_gaps.get('equalized_odds_max_gap'):.4f}" if fb_gaps.get('equalized_odds_max_gap') is not None else "Equalized Odds Max Gap: null")

    print(f"\n--- Validation Deltas (FairBias - Baseline) ---")
    print(f"Utility Deltas: {json.dumps(comp['utility_deltas'], indent=2)}")
    print(f"Fairness Gap Deltas: {json.dumps(comp['fairness_gap_deltas'], indent=2)}")
    print(f"Test Evaluated: {manifest['test_evaluated']}")
    print(f"=== Preflight Completed Successfully ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
