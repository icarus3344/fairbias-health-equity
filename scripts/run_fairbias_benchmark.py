#!/usr/bin/env python3
"""CLI runner for FairBias benchmark experiments with reproducibility and Pareto checkpointing."""

import argparse
import json
import sys

from fairbias.config import FairBiasConfig
from fairbias.pipeline import run_fairbias_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible, leakage-free FairBias benchmarking experiments."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="compas",
        choices=["compas", "credit"],
        help="Dataset name (compas or credit)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Custom dataset path",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Random seed(s) for multiple runs",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5,
        help="Maximum mitigation iterations",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="LR",
        help="Classifier type (LR, DT, RF, GBDT, XGB, LGBM, CatBoost)",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="EO",
        choices=["EO", "SP"],
        help="Fairness metric for Pareto best-iteration checkpoint selection",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="engineering",
        choices=["official", "engineering"],
        help=(
            "Algorithm mode: 'official' = official_unweighted_reproduction "
            "(MDS fixed dim=2, official interleaved power stream, no "
            "iteration budget, no Pareto rollback); 'engineering' = "
            "engineering_bounded (automatic MDS dim, six-value grid, "
            "bounded iterations, Pareto checkpoint)."
        ),
    )
    parser.add_argument(
        "--enable-ae",
        action="store_true",
        help="Enable accuracy enhancement module (engineering mode only)",
    )

    args = parser.parse_args()

    print(f"=== Starting FairBias Benchmark Run (Dataset: {args.dataset.upper()}) ===")
    
    results = []
    for seed in args.seeds:
        print(f"\n--- Running Seed {seed} ---")
        if args.dataset == "compas":
            cfg = FairBiasConfig.compas_default(
                random_seed=seed,
                max_iterations=args.max_iter,
                classifier=args.classifier,
                selection_metric=args.selection_metric,
                use_accuracy_enhancement=args.enable_ae,
                mode=args.mode,
            )
        else:
            cfg = FairBiasConfig.credit_default(
                random_seed=seed,
                max_iterations=args.max_iter,
                classifier=args.classifier,
                selection_metric=args.selection_metric,
                use_accuracy_enhancement=args.enable_ae,
                mode=args.mode,
            )

        if args.data_path:
            cfg = FairBiasConfig(
                dataset_name=cfg.dataset_name,
                dataset_path=args.data_path,
                label_Y=cfg.label_Y,
                label_O=cfg.label_O,
                random_seed=seed,
                max_iterations=args.max_iter,
                classifier=args.classifier,
                selection_metric=args.selection_metric,
                use_accuracy_enhancement=args.enable_ae,
                algorithm_mode=cfg.algorithm_mode,
            )

        res = run_fairbias_pipeline(cfg)
        results.append(res)

        print(f"Run ID: {res.run_id}")
        print(f"Algorithm mode: {res.algorithm_mode}")
        print(f"Initial ACC: {res.initial_metrics['ACC']:.4f}, Initial EO: {res.initial_metrics['EO']}, Initial SP: {res.initial_metrics['SP']}")
        print(f"Termination: converged={res.termination['converged']} "
              f"reason={res.termination['termination_reason']} "
              f"terminal_iteration={res.termination['terminal_iteration']}")
        if res.algorithm_mode == "official_unweighted_reproduction":
            print(f"Official-Unweighted-Reproduction Final ACC: "
                  f"{res.greedy_terminal_metrics['ACC']:.4f}, "
                  f"EO: {res.greedy_terminal_metrics['EO']}, "
                  f"SP: {res.greedy_terminal_metrics['SP']} "
                  f"(test, greedy termination state — sole reported state, "
                  f"no Pareto rollback in this mode)")
        else:
            print(f"Best Iteration (pareto_engineering): {res.best_iteration}")
            print(f"Selection Reason: {res.best_selection_reason}")
            print(f"Configured-Greedy-Terminal Final ACC: "
                  f"{res.greedy_terminal_metrics['ACC']:.4f}, "
                  f"EO: {res.greedy_terminal_metrics['EO']}, "
                  f"SP: {res.greedy_terminal_metrics['SP']} "
                  f"(test, configured greedy termination state — no "
                  f"paper-alignment claim)")
            print(f"Pareto-Engineering Final ACC: "
                  f"{res.pareto_engineering_metrics['ACC']:.4f}, "
                  f"EO: {res.pareto_engineering_metrics['EO']}, "
                  f"SP: {res.pareto_engineering_metrics['SP']} "
                  f"(engineering extension, not the paper output)")
        print(f"Output saved to: {res.output_file}")

    print("\n=== All Benchmark Runs Completed Successfully ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
