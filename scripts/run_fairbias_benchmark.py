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
        "--enable-ae",
        action="store_true",
        help="Enable accuracy enhancement module",
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
            )
        else:
            cfg = FairBiasConfig.credit_default(
                random_seed=seed,
                max_iterations=args.max_iter,
                classifier=args.classifier,
                selection_metric=args.selection_metric,
                use_accuracy_enhancement=args.enable_ae,
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
            )

        res = run_fairbias_pipeline(cfg)
        results.append(res)

        print(f"Run ID: {res.run_id}")
        print(f"Initial ACC: {res.initial_metrics['ACC']:.4f}, Initial EO: {res.initial_metrics['EO']}, Initial SP: {res.initial_metrics['SP']}")
        print(f"Termination: converged={res.termination['converged']} "
              f"reason={res.termination['termination_reason']} "
              f"terminal_iteration={res.termination['terminal_iteration']}")
        print(f"Best Iteration (pareto_engineering): {res.best_iteration}")
        print(f"Selection Reason: {res.best_selection_reason}")
        print(f"Paper-Strict Final ACC: {res.paper_strict_metrics['ACC']:.4f}, "
              f"EO: {res.paper_strict_metrics['EO']}, SP: {res.paper_strict_metrics['SP']} "
              f"(greedy termination state)")
        print(f"Pareto-Engineering Final ACC: {res.pareto_engineering_metrics['ACC']:.4f}, "
              f"EO: {res.pareto_engineering_metrics['EO']}, SP: {res.pareto_engineering_metrics['SP']} "
              f"(engineering extension, not the paper output)")
        print(f"Output saved to: {res.output_file}")

    print("\n=== All Benchmark Runs Completed Successfully ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
