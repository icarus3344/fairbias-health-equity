#!/usr/bin/env python3
"""CLI entrypoint for executing the MEPS longitudinal fairness research pipeline."""

from __future__ import annotations

import argparse
import pathlib
import sys

# Deterministically add src directory to sys.path
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_DIR = str(_REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from meps_fairness.pipeline import run_pipeline


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute MEPS longitudinal coverage prediction and fairness mitigation pipeline."
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "formal"],
        default="smoke",
        help="Pipeline execution mode (default: smoke). Smoke mode is non-evidentiary verification.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260828,
        help="Random seed for reproducible partitioning and bootstrap (default: 20260828).",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=25,
        help="Number of design-aware stratified PSU bootstrap replicates (default: 25).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="runs",
        help="Root directory for no-clobber execution run directories (default: runs).",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    ns = parse_args(args)
    print(f"Starting MEPS Fairness Pipeline (mode={ns.mode}, seed={ns.seed}, bootstraps={ns.bootstraps})...")

    result = run_pipeline(
        mode=ns.mode,
        seed=ns.seed,
        repo_root=_REPO_ROOT,
        n_bootstraps=ns.bootstraps,
        output_root=ns.output_root,
    )

    print("\n=== Execution Summary ===")
    print(f"Run ID: {result.run_id}")
    print(f"Status: {result.status}")
    print(f"Runtime: {result.runtime_seconds:.2f} seconds")
    print(f"Peak RSS: {result.peak_rss_gb:.2f} GB")
    print(f"Output Directory: {result.output_directory}")
    print(f"Panel 27 Holdout Status: {result.panel27_status}")
    print("=========================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
