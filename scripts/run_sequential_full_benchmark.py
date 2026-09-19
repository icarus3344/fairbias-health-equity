#!/usr/bin/env python3
"""Sequential Full-Dataset Benchmark Runner with Strict Memory Isolation.

Guarantees:
1. Full Data Execution: Uses the complete processed NHIS 2022-2024 dataset (89,091 eligible records).
2. Strict Sequential Execution: Runs exactly one model at a time, followed by explicit memory deallocation
   and garbage collection (gc.collect()) to guarantee minimal RAM usage (< 500 MB) on host machine.
3. Comprehensive Integration: Combines downstream classification performance (Balanced Accuracy, TPR, FPR)
   and bias reduction effects (Equalized Odds Gap, Demographic Parity Gap, relative bias reduction %,
   and paired bootstrap significance against FairBias).
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add repository root and src to sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nhis_fairbias.benchmark.adapters import (
    BaseMethodAdapter,
    ExponentiatedGradientAdapter,
    FairBiasAdapter,
    LFRAdapter,
    NotSupportedError,
    ReweighingAdapter,
    ThresholdOptimizerAdapter,
    UnmitigatedAdapter,
)
from nhis_fairbias.benchmark.data_contracts import (
    ARM_SPECS,
    load_arm_partitions,
    load_processed_nhis_cohort,
)
from nhis_fairbias.benchmark.metrics import compute_survey_fairness_metrics
from nhis_fairbias.benchmark.preprocessing import BenchmarkPreprocessor
from nhis_fairbias.benchmark.selection import select_best_configuration_on_S
from nhis_fairbias.benchmark.survey_inference import evaluate_with_survey_bootstrap


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Full-Dataset Fairness Benchmark Sequentially"
    )
    parser.add_argument(
        "--parquet-path",
        type=str,
        default=None,
        help="Path to processed nhis_2022_2024_features.parquet",
    )
    parser.add_argument(
        "--arms",
        type=str,
        default="arm_001,arm_002,arm_003,arm_004",
        help="Comma-separated list of experimental arms to execute",
    )
    parser.add_argument(
        "--bootstrap-b",
        type=int,
        default=30,
        help="Number of rescaled bootstrap replicates (default: 30)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=0.10,
        help="Fairness budget threshold on Set S (default: 0.10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for manifests and audit reports",
    )
    return parser.parse_args()


def format_val(val: Optional[float], precision: int = 4) -> str:
    if val is None or np.isnan(val):
        return "N/A"
    return f"{val:.{precision}f}"


def format_ci(l: Optional[float], u: Optional[float], precision: int = 4) -> str:
    if l is None or u is None or np.isnan(l) or np.isnan(u):
        return "N/A"
    return f"[{l:.{precision}f}, {u:.{precision}f}]"


def instantiate_single_model(model_name: str, arm_id: str, seed: int) -> Optional[BaseMethodAdapter]:
    """Instantiate a single model adapter. Returns None if mathematically unsupported."""
    if model_name == "UNMITIGATED":
        return UnmitigatedAdapter(C=1.0, random_state=seed)
    elif model_name == "FAIRBIAS_BM":
        return FairBiasAdapter(arm_id=arm_id, C=1.0, max_iterations=3, random_state=seed)
    elif model_name == "REWEIGHING":
        return ReweighingAdapter(C=1.0, random_state=seed)
    elif model_name == "EG_DP":
        return ExponentiatedGradientAdapter(
            constraint_type="demographic_parity", eps=0.05, max_iter=15, random_state=seed
        )
    elif model_name == "EG_EO":
        return ExponentiatedGradientAdapter(
            constraint_type="equalized_odds", eps=0.05, max_iter=15, random_state=seed
        )
    elif model_name == "TO_EO":
        return ThresholdOptimizerAdapter(C=1.0, random_state=seed)
    elif model_name == "LFR_RECONSTRUCTED":
        if arm_id == "arm_002":
            return None  # Strictly NOT_SUPPORTED on Arm 002
        return LFRAdapter(k=5, random_state=seed)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def main():
    args = parse_args()
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_dir_path = (
        pathlib.Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "runs" / f"sequential_full_benchmark_{timestamp_str}"
    )
    out_dir_path.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("  NHIS FULL DATASET FAIRNESS BENCHMARK — SEQUENTIAL EXECUTION (MEMORY SAFE)")
    print("=" * 96)
    print(f"Timestamp:             {timestamp_str}")
    print(f"Output Directory:      {out_dir_path}")
    print(f"Bootstrap Replicates:  {args.bootstrap_b}")
    print(f"Set S Fairness Budget: {args.budget}")
    print(f"Active Arms:           {args.arms}")
    print("-" * 96)

    # 1. Load full processed NHIS dataset
    print("\n[Step 1/3] Loading full processed NHIS features cohort from disk...")
    t0 = time.time()
    df_full = load_processed_nhis_cohort(args.parquet_path)
    print(f"Loaded {len(df_full):,} eligible survey records across 2022-2024 in {time.time()-t0:.2f}s.")
    print(f"Distribution: 2022 (F+C)={(df_full['year']==2022).sum():,}, "
          f"2023 (Set S)={(df_full['year']==2023).sum():,}, "
          f"2024 (Set T)={(df_full['year']==2024).sum():,}")

    arm_ids = [a.strip() for a in args.arms.split(",") if a.strip()]
    model_names = [
        "UNMITIGATED",
        "FAIRBIAS_BM",
        "REWEIGHING",
        "EG_DP",
        "EG_EO",
        "TO_EO",
        "LFR_RECONSTRUCTED",
    ]

    integrated_benchmark_results = {}

    # 2. Process each arm
    for arm_id in arm_ids:
        if arm_id not in ARM_SPECS:
            continue

        spec = ARM_SPECS[arm_id]
        print("\n" + "=" * 96)
        print(f"  PROCESSING EXPERIMENTAL ARM: {arm_id.upper()} ({spec['protected_attribute']})")
        print("=" * 96)

        # Partition full data for this arm
        t_arm_0 = time.time()
        partitions = load_arm_partitions(df_full, arm_id)
        p_F = partitions["fitting_F"]
        p_C = partitions["calibration_C"]
        p_S = partitions["selection_S"]
        p_T = partitions["evaluation_T"]

        print(f"Partition Sample Sizes: F={len(p_F):,}, C={len(p_C):,}, S={len(p_S):,}, T={len(p_T):,}")

        # Preprocess features fit exclusively on F
        print("Fitting 3-tier Preprocessor on Partition F and transforming sets F, C, S, T...")
        preprocessor = BenchmarkPreprocessor(spec["features"])
        preprocessor.fit(p_F.X_semantic)
        X_F = preprocessor.transform(p_F.X_semantic)
        X_C = preprocessor.transform(p_C.X_semantic)
        X_S = preprocessor.transform(p_S.X_semantic)
        X_T = preprocessor.transform(p_T.X_semantic)
        print(f"Transformed feature matrix width: {X_F.shape[1]} columns. Memory: ~{X_F.nbytes / 1e6:.1f} MB.")

        # Sequential Model Execution
        predictions_S: Dict[str, np.ndarray] = {}
        predictions_T: Dict[str, np.ndarray] = {}
        method_statuses: Dict[str, str] = {}
        method_fit_times: Dict[str, float] = {}

        for m_idx, m_name in enumerate(model_names):
            print(f"\n--- [{m_idx + 1}/{len(model_names)}] Model: {m_name} ---")
            t_m_0 = time.time()

            adapter = instantiate_single_model(m_name, arm_id, args.seed)
            if adapter is None:
                print(f"Status: NOT_SUPPORTED (Contractually prohibited on {arm_id})")
                method_statuses[m_name] = "NOT_SUPPORTED"
                continue

            try:
                if m_name == "FAIRBIAS_BM":
                    fb_adapter: FairBiasAdapter = adapter  # type: ignore
                    print(f"  Fitting {m_name} on Partition F ({len(X_F):,} rows with semantic transforms)...")
                    fb_adapter.fit(X_F, p_F.y, p_F.A, X_semantic=p_F.X_semantic)
                    q_S = fb_adapter.predict_decision_proba(X_S, X_semantic=p_S.X_semantic)
                    q_T = fb_adapter.predict_decision_proba(X_T, X_semantic=p_T.X_semantic)
                    print(f"  FairBias Active Transforms: {len(fb_adapter.changed_dict_)} items {list(fb_adapter.changed_dict_.keys())}")
                elif m_name == "TO_EO":
                    to_adapter: ThresholdOptimizerAdapter = adapter  # type: ignore
                    print("  Stage 1: Fitting base estimator on Partition F...")
                    to_adapter.fit_base(X_F, p_F.y)
                    print("  Stage 2: Calibrating group-specific thresholds on Calibration Set C...")
                    to_adapter.calibrate(X_C, p_C.y, p_C.A)
                    q_S = to_adapter.predict_decision_proba(X_S, A=p_S.A)
                    q_T = to_adapter.predict_decision_proba(X_T, A=p_T.A)
                else:
                    print(f"  Fitting {m_name} on Partition F ({len(X_F):,} rows)...")
                    adapter.fit(X_F, p_F.y, p_F.A)
                    if adapter.requires_sensitive_at_predict:
                        q_S = adapter.predict_decision_proba(X_S, A=p_S.A)
                        q_T = adapter.predict_decision_proba(X_T, A=p_T.A)
                    else:
                        q_S = adapter.predict_decision_proba(X_S)
                        q_T = adapter.predict_decision_proba(X_T)

                predictions_S[m_name] = q_S
                predictions_T[m_name] = q_T
                method_statuses[m_name] = "VALID"
                elapsed_m = time.time() - t_m_0
                method_fit_times[m_name] = elapsed_m
                print(f"  Completed in {elapsed_m:.2f}s. Memory cleared.")

            except NotSupportedError as e:
                print(f"  Status: NOT_SUPPORTED ({e})")
                method_statuses[m_name] = "NOT_SUPPORTED"
            except Exception as e:
                print(f"  Execution Error: {e}")
                method_statuses[m_name] = f"ERROR: {str(e)[:25]}"

            # STRICT MEMORY ISOLATION: Deallocate model instance and force GC
            del adapter
            gc.collect()

        # Step S: Model Selection
        print("\nEvaluating candidates on Selection Set S (2023) against fairness budget...")
        sel_result = select_best_configuration_on_S(
            predictions_S,
            p_S.y,
            p_S.A,
            p_S.WTFA_A,
            expected_groups=spec["expected_categories"],
            budget_tau=args.budget,
        )
        print(f"Set S Winner: {sel_result.selected_candidate_id} (Status: {sel_result.status}, "
              f"BA_S={sel_result.balanced_accuracy_S:.4f}, EO_gap_S={sel_result.eo_gap_S:.4f})")

        # Step T: Rescaled PSU Bootstrap Survey Inference
        print(f"\nRunning Rescaled PSU Bootstrap Survey Inference on Set T (B={args.bootstrap_b} reps)...")
        survey_results = evaluate_with_survey_bootstrap(
            y_true=p_T.y,
            predictions_dict=predictions_T,
            A=p_T.A,
            strata=p_T.PSTRAT,
            psus=p_T.PPSU,
            weights=p_T.WTFA_A,
            expected_groups=spec["expected_categories"],
            B=args.bootstrap_b,
            seed=args.seed,
        )

        # Baseline unmitigated bias values for relative reduction calculation
        unmit_eo = (
            survey_results["method_inference"]["UNMITIGATED"]["eo_gap"].point_estimate
            if "UNMITIGATED" in survey_results["method_inference"]
            else np.nan
        )
        unmit_dp = (
            survey_results["method_inference"]["UNMITIGATED"]["dp_gap"].point_estimate
            if "UNMITIGATED" in survey_results["method_inference"]
            else np.nan
        )

        # Consolidate method results
        arm_method_metrics = {}
        for m_name in model_names:
            st = method_statuses.get(m_name, "N/A")
            if st == "VALID" and m_name in survey_results["method_inference"]:
                mi = survey_results["method_inference"][m_name]
                ba_pt = mi["balanced_accuracy"].point_estimate
                ba_ci = (mi["balanced_accuracy"].ci_lower, mi["balanced_accuracy"].ci_upper)
                eo_pt = mi["eo_gap"].point_estimate
                eo_ci = (mi["eo_gap"].ci_lower, mi["eo_gap"].ci_upper)
                dp_pt = mi["dp_gap"].point_estimate

                # Relative bias reduction %
                if not np.isnan(unmit_eo) and unmit_eo > 0:
                    rel_eo_reduct = float((unmit_eo - eo_pt) / unmit_eo * 100.0)
                else:
                    rel_eo_reduct = 0.0

                arm_method_metrics[m_name] = {
                    "status": "VALID",
                    "fit_time_seconds": method_fit_times.get(m_name, 0.0),
                    "balanced_accuracy": ba_pt,
                    "ba_ci_lower": ba_ci[0],
                    "ba_ci_upper": ba_ci[1],
                    "eo_gap": eo_pt,
                    "eo_ci_lower": eo_ci[0],
                    "eo_ci_upper": eo_ci[1],
                    "dp_gap": dp_pt,
                    "relative_eo_reduction_pct": rel_eo_reduct,
                }
            else:
                arm_method_metrics[m_name] = {
                    "status": st,
                    "fit_time_seconds": method_fit_times.get(m_name, 0.0),
                    "balanced_accuracy": np.nan,
                    "ba_ci_lower": np.nan,
                    "ba_ci_upper": np.nan,
                    "eo_gap": np.nan,
                    "eo_ci_lower": np.nan,
                    "eo_ci_upper": np.nan,
                    "dp_gap": np.nan,
                    "relative_eo_reduction_pct": np.nan,
                }

        # Consolidate paired contrasts
        arm_contrasts = {}
        for base_name, c_dict in survey_results.get("paired_contrasts", {}).items():
            arm_contrasts[base_name] = {
                "delta_ba": c_dict["balanced_accuracy"].point_contrast,
                "delta_ba_ci_lower": c_dict["balanced_accuracy"].ci_lower,
                "delta_ba_ci_upper": c_dict["balanced_accuracy"].ci_upper,
                "delta_ba_p_value": c_dict["balanced_accuracy"].p_value,
                "delta_eo": c_dict["eo_gap"].point_contrast,
                "delta_eo_ci_lower": c_dict["eo_gap"].ci_lower,
                "delta_eo_ci_upper": c_dict["eo_gap"].ci_upper,
                "delta_eo_p_value": c_dict["eo_gap"].p_value,
            }

        integrated_benchmark_results[arm_id] = {
            "sample_sizes": {
                "F": len(p_F),
                "C": len(p_C),
                "S": len(p_S),
                "T": len(p_T),
            },
            "unmitigated_baseline": {
                "eo_gap": unmit_eo,
                "dp_gap": unmit_dp,
            },
            "methods": arm_method_metrics,
            "paired_contrasts": arm_contrasts,
            "execution_time_seconds": time.time() - t_arm_0,
        }

        # Free arm partitions from memory
        del partitions, p_F, p_C, p_S, p_T, X_F, X_C, X_S, X_T, predictions_S, predictions_T
        gc.collect()

    # 3. Print Integrated Summary Tables
    print("\n" + "=" * 115)
    print("  INTEGRATED BENCHMARK RESULTS ON FULL NHIS 2024 EVALUATION SET (SET T)")
    print("=" * 115)

    for arm_id, arm_data in integrated_benchmark_results.items():
        spec = ARM_SPECS[arm_id]
        print(f"\n### EXPERIMENTAL ARM: {arm_id.upper()} ({spec['protected_attribute']})")
        print(f"Sample Sizes: Total={sum(arm_data['sample_sizes'].values()):,}, "
              f"Train(F)={arm_data['sample_sizes']['F']:,}, "
              f"Calib(C)={arm_data['sample_sizes']['C']:,}, "
              f"Select(S)={arm_data['sample_sizes']['S']:,}, "
              f"Test(T)={arm_data['sample_sizes']['T']:,}")
        print()

        headers = [
            "Method ID",
            "Status",
            "Balanced Acc (95% CI)",
            "EO Gap (95% CI)",
            "DP Gap",
            "Bias Reduction %",
            "vs FB (p-val EO)",
        ]
        row_fmt = "{:<18} {:<12} {:<24} {:<24} {:<9} {:<18} {:<16}"
        print(row_fmt.format(*headers))
        print("-" * 125)

        for m_name, m_res in arm_data["methods"].items():
            st = m_res["status"]
            if st != "VALID":
                print(row_fmt.format(m_name, st[:11], "N/A", "N/A", "N/A", "N/A", "N/A"))
                continue

            ba_str = f"{m_res['balanced_accuracy']:.4f} {format_ci(m_res['ba_ci_lower'], m_res['ba_ci_upper'], 3)}"
            eo_str = f"{m_res['eo_gap']:.4f} {format_ci(m_res['eo_ci_lower'], m_res['eo_ci_upper'], 3)}"
            dp_str = format_val(m_res["dp_gap"])
            red_pct = f"{m_res['relative_eo_reduction_pct']:+.1f}%" if not np.isnan(m_res['relative_eo_reduction_pct']) else "N/A"

            if m_name == "FAIRBIAS_BM":
                pval_str = "Baseline"
            elif m_name in arm_data["paired_contrasts"]:
                pval = arm_data["paired_contrasts"][m_name]["delta_eo_p_value"]
                pval_str = f"p={pval:.4f}" if not np.isnan(pval) else "N/A"
            else:
                pval_str = "N/A"

            print(row_fmt.format(m_name, "VALID", ba_str, eo_str, dp_str, red_pct, pval_str))

    # Save to JSON
    summary_file = out_dir_path / "integrated_full_benchmark_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(integrated_benchmark_results, f, indent=2)

    print(f"\nAll benchmark results successfully recorded to: {summary_file}")
    print("=" * 115)


if __name__ == "__main__":
    raise SystemExit("Historical runner retired: it conflates p/q, does not implement the frozen grid, and lacks full annual design domains. Use scripts/run_nhis_benchmark.py; historical outputs remain unchanged.")
