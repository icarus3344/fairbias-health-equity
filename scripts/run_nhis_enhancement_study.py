"""Execution CLI for Gate D8: Exploratory Accuracy Enhancement Study on NHIS Arms."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import secrets
import sys
import time
from typing import Any, Dict, List

import pandas as pd

from nhis_fairbias.d8_enhancement_runner import D8EnhancementRunner, D8ExecutionMode
from nhis_fairbias.d6_temporal_runner import FROZEN_D6_ARMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Accuracy Enhancement benchmark on the four NHIS temporal study arms."
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="all",
        choices=["all", "D6_ARM_001", "D6_ARM_002", "D6_ARM_003", "D6_ARM_004"],
        help="Target study arm to evaluate (default: 'all').",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on small subsample for fast smoke verification.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for study results. If not specified, a unique timestamped directory is created.",
    )
    parser.add_argument(
        "--allow-real-data",
        action="store_true",
        help="Explicit permission flag required to read real NHIS microdata parquet.",
    )
    parser.add_argument(
        "--baseline-reproduction-only",
        action="store_true",
        help="Execute only baseline and canonical FairBias conditions without enhancement search.",
    )
    parser.add_argument(
        "--execution-mode",
        "--mode",
        type=str,
        default=D8ExecutionMode.SUBSTANTIVE_D6_GEOMETRY.value,
        choices=[m.value for m in D8ExecutionMode],
        help="Explicit study execution mode (default: SUBSTANTIVE_D6_GEOMETRY).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for model fitting and evaluation (default: 0).",
    )
    parser.add_argument(
        "--r4-primary-authorized",
        action="store_true",
        help="Explicit supervisor authorization flag for the single primary Gate D8-R4 real-data execution.",
    )
    return parser.parse_args()


def build_comparison_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for arm_id, arm_data in results.items():
        conds = arm_data["conditions"]
        for cond_name, cond_res in conds.items():
            test_res = cond_res.get("test")
            train_res = cond_res.get("train")
            rows.append({
                "arm_id": arm_id,
                "protected_attribute": arm_data["protected_attribute"],
                "condition": cond_name,
                "num_transforms": cond_res.get("num_transforms", 0),
                "test_auroc": round(test_res["auroc"], 4) if test_res and "auroc" in test_res else None,
                "test_auprc": round(test_res["auprc"], 4) if test_res and "auprc" in test_res else None,
                "test_auprc_trapezoidal": round(test_res.get("auprc_trapezoidal", test_res.get("auprc")), 4) if test_res else None,
                "test_average_precision": round(test_res.get("average_precision", test_res.get("auprc")), 4) if test_res else None,
                "test_accuracy": round(test_res["accuracy"], 4) if test_res and "accuracy" in test_res else None,
                "test_positives_0_5": test_res.get("predicted_positive_count") if test_res else None,
                "test_positive_rate": round(test_res["predicted_positive_rate"], 5) if test_res and "predicted_positive_rate" in test_res else None,
                "test_max_dphi": round(test_res["max_dphi"], 5) if test_res and "max_dphi" in test_res else None,
                "train_max_dphi": round(train_res["max_dphi"], 5) if train_res and "max_dphi" in train_res else None,
                "epsilon_threshold": round(arm_data["epsilon_threshold"], 5),
                "epsilon_threshold_source": arm_data.get("epsilon_threshold_source"),
                "frozen_reference_artifact": arm_data.get("frozen_reference_artifact"),
                "computed_initial_threshold_diagnostic": round(arm_data.get("computed_initial_threshold_diagnostic", 0.0), 5) if arm_data.get("computed_initial_threshold_diagnostic") is not None else None,
                "fairness_feasible": cond_res.get("fairness_feasible", False),
                "termination_reason": cond_res.get("termination_reason", "unknown"),
                "test_dp_difference": round(test_res["demographic_parity_difference"], 4) if test_res and "demographic_parity_difference" in test_res else None,
                "test_eo_difference": round(test_res["equal_opportunity_difference"], 4) if test_res and "equal_opportunity_difference" in test_res else None,
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    # Resolve execution mode
    if args.baseline_reproduction_only:
        mode = D8ExecutionMode.BASELINE_REPRODUCTION
    else:
        try:
            mode = D8ExecutionMode(args.execution_mode)
        except ValueError:
            raise ValueError(
                f"Unknown execution mode: '{args.execution_mode}'. Valid modes are: {[m.value for m in D8ExecutionMode]}"
            )

    if args.allow_real_data:
        if mode == D8ExecutionMode.BASELINE_REPRODUCTION:
            pass
        elif mode == D8ExecutionMode.SUBSTANTIVE_D6_GEOMETRY:
            if not args.r4_primary_authorized:
                raise ValueError(
                    "--allow-real-data is strictly prohibited without --baseline-reproduction-only "
                    "or explicit --r4-primary-authorized flag. Gate D8-R4 primary authorization required."
                )
        else:
            raise ValueError(
                f"--allow-real-data is strictly prohibited without --baseline-reproduction-only (prohibited with mode '{mode.value}')."
            )
    elif args.r4_primary_authorized:
        if mode != D8ExecutionMode.SUBSTANTIVE_D6_GEOMETRY:
            raise ValueError(
                f"--r4-primary-authorized is only valid with SUBSTANTIVE_D6_GEOMETRY, got '{mode.value}'."
            )

    # Collision-proof output directory resolution
    if args.output_dir is not None:
        out_dir = pathlib.Path(args.output_dir)
        if out_dir.exists():
            raise FileExistsError(
                f"Output directory {out_dir} already exists. "
                "Refusing to reuse or overwrite existing directory."
            )
        out_dir.mkdir(parents=True, exist_ok=False)
    else:
        utc_ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rand_suffix = secrets.token_hex(4)
        out_dir = pathlib.Path("runs/d8_enhancement_study") / f"{utc_ts}_{rand_suffix}"
        out_dir.mkdir(parents=True, exist_ok=False)

    target_arms = (
        sorted(list(FROZEN_D6_ARMS.keys()))
        if args.arm == "all"
        else [args.arm]
    )

    run_id = f"d8_run_{out_dir.name}"
    print(f"=== Starting D8 Accuracy Enhancement Study ===")
    print(f"Run ID: {run_id}")
    print(f"Target arms: {target_arms}")
    print(f"Execution mode: {mode.value}")
    print(f"Smoke test mode: {args.smoke_test}")
    print(f"Allow real data: {args.allow_real_data}")
    print(f"R4 primary authorized: {args.r4_primary_authorized}")
    print(f"Baseline reproduction only: {mode == D8ExecutionMode.BASELINE_REPRODUCTION}")
    print(f"Random seed: {args.random_seed}")
    print(f"Output directory: {out_dir}")

    # Write initial startup manifest
    manifest_path = out_dir / "execution_manifest.json"
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    init_manifest = {
        "run_id": run_id,
        "status": "RUNNING",
        "started_at_utc": started_at,
        "execution_mode": mode.value,
        "smoke_test": args.smoke_test,
        "allow_real_data": args.allow_real_data,
        "r4_primary_authorized": args.r4_primary_authorized,
        "baseline_reproduction_only": mode == D8ExecutionMode.BASELINE_REPRODUCTION,
        "random_seed": args.random_seed,
        "target_arms": target_arms,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(init_manifest, f, indent=2)

    try:
        runner = D8EnhancementRunner(
            mode=mode,
            smoke_test=args.smoke_test,
            run_id=run_id,
            allow_real_data=args.allow_real_data,
            r4_primary_authorized=args.r4_primary_authorized,
            random_seed=args.random_seed,
        )

        study_results: Dict[str, Any] = {}
        t0 = time.time()

        for arm_id in target_arms:
            print(f"\n[{arm_id}] Evaluating {arm_id} ({FROZEN_D6_ARMS[arm_id]['protected_attribute']})...")
            t_arm0 = time.time()
            res = runner.run_arm(arm_id)
            study_results[arm_id] = res
            print(f"[{arm_id}] Finished in {time.time() - t_arm0:.1f}s.")

        total_time = time.time() - t0
        print(f"\n=== All arms completed in {total_time:.1f}s ===")

        # 1. Save JSON results
        json_path = out_dir / "d8_enhancement_study_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(study_results, f, indent=2)
        print(f"Saved raw results to {json_path}")

        # 2. Build and save comparative table
        df_cmp = build_comparison_dataframe(study_results)
        csv_path = out_dir / "d8_enhancement_comparison.csv"
        df_cmp.to_csv(csv_path, index=False)
        print(f"Saved summary comparison CSV to {csv_path}")

        # 3. Save candidate audit events (JSON Lines)
        audit_path = out_dir / "d8_candidate_audit_events.jsonl"
        with open(audit_path, "w", encoding="utf-8") as f:
            for ev in runner.audit_events:
                f.write(json.dumps(ev.to_dict()) + "\n")
        print(f"Saved {len(runner.audit_events)} candidate audit events to {audit_path}")

        # 4. Save completed execution manifest
        def _file_sha(p: pathlib.Path) -> str:
            return hashlib.sha256(p.read_bytes()).hexdigest()

        manifest = {
            "run_id": run_id,
            "status": "COMPLETED",
            "started_at_utc": started_at,
            "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "duration_seconds": total_time,
            "execution_mode": mode.value,
            "smoke_test": args.smoke_test,
            "allow_real_data": args.allow_real_data,
            "baseline_reproduction_only": mode == D8ExecutionMode.BASELINE_REPRODUCTION,
            "random_seed": args.random_seed,
            "target_arms": target_arms,
            "output_files": {
                "results_json": {"path": str(json_path), "sha256": _file_sha(json_path)},
                "comparison_csv": {"path": str(csv_path), "sha256": _file_sha(csv_path)},
                "audit_events_jsonl": {"path": str(audit_path), "sha256": _file_sha(audit_path)},
            },
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved execution manifest to {manifest_path}")
    except Exception as exc:
        import traceback
        fail_manifest = {
            "run_id": run_id,
            "status": "FAILED",
            "started_at_utc": started_at,
            "failed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "execution_mode": mode.value,
            "smoke_test": args.smoke_test,
            "allow_real_data": args.allow_real_data,
            "baseline_reproduction_only": mode == D8ExecutionMode.BASELINE_REPRODUCTION,
            "random_seed": args.random_seed,
            "target_arms": target_arms,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(fail_manifest, f, indent=2)
        raise

    # Print markdown table summary
    print("\n### Downstream Test Utility and Fairness Summary:\n")
    cols_to_display = [
        "arm_id",
        "condition",
        "test_auroc",
        "test_auprc",
        "test_positives_0_5",
        "test_max_dphi",
        "fairness_feasible",
        "termination_reason",
    ]
    print(df_cmp[cols_to_display].to_string(index=False))


if __name__ == "__main__":
    main()
