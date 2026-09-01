#!/usr/bin/env python3
"""Run the authorized HC-244 / Panel 26 grouped OOF robustness analysis."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
import subprocess
import sys
import time
from typing import Any, Iterable

import pandas as pd


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meps_fairness.crossfit import (  # noqa: E402
    ANALYSIS_SUBTYPE,
    FAIRNESS_LABEL,
    RESULT_LABEL,
    SEED,
    run_panel26_oof_robustness,
)


EXPECTED_BRANCH = "research/meps-hc252-longitudinal"
REQUIRED_BASELINE_ANCESTOR = "59b90eb3545ae4948475b8eb45f6f739d855b8bc"
EXPECTED_HC244_SHA256 = "5cf983c94fd9ed8d8377c9ad27bebd905c545327eca823a8c8412bf4e66eaa70"
HC244_RELATIVE_PATH = pathlib.Path("data/interim/meps/h244/h244.dta")
ARCHIVE_RELATIVE_PATH = pathlib.Path("archive/baseline_v0.3")

ARTIFACT_NAMES: tuple[str, ...] = (
    "oof_summary.json",
    "candidate_model_oof_metrics.csv",
    "fold_metrics.csv",
    "model_selection_stability.csv",
    "fairness_audit.csv",
    "weight_scale_sensitivity.csv",
    "calibration_bins.csv",
    "RUN_README.md",
    "execution_manifest.json",
)


def _git_output(repo_root: pathlib.Path, args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_baseline_ancestor(repo_root: pathlib.Path, baseline: str, head: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", baseline, head],
        cwd=repo_root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _archive_preflight(repo_root: pathlib.Path) -> dict[str, Any]:
    tracked_diff = subprocess.run(
        ["git", "diff", "--quiet", "--", str(ARCHIVE_RELATIVE_PATH)],
        cwd=repo_root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    staged_diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", str(ARCHIVE_RELATIVE_PATH)],
        cwd=repo_root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if tracked_diff.returncode != 0 or staged_diff.returncode != 0:
        raise RuntimeError("archive/baseline_v0.3 has tracked working-tree or index changes")
    return {
        "tracked_worktree_diff": False,
        "staged_diff": False,
        "status_snapshot": _git_output(
            repo_root,
            ["status", "--short", "--untracked-files=all", "--", str(ARCHIVE_RELATIVE_PATH)],
        ),
    }


def preflight(repo_root: pathlib.Path) -> dict[str, Any]:
    """Verify repository and authorized source conditions before Stata loading."""
    branch = _git_output(repo_root, ["branch", "--show-current"])
    head = _git_output(repo_root, ["rev-parse", "HEAD"])
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Unexpected branch: {branch!r}; expected {EXPECTED_BRANCH!r}")
    if not _is_baseline_ancestor(repo_root, REQUIRED_BASELINE_ANCESTOR, head):
        raise RuntimeError(
            "Required accepted ancestor is not present: "
            f"ancestor {REQUIRED_BASELINE_ANCESTOR}, HEAD {head}"
        )

    data_path = repo_root / HC244_RELATIVE_PATH
    if not data_path.is_file():
        raise FileNotFoundError(f"Authorized HC-244 data file not found: {data_path}")
    observed_sha256 = _sha256_file(data_path)
    if observed_sha256 != EXPECTED_HC244_SHA256:
        raise RuntimeError(
            "HC-244 SHA-256 mismatch; analysis stopped before loading data: "
            f"observed {observed_sha256}, expected {EXPECTED_HC244_SHA256}"
        )

    return {
        "branch": branch,
        "head": head,
        "required_baseline_ancestor": REQUIRED_BASELINE_ANCESTOR,
        "hc244_path": str(HC244_RELATIVE_PATH),
        "hc244_sha256": observed_sha256,
        "hc244_expected_sha256": EXPECTED_HC244_SHA256,
        "archive_baseline": _archive_preflight(repo_root),
    }


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot JSON-serialize {type(value)!r}")


def _write_json_no_clobber(path: pathlib.Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")


def _write_csv_no_clobber(
    path: pathlib.Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _create_run_dir(repo_root: pathlib.Path, seed: int) -> tuple[str, pathlib.Path]:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"panel26_oof_robustness_{timestamp}_{seed}"
    run_dir = repo_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def _write_run_readme(
    path: pathlib.Path,
    run_id: str,
    preflight_record: dict[str, Any],
    aggregate: dict[str, Any],
    runtime_seconds: float,
) -> None:
    summary = aggregate["oof_summary"]
    lines = [
        "# HC-244 / Panel 26 grouped OOF robustness",
        "",
        f"`{RESULT_LABEL}`",
        "",
        f"Analysis subtype: `{ANALYSIS_SUBTYPE}`",
        f"Fairness label: `{FAIRNESS_LABEL}`",
        "",
        "This is a five-fold grouped cross-fitted development analysis. It is "
        "not temporal validation, multi-panel evidence, population-pooled "
        "evidence, final fairness evidence, or publication-ready inference.",
        "",
        "## Frozen design",
        "",
        "- Source: HC-244 / Panel 26 only.",
        f"- Seed: `{summary['seed']}`; outer folds: `{summary['protocol']['outer_folds']}`.",
        "- Outer assignment unit: DUID; row-wise K-fold is not used.",
        "- Inner development split: grouped 75% train / 25% validation.",
        "- Preprocessing: fit on inner train only.",
        "- Selection: inner-validation weighted AUPRC, then weighted AUROC, then alphabetical model name.",
        "- Calibration and 10% capacity thresholds: inner validation only; one frozen threshold per outer fold.",
        "- Reported metrics: original LONGWT in both fitting modes.",
        "- No hyperparameter tuning and no participant-level prediction artifact.",
        "",
        "## Cohort and preflight",
        "",
        f"- Eligible records: `{summary['cohort']['eligible_record_count']}`; positive events: `{summary['cohort']['positive_events']}`; DUIDs: `{summary['cohort']['duid_count']}`.",
        f"- Branch: `{preflight_record['branch']}`; HEAD: `{preflight_record['head']}`.",
        f"- HC-244 SHA-256: `{preflight_record['hc244_sha256']}`.",
        f"- Runtime seconds: `{runtime_seconds:.6f}`.",
        f"- Run ID: `{run_id}`.",
        "",
        "## Mode summaries",
        "",
    ]
    for mode, mode_summary in summary["modes"].items():
        selected = mode_summary["selected_pipeline_oof"]
        lines.extend(
            [
                f"### `{mode}`",
                "",
                f"- Selected-pipeline raw AUROC/AUPRC: `{selected['raw']['weighted_auroc']:.8f}` / `{selected['raw']['weighted_auprc']:.8f}`.",
                f"- Selected-pipeline calibrated Brier: `{selected['calibrated']['weighted_brier']:.8f}`.",
                f"- Weighted selection rate: `{selected['operational']['selection_rate']:.8f}`.",
                f"- Model-selection frequency: `{mode_summary['model_selection_frequency']}`.",
                f"- Fairness endpoint: `{mode_summary['fairness']['primary_fairness_endpoint']}`.",
            ]
        )
        for dimension, dimension_summary in mode_summary["fairness"]["dimensions"].items():
            lines.extend(
                [
                    f"- `{dimension}` estimability: `{dimension_summary['endpoint']}`.",
                    f"  - Full max pairwise TPR gap: `{dimension_summary['full_max_pairwise_tpr_gap']}`.",
                    "  - Partial max pairwise TPR gap among unsuppressed groups: "
                    f"`{dimension_summary['partial_max_pairwise_tpr_gap_unsuppressed_groups']}`.",
                ]
            )
        lines.append("")
    sensitivity = summary["weight_scale_sensitivity"]
    lines.extend(
        [
            "## Weight-scale sensitivity",
            "",
            f"- Selected-pipeline OOF AUPRC absolute difference: `{sensitivity['selected_pipeline_oof_auprc_absolute_difference']:.8f}`.",
            f"- Selected-pipeline classification: `{sensitivity['selected_pipeline_classification']}`.",
            f"- Maximum candidate-model OOF AUPRC absolute difference: `{sensitivity['maximum_candidate_model_auprc_absolute_difference']:.8f}` (`{sensitivity['maximum_candidate_model_name']}`).",
            f"- Candidate-model classification: `{sensitivity['candidate_model_classification']}`.",
            "",
            "## Calibrated AUROC scope",
            "",
            f"- {summary['calibrated_auroc_interpretation_note']}",
            "",
            "The run directory is local and ignored by Git. It contains aggregate "
            "artifacts only. The archive baseline was not modified. No other MEPS "
            "panel or artifact was loaded by this run.",
            "",
        ]
    )
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _write_aggregate_artifacts(
    run_dir: pathlib.Path,
    aggregate: dict[str, Any],
    run_id: str,
    preflight_record: dict[str, Any],
    runtime_seconds: float,
) -> None:
    """Write the fixed aggregate artifact set; never write row-level arrays."""
    summary = aggregate["oof_summary"]
    summary["run_id"] = run_id
    summary["artifacts"] = list(ARTIFACT_NAMES)
    _write_json_no_clobber(run_dir / "oof_summary.json", summary)

    _write_csv_no_clobber(
        run_dir / "candidate_model_oof_metrics.csv",
        aggregate["candidate_model_oof_metrics"],
        [
            "result_label",
            "analysis_subtype",
            "weight_mode",
            "model_name",
            "weighted_auroc",
            "weighted_auprc",
            "weighted_brier",
            "weighted_observed_prevalence",
            "auprc_weighted_prevalence_ratio",
        ],
    )
    _write_csv_no_clobber(
        run_dir / "fold_metrics.csv",
        aggregate["fold_metrics"],
        [
            "weight_mode",
            "record_type",
            "outer_fold",
            "model_name",
            "selected_model",
            "outer_test_record_count",
            "outer_test_positive_events",
            "frozen_threshold",
            "raw_auroc",
            "raw_auprc",
            "raw_brier",
            "calibrated_auroc",
            "calibrated_auprc",
            "calibrated_brier",
            "tpr",
            "tnr",
            "fpr",
            "ppv",
            "f1",
            "selection_rate",
        ],
    )
    _write_csv_no_clobber(
        run_dir / "model_selection_stability.csv",
        aggregate["model_selection_stability"],
        [
            "result_label",
            "analysis_subtype",
            "weight_mode",
            "record_type",
            "outer_fold",
            "model_name",
            "selected",
            "folds_won",
            "validation_weighted_auprc",
            "validation_weighted_auroc",
            "outer_fold_raw_auprc",
            "selected_model_outer_fold_auprc",
        ],
    )
    _write_csv_no_clobber(
        run_dir / "fairness_audit.csv",
        aggregate["fairness_audit"],
        [
            "result_label",
            "analysis_subtype",
            "weight_mode",
            "fairness_label",
            "primary_fairness_endpoint",
            "dimension",
            "row_type",
            "group",
            "status",
            "sample_n",
            "positive_events",
            "negative_events",
            "kish_effective_n",
            "tpr",
            "fpr",
            "ppv",
            "selection_rate",
            "full_max_pairwise_tpr_gap",
            "partial_max_pairwise_tpr_gap_unsuppressed_groups",
            "dimension_endpoint",
            "suppression_reasons",
        ],
    )
    _write_csv_no_clobber(
        run_dir / "weight_scale_sensitivity.csv",
        aggregate["weight_scale_sensitivity"],
        [
            "result_label",
            "analysis_subtype",
            "metric",
            "model_name",
            "legacy_raw_longwt",
            "mean1_normalized_longwt",
            "absolute_difference",
            "changed",
            "classification",
        ],
    )
    _write_csv_no_clobber(
        run_dir / "calibration_bins.csv",
        aggregate["calibration_bins"],
        [
            "result_label",
            "analysis_subtype",
            "weight_mode",
            "probability_type",
            "bin",
            "lower_edge",
            "upper_edge",
            "sample_n",
            "weighted_n",
            "weighted_predicted_mean",
            "weighted_observed_event_prevalence",
        ],
    )
    _write_run_readme(
        run_dir / "RUN_README.md",
        run_id,
        preflight_record,
        aggregate,
        runtime_seconds,
    )


def execute(repo_root: pathlib.Path, seed: int) -> pathlib.Path:
    preflight_record = preflight(repo_root)
    data_path = repo_root / HC244_RELATIVE_PATH
    start = time.perf_counter()

    # This is the only authorized MEPS data load in this runner.
    raw_df = pd.read_stata(data_path, convert_categoricals=False)
    aggregate = run_panel26_oof_robustness(
        raw_df,
        seed=int(seed),
        source_path=str(HC244_RELATIVE_PATH),
        source_sha256=preflight_record["hc244_sha256"],
    )
    runtime_seconds = time.perf_counter() - start

    run_id, run_dir = _create_run_dir(repo_root, int(seed))
    _write_aggregate_artifacts(
        run_dir,
        aggregate,
        run_id,
        preflight_record,
        runtime_seconds,
    )

    output_hashes = {
        name: _sha256_file(run_dir / name)
        for name in ARTIFACT_NAMES
        if name != "execution_manifest.json"
    }
    post_archive = _archive_preflight(repo_root)
    if post_archive != preflight_record["archive_baseline"]:
        raise RuntimeError("archive/baseline_v0.3 changed during the run")
    manifest = {
        "result_label": RESULT_LABEL,
        "analysis_subtype": ANALYSIS_SUBTYPE,
        "fairness_label": FAIRNESS_LABEL,
        "run_id": run_id,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seed": int(seed),
        "command": "PYTHONPATH=src python scripts/run_panel26_oof_robustness.py --seed 20260828",
        "preflight": preflight_record,
        "postflight": {"archive_baseline": post_archive},
        "runtime_seconds": runtime_seconds,
        "output_hashes": output_hashes,
        "row_counts": {
            "eligible_cohort": aggregate["oof_summary"]["cohort"]["eligible_record_count"],
            "candidate_model_oof_metrics": len(aggregate["candidate_model_oof_metrics"]),
            "fold_metrics": len(aggregate["fold_metrics"]),
            "model_selection_stability": len(aggregate["model_selection_stability"]),
            "fairness_audit": len(aggregate["fairness_audit"]),
            "weight_scale_sensitivity": len(aggregate["weight_scale_sensitivity"]),
            "calibration_bins": len(aggregate["calibration_bins"]),
        },
        "data_boundary": "HC-244 / Panel 26 local development file only",
        "aggregate_artifacts_only": True,
        "row_level_predictions_written": False,
        "participant_level_prediction_files": [],
        "panel27_accessed": False,
        "hc217_accessed": False,
        "other_meps_artifacts_accessed": False,
        "no_clobber": True,
    }
    _write_json_no_clobber(run_dir / "execution_manifest.json", manifest)
    if {path.name for path in run_dir.iterdir()} != set(ARTIFACT_NAMES):
        raise RuntimeError("Run directory contains an unexpected non-aggregate artifact")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute the HC-244 / Panel 26 grouped OOF robustness analysis."
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--repo-root", type=pathlib.Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    run_dir = execute(args.repo_root.resolve(), int(args.seed))
    print(RESULT_LABEL)
    print(ANALYSIS_SUBTYPE)
    print(f"run_directory={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
