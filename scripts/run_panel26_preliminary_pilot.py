#!/usr/bin/env python3
"""Run the authorized HC-244 / Panel 26 preliminary development pilot."""

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

from meps_fairness.pilot import (  # noqa: E402
    EVIDENCE_SCOPE_NOTE,
    RESULT_LABEL,
    SEED,
    run_panel26_pilot,
)


EXPECTED_BRANCH = "research/meps-hc252-longitudinal"
REQUIRED_BASELINE_ANCESTOR = "bc43038b743e8c5e6ff7e990db012e31a3819282"
EXPECTED_HC244_SHA256 = "5cf983c94fd9ed8d8377c9ad27bebd905c545327eca823a8c8412bf4e66eaa70"
HC244_RELATIVE_PATH = pathlib.Path("data/interim/meps/h244/h244.dta")
ARCHIVE_RELATIVE_PATH = pathlib.Path("archive/baseline_v0.3")


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
    )
    staged_diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", str(ARCHIVE_RELATIVE_PATH)],
        cwd=repo_root,
        check=False,
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
    """Verify immutable repository/data conditions before Stata loading."""
    branch = _git_output(repo_root, ["branch", "--show-current"])
    head = _git_output(repo_root, ["rev-parse", "HEAD"])
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Unexpected branch: {branch!r}; expected {EXPECTED_BRANCH!r}")
    if not _is_baseline_ancestor(repo_root, REQUIRED_BASELINE_ANCESTOR, head):
        raise RuntimeError(
            "Required baseline is not an ancestor of current HEAD: "
            f"baseline {REQUIRED_BASELINE_ANCESTOR}, HEAD {head}"
        )

    data_path = repo_root / HC244_RELATIVE_PATH
    if not data_path.is_file():
        raise FileNotFoundError(f"HC-244 data file not found: {data_path}")
    observed_sha256 = _sha256_file(data_path)
    if observed_sha256 != EXPECTED_HC244_SHA256:
        raise RuntimeError(
            "HC-244 SHA-256 mismatch; pilot stopped before loading data: "
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


def _write_csv_no_clobber(path: pathlib.Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _artifact_sha256(path: pathlib.Path) -> str:
    return _sha256_file(path)


def _create_run_dir(repo_root: pathlib.Path, seed: int) -> tuple[str, pathlib.Path]:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"panel26_preliminary_{timestamp}_{seed}"
    run_dir = repo_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def _write_run_readme(
    path: pathlib.Path,
    run_id: str,
    preflight_record: dict[str, Any],
    result: dict[str, Any],
    runtime_seconds: float,
) -> None:
    summary = result["pilot_summary"]
    fairness = result["fairness_audit"]
    text = f"""# HC-244 / Panel 26 preliminary pilot

`{RESULT_LABEL}`

## Evidence scope

This is a single-panel development result from the HC-244 / Panel 26 local
development file. It is **{RESULT_LABEL}**.

The result is explicitly **{EVIDENCE_SCOPE_NOTE}**

## Frozen protocol

- Seed: `{summary['seed']}`
- Model selection: validation weighted AUPRC, then validation weighted AUROC, then alphabetical model name.
- Train: model fitting only.
- Validation: model selection, Platt calibration, and 10% weighted-capacity threshold freezing.
- Existing `cal` partition: untouched final development test.
- Refit on train plus validation: `False`.
- Final-test threshold recalculation: `False`.
- Selected model: `{summary['selected_model']}`.
- Frozen validation-derived threshold: `{summary['frozen_threshold']:.17g}`.
- Validation weighted selection rate: `{summary['validation_weighted_selection_rate']:.17g}`.
- Final-test weighted selection rate: `{summary['test_weighted_selection_rate']:.17g}`.

## Fairness boundary

Fairness uses the same frozen validation-derived threshold. Test-set capacity is
not recomputed. Primary dimensions are `RACETHX` and `SEX`, with suppression
requirements of n >= 100, positive events >= 20, negative events >= 20, and
Kish effective n >= 50. The primary endpoint is:

`{fairness['primary_fairness_endpoint']}`

## Preflight evidence

- Branch: `{preflight_record['branch']}`
- HEAD: `{preflight_record['head']}`
- HC-244 SHA-256: `{preflight_record['hc244_sha256']}`
- Archive tracked working-tree diff: `{preflight_record['archive_baseline']['tracked_worktree_diff']}`
- Archive staged diff: `{preflight_record['archive_baseline']['staged_diff']}`
- Runtime seconds: `{runtime_seconds:.6f}`
- Run ID: `{run_id}`

No Panel 27 data was loaded or used by this run. The run directory is local,
ignored by Git, and must not be treated as a publication or holdout result.
"""
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def execute(repo_root: pathlib.Path, seed: int) -> pathlib.Path:
    preflight_record = preflight(repo_root)
    data_path = repo_root / HC244_RELATIVE_PATH
    start = time.perf_counter()

    # This is the first point at which the authorized HC-244 file is loaded.
    raw_df = pd.read_stata(data_path, convert_categoricals=False)
    result = run_panel26_pilot(
        raw_df,
        seed=seed,
        source_path=str(HC244_RELATIVE_PATH),
        source_sha256=preflight_record["hc244_sha256"],
    )
    runtime_seconds = time.perf_counter() - start

    run_id, run_dir = _create_run_dir(repo_root, seed)
    artifact_names = [
        "pilot_summary.json",
        "cohort_summary.json",
        "model_validation_comparison.csv",
        "test_metrics.json",
        "fairness_audit.csv",
        "calibration_bins.csv",
        "RUN_README.md",
        "execution_manifest.json",
    ]
    result["pilot_summary"]["run_id"] = run_id
    result["pilot_summary"]["artifacts"] = artifact_names
    result["cohort_summary"]["run_id"] = run_id

    _write_json_no_clobber(run_dir / "pilot_summary.json", result["pilot_summary"])
    _write_json_no_clobber(run_dir / "cohort_summary.json", result["cohort_summary"])
    _write_json_no_clobber(run_dir / "test_metrics.json", result["test_metrics"])

    validation_rows = []
    for row in result["model_validation_comparison"]:
        validation_rows.append(
            {
                "result_label": RESULT_LABEL,
                "evidence_scope_note": EVIDENCE_SCOPE_NOTE,
                "model_name": row["model_name"],
                "validation_selection_rank": row["validation_selection_rank"],
                "selected": row["model_name"] == result["pilot_summary"]["selected_model"],
                "validation_weighted_auprc": row["validation_weighted_auprc"],
                "validation_weighted_auroc": row["validation_weighted_auroc"],
                "validation_weighted_brier_score": row["validation_weighted_brier_score"],
                "validation_predicted_mean_risk": row["validation_predicted_mean_risk"],
                "validation_weighted_observed_event_prevalence": row[
                    "validation_weighted_observed_event_prevalence"
                ],
                "configuration": json.dumps(row["configuration"], sort_keys=True),
            }
        )
    _write_csv_no_clobber(
        run_dir / "model_validation_comparison.csv",
        validation_rows,
        [
            "result_label",
            "evidence_scope_note",
            "model_name",
            "validation_selection_rank",
            "selected",
            "validation_weighted_auprc",
            "validation_weighted_auroc",
            "validation_weighted_brier_score",
            "validation_predicted_mean_risk",
            "validation_weighted_observed_event_prevalence",
            "configuration",
        ],
    )

    fairness_rows: list[dict[str, Any]] = []
    for dimension, dimension_record in result["fairness_audit"]["dimensions"].items():
        for group, group_record in dimension_record["subgroups"].items():
            fairness_rows.append(
                {
                    "result_label": RESULT_LABEL,
                    "evidence_scope_note": EVIDENCE_SCOPE_NOTE,
                    "dimension": dimension,
                    "group": group,
                    "threshold": result["fairness_audit"]["threshold"],
                    "status": group_record["status"],
                    "sample_n": group_record["sample_n"],
                    "positive_events": group_record["positive_events"],
                    "negative_events": group_record["negative_events"],
                    "kish_effective_n": group_record["kish_effective_n"],
                    "tpr": group_record["tpr"],
                    "fpr": group_record["fpr"],
                    "ppv": group_record["ppv"],
                    "selection_rate": group_record["selection_rate"],
                    "dimension_max_pairwise_tpr_gap": dimension_record["max_pairwise_tpr_gap"],
                    "primary_fairness_endpoint": result["fairness_audit"]["primary_fairness_endpoint"],
                }
            )
    _write_csv_no_clobber(
        run_dir / "fairness_audit.csv",
        fairness_rows,
        [
            "result_label",
            "evidence_scope_note",
            "dimension",
            "group",
            "threshold",
            "status",
            "sample_n",
            "positive_events",
            "negative_events",
            "kish_effective_n",
            "tpr",
            "fpr",
            "ppv",
            "selection_rate",
            "dimension_max_pairwise_tpr_gap",
            "primary_fairness_endpoint",
        ],
    )

    calibration_rows = []
    for row in result["calibration_bins"]:
        calibration_rows.append(
            {
                "result_label": RESULT_LABEL,
                "evidence_scope_note": EVIDENCE_SCOPE_NOTE,
                **row,
            }
        )
    _write_csv_no_clobber(
        run_dir / "calibration_bins.csv",
        calibration_rows,
        [
            "result_label",
            "evidence_scope_note",
            "partition",
            "partition_role",
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
        result,
        runtime_seconds,
    )

    artifact_hashes = {
        name: _artifact_sha256(run_dir / name)
        for name in artifact_names
        if name != "execution_manifest.json"
    }
    manifest = {
        "result_label": RESULT_LABEL,
        "scope_note": EVIDENCE_SCOPE_NOTE,
        "run_id": run_id,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seed": seed,
        "command": "python scripts/run_panel26_preliminary_pilot.py",
        "preflight": preflight_record,
        "runtime_seconds": runtime_seconds,
        "artifacts": artifact_hashes,
        "no_clobber": True,
        "data_boundary": "HC-244 / Panel 26 local development microdata only",
    }
    _write_json_no_clobber(run_dir / "execution_manifest.json", manifest)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute the HC-244 / Panel 26 preliminary single-panel development pilot."
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--repo-root", type=pathlib.Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    run_dir = execute(repo_root, seed=args.seed)
    print(f"{RESULT_LABEL}")
    print(f"run_directory={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
