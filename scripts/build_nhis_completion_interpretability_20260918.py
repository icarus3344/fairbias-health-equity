#!/usr/bin/env python3
"""Build descriptive completion-cost and trace-interpretability figures.

This script consumes only JSON completion metadata from the independently
verified completed80 backup.  It deliberately does not load policy.joblib,
prepared inputs, S/T outputs, or any prediction/performance artifact.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_JOBS = 80
EXPECTED_MANIFEST = "6607256f5efa027e0c954e7e0a0021d886074bf6dbfde0a52f1e5c5ba69ba0f0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def as_label(method: str, backbone: str) -> str:
    family = "BM-AE" if method == "FAIRBIAS_BM_AE" else "Joint"
    return f"{family} / {backbone}"


def read_verified_records(backup_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    control = backup_root / "control" / "audit.json"
    audit = json.loads(control.read_text())
    verification_path = backup_root / "control" / "verification.json"
    manifest_path = backup_root / "control" / "files_manifest.json"
    verification = json.loads(verification_path.read_text())
    files_manifest = json.loads(manifest_path.read_text())
    audit_sha = sha256_file(control)
    manifest_sha = sha256_file(manifest_path)
    if verification.get("audit_sha256") != audit_sha or files_manifest.get("audit_sha256") != audit_sha:
        raise RuntimeError("backup audit hash is inconsistent across verification records")
    if verification.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("backup file-manifest hash is inconsistent with verification record")
    if verification.get("status") != "LOCAL_ALL80_BACKUP_VERIFIED_STOP_READY":
        raise RuntimeError(f"unexpected backup verification status: {verification.get('status')!r}")
    if files_manifest.get("file_count") != 476 or files_manifest.get("total_bytes") != 1673362569:
        raise RuntimeError("unexpected verified backup file inventory")
    if audit.get("status") != "ALL_80_FC_ARTIFACTS_AND_RELOADS_VERIFIED":
        raise RuntimeError(f"unexpected audit status: {audit.get('status')!r}")
    if audit.get("S_T_evaluated") or audit.get("formal_benchmark_admission"):
        raise RuntimeError("audit indicates S/T evaluation or formal admission")
    if len(audit.get("jobs", [])) != EXPECTED_JOBS:
        raise RuntimeError("audit does not contain exactly 80 jobs")
    if audit.get("manifest_sha256") != EXPECTED_MANIFEST:
        raise RuntimeError("unexpected frozen manifest hash")

    result_paths = {
        path.parent.name: path
        for path in (backup_root / "snapshot").rglob("result.json")
    }
    if len(result_paths) != EXPECTED_JOBS:
        raise RuntimeError(f"expected 80 unique result directories, found {len(result_paths)}")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    result_hashes = 0
    model_hashes = 0
    for job in audit["jobs"]:
        job_id = job["job_id"]
        if job_id in seen:
            raise RuntimeError(f"duplicate job id in audit: {job_id}")
        seen.add(job_id)
        result_path = result_paths.get(job_id)
        if result_path is None:
            raise RuntimeError(f"missing result for audited job {job_id}")
        model_paths = sorted(result_path.parent.glob("*.joblib"))
        if len(model_paths) != 1:
            raise RuntimeError(f"expected one model artifact for {job_id}, found {len(model_paths)}")
        if sha256_file(result_path) != job["result_sha256"]:
            raise RuntimeError(f"result hash mismatch for {job_id}")
        result_hashes += 1
        if sha256_file(model_paths[0]) != job["model_sha256"]:
            raise RuntimeError(f"model hash mismatch for {job_id}")
        model_hashes += 1

        result = json.loads(result_path.read_text())
        if result.get("job_id") != job_id:
            raise RuntimeError(f"result job id mismatch for {job_id}")
        if result.get("manifest_sha256") != audit["manifest_sha256"]:
            raise RuntimeError(f"manifest mismatch for {job_id}")
        if result.get("formal_benchmark_admission") or result.get("S_T_evaluated"):
            raise RuntimeError(f"forbidden evaluation flag for {job_id}")
        if result.get("status") != "FC_COMPLETE_FEASIBLE" or not result.get("reload_exact"):
            raise RuntimeError(f"completion/reload status mismatch for {job_id}")
        loaded_modules = result.get("loaded_modules", {})
        # The audit records the frozen supervisor module set.  Completed
        # workers may additionally record an internal cache helper; require
        # every audited module to match while retaining that harmless superset.
        if any(loaded_modules.get(key) != value for key, value in audit["loaded_modules"].items()):
            raise RuntimeError(f"loaded-module provenance mismatch for {job_id}")
        config = result["config"]
        provenance = result["completion"]["model_provenance"]
        records.append(
            {
                "job_id": job_id,
                "method": config["method"],
                "backbone": config["backbone"],
                "arm": config["arm_id"],
                "seed": int(result["seed"]),
                "elapsed_seconds": float(result["elapsed_seconds"]),
                "adapter_elapsed_seconds": float(result["completion"]["elapsed_seconds"]),
                "search_attempts": len(result["completion"]["attempts"]),
                "search_attempt_elapsed_seconds": sum(float(a["elapsed_seconds"]) for a in result["completion"]["attempts"]),
                "calibration_rows": int(job["calibration_rows"]),
                "bm_commits": int(provenance["bm_commits"]),
                "ae_commits": int(provenance["ae_commits"]),
                "trace_count": int(provenance["trace_count"]),
                "bm_model_fits": int(provenance["bm_model_fits"]),
                "ae_model_fits": int(provenance["ae_model_fits"]),
                "utility_evaluations": int(provenance["utility_evaluations"]),
                "geometry_evaluations": int(provenance["geometry_evaluations"]),
                "bm_trace": provenance.get("bm_trace", []),
                "ae_audit": provenance.get("ae_audit", []),
                "changed_dict": provenance.get("changed_dict", {}),
            }
        )

    if seen != set(result_paths):
        raise RuntimeError("result inventory contains an unaudited job")
    audit_summary = {
        "audit_status": audit["status"],
        "manifest_sha256": audit["manifest_sha256"],
        "audited_jobs": len(records),
        "result_hashes_verified": result_hashes,
        "model_hashes_verified": model_hashes,
        "loaded_module_sets_verified": len(records),
        "calibration_rows_total": sum(row["calibration_rows"] for row in records),
        "calibration_rows_per_model_counts": dict(Counter(row["calibration_rows"] for row in records)),
        "formal_benchmark_admission": False,
        "S_T_evaluated": False,
        "backup_audit_sha256": audit_sha,
        "backup_files_manifest_sha256": manifest_sha,
    }
    return records, audit_summary


def feature_frequency(records: list[dict[str, Any]], backup_root: Path) -> dict[str, Any]:
    """Count models with any committed BM operation, never repeated steps."""
    runtime = backup_root / "snapshot" / "fairbias_completion_v2_20260917"
    source_manifest = json.loads((runtime / "control/manifest.json").read_text())
    paths = {"features": "configs/nhis/features.json", "arm_contract": "src/nhis_fairbias/benchmark/data_contracts.py"}
    for path in paths.values():
        if sha256_file(runtime / path) != source_manifest["sources"][path]:
            raise RuntimeError("feature-label or arm-schema hash mismatch")
    registry = json.loads((runtime / paths["features"]).read_text())
    constants = {}
    for node in ast.parse((runtime / paths["arm_contract"]).read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"PRIMARY_CORE_21", "DISABILITY_COMPONENTS_6"}:
                    constants[target.id] = ast.literal_eval(node.value)
    features = registry["feature_lists"]["primary_core_features"]
    if tuple(features) != constants["PRIMARY_CORE_21"]:
        raise RuntimeError("primary feature registry differs from frozen arm contract")
    excluded = set(constants["DISABILITY_COMPONENTS_6"])
    specs = {v["harmonized_name"]: v for v in registry["primary_core"].values()}
    short = {
        "agep_a": "Age (top-coded at 85)", "educp_a": "Educational attainment",
        "region": "Household region", "pcnt18uptc": "Adults in household",
        "pcntlt18tc": "Children in household", "ratcat_a": "Family income / poverty ratio",
        "empwrklsw1_a": "Worked last week", "empwrkft1_a": "Full-time / part-time work",
        "notcov_a": "Health insurance coverage", "phstat_a": "General health",
        "hypev_a": "Ever diagnosed hypertension", "chlev_a": "Ever diagnosed high cholesterol",
        "dibev_a": "Ever diagnosed diabetes", "asev_a": "Ever diagnosed asthma",
        "visiondf_a": "Difficulty seeing", "hearingdf_a": "Difficulty hearing",
        "diff_a": "Difficulty walking / climbing", "comdiff_a": "Difficulty communicating",
        "uppslfcr_a": "Difficulty with self-care", "cogmemdff_a": "Difficulty remembering / focusing",
        "usualpl_a": "Usual place for health care",
    }
    cells = [(f"arm_{a:03d}", b) for a in range(1, 5) for b in ("LR", "GBDT")]
    methods = {}
    for method in ("FAIRBIAS_BM_AE", "FAIRBIAS_JOINT"):
        matrix = {feature: [] for feature in features}
        for arm, backbone in cells:
            jobs = [r for r in records if (r["method"], r["arm"], r["backbone"]) == (method, arm, backbone)]
            if len(jobs) != 5 or {r["seed"] for r in jobs} != {0, 7, 19, 37, 73}:
                raise RuntimeError("feature frequency requires the complete registered five-seed set")
            touched = [{t["selected_feature"] for t in r["bm_trace"]} for r in jobs]
            if any(not values <= set(features) for values in touched):
                raise RuntimeError("trace feature absent from frozen registry")
            if arm == "arm_004" and any(values & excluded for values in touched):
                raise RuntimeError("Arm004 trace contains an excluded disability component")
            for feature in features:
                matrix[feature].append(None if arm == "arm_004" and feature in excluded else sum(feature in v for v in touched))
        methods[method] = matrix
    return {"definition": "number of five registered seed models with at least one committed BM category-merge, numeric-power or drop event for this feature; repeated steps count once",
            "interpretation": "descriptive seed stability of BM search operations, not causal influence or final-model feature importance",
            "denominator": 5, "columns": [{"arm": a, "backbone": b} for a, b in cells],
            "feature_order": features, "labels": {k: {"short_english": short[k], "official_description": specs[k]["official_description"]} for k in features},
            "arm004_excluded_features": sorted(excluded), "methods": methods,
            "sources": {k: {"path": str(runtime / v), "sha256": sha256_file(runtime / v)} for k, v in paths.items()}}


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_method: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    operation_counts: Counter[str] = Counter()
    semantic_counts: Counter[str] = Counter()
    operation_by_semantic: defaultdict[str, Counter[str]] = defaultdict(Counter)
    feature_counts: Counter[str] = Counter()
    audit_status: Counter[str] = Counter()
    audit_status_by_method: defaultdict[str, Counter[str]] = defaultdict(Counter)
    audit_decision_counts: Counter[str] = Counter()
    audit_decisions_by_condition: defaultdict[str, Counter[str]] = defaultdict(Counter)
    bm_events = 0
    accepted_events = 0
    for record in records:
        group = as_label(record["method"], record["backbone"])
        by_group[group].append(record)
        by_method[record["method"]].append(record)
        for trace in record["bm_trace"]:
            bm_events += 1
            semantic = str(trace.get("feature_semantic_type", "unknown"))
            semantic_counts[semantic] += 1
            transform = trace.get("accepted_transformation")
            if trace.get("dropped") is True and transform == "dropped":
                operation = "dropped"
            elif isinstance(transform, dict) and set(transform) == {"power"}:
                if trace.get("numerical_exponent") is None or semantic != "numerical":
                    raise RuntimeError("numeric power trace lacks its semantic/exponent contract")
                operation = "numeric_power"
            elif isinstance(transform, dict) and transform and "power" not in transform:
                if semantic != "categorical" or trace.get("categorical_merge_mapping") != transform:
                    raise RuntimeError("categorical merge trace lacks its mapping contract")
                operation = "categorical_merge"
            else:
                raise RuntimeError("completed80 contains an unclassified or uncommitted BM trace")
            # A dictionary contains a cumulative source->destination mapping,
            # not one operation per key. Count this committed BM event once.
            operation_counts[operation] += 1
            operation_by_semantic[semantic][operation] += 1
            accepted_events += 1
        for feature in record["changed_dict"]:
            feature_counts[feature] += 1
        for audit_row in record["ae_audit"]:
            status = str(audit_row.get("validity_status", "unknown"))
            audit_status[status] += 1
            audit_status_by_method[audit_row.get("condition", "unknown")][status] += 1
            if status == "VALID":
                decision = "COMMITTED" if audit_row.get("accepted") is True else "VALID_NOT_COMMITTED"
            elif status in {"FAIRNESS_CAP_EXCEEDED", "CYCLE_DETECTED"} and audit_row.get("accepted") is False:
                decision = status
            else:
                raise RuntimeError("unexpected AE validity/acceptance combination")
            audit_decision_counts[decision] += 1
            audit_decisions_by_condition[audit_row.get("condition", "unknown")][decision] += 1

    if bm_events != sum(operation_counts.values()) or bm_events != sum(r["bm_commits"] for r in records):
        raise RuntimeError("one-operation-per-committed-BM-event invariant failed")
    if audit_decision_counts["COMMITTED"] != sum(r["ae_commits"] for r in records):
        raise RuntimeError("AE acceptance count differs from committed-step provenance")

    def summaries(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for label, rows in sorted(groups.items()):
            costs = [row["elapsed_seconds"] for row in rows]
            out[label] = {
                "n": len(rows),
                "elapsed_seconds_total": round(sum(costs), 6),
                "elapsed_seconds_mean": round(sum(costs) / len(costs), 6),
                "elapsed_seconds_median": round(statistics.median(costs), 6),
                "elapsed_seconds_min": round(min(costs), 6),
                "elapsed_seconds_max": round(max(costs), 6),
                "trace_events_total": sum(row["trace_count"] for row in rows),
                "bm_model_fits_total": sum(row["bm_model_fits"] for row in rows),
                "ae_model_fits_total": sum(row["ae_model_fits"] for row in rows),
                "utility_evaluations_total": sum(row["utility_evaluations"] for row in rows),
                "geometry_evaluations_total": sum(row["geometry_evaluations"] for row in rows),
                "search_attempt_count": sum(row["search_attempts"] for row in rows),
                "bm_commits_total": sum(row["bm_commits"] for row in rows),
                "ae_commits_total": sum(row["ae_commits"] for row in rows),
            }
        return out

    return {
        "schema_version": "completed80_interpretability_v2",
        "trace_scope": "all BM and AE events in the final successful delegate; earlier replay delegates are not retained in model_provenance",
        "observed_search_attempt_counts": dict(Counter(row["search_attempts"] for row in records)),
        "elapsed_scope": "one worker process: adapter fit including internal replays, C calibration/prediction, policy save/reload and C reload prediction; excludes prepared loading, prior process attempts and later source verification",
        "elapsed_seconds_total": sum(row["elapsed_seconds"] for row in records),
        "elapsed_process_hours_total": sum(row["elapsed_seconds"] for row in records) / 3600,
        "groups": summaries(by_group),
        "methods": summaries(by_method),
        "bm_trace_events": bm_events,
        "bm_trace_accepted_events": accepted_events,
        "bm_trace_semantic_type_counts": dict(sorted(semantic_counts.items())),
        "bm_trace_operation_counts": dict(operation_counts.most_common()),
        "bm_trace_operation_by_semantic": {
            key: dict(value.most_common()) for key, value in sorted(operation_by_semantic.items())
        },
        "changed_feature_trace_counts": dict(feature_counts.most_common()),
        "ae_audit_status_counts": dict(audit_status.most_common()),
        "ae_audit_status_by_condition": {
            key: dict(value.most_common()) for key, value in sorted(audit_status_by_method.items())
        },
        "ae_audit_decision_counts": dict(audit_decision_counts),
        "ae_audit_decisions_by_condition": {key: dict(value) for key, value in sorted(audit_decisions_by_condition.items())},
    }


def make_figures(records: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = {"BM-AE": "#2a6fbb", "Joint": "#d95f02"}

    labels = sorted(summary["groups"])
    group_rows = defaultdict(list)
    for row in records:
        group_rows[as_label(row["method"], row["backbone"])].append(row)
    x = np.arange(len(labels))
    counts = [summary["groups"][label]["n"] for label in labels]
    colors = [palette[label.split(" /")[0]] for label in labels]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1.05, 1.65]})
    axes[0].bar(x, counts, color=colors, width=0.72, edgecolor="white")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylabel("Verified completed fits")
    axes[0].set_ylim(0, max(counts) * 1.28)
    axes[0].set_title("Completion coverage")
    for idx, count in enumerate(counts):
        axes[0].text(idx, count + 0.6, f"{count}/20", ha="center", va="bottom", fontsize=9)
    axes[0].text(
        0.02,
        0.04,
        "All 80 records: FC_COMPLETE_FEASIBLE",
        transform=axes[0].transAxes,
        fontsize=8,
        color="#555555",
    )
    box_data = [[row["elapsed_seconds"] / 60.0 for row in group_rows[label]] for label in labels]
    box = axes[1].boxplot(box_data, patch_artist=True, tick_labels=labels, widths=0.58, showmeans=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_ylabel("Worker process elapsed time (minutes)")
    axes[1].set_title("Per-model process elapsed time")
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.suptitle("Completed80 computation cost and coverage", y=1.02, fontsize=14, fontweight="bold")
    fig.text(
        0.01,
        -0.02,
        "Includes fitting and C policy checks in each completed process. Not CPU time, rental wall time, or historical retry cost.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(output_dir / f"completion_cost_coverage.{ext}", bbox_inches="tight")
    plt.close(fig)

    frequencies = summary.get("feature_frequency")
    if frequencies is not None:
        features = frequencies["feature_order"]
        columns = frequencies["columns"]
        fig, axes = plt.subplots(1, 2, figsize=(14.4, 9.6), sharey=True)
        cmap = plt.get_cmap("YlGnBu").copy()
        cmap.set_bad("#dddddd")
        for axis, method, label in zip(axes, ("FAIRBIAS_BM_AE", "FAIRBIAS_JOINT"), ("BM-AE", "Joint")):
            matrix = np.array([[np.nan if n is None else n / 5 for n in frequencies["methods"][method][f]] for f in features])
            graphic = axis.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
            axis.set_title(label, fontsize=14)
            axis.set_xticks(np.arange(8), [f"Arm{int(c['arm'][-3:])}\n{c['backbone']}" for c in columns], fontsize=9)
            axis.set_yticks(np.arange(len(features)), [frequencies["labels"][f]["short_english"] for f in features], fontsize=10)
            for i, feature in enumerate(features):
                for j, n in enumerate(frequencies["methods"][method][feature]):
                    axis.text(j, i, "N/A" if n is None else f"{n}/5", ha="center", va="center", fontsize=8,
                              color="#666666" if n is None else "white" if n >= 3 else "#222222")
            axis.set_xticks(np.arange(-.5, 8, 1), minor=True)
            axis.set_yticks(np.arange(-.5, len(features), 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=.5)
            axis.tick_params(which="minor", bottom=False, left=False)
        fig.suptitle("Features transformed by BM across five registered seeds", fontsize=16, y=.97, fontweight="bold")
        fig.subplots_adjust(left=.22, right=.90, bottom=.12, top=.91, wspace=.08)
        color_axis = fig.add_axes([.92, .23, .015, .55])
        colorbar = fig.colorbar(graphic, cax=color_axis, ticks=np.linspace(0, 1, 6))
        colorbar.ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
        colorbar.set_label("Models with at least one committed BM operation")
        fig.text(.02, .035, "One model is counted once per feature, across all its BM steps. Gray: feature excluded from Arm4.\nMerge, power and drop operations are combined. Seed stability describes the search; it is not a causal effect or final-model importance.", fontsize=9, color="#555555")
        for ext in ("png", "svg"):
            fig.savefig(output_dir / f"feature_operation_seed_frequency.{ext}", bbox_inches="tight")
        plt.close(fig)

    operations = ["categorical_merge", "numeric_power", "dropped"]
    operation_labels = ["Category merge", "Numeric power", "Feature dropped"]
    operation_values = [summary["bm_trace_operation_counts"][op] for op in operations]
    semantic = sorted(summary["bm_trace_operation_by_semantic"])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), gridspec_kw={"width_ratios": [1.15, 1.4]})
    y = np.arange(len(operations))
    left = np.zeros(len(operations))
    semantic_colors = {"categorical": "#7b3294", "numerical": "#008837", "unknown": "#999999"}
    for sem in semantic:
        vals = [summary["bm_trace_operation_by_semantic"].get(sem, {}).get(op, 0) for op in operations]
        axes[0].barh(y, vals, left=left, color=semantic_colors.get(sem, "#999999"), label=sem.capitalize())
        left += np.array(vals)
    axes[0].set_yticks(y, operation_labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Committed BM events (one event, one operation)")
    axes[0].set_title("BM transformations: 560 events")
    for index, value in enumerate(operation_values):
        axes[0].text(value + 4, index, str(value), va="center", fontsize=9)
    axes[0].set_xlim(0, max(operation_values) * 1.15)
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].grid(axis="x", color="#dddddd", linewidth=0.6)

    conditions = sorted(summary["ae_audit_decisions_by_condition"])
    statuses = ["COMMITTED", "VALID_NOT_COMMITTED", "FAIRNESS_CAP_EXCEEDED", "CYCLE_DETECTED"]
    status_labels = {"COMMITTED": "Committed", "VALID_NOT_COMMITTED": "Valid, not committed", "FAIRNESS_CAP_EXCEEDED": "Geometry cap exceeded", "CYCLE_DETECTED": "Cycle detected"}
    status_colors = {"COMMITTED": "#1b9e77", "VALID_NOT_COMMITTED": "#91cf60", "FAIRNESS_CAP_EXCEEDED": "#d95f02", "CYCLE_DETECTED": "#7570b3"}
    bottom = np.zeros(len(conditions))
    for status in statuses:
        vals = [summary["ae_audit_decisions_by_condition"].get(condition, {}).get(status, 0) for condition in conditions]
        axes[1].bar(conditions, vals, bottom=bottom, color=status_colors[status], label=status_labels[status])
        bottom += np.array(vals)
    axes[1].set_ylabel("AE audit records")
    axes[1].set_title("AE candidate audits: 37,515 records")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.suptitle("Transformations and candidate decisions in completed F/C searches", y=1.02, fontsize=14, fontweight="bold")
    fig.text(
        0.01,
        -0.02,
        "Final successful searches: all committed BM steps and recorded AE candidate audits. All 80 jobs had one search attempt.\nGeometry feasibility is not population fairness; these counts do not include low-level MDS iterations.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(output_dir / f"feature_trace_feasibility.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=Path("artifacts/nhis/completed80_backup_20260918"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/paper/figures_completion_v2_20260918"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("output directory already exists; preserve historical derived figures")
    records, audit_summary = read_verified_records(args.backup_root)
    summary = aggregate(records)
    summary["feature_frequency"] = feature_frequency(records, args.backup_root)
    summary["audit"] = audit_summary
    summary["source_scope"] = {
        "metadata_only": True,
        "policy_models_loaded": False,
        "prepared_inputs_loaded": False,
        "S_T_evaluated": False,
        "formal_benchmark_admission": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    json_dump(args.output_dir / "completion_interpretability_aggregates.json", summary)
    make_figures(records, summary, args.output_dir)
    print(json.dumps({"audit": audit_summary, "summary": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
