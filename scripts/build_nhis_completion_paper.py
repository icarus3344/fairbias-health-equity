#!/usr/bin/env python3
"""Build tables, figures, and a Chinese paper outline from a merged T summary.

The input is the JSON emitted by ``completion_evaluation.merge_completion``.
This is a reporting-only reader: it does not load policies, prediction arrays,
microdata, or selection code, and it never turns missing values into zero.

Examples
--------
    .venv311/bin/python scripts/build_nhis_completion_paper.py \
        --input path/to/summary_T_merged.json

    .venv311/bin/python scripts/build_nhis_completion_paper.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterable


ARMS = ("arm_001", "arm_002", "arm_003", "arm_004")
BACKBONES = ("LR", "GBDT")
FAIRBIAS_METHODS = ("FAIRBIAS_BM_AE", "FAIRBIAS_JOINT")
Q_ONLY_METHODS = ("EG_DP", "EG_EO", "TO_EO", "OXONFAIR_EO")
METHOD_ORDER = (
    "FAIRBIAS_BM_AE",
    "FAIRBIAS_JOINT",
    "UNMITIGATED",
    "FAIRBIAS_BM",
    "REWEIGHING",
    "LFR_RECONSTRUCTED",
    "EG_DP",
    "EG_EO",
    "TO_EO",
    "FRAPPE_EO",
    "OXONFAIR_EO",
)
METHOD_LABELS = {
    "FAIRBIAS_BM_AE": "FairBias BM+AE",
    "FAIRBIAS_JOINT": "FairBias Joint",
    "UNMITIGATED": "Unmitigated",
    "FAIRBIAS_BM": "FairBias BM",
    "REWEIGHING": "Reweighing",
    "LFR_RECONSTRUCTED": "LFR",
    "EG_DP": "EG-DP",
    "EG_EO": "EG-EO",
    "TO_EO": "Threshold EO",
    "FRAPPE_EO": "FRAPPE EO",
    "OXONFAIR_EO": "OxonFair EO",
}
METHOD_SHORT_LABELS = {
    "FAIRBIAS_BM_AE": "BM+AE",
    "FAIRBIAS_JOINT": "Joint",
    "UNMITIGATED": "Unmit",
    "FAIRBIAS_BM": "BM",
    "REWEIGHING": "RW",
    "LFR_RECONSTRUCTED": "LFR",
    "EG_DP": "EG-DP",
    "EG_EO": "EG-EO",
    "TO_EO": "TO-EO",
    "FRAPPE_EO": "FRAPPE",
    "OXONFAIR_EO": "Oxon",
}
ARM_SPECS = {
    "arm_001": {"protected_attribute": "SEX_A", "outcome": "MEDDL12M_A", "features": 21},
    "arm_002": {"protected_attribute": "HISPALLP_A", "outcome": "MEDDL12M_A", "features": 21},
    "arm_003": {"protected_attribute": "DISAB3_A", "outcome": "MEDDL12M_A", "features": 21},
    "arm_004": {"protected_attribute": "DISAB3_A", "outcome": "MEDDL12M_A", "features": 15},
}
METRIC_NAMES = OrderedDict(
    (
        ("balanced_accuracy", "BA"),
        ("average_precision", "AP"),
        ("auroc", "AUC"),
        ("brier", "Brier"),
        ("eo_gap", "EO"),
        ("dp_gap", "DP"),
    )
)
NUMERIC_KEYS = {"mean", "estimate", "lower", "upper", "ci_lower", "ci_upper"}
PAIR_INTERVAL_FIELDS = {
    "BA": {"nominal": "taylor_95", "family": "family_interval", "union": "union_family_interval"},
    "EO": {"nominal": "projection_95", "family": "projection_family", "union": "projection_union_family"},
}


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value: Any) -> float | None:
    return float(value) if is_number(value) else None


def metric_mean(value: Any) -> float | None:
    if is_number(value):
        return float(value)
    if not isinstance(value, dict):
        return None
    for key in ("mean", "estimate", "point_estimate"):
        result = number(value.get(key))
        if result is not None:
            return result
    return None


def interval(value: Any) -> tuple[float | None, float | None, str]:
    if not isinstance(value, dict):
        return None, None, "NOT_ESTIMABLE"
    low = number(value.get("lower", value.get("ci_lower")))
    high = number(value.get("upper", value.get("ci_upper")))
    status = str(value.get("status", "VALID" if low is not None and high is not None else "NOT_ESTIMABLE"))
    if low is None or high is None:
        return None, None, status
    return low, high, status


def select_interval(metric: Any, preference: Iterable[str]) -> tuple[float | None, float | None, str, str]:
    if not isinstance(metric, dict):
        return None, None, "NOT_ESTIMABLE", "none"
    for key in preference:
        if key in metric:
            low, high, status = interval(metric[key])
            if low is not None and high is not None and status == "VALID":
                return low, high, status, key
    for key in preference:
        if key in metric:
            low, high, status = interval(metric[key])
            return low, high, status, key
    return None, None, "NOT_ESTIMABLE", "none"


def selection_metric(row: dict[str, Any], metric_name: str) -> tuple[float | None, float | None, float | None, str, str]:
    metric = row.get(metric_name)
    value = metric_mean(metric)
    if metric_name == "balanced_accuracy":
        preference = ("taylor_95", "bootstrap_descriptive")
    elif metric_name in {"eo_gap", "dp_gap"}:
        preference = ("projection_95", "bootstrap_descriptive")
    else:
        preference = ("bootstrap_descriptive", "taylor_95")
    low, high, status, source = select_interval(metric, preference)
    return value, low, high, status, source


def pair_metric(row: dict[str, Any], metric_name: str) -> tuple[float | None, float | None, float | None, str, str]:
    metric = row.get("delta_" + metric_name)
    value = metric_mean(metric)
    if metric_name == "balanced_accuracy":
        preference = ("family_interval", "taylor_95", "bootstrap_descriptive", "union_family_interval")
    elif metric_name in {"eo_gap", "dp_gap"}:
        preference = ("projection_family", "projection_95", "projection_union_family")
    else:
        preference = ("bootstrap_descriptive", "taylor_95")
    low, high, status, source = select_interval(metric, preference)
    return value, low, high, status, source


def fmt(value: Any) -> str:
    if not is_number(value):
        return "NA"
    return f"{float(value):.10g}"


def display_method(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def row_label(row: dict[str, Any]) -> str:
    # The fixed-BM anchor is retained as a separate row in the full table,
    # while the scatter keeps the registered nine-control visual vocabulary.
    return display_method(str(row.get("method", "NA")))


def short_method(method: str) -> str:
    return METHOD_SHORT_LABELS.get(method, method)


def validate_summary(summary: dict[str, Any], strict: bool = True) -> None:
    if not isinstance(summary, dict):
        raise ValueError("MERGED_SUMMARY_MUST_BE_OBJECT")
    if strict:
        if summary.get("schema_version") != "completion_merged_summary_v1":
            raise ValueError("MERGED_SCHEMA_VERSION_REQUIRED")
        if summary.get("status") != "COMPLETE" or summary.get("known_T") is not True:
            raise ValueError("COMPLETE_KNOWN_T_SUMMARY_REQUIRED")
        if summary.get("family_sizes") != {"primary": 20, "secondary": 316, "union": 336}:
            raise ValueError("REGISTERED_FAMILY_SIZES_REQUIRED")
    if not isinstance(summary.get("selections"), list) or not isinstance(summary.get("paired_contrasts"), list):
        raise ValueError("SELECTIONS_AND_PAIRED_CONTRASTS_REQUIRED")
    expected_selections = 96 if strict else 1
    expected_pairs = 168 if strict else 1
    if strict and len(summary["selections"]) != expected_selections:
        raise ValueError(f"EXPECTED_{expected_selections}_SELECTION_ROWS")
    if strict and len(summary["paired_contrasts"]) != expected_pairs:
        raise ValueError(f"EXPECTED_{expected_pairs}_PAIR_ROWS")
    if not strict and (len(summary["selections"]) < expected_selections or len(summary["paired_contrasts"]) < expected_pairs):
        raise ValueError("MINI_SCHEMA_NEEDS_AT_LEAST_ONE_SELECTION_AND_PAIR")
    selection_ids = {row.get("selection_id") for row in summary["selections"]}
    if None in selection_ids or len(selection_ids) != len(summary["selections"]):
        raise ValueError("SELECTION_ID_COVERAGE_INVALID")
    pair_ids = {row.get("contrast_id") for row in summary["paired_contrasts"]}
    if None in pair_ids or len(pair_ids) != len(summary["paired_contrasts"]):
        raise ValueError("CONTRAST_ID_COVERAGE_INVALID")
    if strict and ({row.get("arm_id") for row in summary["selections"]} != set(ARMS)):
        raise ValueError("FOUR_ARM_COVERAGE_REQUIRED")
    if strict:
        family_counts = {family: sum(1 for row in summary["paired_contrasts"] if row.get("family") == family) for family in ("primary", "secondary")}
        if family_counts != {"primary": 10, "secondary": 158}:
            raise ValueError("REGISTERED_PAIR_FAMILY_COUNTS_REQUIRED")


def condition_row(row: dict[str, Any], known_t: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "arm_id": row.get("arm_id", "NA"),
        "protected_attribute": ARM_SPECS.get(row.get("arm_id"), {}).get("protected_attribute", "NA"),
        "backbone": row.get("backbone", "NA"),
        "method": row.get("method", "NA"),
        "method_label": row_label(row),
        "origin": row.get("origin", "NA"),
        "selection_id": row.get("selection_id", "NA"),
        "training_weighted": row.get("training_weighted", "NA"),
        "tau": row.get("tau", "NA"),
        "seed_count": row.get("seed_count", len(row.get("model_ids", row.get("frozen_seed_models", [])))),
        "sample_n": row.get("sample_n", "NA"),
        "design_n": row.get("design_n", "NA"),
        "df": row.get("df", "NA"),
        "S_status": row.get("status", row.get("S_status", row.get("selection_status", "NA"))),
        "T_evaluation_status": row.get("evaluation_status", "NA"),
        "known_T": known_t,
        "retrospective_T": "knownT retrospective",
    }
    for name, short in METRIC_NAMES.items():
        raw_metric = row.get(name)
        value, low, high, status, source = selection_metric(row, name)
        result[f"T_{short}"] = value
        result[f"T_{short}_status"] = raw_metric.get("status", "NA") if isinstance(raw_metric, dict) else "NA"
        result[f"T_{short}_seed_sd"] = raw_metric.get("seed_sd", "NA") if isinstance(raw_metric, dict) else "NA"
        result[f"T_{short}_ci_low"] = low
        result[f"T_{short}_ci_high"] = high
        result[f"T_{short}_ci_status"] = status
        result[f"T_{short}_ci_source"] = source
        unweighted = row.get("unweighted_" + name)
        result[f"unweighted_{short}_mean"] = metric_mean(unweighted)
        result[f"unweighted_{short}_seed_sd"] = unweighted.get("seed_sd", "NA") if isinstance(unweighted, dict) else "NA"
    return result


def pair_row(row: dict[str, Any], selections_by_id: dict[str, dict[str, Any]], known_t: bool) -> dict[str, Any]:
    ref = selections_by_id.get(row.get("reference_selection_id"), {})
    base = selections_by_id.get(row.get("comparison_selection_id"), {})
    result: dict[str, Any] = {
        "contrast_id": row.get("contrast_id", "NA"),
        "arm_id": row.get("arm_id", ref.get("arm_id", "NA")),
        "backbone": row.get("backbone", ref.get("backbone", "NA")),
        "family": row.get("family", "NA"),
        "kind": row.get("kind", "NA"),
        "reference_method": row.get("reference_method", ref.get("method", "NA")),
        "comparison_method": row.get("comparison_method", base.get("method", "NA")),
        "reference_selection_id": row.get("reference_selection_id", "NA"),
        "comparison_selection_id": row.get("comparison_selection_id", "NA"),
        "difference_direction": row.get("difference_direction", "reference_minus_comparison"),
        "family_endpoint_slots": row.get("family_endpoint_slots", "NA"),
        "union_endpoint_slots": row.get("union_endpoint_slots", "NA"),
        "reference_seed_count": len(ref.get("model_ids", ref.get("frozen_seed_models", []))),
        "comparison_seed_count": len(base.get("model_ids", base.get("frozen_seed_models", []))),
        "evaluation_status": row.get("evaluation_status", "NA"),
        "known_T": known_t,
        "retrospective_T": "knownT retrospective",
    }
    for name, short in METRIC_NAMES.items():
        value, low, high, status, source = pair_metric(row, name)
        result[f"delta_{short}"] = value
        result[f"delta_{short}_ci_low"] = low
        result[f"delta_{short}_ci_high"] = high
        result[f"delta_{short}_ci_status"] = status
        result[f"delta_{short}_ci_source"] = source
        if short in PAIR_INTERVAL_FIELDS:
            metric = row.get("delta_" + name)
            for label, field_name in PAIR_INTERVAL_FIELDS[short].items():
                interval_value = metric.get(field_name) if isinstance(metric, dict) else None
                interval_low, interval_high, interval_status = interval(interval_value)
                result[f"delta_{short}_ci_{label}_low"] = interval_low
                result[f"delta_{short}_ci_{label}_high"] = interval_high
                result[f"delta_{short}_ci_{label}_status"] = interval_status
                result[f"delta_{short}_ci_{label}_source"] = field_name if interval_value is not None else "none"
        else:
            for label in ("nominal", "family", "union"):
                result[f"delta_{short}_ci_{label}_low"] = None
                result[f"delta_{short}_ci_{label}_high"] = None
                result[f"delta_{short}_ci_{label}_status"] = "NOT_ESTIMABLE"
                result[f"delta_{short}_ci_{label}_source"] = "none"
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            def csv_value(value: Any) -> Any:
                if isinstance(value, bool):
                    return str(value).lower()
                if isinstance(value, (int, float)):
                    return fmt(value)
                return "NA" if value is None else value
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def plot_figures(condition_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]], output: Path, strict: bool) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    def save_figure(fig: Any, path: Path) -> None:
        # Suppress Matplotlib's current-time SVG metadata so repeated exports
        # have stable hashes for review and archival manifests.
        if path.suffix.lower() == ".svg":
            fig.savefig(path, metadata={"Date": None}, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 300, "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    method_colors = {
        "FAIRBIAS_BM_AE": "#1b9e77",
        "FAIRBIAS_JOINT": "#d95f02",
        "UNMITIGATED": "#666666",
        "FAIRBIAS_BM": "#7570b3",
        "REWEIGHING": "#e7298a",
        "LFR_RECONSTRUCTED": "#66a61e",
        "EG_DP": "#e6ab02",
        "EG_EO": "#a6761d",
        "TO_EO": "#1f78b4",
        "FRAPPE_EO": "#b2df8a",
        "OXONFAIR_EO": "#fb9a99",
    }
    markers = {"FAIRBIAS_BM_AE": "*", "FAIRBIAS_JOINT": "D"}

    method_markers = {
        "FAIRBIAS_BM_AE": "*", "FAIRBIAS_JOINT": "D", "UNMITIGATED": "o",
        "FAIRBIAS_BM": "P", "REWEIGHING": "s", "LFR_RECONSTRUCTED": "^",
        "EG_DP": "v", "EG_EO": "X", "TO_EO": "d", "FRAPPE_EO": "h", "OXONFAIR_EO": "8",
    }
    handles = [plt.Line2D([0], [0], marker=method_markers.get(method, "o"), color="w", label=display_method(method),
                           markerfacecolor=method_colors.get(method, "#333333"),
                           markeredgecolor="black" if method in FAIRBIAS_METHODS else "white",
                           markersize=8 if method in FAIRBIAS_METHODS else 6)
               for method in METHOD_ORDER]

    # The primary figure excludes fixed-BM anchors, which remain in the complete
    # tables.  This avoids presenting two BM origins as one visual point.
    visual_rows = [r for r in condition_rows if r.get("origin") in {"parent_tau10", "adaptive_completed80"}]

    def make_scatter(y_metric: str, filename: str, title: str) -> None:
        fig, axes = plt.subplots(4, 2, figsize=(10.8, 12.2), sharex=False, sharey=False)
        for arm_i, arm in enumerate(ARMS):
            for backbone_i, backbone in enumerate(BACKBONES):
                ax = axes[arm_i, backbone_i]
                subset = [r for r in visual_rows if r["arm_id"] == arm and r["backbone"] == backbone]
                missing = sum(1 for r in subset if not is_number(r.get("T_BA")) or not is_number(r.get(f"T_{y_metric}")))
                for row in sorted(subset, key=lambda x: METHOD_ORDER.index(x["method"]) if x["method"] in METHOD_ORDER else 999):
                    x, y, method = row.get("T_BA"), row.get(f"T_{y_metric}"), row["method"]
                    if not is_number(x) or not is_number(y):
                        continue
                    ax.scatter(x, y, s=75 if method in FAIRBIAS_METHODS else 34,
                               marker=method_markers.get(method, "o"), color=method_colors.get(method, "#333333"),
                               edgecolor="black" if method in FAIRBIAS_METHODS else "white",
                               linewidth=0.65, alpha=0.92, zorder=3)
                # Annotate FairBias only.  Exact coordinate matches get one label;
                # controls use the stable legend instead of crowded point labels.
                fairbias_points: dict[tuple[float, float], list[str]] = defaultdict(list)
                for row in subset:
                    x, y, method = row.get("T_BA"), row.get(f"T_{y_metric}"), row["method"]
                    if method in FAIRBIAS_METHODS and is_number(x) and is_number(y):
                        fairbias_points[(round(float(x), 12), round(float(y), 12))].append(method)
                for (x, y), methods_at_point in fairbias_points.items():
                    method_set = set(methods_at_point)
                    if method_set == set(FAIRBIAS_METHODS):
                        label = "Joint / BM+AE"
                        offset = (4, 4)
                    else:
                        label = " / ".join(short_method(m) for m in FAIRBIAS_METHODS if m in method_set)
                        offset = (4, -12) if "FAIRBIAS_BM_AE" in method_set else (4, 4)
                    ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=6.5,
                                fontweight="bold", color="#333333")
                ax.set_title(f"{arm} · {backbone}")
                ax.grid(color="#dddddd", linewidth=0.5)
                ax.text(0.02, 0.97, f"{missing} NA omitted" if missing else "All available",
                        transform=ax.transAxes, va="top", fontsize=6.5, color="#666666")
                ax.set_xlabel("T balanced accuracy")
                ax.set_ylabel(f"T {y_metric} gap")
        fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=7,
                   bbox_to_anchor=(0.5, 0.015))
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
        fig.text(0.01, 0.002, "Descriptive knownT retrospective; parent_fixed_BM anchors are retained in tables, not this primary visual. NA omitted, never replaced by zero.", fontsize=7.5, color="#555555")
        fig.tight_layout(rect=(0, 0.09, 1, 0.96))
        for ext in ("png", "svg"):
            path = output / f"{filename}.{ext}"
            save_figure(fig, path)
            created.append(path)
        plt.close(fig)

    make_scatter("EO", "prediction_fairness_scatter", "Prediction and fairness on NHIS 2024")
    make_scatter("DP", "prediction_fairness_scatter_dp", "Prediction and fairness on NHIS 2024 · DP gap")

    pair_subset = [r for r in pair_rows if r["kind"] == "joint_vs_sequential" or (r["reference_method"] == "FAIRBIAS_JOINT" and r["comparison_method"] == "FAIRBIAS_BM_AE")]
    pair_subset.sort(key=lambda r: (ARMS.index(r["arm_id"]) if r["arm_id"] in ARMS else 999, BACKBONES.index(r["backbone"]) if r["backbone"] in BACKBONES else 999))
    arm_labels = {"arm_001": "Sex", "arm_002": "Race", "arm_003": "Disability", "arm_004": "Reduced features"}
    labels = [f"{arm_labels.get(r['arm_id'], r['arm_id'])}\n{r['backbone']}" for r in pair_subset]
    y = np.arange(len(pair_subset))
    # The frozen paired registry provides formal BA/EO intervals only.  Keep
    # DP columns in the complete CSV as NA and omit a DP forest/panel entirely.
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.6), sharey=True)
    for ax, short, title in zip(axes, ("BA", "EO"), ("Δ BA", "Δ EO gap")):
        values, lows, highs = [], [], []
        for row in pair_subset:
            value, low, high = row.get(f"delta_{short}"), row.get(f"delta_{short}_ci_low"), row.get(f"delta_{short}_ci_high")
            values.append(value if is_number(value) else np.nan)
            lows.append(low if is_number(low) else np.nan)
            highs.append(high if is_number(high) else np.nan)
        values_np, lows_np, highs_np = np.asarray(values, dtype=float), np.asarray(lows, dtype=float), np.asarray(highs, dtype=float)
        for idx in range(len(pair_subset)):
            if np.isfinite(values_np[idx]) and np.isfinite(lows_np[idx]) and np.isfinite(highs_np[idx]):
                ax.errorbar(values_np[idx], idx, xerr=[[values_np[idx] - lows_np[idx]], [highs_np[idx] - values_np[idx]]], fmt="o", color="#2c7fb8", capsize=3)
            else:
                ax.plot(0, idx, marker="x", color="#999999", markersize=7)
        ax.axvline(0, color="#555555", linewidth=0.8)
        ax.set_title(title)
        ax.set_yticks(y, labels)
        ax.grid(axis="x", color="#dddddd", linewidth=0.5)
        ax.set_xlabel("Joint − BM+AE")
    axes[0].set_ylabel("Arm · backbone")
    fig.suptitle("Joint − BM+AE paired differences with recorded CIs", fontsize=13, fontweight="bold", y=0.995)
    fig.text(0.01, 0.005, "Recorded interval fields; family-adjusted fairness intervals use the secondary family (316 endpoints). Wide EO intervals are uncertainty, not improvement. DP is unavailable.", fontsize=7.8, color="#555555")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    for ext in ("png", "svg"):
        path = output / f"joint_minus_bmae_paired_ci.{ext}"
        save_figure(fig, path)
        created.append(path)
    plt.close(fig)
    return created


def write_outline(output: Path, condition_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]], strict: bool) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    missing = sum(1 for row in condition_rows for key in ("T_BA", "T_AP", "T_AUC", "T_Brier", "T_EO", "T_DP") if not is_number(row.get(key)))
    probability_inapplicable = sum(
        1 for row in condition_rows
        if row.get("method") in Q_ONLY_METHODS
        for key in ("T_AP", "T_AUC", "T_Brier")
        if not is_number(row.get(key))
    )
    other_missing = missing - probability_inapplicable
    primary = sum(1 for row in pair_rows if row.get("family") == "primary")
    secondary = sum(1 for row in pair_rows if row.get("family") == "secondary")
    not_supported = sum(1 for row in condition_rows if str(row.get("S_status")).upper() == "NOT_SUPPORTED")
    no_valid_fixed_ablation = sum(1 for row in condition_rows if str(row.get("S_status")).upper() == "NO_VALID_FIXED_ABLATION")
    status_missing = not_supported + no_valid_fixed_ablation
    fixed_bm_missing = sum(1 for row in condition_rows if str(row.get("S_status")).upper() == "NO_VALID_FIXED_ABLATION")
    fairbias_groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in condition_rows:
        if row.get("method") in FAIRBIAS_METHODS:
            fairbias_groups[(row["arm_id"], row["backbone"])][row["method"]] = row
    equal_all = 0
    equal_denominator = 0
    for group in fairbias_groups.values():
        if not all(method in group for method in FAIRBIAS_METHODS):
            continue
        equal_denominator += 1
        if all(
            is_number(group["FAIRBIAS_JOINT"].get("T_" + metric))
            and is_number(group["FAIRBIAS_BM_AE"].get("T_" + metric))
            and abs(group["FAIRBIAS_JOINT"]["T_" + metric] - group["FAIRBIAS_BM_AE"]["T_" + metric]) < 1e-12
            for metric in ("BA", "AP", "AUC", "Brier", "EO", "DP")
        ):
            equal_all += 1
    outline = f"""# FairBias completion results：中文论文脉络（knownT retrospective）

> 本文档由 `build_nhis_completion_paper.py` 根据 merged completion summary 自动生成。当前版本只负责整理已冻结摘要中的结果字段，不重新评价 T、不改变选择政策、不加载微观记录，也不预设任何方法优越性。

## 摘要写作边界

研究问题是：在 NHIS 重复横断面调查中，围绕医疗可及性相关的延迟照护结局，公平表示学习是否在跨年迁移、复杂调查推断和可解释诊断之间保持可审计性。本批评价的主结局为本地变量契约中的 `MEDDL12M_A`（过去 12 个月因费用延迟医疗照护的二元结局）；`MEDNG12M_A` 仅为注册中的敏感性结局，本批 completion 未评价该结局。结果只能表述为固定模型在已知 T 上的回顾性计算扩展，不能写成临床获益或真实部署效果。

## 1. 临床问题与研究动机

医疗可及性研究需要同时关注预测识别和不同群体的错误分布。四个研究臂使用 `SEX_A`、`HISPALLP_A` 或 `DISAB3_A` 作为受保护维度，围绕同一主要结局比较普通预测、FairBias BM+AE、FairBias Joint 与已注册对照。这里的“临床问题”是识别费用相关延迟照护风险及其群体差异，不是对个体进行诊断或给出治疗建议。

## 2. 数据、队列与跨年验证

本地研究契约定义 2022 年为开发训练、2023 年为开发验证、2024 年为冻结测试；F/C/S/T 的职责必须在正文和图中分开。主要臂使用 21 个特征，`arm_004` 使用 15 个特征。合并摘要包含 {len(condition_rows)} 条条件记录和 {len(pair_rows)} 条完整配对记录；本次输出保留所有分母。状态上共有 {status_missing} 条未完成条件，其中 {not_supported} 条为 `NOT_SUPPORTED`、{no_valid_fixed_ablation} 条为 `NO_VALID_FIXED_ABLATION`；后者即固定 BM anchor 无 T 可估计指标的 {fixed_bm_missing} 条，不能与前者重复相加。指标格共有 {missing} 个 NA，其中 {probability_inapplicable} 个来自 q-only 对照的 AP/AUC/Brier 概率语义不适用，其余 {other_missing} 个来自不可支持、固定 anchor 或其他不可估计状态；均未用 0 填补。八个 Joint/BM+AE 条件中，有 {equal_all}/{equal_denominator} 个条件的六项点估计完全相同；这只是结果重合的描述，不是优越性证据。

`known_T` 仅说明 T 已在历史工作流中可见；图表统一标注 `knownT retrospective`。因此，这些结果可以描述为冻结策略条件下的回顾性域估计，不能被包装成严格的前瞻性外部验证。

## 3. 方法、对照与可解释机制

FairBias BM+AE 与 Joint 的方法名称、骨干、研究臂和 origin 直接沿用输入摘要。图中保留 FairBias BM+AE、Joint 与九类注册对照的统一视觉词汇；固定 BM anchor 若与 tau=.10 的 BM 使用相同方法标签，仍在完整 CSV 中按 origin 分行，不在图上静默合并为新的方法。操作痕迹和模型输出应分开解释：几何/表示变换记录说明算法执行了哪些候选操作，不能直接证明统计公平性改善；输出 EO/DP 是群体指标，也不能证明特征变换具有因果效果。正文可把“发生了什么操作”与“最终群体指标如何变化”并列展示，但不能把两者写成机制已经得到因果验证。

## 4. 评价与 survey 推断

条件表报告 T BA、AP、AUC、Brier、EO 和 DP，以及摘要中实际存在的 CI、seed SD、sample/design/df 和 unweighted 均值。配对表报告参考方法减比较方法的方向，并把 BA/EO 的 nominal、family、union 三套区间分别展开；Joint−BM+AE 图使用记录的区间字段。主要推断族规模为 BA/EO 的 primary 20、secondary 316、union 336 个端点；Family EO 区间可能很宽，宽区间应被写成不确定性，而不能称为改善或劣化。若字段为 `NA` 或 `NOT_ESTIMABLE`，正文必须保留该状态和原分母。

## 5. 结果组织

建议按以下顺序叙述：先报告 96 条条件和 168 条配对的完整覆盖及缺失；再展示四臂×双骨干的 BA–EO/DP 散点；随后报告 Joint−BM+AE 的 BA/EO 配对差值与 CI；最后分开展示跨年、骨干和对照方法的敏感性。散点图只在 BA 与公平性指标均可用时绘点，NA 不作为零值参与视觉比较。部分注册对照在某些臂/骨干上具有更高 BA 或更小公平性 gap，且不同指标方向不一致；因此正文应逐条件报告，不能写成 FairBias 总体优越。DP 的完整配对 CI 不在冻结摘要中提供，配对森林图不展示 DP，CSV 中保留为 NA。

当前输出文件：

- [96 条件完整表](completion_conditions_96.csv)
- [168 配对完整表](completion_pairs_168.csv)
- [8 条新条件 Joint/BM+AE 汇总表](completion_fairbias_8_conditions.csv)
- [预测–EO 公平性主散点图](prediction_fairness_scatter.svg)
- [预测–DP 公平性补充散点图](prediction_fairness_scatter_dp.svg)
- [Joint−BM+AE 配对 CI 图](joint_minus_bmae_paired_ci.svg)

## 6. 讨论重点

讨论应回答：哪些臂和骨干在固定 T 上仍可估计，哪些指标区间受 survey 设计或 Family 校正影响；FairBias 操作是否伴随预测代价；Joint 与 BM+AE 的差值是否跨臂一致。这里的“代价”是结果层面的描述，不是重新选择模型的依据，也不是事后改变注册对照的理由。

## 7. 限制与投稿缺口

主要限制包括单一 NHIS 来源、自报/回顾性结局、受保护组支持度差异、固定策略条件推断、known-T retrospective 边界、Family EO 区间宽、对照方法 origin 与 FairBias completion 的历史预算不同，以及缺失/不可估计指标。投稿前仍需完成完整的 cohort/label 叙述、survey 设计和自由度说明、对本扩展事先冻结端点且已知历史 T 后进行回顾性扩展的措辞审校、缺失分母表、可解释操作的非因果措辞审校，以及独立审阅图表是否把描述性差异误读为优越性。

## 8. 证据与复核

本输出仅依赖 `--input` 指向的 merged summary；脚本不会读取其他 T 结果，也不会生成微观记录。真实摘要运行时应在附录记录输入 SHA-256、schema 版本、96/168 行数、NA 计数、各 CI 来源和输出文件 SHA-256。
"""
    path = output / "FAIRBIAS_COMPLETION_RESULTS_OUTLINE_20260918.md"
    path.write_text(outline, encoding="utf-8")
    return path


def build(input_path: Path, output: Path, strict: bool = True) -> dict[str, Any]:
    summary = json.loads(input_path.read_text(encoding="utf-8"))
    validate_summary(summary, strict=strict)
    known_t = bool(summary.get("known_T", True))
    selections = [condition_row(row, known_t) for row in summary["selections"]]
    by_id = {row.get("selection_id"): row for row in summary["selections"]}
    pairs = [pair_row(row, by_id, known_t) for row in summary["paired_contrasts"]]
    write_csv(output / "completion_conditions_96.csv", selections)
    write_csv(output / "completion_pairs_168.csv", pairs)

    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in selections:
        if row["method"] in FAIRBIAS_METHODS:
            grouped[(row["arm_id"], row["backbone"])][row["method"]] = row
    summary_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for backbone in BACKBONES:
            base = {"arm_id": arm, "backbone": backbone, "known_T": known_t, "retrospective_T": "knownT retrospective"}
            for method in FAIRBIAS_METHODS:
                row = grouped.get((arm, backbone), {}).get(method, {})
                prefix = "Joint" if method == "FAIRBIAS_JOINT" else "BM_AE"
                base[prefix + "_selection_id"] = row.get("selection_id")
                base[prefix + "_seed_count"] = row.get("seed_count")
                for short in ("BA", "AP", "AUC", "Brier", "EO", "DP"):
                    for suffix in ("", "_ci_low", "_ci_high"):
                        base[f"{prefix}_{short}{suffix}"] = row.get(f"T_{short}{suffix}")
            summary_rows.append(base)
    write_csv(output / "completion_fairbias_8_conditions.csv", summary_rows)
    created = plot_figures(selections, pairs, output, strict)
    outline = write_outline(output, selections, pairs, strict)
    metadata = {
        "input": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "schema_version": summary.get("schema_version", "synthetic_mini"),
        "export_version": summary.get("export_version", "NA"),
        "known_T": known_t,
        "retrospective_label": "knownT retrospective",
        "selection_rows": len(selections),
        "pair_rows": len(pairs),
        "primary_pair_rows": sum(1 for row in pairs if row.get("family") == "primary"),
        "secondary_pair_rows": sum(1 for row in pairs if row.get("family") == "secondary"),
        "family_sizes": summary.get("family_sizes", {"primary": 20, "secondary": 316, "union": 336}),
        "na_metric_cells": sum(1 for row in selections for short in ("BA", "AP", "AUC", "Brier", "EO", "DP") if not is_number(row.get("T_" + short))),
        "q_only_probability_na_cells": sum(
            1 for row in selections if row.get("method") in Q_ONLY_METHODS
            for short in ("AP", "AUC", "Brier") if not is_number(row.get("T_" + short))
        ),
        "selection_status_counts": dict(sorted(Counter(str(row.get("S_status", "NA")) for row in selections).items())),
        "outputs": [str(path.name) for path in sorted(output.iterdir()) if path.is_file()],
        "figures": [str(path.name) for path in created],
        "outline": outline.name,
        "policy": "reporting-only; no selection change, no other T read, no microdata or prediction-array read",
    }
    metadata["output_sha256"] = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file()
        and path.name not in {"build_metadata.json", "FAIRBIAS_COMPLETION_PAPER_WORKER_20260918.md"}
    }
    (output / "build_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def synthetic_summary() -> dict[str, Any]:
    def metric(mean: float | None, low: float | None = None, high: float | None = None) -> dict[str, Any]:
        value: dict[str, Any] = {"mean": mean, "status": "VALID" if mean is not None else "NOT_ESTIMABLE"}
        value["bootstrap_descriptive"] = {"status": "VALID", "lower": low, "upper": high} if low is not None and high is not None else {"status": "NOT_ESTIMABLE"}
        return value

    def row(method: str, sid: str, missing_ap: bool = False) -> dict[str, Any]:
        return {
            "selection_id": sid, "arm_id": "arm_001", "backbone": "LR", "method": method,
            "origin": "adaptive_completed80" if method.startswith("FAIRBIAS") else "parent_tau10",
            "training_weighted": False, "tau": 0.1, "status": "FEASIBLE", "model_ids": [sid + "_s0"],
            "evaluation_status": "VALID", "balanced_accuracy": metric(0.7, 0.68, 0.72),
            "average_precision": metric(None) if missing_ap else metric(0.5, 0.45, 0.55),
            "auroc": metric(0.75, 0.70, 0.80), "brier": metric(0.20, 0.18, 0.22),
            "eo_gap": {"mean": 0.08, "projection_95": {"status": "VALID", "lower": 0.02, "upper": 0.14}},
            "dp_gap": {"mean": 0.05, "projection_95": {"status": "VALID", "lower": 0.01, "upper": 0.09}},
        }

    selections = [row("FAIRBIAS_BM_AE", "completion:arm_001:LR:FAIRBIAS_BM_AE"), row("FAIRBIAS_JOINT", "completion:arm_001:LR:FAIRBIAS_JOINT"), row("UNMITIGATED", "parent:arm_001:LR:UNMITIGATED", True)]
    pairs = [{
        "contrast_id": "contrast:joint-bmae", "reference_selection_id": selections[1]["selection_id"], "comparison_selection_id": selections[0]["selection_id"],
        "arm_id": "arm_001", "backbone": "LR", "family": "secondary", "kind": "joint_vs_sequential",
        "reference_method": "FAIRBIAS_JOINT", "comparison_method": "FAIRBIAS_BM_AE", "difference_direction": "reference_minus_comparison",
        "family_endpoint_slots": 316, "union_endpoint_slots": 336, "evaluation_status": "VALID",
        "delta_balanced_accuracy": {"estimate": 0.01, "family_interval": {"status": "VALID", "lower": -0.03, "upper": 0.05}},
        "delta_average_precision": 0.02, "delta_auroc": 0.01, "delta_brier": -0.01,
        "delta_eo_gap": {"estimate": -0.01, "projection_family": {"status": "VALID", "lower": -0.10, "upper": 0.08}},
        "delta_dp_gap": {"estimate": -0.02, "projection_family": {"status": "VALID", "lower": -0.08, "upper": 0.04}},
    }]
    return {"schema_version": "synthetic_mini", "status": "COMPLETE", "known_T": True, "selections": selections, "paired_contrasts": pairs}


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="completion_paper_selftest_") as tmp:
        root = Path(tmp)
        source = root / "synthetic_merged.json"
        source.write_text(json.dumps(synthetic_summary()), encoding="utf-8")
        output = root / "output"
        metadata = build(source, output, strict=False)
        assert metadata["selection_rows"] == 3
        assert metadata["pair_rows"] == 1
        csv_text = (output / "completion_conditions_96.csv").read_text()
        assert "NA" in csv_text
        assert "NA,NA" in csv_text or ",NA," in csv_text
        assert (output / "prediction_fairness_scatter.png").exists()
        assert (output / "joint_minus_bmae_paired_ci.svg").exists()
        print(json.dumps({"status": "SELF_TEST_PASS", "output_files": metadata["outputs"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Merged completion summary JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/paper/completion_results_20260918"))
    parser.add_argument("--allow-mini", action="store_true", help="Allow a reduced synthetic schema for development tests")
    parser.add_argument("--self-test", action="store_true", help="Run a temporary synthetic schema test and write no repository outputs")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.input is None:
        parser.error("--input is required unless --self-test is used")
    metadata = build(args.input, args.output_dir, strict=not args.allow_mini)
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
