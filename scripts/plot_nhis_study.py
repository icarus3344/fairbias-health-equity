"""Plot frozen NHIS benchmark aggregates without reading models or microdata.

The inputs are the registration, selection-freeze, and summary_T JSON files
produced by the frozen evaluation workflow.  This script only consumes their
aggregate fields and writes a new, non-overwriting output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _nested(row: dict[str, Any], name: str, key: str) -> Any:
    value = row.get(name)
    if isinstance(value, dict):
        return value.get(key)
    return None


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _method_color(method: str, colors: dict[str, Any]) -> Any:
    if method not in colors:
        colors[method] = plt.get_cmap("tab20")(len(colors) % 20)
    return colors[method]


def _registration_index(registration: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["candidate_id"]): row for row in registration.get("candidates", [])
            if isinstance(row, dict) and row.get("candidate_id") is not None}


def _s_rows(registration: dict[str, Any], selection: dict[str, Any]) -> list[dict[str, Any]]:
    configs = _registration_index(registration)
    rows: list[dict[str, Any]] = []
    for summary in selection.get("candidate_summaries", []):
        if not isinstance(summary, dict) or summary.get("status") != "VALID":
            continue
        cid = str(summary.get("candidate_id", ""))
        config = configs.get(cid, {})
        ba = _first(summary, "balanced_accuracy", "balanced_accuracy_S")
        eo = _first(summary, "eo_gap", "eo_gap_S")
        if not (_finite(ba) and _finite(eo)):
            continue
        rows.append({"candidate_id": cid, "arm_id": config.get("arm_id", summary.get("arm_id", "UNKNOWN")),
                     "backbone": config.get("backbone", summary.get("backbone", "UNKNOWN")),
                     "method": config.get("method", summary.get("method", "UNKNOWN")),
                     "training_weighted": bool(config.get("training_weighted", summary.get("training_weighted", False))),
                     "balanced_accuracy": float(ba), "eo_gap": float(eo)})
    return rows


def _plot_s_scatter(rows: list[dict[str, Any]], output: Path, *, title: str,
                    arms: Iterable[str], backbone: str | None = None,
                    training_weighted: bool = False) -> None:
    arms = list(arms)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False, sharex=True, sharey=True)
    colors: dict[str, Any] = {}
    for ax, arm in zip(axes.flat, arms):
        subset = [r for r in rows if r["arm_id"] == arm and r["training_weighted"] == training_weighted
                  and (backbone is None or r["backbone"] == backbone)]
        for row in subset:
            ax.scatter(row["eo_gap"], row["balanced_accuracy"], color=_method_color(row["method"], colors),
                       s=38, alpha=.85, edgecolor="white", linewidth=.35)
        ax.axvline(.10, color="black", linestyle="--", linewidth=.8, label="EO threshold = 0.10")
        ax.set_title(str(arm))
        ax.grid(alpha=.2)
    for ax in axes[-1, :]:
        ax.set_xlabel("EO gap on S (lower is more equal)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Balanced accuracy on S")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                      markeredgecolor="white", markersize=7, label=method)
               for method, color in colors.items()]
    handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=.8, label="EO threshold = 0.10"))
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, .18, 1, .95))
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def _paired_value(row: dict[str, Any], metric: str, key: str) -> Any:
    flat = f"delta_{metric}_{key}"
    nested = _nested(row, f"delta_{metric}", key)
    return nested if nested is not None else row.get(flat)


def _paired_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    observed = [r for r in summary.get("paired_contrasts", []) if isinstance(r, dict)]
    expected = [(arm, method) for arm in ("arm_001", "arm_003")
                for method in ("REWEIGHING", "LFR_RECONSTRUCTED", "EG_DP", "EG_EO", "TO_EO")]
    result = []
    for arm, comparator in expected:
        candidates = [r for r in observed if r.get("arm_id") == arm and r.get("backbone") == "LR"
                      and str(r.get("comparison_method", r.get("method", ""))).upper() == comparator
                      and r.get("in_primary_family_20") is True]
        raw = candidates[0] if candidates else {"arm_id": arm, "backbone": "LR",
                                                 "comparison_method": comparator,
                                                 "in_primary_family_20": True,
                                                 "status": "NOT_ESTIMABLE"}
        ba = _paired_value(raw, "balanced_accuracy", "estimate")
        eo = _paired_value(raw, "eo_gap", "estimate")
        ba_interval = _paired_value(raw, "balanced_accuracy", "family_20_interval")
        eo_interval = _paired_value(raw, "eo_gap", "projection_primary_family")
        ba_lo = _first(raw, "delta_ba_family20_lower")
        ba_hi = _first(raw, "delta_ba_family20_upper")
        eo_lo = _first(raw, "delta_eo_family20_lower")
        eo_hi = _first(raw, "delta_eo_family20_upper")
        if isinstance(ba_interval, dict):
            ba_lo, ba_hi = ba_interval.get("lower"), ba_interval.get("upper")
        if isinstance(eo_interval, dict):
            eo_lo, eo_hi = eo_interval.get("lower"), eo_interval.get("upper")
        status = str(raw.get("status") or raw.get("delta_balanced_accuracy", {}).get("status") or "VALID")
        result.append({"label": f'{raw.get("arm_id", "?")} | FairBias minus {raw.get("comparison_method", "?")}',
                       "method": str(raw.get("comparison_method", raw.get("method", "UNKNOWN"))),
                       "feasible_on_S": raw.get("feasible_on_S"),
                       "estimate_ba": float(ba) if _finite(ba) else None,
                       "ba_low": float(ba_lo) if _finite(ba_lo) else None,
                       "ba_high": float(ba_hi) if _finite(ba_hi) else None,
                       "estimate_eo": float(eo) if _finite(eo) else None,
                       "eo_low": float(eo_lo) if _finite(eo_lo) else None,
                       "eo_high": float(eo_hi) if _finite(eo_hi) else None,
                       "status": status})
    return result


def _plot_forest(rows: list[dict[str, Any]], output: Path, *, metric: str, title: str,
                 good_label: str, low: float, high: float) -> None:
    fig, ax = plt.subplots(figsize=(12, max(4.5, .32 * len(rows) + 1.4)))
    ys = list(range(len(rows)))
    for y, row in zip(ys, rows):
        est, lo, hi = row[f"estimate_{metric}"], row[f"{metric}_low"], row[f"{metric}_high"]
        label = row["label"] + (" [S-boundary]" if row.get("feasible_on_S") is False else "")
        if est is None or lo is None or hi is None:
            ax.text(.98, y, "NE", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8)
            continue
        ax.hlines(y, lo, hi, color="tab:blue", linewidth=1.4)
        ax.plot(est, y, "o", color="tab:blue", markersize=4)
    ax.axvline(0, color="black", linewidth=.8)
    finite_bounds = [v for r in rows for v in (r[f"{metric}_low"], r[f"{metric}_high"], r[f"estimate_{metric}"]) if _finite(v)]
    if finite_bounds:
        left, right = min(0., min(finite_bounds)), max(0., max(finite_bounds))
        span = max(right - left, .1)
        low, high = left - .12 * span, right + .12 * span
    ax.set_xlim(low, high)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] + (" [S-boundary]" if r.get("feasible_on_S") is False else "") for r in rows], fontsize=8)
    ax.set_ylim(-.6, len(rows)-.4)
    ax.invert_yaxis()
    ax.set_xlabel(f"Δ{metric.upper()} (FairBias minus comparator; {good_label})")
    ax.set_title(title)
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--selection-freeze", type=Path, required=True)
    parser.add_argument("--summary-t", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    inputs = [args.registration, args.selection_freeze, args.summary_t]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    registration, selection, summary = map(_read, inputs)
    s_rows = _s_rows(registration, selection)
    if not s_rows:
        raise ValueError("selection_freeze contains no finite VALID candidate summaries")
    t_rows = _paired_rows(summary)
    args.output_dir.mkdir(parents=True)
    arms = sorted({str(r["arm_id"]) for r in s_rows})
    while len(arms) < 4:
        arms.append(f"UNOBSERVED_{len(arms) + 1}")
    _plot_s_scatter(s_rows, args.output_dir / "S_scatter_all_backbones_by_arm", title="S: all VALID unweighted configurations", arms=arms[:4])
    backbones = sorted({str(r["backbone"]) for r in s_rows if not r["training_weighted"]})
    for backbone in backbones:
        _plot_s_scatter(s_rows, args.output_dir / f"S_scatter_{backbone}_by_arm", title=f"S: {backbone}", arms=arms[:4], backbone=backbone)
    if any(r["training_weighted"] for r in s_rows):
        _plot_s_scatter(s_rows, args.output_dir / "S_scatter_weighted_sensitivity_by_arm", title="S: weighted sensitivity", arms=arms[:4], training_weighted=True)
    if t_rows:
        _plot_forest(t_rows, args.output_dir / "T_family20_delta_BA", metric="ba", title="T primary family 20: paired ΔBA", good_label="ΔBA > 0 is favorable", low=-1, high=1)
        _plot_forest(t_rows, args.output_dir / "T_family20_delta_EO", metric="eo", title="T primary family 20: paired ΔEO", good_label="ΔEO < 0 is favorable", low=-1, high=1)
    else:
        for name in ("T_family20_delta_BA", "T_family20_delta_EO"):
            fig, ax = plt.subplots(figsize=(8, 2.5)); ax.text(.5, .5, "No estimable family-20 rows", ha="center", va="center"); ax.axis("off")
            fig.savefig(args.output_dir / f"{name}.png", dpi=180); fig.savefig(args.output_dir / f"{name}.pdf"); plt.close(fig)
    outputs = sorted(p for p in args.output_dir.iterdir() if p.is_file())
    manifest = {"inputs": {str(p): _sha(p) for p in inputs}, "s_valid_rows": len(s_rows),
                "t_family20_rows": len(t_rows), "outputs": {p.name: {"sha256": _sha(p), "bytes": p.stat().st_size} for p in outputs},
                "notes": ["Aggregate JSON only; no model, NPZ, or microdata reads.", "No Pareto-optimal claim is computed or made."]}
    (args.output_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), "s_valid_rows": len(s_rows), "t_family20_rows": len(t_rows), "outputs": [p.name for p in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
