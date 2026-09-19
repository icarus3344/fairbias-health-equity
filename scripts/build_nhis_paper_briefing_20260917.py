#!/usr/bin/env python3
"""Build an aggregate-only NHIS FairBias paper briefing and draft figures.

The renderer reads frozen JSON summaries and cohort/individual metric summaries.
It never deserializes models or prediction arrays. Interpretability records
are joined to the frozen selected models and independently hash-verified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/nhis/completed_main_analysis_backup_20260917/snapshot/artifacts/nhis/analysis_catalog_20260917_064230Z"
EVAL = BASE / "main_evaluation_v2"
MERGED = EVAL / "merged"
OUT = ROOT / "docs/paper"
FIGDIR = OUT / "figures_20260917"
SUMMARY_SHA = "f2c3edee002558de514a7dadc7598575e9be340a73d7e2e79d360fdf7cd4690f"
MANIFEST_SHA = "9d532449976be038d8d0e6f530d2c456ce4ea97727cba843c4d1b358320465b3"
STUDY_SHA = "b9531ebba6c95e66489255eb0a232ce2a67b585e48799673d8d0c6dad1a10b1e"
SELECTION_SHA = "ef11fe4864e0fc5d497c3c5731d1d633ecf16f2b038cca2ab03d661906b87838"
ARMS = ("arm_001", "arm_002", "arm_003", "arm_004")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return "NA" if value is None else f"{100 * float(value):.2f}%"


def verify_inputs():
    assert sha(MERGED / "summary_T.json") == SUMMARY_SHA
    assert sha(MERGED / "evaluation_manifest.json") == MANIFEST_SHA
    manifest = read(MERGED / "evaluation_manifest.json")
    assert manifest["study_freeze_sha256"] == STUDY_SHA
    assert manifest["selection_sha256"] == SELECTION_SHA
    assert manifest["selected_models"] and len(manifest["selected_models"]) == 925
    study_path = BASE / "main_analysis_v3/study_freeze.json"
    selection_path = BASE / "main_analysis_v2/selection_freeze.json"
    assert sha(study_path) == STUDY_SHA
    assert sha(selection_path) == SELECTION_SHA
    for shard in manifest["shards"]:
        for field in ("manifest", "summary"):
            item = shard[field]
            rel = item["path"].split("/main_evaluation_v2/", 1)[1]
            assert sha(EVAL / rel) == item["sha256"]
    for arm in ARMS:
        am = read(EVAL / arm / "evaluation_manifest.json")
        for name, digest in am["artifacts"].items():
            path = EVAL / arm / name
            assert path.is_file() and sha(path) == digest, (arm, name)
    return manifest, read(study_path)


def selected_rows(summary):
    return [r for r in summary["selections"] if r.get("tau") == 0.1 and r.get("backbone") in {"LR", "GBDT"}
            and r.get("training_weighted") is False]


def cohort_rows():
    rows = []
    for arm in ARMS:
        d = read(EVAL / arm / f"{arm}_cohort_T.json")
        o = d["overall"]
        rows.append({"arm": arm, "n": o["n"], "events": o["event_count"], "weight_sum": o["weight_sum"],
                     "kish_ess": o["kish_ess"], "strata": d["design"]["strata_count"],
                     "psus": d["design"]["psu_count"], "status": o["status"]})
    return rows


def write_tables(summary, manifest):
    rows = selected_rows(summary)
    out = OUT / "FAIRBIAS_FROZEN_RESULT_TABLES_20260917.md"
    methods = sorted({r["method"] for r in rows})
    lines = ["# Frozen NHIS FairBias result tables (aggregate-only briefing)", "",
             "This is a draft briefing generated from the hash-verified v2 T aggregate. It does not select a winner or establish a superiority claim.", "",
             f"Summary SHA-256: `{SUMMARY_SHA}`; evaluation manifest SHA-256: `{MANIFEST_SHA}`.", "",
             "## Cohort and survey design", "", "| Arm | Eligible n | Events | Weight sum | Kish ESS | Strata | PSUs | Status |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in cohort_rows():
        lines.append(f"| {r['arm']} | {r['n']:,} | {r['events']:,} | {r['weight_sum']:,.1f} | {r['kish_ess']:,.1f} | {r['strata']} | {r['psus']} | {r['status']} |")
    lines += ["", "## LR/GBDT, tau=0.10, unweighted training, survey-weighted T evaluation", "",
             "All states are retained. BA/EO/DP are survey-weighted T estimates; `NO_VALID_MODEL` rows are not silently dropped. Unweighted sensitivity values are available in the aggregate source and remain secondary.", "",
              "| Arm | Backbone | Method | S status | T status | BA | BA Taylor 95% | EO | EO primary-family band | DP | AP | AUROC | Brier |", "|---|---|---|---|---|---:|---|---:|---|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda x: (x["arm_id"], x["backbone"], x["method"])):
        ba = r.get("balanced_accuracy", {}); eo = r.get("eo_gap", {}); dp = r.get("dp_gap", {})
        t = ba.get("taylor_95", {})
        band = eo.get("projection_primary_family", {})
        def interval(x): return "NA" if x.get("status") != "VALID" else f"[{pct(x.get('lower'))}, {pct(x.get('upper'))}]"
        brier = r.get('brier', {}).get('mean'); brier_text = "NA" if brier is None else f"{brier:.4f}"
        lines.append(f"| {r['arm_id']} | {r['backbone']} | {r['method']} | {r.get('status')} | {r.get('evaluation_status')} | {pct(ba.get('mean'))} | {interval(t)} | {pct(eo.get('mean'))} | {interval(band)} | {pct(dp.get('mean'))} | {pct(r.get('average_precision', {}).get('mean'))} | {pct(r.get('auroc', {}).get('mean'))} | {brier_text} |")
    lines += ["", "## Compact matched-backbone table", "", "LR-only primary operating point; this keeps the primary comparison readable while the full table above retains every LR/GBDT method state.", "", "| Arm | FairBias BA | FairBias EO | Unmitigated BA | Unmitigated EO |", "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        a = [r for r in rows if r["arm_id"] == arm and r["backbone"] == "LR" and r["method"] in {"FAIRBIAS_BM", "UNMITIGATED"}]
        by = {r["method"]: r for r in a}; f, u = by.get("FAIRBIAS_BM", {}), by.get("UNMITIGATED", {})
        lines.append(f"| {arm} | {pct(f.get('balanced_accuracy', {}).get('mean'))} | {pct(f.get('eo_gap', {}).get('mean'))} | {pct(u.get('balanced_accuracy', {}).get('mean'))} | {pct(u.get('eo_gap', {}).get('mean'))} |")
    lines += ["", "## Compact modern matched-backbone table", "", "All registered modern backbones at tau=0.10, training-unweighted. This table is descriptive and preserves unavailable states.", "", "| Arm | Backbone | Method | BA | EO | S status / T status |", "|---|---|---|---:|---:|---|"]
    for r in sorted([r for r in summary["selections"] if r.get("backbone") in {"FAIRGBM_BASE", "MLP", "TABM"} and r.get("tau") == .1 and r.get("training_weighted") is False], key=lambda x: (x["arm_id"], x["backbone"], x["method"])):
        lines.append(f"| {r['arm_id']} | {r['backbone']} | {r['method']} | {pct(r.get('balanced_accuracy', {}).get('mean'))} | {pct(r.get('eo_gap', {}).get('mean'))} | {r.get('status')} / {r.get('evaluation_status')} |")
    lines += ["", "## Primary paired BA family", "", "Positive values mean FairBias-BM minus baseline. Family intervals use the registered 20-endpoint correction; EO family projections are shown separately and are not normal max-gap p-values.", "",
              "| Arm | Baseline | BA difference (pp) | Family 95% interval (pp) | S feasible |", "|---|---|---:|---:|---|"]
    for p in summary["paired_contrasts"]:
        if not p.get("in_primary_family_20"): continue
        d = p["delta_balanced_accuracy"]; ci = d.get("family_20_interval", {})
        lines.append(f"| {p['arm_id']} | {p['comparison_method']} | {100*d['estimate']:.2f} | [{100*ci.get('lower', float('nan')):.2f}, {100*ci.get('upper', float('nan')):.2f}] | {p['feasible_on_S']} |")
    lines += ["", "## Primary paired EO family", "", "Projection intervals are reported as registered simultaneous EO gap bounds; no normal max-gap p-value is implied.", "", "| Arm | Baseline | EO difference | Primary-family interval |", "|---|---|---:|---:|"]
    for p in summary["paired_contrasts"]:
        if not p.get("in_primary_family_20"): continue
        d = p["delta_eo_gap"].get("projection_primary_family", {})
        est = p["delta_eo_gap"].get("estimate"); est_text = "NA" if est is None else f"{100*est:.2f} pp"
        lines.append(f"| {p['arm_id']} | {p['comparison_method']} | {est_text} | [{100*d['lower']:.2f}, {100*d['upper']:.2f}] pp |")
    lines += ["", "## Group support in the 2024 analysis domains", "", "Category codes follow the registered Arm specification. Kish ESS describes weight dispersion, not full cluster-adjusted effective sample size.", "", "| Arm | Group code | n | Events | Weighted event rate | Kish ESS |", "|---|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        for group, d in read(EVAL / arm / f"{arm}_cohort_T.json")["groups"].items():
            lines.append(f"| {arm} | {group} | {d['n']} | {d['event_count']} | {pct(d['weighted_event_rate'])} | {d['kish_ess']:.1f} |")
    lines += ["", "## Figure captions", "", "Figure 1: LR only, all supported main comparison methods at the registered tau=0.10. Solid symbols are S-feasible; open symbols retain S-infeasible boundary configurations. Missing methods are described in the tables. Means only, no uncertainty shown. Overlapping points are not separated artificially.", "", "Figure 2: Exact S-selected/boundary LR FairBias models, all five registered seeds per arm. Initial F feature geometry score is normalized by each seed's within-arm maximum before averaging. Gray NA cells are features excluded from Arm 004; unchanged present features have operation frequency zero. D/M/P denote deletion/category merge/power transformation. Final/initial max-dphi ratios are endpoints, not a global iteration trajectory. No causal or SHAP claim.", "", "Figure 3: All ten primary BA pairs, FairBias minus comparator in percentage points, family-adjusted for the 20 registered BA/EO endpoints. Intervals condition on fixed policies; EO intervals appear in the accompanying table."]
    lines += ["", "## Interpretation boundary", "", "The aggregate is conditional on fixed selected policies. Bootstrap intervals are descriptive; survey Taylor intervals are approximate public-use-design inference. q-only methods have unavailable event-risk metrics. 2024 is retrospective. Supplemental branches remain separate.", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "figure.dpi": 160})


def save(fig, stem):
    fig.tight_layout()
    fig.savefig(FIGDIR / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(FIGDIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def fig1(summary):
    names = {"FAIRBIAS_BM": "FairBias-BM", "UNMITIGATED": "Unmitigated", "REWEIGHING": "Reweighing", "LFR_RECONSTRUCTED": "LFR", "EG_DP": "EG-DP", "EG_EO": "EG-EO", "TO_EO": "ThresholdOpt", "FRAPPE_EO": "FRAPPE", "OXONFAIR_EO": "OxonFair"}
    colors = dict(zip(names, ["#cc4555", "#424b57", "#007c91", "#55a99a", "#4477aa", "#90b3d7", "#87639d", "#d99834", "#688349"]))
    rows = [r for r in selected_rows(summary) if r["backbone"] == "LR" and r["method"] in names]
    titles = ["A  Sex", "B  Race / ethnicity (7 groups)", "C  Disability", "D  Disability, feature ablation"]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), sharex=True, sharey=True)
    for ax, arm, title in zip(axes.flat, ARMS, titles):
        for r in sorted([r for r in rows if r["arm_id"] == arm], key=lambda r: r["method"] == "FAIRBIAS_BM"):
            ba = r.get("balanced_accuracy", {}).get("mean"); eo = r.get("eo_gap", {}).get("mean")
            if ba is None or eo is None: continue
            emph = r["method"] == "FAIRBIAS_BM"
            ax.scatter(eo, ba, s=110 if emph else 48, marker="D" if emph else "o", facecolors=colors[r["method"]] if r["status"] == "FEASIBLE" else "none", edgecolors=colors[r["method"]], linewidths=1.5, zorder=4 if emph else 3)
        ax.axvline(.1, color="#adb5bd", lw=.8, ls="--")
        ax.set_title(title, loc="left", weight="bold", fontsize=11)
        ax.set_xlim(-.015, .47); ax.set_ylim(.49, .725)
        ax.set_xlabel("EO gap (lower is better)"); ax.set_ylabel("Balanced accuracy (higher is better)")
    handles = [plt.Line2D([0], [0], marker="D" if m == "FAIRBIAS_BM" else "o", color="none", markerfacecolor=colors[m], markeredgecolor=colors[m], label=names[m], markersize=7) for m in names]
    fig.suptitle("Prediction and fairness across NHIS evaluation arms", fontsize=16, weight="bold", y=.99)
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(.5, .04), frameon=False)
    fig.text(.5, .015, "LR, tau = 0.10; survey-weighted 2024 means. Filled = feasible on 2023 S; open = boundary. No CI shown.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .95))
    fig.savefig(FIGDIR / "fig1_four_arm_accuracy_eo.png", dpi=240, bbox_inches="tight")
    fig.savefig(FIGDIR / "fig1_four_arm_accuracy_eo.svg", bbox_inches="tight")
    plt.close(fig)


def trace_records(summary, study):
    model_map = {m["model_id"]: m for m in study["selected_models"]}
    records = []
    for row in summary["selections"]:
        if row.get("method") != "FAIRBIAS_BM" or row.get("backbone") != "LR" or row.get("tau") != .1 or row.get("training_weighted") is not False:
            continue
        for mid in row.get("model_ids", []):
            m = model_map[mid]; rel = m["result_path"].split("/root/autodl-tmp/fairbias/", 1)[1]
            path = ROOT / rel
            assert path.is_file() and sha(path) == m["result_sha256"], (mid, path)
            d = read(path)["algorithm_manifest"]
            assert {"reference_epsilon", "initial_max_dphi", "final_max_dphi", "iterations", "changed_dict", "transform_trace"} <= set(d)
            eps = d["reference_epsilon"].get("__protected__", d["reference_epsilon"])
            records.append({"arm": row["arm_id"], "seed": m["seed"], "model_id": mid, "result_path": rel, "result_sha256": m["result_sha256"], "reference_epsilon": eps, "initial_max_dphi": d["initial_max_dphi"], "final_max_dphi": d["final_max_dphi"], "iterations": d["iterations"], "changed_dict": d["changed_dict"], "transform_trace": d["transform_trace"]})
    assert len(records) == 20 and {r["arm"] for r in records} == set(ARMS)
    return records


def fig2_trace(summary, study):
    records = trace_records(summary, study)
    features = sorted({f for r in records for f in r["reference_epsilon"]})
    labels = {"agep_a":"Age", "asev_a":"Asthma history", "chlev_a":"High cholesterol", "cogmemdff_a":"Memory difficulty", "comdiff_a":"Communication difficulty", "dibev_a":"Diabetes", "diff_a":"Walking difficulty", "educp_a":"Education", "empwrkft1_a":"Full-time work", "empwrklsw1_a":"Worked last week", "hearingdf_a":"Hearing difficulty", "hypev_a":"Hypertension", "notcov_a":"No insurance", "pcnt18uptc":"Adults in household", "pcntlt18tc":"Children in household", "phstat_a":"Self-rated health", "ratcat_a":"Income / poverty ratio", "region":"Region", "uppslfcr_a":"Self-care difficulty", "usualpl_a":"Usual place of care", "visiondf_a":"Vision difficulty"}
    norm = np.full((len(features), 4), np.nan); freq = norm.copy(); operations = {}; ratios = []
    for ai, arm in enumerate(ARMS):
        rr = [r for r in records if r["arm"] == arm]
        assert {r["seed"] for r in rr} == {0, 7, 19, 37, 73}
        for fi, feature in enumerate(features):
            present = [r for r in rr if feature in r["reference_epsilon"]]
            if not present: continue
            assert len(present) == 5
            norm[fi, ai] = np.mean([r["reference_epsilon"][feature] / max(r["reference_epsilon"].values()) for r in present])
            cats = []
            for r in present:
                v = r["changed_dict"].get(feature)
                cats.append("D" if v == "dropped" else "P" if isinstance(v, dict) and "power" in v else "M" if isinstance(v, dict) else "")
            freq[fi, ai] = sum(bool(c) for c in cats) / 5
            operations[(fi, ai)] = "/".join(sorted(set(cats) - {""}))
        ratios.append([r["final_max_dphi"] / r["initial_max_dphi"] for r in rr])
    assert np.isnan(norm[:, 3]).sum() == 6 and np.isnan(freq[:, 3]).sum() == 6
    assert np.all(freq[:, 0] == 0)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 8), gridspec_kw={"width_ratios":[1.6, 1.25, 1.1]})
    palettes = [plt.get_cmap("viridis").copy(), plt.get_cmap("YlOrRd").copy()]
    for cmap in palettes: cmap.set_bad("#d9dde2")
    arm_labels = ["1: Sex", "2: Race*", "3: Disability", "4: Ablation"]
    for idx, data, title, cb in [(0, norm, "A  Initial feature geometry", "Normalized score (mean of five seeds)"), (1, freq, "B  Recorded transformations", "Fraction of seeds with a change")]:
        ax=axes[idx];im=ax.imshow(np.ma.masked_invalid(data), aspect="auto", cmap=palettes[idx], vmin=0, vmax=1)
        ax.grid(False);ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=13)
        ax.set_xticks(range(4), arm_labels, rotation=40, ha="right", fontsize=9)
        ax.set_yticks(range(len(features)), [labels.get(f,f) for f in features] if idx == 0 else [], fontsize=10)
        for fi in range(len(features)):
            for ai in range(4):
                if np.isnan(data[fi,ai]): ax.text(ai,fi,"NA",ha="center",va="center",fontsize=8,color="#66717d")
                elif idx == 1 and operations.get((fi,ai)): ax.text(ai,fi,operations[(fi,ai)],ha="center",va="center",fontsize=11,weight="bold",color="white" if data[fi,ai]>.55 else "#273240")
        c=fig.colorbar(im,ax=ax,orientation="horizontal",fraction=.045,pad=.15);c.set_label(cb,fontsize=9)
    ax=axes[2];means=np.array([np.mean(r) for r in ratios]); ypos=np.arange(4)
    ax.barh(ypos, means, color=["#a8b0b9", "#dfb56c", "#cc4555", "#754d74"],height=.55)
    for ai,vals in enumerate(ratios):
        ax.scatter(vals,np.full(5,ai),color="#273240",s=14,zorder=3)
        ax.text(means[ai]+.025,ai,f"{means[ai]:.3f}",va="center",fontsize=11)
    ax.set_yticks(ypos,arm_labels);ax.invert_yaxis();ax.set_xlim(0,1.17)
    ax.set_xlabel("Final / initial max geometry score");ax.set_title("C  Geometry endpoints",loc="left",fontsize=12,weight="bold",pad=13)
    fig.suptitle("What did FairBias actually change?",fontsize=18,weight="bold",y=.99)
    fig.text(.02,.035,"Fitting-partition diagnostics for the frozen LR models at tau = 0.10. *Arm 2 uses its S-infeasible boundary model. Gray = excluded feature.",fontsize=9)
    fig.text(.02,.012,"D = deletion; M = category merge; P = power transform. Geometry scores are not EO, causal effects, or SHAP importance.",fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.95),w_pad=2.1)
    fig.savefig(FIGDIR / "fig2_selected_lr_bm_interpretability_traces.png",dpi=240,bbox_inches="tight")
    fig.savefig(FIGDIR / "fig2_selected_lr_bm_interpretability_traces.svg",bbox_inches="tight")
    plt.close(fig)
    return records


def fig3(summary):
    ps = [p for p in summary["paired_contrasts"] if p.get("in_primary_family_20")]
    labels = [f"{p['arm_id'].replace('_',' ')} · {p['comparison_method']}" for p in ps]
    y = np.arange(len(ps)); est = np.array([100*p["delta_balanced_accuracy"]["estimate"] for p in ps])
    lo = np.array([100*p["delta_balanced_accuracy"]["family_20_interval"]["lower"] for p in ps]); hi = np.array([100*p["delta_balanced_accuracy"]["family_20_interval"]["upper"] for p in ps])
    fig, ax = plt.subplots(figsize=(10, 6.5)); ax.axvline(0, color="#333", lw=1); ax.errorbar(est, y, xerr=[est-lo, hi-est], fmt="o", color="#d1495b", ecolor="#17324d", capsize=3, lw=1.4)
    ax.set_yticks(y, labels); ax.set_xlabel("FairBias-BM minus baseline balanced accuracy (percentage points)"); ax.set_title("Primary paired BA contrasts, family-adjusted intervals", fontsize=14, fontweight="bold"); ax.text(.01, -.13, "Family = 20 registered BA/EO endpoints. Positive favors FairBias; intervals are conditional on frozen policies.", transform=ax.transAxes, fontsize=8)
    save(fig, "fig3_primary_ba_paired_forest")


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    style(); manifest, study = verify_inputs(); summary = read(MERGED / "summary_T.json")
    write_tables(summary, manifest); fig1(summary); records = fig2_trace(summary, study); fig3(summary)
    trace = {"status": "VERIFIED", "record_count": len(records), "fields": ["reference_epsilon", "initial_max_dphi", "final_max_dphi", "changed_dict", "transform_trace"], "model_or_prediction_arrays_loaded": False, "summary_sha256": SUMMARY_SHA, "evaluation_manifest_sha256": MANIFEST_SHA, "records": records}
    (OUT / "figures_20260917" / "interpretability_trace_audit.json").write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    output_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(FIGDIR.iterdir()) if p.is_file() and p.name != "output_hashes.json"}
    for p in [OUT / "FAIRBIAS_FROZEN_RESULT_TABLES_20260917.md", Path(__file__)]:
        output_hashes[str(p.relative_to(ROOT))] = sha(p)
    (FIGDIR / "output_hashes.json").write_text(json.dumps(output_hashes, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tables": str(OUT / "FAIRBIAS_FROZEN_RESULT_TABLES_20260917.md"), "figures": sorted(p.name for p in FIGDIR.iterdir())}, indent=2))


if __name__ == "__main__":
    main()
