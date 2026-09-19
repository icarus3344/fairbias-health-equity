#!/usr/bin/env python3
"""Independent, aggregate-only export audit; no policy or raw-T loading.

Checks paper CSVs against the accepted summary and recomputes F/C/S cohort
aggregates from hash-bound prepared inputs. Does not import the exporters.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BASE = ROOT / "artifacts/nhis/completion_evaluation_20260918"
PAPER = ROOT / "docs/paper/completion_results_20260918"
COHORT = ROOT / "docs/paper/cohort_tables_v2_20260918"
MERGED_SHA = "641fe2ec523c8998cfe4c1f48ae2d5fb1e6ecbe9a5eb9e20b16af33d47c6b406"
ADMISSION_SHA = "2a83612b23048a54ce09196ec553b9881ee87bf5b5930f36fc53bc0e52d1a628"
COUNTS: Counter = Counter()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok, label):
    if not ok:
        raise RuntimeError(label)
    COUNTS[label.split(":", 1)[0]] += 1


def read(path):
    return json.loads(Path(path).read_text())


def table(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def number(actual, expected, label):
    if expected is None:
        require(actual == "NA", label)
    else:
        require(actual != "NA" and math.isfinite(float(actual))
                and math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-10), label)


def main():
    require(sha(BASE / "merged_summary_v1.json") == MERGED_SHA, "input:merged")
    require(sha(BASE / "control/admission_v1.json") == ADMISSION_SHA, "input:admission")
    merged = read(BASE / "merged_summary_v1.json")
    admission = read(BASE / "control/admission_v1.json")
    require(len(admission["analysis_files"]) == 95, "source:coverage")
    for name, digest in admission["analysis_files"].items():
        require(sha(ROOT / name) == digest, "source:" + name)
    metadata = read(PAPER / "build_metadata.json")
    require(metadata["input_sha256"] == MERGED_SHA, "binding:paper")
    for name, digest in metadata["output_sha256"].items():
        require(sha(PAPER / name) == digest, "output_hash:" + name)
    cm = read(COHORT / "hash_manifest.json")
    require(sha(cm["created_by"]) == cm["script_sha256"], "binding:cohort_source")
    for name, ref in cm["inputs"].items():
        require(sha(ref["path"]) == ref["sha256"], "input_hash:" + name)
    for name, digest in cm["outputs"].items():
        require(sha(COHORT / name) == digest, "output_hash:" + name)

    metrics = {"BA": "balanced_accuracy", "AP": "average_precision", "AUC": "auroc",
               "Brier": "brier", "EO": "eo_gap", "DP": "dp_gap"}
    cond = table(PAPER / "completion_conditions_96.csv")
    cond_map = {r["selection_id"]: r for r in cond}
    require(len(cond_map) == len(cond) == len(merged["selections"]) == 96, "coverage:conditions")
    for row in merged["selections"]:
        out = cond_map[row["selection_id"]]
        for field in ("arm_id", "backbone", "method", "origin"):
            require(out[field] == row[field], "condition_identity:" + field)
        require(out["S_status"] == row["status"], "condition_identity:S_status")
        require(out["T_evaluation_status"] == row["evaluation_status"], "condition_identity:T_status")
        for field in ("sample_n", "design_n", "df"):
            number(out[field], row.get(field), "condition_design:" + field)
        for label, key in metrics.items():
            metric = row.get(key) or {}
            unweighted = row.get("unweighted_" + key) or {}
            number(out["T_" + label], metric.get("mean"), "condition_point:" + label)
            number(out["T_" + label + "_seed_sd"], metric.get("seed_sd"), "condition_seed_sd:" + label)
            for part in ("mean", "seed_sd"):
                number(out[f"unweighted_{label}_{part}"], unweighted.get(part), "condition_unweighted:" + label)
            interval_key = "taylor_95" if label == "BA" else "projection_95"
            ci = metric.get(interval_key) or {}
            for part, source in (("low", "lower"), ("high", "upper")):
                number(out[f"T_{label}_ci_{part}"], ci.get(source), "condition_ci:" + label)
    require(Counter(r["status"] for r in merged["selections"])["NOT_SUPPORTED"] == 4, "coverage:unsupported")
    require(Counter(r["status"] for r in merged["selections"])["NO_VALID_FIXED_ABLATION"] == 2, "coverage:missing_BM")

    pairs = table(PAPER / "completion_pairs_168.csv")
    pair_map = {r["contrast_id"]: r for r in pairs}
    require(len(pair_map) == len(pairs) == len(merged["paired_contrasts"]) == 168, "coverage:pairs")
    for row in merged["paired_contrasts"]:
        out = pair_map[row["contrast_id"]]
        for field in ("difference_direction", "reference_selection_id", "comparison_selection_id", "family", "evaluation_status"):
            require(out[field] == row[field], "pair_identity:" + field)
        for label, key in metrics.items():
            value = row.get("delta_" + key)
            metric = value if isinstance(value, dict) else {}
            number(out["delta_" + label], metric.get("estimate") if metric else value, "pair_point:" + label)
            levels = ({"nominal": "taylor_95", "family": "family_interval", "union": "union_family_interval"}
                      if label == "BA" else {"nominal": "projection_95", "family": "projection_family", "union": "projection_union_family"})
            for level, ci_key in levels.items():
                ci = metric.get(ci_key) or {}
                for part, source in (("low", "lower"), ("high", "upper")):
                    number(out[f"delta_{label}_ci_{level}_{part}"], ci.get(source), "pair_ci:" + label)
            for part in ("low", "high"):
                require(out[f"delta_{label}_ci_{part}"] == out[f"delta_{label}_ci_family_{part}"], "pair_ci:default_family")
    eight = table(PAPER / "completion_fairbias_8_conditions.csv")
    require(len(eight) == 8, "coverage:eight_cells")
    for row in eight:
        for method in ("BM_AE", "Joint"):
            selected = cond_map[row[method + "_selection_id"]]
            for label in metrics:
                require(row[f"{method}_{label}"] == selected["T_" + label], "eight_cell_point:" + label)
                for part in ("low", "high"):
                    require(row[f"{method}_{label}_ci_{part}"] == selected[f"T_{label}_ci_{part}"], "eight_cell_ci:" + label)

    rows = table(COHORT / "Table1.csv")
    require(len(rows) == 68, "coverage:cohort")
    require(sum(r["group_code"] == "ALL" for r in rows) == 16, "coverage:overall")
    role_names = {"F": "fitting_F", "C": "calibration_C", "S": "selection_S"}
    for arm in ("arm_001", "arm_002", "arm_003", "arm_004"):
        ref = cm["inputs"][arm + ".prepared"]
        payload = joblib.load(ref["path"])  # Already hash-verified above.
        require(payload["data_identity"] == ref["data_identity"], "cohort_binding:identity")
        for part, role in role_names.items():
            data = payload["partitions"][role]
            y, group, weight = np.asarray(data.y), np.asarray(data.A), np.asarray(data.WTFA_A, dtype=float)
            for row in (r for r in rows if r["arm_id"] == arm and r["partition"] == part):
                mask = np.ones(len(y), dtype=bool) if row["group_code"] == "ALL" else group == int(row["group_code"])
                w, yy = weight[mask], y[mask]
                expected = {"n": len(yy), "event_count": int(np.count_nonzero(yy == 1)),
                            "weight_sum": w.sum(), "weighted_event_rate": np.average(yy, weights=w),
                            "kish_ess": w.sum() ** 2 / np.square(w).sum(),
                            "weight_min": w.min(), "weight_max": w.max(),
                            "weight_p25": np.percentile(w, 25), "weight_p50": np.percentile(w, 50), "weight_p75": np.percentile(w, 75),
                            "analysis_domain_strata_count": len(set(np.asarray(data.PSTRAT)[mask])),
                            "analysis_domain_psu_count": len(set(zip(np.asarray(data.PSTRAT)[mask], np.asarray(data.PPSU)[mask]))),
                            "inference_design_strata_count": len(set(data.annual_design.strata)),
                            "inference_design_psu_count": len(set(zip(data.annual_design.strata, data.annual_design.psus)))}
                for key, value in expected.items():
                    number(row[key], value, "cohort_FCS_independent:" + key)
        del payload
        t = read(cm["inputs"][arm + ".cohort_T"]["path"])
        for row in (r for r in rows if r["arm_id"] == arm and r["partition"] == "T"):
            expected = t["overall"] if row["group_code"] == "ALL" else t["groups"][row["group_code"]]
            for key in ("n", "event_count", "weight_sum", "weighted_event_rate", "kish_ess"):
                number(row[key], expected.get(key), "cohort_T_saved_aggregate:" + key)
            require(row["inference_design_strata_count"] == "52" and row["inference_design_psu_count"] == "662", "cohort_T:annual_design")
            require(all(row[x] == "NA" for x in ("weight_min", "weight_max", "weight_p25", "weight_p50", "weight_p75")), "cohort_T:unavailable_diagnostics")
        for part in ("F", "C", "S", "T"):
            subset = [r for r in rows if r["arm_id"] == arm and r["partition"] == part]
            overall = next(r for r in subset if r["group_code"] == "ALL")
            for key in ("n", "event_count"):
                require(sum(int(r[key]) for r in subset if r["group_code"] != "ALL") == int(overall[key]), "cohort:partition_sum")
    result = {"status": "PASS", "schema_version": "completion_paper_numerical_review_v1",
              "merged_sha256": MERGED_SHA, "verification_source_sha256": sha(__file__),
              "checks": dict(COUNTS), "total_checks": sum(COUNTS.values()),
              "scope": "96 condition /168 pair /8 compact-cell CSVs;68 cohort rows;hashes;independent F/C/S aggregates;T saved aggregates only",
              "limits": ["No new policy loading, model fitting, selection or T prediction", "No raw T loaded", "Not a new rederivation of survey inference"]}
    destination = BASE / "control/paper_numerical_verification_v1.json"
    with destination.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
