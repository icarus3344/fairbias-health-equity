#!/usr/bin/env python3
"""Bounded synthetic coverage check for the production Taylor function.

This is aggregate-only: it creates finite synthetic populations, samples PSUs,
and calls ``linearized_survey_inference``.  It does not read NHIS files.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from nhis_fairbias.benchmark.survey_linearization import linearized_survey_inference

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/reports/fairbias_codex_takeover_20260915_154733Z/survey_linearization_coverage_report.md"
SEED = 20260914
M = 200


def _metrics(y, q, a, w):
    pos, neg = y == 1, y == 0
    tpr = np.dot(w[pos], q[pos]) / w[pos].sum()
    fpr = np.dot(w[neg], q[neg]) / w[neg].sum()
    sr = [np.dot(w[a == g], q[a == g]) / w[a == g].sum() for g in (0, 1)]
    tprs = [np.dot(w[(a == g) & pos], q[(a == g) & pos]) / w[(a == g) & pos].sum() for g in (0, 1)]
    fprs = [np.dot(w[(a == g) & neg], q[(a == g) & neg]) / w[(a == g) & neg].sum() for g in (0, 1)]
    return {
        "ba": float((tpr + 1.0 - fpr) / 2.0),
        "eo": float(max(max(tprs) - min(tprs), max(fprs) - min(fprs))),
    }


def _population(rng, *, rare=False, tied=False):
    H, P, R = 3, 4, 8
    strata = np.repeat(np.arange(H), P * R)
    psus = np.concatenate([np.repeat(np.arange(P), R) for _ in range(H)])
    a = rng.binomial(1, 0.05 if rare else 0.5, len(strata))
    y = rng.binomial(1, np.where(a == 1, 0.30 if rare else 0.35, 0.30))
    w = rng.lognormal(0.0, 0.25, len(strata))
    q1 = np.clip(y * 0.75 + (1 - y) * 0.20 + rng.normal(0, 0.04, len(y)), 0, 1)
    q2 = q1.copy() if tied else np.clip(q1 + np.where(a == 1, 0.04, -0.01), 0, 1)
    return strata, psus, y, a, w, q1, q2


def _sample(rng, strata, psus):
    mask = np.zeros(len(strata), dtype=bool)
    for h in np.unique(strata):
        chosen = rng.choice(np.unique(psus[strata == h]), size=2, replace=False)
        mask |= (strata == h) & np.isin(psus, chosen)
    return mask


def run_scenario(name, *, rare=False, tied=False):
    offsets = {"null_tied_equal_groups": 11, "nonzero_gap_equal_groups": 23,
               "null_tied_rare_group": 37, "nonzero_gap_rare_group": 49}
    rng = np.random.default_rng(SEED + offsets[name])
    covered = {k: 0 for k in ("ba", "eo", "paired_ba")}
    valid = {k: 0 for k in covered}
    failures = {k: 0 for k in covered}
    for _ in range(M):
        s, p, y, a, w, q1, q2 = _population(rng, rare=rare, tied=tied)
        target1, target2 = _metrics(y, q1, a, w), _metrics(y, q2, a, w)
        mask = _sample(rng, s, p)
        result = linearized_survey_inference(
            y[mask], {"FAIRBIAS_BM": q1[mask], "BASE": q2[mask]}, a[mask],
            s[mask], p[mask], w[mask], [0, 1], alpha=0.05,
        )
        if result["status"] != "VALID":
            for key in failures:
                failures[key] += 1
            continue
        methods = result["methods"]
        ba = methods["FAIRBIAS_BM"]["balanced_accuracy"]
        eo = methods["FAIRBIAS_BM"]["eo_gap_interval"]
        paired = result["paired"]["BASE"]["balanced_accuracy"]
        checks = {"ba": (ba.ci_lower, ba.ci_upper, target1["ba"]),
                  "eo": (eo[0], eo[1], target1["eo"]),
                  "paired_ba": (paired.ci_lower, paired.ci_upper, target1["ba"] - target2["ba"])}
        for key, (lo, hi, target) in checks.items():
            finite = bool(np.isfinite(lo) and np.isfinite(hi) and np.isfinite(target))
            valid[key] += int(finite)
            failures[key] += int(not finite)
            covered[key] += int(finite and lo <= target <= hi)
    return {"M": M, "covered": covered, "valid_intervals": valid, "failures": failures,
            "valid_rate": {k: valid[k] / M for k in valid},
            "coverage_unconditional": {k: covered[k] / M for k in covered},
            "coverage_conditional": {k: (covered[k] / valid[k] if valid[k] else None) for k in covered}}


def main():
    start = time.perf_counter()
    scenarios = {
        "null_tied_equal_groups": run_scenario("null_tied_equal_groups", tied=True),
        "nonzero_gap_equal_groups": run_scenario("nonzero_gap_equal_groups"),
        "null_tied_rare_group": run_scenario("null_tied_rare_group", rare=True, tied=True),
        "nonzero_gap_rare_group": run_scenario("nonzero_gap_rare_group", rare=True),
    }
    elapsed = time.perf_counter() - start
    report = [
        "Gate: R2 production Taylor linearization coverage validation",
        "Status: COMPLETE — descriptive synthetic validation only; no formal survey-method acceptance.",
        "Files changed: scripts/validate_linearization_coverage.py; docs/reports/fairbias_codex_takeover_20260915_154733Z/survey_linearization_coverage_report.md",
        "Commands executed: `PYTHONPATH=src .venv311/bin/python -B scripts/validate_linearization_coverage.py`",
        "Permissions requested: None; synthetic arrays only, no NHIS/data/network/install operations.",
        "Tests executed: M=200 per scenario; production `linearized_survey_inference`; four preregistered scenarios.",
        f"Exact test results: elapsed_seconds={elapsed:.3f}; seed={SEED}; M={M}; see aggregate results below.",
        "Input hashes: Not applicable; no external inputs.",
        "Output hashes: Not generated; aggregate-only report.",
        "Row counts: Each synthetic population has 96 rows; each sampled design has 48 rows.",
        "Assumptions: Three strata, four PSUs per stratum, two sampled PSUs per stratum; Taylor covariance omits FPC; target is the full finite population.",
        "Unresolved issues: Coverage is descriptive and finite-population specific; rare-group and nonsmooth EO results are not formal guarantees.",
        "Git diff summary: Added one synthetic validation script and its aggregate report.",
        "Proposed next step: Parent review of coverage and evaluation-boundary findings before any scientific claim.",
        "", "## Aggregate results", "", "```json", json.dumps({"elapsed_seconds": elapsed, "seed": SEED, "scenarios": scenarios}, indent=2), "```",
        "", "## Interpretation", "",
        "BA uses the production single-metric t interval. EO uses production simultaneous rate intervals followed by conservative projection. Paired BA uses the shared influence-difference covariance and single-metric t interval. Identical policies are expected to have exact paired BA cancellation. These results do not validate NHIS variance or finite-sample coverage.",
        "", "STOP — waiting for Codex review.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "elapsed_seconds": elapsed, "scenarios": scenarios}, indent=2))


if __name__ == "__main__":
    main()
