"""Survey statistics for an independently frozen completion contrast registry.

This module neither selects models nor authorizes access to S/T. The caller
validates complete seed sets, provenance, prediction order and the global
contrast registry before supplying one complete arm. All intervals condition
on the fitted models; seed variation is reported separately.
"""
from __future__ import annotations

from collections import Counter
import copy
import dataclasses
from pathlib import Path

import numpy as np
from scipy import stats

from .catalog_evaluation import _normalized_annual_design
from .data_contracts import ARM_SPECS
from .experiment_evaluation import (
    _linear_record, _project_mean, _projection_difference,
    _replicate_interval, _summary,
)
from .experiment_worker import write_json
from .survey_batch import bootstrap_metric_arrays
from .survey_linearization import linearized_survey_inference

BOOTSTRAP_SEED = 20260914
ENDPOINTS = ("balanced_accuracy", "eo_gap")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _adjusted_ba(record, df, slots):
    """Adjust the paired Taylor interval without recomputing its covariance."""
    if (record is None or record.status != "VALID" or df <= 0
            or not np.all(np.isfinite([record.point_estimate, record.std_error]))):
        return {"status": "NOT_ESTIMABLE", "family_endpoint_slots": slots}
    alpha = .05 / slots
    half = float(stats.t.ppf(1 - alpha / 2, df)) * record.std_error
    return {"status": "VALID", "estimate": record.point_estimate,
            "se": record.std_error, "lower": record.point_estimate - half,
            "upper": record.point_estimate + half, "alpha": alpha,
            "family_endpoint_slots": slots, "conditional_on_fixed_models": True,
            "label": "Bonferroni-t interval for paired mean-seed BA"}


def evaluate_statistics(T, arm_id, selections, q_models, individual, output, *,
                        contrasts, family_sizes, B=2000):
    """Return selection rows and every registered pair, including unavailable ones.

    ``family_sizes`` counts global BA/EO endpoint slots, not pair rows and not
    only estimable endpoints. ``contrasts`` contains this arm's complete subset
    of the pre-frozen registry. Nominal, within-family and union-family bands
    are distinct. The caller must not replace the global sizes with arm counts.
    """
    _require(arm_id in ARM_SPECS, "Unknown statistics arm")
    _require(type(B) is int and 2 <= B <= 10000, "Invalid bootstrap count")
    _require(isinstance(family_sizes, dict)
             and set(family_sizes) in ({"primary", "secondary"}, {"primary", "secondary", "union"})
             and all(type(v) is int and v >= 0 for v in family_sizes.values()),
             "Global primary/secondary endpoint counts are required")
    union_size = family_sizes["primary"] + family_sizes["secondary"]
    _require("union" not in family_sizes or family_sizes["union"] == union_size,
             "Union endpoint count must equal primary plus secondary")
    _require(isinstance(selections, list) and isinstance(contrasts, list), "Invalid frozen rows")
    by_id = {s["selection_id"]: s for s in selections}
    _require(len(by_id) == len(selections), "Duplicate selection identity")
    _require(all(s["arm_id"] == arm_id and isinstance(s["model_ids"], list)
                 and len(s["model_ids"]) == len(set(s["model_ids"])) for s in selections),
             "Selection arm or model membership mismatch")
    _require(len({s["study_branch"] for s in selections}) <= 1, "Mixed study branches")
    expected_models = {k for s in selections for k in s["model_ids"]}
    _require(set(q_models) == set(individual) == expected_models, "Incomplete arm predictions or metrics")
    _require(len({c["contrast_id"] for c in contrasts}) == len(contrasts), "Duplicate contrast identity")
    counts = Counter()
    seen_pairs = set()
    for c in contrasts:
        ref, base, family = (c["reference_selection_id"], c["comparison_selection_id"], c["family"])
        _require(ref in by_id and base in by_id and ref != base and family in {"primary", "secondary"},
                 "Unknown or invalid contrast membership")
        _require((ref, base, family) not in seen_pairs, "Duplicate contrast endpoints")
        seen_pairs.add((ref, base, family))
        _require(all(by_id[ref].get(k) == by_id[base].get(k)
                     for k in ("backbone", "training_weighted")),
                 "Contrast conditions differ")
        matched_fixed_bm = (c.get("kind") == "parent_fixed_BM"
                            and by_id[base]["method"] == "FAIRBIAS_BM"
                            and by_id[base]["status"] in {"FIXED_ABLATION", "NO_VALID_FIXED_ABLATION"}
                            and by_id[base]["tau"] is None and by_id[ref]["tau"] == .10)
        _require(by_id[ref]["tau"] == by_id[base]["tau"] or matched_fixed_bm,
                 "Contrast tau differs without a registered fixed BM anchor")
        counts[family] += len(ENDPOINTS)
    _require(all(family_sizes[f] >= n for f, n in counts.items()),
             "Family size cannot omit registered endpoint slots")
    output = Path(output)
    individual_path = output / (arm_id + "_individual_model_metrics.json")
    replicate_path = output / (arm_id + "_replicate_metric_arrays.npz")
    _require(not individual_path.exists() and not replicate_path.exists(), "Statistics outputs already exist")

    design = _normalized_annual_design(T)
    args = (design.expand(T.y), q_models, design.expand(T.A), design.strata,
            design.psus, design.weights, ARM_SPECS[arm_id]["expected_categories"])
    linear = (linearized_survey_inference(*args, domain_mask=design.domain_mask, alpha=.05)
              if q_models else {"status": "NO_VALID_MODEL", "methods": {}})
    valid_design = linear["status"] == "VALID"
    df = linear.get("degrees_of_freedom", 0)
    adjusted = {name: linearized_survey_inference(*args, domain_mask=design.domain_mask,
                                                 alpha=.05 / size)
                for name, size in {**family_sizes, "union": union_size}.items()
                if size > 0 and q_models}
    replicates = (bootstrap_metric_arrays(*args, domain_mask=design.domain_mask,
                                          B=B, seed=BOOTSTRAP_SEED)
                  if valid_design else {k: np.full((B, 3), np.nan) for k in q_models})
    individual = copy.deepcopy(individual)
    for key, entry in individual.items():
        entry["simultaneous_rate_intervals_95"] = {
            label: dataclasses.asdict(metric) for label, metric in
            linear.get("methods", {}).get(key, {}).get("rates", {}).items()}

    rows, mean_q, mean_replicates = [], {}, {}
    for selection in selections:
        key, models = selection["selection_id"], selection["model_ids"]
        row = {**selection, "sample_n": len(T), "design_n": len(design.weights),
               "df": df, "frozen_seed_models": list(models),
               "conditional_on_fixed_models": True,
               "evaluation_status": ("VALID" if valid_design else "DESIGN_NOT_ESTIMABLE")
                                    if models else "NO_VALID_MODEL"}
        if models:
            mean_replicates[key] = np.mean([replicates[k] for k in models], axis=0)
            mean_q[key] = np.mean([q_models[k] for k in models], axis=0)
            for index, metric in enumerate(("balanced_accuracy", "dp_gap", "eo_gap")):
                row[metric] = _summary([individual[k]["weighted"][metric] for k in models])
                row["unweighted_" + metric] = _summary([individual[k]["unweighted"][metric] for k in models])
                row[metric]["bootstrap_descriptive"] = _replicate_interval(
                    row[metric]["mean"], mean_replicates[key][:, index], df)
            for metric in ("average_precision", "auroc", "brier"):
                row[metric] = _summary([individual[k]["risk"].get(metric, np.nan) for k in models])
                row["unweighted_" + metric] = _summary([
                    individual[k]["unweighted_risk"].get(metric, np.nan) for k in models])
                if all("base_risk" in individual[k] for k in models):
                    row["untouched_base_" + metric] = _summary([
                        individual[k]["base_risk"].get(metric, np.nan) for k in models])
                    row["unweighted_untouched_base_" + metric] = _summary([
                        individual[k]["unweighted_base_risk"].get(metric, np.nan) for k in models])
            for metric in ("dp_gap", "eo_gap"):
                row[metric]["projection_95"] = _project_mean(linear, models, metric)
                row[metric]["projection_families"] = {
                    family: _project_mean(value, models, metric) for family, value in adjusted.items()}
            ba = linearized_survey_inference(args[0], {"MEAN_POLICY": mean_q[key]}, *args[2:],
                                             domain_mask=design.domain_mask, alpha=.05)
            row["balanced_accuracy"]["taylor_95"] = _linear_record(
                ba.get("methods", {}).get("MEAN_POLICY", {}).get("balanced_accuracy"))
            if valid_design and any(row[m]["projection_95"]["status"] != "VALID"
                                    for m in ("dp_gap", "eo_gap")):
                row["evaluation_status"] = "PARTIALLY_ESTIMABLE"
        rows.append(row)

    evaluated = {r["selection_id"]: r for r in rows}
    pairs = []
    for contrast in contrasts:
        ref = evaluated[contrast["reference_selection_id"]]
        base = evaluated[contrast["comparison_selection_id"]]
        family = contrast["family"]
        pair = {**contrast, "arm_id": arm_id, "backbone": ref["backbone"],
                "tau": ref["tau"], "training_weighted": ref["training_weighted"],
                "study_branch": ref["study_branch"], "reference_method": ref["method"],
                "comparison_method": base["method"], "reference_model_ids": ref["model_ids"],
                "comparison_model_ids": base["model_ids"], "family_endpoint_slots": family_sizes[family],
                "union_endpoint_slots": union_size, "endpoints": list(ENDPOINTS),
                "difference_direction": "reference_minus_comparison",
                "feasible_on_S": ref["status"] == base["status"] == "FEASIBLE",
                "conditional_on_fixed_models": True}
        ref_id, base_id = ref["selection_id"], base["selection_id"]
        if ref_id not in mean_q or base_id not in mean_q:
            pair["evaluation_status"] = "NO_VALID_MODEL"
            for metric in ENDPOINTS:
                pair["delta_" + metric] = {"estimate": None, "status": "NOT_ESTIMABLE"}
            pairs.append(pair)
            continue
        diffs = mean_replicates[ref_id] - mean_replicates[base_id]
        for metric, index in (("balanced_accuracy", 0), ("eo_gap", 2)):
            f, b = ref[metric]["mean"], base[metric]["mean"]
            point = f - b if f is not None and b is not None else None
            pair["delta_" + metric] = {"estimate": point,
                "bootstrap_descriptive": _replicate_interval(point, diffs[:, index], df)}
        paired = linearized_survey_inference(
            args[0], {"REFERENCE": mean_q[ref_id], "COMPARISON": mean_q[base_id]}, *args[2:],
            domain_mask=design.domain_mask, alpha=.05, reference_method="REFERENCE")
        record = paired.get("paired", {}).get("COMPARISON", {}).get("balanced_accuracy")
        pair["delta_balanced_accuracy"]["taylor_95"] = _linear_record(record)
        pair["delta_balanced_accuracy"]["family_interval"] = _adjusted_ba(record, df, family_sizes[family])
        pair["delta_balanced_accuracy"]["union_family_interval"] = _adjusted_ba(record, df, union_size)
        for label, source in (("projection_95", None), ("projection_family", family),
                              ("projection_union_family", "union")):
            f = ref["eo_gap"]["projection_95"] if source is None else ref["eo_gap"]["projection_families"][source]
            b = base["eo_gap"]["projection_95"] if source is None else base["eo_gap"]["projection_families"][source]
            pair["delta_eo_gap"][label] = _projection_difference(f, b)
        for metric in ("average_precision", "auroc", "brier"):
            f, b = ref[metric]["mean"], base[metric]["mean"]
            pair["delta_" + metric] = f - b if f is not None and b is not None else None
        statuses = (pair["delta_balanced_accuracy"]["taylor_95"]["status"],
                    pair["delta_eo_gap"]["projection_95"]["status"])
        pair["evaluation_status"] = ("DESIGN_NOT_ESTIMABLE" if not valid_design else
                                      "VALID" if all(s == "VALID" for s in statuses) else "PARTIALLY_ESTIMABLE")
        pairs.append(pair)

    output.mkdir(parents=True, exist_ok=True)
    write_json(individual_path, individual)
    np.savez_compressed(replicate_path, **replicates)
    return rows, pairs
