"""Evaluation-only loading of frozen policies; this module contains no fitting."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import joblib
import numpy as np
from scipy import stats

from .data_contracts import ARM_SPECS, load_arm_partitions, load_local_nhis_cohort
from .experiment_worker import file_sha, write_json
from .metrics import compute_survey_fairness_metrics
from .predictions import PredictionBundle
from .risk_metrics import compute_risk_metrics
from .survey_batch import bootstrap_metric_arrays
from .survey_linearization import linearized_survey_inference
from .cohort_reporting import summarize_partition, summarize_eligibility


def _summary(values):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        return {"mean": None, "seed_sd": None, "seed_range": None, "status": "NOT_ESTIMABLE"}
    return {"mean": float(values.mean()), "seed_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "seed_range": [float(values.min()), float(values.max())], "status": "VALID"}


def _replicate_interval(point, replicates, df, alpha=.05):
    vals = np.asarray(replicates, dtype=float)
    good = vals[np.isfinite(vals)]
    if point is None or not np.isfinite(point) or df <= 0 or len(good) < 2 or len(good) / max(len(vals), 1) < .95:
        return {"status": "NOT_ESTIMABLE", "valid_fraction": len(good) / max(len(vals), 1)}
    se = float(good.std(ddof=1))
    half = float(stats.t.ppf(1 - alpha/2, df)) * se
    return {"status": "VALID", "estimate": point, "se": se, "lower": point-half, "upper": point+half,
            "alpha": alpha, "valid_fraction": len(good)/len(vals), "conditional_on_fixed_models": True}


def _project_mean(linear, keys, metric):
    bands = [linear.get("methods", {}).get(k, {}).get(metric + "_interval", (np.nan, np.nan, "NOT_ESTIMABLE")) for k in keys]
    if not bands or any(v[2] != "VALID" or not np.all(np.isfinite(v[:2])) for v in bands):
        return {"status": "NOT_ESTIMABLE", "lower": None, "upper": None}
    return {"status": "VALID", "lower": float(np.mean([v[0] for v in bands])), "upper": float(np.mean([v[1] for v in bands])),
            "label": "conservative mean of simultaneously covered seed gap projections"}


def _projection_difference(f, b):
    if f["status"] != "VALID" or b["status"] != "VALID":
        return {"status": "NOT_ESTIMABLE", "lower": None, "upper": None}
    return {"status": "VALID", "lower": f["lower"]-b["upper"], "upper": f["upper"]-b["lower"],
            "label": "conservative simultaneous gap projection; no normal max-gap p-value"}


def _linear_record(record):
    if record is None:
        return {"status": "NOT_ESTIMABLE"}
    return {"status": record.status, "estimate": record.point_estimate, "se": record.std_error,
            "lower": record.ci_lower, "upper": record.ci_upper, "conditional_on_fixed_models": True,
            "label": "Taylor linearization with stratum-centered PSU influence and t critical value"}


def freeze_study(runs, output, *, repo_root=None, extension_resolutions=None):
    """Bind complete selections and analysis implementation before any T I/O."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    batches, evidence, fixed_ablations = {}, {}, {}
    resolutions = dict(extension_resolutions or {})
    if not runs:
        raise ValueError("At least one complete registered batch required")
    for run in map(Path, runs):
        registration = json.loads((run / "registration.json").read_text())
        selected = json.loads((run / "selection_freeze.json").read_text())
        if selected["registration_sha256"] != file_sha(run / "registration.json"):
            raise ValueError("Registration differs from selection freeze")
        if set(registration["pending_predeclared_extensions"]) - set(resolutions):
            raise ValueError("Unresolved extension admission")
        for rel, expected in registration["source_files"].items():
            if file_sha(root / rel) != expected:
                raise ValueError("Training implementation changed before study freeze: " + rel)
        for rel, expected in selected["selected_artifacts"].items():
            if file_sha(run / rel) != expected:
                raise ValueError("Selected model changed before study freeze")
        batches[str(run.resolve())] = file_sha(run / "selection_freeze.json")
        evidence[str((run / "registration.json").resolve())] = file_sha(run / "registration.json")
        # Retain predeclared fixed ablations even when S's tuned winner is a
        # different capacity/epsilon. This selection is parameter-based only.
        valid_ids = {r.get("candidate_id") for r in selected["candidate_summaries"] if r.get("status") == "VALID"}
        fixed_rows, fixed_files = [], {}
        for config in registration["candidates"]:
            params = config.get("params", {})
            base = params.get("estimator_params", {})
            bm_anchor = (config["method"] == "FAIRBIAS_BM" and not config.get("training_weighted") and params.get("epsilon_ratio") == 1.
                and ((config["backbone"] == "LR" and params.get("C", 1.) == 1.) or
                     (config["backbone"] == "GBDT" and base.get("n_estimators") == 100 and base.get("max_depth") == 2)))
            if not bm_anchor and config.get("family") not in {"fixed_ae_ablation", "fixed_geometry_ablation"}:
                continue
            cid = config["candidate_id"]
            valid = cid in valid_ids
            fixed_rows.append({"arm_id": config["arm_id"], "backbone": config["backbone"], "method": config["method"],
                "training_weighted": False, "tau": None, "status": "FIXED_ABLATION" if valid else "NO_VALID_FIXED_ABLATION",
                "selected_candidate_id": cid if valid else None, "registered_fixed_candidate_id": cid})
            if valid:
                for seed in config["seeds"]:
                    rel = f'jobs/{cid}_s{seed}/model.joblib'
                    fixed_files[rel] = file_sha(run / rel)
        fixed_ablations[str(run.resolve())] = {"selections": fixed_rows, "artifacts": fixed_files}
    for value in resolutions.values():
        if value.get("status") not in {"ADMITTED_AND_FROZEN", "NOT_SUPPORTED"} or not value.get("evidence_files"):
            raise ValueError("Resolution needs final disposition and concrete evidence")
        for path, expected in value["evidence_files"].items():
            if file_sha(path) != expected:
                raise ValueError("Extension admission evidence hash mismatch")
            evidence[str(Path(path).resolve())] = expected
    names = ("experiment_evaluation.py", "survey_batch.py", "survey_linearization.py", "survey_inference.py",
             "metrics.py", "risk_metrics.py", "data_contracts.py", "cohort_reporting.py", "paper_reporting.py")
    analysis = {str(Path(__file__).with_name(name).resolve()): file_sha(Path(__file__).with_name(name)) for name in names}
    write_json(output, {"status": "COMPLETE_SELECTION_FREEZE", "batch_selection_hashes": batches,
        "analysis_files": analysis, "evidence_files": evidence, "extension_resolutions": resolutions,
        "fixed_ablations": fixed_ablations,
        "primary_contrast_family_size": 20, "bootstrap_replicates": 2000,
        "primary_claim": "FairBias-BM application study; retrospective 2024 fixed-policy evaluation"})


def evaluate_frozen_run(run, output, *, study_freeze, repo_root=None, B=2000):
    """Require a supervisor's complete-study freeze before opening 2024.

    A study freeze resolves every predeclared extension (admitted batch or an
    evidenced NOT_SUPPORTED condition). This prevents early T results from
    influencing admission, capacity, or subsequent development choices.
    """
    run, output, study_freeze = Path(run), Path(output), Path(study_freeze)
    if B != 2000 or isinstance(B, bool):
        raise ValueError("Formal evaluation uses the registered B=2000; other budgets require a separate registration")
    final = json.loads(study_freeze.read_text())
    selected = json.loads((run / "selection_freeze.json").read_text())
    registration = json.loads((run / "registration.json").read_text())
    if final.get("status") != "COMPLETE_SELECTION_FREEZE":
        raise ValueError("Complete study selection has not been frozen")
    required_analysis = {str(Path(__file__).with_name(name).resolve()) for name in (
        "experiment_evaluation.py", "survey_batch.py", "survey_linearization.py", "survey_inference.py",
        "metrics.py", "risk_metrics.py", "data_contracts.py", "cohort_reporting.py", "paper_reporting.py")}
    if not required_analysis.issubset(final.get("analysis_files", {})):
        raise ValueError("Study freeze does not bind the complete analysis implementation")
    for path, expected in {**final["analysis_files"], **final.get("evidence_files", {})}.items():
        if file_sha(path) != expected:
            raise ValueError("Study analysis or evidence changed before T evaluation: " + path)
    if final.get("batch_selection_hashes", {}).get(str(run.resolve())) != file_sha(run / "selection_freeze.json"):
        raise ValueError("Study freeze does not cover this exact batch selection")
    if selected["registration_sha256"] != file_sha(run / "registration.json"):
        raise ValueError("Registration hash mismatch")
    unresolved = set(registration["pending_predeclared_extensions"]) - set(final.get("extension_resolutions", {}))
    if unresolved:
        raise ValueError("Unresolved predeclared extensions: " + str(sorted(unresolved)))
    if any(r.get("status") not in {"ADMITTED_AND_FROZEN", "NOT_SUPPORTED"} for r in final.get("extension_resolutions", {}).values()):
        raise ValueError("All extension resolutions must have a final admission disposition")
    fixed = final.get("fixed_ablations", {}).get(str(run.resolve()), {})
    selected["selections"] += fixed.get("selections", [])
    selected["selected_artifacts"].update(fixed.get("artifacts", {}))
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    for rel, expected in registration["source_files"].items():
        if file_sha(root / rel) != expected:
            raise ValueError("Registered implementation changed before frozen evaluation: " + rel)
    for rel, expected in selected["selected_artifacts"].items():
        if file_sha(run / rel) != expected:
            raise ValueError("Frozen model hash mismatch")
    output.mkdir(parents=True, exist_ok=False)
    cohort = load_local_nhis_cohort(root, years=(2024,))
    write_json(output / "cohort_eligibility_T.json", summarize_eligibility(cohort))
    write_json(output / "evaluation_manifest.json", {"study_freeze_sha256": file_sha(study_freeze),
        "selection_sha256": file_sha(run / "selection_freeze.json"), "evaluator_sha256": file_sha(__file__),
        "source_provenance": cohort.attrs["source_provenance"], "B": B, "bootstrap_seed": 20260914,
        "claim": "Retrospective annual domain estimation; fixed learned policies; no fit/refit or configuration selection on T"})
    summaries, all_paired = [], []
    for arm, spec in ARM_SPECS.items():
        T = load_arm_partitions(cohort, arm)["evaluation_T"]
        design = T.annual_design
        arm_selections = [s for s in selected["selections"] if s["arm_id"] == arm]
        ids = {s.get("selected_candidate_id") or s.get("boundary_candidate_id") for s in arm_selections} - {None}
        q_models, individual = {}, {}
        for rel in selected["selected_artifacts"]:
            model_key = Path(rel).parent.name
            candidate = model_key.rsplit("_s", 1)[0]
            if candidate not in ids:
                continue
            model = joblib.load(run / rel)
            X = T.X_semantic if model["semantic_input"] else model["preprocessor"].transform(T.X_semantic)
            bundle = model["policy"].predict(X, T.A)
            q_models[model_key] = design.expand(bundle.q_decision)
            individual[model_key] = {"weighted": compute_survey_fairness_metrics(T.y, bundle.q_decision, T.A, T.WTFA_A, spec["expected_categories"]),
                "unweighted": compute_survey_fairness_metrics(T.y, bundle.q_decision, T.A, np.ones(len(T)), spec["expected_categories"]),
                "risk": compute_risk_metrics(T.y, bundle, T.WTFA_A, T.A, spec["expected_categories"]),
                "unweighted_risk": compute_risk_metrics(T.y, bundle, np.ones(len(T)), T.A, spec["expected_categories"])}
            arrays = {"q": bundle.q_decision}
            if bundle.p_event is not None:
                arrays["p"] = bundle.p_event
                individual[model_key]["threshold_05"] = compute_survey_fairness_metrics(T.y, (bundle.p_event>=.5).astype(float), T.A, T.WTFA_A, spec["expected_categories"])
            elif hasattr(model["policy"]._adapter, "predict_event_probability"):
                p = model["policy"]._adapter.predict_event_probability(X, T.A)
                arrays["base_p"] = p
                individual[model_key]["base_risk"] = compute_risk_metrics(T.y, PredictionBundle(p, bundle.q_decision), T.WTFA_A, T.A, spec["expected_categories"])
                individual[model_key]["unweighted_base_risk"] = compute_risk_metrics(T.y, PredictionBundle(p, bundle.q_decision), np.ones(len(T)), T.A, spec["expected_categories"])
            np.savez_compressed(output / (arm + "_" + model_key + "_predictions_T.npz"), **arrays)
        if not q_models:
            summaries.extend({**s, "evaluation_status": "NO_VALID_MODEL"} for s in arm_selections)
            continue
        write_json(output / (arm + "_cohort_T.json"), summarize_partition(T, spec["expected_categories"],
            fitted_preprocessor=model["preprocessor"]))
        args = (design.expand(T.y), q_models, design.expand(T.A), design.strata, design.psus, design.weights, spec["expected_categories"])
        linear = linearized_survey_inference(*args, domain_mask=design.domain_mask, alpha=.05)
        adjusted = linearized_survey_inference(*args, domain_mask=design.domain_mask, alpha=.05/20)
        design_valid = linear["status"] == "VALID"
        replicates = (bootstrap_metric_arrays(*args, domain_mask=design.domain_mask, B=B) if design_valid else
                      {k: np.full((B, 3), np.nan) for k in q_models})
        for key in individual:
            individual[key]["simultaneous_rate_intervals_95"] = {
                label: dataclasses.asdict(metric) for label, metric in
                linear.get("methods", {}).get(key, {}).get("rates", {}).items()}
        # No serialization of per-person arrays into public aggregate JSON.
        write_json(output / (arm + "_individual_model_metrics.json"), individual)
        df = linear.get("degrees_of_freedom", 0)
        condition_replicates, condition_q, arm_rows = {}, {}, []
        for position, selection in enumerate(arm_selections):
            cid = selection.get("selected_candidate_id") or selection.get("boundary_candidate_id")
            keys = [k for k in q_models if k.rsplit("_s", 1)[0] == cid]
            row = {**selection, "sample_n": len(T), "design_n": len(design.weights), "df": df,
                   "evaluation_status": ("VALID" if design_valid else "DESIGN_NOT_ESTIMABLE") if keys else "NO_VALID_MODEL", "frozen_seed_models": keys}
            if keys:
                mean_reps = np.mean([replicates[k] for k in keys], axis=0)
                for j, metric in enumerate(("balanced_accuracy", "dp_gap", "eo_gap")):
                    row[metric] = _summary([individual[k]["weighted"][metric] for k in keys])
                    row["unweighted_"+metric] = _summary([individual[k]["unweighted"][metric] for k in keys])
                    row[metric]["bootstrap_descriptive"] = _replicate_interval(row[metric]["mean"], mean_reps[:, j], df)
                for metric in ("average_precision", "auroc", "brier"):
                    values = [individual[k]["risk"].get(metric, np.nan) for k in keys]
                    row[metric] = _summary(values)
                    row["unweighted_"+metric] = _summary([individual[k]["unweighted_risk"].get(metric, np.nan) for k in keys])
                    if all("base_risk" in individual[k] for k in keys):
                        row["untouched_base_"+metric] = _summary([individual[k]["base_risk"].get(metric, np.nan) for k in keys])
                        row["unweighted_untouched_base_"+metric] = _summary([individual[k]["unweighted_base_risk"].get(metric, np.nan) for k in keys])
                for metric in ("dp_gap", "eo_gap"):
                    row[metric]["projection_95"] = _project_mean(linear, keys, metric)
                    row[metric]["projection_primary_family"] = _project_mean(adjusted, keys, metric)
                if design_valid and any(row[m]["projection_95"]["status"] != "VALID" for m in ("dp_gap", "eo_gap")):
                    row["evaluation_status"] = "PARTIALLY_ESTIMABLE"
                condition_replicates[position] = mean_reps
                # BA is linear in q at fixed Y and weights; using mean q here
                # exactly equals mean seed BA. This identity is NOT used for EO.
                condition_q[position] = np.mean([q_models[k] for k in keys], axis=0)
                mean_ba = linearized_survey_inference(args[0], {"MEAN_POLICY": condition_q[position]}, *args[2:],
                    domain_mask=design.domain_mask, alpha=.05)
                row["balanced_accuracy"]["taylor_95"] = _linear_record(mean_ba.get("methods", {}).get("MEAN_POLICY", {}).get("balanced_accuracy"))
            summaries.append(row)
            arm_rows.append(row)
        for i, fair in enumerate(arm_rows):
            if fair["method"] != "FAIRBIAS_BM" or i not in condition_replicates:
                continue
            for j, base in enumerate(arm_rows):
                if base["method"] == "FAIRBIAS_BM" or j not in condition_replicates:
                    continue
                if any(fair.get(k) != base.get(k) for k in ("backbone", "training_weighted", "tau")):
                    continue
                if (fair["status"] == "FIXED_ABLATION") != (base["status"] == "FIXED_ABLATION"):
                    continue
                primary = (arm in ("arm_001", "arm_003") and fair["backbone"] == "LR" and not fair["training_weighted"]
                           and fair["tau"] == .10 and base["method"] in ("REWEIGHING", "LFR_RECONSTRUCTED", "EG_DP", "EG_EO", "TO_EO"))
                row = {"arm_id": arm, "backbone": fair["backbone"], "tau": fair["tau"], "training_weighted": fair["training_weighted"],
                       "reference_method": "FAIRBIAS_BM", "comparison_method": base["method"], "in_primary_family_20": primary,
                       "feasible_on_S": fair["status"] == base["status"] == "FEASIBLE"}
                diff_reps = condition_replicates[i] - condition_replicates[j]
                for k, idx in (("balanced_accuracy", 0), ("eo_gap", 2)):
                    fp, bp = fair[k]["mean"], base[k]["mean"]
                    point = fp-bp if fp is not None and bp is not None else None
                    row["delta_"+k] = {"estimate": point, "bootstrap_descriptive": _replicate_interval(point, diff_reps[:, idx], df)}
                paired_args = (args[0], {"FAIRBIAS_BM": condition_q[i], "BASE": condition_q[j]}, *args[2:])
                paired_linear = linearized_survey_inference(*paired_args, domain_mask=design.domain_mask, alpha=.05)
                record = paired_linear.get("paired", {}).get("BASE", {}).get("balanced_accuracy")
                row["delta_balanced_accuracy"].update(_linear_record(record))
                if primary and record is not None:
                    half = float(stats.t.ppf(1-(.05/20)/2, df))*record.std_error
                    row["delta_balanced_accuracy"]["family_20_interval"] = {"status": record.status,
                        "lower": record.point_estimate-half, "upper": record.point_estimate+half, "alpha": .05/20}
                for band in ("projection_95", "projection_primary_family"):
                    f, b = fair["eo_gap"][band], base["eo_gap"][band]
                    row["delta_eo_gap"][band] = _projection_difference(f, b)
                for k in ("average_precision", "auroc", "brier"):
                    fp, bp = fair[k]["mean"], base[k]["mean"]
                    row["delta_"+k] = fp-bp if fp is not None and bp is not None else None
                all_paired.append(row)
        np.savez_compressed(output / (arm+"_replicate_metric_arrays.npz"), **replicates)
    write_json(output / "summary_T.json", {"selections": summaries, "paired_contrasts": all_paired,
        "evaluation_manifest_sha256": file_sha(output / "evaluation_manifest.json"),
        "intervals": "EO bootstrap is descriptive; projection bands address nonsmooth max/range using simultaneous rate intervals",
        "training_variation": "seed SD/range is separate from design SE; no seed is selected"})
    return summaries


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "evaluate"))
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study-freeze", type=Path)
    options = parser.parse_args()
    if options.phase == "freeze":
        freeze_study(options.run, options.output)
    else:
        if len(options.run) != 1 or options.study_freeze is None:
            parser.error("evaluate requires exactly one run and --study-freeze")
        evaluate_frozen_run(options.run[0], options.output, study_freeze=options.study_freeze)
