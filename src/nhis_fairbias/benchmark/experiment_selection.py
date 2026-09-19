"""Freeze configurations using means of seed metrics, never the best seed."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .experiment_registry import OPERATING_POINTS
from .experiment_worker import file_sha, write_json


def aggregate_configuration(config, records):
    by_seed = {r["seed"]: r for r in records}
    if len(by_seed) != len(records) or set(by_seed) != set(config["seeds"]):
        return {"status": "INCOMPLETE_SEEDS", "candidate_id": config["candidate_id"]}
    if any(r["status"] != "VALID" for r in records):
        return {"status": "INCOMPLETE_CONFIGURATION", "candidate_id": config["candidate_id"],
                "seed_statuses": {str(k): r["status"] for k, r in by_seed.items()}}
    result = {"candidate_id": config["candidate_id"], "status": "VALID", "complexity": config["complexity"],
              "seed_metrics": {str(k): r["metrics_S"] for k, r in by_seed.items()}}
    for metric in ("balanced_accuracy", "eo_gap", "dp_gap"):
        values = [r["metrics_S"].get(metric) for r in records]
        result[metric] = float(np.mean(values)) if all(v is not None and np.isfinite(v) for v in values) else None
    ap = [r["risk_S"].get("average_precision") for r in records]
    result["average_precision"] = float(np.mean(ap)) if all(v is not None and np.isfinite(v) for v in ap) else None
    return result


def select_aggregate(candidates, tau):
    valid = [c for c in candidates if c["status"] == "VALID" and c["balanced_accuracy"] is not None and c["eo_gap"] is not None]
    feasible = [c for c in valid if c["eo_gap"] <= tau]
    if feasible:
        winner = min(feasible, key=lambda c: (-c["balanced_accuracy"], c["eo_gap"], c["complexity"], c["candidate_id"]))
        return {"status": "FEASIBLE", "selected_candidate_id": winner["candidate_id"], "tau": tau}
    boundary = min(valid, key=lambda c: (c["eo_gap"], -c["balanced_accuracy"], c["complexity"], c["candidate_id"])) if valid else None
    return {"status": "NO_FEASIBLE_CONFIGURATION" if valid else "NOT_ESTIMABLE", "selected_candidate_id": None,
            "boundary_candidate_id": boundary["candidate_id"] if boundary else None, "tau": tau}


def freeze_selection(run):
    run = Path(run)
    registration = json.loads((run / "registration.json").read_text())
    matrix = registration["candidates"]
    summaries, conditions, artifacts = [], defaultdict(list), {}
    unfinished = []
    for config in matrix:
        if config["status"] != "REGISTERED":
            summaries.append(config)
            continue
        records = []
        for seed in config["seeds"]:
            path = run / "jobs" / f'{config["candidate_id"]}_s{seed}'
            if not (path / "result.json").exists():
                unfinished.append(path.name)
                continue
            receipt = json.loads((path / "receipt.json").read_text())
            mandatory = {"job.json", "result.json", "worker.log"}
            if not mandatory.issubset(receipt.get("files", {})):
                raise ValueError("Job receipt omits mandatory evidence: " + path.name)
            if any(file_sha(path / name) != expected for name, expected in receipt["files"].items()):
                raise ValueError("Job evidence hash mismatch: " + path.name)
            result = json.loads((path / "result.json").read_text())
            job = json.loads((path / "job.json").read_text())
            expected_data = registration["prepared"][config["arm_id"]]
            if job.get("config") != config or job.get("seed") != seed or job.get("source_identity") != registration["source_identity"]:
                raise ValueError("Job identity differs from registration: " + path.name)
            if job.get("data_identity") != expected_data["data_identity"] or job.get("data_sha256") != expected_data["sha256"]:
                raise ValueError("Job data identity differs from registration: " + path.name)
            if result.get("candidate_id") != config["candidate_id"] or result.get("seed") != seed:
                raise ValueError("Result candidate/seed identity differs from registered job: " + path.name)
            if result["status"] == "VALID" and (result.get("source_identity") != registration["source_identity"] or result.get("data_identity") != expected_data["data_identity"]):
                raise ValueError("Valid result has wrong source/data identity: " + path.name)
            if result["status"] == "VALID" and not {"model.joblib", "predictions_S.npz"}.issubset(receipt["files"]):
                raise ValueError("Valid job receipt omits model or S predictions: " + path.name)
            if result["status"] == "VALID" and (receipt["returncode"] != 0 or receipt["termination"] is not None or not result.get("reload_verified")):
                raise ValueError("Successful job lacks successful independent receipt: " + path.name)
            records.append(result)
        aggregated = aggregate_configuration(config, records)
        summaries.append({"config": config, **aggregated})
        key = (config["arm_id"], config["backbone"], config["method"], config["training_weighted"])
        conditions[key].append(aggregated)
    if unfinished:
        raise ValueError(f"Cannot freeze selection: {len(unfinished)} registered seed jobs remain unexecuted")
    selections = []
    for (arm, backbone, method, weighted), candidates in sorted(conditions.items()):
        for tau in OPERATING_POINTS:
            chosen = select_aggregate(candidates, tau)
            chosen.update(arm_id=arm, backbone=backbone, method=method, training_weighted=weighted)
            selections.append(chosen)
        if method == "UNMITIGATED":
            valid = [c for c in candidates if c["status"] == "VALID" and c["average_precision"] is not None]
            chosen = min(valid, key=lambda c: (-c["average_precision"], -c["balanced_accuracy"], c["complexity"], c["candidate_id"])) if valid else None
            selections.append({"arm_id": arm, "backbone": backbone, "method": method, "training_weighted": weighted,
                "status": "PREDICTION_REFERENCE" if chosen else "NOT_ESTIMABLE", "tau": None,
                "selected_candidate_id": chosen["candidate_id"] if chosen else None})
    selected_ids = {s.get("selected_candidate_id") or s.get("boundary_candidate_id") for s in selections} - {None}
    for config in matrix:
        if config["candidate_id"] not in selected_ids:
            continue
        for seed in config["seeds"]:
            rel = f'jobs/{config["candidate_id"]}_s{seed}/model.joblib'
            artifacts[rel] = file_sha(run / rel)
    write_json(run / "selection_freeze.json", {"registration_sha256": file_sha(run / "registration.json"),
        "aggregation": "mean of each seed's metric; incomplete seed sets excluded", "selections": selections,
        "candidate_summaries": summaries, "selected_artifacts": artifacts,
        "pending_extensions": registration["pending_predeclared_extensions"],
        "evaluation_authorized": not registration["pending_predeclared_extensions"]})
    return selections
