"""One isolated, auditable F/C/S fit. Never reads the 2024 input file."""
from __future__ import annotations

import copy
import json
import hashlib
import math
import os
import pathlib
import resource
import sys
import time
import traceback
import warnings

import joblib
import numpy as np

from .adapters.base import NotSupportedError
from .experiment_registry import identity, make_adapter
from .metrics import compute_survey_fairness_metrics
from .predictions import FrozenDecisionPolicy, PredictionBundle
from .risk_metrics import compute_risk_metrics


# ``resource.getrusage().ru_maxrss`` is reported in bytes on macOS and KiB on
# Linux.  Keep the conversion in one small helper so scheduler receipts and
# worker results use the same unit on every supported platform.
_THREAD_CONTROL_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _ru_maxrss_bytes(usage=None, platform=None):
    """Return ``ru_maxrss`` in bytes for macOS and Linux-style hosts.

    Python exposes the native value without a unit marker.  macOS reports
    bytes, while Linux reports KiB; converting at the worker boundary avoids
    making the scheduler or downstream reports platform-dependent.  The
    optional arguments keep the conversion directly testable without a real
    high-memory allocation or platform switch.
    """
    if usage is None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
    if platform is None:
        platform = sys.platform
    value = float(usage.ru_maxrss)
    if not math.isfinite(value) or value < 0:
        raise ValueError("ru_maxrss must be a finite non-negative value")
    # macOS is the supported byte-reporting platform; Linux and other Unix
    # platforms use the POSIX/Linux KiB convention exposed by Python here.
    multiplier = 1 if platform == "darwin" else 1024
    return int(value * multiplier)


def _resource_telemetry():
    """Capture normalized process usage and the scheduler's thread controls."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    thread_environment = {name: os.environ.get(name) for name in _THREAD_CONTROL_ENV_VARS}
    return {
        "max_rss_bytes": _ru_maxrss_bytes(usage),
        "ru_maxrss_native": float(usage.ru_maxrss),
        "ru_maxrss_unit": "bytes" if sys.platform == "darwin" else "KiB",
        "cpu_user_seconds": float(usage.ru_utime),
        "cpu_system_seconds": float(usage.ru_stime),
        "thread_controls": thread_environment,
        "thread_controls_all_one": all(value == "1" for value in thread_environment.values()),
    }


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def write_json(path, value):
    path = pathlib.Path(path)
    with path.open("x") as handle:
        json.dump(json_value(value), handle, indent=2, allow_nan=False)


def file_sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def representation_key(config, seed, data_identity):
    method, params = config["method"], config["params"]
    if method == "FAIRBIAS_BM" or method.startswith("FAIRBIAS_GEOMETRY_"):
        defaults = {"epsilon_ratio": .5, "max_iterations": 50, "max_geometry_evaluations": 20000,
                    "geometry_profile": "stress_elbow", "phi_threshold": 100., "algorithm_version": "application_v1"}
        spec = {k: params.get(k, v) for k, v in defaults.items()}
        family = "FAIRBIAS_BM"
    elif method == "LFR_RECONSTRUCTED":
        spec = {k: params[k] for k in ("k", "Az", "Ax", "Ay", "maxiter", "maxfun")}
        family = method
    else:
        return None
    return identity({"data": data_identity, "method": family, "seed": seed, "representation": spec})


def _fit_with_representation_cache(adapter, config, seed, data, cache):
    F = data["partitions"]["fitting_F"]
    weighted = F.WTFA_A / np.mean(F.WTFA_A) if config.get("training_weighted") else None
    method = config["method"]
    if method == "FAIRBIAS_BM" or method.startswith("FAIRBIAS_GEOMETRY_"):
        key = representation_key(config, seed, data["data_identity"])
        model_path, status_path = cache / (key + ".joblib"), cache / (key + ".json")
        if status_path.exists():
            recorded = json.loads(status_path.read_text())
            if recorded["status"] != "VALID":
                raise RuntimeError("CACHED_" + recorded["status"])
            if recorded.get("key") != key or file_sha(model_path) != recorded.get("model_sha256"):
                raise ValueError("Representation cache identity/hash mismatch")
            learned = joblib.load(model_path)
            learned.C, learned.backbone = adapter.C, adapter.backbone
            learned.estimator_params = adapter.estimator_params
            adapter = learned
        else:
            try:
                adapter.fit_representation(F.X_semantic, F.y, F.A)
                if not adapter.converged_:
                    raise RuntimeError(str(adapter.termination_reason_).upper())
                joblib.dump(adapter, model_path, compress=3)
                write_json(status_path, {"status": "VALID", "key": key, "model_sha256": file_sha(model_path), "manifest": adapter.fit_manifest_})
            except Exception as exc:
                write_json(status_path, {"status": str(exc), "manifest": adapter.fit_manifest_})
                raise
        adapter.fit_predictor(F.X_semantic, F.y, sample_weight=weighted)
        return adapter, key
    if method == "LFR_RECONSTRUCTED":
        key = representation_key(config, seed, data["data_identity"])
        model_path, status_path = cache / (key + ".joblib"), cache / (key + ".json")
        if status_path.exists():
            recorded = json.loads(status_path.read_text())
            if recorded["status"] != "VALID":
                raise RuntimeError("CACHED_" + recorded["status"])
            if recorded.get("key") != key or file_sha(model_path) != recorded.get("model_sha256"):
                raise ValueError("Representation cache identity/hash mismatch")
            learned = joblib.load(model_path)
            adapter.lfr = learned.lfr
            adapter.privileged_val_, adapter.unprivileged_val_ = learned.privileged_val_, learned.unprivileged_val_
            adapter.optimization_result_, adapter.converged_ = learned.optimization_result_, learned.converged_
            adapter.clf.fit(adapter._transform_X(data["X_F"], F.A), F.y)
        else:
            adapter.fit(data["X_F"], F.y, F.A)
            status = "VALID" if adapter.converged_ else "BUDGET_EXHAUSTED_OR_OPTIMIZATION_FAILURE"
            if adapter.converged_:
                joblib.dump(adapter, model_path, compress=3)
            write_json(status_path, {"status": status, "key": key,
                "model_sha256": file_sha(model_path) if adapter.converged_ else None,
                "diagnostics": adapter.optimization_result_})
            if not adapter.converged_:
                raise RuntimeError(status)
        return adapter, key
    adapter.fit(data["X_F"], F.y, F.A, sample_weight=weighted)
    return adapter, None


def execute_job(job_path):
    job_path = pathlib.Path(job_path)
    job = json.loads(job_path.read_text())
    out = job_path.parent
    config, seed = job["config"], job["seed"]
    start = time.monotonic()
    result = {"candidate_id": config["candidate_id"], "seed": seed, "status": "STARTED",
              "source_identity": job["source_identity"], "training_weighted": config.get("training_weighted", False)}
    adapter = None
    try:
        if file_sha(job["data_path"]) != job["data_sha256"]:
            raise ValueError("Prepared F/C/S file hash mismatch")
        data = joblib.load(job["data_path"])
        if data["data_identity"] != job["data_identity"]:
            raise ValueError("Prepared F/C/S data identity mismatch")
        if "evaluation_T" in data["partitions"] or set(p.year for p in data["partitions"].values()) - {2022, 2023}:
            raise ValueError("F/C/S worker refuses evaluation data")
        F, C, S = [data["partitions"][r] for r in ("fitting_F", "calibration_C", "selection_S")]
        adapter = make_adapter(config, seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if config["method"] in ("TO_EO", "OXONFAIR_EO", "FRAPPE_EO"):
                adapter.fit_base(data["X_F"], F.y)
                adapter.calibrate(data["X_C"], C.y, C.A)
                cache_key = None
            elif config["method"] in ("FAIRBIAS_BM_AE", "FAIRBIAS_JOINT"):
                adapter.fit_development(F.X_semantic, F.y, F.A, C.X_semantic, C.y, C.A,
                    metadata={"F_ids": F.record_keys, "C_ids": C.record_keys})
                cache_key = None
            else:
                adapter, cache_key = _fit_with_representation_cache(adapter, config, seed, data, pathlib.Path(job["cache_path"]))
            semantic = config["method"].startswith("FAIRBIAS")
            policy = FrozenDecisionPolicy().fit_calibration(adapter, C.X_semantic if semantic else data["X_C"], C.y, C.A,
                metadata={"role": "calibration_C", "year": 2022})
            bundle = policy.predict(S.X_semantic if semantic else data["X_S"], S.A)
        groups = S.metadata["expected_categories"]
        result.update(status="VALID", data_identity=data["data_identity"], representation_cache_key=cache_key,
            sample_sizes={"F": len(F), "C": len(C), "S": len(S)}, threshold=policy.threshold,
            output_type=adapter.output_type, capabilities=adapter.get_capabilities(),
            metrics_S=compute_survey_fairness_metrics(S.y, bundle.q_decision, S.A, S.WTFA_A, groups),
            unweighted_metrics_S=compute_survey_fairness_metrics(S.y, bundle.q_decision, S.A, np.ones(len(S)), groups),
            risk_S=compute_risk_metrics(S.y, bundle, S.WTFA_A, S.A, groups),
            algorithm_manifest=getattr(adapter, "fit_manifest_", getattr(adapter, "provenance_", {})), optimization=getattr(adapter, "optimization_result_", {}),
            warnings=[{"category": key[0], "message": key[1], "count": count} for key, count in
                      __import__("collections").Counter((w.category.__name__, str(w.message)) for w in caught).items()])
        arrays = {"q": bundle.q_decision}
        if bundle.p_event is not None:
            arrays["p"] = bundle.p_event
            result["threshold_05_metrics_S"] = compute_survey_fairness_metrics(S.y, (bundle.p_event >= .5).astype(float), S.A, S.WTFA_A, groups)
        elif hasattr(adapter, "predict_event_probability"):
            base_p = adapter.predict_event_probability(data["X_S"], A=S.A)
            arrays["base_p"] = base_p
            result["base_risk_S"] = compute_risk_metrics(S.y, PredictionBundle(base_p, bundle.q_decision), S.WTFA_A, S.A, groups)
        np.savez_compressed(out / "predictions_S.npz", **arrays)
        artifact = {"policy": policy, "preprocessor": data["preprocessor"], "semantic_input": semantic,
                    "config": config, "seed": seed, "data_identity": data["data_identity"]}
        joblib.dump(artifact, out / "model.joblib", compress=3)
        reloaded = joblib.load(out / "model.joblib")
        probe = reloaded["policy"].predict(S.X_semantic.iloc[:32] if semantic else data["X_S"][:32], S.A[:32])
        if not np.allclose(probe.q_decision, bundle.q_decision[:32], rtol=0, atol=1e-12):
            raise ValueError("Persisted policy or prediction batch contract failed")
        result["reload_verified"] = True
    except NotSupportedError as exc:
        result.update(status="NOT_SUPPORTED", error=str(exc))
    except Exception as exc:
        reason = str(exc)
        result.update(status="BUDGET_EXHAUSTED" if "BUDGET_EXHAUSTED" in reason else "FAILED",
                      error=reason, error_type=type(exc).__name__, traceback=traceback.format_exc())
    if adapter is not None and result["status"] != "VALID":
        result["algorithm_manifest"] = getattr(adapter, "fit_manifest_", getattr(adapter, "provenance_", {}))
        result["termination_reason"] = getattr(adapter, "termination_reason_", None)
    result["elapsed_seconds"] = time.monotonic() - start
    telemetry = _resource_telemetry()
    result["max_rss_bytes"] = telemetry["max_rss_bytes"]
    result["resource_telemetry"] = telemetry
    write_json(out / "result.json", result)
    return result
