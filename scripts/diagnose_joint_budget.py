"""Bounded, aggregate-only F/C diagnostic; never evaluates or selects on S/T.

Run in an independent process. Hooks are restored even on the diagnostic
deadline. Existing registered source and result files are read-only inputs.
"""
from __future__ import annotations

import argparse
import cProfile
from collections import Counter
import functools
import hashlib
import json
import os
import pickle
from pathlib import Path
import pstats
import resource
import shutil
import signal
import time
from contextlib import ExitStack
from unittest.mock import patch


class DiagnosticDeadline(BaseException):
    """Escape candidate-level Exception handlers, without declaring success."""


class DiagnosticGeometryStop(BaseException):
    """Stop after a predeclared work prefix for equivalent-work timing."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    import joblib
    from fairbias import bias_metric
    from fairbias.evaluator import FairEvaluator
    from fairbias.enhancement import FairAccuracyEnhancement
    from fairbias.mitigation import FairBiasMitigation
    from nhis_fairbias.benchmark.adapters.adapter_fairbias_ae import FairBiasAEAdapter

    root = Path(__file__).resolve().parents[1]
    run_path = args.run.resolve()
    registration_path = run_path / "registration.json"
    registration = json.loads(registration_path.read_text())
    for name, expected in registration["source_files"].items():
        if sha(root / name) != expected:
            raise ValueError("Registered source mismatch: " + name)
    config = next(c for c in registration["candidates"] if c["candidate_id"] == args.candidate)
    if config["method"] not in {"FAIRBIAS_JOINT", "FAIRBIAS_BM_AE"} or args.seed not in config["seeds"]:
        raise ValueError("Not a registered AE/Joint job")
    output = args.output.resolve()
    if output.exists() or output == run_path or run_path in output.parents:
        raise ValueError("Output must be a fresh directory outside the registered run")
    output.mkdir(parents=True)
    shutil.copyfile(__file__, output / "diagnostic_source.py")
    prepared = run_path / "prepared" / (config["arm_id"] + ".joblib")
    expected = registration["prepared"][config["arm_id"]]
    if sha(prepared) != expected["sha256"]:
        raise ValueError("Prepared file hash mismatch")
    # Prepared payload contains F/C/S. Only F/C are retained and accessed;
    # no S columns, labels, scores, or predictions are examined.
    data = joblib.load(prepared)
    if data["data_identity"] != expected["data_identity"]:
        raise ValueError("Prepared identity mismatch")
    F, C = data["partitions"]["fitting_F"], data["partitions"]["calibration_C"]
    del data
    params = dict(config["params"])
    adapter = FairBiasAEAdapter(backbone=config["backbone"], random_state=args.seed, **params)
    counters, phase, durations = {}, ["initialization"], {}
    engines, commits = {}, {"BM": 0, "AE": 0}
    events = (output / "events.jsonl").open("x", buffering=1)
    started = time.monotonic()

    def event(kind, **fields):
        events.write(json.dumps({"elapsed": time.monotonic() - started, "event": kind,
                                 "phase": phase[-1], **fields}, allow_nan=False) + "\n")
        events.flush()

    def timed(name, fn):
        @functools.wraps(fn)
        def wrapped(*a, **kw):
            counters[name] = counters.get(name, 0) + 1
            tick = time.monotonic()
            is_phase = name in {"BM", "AE"}
            if is_phase:
                engines[name] = a[0]
                phase.append(name)
                event("phase_start", call=counters[name])
            try:
                value = fn(*a, **kw)
                if name == "geometry":
                    event("geometry_value", calls=counters[name],
                          digest=hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest())
                    if args.geometry_prefix and counters[name] >= args.geometry_prefix:
                        raise DiagnosticGeometryStop()
                if is_phase:
                    commits[name] += int(value[-1] is not None)
                    event("phase_end", call=counters[name], committed=value[-1] is not None,
                          state_digest=hashlib.sha256(pickle.dumps(value[1], protocol=5)).hexdigest())
                return value
            finally:
                durations[name] = durations.get(name, 0) + time.monotonic() - tick
                if name in {"geometry", "utility"}:
                    event(name, calls=counters[name], seconds=time.monotonic() - tick)
                if is_phase:
                    phase.pop()
        return wrapped

    def deadline(signum, frame):
        raise DiagnosticDeadline()

    profiler = cProfile.Profile()
    old_signal = signal.signal(signal.SIGALRM, deadline)
    result = {"diagnostic_only": True, "candidate_id": args.candidate, "seed": args.seed,
              "config": config, "registration_sha256": sha(registration_path),
              "source_identity": registration["source_identity"], "prepared_sha256": sha(prepared),
              "sample_sizes": {"F": len(F), "C": len(C)}, "S_T_evaluated": False,
              "wall_seconds_limit": args.seconds, "optimized": args.optimized,
              "geometry_prefix": args.geometry_prefix, "profile_enabled": args.profile,
              "diagnostic_script_sha256": sha(__file__), "status": "STARTED"}
    if args.optimized:
        result["optimization_source_hashes"] = {name: sha(root / name) for name in (
            "src/nhis_fairbias/benchmark/joint_budget_optimization.py",)}
        for name in result["optimization_source_hashes"]:
            shutil.copyfile(root / name, output / Path(name).name)
    optimization = None
    try:
        with ExitStack() as stack:
            for obj, attr, label in (
                (FairEvaluator, "calculate_epsilon", "geometry"),
                (FairBiasAEAdapter, "_utility", "utility"),
                (FairBiasMitigation, "mitigate_step", "BM"),
                (FairAccuracyEnhancement, "enhance_step", "AE"),
                (bias_metric, "compute_pairwise_divergences", "divergence"),
                (bias_metric, "compute_shapley_distance_matrix", "shapley"),
                (bias_metric, "_find_optimal_mds_components", "elbow"),
            ):
                stack.enter_context(patch.object(obj, attr, timed(label, getattr(obj, attr))))
            if args.optimized:
                from nhis_fairbias.benchmark.joint_budget_optimization import exact_geometry_acceleration
                optimization = stack.enter_context(exact_geometry_acceleration())
            signal.setitimer(signal.ITIMER_REAL, args.seconds)
            if args.profile:
                profiler.enable()
            adapter.fit_development(F.X_semantic, F.y, F.A, C.X_semantic, C.y, C.A,
                                    metadata={"F_ids": F.record_keys, "C_ids": C.record_keys})
            result["status"] = "FIT_COMPLETED_DIAGNOSTIC_ONLY"
    except DiagnosticDeadline:
        result["status"] = "DIAGNOSTIC_TIME_LIMIT"
    except DiagnosticGeometryStop:
        result["status"] = "DIAGNOSTIC_PREFIX_COMPLETED"
    except Exception as exc:
        # No raw rows or exception locals are included.
        result.update(status="FIT_EXCEPTION", error_type=type(exc).__name__, error=str(exc))
    finally:
        profiler.disable()
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_signal)
        result.update(elapsed=time.monotonic() - started, calls=counters, inclusive_seconds=durations,
                      utility_evaluations=getattr(adapter, "_utility_evaluations_", 0),
                      geometry_evaluations=getattr(adapter, "_geometry_evaluations_", 0),
                      mds_fits=len(getattr(adapter, "mds_diagnostics_", [])),
                      termination_reason=getattr(adapter, "termination_reason_", None),
                      optimization=optimization, completed_phase_commits=commits,
                      mds_status_counts=dict(Counter(r["status"] for r in getattr(adapter, "mds_diagnostics_", []))),
                      mds_recorded_best_iterations_sum=sum(r["iterations"] for r in getattr(adapter, "mds_diagnostics_", [])))
        ae_events = getattr(engines.get("AE"), "audit_trail", [])
        result["ae_audit"] = {"events": len(ae_events),
            "accepted": sum(int(e.accepted) for e in ae_events),
            "reasons": dict(Counter((e.rejection_reason or "ACCEPTED").split(":", 1)[0] for e in ae_events))}
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result["resource"] = {"user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime,
                              "maxrss_native": usage.ru_maxrss}
        result["registered_source_unchanged_after"] = all(
            sha(root / name) == expected for name, expected in registration["source_files"].items())
        result["optimization_source_unchanged_after"] = all(
            sha(root / name) == expected for name, expected in result.get("optimization_source_hashes", {}).items())
        if args.profile:
            stats = pstats.Stats(profiler)
            result["profile"] = [{"file": k[0], "line": k[1], "function": k[2],
                                  "primitive_calls": v[0], "calls": v[1], "self_seconds": v[2],
                                  "cumulative_seconds": v[3]}
                                 for k, v in sorted(stats.stats.items(), key=lambda kv: -kv[1][3])[:80]]
        event("finished", status=result["status"])
        events.close()
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "elapsed", "calls", "inclusive_seconds")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--geometry-prefix", type=int, default=0)
    options = parser.parse_args()
    if not 1 <= options.seconds <= 1800:
        parser.error("diagnostic deadline must be 1..1800 seconds")
    if options.geometry_prefix < 0:
        parser.error("geometry prefix must be non-negative")
    run(options)
