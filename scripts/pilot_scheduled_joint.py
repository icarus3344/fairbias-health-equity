"""Independent aggregate-only F/C pilot of exact MDS and scheduled Joint.

This script never imports the formal worker or evaluates S/T. Run one fit per
isolated process with single-thread numerical libraries. Historical inputs are
read-only; output must be a fresh sibling directory outside the registered run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import functools
import hashlib
import json
from pathlib import Path
import pickle
import resource
import shutil
import signal
import sys
import time
from unittest.mock import patch


class PilotHardStop(BaseException):
    """Hard resource interruption; never a completed fit."""


class PilotPrefixStop(BaseException):
    """Predeclared equivalent-work timing endpoint."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_modules(root, expected):
    """Bind loaded project module origins to the source snapshot before fit."""
    observed = {}
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] not in {"fairbias", "nhis_fairbias"}:
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            raise ValueError("Project module has no verifiable source: " + name)
        path = Path(origin).resolve()
        if not path.is_relative_to(root / "src"):
            raise ValueError("Project module loaded outside source tree: " + name)
        rel = str(path.relative_to(root))
        if rel not in expected or sha(path) != expected[rel]:
            raise ValueError("Loaded project module source mismatch: " + name)
        observed[name] = {"source": rel, "sha256": expected[rel]}
    return observed


def run(args):
    root = Path(__file__).resolve().parents[1]
    # Snapshot before any project imports. This also covers auxiliary package
    # files outside the old registration; they cannot be silently shadowed.
    project_hashes = {str(p.relative_to(root)): sha(p)
                      for package in ("fairbias", "nhis_fairbias")
                      for p in (root / "src" / package).rglob("*.py")}
    import joblib
    from fairbias.evaluator import FairEvaluator
    from fairbias.enhancement import FairAccuracyEnhancement
    from fairbias.mitigation import FairBiasMitigation
    from nhis_fairbias.benchmark.adapters.adapter_fairbias_ae import FairBiasAEAdapter
    from nhis_fairbias.benchmark.joint_mds_numpy import numpy_mds_acceleration

    registered_run = args.run.resolve()
    registration_path = registered_run / "registration.json"
    registration = json.loads(registration_path.read_text())
    for name, digest in registration["source_files"].items():
        if sha(root / name) != digest:
            raise ValueError("Registered source mismatch: " + name)
    config = next(c for c in registration["candidates"] if c["candidate_id"] == args.candidate)
    if config["method"] != "FAIRBIAS_JOINT" or args.seed not in config["seeds"]:
        raise ValueError("Pilot requires a registered Joint candidate and seed")
    output = args.output.resolve()
    if output.exists() or output == registered_run or registered_run in output.parents:
        raise ValueError("Output must be fresh and outside the registered run")
    output.mkdir(parents=True)
    sources = [
        "scripts/pilot_scheduled_joint.py",
        "src/nhis_fairbias/benchmark/joint_budget_optimization.py",
        "src/nhis_fairbias/benchmark/joint_mds_numpy.py",
    ]
    if args.variant == "scheduled_joint_v1":
        sources.append("src/nhis_fairbias/benchmark/adapters/adapter_fairbias_scheduled.py")
        from nhis_fairbias.benchmark.adapters.adapter_fairbias_scheduled import ScheduledJointAdapter
    hashes = {name: sha(root / name) for name in sources}
    for name in sources:
        shutil.copyfile(root / name, output / Path(name).name)
    modules_before_load = runtime_modules(root, project_hashes)
    prepared = registered_run / "prepared" / (config["arm_id"] + ".joblib")
    expected = registration["prepared"][config["arm_id"]]
    if sha(prepared) != expected["sha256"]:
        raise ValueError("Prepared file hash mismatch")
    # The payload contains F/C/S. Do not inspect or retain S.
    data = joblib.load(prepared)
    if data["data_identity"] != expected["data_identity"]:
        raise ValueError("Prepared identity mismatch")
    F, C = data["partitions"]["fitting_F"], data["partitions"]["calibration_C"]
    del data
    modules_before_fit = runtime_modules(root, project_hashes)
    params = dict(config["params"])
    if args.variant == "scheduled_joint_v1":
        adapter = ScheduledJointAdapter(
            backbone=config["backbone"], random_state=args.seed,
            ae_every_bm_commits=args.ae_interval, search_seconds=args.search_seconds, **params,
        )
    else:
        adapter = FairBiasAEAdapter(backbone=config["backbone"], random_state=args.seed, **params)
    started = time.monotonic()
    counters, durations, engines, commits = {}, {}, {}, {"BM": 0, "AE": 0}
    events = (output / "events.jsonl").open("x", buffering=1)

    def event(kind, **fields):
        events.write(json.dumps({"elapsed": time.monotonic() - started, "event": kind,
                                 **fields}, allow_nan=False) + "\n")

    def timed(name, fn):
        @functools.wraps(fn)
        def wrapped(*a, **kw):
            counters[name] = counters.get(name, 0) + 1
            tick = time.monotonic()
            if name in {"BM", "AE"}:
                engines[name] = a[0]
                event("phase_start", engine=name, call=counters[name])
            try:
                value = fn(*a, **kw)
                if name == "geometry":
                    event("geometry_value", call=counters[name],
                          digest=hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest())
                    if args.geometry_prefix and counters[name] >= args.geometry_prefix:
                        raise PilotPrefixStop()
                if name in {"BM", "AE"}:
                    # Engine proposals are recorded separately from adapter commits:
                    # a later full verification or budget check can reject a proposal.
                    commits[name] += int(value[-1] is not None)
                    event("engine_return", engine=name, call=counters[name],
                          proposed_change=value[-1] is not None,
                          state_digest=hashlib.sha256(pickle.dumps(value[1], protocol=5)).hexdigest())
                return value
            finally:
                durations[name] = durations.get(name, 0.0) + time.monotonic() - tick
                if name in {"geometry", "utility"}:
                    event(name, call=counters[name], seconds=time.monotonic() - tick)
        return wrapped

    def deadline(signum, frame):
        raise PilotHardStop("HARD_WALL_LIMIT")

    def memory_check(signum, frame):
        # Linux ru_maxrss is KiB. The launcher runs this only on Linux.
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 4 * 1024 * 1024:
            raise PilotHardStop("RSS_LIMIT")

    old_alarm = signal.signal(signal.SIGALRM, deadline)
    old_timer = signal.signal(signal.SIGPROF, memory_check)
    result = {
        "diagnostic_only": True, "variant": args.variant, "candidate_id": args.candidate,
        "seed": args.seed, "config": config, "ae_interval": args.ae_interval,
        "search_seconds": args.search_seconds, "hard_seconds": args.hard_seconds,
        "geometry_prefix": args.geometry_prefix, "S_T_evaluated": False,
        "registration_sha256": sha(registration_path), "source_identity": registration["source_identity"],
        "prepared_sha256": sha(prepared), "pilot_source_hashes": hashes,
        "loaded_project_modules_before_load": modules_before_load,
        "loaded_project_modules_before_fit": modules_before_fit,
        "sample_sizes": {"F": len(F), "C": len(C)}, "status": "STARTED",
    }
    (output / "pilot_inputs.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    optimization = None
    try:
        with ExitStack() as stack:
            for obj, attr, label in (
                (FairEvaluator, "calculate_epsilon", "geometry"),
                (FairBiasAEAdapter, "_utility", "utility"),
                (FairBiasMitigation, "mitigate_step", "BM"),
                (FairAccuracyEnhancement, "enhance_step", "AE"),
            ):
                stack.enter_context(patch.object(obj, attr, timed(label, getattr(obj, attr))))
            optimization = stack.enter_context(numpy_mds_acceleration())
            signal.setitimer(signal.ITIMER_REAL, args.hard_seconds)
            signal.setitimer(signal.ITIMER_PROF, 2, 2)
            adapter.fit_development(F.X_semantic, F.y, F.A, C.X_semantic, C.y, C.A,
                                    metadata={"F_ids": F.record_keys, "C_ids": C.record_keys})
            if not adapter.is_fitted_:
                raise RuntimeError("Fit returned without fitted model")
            result["status"] = "FIT_COMPLETED_DIAGNOSTIC_ONLY"
    except PilotHardStop as exc:
        result.update(status="PILOT_HARD_LIMIT", hard_limit_reason=str(exc))
    except PilotPrefixStop:
        result["status"] = "PILOT_PREFIX_COMPLETED"
    except Exception as exc:
        # Error messages from arbitrary models can include data. Keep only type.
        result.update(status="FIT_EXCEPTION", error_type=type(exc).__name__)
    except BaseException as exc:
        result.update(status="INTERRUPTED", error_type=type(exc).__name__)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.setitimer(signal.ITIMER_PROF, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        signal.signal(signal.SIGPROF, old_timer)
        provenance = getattr(adapter, "provenance_", {})
        result.update(
            elapsed=time.monotonic() - started, calls=counters, inclusive_seconds=durations,
            termination_reason=getattr(adapter, "termination_reason_", None),
            convergence_verified=getattr(adapter, "convergence_verified_", None),
            incumbent_available=getattr(adapter, "incumbent_available_", None),
            fitted=bool(getattr(adapter, "is_fitted_", False)),
            utility_evaluations=getattr(adapter, "_utility_evaluations_", 0),
            geometry_evaluations=getattr(adapter, "_geometry_evaluations_", 0),
            engine_proposal_counts=commits, optimization=optimization,
            changed_state_digest=hashlib.sha256(pickle.dumps(getattr(adapter, "changed_dict_", {}), protocol=5)).hexdigest(),
            controller_summary={k: provenance[k] for k in (
                "bm_commits", "ae_commits", "final_max_dphi", "epsilon_threshold",
                "termination_reason", "budget_reason", "convergence_verified",
                "schedule", "ae_deferred", "search_seconds", "incumbent_available",
            ) if k in provenance},
            controller_traces=getattr(adapter, "candidate_traces_", []),
        )
        mds = getattr(adapter, "mds_diagnostics_", [])
        result["mds"] = {"fits": len(mds),
            "status_counts": dict(Counter(r["status"] for r in mds)),
            "recorded_best_iterations_sum": sum(r["iterations"] for r in mds)}
        if args.variant == "strict_numpy" and result["fitted"]:
            complete = all(r["status"] == "CONVERGED_BEFORE_CAP" for r in mds)
            result["convergence_verified"] = bool(complete and mds and
                result["termination_reason"] == "STRICT_FEASIBLE_SEARCH_EXHAUSTED")
            if not result["convergence_verified"]:
                result["status"] = "FIT_WITH_UNRESOLVED_GEOMETRY"
        audit = getattr(engines.get("AE"), "audit_trail", [])
        result["ae_audit_summary"] = {"events": len(audit),
            "selected_by_engine": sum(int(e.accepted) for e in audit),
            "reasons": dict(Counter((e.rejection_reason or "SELECTED_BY_ENGINE").split(":", 1)[0] for e in audit))}
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result["resource"] = {"user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime,
                              "maxrss_kib": usage.ru_maxrss}
        result["registered_source_unchanged_after"] = all(
            sha(root / name) == digest for name, digest in registration["source_files"].items())
        result["pilot_source_unchanged_after"] = all(sha(root / name) == digest for name, digest in hashes.items())
        try:
            result["loaded_project_modules_after_fit"] = runtime_modules(root, project_hashes)
            result["runtime_module_origins_verified"] = True
        except ValueError:
            result["runtime_module_origins_verified"] = False
        if not result["registered_source_unchanged_after"] or not result["pilot_source_unchanged_after"]:
            result["status"] = "INTEGRITY_FAILURE"
        if not result["runtime_module_origins_verified"]:
            result["status"] = "INTEGRITY_FAILURE"
        event("finished", status=result["status"])
        events.close()
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "elapsed", "termination_reason", "calls")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["strict_numpy", "scheduled_joint_v1"], required=True)
    parser.add_argument("--ae-interval", type=int, default=3)
    parser.add_argument("--search-seconds", type=float, default=900)
    parser.add_argument("--hard-seconds", type=int, default=960)
    parser.add_argument("--geometry-prefix", type=int, default=0)
    options = parser.parse_args()
    if not 1 <= options.hard_seconds <= 1800:
        parser.error("hard deadline must be 1..1800 seconds")
    if not 0 < options.search_seconds <= options.hard_seconds - 30:
        parser.error("search deadline must reserve at least 30 seconds for final fitting")
    if options.ae_interval < 1 or options.geometry_prefix < 0:
        parser.error("interval must be positive and geometry prefix nonnegative")
    run(options)
