#!/usr/bin/env python3
"""Auditable staged benchmark. Development never loads the 2024 source file."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_manifest():
    paths = sorted(set((ROOT / "src/fairbias").glob("*.py")) | set((ROOT / "src/nhis_fairbias").rglob("*.py")))
    # Evaluation-only analysis receives a separate hash at complete study
    # freeze. Every training adapter is bound to this registration.
    pending = {"experiment_evaluation.py", "survey_batch.py", "survey_linearization.py",
               "cohort_reporting.py", "paper_reporting.py"}
    paths = [p for p in paths if p.name not in pending]
    paths += [Path(__file__).resolve()] + sorted((ROOT / "configs/nhis").glob("*.json"))
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def prepare(run):
    import joblib
    from nhis_fairbias.benchmark.data_contracts import ARM_SPECS, load_arm_partitions, load_local_nhis_cohort
    from nhis_fairbias.benchmark.preprocessing import BenchmarkPreprocessor
    from nhis_fairbias.benchmark.experiment_registry import enumerate_candidates, identity
    from nhis_fairbias.benchmark.experiment_worker import write_json
    run.mkdir(parents=True, exist_ok=False)
    (run / "prepared").mkdir()
    (run / "jobs").mkdir()
    (run / "representation_cache").mkdir()
    sources = source_manifest()
    cohort = load_local_nhis_cohort(ROOT, years=(2022, 2023))
    prepared = {}
    for arm in ARM_SPECS:
        partitions = load_arm_partitions(cohort, arm)
        partitions.pop("evaluation_T")
        if any(len(p) == 0 for p in partitions.values()):
            raise ValueError("F/C/S partition empty under registered split")
        F, C, S = [partitions[k] for k in ("fitting_F", "calibration_C", "selection_S")]
        preprocessor = BenchmarkPreprocessor(ARM_SPECS[arm]["features"]).fit(F.X_semantic)
        data_identity = identity({"sources": cohort.attrs["source_provenance"], "arm": arm,
                                  "split": F.metadata["psu_assignment_sha256"], "code": sources})
        payload = {"partitions": partitions, "preprocessor": preprocessor,
                   "X_F": preprocessor.transform(F.X_semantic), "X_C": preprocessor.transform(C.X_semantic),
                   "X_S": preprocessor.transform(S.X_semantic), "data_identity": data_identity}
        path = run / "prepared" / (arm + ".joblib")
        joblib.dump(payload, path, compress=3)
        prepared[arm] = {"path": str(path), "sha256": sha(path), "data_identity": data_identity,
                         "sizes": {k: len(p) for k, p in partitions.items()}}
    configs = enumerate_candidates()
    from nhis_fairbias.benchmark.experiment_extensions import enumerate_extensions
    configs += enumerate_extensions()
    write_json(run / "registration.json", {"version": "codex_application_v1_20260916",
        "source_files": sources, "source_identity": identity(sources), "prepared": prepared,
        "raw_sources": cohort.attrs["source_provenance"], "candidates": configs,
        "resources": {"fit_seconds": 1800, "worker_rss_bytes": 4 * 1024**3, "concurrent_fits": 1, "threads": 1},
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata.get("Name")},
        "claim": "Retrospective cross-year evaluation. F/C/S development; no new 2024 model evaluation before complete selection freeze.",
        "pending_predeclared_extensions": [],
        "registered_extensions": ["FRAPPE", "FAIRGBM", "TABM_COMPACT", "BM_AE_FIXED", "JOINT_FIXED", "ARM004_GEOMETRY_FIXED"],
        "optional_not_in_candidate_family": {"TABICL_V1": "Deferred: CPU context/official checkpoint admission not completed; not an evaluated comparator.",
            "TABM_PAPER_CAPACITY": "Compact CPU variant registered explicitly; paper-default-capacity reproduction not claimed."}})
    print(json.dumps({"prepared_years": [2022, 2023], "configurations": len(configs),
                      "fit_jobs_before_representation_reuse": sum(len(c["seeds"]) for c in configs)}), flush=True)


def validate_source(registration):
    for rel, expected in registration["source_files"].items():
        if sha(ROOT / rel) != expected:
            raise ValueError("Registered source changed: " + rel + "; create a new registered run")


def develop(run, methods=None, backbones=None, max_jobs=None):
    from nhis_fairbias.benchmark.experiment_worker import write_json, representation_key
    registration = json.loads((run / "registration.json").read_text())
    validate_source(registration)
    limits = registration["resources"]
    done = 0
    for config in registration["candidates"]:
        if config["status"] != "REGISTERED" or (methods and config["method"] not in methods) or (backbones and config["backbone"] not in backbones):
            continue
        for seed in config["seeds"]:
            path = run / "jobs" / f'{config["candidate_id"]}_s{seed}'
            if (path / "result.json").exists():
                continue
            if path.exists():
                raise RuntimeError("Incomplete prior attempt requires explicit recovery: " + str(path))
            validate_source(registration)
            path.mkdir()
            data = registration["prepared"][config["arm_id"]]
            job = {"config": config, "seed": seed, "data_path": data["path"], "data_identity": data["data_identity"],
                   "data_sha256": data["sha256"], "cache_path": str(run / "representation_cache"),
                   "source_identity": registration["source_identity"]}
            write_json(path / "job.json", job)
            env = dict(os.environ)
            env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
            env["PYTHONHASHSEED"] = "0"
            executable = sys.executable
            if config["method"] == "FRAPPE_EO":
                executable = str(ROOT / "artifacts/nhis/benchmark_dependencies_20260916/frappe_env/.venv/bin/python")
                env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / ".venv311/lib/python3.11/site-packages")])
                env["TF_CPP_MIN_LOG_LEVEL"] = "2"
            start = time.monotonic()
            termination, peak_rss = None, 0
            with (path / "worker.log").open("x") as log:
                proc = subprocess.Popen([executable, "-B", str(Path(__file__).resolve()), "worker", "--job", str(path / "job.json")],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while proc.poll() is None:
                    usage = subprocess.run(["ps", "-o", "rss=", "-p", str(proc.pid)], capture_output=True, text=True)
                    if usage.returncode == 0 and usage.stdout.strip():
                        peak_rss = max(peak_rss, int(usage.stdout.strip()) * 1024)
                    if peak_rss > limits["worker_rss_bytes"] or time.monotonic() - start > limits["fit_seconds"]:
                        termination = "MEMORY_LIMIT" if peak_rss > limits["worker_rss_bytes"] else "TIME_LIMIT"
                        import signal
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                        break
                    time.sleep(1)
            if not (path / "result.json").exists():
                write_json(path / "result.json", {"candidate_id": config["candidate_id"], "seed": seed,
                    "status": termination or "WORKER_FAILED", "returncode": proc.returncode,
                    "elapsed_seconds": time.monotonic() - start, "observed_peak_rss_bytes": peak_rss})
            result = json.loads((path / "result.json").read_text())
            if termination is not None:
                key = representation_key(config, seed, data["data_identity"])
                if key is not None:
                    cache_status = run / "representation_cache" / (key + ".json")
                    if not cache_status.exists():
                        write_json(cache_status, {"status": termination, "key": key,
                            "source_job": path.name, "termination": "supervised process resource stop"})
            write_json(path / "receipt.json", {"returncode": proc.returncode, "termination": termination,
                "observed_peak_rss_bytes": peak_rss, "files": {p.name: sha(p) for p in path.iterdir() if p.is_file()}})
            print(json.dumps({"arm": config["arm_id"], "method": config["method"], "backbone": config["backbone"],
                              "seed": seed, "status": result["status"], "seconds": round(time.monotonic()-start, 2)}), flush=True)
            done += 1
            if max_jobs is not None and done >= max_jobs:
                return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "develop", "worker", "select"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--backbones", nargs="+")
    parser.add_argument("--max-jobs", type=int)
    args = parser.parse_args()
    if args.phase == "worker":
        from nhis_fairbias.benchmark.experiment_worker import execute_job
        execute_job(args.job)
    elif args.phase == "prepare":
        prepare(args.run.resolve())
    elif args.phase == "select":
        from nhis_fairbias.benchmark.experiment_selection import freeze_selection
        registration = json.loads((args.run / "registration.json").read_text())
        validate_source(registration)
        freeze_selection(args.run.resolve())
    else:
        develop(args.run.resolve(), args.methods, args.backbones, args.max_jobs)


if __name__ == "__main__":
    main()
