#!/usr/bin/env python3
"""Finish one supervisor-admitted local study after its development process.

This is a single execution chain, not a recurring scheduler. It never chooses
new candidates, retries failed fits, changes a budget, or uploads any data.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhis_fairbias.benchmark.experiment_worker import file_sha, write_json


def runtime_signature(executable, environment):
    names = ["numpy", "pandas", "scipy", "scikit-learn", "fairlearn", "aif360", "torch", "joblib",
             "cloudpickle", "oxonfair", "fairret", "rtdl-num-embeddings", "tensorflow-macos", "tensorflow-model-remediation"]
    code = "\n".join(["import importlib.metadata as m,json,sys", "packages={}",
        "for n in json.loads(sys.argv[1]):", " try: packages[n]=m.version(n)",
        " except m.PackageNotFoundError: packages[n]=None",
        "print(json.dumps({'python':sys.version,'packages':packages},sort_keys=True))"])
    result = subprocess.run([str(executable), "-B", "-c", code, json.dumps(names)], env=environment,
                            cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def verify_admission(path, run):
    record = json.loads(path.read_text())
    if record.get("status") != "SUPERVISOR_ACCEPTED_FOR_COMPLETION":
        raise ValueError("Missing supervisor completion admission")
    if record.get("registration_sha256") != file_sha(run / "registration.json"):
        raise ValueError("Completion admission covers a different registration")
    for name, expected in record["files"].items():
        if file_sha(ROOT / name) != expected:
            raise ValueError("Admitted implementation or evidence changed: " + name)
    return record


def complete(run, admission, development_pid):
    run, admission = run.resolve(), admission.resolve()
    verify_admission(admission, run)
    journal = run / "completion_events.jsonl"
    if journal.exists():
        raise FileExistsError("Completion attempt already exists; inspect it before a new attempt")

    def event(phase, status, **details):
        value = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "phase": phase, "status": status, **details}
        with journal.open("a") as handle:
            handle.write(json.dumps(value, allow_nan=False) + "\n")
        print(json.dumps(value), flush=True)

    environment = dict(os.environ)
    environment.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
    environment["PYTHONHASHSEED"] = "0"
    environment["TF_CPP_MIN_LOG_LEVEL"] = "2"
    environment["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"),
        str(ROOT / "artifacts/nhis/benchmark_dependencies_20260916/site-packages"),
        str(ROOT / ".venv311/lib/python3.11/site-packages")])
    main_python = ROOT / ".venv311/bin/python"
    tf_python = ROOT / "artifacts/nhis/benchmark_dependencies_20260916/frappe_env/.venv/bin/python"

    def execute(phase, command):
        approved = verify_admission(admission, run)
        for executable, signature in approved.get("runtime_signatures", {}).items():
            if runtime_signature(executable, environment) != signature:
                raise ValueError("Admitted dependency versions changed: " + executable)
        event(phase, "RUNNING")
        log_path = run / ("completion_"+phase+".log")
        with log_path.open("x") as handle:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, env=environment,
                                    stdout=handle, stderr=subprocess.STDOUT)
        event(phase, "COMPLETE" if result.returncode == 0 else "FAILED",
              returncode=result.returncode, log_sha256=file_sha(log_path))
        if result.returncode:
            raise RuntimeError("Completion phase failed: " + phase)

    try:
        event("development", "WAITING", process_id=development_pid)
        seen, counts = set(), Counter()
        registration = json.loads((run / "registration.json").read_text())
        expected = sum(len(c.get("seeds", [])) for c in registration["candidates"] if c["status"] == "REGISTERED")
        while development_pid:
            receipts = set((run / "jobs").glob("*/receipt.json"))
            for receipt in receipts - seen:
                result = json.loads(receipt.with_name("result.json").read_text())
                counts[result["status"]] += 1
            if receipts != seen:
                seen = receipts
                status = {"phase": "F_C_S_DEVELOPMENT", "completed_seed_jobs": len(seen), "registered_seed_jobs": expected,
                          "statuses": dict(counts), "new_2024_model_evaluation_started": False,
                          "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
                temporary = run / "live_status.json.tmp"
                temporary.write_text(json.dumps(status, indent=2))
                temporary.replace(run / "live_status.json")
                event("development", "PROGRESS", completed=len(seen), total=expected, counts=dict(counts))
            observed = subprocess.run(["ps", "-o", "args=", "-p", str(development_pid)], capture_output=True, text=True)
            if observed.returncode != 0 or not observed.stdout.strip():
                break
            if "run_nhis_benchmark.py develop" not in observed.stdout:
                raise ValueError("Development process identity changed")
            time.sleep(30)
        # This checks every registered seed and every command/artifact receipt.
        # An interrupted development process cannot silently authorize T.
        execute("selection", [main_python, "-B", ROOT / "scripts/run_nhis_benchmark.py", "select", "--run", run])
        study_freeze = run / "complete_study_freeze.json"
        execute("freeze", [main_python, "-B", "-m", "nhis_fairbias.benchmark.experiment_evaluation",
            "freeze", "--run", run, "--output", study_freeze])
        output = run / "evaluation_T"
        (run / "live_status.json").write_text(json.dumps({"phase": "FROZEN_T_EVALUATION", "new_2024_model_evaluation_started": True,
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2))
        execute("evaluation", [tf_python, "-B", "-m", "nhis_fairbias.benchmark.experiment_evaluation",
            "evaluate", "--run", run, "--output", output, "--study-freeze", study_freeze])
        verify_admission(admission, run)
        from nhis_fairbias.benchmark.paper_reporting import write_paper_reporting
        reports = write_paper_reporting(run / "registration.json", run / "selection_freeze.json", output / "summary_T.json",
            sorted(output.glob("*_individual_model_metrics.json")), run / "paper_results",
            job_result_paths=sorted((run / "jobs").glob("*/result.json")))
        event("tables", "COMPLETE", report=str(reports["markdown"]))
        execute("plots", [main_python, "-B", ROOT / "scripts/plot_nhis_study.py", "--registration", run / "registration.json",
            "--selection-freeze", run / "selection_freeze.json", "--summary-t", output / "summary_T.json",
            "--output-dir", run / "paper_figures"])
        write_json(run / "completion_result.json", {"status": "COMPLETED_FIXED_STUDY",
            "admission_sha256": file_sha(admission), "study_freeze_sha256": file_sha(study_freeze),
            "summary_T_sha256": file_sha(output / "summary_T.json"), "paper_report": str(reports["markdown"]),
            "claim": "Computed registered fixed-study results and drafts; no presumption of superiority or journal acceptance."})
        event("study", "COMPLETE")
        (run / "live_status.json").write_text(json.dumps({"phase": "COMPLETED_FIXED_STUDY", "paper_report": str(reports["markdown"]),
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2))
    except Exception as exc:
        event("study", "STOPPED_ON_ERROR", error_type=type(exc).__name__, error=str(exc))
        (run / "live_status.json").write_text(json.dumps({"phase": "STOPPED_ON_ERROR", "error": str(exc),
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--development-pid", type=int, default=0)
    options = parser.parse_args()
    complete(options.run, options.admission, options.development_pid)
