"""Start the admitted CPU queue after the bounded real-data pilots finish.

This controller performs no selection or 2024 evaluation. It preserves the
registered method budgets and leaves LFR/FRAPPE for their separate admissions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from nhis_fairbias.benchmark.parallel_execution import file_sha, verify_completed_job
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    registration = json.loads((run / "registration.json").read_text())
    # Block behind the pilot's exclusive lock without stopping its workers.
    with (run / "parallel_scheduler.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        counts = Counter()
        for config in registration["candidates"]:
            for seed in config["seeds"]:
                path = run / "jobs" / f'{config["candidate_id"]}_s{seed}'
                result = verify_completed_job(path, config, seed, registration)
                if result:
                    if result["status"] != "VALID":
                        raise RuntimeError(f"Pilot admission failed: {path.name}: {result['status']}")
                    counts[config["method"]] += 1
        if counts != {"UNMITIGATED": 24, "FAIRBIAS_BM": 16}:
            raise RuntimeError(f"Pilot receipts are incomplete: {dict(counts)}")
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    methods = sorted({c["method"] for c in registration["candidates"] if
                      c["status"] == "REGISTERED" and c["method"] not in
                      {"LFR_RECONSTRUCTED", "FRAPPE_EO"}})
    command = [sys.executable, "-B", str(ROOT / "scripts/run_nhis_benchmark_parallel.py"),
               "develop", "--run", str(run), "--parallel-policy", str(run / "parallel_policy.json"),
               "--workers", "12", "--max-rss-gib", "64", "--methods", *methods]
    admitted = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "controller_pid": os.getpid(), "controller_sha256": file_sha(Path(__file__)),
                "pilot_valid_counts": dict(counts), "command": command,
                "registration_sha256": file_sha(run / "registration.json"),
                "phase": "F_C_S_DEVELOPMENT_ONLY", "methods": methods,
                "workers": 12, "deferred_methods": ["LFR_RECONSTRUCTED", "FRAPPE_EO"]}
    with (run / "full_cpu_queue_admission.json").open("x") as handle:
        json.dump(admitted, handle, indent=2)
    print(json.dumps(admitted), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    with (run / "full_cpu_queue_exit.json").open("x") as handle:
        json.dump({"returncode": completed.returncode, "development_only": True,
                   "selection_or_test_evaluation_started": False}, handle, indent=2)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
