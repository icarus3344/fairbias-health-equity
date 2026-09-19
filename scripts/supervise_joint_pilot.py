"""Linux external supervisor for one independent Joint diagnostic child.

The RSS threshold is sampled every 0.5 seconds; it is not a kernel-enforced
memory cap. A completed supervisor receipt never establishes fit success.
Importing this module does not launch a process or inspect data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time


THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE", "MKL_DYNAMIC": "FALSE", "PYTHONDONTWRITEBYTECODE": "1",
}


def read_proc_rss_bytes(pid):
    """Read the main child's resident bytes; an exited process returns None."""
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as source:
            for line in source:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except FileNotFoundError:
        return None
    return None


def _nice_child():
    os.nice(19)


def _signal_group(pid, signum):
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        return False
    return True


def _stop_group(process, grace_seconds):
    term_sent = _signal_group(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        process.poll()  # Reap the leader, while still watching its process group.
        if not _signal_group(process.pid, 0):
            return term_sent, False
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    kill_sent = _signal_group(process.pid, signal.SIGKILL)
    process.wait()
    return term_sent, kill_sent


def supervise_command(command, *, receipt_path, stdout_path, elapsed_limit_seconds,
                      memory_limit_bytes=4 * 1024 ** 3, sample_seconds=0.5,
                      grace_seconds=5.0, environment=None, cwd=None,
                      rss_reader=read_proc_rss_bytes, receipt_metadata=None):
    """Supervise an isolated process group; command/RSS injection aids testing.

    The production CLI fixes the limits, environment and command. This helper
    can run short synthetic subprocesses without loading benchmark data.
    Receipt/log paths must both be absent. Only files created by this invocation
    are updated, and the child's output directory is never created here.
    """
    if elapsed_limit_seconds <= 0 or sample_seconds <= 0 or grace_seconds < 0:
        raise ValueError("Invalid supervision timing")
    if memory_limit_bytes <= 0:
        raise ValueError("Memory threshold must be positive")
    receipt_path, stdout_path = Path(receipt_path), Path(stdout_path)
    if receipt_path == stdout_path or receipt_path.exists() or stdout_path.exists():
        raise FileExistsError("Supervisor receipt and log must be distinct fresh files")
    env = dict(os.environ if environment is None else environment)
    env.update(THREAD_ENVIRONMENT)
    record = {
        "supervisor_only": True, "fit_success_established": False,
        "status": "STARTING", "command": list(command), "cwd": str(cwd) if cwd else None,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "thread_environment": dict(THREAD_ENVIRONMENT), "requested_nice": 19,
        "elapsed_limit_seconds": elapsed_limit_seconds,
        "memory_limit_bytes": memory_limit_bytes, "sample_seconds": sample_seconds,
        "termination_grace_seconds": grace_seconds,
        "rss_scope": "main_child_only", "rss_limit_kind": "sampled_threshold_not_kernel_cap",
        "rss_caveat": "Allocations between samples can exceed the threshold; descendants are not summed.",
        "peak_observed_rss_bytes": 0, "rss_samples": 0, "rss_unavailable_samples": 0,
        "termination_reason": None, "exit_code": None,
        "sigterm_sent": False, "sigkill_sent": False,
        "stdout_path": str(stdout_path.resolve()),
        "metadata": dict(receipt_metadata or {}),
    }
    process = None
    start = time.monotonic()
    with stdout_path.open("x", encoding="utf-8") as child_log:
        with receipt_path.open("x", encoding="utf-8") as receipt:
            def save():
                receipt.seek(0)
                json.dump(record, receipt, indent=2, allow_nan=False)
                receipt.write("\n")
                receipt.truncate()
                receipt.flush()
                os.fsync(receipt.fileno())

            save()
            try:
                start = time.monotonic()
                process = subprocess.Popen(
                    list(command), stdin=subprocess.DEVNULL, stdout=child_log,
                    stderr=subprocess.STDOUT, env=env, cwd=cwd,
                    start_new_session=True, preexec_fn=_nice_child,
                )
                record.update(status="RUNNING", child_pid=process.pid, process_group_id=process.pid)
                record["observed_nice"] = os.getpriority(os.PRIO_PROCESS, process.pid)
                save()
                while process.poll() is None:
                    elapsed = time.monotonic() - start
                    if elapsed >= elapsed_limit_seconds:
                        record["termination_reason"] = "EXTERNAL_WALL_LIMIT"
                        break
                    try:
                        rss = rss_reader(process.pid)
                    except Exception as exc:
                        record.update(termination_reason="RSS_MONITOR_ERROR", monitor_error_type=type(exc).__name__)
                        break
                    if rss is None:
                        record["rss_unavailable_samples"] += 1
                    else:
                        record["rss_samples"] += 1
                        record["peak_observed_rss_bytes"] = max(record["peak_observed_rss_bytes"], int(rss))
                        if rss > memory_limit_bytes:
                            record["termination_reason"] = "EXTERNAL_RSS_LIMIT"
                            break
                    time.sleep(min(sample_seconds, max(0.0, elapsed_limit_seconds - elapsed)))
                if record["termination_reason"] is not None:
                    record["status"] = "CHILD_TERMINATED_BY_SUPERVISOR"
                    record["sigterm_sent"], record["sigkill_sent"] = _stop_group(process, grace_seconds)
                else:
                    record["status"] = "CHILD_PROCESS_EXITED"
                    record["termination_reason"] = "PROCESS_EXIT"
                record["exit_code"] = process.wait()
            except BaseException as exc:
                record.update(status="SUPERVISOR_EXCEPTION", termination_reason="SUPERVISOR_EXCEPTION",
                              supervisor_error_type=type(exc).__name__)
                if process is not None:
                    record["sigterm_sent"], record["sigkill_sent"] = _stop_group(process, grace_seconds)
                    record["exit_code"] = process.wait()
                raise
            finally:
                record["elapsed_seconds"] = time.monotonic() - start
                record["finished_utc"] = datetime.now(timezone.utc).isoformat()
                save()
    return record


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_pilot(args):
    if sys.platform != "linux":
        raise RuntimeError("The Joint pilot launcher is Linux-only")
    root = Path(__file__).resolve().parents[1]
    registered = args.run.resolve()
    output = args.output.resolve()
    if output.exists() or output == registered or registered in output.parents:
        raise ValueError("Child output must be fresh and outside the registered run")
    receipt = output.with_name(output.name + ".supervisor.json")
    log = output.with_name(output.name + ".stdout.log")
    if receipt.exists() or log.exists():
        raise FileExistsError("Supervisor artifacts already exist")
    sources = [
        Path(__file__).resolve(), root / "scripts/pilot_scheduled_joint.py",
        root / "src/nhis_fairbias/benchmark/joint_budget_optimization.py",
        root / "src/nhis_fairbias/benchmark/joint_mds_numpy.py",
    ]
    if args.variant == "scheduled_joint_v1":
        sources.append(root / "src/nhis_fairbias/benchmark/adapters/adapter_fairbias_scheduled.py")
    metadata = {
        "runtime_source_hashes": {str(path.relative_to(root)): _sha(path) for path in sources},
        "python_executable": str(Path(sys.executable).resolve()),
        "python_executable_sha256": _sha(Path(sys.executable).resolve()),
        "python_version": sys.version, "platform": platform.platform(),
        "child_output": str(output), "registered_run": str(registered),
        "whole_child_process_supervised": True,
        "child_fit_alarm_seconds": args.hard_seconds,
    }
    command = [sys.executable, "-B", str(root / "scripts/pilot_scheduled_joint.py")]
    for flag, value in (
        ("--run", registered), ("--candidate", args.candidate), ("--variant", args.variant),
        ("--seed", args.seed), ("--output", output), ("--hard-seconds", args.hard_seconds),
        ("--search-seconds", args.search_seconds), ("--ae-interval", args.ae_interval),
        ("--geometry-prefix", args.geometry_prefix),
    ):
        command.extend([flag, str(value)])
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    output.parent.mkdir(parents=True, exist_ok=True)
    return supervise_command(
        command, receipt_path=receipt, stdout_path=log,
        elapsed_limit_seconds=args.hard_seconds + 15, memory_limit_bytes=4 * 1024 ** 3,
        sample_seconds=0.5, grace_seconds=5.0, environment=environment,
        cwd=root, receipt_metadata=metadata,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--variant", choices=["strict_numpy", "scheduled_joint_v1"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hard-seconds", type=int, default=960)
    parser.add_argument("--search-seconds", type=float, default=900)
    parser.add_argument("--ae-interval", type=int, default=3)
    parser.add_argument("--geometry-prefix", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.hard_seconds <= 1800 or not 0 < args.search_seconds <= args.hard_seconds - 30:
        parser.error("hard deadline must be 1..1800 seconds and reserve 30 seconds after search")
    if args.ae_interval < 1 or args.geometry_prefix < 0:
        parser.error("interval must be positive and geometry prefix nonnegative")
    result = run_pilot(args)
    print(json.dumps({key: result[key] for key in ("status", "termination_reason", "exit_code", "elapsed_seconds")}))
    return 0 if result["status"] == "CHILD_PROCESS_EXITED" and result["exit_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
