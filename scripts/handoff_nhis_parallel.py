"""Linux operational drain/resume without changing registered scientific source.

The old dispatcher is stopped at its polling boundary. Its existing workers
finish under resource monitoring. Kernel zombie wait status, rather than a
guessed return code, is retained before receipts are sealed with the original
scheduler. This helper is intentionally outside the scientific source manifest.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nhis_fairbias.benchmark.parallel_execution import (  # noqa: E402
    ExclusiveRunLock, JobSpec, ParallelBenchmarkScheduler, ResourceLimits,
    RunningJob, SchedulerError, _descendant_rss, _effective_cpu_count, file_sha, replace_json,
    verify_completed_job, write_json_once,
)


def proc(pid):
    raw = Path(f"/proc/{pid}/stat").read_text()
    fields = raw.rsplit(")", 1)[1].split()
    return {"pid": pid, "state": fields[0], "ppid": int(fields[1]),
            "start_ticks": int(fields[19]), "wait_status": int(fields[49])}


def children(pid):
    found = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            state = proc(int(p.name))
            if state["ppid"] == pid:
                state["argv"] = [a.decode() for a in (p / "cmdline").read_bytes().split(b"\0") if a]
                found[int(p.name)] = state
        except FileNotFoundError:
            continue
    return found


def same_process(expected):
    current = proc(expected["pid"])
    if current["start_ticks"] != expected["start_ticks"] or current["ppid"] != expected["ppid"]:
        raise SchedulerError("process identity changed during handoff")
    return current


def stop_at_boundary(pid, run, runner):
    """Reject races before modifying any job evidence; resume on failed capture."""
    parent = proc(pid)
    args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    if (str(run).encode() not in args or b"develop" not in args or
            str(runner.parent / "run_nhis_benchmark_parallel.py").encode() not in args or
            Path(f"/proc/{pid}/cwd").resolve() != runner.parent.parent):
        raise SchedulerError("PID is not the requested development dispatcher")
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        same_process(parent)
        before = children(pid)
        if not before or any("--job" not in p["argv"] for p in before.values()):
            time.sleep(.02)
            continue
        if "nanosleep" not in Path(f"/proc/{pid}/wchan").read_text():
            time.sleep(.01)
            continue
        os.kill(pid, signal.SIGSTOP)
        keep_stopped = False
        try:
            for _ in range(100):
                if same_process(parent)["state"] == "T":
                    break
                time.sleep(.001)
            else:
                raise SchedulerError("dispatcher did not stop")
            after = children(pid)
            if set(after) != set(before):
                continue
            paths = set()
            for child, old in before.items():
                same_process(old)
                argv = old["argv"]
                if str(runner) not in argv or "worker" not in argv:
                    raise SchedulerError("unexpected dispatcher child")
                path = Path(argv[argv.index("--job") + 1]).resolve()
                if path.name != "job.json" or path.parent.parent != run / "jobs":
                    raise SchedulerError("worker is outside requested run")
                old["job_path"] = str(path.parent)
                if os.getpgid(child) != child or os.getsid(child) != child:
                    raise SchedulerError("worker is not its own process group and session leader")
                paths.add(path.parent)
            incomplete = {p for p in (run / "jobs").iterdir() if not (p / "receipt.json").exists()}
            if incomplete != paths or list((run / "jobs").glob("*/.*.tmp")):
                continue
            keep_stopped = True
            return parent, before
        finally:
            if not keep_stopped:
                os.kill(pid, signal.SIGCONT)
        time.sleep(.01)
    raise SchedulerError("could not capture a stable polling boundary; old queue remains running")


class ObservedProcess:
    def __init__(self, info):
        self.info, self.pid, self.returncode = info, info["pid"], None

    def poll(self):
        state = same_process(self.info)
        if state["state"] == "Z":
            self.returncode = os.waitstatus_to_exitcode(state["wait_status"])
        return self.returncode

    def wait(self, timeout=None):
        deadline = time.monotonic() + (timeout if timeout is not None else 10)
        while self.poll() is None:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(.05)
        return self.returncode


def drain(old_pid, scheduler, audit):
    run = scheduler.run_path
    scheduler._validate_sources()
    scheduler._validate_parallel_policy()
    scheduler._write_manifest()
    registration = scheduler.registration
    old_policy = run / "parallel_policy.json"
    old_manifest = json.loads((run / "parallel_scheduler_manifest.json").read_text())
    if file_sha(old_policy) != old_manifest["parallel_policy_sha256"]:
        raise SchedulerError("old policy hash differs from the original manifest")
    parent, captured = stop_at_boundary(old_pid, run, scheduler.runner)
    sealed = False
    try:
        write_json_once(audit / "captured.json", {"parent": parent, "workers": captured,
            "helper_sha256": file_sha(Path(__file__)), "registration_sha256": file_sha(scheduler.registration_path),
            "scheduler_manifest": json.loads((run / "parallel_scheduler_manifest.json").read_text()),
            "new_policy": json.loads(scheduler.policy_path.read_text()),
            "new_policy_sha256": file_sha(scheduler.policy_path),
            "old_policy": json.loads(old_policy.read_text()), "old_policy_sha256": file_sha(old_policy),
            "runner_sha256": file_sha(scheduler.runner), "source_script_sha256": file_sha(scheduler.source_script),
            "drain_session_id": scheduler._session_id})
        configs = {c["candidate_id"]: c for c in registration["candidates"]}
        for pid, info in captured.items():
            path = Path(info["job_path"])
            payload = json.loads((path / "job.json").read_text())
            config = configs[payload["config"]["candidate_id"]]
            data = registration["prepared"][config["arm_id"]]
            if (payload["config"] != config or payload["seed"] not in config["seeds"] or
                    payload["data_sha256"] != data["sha256"] or
                    payload["data_identity"] != data["data_identity"] or
                    payload["source_identity"] != registration["source_identity"]):
                raise SchedulerError("captured job differs from registration")
            process_age = float(Path("/proc/uptime").read_text().split()[0]) - info["start_ticks"] / os.sysconf("SC_CLK_TCK")
            # job.json precedes Popen; this conservatively retains the old budget.
            age = max(0., process_age, time.time() - (path / "job.json").stat().st_mtime)
            spec = JobSpec(config, payload["seed"], path, data, payload["source_identity"], payload.get("representation_key"))
            # An independent read handle supplies close() without changing the worker log.
            state = RunningJob(spec, ObservedProcess(info), (path / "worker.log").open("r"), time.monotonic() - age)
            scheduler._running[pid] = state
            scheduler.jobs.append(spec)
        while scheduler._running:
            if same_process(parent)["state"] != "T":
                raise SchedulerError("old dispatcher unexpectedly resumed")
            for pid, state in list(scheduler._running.items()):
                sample = _descendant_rss(pid)
                if sample is None:
                    raise SchedulerError("captured worker disappeared before wait status was read")
                state.last_rss_bytes = max(state.last_rss_bytes, sample)
                if state.process.poll() is None:
                    if state.last_rss_bytes > scheduler.limits.worker_rss_bytes:
                        scheduler._terminate(state, "MEMORY_LIMIT")
                    elif time.monotonic() - state.started > scheduler.limits.fit_seconds:
                        scheduler._terminate(state, "TIME_LIMIT")
            if sum(s.last_rss_bytes for s in scheduler._running.values()) > scheduler.limits.total_rss_bytes:
                for state in scheduler._running.values():
                    if state.process.poll() is None:
                        scheduler._terminate(state, "TOTAL_MEMORY_LIMIT")
            replace_json(audit / "status.json", {"status": "DRAINING", "remaining": len(scheduler._running),
                "finalized": scheduler._finalized_count, "parent_pid": old_pid, "timestamp": time.time()})
            for pid, state in list(scheduler._running.items()):
                if state.process.poll() is None:
                    continue
                kernel = same_process(captured[pid])
                write_json_once(audit / f"kernel_exit_{pid}.json", {**kernel,
                    "job": state.spec.path.name, "returncode": state.process.returncode,
                    "elapsed_seconds": time.monotonic() - state.started,
                    "rss_observation": "handoff start through exit; earlier peak only available in worker result"})
                sealed = True
                scheduler._finalize(state)
                del scheduler._running[pid]
            if scheduler._running:
                time.sleep(.5)
        # Verify every captured receipt while the old parent still owns its lock.
        for spec in scheduler.jobs:
            verify_completed_job(spec.path, spec.config, spec.seed, registration)
        same_process(parent)
        # Never resume its in-memory _running table after external receipt seals.
        # All compute children have already exited; only the dispatcher is killed.
        os.kill(old_pid, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while Path(f"/proc/{old_pid}").exists() and time.monotonic() < deadline:
            if same_process(parent)["state"] == "Z":
                break
            time.sleep(.05)
        else:
            if Path(f"/proc/{old_pid}").exists():
                raise SchedulerError("old dispatcher did not exit")
        write_json_once(audit / "drain_complete.json", {"status": "DRAINED", "captured": len(captured),
            "sealed": scheduler._finalized_count, "failed": scheduler._failed_count,
            "old_dispatcher_intentional_signal": "SIGKILL of stopped dispatcher after all captured workers exited",
            "timestamp": time.time()})
    except BaseException:
        # Before receipt writes the original owner can safely resume. After writes
        # preserve its stopped state for recovery rather than double-finalizing.
        if not sealed:
            os.kill(old_pid, signal.SIGCONT)
        replace_json(audit / "status.json", {"status": "NEEDS_RECOVERY" if sealed else "ABORTED_OLD_QUEUE_RESUMED",
            "parent_pid": old_pid, "timestamp": time.time()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--old-pid", type=int)
    parser.add_argument("--methods", nargs="+")
    args = parser.parse_args()
    run = args.run.resolve()
    if args.workers > _effective_cpu_count() or args.workers * 4 * 1024**3 > 64 * 1024**3:
        parser.error("requested workers exceed effective CPU quota or aggregate memory reserve")
    limits = ResourceLimits(workers=args.workers, max_workers=args.workers)
    scheduler = ParallelBenchmarkScheduler(run, runner=ROOT / "scripts/run_nhis_benchmark.py",
        source_script=ROOT / "scripts/run_nhis_benchmark_parallel.py", limits=limits,
        methods=args.methods, policy_path=args.policy)
    audit = run / "operational_handoffs" / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    audit.mkdir(parents=True)
    with ExclusiveRunLock(run / "operational_handoff.lock"):
        write_json_once(audit / "request.json", {"workers": args.workers, "old_pid": args.old_pid,
            "methods": args.methods, "helper_sha256": file_sha(Path(__file__)),
            "policy_sha256": file_sha(args.policy), "old_live_status": json.loads((run / "parallel_scheduler_live_status.json").read_text())})
        if args.old_pid:
            drain(args.old_pid, scheduler, audit)
        # New instance has fresh counters, session id, and representation state.
        resumed = ParallelBenchmarkScheduler(run, runner=scheduler.runner, source_script=scheduler.source_script,
            limits=limits, methods=args.methods, policy_path=args.policy)
        write_json_once(audit / "resume_started.json", {"timestamp": time.time(), "pid": os.getpid(),
            "workers": args.workers, "helper_sha256": file_sha(Path(__file__)),
            "limits": {"workers": limits.workers, "max_workers": limits.max_workers,
                       "total_rss_bytes": limits.total_rss_bytes, "worker_rss_bytes": limits.worker_rss_bytes,
                       "fit_seconds": limits.fit_seconds, "poll_seconds": limits.poll_seconds},
            "registration_sha256": file_sha(resumed.registration_path),
            "runner_sha256": file_sha(resumed.runner), "source_script_sha256": file_sha(resumed.source_script),
            "policy_sha256": file_sha(args.policy), "session_id": resumed._session_id})
        summary = resumed.run()
        write_json_once(audit / "queue_exit.json", summary)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
