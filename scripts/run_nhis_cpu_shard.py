#!/usr/bin/env python3
"""Run one supervisor-authorized shard of the registered NHIS benchmark.

This is an operational wrapper.  It never changes registration or scientific
source; the plan only restricts which already-registered job directories this
scheduler may create or resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import signal
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhis_fairbias.benchmark.parallel_execution import (  # noqa: E402
    ParallelBenchmarkScheduler, ResourceLimits, SchedulerError, _effective_cpu_count,
    _representation_key, file_sha, write_json_once, verify_completed_job,
)

SHARD_PLAN_VERSION = "nhis_cpu_shard_plan_v1_20260917"
SHARD_AUDIT_VERSION = "nhis_cpu_shard_audit_v1_20260917"

def await_released_run_lock(run: Path, timeout: float = 10.0):
    """Wait for actual kernel lock release after a drained dispatcher exits."""
    from nhis_fairbias.benchmark.parallel_execution import ExclusiveRunLock
    deadline = time.monotonic() + timeout
    while True:
        try:
            with ExclusiveRunLock(run / 'parallel_scheduler.lock'):
                return
        except SchedulerError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.05)


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise SchedulerError(f"unreadable JSON: {path}") from exc


def _job_name(candidate_id: str, seed: int) -> str:
    return f"{candidate_id}_s{seed}"


def _sha(path: Path) -> str:
    return file_sha(path)


def _safe_stop_at_boundary(pid: int, run: Path, runner: Path):
    """Capture either normal-dispatcher or handoff-parent polling boundary."""
    from handoff_nhis_parallel import children, proc, same_process
    parent = proc(pid)
    args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    run_b = str(run).encode()
    handoff = str(ROOT / "scripts/handoff_nhis_parallel.py").encode()
    normal = str(ROOT / "scripts/run_nhis_benchmark_parallel.py").encode()
    valid_parent = ((normal in args and b"develop" in args) or
                    (handoff in args and b"--run" in args))
    if (run_b not in args or not valid_parent
            or Path(f"/proc/{pid}/cwd").resolve() != runner.parent.parent):
        raise SchedulerError("PID is not a requested development or handoff dispatcher")
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
        keep = False
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
            keep = True
            return parent, before
        finally:
            if not keep:
                os.kill(pid, signal.SIGCONT)
        time.sleep(.01)
    raise SchedulerError("could not capture a stable polling boundary; old queue remains running")


class ShardedScheduler(ParallelBenchmarkScheduler):
    """ParallelBenchmarkScheduler restricted by an operator-approved, hash-bound plan."""

    def __init__(self, *args: Any, plan_path: Path, role: str,
                 activation_path: Path | None = None, **kwargs: Any) -> None:
        if role not in {"source", "cpu"}:
            raise ValueError("role must be source or cpu")
        self.plan_path = Path(plan_path).resolve()
        self.role = role
        self.activation_path = Path(activation_path).resolve() if activation_path else None
        super().__init__(*args, **kwargs)
        self.plan = _load(self.plan_path)
        self._validate_plan()
        self._owned_names = set(self.plan["assignments"][role])
        self._shard_lock_path = self.run_path / f"parallel_scheduler_{role}.lock"

    def _universe(self) -> tuple[dict[str, tuple[Mapping[str, Any], int]], dict[str, str]]:
        jobs: dict[str, tuple[Mapping[str, Any], int]] = {}
        keys: dict[str, str] = {}
        for config in self.registration.get("candidates", []):
            if config.get("status") != "REGISTERED":
                continue
            data = self.registration["prepared"][config["arm_id"]]
            for seed in config["seeds"]:
                name = _job_name(str(config["candidate_id"]), int(seed))
                if name in jobs:
                    raise SchedulerError(f"duplicate registered job name: {name}")
                jobs[name] = (config, int(seed))
                key = _representation_key(config, int(seed), data["data_identity"])
                if key is not None:
                    keys[name] = key
        return jobs, keys

    def _validate_plan(self) -> None:
        plan = self.plan
        if plan.get("schema_version") != SHARD_PLAN_VERSION:
            raise SchedulerError("unsupported shard plan schema")
        if plan.get("registration_sha256") != _sha(self.registration_path):
            raise SchedulerError("shard plan registration hash mismatch")
        if plan.get("source_identity") != self.registration.get("source_identity"):
            raise SchedulerError("shard plan source identity mismatch")
        assignments = plan.get("assignments")
        if not isinstance(assignments, dict) or set(assignments) != {"source", "cpu"}:
            raise SchedulerError("assignments must contain exactly source and cpu")
        if any(not isinstance(v, list) for v in assignments.values()):
            raise SchedulerError("shard assignments must be lists")
        universe, keys = self._universe()
        source, cpu = map(set, (assignments["source"], assignments["cpu"]))
        if len(source) != len(assignments["source"]) or len(cpu) != len(assignments["cpu"]):
            raise SchedulerError("duplicate job in shard assignment")
        if source & cpu:
            raise SchedulerError("shard assignments overlap")
        if source | cpu != set(universe):
            raise SchedulerError("shard assignments are not the exact registered job universe")
        if (source | cpu) - set(universe):
            raise SchedulerError("unknown job in shard assignment")
        owners_by_key: dict[str, str] = {}
        for name, key in keys.items():
            owner = "source" if name in source else "cpu"
            if key in owners_by_key and owners_by_key[key] != owner:
                raise SchedulerError(f"representation key split across shards: {key}")
            owners_by_key[key] = owner
        for name in cpu:
            config, seed = universe[name]
            if config.get("method") != "FRAPPE_EO":
                raise SchedulerError("cpu shard may own only FRAPPE_EO jobs")
            if self.role == "source" and (self.run_path / "jobs" / name).exists():
                raise SchedulerError(f"cpu shard job is already started: {name}")
        hosts = plan.get("hostnames", {})
        if not isinstance(hosts, dict) or hosts.get(self.role) != os.uname().nodename:
            raise SchedulerError("shard plan hostname binding mismatch")
        for relative, expected in self.registration.get("source_files", {}).items():
            if not (self.root / relative).is_file() or _sha(self.root / relative) != expected:
                raise SchedulerError(f"registered source changed: {relative}")
        self._validate_activation()

    def _validate_activation(self) -> None:
        if self.activation_path is None or not self.activation_path.is_file():
            raise SchedulerError("explicit shard activation JSON is required")
        activation = _load(self.activation_path)
        expected_plan = _sha(self.plan_path)
        if activation.get("schema_version") != SHARD_AUDIT_VERSION:
            raise SchedulerError("unsupported shard activation schema")
        if activation.get("plan_sha256") != expected_plan or activation.get("role") != self.role:
            raise SchedulerError("activation does not bind to this plan and role")
        if activation.get("registration_sha256") != _sha(self.registration_path):
            raise SchedulerError("activation registration hash mismatch")
        if activation.get("source_identity") != self.registration.get("source_identity"):
            raise SchedulerError("activation source identity mismatch")
        if self.role == "cpu":
            for field in ("source_drain_complete_sha256", "source_request_sha256"):
                value = activation.get(field)
                if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise SchedulerError(f"CPU activation lacks valid {field}")

    def discover(self) -> list[Any]:
        jobs = super().discover()
        self.resumed = 0
        for name, (config, seed) in self._universe()[0].items():
            if name not in self._owned_names:
                continue
            if self.methods and config['method'] not in self.methods:
                continue
            if self.backbones and config.get('backbone') not in self.backbones:
                continue
            if verify_completed_job(self.run_path / 'jobs' / name, config, seed, self.registration) is not None:
                self.resumed += 1
        self.jobs = [job for job in jobs if job.path.name in self._owned_names]
        return self.jobs

    def _write_shard_audit(self) -> None:
        payload = {
            "schema_version": SHARD_AUDIT_VERSION,
            "role": self.role,
            "plan_sha256": _sha(self.plan_path),
            "registration_sha256": _sha(self.registration_path),
            "source_identity": self.registration["source_identity"],
            "source_script_sha256": _sha(self.source_script),
            "scheduler_module_sha256": _sha(Path(__file__).parents[1] / "src/nhis_fairbias/benchmark/parallel_execution.py"),
            "policy_sha256": _sha(self.policy_path) if self.policy_path else None,
            "assignments": self.plan["assignments"],
            "hostname": os.uname().nodename,
            "pid": os.getpid(),
            "activation_sha256": _sha(self.activation_path),
            "wrapper_sha256": _sha(Path(__file__)),
            "admission": ({"enabled": self.role == "cpu", "floor_gib": 2.25,
                           "aggregate_reserve_gib": 2.0, "host_reserve_gib": 6.0,
                           "max_workers": self.limits.workers} if self.role == "cpu"
                          else {"enabled": False}),
        }
        target = self.run_path / f"shard_{self.role}_activation.json"
        if target.exists():
            existing = _load(target)
            if (existing.get("plan_sha256") != payload["plan_sha256"] or
                    existing.get("role") != self.role or
                    existing.get("assignments") != payload["assignments"]):
                raise SchedulerError("conflicting shard audit ownership")
        elif not write_json_once(target, payload):
            raise SchedulerError("shard audit creation raced")

    def _launchable_index(self, pending: list[Any], slots: int) -> int | None:
        if self.role != "cpu":
            return super()._launchable_index(pending, slots)
        floor = int(2.25 * 1024**3)
        largest = max((s.last_rss_bytes for s in self._running.values()), default=0)
        reserve = max(floor, int(largest * 1.2))
        projected = sum(max(s.last_rss_bytes, reserve) for s in self._running.values()) + reserve
        if projected > self.limits.total_rss_bytes - 2 * 1024**3:
            return None
        try:
            current = int(Path("/sys/fs/cgroup/memory.current").read_text())
            maximum = Path("/sys/fs/cgroup/memory.max").read_text().strip()
            if maximum != "max" and current + reserve > int(maximum) - 6 * 1024**3:
                return None
        except (FileNotFoundError, OSError, ValueError):
            pass
        return super()._launchable_index(pending, slots)

    def run(self) -> dict[str, int]:
        self._validate_sources()
        self._validate_parallel_policy()
        self._write_shard_audit()
        return super().run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--activation", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("source", "cpu"))
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--methods", nargs="+", help="registered methods to include")
    parser.add_argument("--old-pid", type=int)
    args = parser.parse_args(argv)
    try:
        policy = _load(args.policy)
        total_rss = int(policy.get("max_total_rss_bytes", 0))
        if total_rss <= 0:
            raise SchedulerError("policy lacks max_total_rss_bytes")
        if args.role == "cpu":
            effective = _effective_cpu_count()
            if args.workers > effective:
                raise SchedulerError("CPU shard workers exceed effective CPU quota")
            try:
                maximum = Path("/sys/fs/cgroup/memory.max").read_text().strip()
                if maximum != "max" and total_rss > int(maximum) - 6 * 1024**3:
                    raise SchedulerError("CPU shard total RSS exceeds cgroup reserve")
            except FileNotFoundError:
                pass
        limits = ResourceLimits(workers=args.workers, max_workers=args.workers,
                                total_rss_bytes=total_rss)
        scheduler = ShardedScheduler(args.run, runner=ROOT / "scripts/run_nhis_benchmark.py",
            source_script=ROOT / "scripts/run_nhis_benchmark_parallel.py", limits=limits,
            policy_path=args.policy, plan_path=args.plan, activation_path=args.activation,
            role=args.role, methods=args.methods)
        if args.old_pid is not None:
            if args.role != "source":
                raise SchedulerError("--old-pid is valid only for the source shard")
            import datetime as dt
            import handoff_nhis_parallel as handoff
            from nhis_fairbias.benchmark.parallel_execution import ExclusiveRunLock
            audit = scheduler.run_path / "operational_handoffs" / (
                dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_cpu_shard")
            audit.mkdir(parents=True)
            with ExclusiveRunLock(scheduler.run_path / "cpu_shard_handoff.lock"):
                write_json_once(audit / "request.json", {
                    "role": "source", "old_pid": args.old_pid,
                    "plan_sha256": _sha(scheduler.plan_path),
                    "activation_sha256": _sha(scheduler.activation_path),
                    "wrapper_sha256": _sha(Path(__file__)),
                })
                original_stop = handoff.stop_at_boundary
                def guarded_stop(pid, run, runner):
                    parent, captured = _safe_stop_at_boundary(pid, run, runner)
                    outside = [info["job_path"] for info in captured.values()
                               if Path(info["job_path"]).name not in scheduler._owned_names]
                    if outside:
                        # drain() invokes its boundary hook before its try block.
                        # This guard therefore owns rollback of its own stop.
                        os.kill(pid, signal.SIGCONT)
                        raise SchedulerError(f"captured job outside source shard: {outside}")
                    return parent, captured
                handoff.stop_at_boundary = guarded_stop
                try:
                    handoff.drain(args.old_pid, scheduler, audit)
                finally:
                    handoff.stop_at_boundary = original_stop
            await_released_run_lock(scheduler.run_path)
            result = ShardedScheduler(args.run, runner=scheduler.runner,
                source_script=scheduler.source_script, limits=limits,
                policy_path=args.policy, plan_path=args.plan,
                activation_path=args.activation, role=args.role, methods=args.methods).run()
        else:
            result = scheduler.run()
    except (SchedulerError, ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"cpu shard refused to run: {exc}\n")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
