#!/usr/bin/env python3
"""Queue a hash-bound numerical recovery run without copying the registry.

The queue owns only the failed LFR seed jobs named by a metadata-only failure
manifest.  It uses a fresh run/cache namespace and counts live FRAPPÉ workers
from the supplied external runs against a shared 28-slot CPU budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from nhis_fairbias.benchmark.experiment_registry import identity  # noqa: E402
from nhis_fairbias.benchmark.parallel_execution import (  # noqa: E402
    JobSpec,
    ParallelBenchmarkScheduler,
    ResourceLimits,
    SchedulerError,
    THREAD_ENVIRONMENT,
    _representation_key,
    _effective_cpu_count,
    file_sha,
    replace_json,
    write_json_once,
)
from scripts.run_nhis_numerical_recovery_worker import (  # noqa: E402
    REQUIRED_SOURCES,
    THREAD_ENVIRONMENT as RECOVERY_THREAD_ENVIRONMENT,
    build_runtime_source_manifest,
    runtime_identity as recovery_runtime_identity,
)


RECOVERY_PLAN_VERSION = "nhis_numerical_recovery_plan_v1_20260917"
RECOVERY_VARIANT = "eg_lfr_numerical_recovery_v1_20260917"
MAX_RECOVERY_WORKERS = 28
HOST_RESERVE_BYTES = 12 * 1024**3


def _json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except Exception as exc:
        raise SchedulerError(f"unreadable JSON: {path}") from exc


def _runtime_plan(path: Path) -> dict[str, Any]:
    plan = _json(path)
    if plan.get("schema_version") != RECOVERY_PLAN_VERSION:
        raise SchedulerError("unsupported numerical recovery plan schema")
    if plan.get("runtime_variant") != RECOVERY_VARIANT:
        raise SchedulerError("unexpected numerical recovery runtime variant")
    environment = plan.get("runtime_environment")
    if environment != RECOVERY_THREAD_ENVIRONMENT:
        raise SchedulerError("recovery runtime environment must match the worker contract")
    files = plan.get("runtime_source_files")
    if not isinstance(files, dict) or not files:
        raise SchedulerError("recovery runtime source manifest is missing")
    worker_script = plan.get("worker_script", "scripts/run_nhis_numerical_recovery_worker.py")
    if worker_script not in files:
        raise SchedulerError("recovery worker is absent from runtime source manifest")
    if not REQUIRED_SOURCES.issubset(files):
        raise SchedulerError("recovery runtime source manifest omits required sources")
    for relative, expected in files.items():
        relative_path = Path(relative)
        source = (ROOT / relative_path).resolve()
        if relative_path.is_absolute() or not source.is_relative_to(ROOT) or not source.is_file():
            raise SchedulerError(f"unsafe or missing recovery source: {relative}")
        if file_sha(source) != expected:
            raise SchedulerError(f"recovery source hash mismatch: {relative}")
    expected_identity = recovery_runtime_identity(files, environment)
    if plan.get("runtime_source_identity") != expected_identity:
        raise SchedulerError("recovery runtime source identity mismatch")
    return plan


def _registered_lfr(registration: Mapping[str, Any]) -> dict[str, tuple[Mapping[str, Any], int]]:
    result: dict[str, tuple[Mapping[str, Any], int]] = {}
    for config in registration.get("candidates", []):
        if config.get("status") != "REGISTERED" or config.get("method") != "LFR_RECONSTRUCTED":
            continue
        for seed in config.get("seeds", []):
            name = f"{config['candidate_id']}_s{int(seed)}"
            if name in result:
                raise SchedulerError(f"duplicate registered LFR job: {name}")
            result[name] = (config, int(seed))
    return result


def _failure_jobs(original_run: Path, manifest_path: Path,
                  registration: Mapping[str, Any], expected_count: int) -> dict[str, dict[str, Any]]:
    manifest = _json(manifest_path)
    if manifest.get("schema") != "failure_recovery_inventory_v1":
        raise SchedulerError("unsupported failure recovery inventory schema")
    registration_path = original_run / "registration.json"
    if manifest.get("registration_sha256") != file_sha(registration_path):
        raise SchedulerError("failure inventory registration hash mismatch")
    registered = _registered_lfr(registration)
    if len(registered) != expected_count:
        raise SchedulerError(f"registration contains {len(registered)} LFR jobs, expected {expected_count}")
    entries = manifest.get("failures")
    if not isinstance(entries, list):
        raise SchedulerError("failure inventory lacks failures list")
    selected: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("config", {}).get("method") != "LFR_RECONSTRUCTED":
            continue
        name = str(entry.get("job_id", ""))
        if name in selected:
            raise SchedulerError(f"duplicate failure inventory job: {name}")
        if name not in registered:
            raise SchedulerError(f"failure inventory job is not registered LFR: {name}")
        config, seed = registered[name]
        if entry.get("config") != config or int(entry.get("seed", -1)) != seed:
            raise SchedulerError(f"failure inventory identity mismatch: {name}")
        original_job = (original_run / "jobs" / name / "job.json").resolve()
        declared_job = Path(entry.get("job_path", "")).resolve()
        if declared_job != original_job or not original_job.is_file():
            raise SchedulerError(f"failure inventory job path mismatch: {name}")
        result_path = original_job.parent / "result.json"
        if not result_path.is_file() or entry.get("job_sha256") != file_sha(original_job):
            raise SchedulerError(f"failure inventory job evidence mismatch: {name}")
        if entry.get("result_sha256") != file_sha(result_path):
            raise SchedulerError(f"failure inventory result evidence mismatch: {name}")
        if entry.get("status") == "VALID":
            raise SchedulerError(f"valid job cannot enter numerical recovery: {name}")
        expected_key = _representation_key(config, seed, registration["prepared"][config["arm_id"]]["data_identity"])
        if entry.get("representation_key") != expected_key:
            raise SchedulerError(f"failure inventory representation key mismatch: {name}")
        selected[name] = {
            "config": config,
            "seed": seed,
            "original_job_path": str(original_job),
            "original_job_sha256": file_sha(original_job),
            "original_result_sha256": file_sha(result_path),
            "original_representation_key": entry.get("representation_key"),
        }
    if set(selected) != set(registered):
        missing = sorted(set(registered) - set(selected))
        extra = sorted(set(selected) - set(registered))
        raise SchedulerError(f"failure inventory does not exactly cover registered LFR: missing={missing[:3]} extra={extra[:3]}")
    if len(selected) != expected_count:
        raise SchedulerError("failure inventory LFR count mismatch")
    return selected


def external_worker_count(runs: Iterable[Path]) -> int:
    """Count live workers whose job paths belong to the supplied runs."""
    roots = {Path(run).resolve() for run in runs}
    seen: set[int] = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        try:
            state = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
            if state == "Z":
                continue
            argv = [part.decode() for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
            if "worker" not in argv or "--job" not in argv:
                continue
            job = Path(argv[argv.index("--job") + 1]).resolve()
            if job.name != "job.json":
                continue
            if any(job.parent.parent == root / "jobs" for root in roots):
                seen.add(pid)
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    return len(seen)


def cgroup_memory() -> tuple[int, int]:
    try:
        current = int(Path("/sys/fs/cgroup/memory.current").read_text())
        maximum_text = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        maximum = int(maximum_text)
    except (FileNotFoundError, OSError, ValueError):
        raise SchedulerError("finite cgroup memory.current and memory.max are required")
    if current < 0 or maximum <= 0 or current > maximum:
        raise SchedulerError("invalid cgroup memory accounting")
    return current, maximum


class NumericalRecoveryScheduler(ParallelBenchmarkScheduler):
    """Scheduler over an exact failed LFR set with a fresh runtime namespace."""

    def __init__(self, namespace: Path, *, original_run: Path, selected: Mapping[str, Mapping[str, Any]],
                 runtime_plan: Mapping[str, Any], runtime_plan_path: Path,
                 external_runs: Iterable[Path], worker: Path, **kwargs: Any) -> None:
        self.original_run = Path(original_run).resolve()
        self.selected = dict(selected)
        self.runtime_plan = dict(runtime_plan)
        self.runtime_plan_path = Path(runtime_plan_path).resolve()
        self.external_runs = tuple(Path(run).resolve() for run in external_runs)
        self.worker_script = Path(worker).resolve()
        namespace_resolved = Path(namespace).resolve()
        if self.original_run in self.external_runs or namespace_resolved in self.external_runs:
            raise SchedulerError("recovery run cannot count itself as an external run")
        # The base scheduler loads the original registration, then all mutable
        # scheduler paths are rebound to the fresh namespace below.
        super().__init__(self.original_run, runner=self.worker_script, **kwargs)
        self.run_path = Path(namespace).resolve()
        self.registration_path = self.original_run / "registration.json"
        self._events_path = self.run_path / "parallel_scheduler_events.jsonl"
        self._live_status_path = self.run_path / "parallel_scheduler_live_status.json"
        self.cache_path = self.run_path / "representation_cache"
        self._selected_names = set(self.selected)
        self._runtime_identity = str(self.runtime_plan["runtime_source_identity"])

    def discover(self) -> list[JobSpec]:
        self._validate_sources()
        self._validate_parallel_policy()
        self._write_manifest()
        jobs: list[JobSpec] = []
        for name in sorted(self._selected_names):
            selected = self.selected[name]
            config = selected["config"]
            seed = int(selected["seed"])
            data = self.registration["prepared"][config["arm_id"]]
            original_key = selected["original_representation_key"]
            if not isinstance(original_key, str):
                raise SchedulerError(f"missing original representation key: {name}")
            jobs.append(JobSpec(config=config, seed=seed,
                                path=self.run_path / "jobs" / name, data=data,
                                source_identity=self.registration["source_identity"],
                                representation_key=original_key))
        self.jobs = jobs
        return jobs

    def _job_payload(self, spec: JobSpec) -> dict[str, Any]:
        payload = super()._job_payload(spec)
        selected = self.selected[spec.path.name]
        payload.update({
            "runtime_variant": self.runtime_plan["runtime_variant"],
            "runtime_environment": self.runtime_plan["runtime_environment"],
            "runtime_plan_sha256": file_sha(self.runtime_plan_path),
            "runtime_source_files": self.runtime_plan["runtime_source_files"],
            "runtime_source_identity": self._runtime_identity,
            "registration_path": str(self.registration_path),
            "registration_sha256": file_sha(self.registration_path),
            "recovery_run_path": str(self.run_path),
            "original_job_path": selected["original_job_path"],
            "original_job_sha256": selected["original_job_sha256"],
            "original_result_sha256": selected["original_result_sha256"],
            "original_representation_key": selected["original_representation_key"],
            "cache_path": str(self.cache_path),
        })
        return payload

    def _environment(self, spec: JobSpec) -> dict[str, str]:
        environment = super()._environment(spec)
        environment.update(self.runtime_plan["runtime_environment"])
        return environment

    def _write_manifest(self) -> None:
        super()._write_manifest()
        self.cache_path.mkdir(parents=True, exist_ok=True)
        namespace_marker = {
            "runtime_variant": self.runtime_plan["runtime_variant"],
            "runtime_source_identity": self._runtime_identity,
            "registration_sha256": file_sha(self.registration_path),
        }
        marker_path = self.cache_path / ".runtime_namespace.json"
        if marker_path.exists() and _json(marker_path) != namespace_marker:
            raise SchedulerError("recovery cache namespace marker mismatch")
        if not marker_path.exists() and not write_json_once(marker_path, namespace_marker):
            raise SchedulerError("recovery cache namespace marker creation raced")
        audit = {
            "schema_version": RECOVERY_PLAN_VERSION,
            "runtime_plan_sha256": file_sha(self.runtime_plan_path),
            "runtime_source_identity": self._runtime_identity,
            "registration_sha256": file_sha(self.registration_path),
            "failure_manifest_sha256": file_sha(self.runtime_plan["failure_manifest_path"]),
            "original_run": str(self.original_run),
            "external_runs": [str(path) for path in self.external_runs],
            "workers": self.limits.workers,
            "host_memory_reserve_bytes": HOST_RESERVE_BYTES,
            "selected_jobs": len(self.selected),
            "cache_path": str(self.cache_path),
        }
        target = self.run_path / "recovery_dispatch_manifest.json"
        if target.exists() and _json(target) != audit:
            raise SchedulerError("recovery dispatch manifest identity mismatch")
        if not target.exists() and not write_json_once(target, audit):
            raise SchedulerError("recovery dispatch manifest creation raced")

    def _seal_failed_cache(self, spec: JobSpec, reason: str | None) -> None:
        """Seal scheduler failures with the worker's required cache sidecar."""
        key = spec.representation_key
        if not key:
            return super()._seal_failed_cache(spec, reason)
        status_path = self.cache_path / f"{key}.json"
        model_path = self.cache_path / f"{key}.joblib"
        sidecar = self.cache_path / f"{key}.runtime.json"
        if status_path.is_file() and sidecar.is_file():
            try:
                evidence = _json(sidecar)
                expected = {
                    "runtime_variant": self.runtime_plan["runtime_variant"],
                    "runtime_source_identity": self._runtime_identity,
                    "registration_sha256": file_sha(self.registration_path),
                    "representation_key": key,
                    "files": {suffix: file_sha(path) for suffix, path in
                              ((".json", status_path), (".joblib", model_path)) if path.is_file()},
                }
                if evidence == expected:
                    return
            except Exception:
                pass
            self._write_cache_orphan(spec, "existing cache evidence failed verification")
            return
        had_partial = status_path.exists() or model_path.exists() or sidecar.exists()
        super()._seal_failed_cache(spec, reason)
        if not status_path.is_file():
            return
        if had_partial or model_path.exists():
            self._write_cache_orphan(spec, "partial or pre-existing cache was not runtime-verified")
            return
        evidence = {
            "runtime_variant": self.runtime_plan["runtime_variant"],
            "runtime_source_identity": self._runtime_identity,
            "registration_sha256": file_sha(self.registration_path),
            "representation_key": key,
            "files": {".json": file_sha(status_path)},
        }
        if sidecar.exists():
            if _json(sidecar) != evidence:
                self._write_cache_orphan(spec, "scheduler cache sidecar identity mismatch")
        elif not write_json_once(sidecar, evidence):
            self._write_cache_orphan(spec, "scheduler cache sidecar creation raced")

    def _write_cache_orphan(self, spec: JobSpec, reason: str) -> None:
        key = spec.representation_key
        if not key:
            return
        path = self.cache_path / f"{key}.scheduler_orphan.json"
        payload = {
            "status": "SCHEDULER_CACHE_ORPHAN",
            "reason": reason,
            "representation_key": key,
            "runtime_source_identity": self._runtime_identity,
        }
        if not path.exists():
            write_json_once(path, payload)

    def _receipt_error(self, state: Any) -> str | None:
        if state.termination is not None:
            return "worker terminated before runtime evidence was accepted"
        path = state.spec.path / "runtime_receipt.json"
        if not path.is_file():
            return "missing runtime receipt"
        try:
            receipt = _json(path)
        except Exception:
            return "unreadable runtime receipt"
        if not isinstance(receipt, dict):
            return "runtime receipt is not an object"
        selected = self.selected[state.spec.path.name]
        required = {
            "runtime_variant": self.runtime_plan["runtime_variant"],
            "runtime_source_identity": self._runtime_identity,
            "registration_sha256": file_sha(self.registration_path),
            "original_job_sha256": selected["original_job_sha256"],
            "original_result_sha256": selected["original_result_sha256"],
            "candidate_id": state.spec.candidate_id,
            "seed": state.spec.seed,
        }
        if any(receipt.get(key) != value for key, value in required.items()):
            return "runtime receipt identity mismatch"
        status = receipt.get("status")
        if not isinstance(status, str) or status not in {"VALID", "FAILED", "BUDGET_EXHAUSTED", "NOT_SUPPORTED"}:
            return "runtime receipt has no accepted terminal status"
        # Exit 1 is the worker's documented exit for a completed, non-VALID fit.
        # It is distinct from an interruption, crash, or missing evidence.
        allowed_exit = (0,) if status == "VALID" else (0, 1)
        if state.process.returncode not in allowed_exit:
            return "worker exit is inconsistent with its terminal status"
        if receipt.get("runtime_checks_passed") is not True:
            return "recovery lacks runtime checks"
        if receipt.get("worker_result_status") != status:
            return "runtime receipt status differs from worker result"
        files = receipt.get("files")
        if not isinstance(files, dict) or files.get("job.json") != file_sha(state.spec.path / "job.json"):
            return "runtime receipt job hash mismatch"
        for name, expected in files.items():
            candidate = Path(name)
            target = state.spec.path / candidate
            if (candidate.is_absolute() or ".." in candidate.parts or candidate.name != name
                    or not target.is_file() or file_sha(target) != expected):
                return "runtime receipt output hash mismatch"
        if "result.json" not in files:
            return "runtime receipt omits result hash"
        try:
            result = _json(state.spec.path / "result.json")
        except Exception:
            return "unreadable worker result"
        if not isinstance(result, dict) or result.get("status") != status:
            return "runtime receipt has inconsistent result status"
        if (result.get("candidate_id") != state.spec.candidate_id
                or result.get("seed") != state.spec.seed):
            return "worker result identity mismatch"
        if status == "VALID":
            if receipt.get("runtime_checks_passed") is not True:
                return "VALID recovery lacks runtime checks"
            if not {"result.json", "model.joblib", "predictions_S.npz"}.issubset(files):
                return "VALID recovery omits required output hashes"
            if (result.get("source_identity") != state.spec.source_identity
                    or result.get("data_identity") != state.spec.data["data_identity"]
                    or result.get("reload_verified") is not True):
                return "VALID result lacks identity or reload evidence"
        return None

    def _write_scheduler_failure(self, state: Any, reason: str) -> None:
        """Seal a killed/malformed attempt as an independent non-VALID result."""
        result_path = state.spec.path / "result.json"
        if result_path.is_file():
            raw = result_path.read_bytes()
            backup = result_path.with_name("result.worker_unadmitted.json")
            if not backup.exists():
                backup.write_bytes(raw)
        replace_json(result_path, {
            "candidate_id": state.spec.candidate_id, "seed": state.spec.seed,
            "status": "SCHEDULER_RUNTIME_FAILED", "scheduler_generated": True,
            "reason": reason, "source_identity": state.spec.source_identity,
            "data_identity": state.spec.data["data_identity"],
        })
        receipt_path = state.spec.path / "runtime_receipt.json"
        if receipt_path.is_file():
            backup = state.spec.path / "runtime_receipt.worker_invalid.json"
            if not backup.exists():
                backup.write_bytes(receipt_path.read_bytes())
        synthetic = {
            "schema_version": "nhis_numerical_recovery_receipt_v1",
            "runtime_variant": self.runtime_plan["runtime_variant"],
            "runtime_source_identity": self._runtime_identity,
            "registration_sha256": file_sha(self.registration_path),
            "original_job_sha256": self.selected[state.spec.path.name]["original_job_sha256"],
            "original_result_sha256": self.selected[state.spec.path.name]["original_result_sha256"],
            "candidate_id": state.spec.candidate_id, "seed": state.spec.seed,
            "status": "SCHEDULER_RUNTIME_FAILED", "runtime_checks_passed": False,
            "error_type": "SchedulerRuntimeFailure", "error_detail": reason,
            "files": {"job.json": file_sha(state.spec.path / "job.json")},
        }
        if receipt_path.exists():
            replace_json(receipt_path, synthetic)
        else:
            write_json_once(receipt_path, synthetic)

    def _finalize(self, state: Any) -> dict[str, Any]:
        error = self._receipt_error(state)
        if error is not None:
            self._write_scheduler_failure(state, error)
        return super()._finalize(state)

    def _launchable_index(self, pending: list[JobSpec], slots: int) -> int | None:
        external = external_worker_count(self.external_runs)
        if len(self._running) + external >= self.limits.workers:
            return None
        current, maximum = cgroup_memory()
        if current + int(1.5 * 1024**3) > maximum - HOST_RESERVE_BYTES:
            return None
        return super()._launchable_index(pending, slots)


def _namespace(root: Path, runtime_identity: str) -> Path:
    if not runtime_identity or any(ch not in "0123456789abcdef" for ch in runtime_identity):
        raise SchedulerError("runtime source identity must be hexadecimal")
    return Path(root).resolve() / f"lfr_recovery_{runtime_identity}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="fresh recovery output root")
    parser.add_argument("--original-run", required=True, type=Path)
    parser.add_argument("--failure-manifest", required=True, type=Path)
    parser.add_argument("--runtime-plan", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--external-run", action="append", required=True)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--workers", type=int, default=MAX_RECOVERY_WORKERS)
    parser.add_argument("--expected-jobs", type=int, default=960)
    args = parser.parse_args(argv)
    try:
        if args.workers < 1 or args.workers > MAX_RECOVERY_WORKERS:
            raise SchedulerError(f"recovery workers must be between 1 and {MAX_RECOVERY_WORKERS}")
        if args.expected_jobs != 960:
            raise SchedulerError("formal numerical recovery requires all 960 registered LFR jobs")
        if args.workers > _effective_cpu_count():
            raise SchedulerError("recovery workers exceed effective CPU quota")
        original_run = args.original_run.resolve()
        if not original_run.is_dir() or not (original_run / "registration.json").is_file():
            raise SchedulerError("original run registration is missing")
        runtime_plan = _runtime_plan(args.runtime_plan.resolve())
        runtime_plan["failure_manifest_path"] = str(args.failure_manifest.resolve())
        registration = _json(original_run / "registration.json")
        selected = _failure_jobs(original_run, args.failure_manifest.resolve(), registration, args.expected_jobs)
        runtime_identity = runtime_plan["runtime_source_identity"]
        namespace = _namespace(args.run, runtime_identity)
        if (namespace.exists() or namespace == original_run
                or original_run in namespace.parents or namespace in original_run.parents):
            raise SchedulerError("recovery namespace must be fresh and outside original run")
        worker = (args.worker or (ROOT / runtime_plan.get("worker_script", "scripts/run_nhis_numerical_recovery_worker.py"))).resolve()
        if not worker.is_file() or worker != (ROOT / runtime_plan.get("worker_script", "scripts/run_nhis_numerical_recovery_worker.py")).resolve():
            raise SchedulerError("recovery worker path is not the plan-bound worker")
        policy = _json(args.policy.resolve())
        total_rss = int(policy.get("max_total_rss_bytes", 0))
        worker_rss = int(policy.get("worker_rss_bytes", 4 * 1024**3))
        fit_seconds = float(policy.get("fit_seconds", 1800.0))
        if total_rss <= 0 or worker_rss <= 0 or fit_seconds <= 0:
            raise SchedulerError("recovery policy has invalid resource limits")
        _, cgroup_max = cgroup_memory()
        if total_rss > cgroup_max - HOST_RESERVE_BYTES:
            raise SchedulerError("recovery policy does not preserve the 12 GiB cgroup reserve")
        namespace.mkdir(parents=True)
        (namespace / "jobs").mkdir()
        (namespace / "representation_cache").mkdir()
        limits = ResourceLimits(workers=args.workers, max_workers=args.workers,
                                total_rss_bytes=total_rss, worker_rss_bytes=worker_rss,
                                fit_seconds=fit_seconds)
        scheduler = NumericalRecoveryScheduler(
            namespace, original_run=original_run, selected=selected,
            runtime_plan=runtime_plan, runtime_plan_path=args.runtime_plan,
            external_runs=args.external_run, worker=worker,
            source_script=ROOT / "scripts/run_nhis_benchmark_parallel.py",
            limits=limits, methods=["LFR_RECONSTRUCTED"],
            policy_path=args.policy.resolve())
        print(json.dumps(scheduler.run()), flush=True)
        return 0
    except (SchedulerError, ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"numerical recovery queue refused to run: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
