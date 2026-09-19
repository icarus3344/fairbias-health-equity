"""Bounded, resume-safe orchestration for the registered F/C/S benchmark.

This module deliberately launches the existing ``experiment_worker`` through
``run_nhis_benchmark.py``.  It only owns scheduling, resource supervision and
evidence sealing; it does not change an adapter, a candidate, a seed or a
scientific budget.
"""
from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Optional
import uuid


SCHEDULER_VERSION = "parallel_scheduler_v1_20260916"
MANDATORY_JOB_FILES = frozenset(("job.json", "result.json", "worker.log"))
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def _representation_key(config: Mapping[str, Any], seed: int, data_identity: str) -> Optional[str]:
    """Mirror the worker's cache-key contract without importing optional adapters."""
    from .experiment_registry import identity
    method, params = config["method"], config["params"]
    if method == "FAIRBIAS_BM" or method.startswith("FAIRBIAS_GEOMETRY_"):
        defaults = {"epsilon_ratio": .5, "max_iterations": 50, "max_geometry_evaluations": 20000,
                    "geometry_profile": "stress_elbow", "phi_threshold": 100., "algorithm_version": "application_v1"}
        spec = {key: params.get(key, value) for key, value in defaults.items()}
        family = "FAIRBIAS_BM"
    elif method == "LFR_RECONSTRUCTED":
        spec = {key: params[key] for key in ("k", "Az", "Ax", "Ay", "maxiter", "maxfun")}
        family = method
    else:
        return None
    return identity({"data": data_identity, "method": family, "seed": seed, "representation": spec})


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _effective_cpu_count() -> int:
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, max(1, int(quota) // int(period)))
    except (FileNotFoundError, OSError, ValueError, ZeroDivisionError):
        pass
    return count


def write_json_once(path: Path, value: Any) -> bool:
    """Create a JSON file atomically and never replace an existing file.

    A hard-link from a same-directory temporary file gives us an atomic
    create-if-absent operation.  This is used for receipts and scheduler
    generated cache status records, where replacement would hide an earlier
    attempt or a competing writer.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    try:
        fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        try:
            os.link(str(temporary), str(path))
            return True
        except FileExistsError:
            return False
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def replace_json(path: Path, value: Any) -> None:
    """Atomically refresh mutable scheduler status without replacing evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class SchedulerError(RuntimeError):
    """Raised when a run cannot be resumed or safely scheduled."""


@dataclasses.dataclass(frozen=True)
class ResourceLimits:
    workers: int = 8
    max_workers: int = 12
    total_rss_bytes: int = 64 * 1024**3
    worker_rss_bytes: int = 4 * 1024**3
    fit_seconds: float = 1800.0
    poll_seconds: float = 0.5

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in (self.workers, self.max_workers)):
            raise ValueError("worker counts must be integers")
        if not 1 <= self.workers <= self.max_workers:
            raise ValueError(f"workers must be between 1 and {self.max_workers}")
        if self.total_rss_bytes <= 0 or self.worker_rss_bytes <= 0:
            raise ValueError("memory limits must be positive")
        if any(not math.isfinite(value) or value <= 0
               for value in (self.fit_seconds, self.poll_seconds)):
            raise ValueError("time and poll limits must be positive")


@dataclasses.dataclass
class JobSpec:
    config: Mapping[str, Any]
    seed: int
    path: Path
    data: Mapping[str, Any]
    source_identity: str
    representation_key: Optional[str]

    @property
    def candidate_id(self) -> str:
        return str(self.config["candidate_id"])


@dataclasses.dataclass
class RunningJob:
    spec: JobSpec
    process: subprocess.Popen[Any]
    log_handle: Any
    started: float
    last_rss_bytes: int = 0
    termination: Optional[str] = None


class ExclusiveRunLock:
    """A process-held lock preventing two schedulers from sharing a run."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.handle: Optional[Any] = None

    def __enter__(self) -> "ExclusiveRunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise SchedulerError(f"another scheduler already owns {self.path}") from exc
        self.handle.write(f"pid={os.getpid()}\n")
        self.handle.flush()
        return self

    def __exit__(self, *_: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _relative_file_map(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for child in sorted(path.iterdir()):
        if not child.is_file() or child.name == "receipt.json":
            continue
        result[child.name] = file_sha(child)
    return result


def verify_completed_job(path: Path, config: Mapping[str, Any], seed: int,
                         registration: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """Return a verified result, or ``None`` when the attempt is absent.

    A result without a complete, matching receipt is intentionally not
    resumable.  The caller must preserve that partial directory and stop.
    """
    path = Path(path)
    if not path.exists():
        return None
    if not path.is_dir():
        raise SchedulerError(f"job path is not a directory: {path}")
    if not all((path / name).is_file() for name in ("job.json", "result.json", "receipt.json")):
        raise SchedulerError(f"incomplete immutable attempt requires recovery: {path}")
    try:
        job = _json(path / "job.json")
        result = _json(path / "result.json")
        receipt = _json(path / "receipt.json")
    except Exception as exc:
        raise SchedulerError(f"unreadable completed attempt: {path}") from exc
    files = receipt.get("files")
    if not isinstance(files, dict) or not MANDATORY_JOB_FILES.issubset(files):
        raise SchedulerError(f"receipt omits mandatory evidence: {path}")
    if receipt.get("scheduler_version") != SCHEDULER_VERSION:
        raise SchedulerError(f"receipt scheduler version mismatch: {path}")
    for name, expected in files.items():
        candidate = Path(name)
        if candidate.is_absolute() or ".." in candidate.parts or candidate.name != name:
            raise SchedulerError(f"receipt contains unsafe relative path: {path}/{name}")
        actual_path = path / name
        if not actual_path.is_file() or file_sha(actual_path) != expected:
            raise SchedulerError(f"receipt hash mismatch: {path}/{name}")
    data = registration["prepared"][config["arm_id"]]
    if (job.get("config") != config or job.get("seed") != seed or
            job.get("source_identity") != registration["source_identity"] or
            job.get("data_identity") != data["data_identity"] or
            job.get("data_sha256") != data["sha256"]):
        raise SchedulerError(f"job identity differs from registration: {path}")
    if job.get("parallel_scheduler_version") != SCHEDULER_VERSION:
        raise SchedulerError(f"job scheduler version mismatch: {path}")
    if result.get("candidate_id") != config["candidate_id"] or result.get("seed") != seed:
        raise SchedulerError(f"result identity differs from registration: {path}")
    if result.get("status") == "VALID":
        if (result.get("source_identity") != registration["source_identity"] or
                result.get("data_identity") != data["data_identity"] or
                not {"model.joblib", "predictions_S.npz"}.issubset(files) or
                receipt.get("returncode") != 0 or receipt.get("termination") is not None or
                not result.get("reload_verified")):
            raise SchedulerError(f"valid attempt lacks independent receipt evidence: {path}")
    return result


def _descendant_rss(root_pid: int) -> Optional[int]:
    """Read a process tree's RSS in bytes; ``ps`` reports KiB on supported Unix."""
    try:
        raw = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SchedulerError("cannot sample process RSS; refusing to continue") from exc
    if not raw.strip():
        raise SchedulerError("process RSS sampler returned no rows; refusing to continue")
    rows: dict[int, tuple[int, int]] = {}
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            pid, ppid, rss_kib = map(int, fields)
        except ValueError:
            continue
        rows[pid] = (ppid, rss_kib)
    if not rows:
        raise SchedulerError("process RSS sample contains no usable rows")
    if root_pid not in rows:
        return None
    if not rows or root_pid not in rows:
        raise SchedulerError("process RSS sampler returned malformed or stale rows; refusing to continue")
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _) in rows.items():
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(rows.get(pid, (0, 0))[1] for pid in descendants) * 1024


class ParallelBenchmarkScheduler:
    """Run registered seed jobs with bounded process concurrency."""

    def __init__(self, run: Path, *, runner: Path, source_script: Path,
                 limits: ResourceLimits = ResourceLimits(), methods: Optional[Iterable[str]] = None,
                 backbones: Optional[Iterable[str]] = None, max_jobs: Optional[int] = None,
                 python_executable: Optional[str] = None, policy_path: Optional[Path] = None):
        self.run_path = Path(run).resolve()
        self.runner = Path(runner).resolve()
        self.source_script = Path(source_script).resolve()
        self.root = self.source_script.parent.parent
        self.limits = limits
        self.methods = set(methods or ())
        self.backbones = set(backbones or ())
        self.max_jobs = max_jobs
        self.python_executable = python_executable or sys.executable
        self.policy_path = Path(policy_path).resolve() if policy_path else None
        self.registration_path = self.run_path / "registration.json"
        self.registration = _json(self.registration_path)
        self.jobs: list[JobSpec] = []
        self.resumed = 0
        self._running: dict[int, RunningJob] = {}
        self._representation_started: set[str] = set()
        self._representation_ready: set[str] = set()
        self._finalized_count = 0
        self._failed_count = 0
        self._events_path = self.run_path / "parallel_scheduler_events.jsonl"
        self._live_status_path = self.run_path / "parallel_scheduler_live_status.json"
        self._session_id = uuid.uuid4().hex
        self._session_started = time.time()

    def _validate_sources(self) -> None:
        for relative, expected in self.registration.get("source_files", {}).items():
            source = self.root / relative
            if not source.is_file() or file_sha(source) != expected:
                raise SchedulerError(f"registered source changed: {relative}")

    def _validate_parallel_policy(self) -> None:
        registered = self.registration.get("resources", {})
        registered_fit = float(registered.get("fit_seconds", self.limits.fit_seconds))
        registered_worker_rss = int(registered.get("worker_rss_bytes", self.limits.worker_rss_bytes))
        if self.limits.fit_seconds != registered_fit:
            raise SchedulerError("fit_seconds must equal the registered per-worker budget")
        if self.limits.worker_rss_bytes != registered_worker_rss:
            raise SchedulerError("worker RSS limit must equal the registered per-worker budget")
        if self.limits.workers == 1:
            return
        if self.policy_path is None or not self.policy_path.is_file():
            raise SchedulerError(
                "workers > 1 requires an explicit supervisor parallel policy JSON "
                "(--parallel-policy); registration.concurrent_fits remains 1"
            )
        try:
            policy = _json(self.policy_path)
        except Exception as exc:
            raise SchedulerError(f"unreadable parallel policy: {self.policy_path}") from exc
        expected_registration = file_sha(self.registration_path)
        if policy.get("status") != "SUPERVISOR_AUTHORIZED_PARALLEL_DEVELOPMENT":
            raise SchedulerError("parallel policy is not supervisor-authorized")
        if policy.get("registration_sha256") != expected_registration:
            raise SchedulerError("parallel policy is bound to a different registration")
        if int(policy.get("max_workers", 0)) < self.limits.workers:
            raise SchedulerError("parallel policy permits fewer workers than requested")
        if int(policy.get("max_total_rss_bytes", 0)) < self.limits.total_rss_bytes:
            raise SchedulerError("parallel policy permits less memory than requested")
        if float(policy.get("fit_seconds", 0)) != self.limits.fit_seconds:
            raise SchedulerError("parallel policy changes the registered fit time")
        if int(policy.get("worker_rss_bytes", 0)) != self.limits.worker_rss_bytes:
            raise SchedulerError("parallel policy changes the registered worker RSS limit")

    def _write_manifest(self) -> None:
        payload = {
            "scheduler_version": SCHEDULER_VERSION,
            "registration_sha256": file_sha(self.registration_path),
            "runner_sha256": file_sha(self.runner),
            "scheduler_module_sha256": file_sha(Path(__file__).resolve()),
            "source_script_sha256": file_sha(self.source_script),
            "parallel_policy_sha256": file_sha(self.policy_path) if self.policy_path else None,
            "workers": self.limits.workers,
            "total_rss_bytes": self.limits.total_rss_bytes,
            "worker_rss_bytes": self.limits.worker_rss_bytes,
            "fit_seconds": self.limits.fit_seconds,
            "thread_environment": {name: "1" for name in THREAD_ENVIRONMENT},
        }
        target = self.run_path / "parallel_scheduler_manifest.json"
        if target.exists():
            existing = _json(target)
            immutable = ("scheduler_version", "registration_sha256", "runner_sha256",
                         "scheduler_module_sha256", "source_script_sha256")
            if any(existing.get(name) != payload[name] for name in immutable):
                raise SchedulerError("parallel scheduler manifest identity mismatch")
        elif not write_json_once(target, payload):
            raise SchedulerError("parallel scheduler manifest creation raced")

    def discover(self) -> list[JobSpec]:
        self._validate_sources()
        self._validate_parallel_policy()
        self._write_manifest()
        jobs: list[JobSpec] = []
        for config in self.registration["candidates"]:
            if config.get("status") != "REGISTERED":
                continue
            if self.methods and config.get("method") not in self.methods:
                continue
            if self.backbones and config.get("backbone") not in self.backbones:
                continue
            data = self.registration["prepared"][config["arm_id"]]
            for seed in config["seeds"]:
                path = self.run_path / "jobs" / f'{config["candidate_id"]}_s{seed}'
                existing = verify_completed_job(path, config, seed, self.registration)
                if existing is not None:
                    self.resumed += 1
                    continue
                key = _representation_key(config, seed, data["data_identity"])
                jobs.append(JobSpec(config=config, seed=seed, path=path, data=data,
                                   source_identity=self.registration["source_identity"],
                                   representation_key=key))
        jobs.sort(key=lambda item: (item.representation_key or "", item.candidate_id, item.seed))
        if self.max_jobs is not None:
            if self.max_jobs < 0:
                raise ValueError("max_jobs must be non-negative")
            jobs = jobs[:self.max_jobs]
        self.jobs = jobs
        return jobs

    def _job_payload(self, spec: JobSpec) -> dict[str, Any]:
        payload = {
            "config": spec.config,
            "seed": spec.seed,
            "data_path": str(Path(spec.data["path"]).resolve()) if "path" in spec.data else str(
                (self.run_path / "prepared" / f'{spec.config["arm_id"]}.joblib').resolve()),
            "data_identity": spec.data["data_identity"],
            "data_sha256": spec.data["sha256"],
            "cache_path": str((self.run_path / "representation_cache").resolve()),
            "source_identity": spec.source_identity,
            "parallel_scheduler_version": SCHEDULER_VERSION,
            "representation_key": spec.representation_key,
        }
        return payload

    def _environment(self, spec: JobSpec) -> dict[str, str]:
        env = dict(os.environ)
        env.update({name: "1" for name in THREAD_ENVIRONMENT})
        env["PYTHONHASHSEED"] = "0"
        env["CUDA_VISIBLE_DEVICES"] = ""
        if spec.config.get("method") == "FRAPPE_EO":
            env["TF_CPP_MIN_LOG_LEVEL"] = "2"
            env["PYTHONPATH"] = os.pathsep.join([
                str(self.root / "src"),
                str(self.root / ".venv311/lib/python3.11/site-packages"),
            ])
        return env

    def _spawn(self, spec: JobSpec) -> RunningJob:
        if spec.path.exists():
            raise SchedulerError(f"attempt path appeared while scheduling: {spec.path}")
        spec.path.mkdir(parents=True)
        write_json_once(spec.path / "job.json", self._job_payload(spec))
        log_handle = (spec.path / "worker.log").open("x")
        executable = self.python_executable
        if spec.config.get("method") == "FRAPPE_EO":
            executable = str(self.root / "artifacts/nhis/benchmark_dependencies_20260916/frappe_env/.venv/bin/python")
        process = subprocess.Popen(
            [executable, "-B", str(self.runner), "worker", "--job", str(spec.path / "job.json")],
            cwd=self.root, env=self._environment(spec), stdout=log_handle,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        return RunningJob(spec=spec, process=process, log_handle=log_handle, started=time.monotonic())

    @staticmethod
    def _terminate(state: RunningJob, reason: str) -> None:
        state.termination = reason
        try:
            os.killpg(state.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            state.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(state.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            state.process.wait()

    def _seal_failed_cache(self, spec: JobSpec, reason: Optional[str]) -> None:
        key = spec.representation_key
        if not key:
            return
        status = self.run_path / "representation_cache" / f"{key}.json"
        if status.exists():
            return
        model = self.run_path / "representation_cache" / f"{key}.joblib"
        payload: dict[str, Any] = {
            "status": reason or "WORKER_FAILED",
            "key": key,
            "source_job": spec.path.name,
            "termination": "parallel scheduler sealed an incomplete representation attempt",
        }
        if model.exists():
            payload["orphan_model_sha256"] = file_sha(model)
        write_json_once(status, payload)

    def _fallback_result(self, state: RunningJob) -> dict[str, Any]:
        reason = state.termination or "WORKER_FAILED"
        return {
            "candidate_id": state.spec.candidate_id,
            "seed": state.spec.seed,
            "status": reason,
            "returncode": state.process.returncode,
            "source_identity": state.spec.source_identity,
            "data_identity": state.spec.data["data_identity"],
            "elapsed_seconds": time.monotonic() - state.started,
            "scheduler_generated": True,
        }

    def _record_event(self, state: RunningJob, result: Mapping[str, Any]) -> None:
        event = {
            "timestamp": time.time(),
            "session_id": self._session_id,
            "event": "job_finalized",
            "job": state.spec.path.name,
            "candidate_id": state.spec.candidate_id,
            "seed": state.spec.seed,
            "status": result.get("status"),
            "termination": state.termination,
            "elapsed_seconds": time.monotonic() - state.started,
            "observed_peak_rss_bytes": state.last_rss_bytes,
            "host_cpu_count": os.cpu_count(),
            "effective_cpu_count": _effective_cpu_count(),
            "workers": self.limits.workers,
            "methods": sorted(self.methods),
            "backbones": sorted(self.backbones),
            "parallel_policy_sha256": file_sha(self.policy_path) if self.policy_path else None,
            "scheduled_count": len(self.jobs),
            "finalized_count": self._finalized_count,
            "failed_count": self._failed_count,
        }
        with self._events_path.open("a") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        live = {
            "status": "RUNNING",
            "scheduler_version": SCHEDULER_VERSION,
            "run": str(self.run_path),
            "scheduled_count": len(self.jobs),
            "finalized_count": self._finalized_count,
            "failed_count": self._failed_count,
            "active_count": len(self._running),
            "host_cpu_count": os.cpu_count(),
            "effective_cpu_count": _effective_cpu_count(),
            "workers": self.limits.workers,
            "methods": sorted(self.methods),
            "backbones": sorted(self.backbones),
            "session_started": self._session_started,
            "parallel_policy_sha256": file_sha(self.policy_path) if self.policy_path else None,
            "last_job": event,
        }
        replace_json(self._live_status_path, live)

    def _finalize(self, state: RunningJob) -> dict[str, Any]:
        state.log_handle.close()
        result_path = state.spec.path / "result.json"
        if not result_path.exists():
            write_json_once(result_path, self._fallback_result(state))
        try:
            result = _json(result_path)
        except Exception as exc:
            raise SchedulerError(f"worker result is unreadable: {result_path}") from exc
        if state.termination or result.get("status") not in {"VALID", "NOT_SUPPORTED"}:
            self._seal_failed_cache(state.spec, state.termination or result.get("status"))
        receipt = {
            "scheduler_version": SCHEDULER_VERSION,
            "session_id": self._session_id,
            "returncode": state.process.returncode,
            "termination": state.termination,
            "observed_peak_rss_bytes": state.last_rss_bytes,
            "files": _relative_file_map(state.spec.path),
        }
        if not write_json_once(state.spec.path / "receipt.json", receipt):
            raise SchedulerError(f"receipt unexpectedly already exists: {state.spec.path}")
        verified = verify_completed_job(state.spec.path, state.spec.config, state.spec.seed, self.registration)
        if verified is None:
            raise SchedulerError(f"newly sealed attempt cannot be verified: {state.spec.path}")
        self._finalized_count += 1
        if verified.get("status") not in {"VALID", "NOT_SUPPORTED"}:
            self._failed_count += 1
        self._record_event(state, verified)
        return verified

    def _launchable_index(self, pending: list[JobSpec], slots: int) -> Optional[int]:
        if slots <= 0:
            return None
        for index, spec in enumerate(pending):
            key = spec.representation_key
            if key is None or key in self._representation_ready:
                return index
            if key not in self._representation_started:
                return index
        return None

    def run(self) -> dict[str, int]:
        started = 0
        with ExclusiveRunLock(self.run_path / "parallel_scheduler.lock"):
            self.discover()
            pending = list(self.jobs)
            replace_json(self._live_status_path, {
                "status": "RUNNING", "session_id": self._session_id,
                "session_started": self._session_started, "workers": self.limits.workers,
                "scheduled_count": len(pending), "resumed_count": self.resumed,
                "finalized_count": 0, "failed_count": 0,
                "methods": sorted(self.methods), "backbones": sorted(self.backbones),
                "effective_cpu_count": _effective_cpu_count(),
            })
            try:
                while pending or self._running:
                    while len(self._running) < self.limits.workers:
                        index = self._launchable_index(pending, self.limits.workers - len(self._running))
                        if index is None:
                            break
                        spec = pending.pop(index)
                        state = self._spawn(spec)
                        self._running[state.process.pid] = state
                        if spec.representation_key:
                            self._representation_started.add(spec.representation_key)
                        started += 1
                    rss_total = 0
                    for pid, state in list(self._running.items()):
                        sample = _descendant_rss(pid)
                        if sample is None and state.process.poll() is None:
                            sample = _descendant_rss(pid)
                            if sample is None and state.process.poll() is None:
                                raise SchedulerError(f"live worker {pid} is missing from RSS samples")
                        state.last_rss_bytes = max(state.last_rss_bytes, sample or 0)
                        rss_total += state.last_rss_bytes
                        if state.process.poll() is None:
                            age = time.monotonic() - state.started
                            if state.last_rss_bytes > self.limits.worker_rss_bytes:
                                self._terminate(state, "MEMORY_LIMIT")
                            elif age > self.limits.fit_seconds:
                                self._terminate(state, "TIME_LIMIT")
                    if rss_total > self.limits.total_rss_bytes:
                        for state in self._running.values():
                            if state.process.poll() is None:
                                self._terminate(state, "TOTAL_MEMORY_LIMIT")
                    for pid, state in list(self._running.items()):
                        if state.process.poll() is None:
                            continue
                        self._finalize(state)
                        if state.spec.representation_key:
                            self._representation_ready.add(state.spec.representation_key)
                        del self._running[pid]
                    if pending or self._running:
                        time.sleep(self.limits.poll_seconds)
            except BaseException:
                for state in self._running.values():
                    if state.process.poll() is None:
                        self._terminate(state, "SCHEDULER_INTERRUPTED")
                    try:
                        state.log_handle.close()
                    except Exception:
                        pass
                self._running.clear()
                replace_json(self._live_status_path, {
                    "status": "INTERRUPTED", "scheduler_version": SCHEDULER_VERSION,
                    "session_id": self._session_id, "session_started": self._session_started,
                    "run": str(self.run_path), "scheduled_count": len(self.jobs),
                    "finalized_count": self._finalized_count, "failed_count": self._failed_count,
                    "active_count": 0, "host_cpu_count": os.cpu_count(),
                    "effective_cpu_count": _effective_cpu_count(), "workers": self.limits.workers,
                    "methods": sorted(self.methods), "backbones": sorted(self.backbones),
                    "parallel_policy_sha256": file_sha(self.policy_path) if self.policy_path else None,
                })
                raise
            replace_json(self._live_status_path, {
                "status": "COMPLETE", "scheduler_version": SCHEDULER_VERSION,
                "session_id": self._session_id, "session_started": self._session_started,
                "run": str(self.run_path), "scheduled_count": len(self.jobs),
                "finalized_count": self._finalized_count, "failed_count": self._failed_count,
                "active_count": 0, "host_cpu_count": os.cpu_count(),
                "effective_cpu_count": _effective_cpu_count(), "workers": self.limits.workers,
                "methods": sorted(self.methods), "backbones": sorted(self.backbones),
                "parallel_policy_sha256": file_sha(self.policy_path) if self.policy_path else None,
            })
        return {"scheduled": started, "completed": self.resumed + self._finalized_count,
                "resumed": self.resumed, "failed": self._failed_count}
