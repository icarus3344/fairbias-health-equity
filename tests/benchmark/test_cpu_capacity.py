"""Synthetic checks for the additive CPU-capacity handoff wrapper."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from nhis_fairbias.benchmark.parallel_execution import ResourceLimits, SCHEDULER_VERSION
from scripts.run_nhis_cpu_capacity import ROOT, SHARD_AUDIT_VERSION, SHARD_PLAN_VERSION, ShardedScheduler


GiB = 1024 ** 3


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capacity_fixture(tmp_path: Path, *, seeds=(0, 1, 2)):
    run = tmp_path / "run"
    (run / "jobs").mkdir(parents=True)
    data = run / "prepared.joblib"
    data.write_bytes(b"synthetic prepared data")
    config = {"candidate_id": "frappe", "arm_id": "a", "backbone": "LR",
              "method": "FRAPPE_EO", "status": "REGISTERED", "seeds": list(seeds),
              "params": {}}
    registration = {"source_files": {}, "source_identity": "synthetic-source",
                    "candidates": [config],
                    "prepared": {"a": {"path": str(data), "data_identity": "d",
                                         "sha256": _sha(data)}}}
    registration_path = run / "registration.json"
    registration_path.write_text(json.dumps(registration))
    plan = {"schema_version": SHARD_PLAN_VERSION,
            "registration_sha256": _sha(registration_path),
            "source_identity": "synthetic-source",
            "hostnames": {"source": os.uname().nodename, "cpu": os.uname().nodename},
            "assignments": {"source": [], "cpu": [f"frappe_s{s}" for s in seeds]},
            "runtime_variant": "frappe_pipeline_deterministic_v1_20260917",
            "runtime_environment": {"TF_DETERMINISTIC_OPS": "1"}}
    from nhis_fairbias.benchmark.experiment_registry import identity

    runtime_files = {
        "scripts/run_nhis_runtime_worker.py": _sha(ROOT / "scripts/run_nhis_runtime_worker.py"),
        "src/nhis_fairbias/benchmark/adapters/adapter_frappe_pipeline.py": _sha(
            ROOT / "src/nhis_fairbias/benchmark/adapters/adapter_frappe_pipeline.py"),
        "src/nhis_fairbias/benchmark/adapters/adapter_frappe.py": _sha(
            ROOT / "src/nhis_fairbias/benchmark/adapters/adapter_frappe.py"),
    }
    plan["runtime_source_files"] = runtime_files
    plan["runtime_source_identity"] = identity({
        "variant": plan["runtime_variant"], "files": runtime_files,
        "environment": plan["runtime_environment"]})
    plan_path = run / "plan.json"
    plan_path.write_text(json.dumps(plan))
    activation = {"schema_version": SHARD_AUDIT_VERSION, "role": "cpu",
                  "plan_sha256": _sha(plan_path), "registration_sha256": _sha(registration_path),
                  "source_identity": "synthetic-source",
                  "source_drain_complete_sha256": "a" * 64,
                  "source_request_sha256": "b" * 64}
    activation_path = run / "activation.json"
    activation_path.write_text(json.dumps(activation))
    return run, registration_path, plan_path, activation_path


def _scheduler(tmp_path: Path, *, workers=30, total=54 * GiB):
    run, registration, plan, activation = _capacity_fixture(tmp_path)
    scheduler = ShardedScheduler(
        run, runner=ROOT / "scripts/run_nhis_runtime_worker.py",
        source_script=ROOT / "scripts/run_nhis_benchmark_parallel.py",
        limits=ResourceLimits(workers=workers, max_workers=workers, total_rss_bytes=total),
        plan_path=plan, activation_path=activation, role="cpu",
        policy_path=run / "parallel_policy.json")
    return scheduler


def test_capacity_admits_thirtieth_worker_at_observed_peak(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler._running = {
        pid: SimpleNamespace(last_rss_bytes=int(1.59 * GiB))
        for pid in range(29)
    }
    assert scheduler._launchable_index([SimpleNamespace(representation_key=None)], 1) == 0


def test_capacity_stops_on_aggregate_pressure(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler._running = {
        pid: SimpleNamespace(last_rss_bytes=int(1.59 * GiB))
        for pid in range(30)
    }
    assert scheduler._launchable_index([SimpleNamespace(representation_key=None)], 1) is None


def test_capacity_stops_on_host_pressure(monkeypatch, tmp_path):
    scheduler = _scheduler(tmp_path)

    class FakeSystemPath:
        def __init__(self, value):
            self.value = value

        def read_text(self):
            if self.value.endswith("memory.current"):
                return str(55 * GiB)
            return str(60 * GiB)

    real_path = Path

    def fake_path(value):
        if str(value) in {"/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"}:
            return FakeSystemPath(str(value))
        return real_path(value)

    import scripts.run_nhis_cpu_capacity as capacity

    monkeypatch.setattr(capacity, "Path", fake_path)
    scheduler._running = {}
    assert scheduler._launchable_index([SimpleNamespace(representation_key=None)], 1) is None


@pytest.mark.skipif(sys.platform != "linux", reason="handoff uses Linux process groups and kernel wait status")
def test_runtime_parent_handoff_preserves_exit_codes_and_pending_job(tmp_path):
    """Drain two live runtime workers, then resume the third registered job."""
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "src").symlink_to(ROOT / "src", target_is_directory=True)
    for name in ("run_nhis_cpu_capacity.py", "handoff_nhis_parallel.py",
                 "run_nhis_benchmark_parallel.py"):
        shutil.copy2(ROOT / "scripts" / name, project / "scripts" / name)
    executable = project / "artifacts/nhis/benchmark_dependencies_20260916/frappe_env/.venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    # The copied parent is deliberately named like the live CPU runtime shard.
    (project / "scripts/run_nhis_runtime_shard.py").write_text(
        '''import json, os, pathlib, subprocess, sys, time\n'''
        '''sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))\n'''
        '''from nhis_fairbias.benchmark.parallel_execution import ExclusiveRunLock, SCHEDULER_VERSION\n'''
        '''run = pathlib.Path(sys.argv[sys.argv.index("--run") + 1]).resolve()\n'''
        '''root = pathlib.Path(__file__).parents[1]\n'''
        '''reg = json.loads((run / "registration.json").read_text())\n'''
        '''plan = json.loads((run / "plan.json").read_text())\n'''
        '''data = reg["prepared"]["a"]\n'''
        '''config = reg["candidates"][0]\n'''
        '''with ExclusiveRunLock(run / "parallel_scheduler.lock"):\n'''
        '''    for seed in (0, 1):\n'''
        '''        path = run / "jobs" / (config["candidate_id"] + f"_s{seed}")\n'''
        '''        path.mkdir(parents=True)\n'''
        '''        payload = {"config": config, "seed": seed, "data_path": str((run / "prepared.joblib").resolve()),\n'''
        '''                   "data_identity": data["data_identity"], "data_sha256": data["sha256"],\n'''
        '''                   "cache_path": str((run / "representation_cache").resolve()),\n'''
        '''                   "source_identity": reg["source_identity"],\n'''
        '''                   "parallel_scheduler_version": SCHEDULER_VERSION, "representation_key": None,\n'''
        '''                   "runtime_variant": plan["runtime_variant"],\n'''
        '''                   "runtime_environment": plan["runtime_environment"],\n'''
        '''                   "runtime_plan_sha256": __import__("hashlib").sha256((run / "plan.json").read_bytes()).hexdigest(),\n'''
        '''                   "runtime_source_files": plan["runtime_source_files"],\n'''
        '''                   "runtime_source_identity": plan["runtime_source_identity"]}\n'''
        '''        (path / "job.json").write_text(json.dumps(payload))\n'''
        '''        log = (path / "worker.log").open("w")\n'''
        '''        subprocess.Popen([sys.executable, "-B", str(root / "scripts/run_nhis_runtime_worker.py"),\n'''
        '''                          "worker", "--job", str(path / "job.json")], cwd=root, stdout=log,\n'''
        '''                         stderr=subprocess.STDOUT, start_new_session=True)\n'''
        '''    while True:\n'''
        '''        time.sleep(0.05)\n''')
    # Runtime worker stub: keep the command identity and emit a real result/exit code.
    (project / "scripts/run_nhis_runtime_worker.py").write_text(
        '''import json, pathlib, sys, time, os\n'''
        '''p = pathlib.Path(sys.argv[sys.argv.index("--job") + 1]); j = json.loads(p.read_text())\n'''
        '''deadline=time.monotonic()+20\n'''
        '''while j["seed"] < 2 and pathlib.Path(f"/proc/{os.getppid()}/stat").read_text().rsplit(")",1)[1].split()[0] != "T":\n'''
        ''' if time.monotonic()>deadline: sys.exit(95)\n'''
        ''' time.sleep(.02)\n'''
        '''time.sleep(.2)\n'''
        '''(p.parent / "result.json").write_text(json.dumps({"status":"NOT_SUPPORTED", "candidate_id":j["config"]["candidate_id"], "seed":j["seed"]}))\n'''
        '''raise SystemExit(7 if j["seed"] == 0 else 0)\n''')

    run = project / "run"
    (run / "jobs").mkdir(parents=True)
    (run / "representation_cache").mkdir()
    data = run / "prepared.joblib"
    data.write_bytes(b"synthetic prepared data")
    config = {"candidate_id": "frappe", "arm_id": "a", "backbone": "LR",
              "method": "FRAPPE_EO", "status": "REGISTERED", "seeds": [0, 1, 2],
              "params": {}}
    registration = {"source_files": {}, "source_identity": "synthetic-source",
                    "candidates": [config],
                    "prepared": {"a": {"path": str(data), "data_identity": "d",
                                         "sha256": _sha(data)}}}
    (run / "registration.json").write_text(json.dumps(registration))
    plan = {"schema_version": SHARD_PLAN_VERSION,
            "registration_sha256": _sha(run / "registration.json"),
            "source_identity": "synthetic-source",
            "hostnames": {"source": os.uname().nodename, "cpu": os.uname().nodename},
            "assignments": {"source": [], "cpu": ["frappe_s0", "frappe_s1", "frappe_s2"]},
            "runtime_variant": "frappe_pipeline_deterministic_v1_20260917",
            "runtime_environment": {"TF_DETERMINISTIC_OPS": "1"}}
    from nhis_fairbias.benchmark.experiment_registry import identity

    runtime_files = {
        "scripts/run_nhis_runtime_worker.py": _sha(project / "scripts/run_nhis_runtime_worker.py"),
        "src/nhis_fairbias/benchmark/adapters/adapter_frappe_pipeline.py": _sha(
            project / "src/nhis_fairbias/benchmark/adapters/adapter_frappe_pipeline.py"),
        "src/nhis_fairbias/benchmark/adapters/adapter_frappe.py": _sha(
            project / "src/nhis_fairbias/benchmark/adapters/adapter_frappe.py"),
    }
    plan["runtime_source_files"] = runtime_files
    plan["runtime_source_identity"] = identity({
        "variant": plan["runtime_variant"], "files": runtime_files,
        "environment": plan["runtime_environment"]})
    plan_path = run / "plan.json"
    plan_path.write_text(json.dumps(plan))
    activation = {"schema_version": SHARD_AUDIT_VERSION, "role": "cpu",
                  "plan_sha256": _sha(plan_path), "registration_sha256": _sha(run / "registration.json"),
                  "source_identity": "synthetic-source", "source_drain_complete_sha256": "a" * 64,
                  "source_request_sha256": "b" * 64}
    activation_path = run / "activation.json"
    activation_path.write_text(json.dumps(activation))
    registration_hash = _sha(run / "registration.json")
    old_policy = {"status": "SUPERVISOR_AUTHORIZED_PARALLEL_DEVELOPMENT",
                  "registration_sha256": registration_hash, "max_workers": 24,
                  "max_total_rss_bytes": 54 * GiB, "worker_rss_bytes": 4 * GiB,
                  "fit_seconds": 1800}
    (run / "parallel_policy.json").write_text(json.dumps(old_policy))
    operations = project / "operations"
    operations.mkdir()
    new_policy = dict(old_policy, max_workers=30)
    new_policy_path = operations / "runtime_cpu_policy.json"
    new_policy_path.write_text(json.dumps(new_policy))
    # Bind the existing scheduler manifest to the original policy bytes.
    manifest = {"scheduler_version": SCHEDULER_VERSION,
                "registration_sha256": registration_hash,
                "runner_sha256": _sha(project / "scripts/run_nhis_runtime_worker.py"),
                "scheduler_module_sha256": _sha(ROOT / "src/nhis_fairbias/benchmark/parallel_execution.py"),
                "source_script_sha256": _sha(project / "scripts/run_nhis_benchmark_parallel.py"),
                "parallel_policy_sha256": _sha(run / "parallel_policy.json")}
    (run / "parallel_scheduler_manifest.json").write_text(json.dumps(manifest))

    parent = subprocess.Popen([
        sys.executable, "-B", str(project / "scripts/run_nhis_runtime_shard.py"),
        "--run", str(run), "--role", "cpu", "--workers", "2"], cwd=project,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        while len(list((run / "jobs").glob("*/job.json"))) < 2:
            if parent.poll() is not None or time.monotonic() > deadline:
                raise AssertionError("synthetic runtime parent did not launch two workers")
            time.sleep(0.02)
        done = subprocess.run([
            sys.executable, "-B", str(project / "scripts/run_nhis_cpu_capacity.py"),
            "--run", str(run), "--plan", str(plan_path), "--activation", str(activation_path),
            "--role", "cpu", "--workers", "30", "--policy", str(new_policy_path),
            "--methods", "FRAPPE_EO", "--old-pid", str(parent.pid)],
            cwd=project, capture_output=True, text=True, timeout=30)
        assert done.returncode == 0, done.stdout + done.stderr
        assert parent.wait(timeout=5) == -9
        job_dirs = sorted((run / "jobs").iterdir())
        assert {path.name for path in job_dirs} == {"frappe_s0", "frappe_s1", "frappe_s2"}
        assert all((path / "receipt.json").is_file() for path in job_dirs)
        receipts = {path.name: json.loads((path / "receipt.json").read_text()) for path in job_dirs}
        assert receipts["frappe_s0"]["returncode"] == 7
        assert receipts["frappe_s1"]["returncode"] == receipts["frappe_s2"]["returncode"] == 0
        assert json.loads((run / "parallel_scheduler_live_status.json").read_text())["status"] == "COMPLETE"
        audits = list((run / "operational_handoffs").glob("*_cpu_capacity"))
        assert len(audits) == 1
        kernel = list(audits[0].glob("kernel_exit_*.json"))
        assert len(kernel) == 2
        assert all(json.loads(path.read_text())["state"] == "Z" for path in kernel)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
