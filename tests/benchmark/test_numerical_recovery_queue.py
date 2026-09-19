"""Focused synthetic contracts for the numerical recovery queue."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.run_nhis_numerical_recovery_queue as queue
from nhis_fairbias.benchmark.experiment_registry import identity
from nhis_fairbias.benchmark.parallel_execution import ResourceLimits, _representation_key


GiB = 1024 ** 3


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, count: int = 4):
    original = tmp_path / "original"
    (original / "jobs").mkdir(parents=True)
    (original / "prepared").mkdir()
    data = original / "prepared" / "a.joblib"
    data.write_bytes(b"synthetic prepared data")
    config = {"candidate_id": "lfr", "arm_id": "a", "backbone": "LR",
              "method": "LFR_RECONSTRUCTED", "status": "REGISTERED",
              "seeds": list(range(count)), "params": {"k": 5, "Az": 0.1,
                                                         "Ax": 0.1, "Ay": 0.1,
                                                         "maxiter": 50, "maxfun": 50}}
    registration = {
        "source_files": {}, "source_identity": "synthetic-source",
        "resources": {"fit_seconds": 1800, "worker_rss_bytes": 4 * GiB},
        "candidates": [config],
        "prepared": {"a": {"path": str(data), "data_identity": "synthetic-data",
                             "sha256": _sha(data)}},
    }
    registration_path = original / "registration.json"
    registration_path.write_text(json.dumps(registration))
    failures = []
    for seed in range(count):
        name = f"lfr_s{seed}"
        job_dir = original / "jobs" / name
        job_dir.mkdir()
        representation_key = _representation_key(config, seed, "synthetic-data")
        job = {"config": config, "seed": seed,
               "data_path": str(data), "data_identity": "synthetic-data",
               "data_sha256": _sha(data), "source_identity": "synthetic-source",
               "representation_key": representation_key}
        job_path = job_dir / "job.json"
        result_path = job_dir / "result.json"
        job_path.write_text(json.dumps(job))
        result_path.write_text(json.dumps({"status": "BUDGET_EXHAUSTED",
                                            "candidate_id": "lfr", "seed": seed}))
        failures.append({"job_id": name, "job_path": str(job_path),
                         "job_sha256": _sha(job_path), "result_sha256": _sha(result_path),
                         "config": config, "seed": seed,
                         "representation_key": job["representation_key"],
                         "status": "BUDGET_EXHAUSTED"})
    failure_path = tmp_path / "failure_inventory.json"
    failure_path.write_text(json.dumps({"schema": "failure_recovery_inventory_v1",
                                        "registration_sha256": _sha(registration_path),
                                        "failures": failures}))
    worker = queue.ROOT / "scripts/run_nhis_numerical_recovery_worker.py"
    files = queue.build_runtime_source_manifest(queue.ROOT)
    plan = {"schema_version": queue.RECOVERY_PLAN_VERSION,
            "runtime_variant": queue.RECOVERY_VARIANT,
            "runtime_environment": dict(queue.RECOVERY_THREAD_ENVIRONMENT),
            "runtime_source_files": files,
            "worker_script": "scripts/run_nhis_numerical_recovery_worker.py"}
    plan["runtime_source_identity"] = queue.recovery_runtime_identity(files, plan["runtime_environment"])
    plan_path = tmp_path / "runtime_plan.json"
    plan_path.write_text(json.dumps(plan))
    plan["failure_manifest_path"] = str(failure_path)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({
        "status": "SUPERVISOR_AUTHORIZED_PARALLEL_DEVELOPMENT",
        "registration_sha256": _sha(registration_path), "max_workers": 28,
        "max_total_rss_bytes": 48 * GiB, "worker_rss_bytes": 4 * GiB,
        "fit_seconds": 1800,
    }))
    namespace = tmp_path / ("lfr_recovery_" + plan["runtime_source_identity"])
    scheduler = queue.NumericalRecoveryScheduler(
        namespace, original_run=original,
        selected=queue._failure_jobs(original, failure_path, registration, count),
        runtime_plan=plan, runtime_plan_path=plan_path,
        external_runs=(tmp_path / "frappe_old", tmp_path / "frappe_new"),
        worker=worker,
        source_script=queue.ROOT / "scripts/run_nhis_benchmark_parallel.py",
        limits=ResourceLimits(workers=28, max_workers=28, total_rss_bytes=48 * GiB),
        methods=["LFR_RECONSTRUCTED"], policy_path=policy_path)
    return scheduler, plan, namespace, failure_path


def test_recovery_discovery_is_exact_and_namespaced(tmp_path):
    scheduler, plan, namespace, _ = _fixture(tmp_path)
    jobs = scheduler.discover()
    assert [job.path.name for job in jobs] == ["lfr_s0", "lfr_s1", "lfr_s2", "lfr_s3"]
    payload = scheduler._job_payload(jobs[0])
    assert payload["runtime_source_identity"] == plan["runtime_source_identity"]
    assert payload["runtime_plan_sha256"] == _sha(scheduler.runtime_plan_path)
    assert payload["recovery_run_path"] == str(namespace)
    assert namespace.name.endswith(plan["runtime_source_identity"])
    assert Path(payload["cache_path"]) == namespace / "representation_cache"
    assert payload["representation_key"] == payload["original_representation_key"]
    assert not (namespace / "registration.json").exists()


def test_recovery_rejects_incomplete_registered_lfr_coverage(tmp_path):
    scheduler, _, _, failure_path = _fixture(tmp_path)
    manifest = json.loads(failure_path.read_text())
    manifest["failures"].pop()
    failure_path.write_text(json.dumps(manifest))
    with pytest.raises(queue.SchedulerError, match="exactly cover"):
        queue._failure_jobs(scheduler.original_run, failure_path,
                            scheduler.registration, expected_count=4)


def test_recovery_counts_external_workers_without_counting_self(monkeypatch, tmp_path):
    scheduler, _, _, _ = _fixture(tmp_path)
    scheduler.discover()
    scheduler._running = {123: object()}
    monkeypatch.setattr(queue, "cgroup_memory", lambda: (0, 60 * GiB))
    monkeypatch.setattr(queue, "external_worker_count", lambda runs: 27)
    assert scheduler._launchable_index(scheduler.jobs, 1) is None
    monkeypatch.setattr(queue, "external_worker_count", lambda runs: 26)
    assert scheduler._launchable_index(scheduler.jobs, 1) == 0


def test_recovery_stops_on_host_memory_pressure(monkeypatch, tmp_path):
    scheduler, _, _, _ = _fixture(tmp_path)
    scheduler.discover()
    scheduler._running = {}
    monkeypatch.setattr(queue, "external_worker_count", lambda runs: 0)
    monkeypatch.setattr(queue, "cgroup_memory", lambda: (49 * GiB, 60 * GiB))
    assert scheduler._launchable_index(scheduler.jobs, 1) is None


def test_recovery_external_runs_cannot_include_output_namespace(tmp_path):
    scheduler, _, namespace, _ = _fixture(tmp_path)
    with pytest.raises(queue.SchedulerError, match="cannot count itself"):
        queue.NumericalRecoveryScheduler(
            namespace, original_run=scheduler.original_run, selected=scheduler.selected,
            runtime_plan=scheduler.runtime_plan, runtime_plan_path=scheduler.runtime_plan_path,
            external_runs=(namespace,), worker=scheduler.worker_script,
            source_script=queue.ROOT / "scripts/run_nhis_benchmark_parallel.py",
            limits=ResourceLimits(workers=28, max_workers=28, total_rss_bytes=48 * GiB),
            methods=["LFR_RECONSTRUCTED"], policy_path=tmp_path / "policy.json")


def test_scheduler_seals_one_intentional_failure_and_finishes_other(tmp_path, monkeypatch):
    scheduler, _, namespace, _ = _fixture(tmp_path, count=2)
    stub = tmp_path / "stub_worker.py"
    stub.write_text(
        "import hashlib, json, pathlib, sys\n"
        "job_path = pathlib.Path(sys.argv[sys.argv.index('--job') + 1])\n"
        "job = json.loads(job_path.read_text())\n"
        "def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()\n"
        "status = 'FAILED' if job['seed'] == 0 else 'VALID'\n"
        "result = {'candidate_id': job['config']['candidate_id'], 'seed': job['seed'], 'status': status,\n"
        "          'source_identity': job['source_identity'], 'data_identity': job['data_identity'],\n"
        "          'reload_verified': status == 'VALID'}\n"
        "(job_path.parent / 'result.json').write_text(json.dumps(result))\n"
        "if status == 'VALID':\n"
        "    (job_path.parent / 'model.joblib').write_bytes(b'model')\n"
        "    (job_path.parent / 'predictions_S.npz').write_bytes(b'predictions')\n"
        "files = {name: sha(job_path.parent / name) for name in ('job.json', 'result.json', 'model.joblib', 'predictions_S.npz') if (job_path.parent / name).is_file()}\n"
        "receipt = {'schema_version': 'nhis_numerical_recovery_receipt_v1',\n"
        " 'runtime_variant': job['runtime_variant'], 'runtime_source_identity': job['runtime_source_identity'],\n"
        " 'registration_sha256': job['registration_sha256'], 'original_job_sha256': job['original_job_sha256'],\n"
        " 'original_result_sha256': job['original_result_sha256'], 'candidate_id': job['config']['candidate_id'],\n"
        " 'seed': job['seed'], 'status': status, 'worker_result_status': status,\n"
        " 'runtime_checks_passed': True, 'files': files}\n"
        "(job_path.parent / 'runtime_receipt.json').write_text(json.dumps(receipt))\n"
    )
    scheduler.worker_script = stub.resolve()
    scheduler.runner = stub.resolve()
    scheduler.limits = ResourceLimits(workers=2, max_workers=28, total_rss_bytes=48 * GiB,
                                      worker_rss_bytes=4 * GiB, fit_seconds=1800, poll_seconds=0.01)
    monkeypatch.setattr(queue, "cgroup_memory", lambda: (0, 60 * GiB))
    monkeypatch.setattr(queue, "external_worker_count", lambda runs: 0)
    summary = scheduler.run()
    assert summary["scheduled"] == 2
    assert summary["failed"] == 1
    assert summary["completed"] == 2
    assert (namespace / "jobs" / "lfr_s0" / "runtime_receipt.json").is_file()
    assert (namespace / "jobs" / "lfr_s1" / "result.json").is_file()
    failed_key = next(job.representation_key for job in scheduler.jobs if job.seed == 0)
    assert (namespace / "representation_cache" / f"{failed_key}.runtime.json").is_file()
