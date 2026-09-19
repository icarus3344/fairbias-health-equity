"""Synthetic subprocess contracts for the bounded benchmark scheduler."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import textwrap

import pytest

from nhis_fairbias.benchmark.experiment_registry import identity
from nhis_fairbias.benchmark.parallel_execution import (
    ParallelBenchmarkScheduler,
    ResourceLimits,
    SchedulerError,
    file_sha,
)
import nhis_fairbias.benchmark.parallel_execution as parallel_execution


FAKE_WORKER = r'''
import json, pathlib, sys, time
job = pathlib.Path(sys.argv[sys.argv.index("--job") + 1])
payload = json.loads(job.read_text())
out = job.parent
key = payload.get("representation_key")
cache = pathlib.Path(payload["cache_path"])
if key:
    status = cache / (key + ".json")
    if not status.exists():
        marker = cache / (key + ".inflight")
        try:
            with marker.open("x") as handle:
                handle.write(out.name)
        except FileExistsError:
            (cache / "CONCURRENT_WRITE").write_text("1")
        time.sleep(0.15)
        model = cache / (key + ".joblib")
        model.write_bytes(b"synthetic representation")
        status.write_text(json.dumps({"status":"VALID", "key":key,
                                      "model_sha256":__import__("hashlib").sha256(model.read_bytes()).hexdigest()}))
        marker.unlink(missing_ok=True)
result = {"candidate_id": payload["config"]["candidate_id"], "seed": payload["seed"],
          "status":"VALID", "source_identity":payload["source_identity"],
          "data_identity":payload["data_identity"], "reload_verified":True}
(out / "model.joblib").write_bytes(b"synthetic model")
(out / "predictions_S.npz").write_bytes(b"synthetic predictions")
(out / "result.json").write_text(json.dumps(result))
'''


def _config(method="UNMITIGATED", backbone="LR", seed_count=1):
    config = {"arm_id": "arm_001", "backbone": backbone, "method": method,
              "training_weighted": False, "params": {}, "complexity": 1.0,
              "seeds": list(range(seed_count)), "status": "REGISTERED"}
    config["candidate_id"] = identity(config)[:20]
    return config


def _run(tmp_path: Path, configs):
    run = tmp_path / "run"
    (run / "jobs").mkdir(parents=True)
    (run / "prepared").mkdir()
    (run / "representation_cache").mkdir()
    data = run / "prepared" / "arm_001.joblib"
    data.write_bytes(b"synthetic prepared data")
    registration = {"source_files": {}, "source_identity": "synthetic-source",
                    "prepared": {"arm_001": {"data_identity": "synthetic-data",
                                                "sha256": file_sha(data)}},
                    "candidates": configs, "pending_predeclared_extensions": []}
    (run / "registration.json").write_text(json.dumps(registration, indent=2))
    return run


def _scheduler(run: Path, tmp_path: Path, *, workers=1, policy=None):
    fake = tmp_path / "fake_worker.py"
    fake.write_text("import sys\n" + textwrap.dedent(FAKE_WORKER))
    return ParallelBenchmarkScheduler(
        run, runner=fake, source_script=Path(__file__).parents[2] / "scripts/run_nhis_benchmark_parallel.py",
        limits=ResourceLimits(workers=workers, poll_seconds=0.02),
        policy_path=policy,
    )


def test_subprocess_run_seals_receipt_and_resume_requires_hashes(tmp_path):
    run = _run(tmp_path, [_config()])
    summary = _scheduler(run, tmp_path).run()
    assert summary["scheduled"] == 1
    assert summary["completed"] == 1
    job = next((run / "jobs").iterdir())
    receipt = json.loads((job / "receipt.json").read_text())
    assert {"job.json", "result.json", "worker.log", "model.joblib", "predictions_S.npz"} <= set(receipt["files"])
    resumed = _scheduler(run, tmp_path).run()
    assert resumed["scheduled"] == 0
    assert resumed["resumed"] == 1
    (job / "model.joblib").write_bytes(b"tampered")
    with pytest.raises(SchedulerError, match="receipt hash mismatch"):
        _scheduler(run, tmp_path).discover()


def test_shared_representation_leader_is_serialized_before_followers(tmp_path):
    configs = [_config("FAIRBIAS_BM", "LR"), _config("FAIRBIAS_BM", "GBDT")]
    run = _run(tmp_path, configs)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "status": "SUPERVISOR_AUTHORIZED_PARALLEL_DEVELOPMENT",
        "registration_sha256": file_sha(run / "registration.json"),
        "max_workers": 2, "max_total_rss_bytes": 80 * 1024**3,
        "fit_seconds": 1800, "worker_rss_bytes": 4 * 1024**3,
    }))
    summary = _scheduler(run, tmp_path, workers=2, policy=policy).run()
    assert summary["scheduled"] == 2
    assert not (run / "representation_cache" / "CONCURRENT_WRITE").exists()


def test_parallel_workers_require_bound_supervisor_policy(tmp_path):
    run = _run(tmp_path, [_config()])
    with pytest.raises(SchedulerError, match="requires an explicit supervisor parallel policy"):
        _scheduler(run, tmp_path, workers=2).discover()


def test_partial_attempt_is_fail_closed(tmp_path):
    config = _config()
    run = _run(tmp_path, [config])
    partial = run / "jobs" / f'{config["candidate_id"]}_s0'
    partial.mkdir()
    (partial / "result.json").write_text("{}")
    with pytest.raises(SchedulerError, match="incomplete immutable attempt"):
        _scheduler(run, tmp_path).discover()


def test_scheduler_version_and_rss_fail_closed(tmp_path, monkeypatch):
    run = _run(tmp_path, [_config()])
    _scheduler(run, tmp_path).run()
    job = next((run / "jobs").iterdir())
    receipt = json.loads((job / "receipt.json").read_text())
    receipt["scheduler_version"] = "stale"
    (job / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(SchedulerError, match="scheduler version mismatch"):
        _scheduler(run, tmp_path).discover()
    monkeypatch.setattr(parallel_execution.subprocess, "run",
                        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("ps unavailable")))
    with pytest.raises(SchedulerError, match="cannot sample process RSS"):
        parallel_execution._descendant_rss(1)


def test_cache_keys_match_production_worker_for_registered_matrix():
    from nhis_fairbias.benchmark.experiment_registry import enumerate_candidates
    from nhis_fairbias.benchmark.experiment_extensions import enumerate_extensions
    from nhis_fairbias.benchmark.experiment_worker import representation_key
    for config in enumerate_candidates() + enumerate_extensions():
        for seed in config['seeds']:
            assert parallel_execution._representation_key(config, seed, 'synthetic') == representation_key(config, seed, 'synthetic')


def test_cgroup_quota_overrides_host_affinity(monkeypatch):
    monkeypatch.setattr(parallel_execution.os, 'sched_getaffinity', lambda _: set(range(192)), raising=False)
    monkeypatch.setattr(Path, 'read_text', lambda self: '1600000 100000')
    assert parallel_execution._effective_cpu_count() == 16


def test_malformed_rss_is_not_treated_as_zero(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(parallel_execution.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='bad output'))
    with pytest.raises(SchedulerError, match='no usable rows'):
        parallel_execution._descendant_rss(123)


def test_run_lock_prevents_second_scheduler(tmp_path):
    from nhis_fairbias.benchmark.parallel_execution import ExclusiveRunLock
    run = _run(tmp_path, [_config()])
    with ExclusiveRunLock(run / 'parallel_scheduler.lock'):
        with pytest.raises(SchedulerError, match='another scheduler'):
            _scheduler(run, tmp_path).run()
    assert not (run / 'parallel_scheduler_manifest.json').exists()
