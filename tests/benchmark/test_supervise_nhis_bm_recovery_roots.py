from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from scripts import supervise_nhis_bm_recovery_roots as roots


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    registered = tmp_path / "registered"
    (registered / "jobs").mkdir(parents=True)
    registration = {"source_files": {}, "source_identity": "source", "prepared": {}, "candidates": []}
    failures = []
    for index in range(8):
        config = {"candidate_id": f"cand{index}", "arm_id": "arm", "method": "FAIRBIAS_BM",
                  "status": "REGISTERED", "seeds": [0, 7], "backbone": "LR",
                  "params": {"max_geometry_evaluations": 20000, "max_iterations": 50}}
        registration["candidates"].append(config)
        for seed in (0, 7):
            name = f"cand{index}_s{seed}"
            job_dir = registered / "jobs" / name
            job_dir.mkdir()
            job = {"config": config, "seed": seed, "source_identity": "source",
                   "data_identity": "data", "data_sha256": "data-sha"}
            result = {"candidate_id": config["candidate_id"], "seed": seed, "status": "FAILED"}
            (job_dir / "job.json").write_text(json.dumps(job))
            (job_dir / "result.json").write_text(json.dumps(result))
            failures.append({"job_id": name, "job_path": str(job_dir / "job.json"),
                             "config": config, "seed": seed,
                             "job_sha256": _sha(job_dir / "job.json"),
                             "result_sha256": _sha(job_dir / "result.json"),
                             "representation_key": f"key-{index}", "status": "FAILED",
                             "cache_status": "NOT_ESTIMABLE: geometry"})
    registration_path = registered / "registration.json"
    registration_path.write_text(json.dumps(registration))
    manifest = tmp_path / "failure_inventory.json"
    manifest.write_text(json.dumps({"schema": "failure_recovery_inventory_v1",
                                    "registration_sha256": _sha(registration_path),
                                    "failures": failures}))
    source = tmp_path / "source"
    source.mkdir()
    return registered, registration_path, manifest, source


def test_waiting_source_gate_does_not_start_or_create_output(tmp_path, capsys):
    registered, _, manifest, source = _fixture(tmp_path)
    (source / "parallel_scheduler_live_status.json").write_text(json.dumps({
        "status": "RUNNING", "scheduled_count": 100, "finalized_count": 99,
        "failed_count": 0, "active_count": 1}))
    output = tmp_path / "bm_root_recovery"
    assert roots.main(["--source-run", str(source), "--registered-run", str(registered),
                       "--failure-manifest", str(manifest), "--output-root", str(output),
                       "--python", sys.executable]) == 0
    assert not output.exists()
    plan = json.loads(capsys.readouterr().out)
    assert plan["source_ready"] is False


def test_selects_eight_unique_roots_and_stable_minimum_job(tmp_path):
    registered, _, manifest, _ = _fixture(tmp_path)
    selected = roots.select_geometry_roots(registered, manifest)
    assert len(selected) == 8
    assert {entry["representation_key"] for entry in selected} == {f"key-{i}" for i in range(8)}
    assert {entry["job_id"] for entry in selected} == {f"cand{i}_s0" for i in range(8)}
    lanes = roots.partition_lanes(selected)
    assert [len(lane) for lane in lanes] == [4, 4]
    assert set(lanes[0][i]["representation_key"] for i in range(4)).isdisjoint(
        {item["representation_key"] for item in lanes[1]})


def test_input_hash_mutation_is_rejected(tmp_path):
    registered, _, manifest, _ = _fixture(tmp_path)
    job = registered / "jobs" / "cand0_s0" / "job.json"
    job.write_text(job.read_text() + "\n")
    with pytest.raises(roots.BMRecoveryGateError, match="evidence hash"):
        roots.select_geometry_roots(registered, manifest)
