from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from nhis_fairbias.benchmark.adapters.adapter_reductions import ExponentiatedGradientAdapter
from nhis_fairbias.benchmark.experiment_worker import file_sha
from scripts.audit_nhis_eg_compatibility import audit_eg_archive


class _Policy:
    def __init__(self, adapter):
        self._adapter = adapter
        self.threshold = None


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _make_archive(tmp_path: Path, *, q_only=True):
    root = tmp_path / "repo"
    archive = tmp_path / "archive"
    jobdir = archive / "jobs" / "eg1_s0"
    root.mkdir()
    jobdir.mkdir(parents=True)
    source = root / "source_marker.py"
    source.write_text("SOURCE = 'frozen'\n")
    source_hash = file_sha(source)
    params = {"constraint_type": "equalized_odds", "eps": 0.01, "max_iter": 5,
              "C": 1.0, "difference_bound": None, "estimator_params": {}}
    config = {"candidate_id": "eg1", "method": "EG_EO", "arm_id": "arm001",
              "backbone": "LR", "training_weighted": False, "params": params,
              "seeds": [0], "status": "REGISTERED"}
    registration = {"source_identity": "synthetic-source", "source_files": {"source_marker.py": source_hash},
                    "prepared": {"arm001": {"data_identity": "synthetic-data", "sha256": "data-sha"}},
                    "candidates": [config]}
    registration_path = archive / "registration.json"
    _write_json(registration_path, registration)
    job = {"candidate_id": "eg1", "seed": 0, "config": config,
           "source_identity": "synthetic-source", "data_identity": "synthetic-data",
           "data_sha256": "data-sha"}
    _write_json(jobdir / "job.json", job)
    adapter = ExponentiatedGradientAdapter(**params, backbone="LR", random_state=0)
    adapter.model = SimpleNamespace(weights_=np.array([0.25, 0.75]))
    artifact = {"policy": _Policy(adapter), "config": config, "seed": 0,
                "data_identity": "synthetic-data"}
    joblib.dump(artifact, jobdir / "model.joblib")
    q = np.array([0.0, 0.25, 0.75, 1.0])
    if q_only:
        np.savez(jobdir / "predictions_S.npz", q=q)
    else:
        np.savez(jobdir / "predictions_S.npz", q=q, p_event=q)
    (jobdir / "worker.log").write_text("synthetic accepted worker\n")
    _write_json(jobdir / "result.json", {"candidate_id": "eg1", "seed": 0,
                "status": "VALID", "reload_verified": True,
                "source_identity": "synthetic-source", "data_identity": "synthetic-data",
                "output_type": "decision_probability_q", "threshold": None})
    required = ("job.json", "result.json", "receipt.json", "model.joblib", "predictions_S.npz", "worker.log")
    receipt = {"returncode": 0, "termination": None,
               "files": {name: file_sha(jobdir / name) for name in required if name != "receipt.json"}}
    _write_json(jobdir / "receipt.json", receipt)
    # The receipt hash manifest is intentionally finalized after receipt exists;
    # audit only requires hashes for the required non-receipt artifacts.
    return archive, root, registration_path


def test_synthetic_eg_archive_is_compatible_without_emitting_q(tmp_path):
    archive, root, registration_path = _make_archive(tmp_path)
    output = tmp_path / "fresh-report"
    report = audit_eg_archive(archive, root, file_sha(registration_path), 1, output)
    assert report["status"] == "COMPATIBLE_STORED_S_Q_ONLY"
    text = (output / "eg_compatibility_report.json").read_text()
    assert "0.25" not in text
    assert report["jobs"][0]["q_count"] == 4


def test_eg_audit_rejects_non_q_npz_and_does_not_create_output(tmp_path):
    archive, root, registration_path = _make_archive(tmp_path, q_only=False)
    with pytest.raises(ValueError, match="q only"):
        audit_eg_archive(archive, root, file_sha(registration_path), 1, tmp_path / "fresh")


def test_eg_audit_rejects_registration_source_drift(tmp_path):
    archive, root, registration_path = _make_archive(tmp_path)
    (root / "source_marker.py").write_text("SOURCE = 'changed'\n")
    with pytest.raises(ValueError, match="source hash mismatch"):
        audit_eg_archive(archive, root, file_sha(registration_path), 1, tmp_path / "fresh")


def test_mixed_archive_ignores_non_eg_and_records_terminal_eg(tmp_path):
    archive, root, registration_path = _make_archive(tmp_path)
    registration = json.loads(registration_path.read_text())
    failed = dict(registration["candidates"][0], candidate_id="eg-failed", seeds=[1])
    registration["candidates"].append(failed)
    registration["candidates"].append({"candidate_id": "other", "method": "UNMITIGATED", "seeds": [0], "status": "REGISTERED"})
    _write_json(registration_path, registration)
    failed_dir = archive / "jobs" / "eg-failed_s1"
    failed_dir.mkdir()
    config = failed
    _write_json(failed_dir / "job.json", {"candidate_id": "eg-failed", "seed": 1, "config": config,
        "source_identity": "synthetic-source", "data_identity": "synthetic-data", "data_sha256": "data-sha"})
    _write_json(failed_dir / "result.json", {"candidate_id": "eg-failed", "seed": 1,
        "status": "FAILED", "source_identity": "synthetic-source", "data_identity": "synthetic-data"})
    (failed_dir / "worker.log").write_text("synthetic failure\n")
    (failed_dir / "receipt.json").write_text("{}")
    receipt = {"returncode": 1, "termination": None,
               "files": {name: file_sha(failed_dir / name) for name in ("job.json", "result.json", "worker.log")}}
    _write_json(failed_dir / "receipt.json", receipt)
    other = archive / "jobs" / "other_s0"
    other.mkdir()
    _write_json(other / "job.json", {"candidate_id": "other", "seed": 0, "config": registration["candidates"][-1]})
    report = audit_eg_archive(archive, root, file_sha(registration_path), 1, tmp_path / "fresh")
    assert report["expected_eg_job_count"] == 2
    assert {row["status"] for row in report["jobs"]} == {"VALID", "FAILED"}
