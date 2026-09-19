import importlib.util
import json
from pathlib import Path

import pytest

from nhis_fairbias.benchmark.experiment_worker import file_sha, write_json

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("study_completion", ROOT / "scripts/complete_nhis_study.py")
completion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(completion)


def _setup(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "jobs").mkdir()
    config = {"candidate_id": "missing_seed", "status": "REGISTERED", "seeds": [0], "params": {},
        "arm_id": "arm_001", "backbone": "LR", "method": "UNMITIGATED", "training_weighted": False}
    write_json(run / "registration.json", {"candidates": [config], "source_files": {}})
    admission = tmp_path / "admission.json"
    write_json(admission, {"status": "SUPERVISOR_ACCEPTED_FOR_COMPLETION",
        "registration_sha256": file_sha(run / "registration.json"), "files": {}})
    return run, admission


def test_actual_selection_failure_prevents_any_t_phase(tmp_path):
    run, admission = _setup(tmp_path)
    with pytest.raises(RuntimeError, match="selection"):
        completion.complete(run, admission, 0)
    events = [json.loads(line) for line in (run / "completion_events.jsonl").read_text().splitlines()]
    assert any(e["phase"] == "selection" and e["status"] == "FAILED" for e in events)
    assert not any(e["phase"] in {"freeze", "evaluation", "plots"} for e in events)
    assert not (run / "evaluation_T").exists()
    assert "registered seed jobs remain unexecuted" in (run / "completion_selection.log").read_text()


def test_changed_admission_evidence_is_rejected_before_process_or_t_io(tmp_path, monkeypatch):
    run, admission = _setup(tmp_path)
    evidence = tmp_path / "approved.py"
    evidence.write_text("original")
    record = json.loads(admission.read_text())
    record["files"] = {str(evidence): file_sha(evidence)}
    admission.write_text(json.dumps(record))
    evidence.write_text("changed")
    monkeypatch.setattr(completion.subprocess, "run", lambda *a, **k: pytest.fail("No process may start after evidence changes"))
    with pytest.raises(ValueError, match="implementation or evidence changed"):
        completion.complete(run, admission, 0)
    assert not (run / "completion_events.jsonl").exists()
