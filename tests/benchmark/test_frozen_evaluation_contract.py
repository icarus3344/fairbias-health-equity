"""End-to-end frozen-evaluation checks using synthetic annual designs only."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from nhis_fairbias.benchmark import experiment_evaluation as evaluation
from nhis_fairbias.benchmark.data_contracts import AnnualSurveyDesign
from nhis_fairbias.benchmark.experiment_evaluation import evaluate_frozen_run, freeze_study
from nhis_fairbias.benchmark.experiment_selection import freeze_selection
from nhis_fairbias.benchmark.experiment_worker import file_sha, write_json
from nhis_fairbias.benchmark.predictions import PredictionBundle
from nhis_fairbias.benchmark.experiment_registry import identity


class _Policy:
    def __init__(self, q):
        self.q = np.asarray(q, dtype=float)
        self._adapter = object()

    def predict(self, X, A):
        assert len(X) == len(self.q)
        return PredictionBundle(self.q, self.q)


class _T(SimpleNamespace):
    def __len__(self):
        return len(self.y)


def _annual(*, missing_group=False, singleton=False):
    n = 8
    strata = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    psus = np.array([10, 10, 11, 11, 20, 20, 21, 21])
    if singleton:
        psus[4:] = 20
    A = np.array([1, 1, 2, 2, 1, 1, 2, 2])
    if missing_group:
        A[:] = 1
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    design = AnnualSurveyDesign(2024, np.arange(n), strata, psus, np.ones(n), np.ones(n, dtype=bool))
    T = _T(
        annual_design=design, X_semantic=pd.DataFrame({"x": np.arange(n)}), y=y, A=A,
        WTFA_A=np.ones(n), record_keys=np.arange(n),
        PSTRAT=strata, PPSU=psus, year=2024, role="evaluation_T", arm_id="arm_001", feature_names=("x",), metadata={},
    )
    return T


def _make_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "jobs").mkdir()
    data_file = run / "prepared.joblib"
    joblib.dump({"synthetic": True}, data_file)
    configs = []
    for method, q in (("FAIRBIAS_BM", [0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8]),
                      ("UNMITIGATED", [0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7])):
        config = {"arm_id": "arm_001", "backbone": "LR", "method": method,
                  "training_weighted": False, "params": {}, "complexity": 1.0,
                  "seeds": [0], "status": "REGISTERED"}
        config["candidate_id"] = identity(config)
        configs.append((config, q))
        job_dir = run / "jobs" / f"{config['candidate_id']}_s0"
        job_dir.mkdir()
        policy_path = job_dir / "model.joblib"
        joblib.dump({"semantic_input": True, "preprocessor": None, "policy": _Policy(q)}, policy_path)
        np.savez(job_dir / "predictions_S.npz", q=np.asarray(q), p=np.asarray(q))
        write_json(job_dir / "worker.log", {"synthetic": True})
        result = {"candidate_id": config["candidate_id"], "seed": 0, "status": "VALID",
                  "source_identity": "synthetic-source", "data_identity": "synthetic-data",
                  "metrics_S": {"balanced_accuracy": .7, "eo_gap": .1, "dp_gap": .1},
                  "risk_S": {"average_precision": .7}, "reload_verified": True}
        write_json(job_dir / "result.json", result)
        job = {"config": config, "seed": 0, "source_identity": "synthetic-source",
               "data_identity": "synthetic-data", "data_sha256": file_sha(data_file)}
        write_json(job_dir / "job.json", job)
        receipt = {"returncode": 0, "termination": None, "files": {
            name: file_sha(job_dir / name) for name in ("job.json", "result.json", "worker.log")}}
        receipt["files"]["model.joblib"] = file_sha(policy_path)
        receipt["files"]["predictions_S.npz"] = file_sha(job_dir / "predictions_S.npz")
        write_json(job_dir / "receipt.json", receipt)
    registration = {"candidates": [c for c, _ in configs], "pending_predeclared_extensions": [],
                    "source_files": {}, "source_identity": "synthetic-source",
                    "prepared": {"arm_001": {"data_identity": "synthetic-data", "sha256": file_sha(data_file)}}}
    write_json(run / "registration.json", registration)
    freeze_selection(run)
    return run


def _fake_cohort_and_loader(monkeypatch, T, years_seen):
    cohort = SimpleNamespace(attrs={"source_provenance": {"synthetic": True}})
    monkeypatch.setattr(evaluation, "load_local_nhis_cohort", lambda root, years: (years_seen.append(tuple(years)) or cohort))
    monkeypatch.setattr(evaluation, "summarize_eligibility", lambda cohort: {"synthetic_loader_fixture": True})
    monkeypatch.setattr(evaluation, "load_arm_partitions", lambda cohort, arm: {"evaluation_T": T} if arm == "arm_001" else {"evaluation_T": _T(annual_design=AnnualSurveyDesign(2024, np.array([0]), np.array([1]), np.array([1]), np.array([1.]), np.array([True])), X_semantic=pd.DataFrame({"x": [0]}), y=np.array([0]), A=np.array([1]), WTFA_A=np.array([1.]))})


def test_frozen_evaluation_synthetic_paired_ba_and_year_boundary(tmp_path, monkeypatch):
    run = _make_run(tmp_path)
    study = tmp_path / "study.json"
    freeze_study([run], study, repo_root=tmp_path)
    years_seen = []
    _fake_cohort_and_loader(monkeypatch, _annual(), years_seen)
    out = tmp_path / "out"
    rows = evaluate_frozen_run(run, out, study_freeze=study, repo_root=tmp_path, B=2000)
    assert years_seen == [(2024,)]
    pairs = list(json.loads((out / "summary_T.json").read_text())["paired_contrasts"])
    assert pairs
    assert pairs[0]["delta_balanced_accuracy"]["status"] == "VALID"
    assert pairs[0]["delta_balanced_accuracy"]["estimate"] == pytest.approx(.1)
    assert pairs[0]["delta_balanced_accuracy"]["se"] == pytest.approx(0.)
    assert pairs[0]["delta_eo_gap"]["projection_95"]["status"] in {"VALID", "NOT_ESTIMABLE"}


def test_frozen_evaluation_rejects_missing_or_changed_freeze_before_cohort_load(tmp_path, monkeypatch):
    run = _make_run(tmp_path)
    study = tmp_path / "study.json"
    freeze_study([run], study, repo_root=tmp_path)
    years_seen = []
    monkeypatch.setattr(evaluation, "load_local_nhis_cohort", lambda *args, **kwargs: years_seen.append(1))
    study.unlink()
    with pytest.raises(FileNotFoundError):
        evaluate_frozen_run(run, tmp_path / "missing", study_freeze=study, repo_root=tmp_path, B=2000)
    freeze_study([run], study, repo_root=tmp_path)
    model = next((run / "jobs").glob("*/model.joblib"))
    model.write_bytes(model.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Frozen model hash mismatch"):
        evaluate_frozen_run(run, tmp_path / "changed", study_freeze=study, repo_root=tmp_path, B=2000)
    assert years_seen == []


def test_singleton_and_missing_group_are_structured_non_estimable(tmp_path, monkeypatch):
    run = _make_run(tmp_path)
    study = tmp_path / "study.json"
    freeze_study([run], study, repo_root=tmp_path)
    for name, T, expected in (("singleton", _annual(singleton=True), "DESIGN_NOT_ESTIMABLE"),
                              ("missing", _annual(missing_group=True), "PARTIALLY_ESTIMABLE")):
        _fake_cohort_and_loader(monkeypatch, T, [])
        rows = evaluate_frozen_run(run, tmp_path / name, study_freeze=study, repo_root=tmp_path, B=2000)
        arm_rows = [r for r in rows if r.get("arm_id") == "arm_001"]
        assert arm_rows and all(r["evaluation_status"] == expected for r in arm_rows)
