"""Production-to-report probes using only the synthetic frozen-run fixture."""
from __future__ import annotations

import json

import joblib
import numpy as np

from nhis_fairbias.benchmark.experiment_evaluation import evaluate_frozen_run, freeze_study
from nhis_fairbias.benchmark.experiment_worker import file_sha
from nhis_fairbias.benchmark.paper_reporting import write_paper_reporting
from nhis_fairbias.benchmark.predictions import PredictionBundle

from test_frozen_evaluation_contract import _annual, _fake_cohort_and_loader, _make_run


class _RisklessAdapter:
    def predict_event_probability(self, X, A):
        return np.full(len(X), 0.6, dtype=float)


class _RisklessPolicy:
    def __init__(self, n):
        self._adapter = _RisklessAdapter()
        self.n = n

    def predict(self, X, A):
        q = np.linspace(0.2, 0.8, self.n)
        return PredictionBundle(None, q)


def _rewrite_receipt_for_models(run):
    for model_path in (run / "jobs").glob("*/model.joblib"):
        job = joblib.load(model_path)
        job["policy"] = _RisklessPolicy(8)
        joblib.dump(job, model_path)
        receipt_path = model_path.parent / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["files"]["model.joblib"] = file_sha(model_path)
        receipt_path.write_text(json.dumps(receipt))
    selection_path = run / "selection_freeze.json"
    selection = json.loads(selection_path.read_text())
    selection["selected_artifacts"] = {
        rel: file_sha(run / rel) for rel in selection["selected_artifacts"]
    }
    selection_path.write_text(json.dumps(selection))


def test_real_evaluation_output_binds_manifest_and_untouched_base_to_paper_export(tmp_path, monkeypatch):
    run = _make_run(tmp_path)
    _rewrite_receipt_for_models(run)
    study = tmp_path / "study.json"
    freeze_study([run], study, repo_root=tmp_path)
    years = []
    _fake_cohort_and_loader(monkeypatch, _annual(), years)
    evaluation_dir = tmp_path / "evaluation"
    evaluate_frozen_run(run, evaluation_dir, study_freeze=study, repo_root=tmp_path, B=2000)
    summary = json.loads((evaluation_dir / "summary_T.json").read_text())
    manifest = json.loads((evaluation_dir / "evaluation_manifest.json").read_text())
    assert summary["evaluation_manifest_sha256"] == file_sha(evaluation_dir / "evaluation_manifest.json")
    assert years == [(2024,)]
    arm_rows = [row for row in summary["selections"] if row.get("arm_id") == "arm_001"]
    assert arm_rows and any("untouched_base_average_precision" in row for row in arm_rows)
    assert all(row['average_precision']['mean'] is None for row in arm_rows)
    assert all(row['untouched_base_average_precision']['mean'] is not None for row in arm_rows)
    individual = list(evaluation_dir.glob("*_individual_model_metrics.json"))
    jobs = list((run / "jobs").glob("*/result.json"))
    report = write_paper_reporting(
        run / "registration.json", run / "selection_freeze.json", evaluation_dir / "summary_T.json",
        individual, tmp_path / "paper", job_result_paths=jobs,
    )
    assert report["csv"].exists()
    assert "untouched_base_average_precision" in report["csv"].read_text()
    import csv
    with report['csv'].open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows and all(row['average_precision_p'] == '' for row in rows)
    assert all(row['untouched_base_average_precision'] != '' for row in rows)
