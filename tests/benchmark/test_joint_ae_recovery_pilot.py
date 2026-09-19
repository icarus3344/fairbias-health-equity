"""Generated-only wrapper checks; prepared S entries raise on access."""
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import signal
import warnings

import joblib
import numpy as np
import pytest

from fairbias.enhancement import FairAccuracyEnhancement
from nhis_fairbias.benchmark.adapters.adapter_joint_ae_recovery import JointAERecoveryAdapter
from test_numerical_recovery_pilot import case as numerical_case
from test_joint_ae_recovery import data as generated_fc


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def case(numerical_case, monkeypatch):
    case = numerical_case
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("joint_ae_pilot_tested", ROOT / "scripts/pilot_nhis_joint_ae_recovery.py")
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    case.pilot = pilot
    XF, yF, AF, XC, yC, AC = generated_fc()
    for role, old, X, y, A in (("fitting_F", case.F, XF, yF, AF),
                               ("calibration_C", case.C, XC, yC, AC)):
        partition = replace(old, X_semantic=X, y=y, A=A,
            record_keys=np.array([role + ":" + str(i) for i in range(len(y))]),
            WTFA_A=np.ones(len(y)), PSTRAT=np.ones(len(y), dtype=int), PPSU=np.arange(len(y)),
            feature_names=tuple(X.columns))
        case.data["partitions"][role] = partition
    case.job["config"].update(method="FAIRBIAS_JOINT", candidate_id="synthetic_joint",
                              params={"mode": "JOINT", "epsilon_ratio": .75})
    joblib.dump(case.data, case.prepared)
    digest = pilot.sha(case.prepared)
    case.job["data_sha256"] = digest
    case.registration["prepared"]["ARM_SYNTHETIC"]["sha256"] = digest
    case.write()
    return case


def test_real_joint_fc_fit_policy_reload_and_no_selection_access(case):
    original_registration = (case.run / "registration.json").read_bytes()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = case.pilot.run(case.job_file, case.output)
    saved = json.loads((case.output / "result.json").read_text())
    assert result["status"] == saved["status"] == "FC_PILOT_PASS"
    assert saved["recovery"]["result"]["status"] == "COMPLETE_FEASIBLE"
    assert saved["recovery"]["search_order_changed"] is False
    assert saved["S_T_evaluated"] is False and saved["formal_benchmark_admission"] is False
    assert saved["partitions_used"] == ["fitting_F", "calibration_C"]
    assert saved["reload_exact"] and saved["loaded_project_modules"]
    assert saved["data_identity"] == "generated-only"
    assert saved["registration_sha256"] == case.pilot.sha(case.run / "registration.json")
    assert set(saved["script_source_hashes"]) == {
        "scripts/pilot_nhis_joint_ae_recovery.py", "scripts/pilot_scheduled_joint.py",
        "scripts/pilot_nhis_numerical_recovery.py"}
    assert case.pilot.sha(case.output / "policy.joblib") == saved["model_sha256"]
    assert (case.run / "registration.json").read_bytes() == original_registration
    assert not (case.output / "result.json.part").exists()


def test_real_bmae_10_then_40_cap_persists_both_failed_attempts_without_policy(case, monkeypatch):
    config = case.job["config"]
    config.update(method="FAIRBIAS_BM_AE", candidate_id="synthetic_bmae",
                  params={"mode": "BM_AE", "epsilon_ratio": 100.0})
    case.write()
    def always_commits(self, X, y, changed, *args, **kwargs):
        count = getattr(self, "synthetic_calls", 0) + 1
        self.synthetic_calls = count
        return X, {"cat": {"d": "a" if count % 2 else "b"}}, "cat"
    monkeypatch.setattr(FairAccuracyEnhancement, "enhance_step", always_commits)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = case.pilot.run(case.job_file, case.output, ae_cap_retry=40)
    saved = json.loads((case.output / "result.json").read_text())
    assert result["status"] == saved["status"] == "FAILED"
    assert saved["error_type"] == "RuntimeError"
    assert saved["recovery"]["result"]["status"] == "NO_MODEL_BUDGET_EXHAUSTED"
    assert [(r["ae_commit_cap"], r["status"]) for r in saved["recovery"]["fit_attempts"]] == [
        (10, "AE_COMMIT_CAP"), (40, "AE_COMMIT_CAP")]
    assert not (case.output / "policy.joblib").exists()


def test_external_stop_keeps_failure_receipt_and_restores_handler(case, monkeypatch):
    previous = signal.getsignal(signal.SIGTERM)
    def stopped(self, *args, **kwargs):
        self.provenance_["recovery"]["synthetic_before_stop"] = True
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    monkeypatch.setattr(JointAERecoveryAdapter, "fit_development", stopped)
    result = case.pilot.run(case.job_file, case.output)
    saved = json.loads((case.output / "result.json").read_text())
    assert result["status"] == saved["status"] == "INTERRUPTED"
    assert saved["error_type"] == "ExternalPilotStop"
    assert saved["recovery"]["synthetic_before_stop"] is True
    assert signal.getsignal(signal.SIGTERM) is previous
    assert not (case.output / "policy.joblib").exists()


def test_existing_output_and_source_mismatch_rejected_before_prepared_load(case, monkeypatch):
    monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("Invalid admission loaded prepared data"))
    case.output.mkdir()
    sentinel = case.output / "result.json"
    sentinel.write_text("old result preserved")
    with pytest.raises(ValueError, match="fresh"):
        case.pilot.run(case.job_file, case.output)
    assert sentinel.read_text() == "old result preserved"
    source = next(iter(case.registration["source_files"]))
    case.registration["source_files"][source] = "0" * 64
    case.write()
    with pytest.raises(ValueError, match="source mismatch"):
        case.pilot.run(case.job_file, case.output.parent / "fresh_second")
