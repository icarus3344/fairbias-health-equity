"""Synthetic readiness contracts, including artifact serialization."""
import json
import hashlib
import os
import subprocess
import sys
from types import SimpleNamespace
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest

from fairbias.evaluator import FairEvaluator
from fairbias.enhancement import FairAccuracyEnhancement
from nhis_fairbias.benchmark.adapters.adapter_fairbias_ae import FairBiasAEAdapter
from nhis_fairbias.benchmark.adapters.adapter_joint_ae_recovery import (
    JointAERecoveryAdapter, RecoveryAdmissionError, classify_recovery_fit,
    make_joint_ae_recovery_adapter,
    fit_joint_ae_recovery_fc,
)
from nhis_fairbias.benchmark.adapters import adapter_joint_ae_recovery as recovery_module


def data():
    training = pd.DataFrame({
        "cat": ["a"] * 12 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4
               + ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 12,
        "z": np.tile([0.0, 1.0], 24),
    })
    X = pd.concat([training, training], ignore_index=True)
    y = np.tile([0, 1], 48)
    A = pd.DataFrame({"A": ([0] * 24 + [1] * 24) * 2})
    return X.iloc[:48], y[:48], A.iloc[:48], X.iloc[48:], y[48:], A.iloc[48:]


@pytest.mark.parametrize("mode", ["JOINT", "BM_AE"])
def test_exact_strict_runtime_preserves_original_fit_and_serializes(tmp_path, mode):
    args = dict(mode=mode, epsilon_ratio=.75, max_geometry_evaluations=100, random_state=0)
    original, recovered = FairBiasAEAdapter(**args), JointAERecoveryAdapter(**args)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        original.fit_development(*data())
        recovered.fit_development(*data())
    assert recovered._delegate.changed_dict_ == original.changed_dict_ == {"cat": {"d": "a"}}
    assert recovered._delegate.candidate_traces_ == original.candidate_traces_
    assert recovered.recovery_result_["status"] == "COMPLETE_FEASIBLE"
    assert recovered.recovery_result_["formal_study_admitted"] is False
    assert recovered.predict_event_probability(data()[3]).tobytes() == original.predict_event_probability(data()[3]).tobytes()
    artifact = tmp_path / "synthetic_model.joblib"
    joblib.dump(recovered, artifact)
    loaded = joblib.load(artifact)
    assert loaded.predict_event_probability(data()[3]).tobytes() == recovered.predict_event_probability(data()[3]).tobytes()
    pair = tmp_path / "synthetic_reload_pair.joblib"
    joblib.dump((recovered, data()[3]), pair)
    expected = hashlib.sha256(recovered.predict_event_probability(data()[3]).tobytes()).hexdigest()
    child = subprocess.run([sys.executable, "-B", "-c",
        "import sys,joblib,hashlib; model,X=joblib.load(sys.argv[1]); "
        "assert hashlib.sha256(model.predict_event_probability(X).tobytes()).hexdigest()==sys.argv[2]",
        str(pair), expected], env=dict(os.environ), capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stderr
    json.dumps(recovered.provenance_, allow_nan=False)


def test_scheduled_budget_incumbent_is_not_complete_or_primary_valid(tmp_path):
    adapter = JointAERecoveryAdapter(mode="JOINT", controller="scheduled", epsilon_ratio=.75,
                                    max_geometry_evaluations=4, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        adapter.fit_development(*data())
    assert adapter.recovery_result_["status"] == "FEASIBLE_BUDGET_LIMITED"
    assert adapter.convergence_verified_ is False
    assert adapter._delegate.changed_dict_ == {"cat": {"d": "a"}}
    artifact = tmp_path / "scheduled.joblib"
    joblib.dump(adapter, artifact)
    assert joblib.load(artifact).predict(data()[3]).tobytes() == adapter.predict(data()[3]).tobytes()


def test_no_incumbent_budget_exit_never_exposes_predictions():
    adapter = JointAERecoveryAdapter(mode="JOINT", controller="scheduled", epsilon_ratio=.75,
                                    max_geometry_evaluations=1, random_state=0)
    with warnings.catch_warnings(), pytest.raises(RuntimeError):
        warnings.simplefilter("ignore", FutureWarning)
        adapter.fit_development(*data())
    assert not adapter.is_fitted_
    with pytest.raises(RuntimeError, match="not admitted"):
        adapter.predict(data()[3])


def test_strict_legacy_geometry_swallow_is_blocked_and_restored(monkeypatch):
    def broken_geometry(*args, **kwargs):
        raise ValueError("synthetic geometry failure")
    monkeypatch.setattr(FairEvaluator, "calculate_epsilon", broken_geometry)
    def legacy_swallow(self, *args):
        try:
            FairEvaluator.calculate_epsilon(None)
        except Exception:
            pass
        pytest.fail("legacy Exception handler must not swallow the transported error")
    monkeypatch.setattr(FairBiasAEAdapter, "fit_development", legacy_swallow)
    adapter = JointAERecoveryAdapter(mode="JOINT")
    with pytest.raises(ValueError, match="synthetic geometry failure"):
        adapter.fit_development(*data())
    assert FairEvaluator.calculate_epsilon is broken_geometry
    assert not adapter.is_fitted_
    assert adapter.provenance_["recovery"]["result"]["error_type"] == "ValueError"


@pytest.mark.parametrize("status", ["MDS_ITERATION_CAP", "NONFINITE_MDS"])
def test_strict_return_with_incomplete_mds_is_rejected(status):
    model = SimpleNamespace(is_fitted_=True, epsilon_threshold_=1.0,
        provenance_={"final_max_dphi": .5}, termination_reason_="STRICT_FEASIBLE_SEARCH_EXHAUSTED",
        mds_diagnostics_=[{"status": status}])
    with pytest.raises(RecoveryAdmissionError):
        classify_recovery_fit(model, controller="strict")


def test_scheduled_geometry_incomplete_retains_distinct_label():
    model = SimpleNamespace(is_fitted_=True, epsilon_threshold_=1.0, convergence_verified_=False,
        provenance_={"final_max_dphi": .5}, termination_reason_="FEASIBLE_GEOMETRY_INCOMPLETE",
        mds_diagnostics_=[{"status": "MDS_ITERATION_CAP"}])
    result = classify_recovery_fit(model, controller="scheduled")
    assert result["status"] == "FEASIBLE_GEOMETRY_INCOMPLETE" and not result["search_complete"]


def test_factory_preserves_config_and_rejects_silent_mode_or_weight_changes():
    config = {"method": "FAIRBIAS_BM_AE", "seeds": [0, 1], "backbone": "LR",
              "params": {"mode": "BM_AE", "epsilon_ratio": .75}}
    before = json.dumps(config, sort_keys=True)
    model = make_joint_ae_recovery_adapter(config, 0)
    assert model.mode == "BM_AE" and json.dumps(config, sort_keys=True) == before
    with pytest.raises(ValueError):
        make_joint_ae_recovery_adapter(config, 7)
    with pytest.raises(ValueError):
        make_joint_ae_recovery_adapter(config, 0, controller="scheduled")
    with pytest.raises(ValueError):
        make_joint_ae_recovery_adapter({**config, "training_weighted": True}, 0)


def test_unknown_component_fingerprint_fails_before_fit(monkeypatch):
    monkeypatch.setattr(recovery_module, "_COMPONENT_HASHES", {
        "src/nhis_fairbias/benchmark/joint_mds_numpy.py": "0" * 64})
    adapter = JointAERecoveryAdapter(mode="JOINT")
    with pytest.raises(RecoveryAdmissionError, match="source fingerprint"):
        adapter.fit_development(*data())
    assert not adapter.is_fitted_


def test_shadowed_component_origin_is_rejected(monkeypatch):
    import fairbias.evaluator as evaluator_module
    monkeypatch.setattr(evaluator_module, "__file__", "/tmp/shadowed/evaluator.py")
    adapter = JointAERecoveryAdapter(mode="JOINT")
    with pytest.raises(RecoveryAdmissionError, match="outside the pinned"):
        adapter.fit_development(*data())


def test_interruption_does_not_retry_and_restores_geometry(monkeypatch):
    original = FairEvaluator.calculate_epsilon
    def interrupted(self, *args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(FairBiasAEAdapter, "fit_development", interrupted)
    adapter = JointAERecoveryAdapter(mode="BM_AE", ae_cap_retry=40)
    with pytest.raises(KeyboardInterrupt):
        adapter.fit_development(*data())
    assert FairEvaluator.calculate_epsilon is original
    assert len(adapter.fit_attempts_) == 1 and not adapter.is_fitted_


def _synthetic_commits(monkeypatch, stop_after=None):
    # Keep geometry feasible but make each declared candidate distinct. This
    # exercises the original adapter's actual commit-limit check before step.
    calls = []
    def enhance(self, X, y, changed, *args, **kwargs):
        count = getattr(self, "synthetic_calls", 0) + 1
        self.synthetic_calls = count
        calls.append(count)
        if stop_after is not None and count > stop_after:
            return X, changed, None
        return X, {"cat": {"d": "a" if count % 2 else "b"}}, "cat"
    monkeypatch.setattr(FairAccuracyEnhancement, "enhance_step", enhance)
    return calls


def test_original_10_cap_attempt_preserved_then_fresh_40_reaches_exhaustion(monkeypatch):
    calls = _synthetic_commits(monkeypatch, stop_after=12)
    adapter = JointAERecoveryAdapter(mode="BM_AE", ae_cap_retry=40, epsilon_ratio=100, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        adapter.fit_development(*data())
    assert calls == list(range(1, 11)) + list(range(1, 14))
    assert [(x["ae_commit_cap"], x["status"], x["seed"]) for x in adapter.fit_attempts_] == [
        (10, "AE_COMMIT_CAP", 0), (40, "FIT_RETURNED", 0)]
    assert adapter._delegate.max_utility_evaluations == 500
    assert adapter._delegate.max_geometry_evaluations == 20000
    assert adapter._delegate.max_bm_steps == 50
    assert adapter.recovery_result_["status"] == "COMPLETE_FEASIBLE"
    assert adapter.runtime_spec["ae_commit_budget_changed"]
    assert not adapter.runtime_spec["search_order_changed"]
    assert adapter.runtime_spec["registered_search_limits"]["outer"] == 10
    json.dumps(adapter.provenance_, allow_nan=False)


def test_40_cap_is_incomplete_and_repeated_fit_restarts_at_registered_10(monkeypatch):
    calls = _synthetic_commits(monkeypatch)
    adapter = JointAERecoveryAdapter(mode="BM_AE", ae_cap_retry=40, epsilon_ratio=100)
    for _ in range(2):
        with warnings.catch_warnings(), pytest.raises(RuntimeError, match="AE commit limit"):
            warnings.simplefilter("ignore", FutureWarning)
            adapter.fit_development(*data())
        assert [x["ae_commit_cap"] for x in adapter.fit_attempts_] == [10, 40]
        assert adapter.fit_attempts_[-1]["status"] == "AE_COMMIT_CAP"
        assert adapter.recovery_result_["status"] == "NO_MODEL_BUDGET_EXHAUSTED"
        assert not adapter.is_fitted_
        with pytest.raises(RuntimeError, match="not admitted"):
            adapter.predict(data()[3])
    assert len(calls) == 100


@pytest.mark.parametrize("label", ["FairBias AE utility evaluations", "FairBias AE geometry evaluations", "BM commit limit"])
def test_other_budgets_do_not_enable_ae_retry(monkeypatch, label):
    def fails(self, *args):
        self.termination_reason_ = "BUDGET_EXHAUSTED"
        raise RuntimeError("BUDGET_EXHAUSTED: " + label)
    monkeypatch.setattr(FairBiasAEAdapter, "fit_development", fails)
    adapter = JointAERecoveryAdapter(mode="BM_AE", ae_cap_retry=40)
    with pytest.raises(RuntimeError):
        adapter.fit_development(*data())
    assert len(adapter.fit_attempts_) == 1
    assert not adapter.is_fitted_


@pytest.mark.parametrize("options", [{"mode": "JOINT"}, {"mode": "BM_AE", "max_outer_iterations": 11},
                                     {"mode": "BM_AE", "controller": "scheduled"}])
def test_ae_retry_cannot_silently_change_other_registered_modes(options):
    with pytest.raises(ValueError):
        JointAERecoveryAdapter(ae_cap_retry=40, **options)


def test_fc_api_rejects_s_t_and_overlap_before_fit():
    XF, yF, AF, XC, yC, AC = data()
    F = SimpleNamespace(role="fitting_F", year=2022, arm_id="synthetic", X_semantic=XF,
                        y=yF, A=AF, record_keys=np.arange(48))
    C = SimpleNamespace(role="calibration_C", year=2022, arm_id="synthetic", X_semantic=XC,
                        y=yC, A=AC, record_keys=np.arange(48, 96))
    config = {"method": "FAIRBIAS_BM_AE", "seeds": [0], "arm_id": "synthetic", "backbone": "LR",
              "params": {"mode": "BM_AE", "epsilon_ratio": .75}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        adapter = fit_joint_ae_recovery_fc(config, 0, F, C)
    assert adapter.recovery_result_["status"] == "COMPLETE_FEASIBLE"
    C.role = "selection_S"
    with pytest.raises(ValueError, match="F/C 2022"):
        fit_joint_ae_recovery_fc(config, 0, F, C)
    C.role = "calibration_C"
    C.record_keys[0] = F.record_keys[0]
    with pytest.raises(ValueError, match="overlap"):
        fit_joint_ae_recovery_fc(config, 0, F, C)
