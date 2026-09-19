import numpy as np
import pandas as pd
import pytest

from nhis_fairbias.benchmark.adapters.adapter_fairbias import FairBiasAdapter
from nhis_fairbias.benchmark.adapters import adapter_fairbias


def _semantic_data(n=24):
    return pd.DataFrame(
        {
            "agep_a": np.linspace(20.0, 80.0, n),
            "status": np.tile(["a", "b"], n // 2),
        }
    )


def test_fairbias_requires_semantic_fitting_data():
    adapter = FairBiasAdapter()
    with pytest.raises(TypeError, match="requires X_semantic DataFrame"):
        adapter.fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]), np.array([1, 2, 1, 2]))


def test_fairbias_gbdt_accepts_direct_semantic_nullable_dataframe():
    X_f = pd.DataFrame({
        "agep_a": pd.Series([20, None, 30, 40] * 6, dtype="Int64"),
        "status": pd.Series([1, 2, None, 1] * 6, dtype="Int64"),
    })
    y = np.array([0, 1, 0, 1] * 6)
    A = np.array([1, 2] * 12)
    adapter = FairBiasAdapter(backbone="GBDT", estimator_params={"n_estimators": 2, "max_depth": 1}, max_iterations=0, epsilon_ratio=100.0)
    adapter.fit(X_f, y, A)
    p = adapter.predict_decision_proba(X_f)
    assert p.shape == (len(X_f),)
    assert np.all((p >= 0) & (p <= 1))
    assert adapter.fit_manifest_["geometry_evaluations"] >= 1


def test_fairbias_invokes_f_only_mitigation_and_records_contract(monkeypatch):
    calls = []
    original = adapter_fairbias.FairBiasMitigation.mitigate_step

    def witness(self, *args, **kwargs):
        calls.append((kwargs["X"].copy(), kwargs["Y"].copy(), kwargs["O"].copy()))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(adapter_fairbias.FairBiasMitigation, "mitigate_step", witness)
    X_f = _semantic_data()
    y = np.array([0, 1] * 12)
    A = np.array([1] * 12 + [2] * 12)
    adapter = FairBiasAdapter(arm_id="arm_001", max_iterations=1, epsilon_ratio=1e-6)
    adapter.fit(np.zeros((len(X_f), 2)), y, A, X_semantic=X_f)

    assert calls, "FairBiasMitigation.mitigate_step was not invoked on F"
    assert calls[0][0].equals(X_f)
    assert adapter.fit_manifest_["source_partition"] == "F"
    assert adapter.fit_manifest_["input_fingerprint"]
    assert adapter.fit_manifest_["termination_reason"] in {"epsilon_reached", "budget_exhausted", "candidate_exhausted"}
    assert adapter.fit_manifest_["converged"] == (adapter.fit_manifest_["termination_reason"] == "epsilon_reached")


def test_fairbias_fitting_is_independent_of_future_partition_mutation():
    X_f = _semantic_data()
    y = np.array([0, 1] * 12)
    A = np.array([1] * 12 + [2] * 12)
    adapter = FairBiasAdapter(max_iterations=1, epsilon_ratio=100.0)
    adapter.fit(np.zeros((len(X_f), 2)), y, A, X_semantic=X_f)
    before = dict(adapter.fit_manifest_)

    X_future = X_f.copy()
    X_future.loc[:, "agep_a"] = X_future["agep_a"] + 1000.0
    X_future.loc[:, "status"] = "future-only"
    X_future.loc[0, "agep_a"] = np.nan
    X_future.loc[1, "status"] = np.nan
    probs = adapter.predict_decision_proba(np.zeros((len(X_future), 2)), X_semantic=X_future)

    assert len(probs) == len(X_future)
    assert adapter.fit_manifest_["input_fingerprint"] == before["input_fingerprint"]
    assert adapter.fit_manifest_["changed_dict"] == before["changed_dict"]


def test_fairbias_high_epsilon_ratio_allows_legal_noop():
    X_f = _semantic_data()
    y = np.array([0, 1] * 12)
    A = np.array([1] * 12 + [2] * 12)
    adapter = FairBiasAdapter(max_iterations=1, epsilon_ratio=100.0)
    adapter.fit(np.zeros((len(X_f), 2)), y, A, X_semantic=X_f)
    assert adapter.changed_dict_ == {}
    assert adapter.fit_manifest_["termination_reason"] == "epsilon_reached"
    assert adapter.fit_manifest_["converged"] is True


def test_fairbias_real_engine_can_produce_nonidentity_transform():
    n = 24
    X_f = pd.DataFrame({
        "agep_a": np.linspace(20.0, 80.0, n),
        "pcnt18uptc": np.linspace(1.0, 12.0, n),
        "status": np.array(["a"] * 12 + ["b"] * 12),
        "region": np.tile(["x", "y", "z"], 8),
    })
    y = np.array([0, 1] * 12)
    A = np.array([1] * 12 + [2] * 12)
    adapter = FairBiasAdapter(max_iterations=1, epsilon_ratio=0.95)
    adapter._learn_transform(adapter._prepare_fit_semantic(X_f), y, A)
    assert adapter.fit_manifest_["termination_reason"] in {"epsilon_reached", "candidate_exhausted", "budget_exhausted"}
    assert adapter.fit_manifest_["changed_dict"], adapter.fit_manifest_


def test_fairbias_budget_exhaustion_is_not_convergence(monkeypatch):
    def stop_without_progress(self, *args, **kwargs):
        return kwargs["X"].copy(), kwargs["changed_dict"], None, "agep_a"

    monkeypatch.setattr(adapter_fairbias.FairBiasMitigation, "mitigate_step", stop_without_progress)
    X_f = _semantic_data()
    y = np.array([0, 1] * 12)
    A = np.array([1] * 12 + [2] * 12)
    adapter = FairBiasAdapter(max_iterations=0, epsilon_ratio=1e-6)
    with pytest.raises(RuntimeError, match="BUDGET_EXHAUSTED"):
        adapter.fit(np.zeros((len(X_f), 2)), y, A, X_semantic=X_f)
    assert adapter.fit_manifest_["termination_reason"] == "budget_exhausted"
    assert adapter.fit_manifest_["converged"] is False
