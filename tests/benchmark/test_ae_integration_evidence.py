"""Synthetic integration evidence for FairBias BM_AE/JOINT orchestration."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import fairbias.enhancement as enhancement_module
import nhis_fairbias.benchmark.adapters.adapter_fairbias_ae as ae_adapter_module
from nhis_fairbias.benchmark.adapters.adapter_fairbias_ae import FairBiasAEAdapter
from fairbias.evaluator import FairEvaluator


def _data():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x": rng.normal(size=16), "z": rng.normal(size=16)})
    y = (X["x"].to_numpy() > 0).astype(int)
    A = pd.DataFrame({"A": np.array([0, 1] * 8)})
    return X, y, A


def _fit(mode="BM_AE", **kwargs):
    X, y, A = _data()
    adapter = FairBiasAEAdapter(
        mode=mode,
        max_outer_iterations=1,
        max_utility_evaluations=20,
        max_geometry_evaluations=200,
        max_bm_steps=2,
        poly_exponents=(1 / 3, 3.0),
        random_state=1,
        **kwargs,
    )
    adapter.fit_development(X.iloc[:8], y[:8], A.iloc[:8], X.iloc[8:], y[8:], A.iloc[8:])
    return adapter, X.iloc[8:]


def _nonidentity_parts():
    n = 48
    idx = np.arange(n)
    X = pd.DataFrame(
        {
            "agep_a": np.linspace(20.0, 80.0, n),
            "pcnt18uptc": np.linspace(1.0, 12.0, n),
            "status": np.array(["a"] * 24 + ["b"] * 24),
            "region": np.tile(["x", "y", "z"], 16),
        }
    )
    y = np.array([0, 1] * 24)
    A = pd.DataFrame({"A": np.array([1, 2] * 24)})
    return X, y, A


def test_fresh_process_joblib_reload_prediction_batch_invariance(tmp_path):
    adapter, Xc = _fit()
    path = tmp_path / "adapter.joblib"
    x_path = tmp_path / "x.joblib"
    joblib.dump(adapter, path)
    joblib.dump(Xc, x_path)
    code = "import joblib,sys; print(joblib.load(sys.argv[1]).predict_decision_proba(joblib.load(sys.argv[2])).tolist())"
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    result = subprocess.check_output([sys.executable, "-c", code, str(path), str(x_path)], env=env, text=True)
    restored = np.asarray(ast.literal_eval(result.strip()), dtype=float)
    direct = adapter.predict_decision_proba(Xc)
    assert np.allclose(restored, direct)
    assert np.allclose(adapter.predict_decision_proba(Xc.iloc[:2]), direct[:2])


def test_budget_exception_restores_scoped_hooks():
    X, y, A = _data()
    original_utility = enhancement_module.evaluate_candidate_utility
    adapter = FairBiasAEAdapter(
        max_outer_iterations=1,
        max_utility_evaluations=1,
        max_geometry_evaluations=1,
        max_bm_steps=1,
        poly_exponents=(1 / 3,),
        random_state=1,
    )
    with pytest.raises(RuntimeError):
        adapter.fit_development(X.iloc[:8], y[:8], A.iloc[:8], X.iloc[8:], y[8:], A.iloc[8:])
    assert enhancement_module.evaluate_candidate_utility is original_utility
    assert adapter.termination_reason_ == "BUDGET_EXHAUSTED"


def test_actual_synthetic_path_records_identity_when_no_nonidentity_witness():
    adapter, _ = _fit()
    assert adapter.is_fitted_ is True
    assert adapter.provenance_["trace_count"] == len(adapter.candidate_traces_)
    # This tiny witness is intentionally recorded as identity if no BM/AE
    # commit occurred; it must not be promoted to a non-identity claim.
    assert adapter.provenance_["bm_commits"] == 0
    assert adapter.provenance_["ae_commits"] == 0
    assert adapter.changed_dict_ == {}


def test_actual_bm_nonidentity_witness_and_ae_result():
    X, y, A = _nonidentity_parts()
    adapter = FairBiasAEAdapter(
        mode="BM_AE",
        epsilon_ratio=0.25,
        max_outer_iterations=2,
        max_bm_steps=5,
        max_utility_evaluations=100,
        max_geometry_evaluations=1000,
        random_state=0,
    )
    adapter.fit_development(X.iloc[:24], y[:24], A.iloc[:24], X.iloc[24:], y[24:], A.iloc[24:])
    assert adapter.provenance_["bm_commits"] > 0
    assert adapter.changed_dict_ == {"agep_a": "dropped", "pcnt18uptc": "dropped"}
    # This actual fixture produced no AE commit; retain that fact explicitly.
    assert adapter.provenance_["ae_commits"] == 0


@pytest.mark.parametrize("mode", ["BM_AE", "JOINT"])
def test_runtime_phase_order_is_bm_then_ae_and_refresh(monkeypatch, mode):
    """Exercise the real orchestration entry point with controlled engines.

    The production wrapper now places the implementation in ``_fit_development``
    and audits it from ``fit_development``.  This test therefore records calls
    made by the actual entry point rather than inspecting source text.  The
    fake engines only make the phase transitions deterministic; the adapter's
    partition validation, scoped hooks, refresh path, final-state checks, and
    restoration still run unchanged.
    """
    events = []
    epsilon_calls = 0

    def fake_calculate_epsilon(self, *args, **kwargs):
        nonlocal epsilon_calls
        epsilon_calls += 1
        events.append("initial_geometry" if epsilon_calls == 1 else "refresh")
        # The first F geometry is infeasible; the first BM commit makes the
        # subsequent refresh feasible, allowing the AE phase to run.
        value = 1.0 if epsilon_calls == 1 else 0.0
        return {"0_1": {"x": value}}

    class FakeBM:
        def __init__(self, *args, **kwargs):
            self.max_search_candidates = 5
            self.step_traces = []
            self.non_convergence = {}
            self.calls = 0

        def mitigate_step(self, X, Y, O, nmi_org, changed_dict, current_epsilon, epsilon_threshold, iteration):
            self.calls += 1
            events.append("BM")
            if self.calls == 1:
                return X, {"x": "dropped"}, current_epsilon, "x"
            return X, dict(changed_dict), current_epsilon, None

    class FakeAE:
        def __init__(self, *args, **kwargs):
            self.audit_trail = []
            self.total_model_fits = 0

        def enhance_step(self, X, Y, changed_dict, O, epsilon_threshold, current_epsilon, iteration, partition):
            events.append("AE")
            return X, dict(changed_dict), None

    monkeypatch.setattr(FairEvaluator, "calculate_epsilon", fake_calculate_epsilon)
    monkeypatch.setattr(ae_adapter_module, "FairBiasMitigation", FakeBM)
    monkeypatch.setattr(ae_adapter_module, "FairAccuracyEnhancement", FakeAE)

    X, y, A = _data()
    adapter = FairBiasAEAdapter(
        mode=mode,
        epsilon_ratio=0.5,
        max_outer_iterations=2,
        max_utility_evaluations=20,
        max_geometry_evaluations=20,
        max_bm_steps=2,
        poly_exponents=(1 / 3,),
        random_state=1,
    )
    adapter.fit_development(X.iloc[:8], y[:8], A.iloc[:8], X.iloc[8:], y[8:], A.iloc[8:])

    assert events[0] == "initial_geometry"
    assert events.index("BM") < events.index("AE")
    assert events.index("BM") < events.index("refresh")
    assert adapter.is_fitted_ is True
    assert adapter.changed_dict_ == {"x": "dropped"}
    assert adapter.provenance_["bm_commits"] == 1
    assert adapter.provenance_["ae_commits"] == 0
