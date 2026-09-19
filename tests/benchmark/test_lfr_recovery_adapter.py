"""Synthetic recovery integration, budget/convergence and artifact reload."""

import joblib
import numpy as np
import pytest
import aif360.algorithms.preprocessing.lfr as upstream

from nhis_fairbias.benchmark.adapters.adapter_lfr import LFRAdapter
from nhis_fairbias.benchmark.adapters.adapter_lfr_recovery import LFRAnalyticRecoveryAdapter
from nhis_fairbias.benchmark.adapters.base import NotSupportedError


def _data():
    rng = np.random.RandomState(9)
    x = rng.normal(size=(48, 6))
    a = np.array([1] * 17 + [2] * 31)
    y = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(int)
    return x, y, a


def test_upstream_initialization_bounds_arguments_and_budget_pass_through(monkeypatch):
    x, y, a = _data()
    calls = []

    def optimizer(func, *args, **kwargs):
        x0 = kwargs["x0"].copy()
        evaluated = func(x0, *kwargs["args"])
        calls.append((kwargs, evaluated))
        loss = evaluated[0] if isinstance(evaluated, tuple) else evaluated
        return x0, loss, {"warnflag": 1, "funcalls": 1, "nit": 0, "task": "synthetic stop"}

    monkeypatch.setattr(upstream.optim, "fmin_l_bfgs_b", optimizer)
    legacy = LFRAdapter(k=3, Ax=0.03, Ay=0.9, Az=8.0, random_state=7, maxiter=12, maxfun=13)
    recovery = LFRAnalyticRecoveryAdapter(k=3, Ax=0.03, Ay=0.9, Az=8.0, random_state=7, maxiter=12, maxfun=13)
    legacy.fit(x, y, a)
    recovery.fit(x, y, a)
    old, new = calls
    assert old[0].keys() == new[0].keys()
    for key in old[0]:
        if key == "approx_grad":
            assert old[0][key] is True and new[0][key] is False
        elif key == "x0":
            np.testing.assert_array_equal(old[0][key], new[0][key])
        elif key == "args":
            for left, right in zip(old[0][key], new[0][key]):
                np.testing.assert_array_equal(left, right)
        else:
            assert old[0][key] == new[0][key]
    assert old[1] == new[1][0]
    assert not recovery.converged_
    assert recovery.optimization_result_["warnflag"] == 1
    assert recovery.name != legacy.name
    assert upstream.optim.fmin_l_bfgs_b is optimizer


def test_synthetic_convergence_uses_fewer_evaluations_and_reload_matches(tmp_path):
    x, y, a = _data()
    before_x, before_y, before_a = x.copy(), y.copy(), a.copy()
    options = dict(k=3, Ax=0.01, Ay=1.0, Az=0.5, random_state=7,
                   maxiter=200, maxfun=300)
    original_optimizer = upstream.optim.fmin_l_bfgs_b
    np.random.seed(551)
    before_rng = np.random.get_state()
    recovery = LFRAnalyticRecoveryAdapter(**options).fit(x, y, a)
    after_rng = np.random.get_state()
    for left, right in zip(before_rng, after_rng):
        np.testing.assert_array_equal(left, right)
    assert upstream.optim.fmin_l_bfgs_b is original_optimizer
    assert recovery.converged_, recovery.optimization_result_
    report = recovery.optimization_result_
    assert report["objective"] < report["initial_objective"]
    assert report["funcalls"] == report["analytic_objective_evaluations"]
    assert report["funcalls"] <= options["maxfun"]
    legacy = LFRAdapter(**options).fit(x, y, a)
    assert not legacy.converged_, legacy.optimization_result_
    assert legacy.optimization_result_["funcalls"] > report["funcalls"]
    assert report["objective"] < legacy.optimization_result_["objective"]
    for actual, saved in [(x, before_x), (y, before_y), (a, before_a)]:
        np.testing.assert_array_equal(actual, saved)
    probability = recovery.predict_event_probability(x, A=a)
    assert probability.shape == (48,) and np.isfinite(probability).all()
    assert np.all((probability >= 0) & (probability <= 1))
    artifact = tmp_path / "synthetic_lfr_recovery.joblib"
    joblib.dump(recovery, artifact)
    restored = joblib.load(artifact)
    np.testing.assert_array_equal(restored.predict_event_probability(x, A=a), probability)
    np.testing.assert_array_equal(restored.lfr.learned_model, recovery.lfr.learned_model)
    assert restored.optimization_result_ == report
    repeated = LFRAnalyticRecoveryAdapter(**options).fit(x, y, a)
    np.testing.assert_array_equal(repeated.lfr.learned_model, recovery.lfr.learned_model)


def test_recovery_retains_sensitive_and_weight_contracts_and_restores_patch():
    x, y, a = _data()
    original_optimizer = upstream.optim.fmin_l_bfgs_b
    with pytest.raises(NotSupportedError, match="weighted"):
        LFRAnalyticRecoveryAdapter().fit(x, y, a, sample_weight=np.ones(len(y)))
    with pytest.raises(NotSupportedError, match="exactly 2"):
        LFRAnalyticRecoveryAdapter().fit(x, y, np.arange(len(y)) % 3)
    assert upstream.optim.fmin_l_bfgs_b is original_optimizer


def test_unverified_upstream_source_fails_before_fit(monkeypatch):
    import nhis_fairbias.benchmark.adapters.adapter_lfr_recovery as recovery_module

    monkeypatch.setattr(recovery_module, "_UPSTREAM_SOURCE_SHA256", {})
    x, y, a = _data()
    with pytest.raises(RuntimeError, match="upstream source differs"):
        LFRAnalyticRecoveryAdapter().fit(x, y, a)
