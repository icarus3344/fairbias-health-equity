"""Admission audit fixtures for the FairBias AE extension.

These tests exercise the repaired admission boundaries.
"""

import pickle
import numpy as np
import pandas as pd
import pytest

from fairbias.evaluator import FairEvaluator
from nhis_fairbias.benchmark.adapters.adapter_fairbias_ae import FairBiasAEAdapter


def _parts(n=20):
    idx = np.arange(n)
    X = pd.DataFrame({"x": idx.astype(float), "cat": np.where(idx % 2, "b", "a")}, index=idx)
    y = (idx % 2).astype(int)
    A = pd.DataFrame({"A": idx % 2}, index=idx)
    return X.iloc[: n // 2], y[: n // 2], A.iloc[: n // 2], X.iloc[n // 2 :], y[n // 2 :], A.iloc[n // 2 :]


def test_nan_initial_geometry_is_rejected_before_reference(monkeypatch):
    def bad_geometry(self, *args, **kwargs):
        return {"A": {"x": float("nan")}}

    monkeypatch.setattr(FairEvaluator, "calculate_epsilon", bad_geometry)
    Xf, yf, Af, Xc, yc, Ac = _parts()
    with pytest.raises(ValueError, match="non-finite|NOT_ESTIMABLE"):
        FairBiasAEAdapter(epsilon_ratio=10.0, max_outer_iterations=1, max_bm_steps=1).fit_development(
            Xf, yf, Af, Xc, yc, Ac
        )


def test_ae_commit_cap_is_explicit_budget_exhausted(monkeypatch):
    from fairbias.enhancement import FairAccuracyEnhancement
    monkeypatch.setattr(FairEvaluator, 'calculate_epsilon', lambda *args, **kw: {'A': {'x': 0., 'cat': 0.}})
    def commit(self, X, y, changed, *args, **kwargs):
        return X, {'cat': {'a': 'merged', 'b': 'merged'}}, 'cat'
    monkeypatch.setattr(FairAccuracyEnhancement, 'enhance_step', commit)
    Xf, yf, Af, Xc, yc, Ac = _parts()
    adapter = FairBiasAEAdapter(mode="BM_AE", epsilon_ratio=10.0, max_outer_iterations=1, max_bm_steps=1)
    with pytest.raises(RuntimeError, match='BUDGET_EXHAUSTED'):
        adapter.fit_development(Xf, yf, Af, Xc, yc, Ac)
    assert adapter.termination_reason_ == "BUDGET_EXHAUSTED"
    assert not adapter.is_fitted_


def test_fitted_adapter_pickle_reload_preserves_predictions():
    Xf, yf, Af, Xc, yc, Ac = _parts()
    adapter = FairBiasAEAdapter(mode="BM_AE", epsilon_ratio=10.0, max_outer_iterations=1, max_bm_steps=1)
    adapter.fit_development(Xf, yf, Af, Xc, yc, Ac)
    restored = pickle.loads(pickle.dumps(adapter))
    np.testing.assert_allclose(adapter.predict_event_probability(Xc), restored.predict_event_probability(Xc))


def test_bm_engine_registered_limits_and_policy():
    Xf, yf, Af, Xc, yc, Ac = _parts()
    adapter = FairBiasAEAdapter(mode="BM_AE", epsilon_ratio=10.0, max_outer_iterations=1, max_bm_steps=50)
    adapter.fit_development(Xf, yf, Af, Xc, yc, Ac)
    engine = adapter.bm_engine_
    assert engine.max_search_candidates == 5  # native categorical merge candidates, not BM commits
    assert adapter.provenance_['limits']['bm'] == 50
    assert engine.power_sequence_policy == "official_stream"
    assert engine.power_revisit_policy == "restart"
    assert engine.failed_attribute_mode == "stop"


def test_identical_f_c_record_ids_are_rejected():
    Xf, yf, Af, _, _, _ = _parts()
    with pytest.raises(ValueError, match="identical|overlap|record"):
        FairBiasAEAdapter(epsilon_ratio=10.0, max_outer_iterations=1, max_bm_steps=1).fit_development(
            Xf, yf, Af, Xf.copy(), yf.copy(), Af.copy(), metadata={"F_ids": list(Xf.index), "C_ids": list(Xf.index)}
        )
