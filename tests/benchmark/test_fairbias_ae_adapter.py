import numpy as np
import pandas as pd

from nhis_fairbias.benchmark.adapters.adapter_fairbias_ae import FairBiasAEAdapter
from nhis_fairbias.benchmark.adapters.base import NotSupportedError


def _data(n=24):
    idx = np.arange(n)
    X = pd.DataFrame({"x": np.linspace(-1, 1, n), "cat": np.where(idx % 3 == 0, "a", "b")}, index=idx)
    y = (idx % 2).astype(int)
    A = pd.DataFrame({"A": idx % 2}, index=idx)
    return X, y, A


def test_bm_ae_uses_f_and_c_and_freezes_prediction_map():
    X, y, A = _data()
    adapter = FairBiasAEAdapter(mode="BM_AE", epsilon_ratio=10.0, max_outer_iterations=1, max_bm_steps=1)
    adapter.fit_development(X.iloc[:12], y[:12], A.iloc[:12], X.iloc[12:], y[12:], A.iloc[12:])
    p1 = adapter.predict_event_probability(X.iloc[12:])
    assert p1.shape == (12,)
    assert np.isfinite(p1).all() and ((p1 >= 0) & (p1 <= 1)).all()
    assert adapter.provenance_["fit_source"] == "F"
    assert adapter.provenance_["selection_source"] == "enhancement_eval"
    assert adapter.provenance_["label_origin"] == {"F": "F", "C": "C"}
    assert adapter.predict_event_probability(X.iloc[12:]).tolist() == p1.tolist()


def test_joint_supports_gbdt_and_preserves_original_labels():
    X, y, A = _data()
    y_before = y.copy()
    adapter = FairBiasAEAdapter(mode="JOINT", backbone="GBDT", max_outer_iterations=1, max_bm_steps=1)
    adapter.fit_development(X.iloc[:12], y[:12], A.iloc[:12], X.iloc[12:], y[12:], A.iloc[12:])
    assert np.array_equal(y, y_before)
    assert adapter.backbone == "GBDT"
    assert isinstance(adapter.changed_dict_, dict)


def test_development_api_rejects_single_class_c():
    X, y, A = _data()
    adapter = FairBiasAEAdapter(max_outer_iterations=1, max_bm_steps=1)
    bad = np.zeros(12, dtype=int)
    try:
        adapter.fit_development(X.iloc[:12], y[:12], A.iloc[:12], X.iloc[12:], bad, A.iloc[12:])
    except ValueError as exc:
        assert "both binary classes" in str(exc)
    else:
        raise AssertionError("single-class C must be rejected")


def test_direct_fit_is_rejected_to_prevent_f_as_c_leakage():
    X, y, A = _data()
    try:
        FairBiasAEAdapter().fit(X, y, A.to_numpy().ravel())
    except NotSupportedError:
        return
    raise AssertionError("direct fit must not reuse F as C")
