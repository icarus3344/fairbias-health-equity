"""Bounded synthetic checks for baseline adapter contracts."""

import numpy as np
import pytest

from nhis_fairbias.benchmark.adapters.adapter_lfr import LFRAdapter
from nhis_fairbias.benchmark.adapters.adapter_reductions import ExponentiatedGradientAdapter
from nhis_fairbias.benchmark.adapters.adapter_reweighing import ReweighingAdapter
from nhis_fairbias.benchmark.adapters.adapter_threshold_optimizer import ThresholdOptimizerAdapter
from nhis_fairbias.benchmark.adapters.adapter_unmitigated import UnmitigatedAdapter
from nhis_fairbias.benchmark.adapters.base import NotSupportedError
from nhis_fairbias.benchmark.adapters.estimators import make_estimator


def _data(n=16):
    x = np.arange(n * 2, dtype=float).reshape(n, 2)
    y = np.asarray([0, 1] * (n // 2))
    a = np.asarray([0] * (n // 2) + [1] * (n // 2))
    return x, y, a


def test_estimator_factory_registered_defaults_and_smoke_override():
    lr = make_estimator("LR")
    assert lr.get_params()["C"] == 1.0
    assert lr.get_params()["max_iter"] == 1000
    gbdt = make_estimator("GBDT")
    assert gbdt.get_params()["n_estimators"] == 100
    assert gbdt.get_params()["max_depth"] == 2
    assert gbdt.get_params()["learning_rate"] == 0.05
    assert gbdt.get_params()["subsample"] == 1.0
    smoke = make_estimator("GBDT", estimator_params={"n_estimators": 2, "max_depth": 2})
    assert smoke.get_params()["n_estimators"] == 2


def test_unmitigated_and_reweighing_expose_validated_positive_p():
    x, y, a = _data()
    baseline = UnmitigatedAdapter().fit(x, y, a)
    p = baseline.predict_event_probability(x)
    assert p.shape == (len(y),) and np.all((p >= 0) & (p <= 1))
    rw = ReweighingAdapter().fit(x, y, a)
    assert np.all(np.isfinite(rw.weights_fair_))
    assert rw.predict_decision_proba(x).shape == (len(y),)


def test_reweighing_rejects_zero_a_by_y_support():
    x, y, a = _data()
    y[:8] = 0
    with pytest.raises(NotSupportedError, match="A×Y"):
        ReweighingAdapter().fit(x, y, a)


def test_eg_keeps_fractional_mixture_q_and_separates_difference_bound():
    x, y, a = _data(24)
    adapter = ExponentiatedGradientAdapter(
        constraint_type="demographic_parity", difference_bound=0.2, max_iter=2
    ).fit(x, y, a)
    q = adapter.predict_decision_proba(x)
    expected = np.asarray(adapter.model._pmf_predict(x))[:, 1]
    assert np.allclose(q, expected)
    assert adapter.eps == pytest.approx(0.01)
    assert adapter.max_iter == 2
    assert adapter.difference_bound == pytest.approx(0.2)


def test_to_requires_explicit_two_stage_fit_and_exposes_base_p():
    x, y, a = _data(24)
    adapter = ThresholdOptimizerAdapter()
    with pytest.raises(NotSupportedError, match="two-stage"):
        adapter.fit(x, y, a)
    adapter.fit_base(x[:16], y[:16])
    adapter.calibrate(x[16:], y[16:], a[16:])
    p = adapter.predict_event_probability(x[16:])
    q = adapter.predict_decision_proba(x[16:], A=a[16:])
    assert p.shape == q.shape == (8,)
    assert np.all((p >= 0) & (p <= 1))
    assert np.all((q >= 0) & (q <= 1))


def test_lfr_rejects_more_than_two_groups_and_predict_transform_uses_placeholder_y():
    x, y, a = _data()
    with pytest.raises(NotSupportedError, match="exactly 2"):
        LFRAdapter().fit(x, y, np.asarray([0, 1, 2, 0] * 4))

    class FakeLFR:
        def __init__(self):
            self.labels_seen = None
            self.features_seen = None

        def transform(self, dataset):
            self.labels_seen = np.asarray(dataset.labels).copy()
            self.features_seen = np.asarray(dataset.features).copy()
            return dataset

    adapter = LFRAdapter()
    adapter.lfr = FakeLFR()
    adapter.privileged_val_, adapter.unprivileged_val_ = 1, 0
    transformed = adapter._transform_X(x[:4], a[:4])
    assert np.array_equal(adapter.lfr.labels_seen.ravel(), np.zeros(4))
    assert adapter.lfr.features_seen.shape == x[:4].shape
    assert transformed.shape == x[:4].shape
