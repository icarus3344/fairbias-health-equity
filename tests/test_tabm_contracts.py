"""Bounded synthetic tests for the source-pinned official TabM adapter."""

import pickle

import numpy as np
import pytest

from nhis_fairbias.benchmark.adapters.adapter_tabm import TabMAdapter, TabMClassifier
from nhis_fairbias.benchmark.adapters.base import NotSupportedError


def _data(n=28):
    rng = np.random.default_rng(20260916)
    X = rng.normal(size=(n, 5)).astype(np.float32)
    y = np.asarray([0, 1] * (n // 2))
    return X, y


def test_tabm_is_official_pinned_model_with_sklearn_like_contract():
    clf = TabMClassifier(epochs=2, batch_size=7, random_state=3)
    assert clf.get_params()["k"] == 4
    assert clf.get_params()["d_block"] == 64
    assert clf.get_params()["input_dim"] is None
    X, y = _data()
    clf.fit(X, y)
    assert clf.classes_.tolist() == [0, 1]
    assert clf.n_features_in_ == 5
    assert clf.get_params()["input_dim"] is None
    assert clf.model_.k == 4


def test_tabm_outputs_batched_valid_risk_p_without_sensitive_attribute():
    X, y = _data()
    adapter = TabMAdapter(epochs=2, batch_size=6, random_state=7).fit(X, y)
    p = adapter.predict_event_probability(X, A=np.arange(len(y)))
    assert p.shape == (len(y),)
    assert np.all(np.isfinite(p))
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert adapter.predict_proba(X).shape == (len(y), 2)


def test_tabm_seed_reproducibility_and_pickle_reload():
    X, y = _data()
    first = TabMClassifier(epochs=2, batch_size=8, random_state=11).fit(X, y)
    second = TabMClassifier(epochs=2, batch_size=8, random_state=11).fit(X, y)
    assert np.array_equal(first.predict_proba(X), second.predict_proba(X))
    restored = pickle.loads(pickle.dumps(first))
    assert np.array_equal(first.predict_proba(X), restored.predict_proba(X))


def test_tabm_rejects_bad_training_inputs_and_failed_refit_invalidates_model():
    X, y = _data()
    clf = TabMClassifier(epochs=1).fit(X, y)
    with pytest.raises(ValueError, match="both classes"):
        clf.fit(X, np.ones(len(y)))
    with pytest.raises(RuntimeError, match="fitted"):
        clf.predict(X)
    with pytest.raises(NotSupportedError, match="sample_weight"):
        TabMClassifier(epochs=1).fit(X, y, sample_weight=np.ones(len(y)))
    with pytest.raises(ValueError, match="real"):
        TabMClassifier(epochs=1).fit(X.astype(complex), y)
    with pytest.raises(ValueError, match="finite"):
        TabMClassifier(epochs=1).fit(np.where(np.indices(X.shape)[0] == 0, np.inf, X), y)

