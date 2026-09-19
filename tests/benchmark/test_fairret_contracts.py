"""Synthetic contract tests for the matched MLP and pinned Fairret adapter."""

import numpy as np
import pytest

from nhis_fairbias.benchmark.adapters.adapter_fairret import (
    FairretAdapter,
    TorchMLPClassifier,
    UnmitigatedMLPAdapter,
)
from nhis_fairbias.benchmark.adapters.base import NotSupportedError
from nhis_fairbias.benchmark.adapters.estimators import make_estimator


def _data():
    rng = np.random.default_rng(20260916)
    X = rng.normal(size=(28, 5)).astype(np.float32)
    A = np.repeat(np.arange(7), 4)
    y = np.tile([0, 1, 0, 1], 7)
    return X, y, A


def test_shared_mlp_is_sklearnish_and_has_fixed_architecture():
    clf = TorchMLPClassifier(input_dim=5, epochs=2, random_state=3)
    assert clf.get_params()["hidden_layers"] == (64, 32)
    assert clf.get_params()["learning_rate"] == pytest.approx(0.001)
    assert clf.get_params()["epochs"] == 2
    assert clf.set_params(epochs=3) is clf
    assert clf.get_params()["epochs"] == 3


def test_fairret_eo_trains_full_f_and_returns_valid_p_independent_of_A():
    X, y, A = _data()
    adapter = FairretAdapter(epochs=3, random_state=11, fairness_coefficient=0.1, fairness_variant="EO")
    adapter.fit(X, y, A)
    p1 = adapter.predict_event_probability(X, A=np.zeros_like(A))
    p2 = adapter.predict_event_probability(X, A=np.arange(len(A)))
    assert np.allclose(p1, p2)
    assert p1.shape == (28,)
    assert np.all(np.isfinite(p1)) and np.all((p1 >= 0.0) & (p1 <= 1.0))
    assert adapter.classifier_.n_epochs_ == 3
    assert adapter.classifier_.finite_loss_count_ == 3
    assert adapter.classifier_.final_loss_ is not None
    assert adapter.loss_instance_ is not None


def test_nonzero_fairret_regularization_changes_parameters_and_lambda_zero_matches_plain():
    X, y, A = _data()
    plain = UnmitigatedMLPAdapter(epochs=3, random_state=5).fit(X, y, A)
    zero = FairretAdapter(epochs=3, random_state=5, fairness_coefficient=0.0).fit(X, y, A)
    fair = FairretAdapter(epochs=3, random_state=5, fairness_coefficient=0.1).fit(X, y, A)
    for lhs, rhs in zip(plain.classifier_.model_.parameters(), zero.classifier_.model_.parameters()):
        assert np.array_equal(lhs.detach().numpy(), rhs.detach().numpy())
    differences = [
        np.max(np.abs(lhs.detach().numpy() - rhs.detach().numpy()))
        for lhs, rhs in zip(zero.classifier_.model_.parameters(), fair.classifier_.model_.parameters())
    ]
    assert max(differences) > 0.0


def test_fairret_seed_reproducibility_and_group_support_contract():
    X, y, A = _data()
    first = FairretAdapter(epochs=2, random_state=19, fairness_coefficient=0.1).fit(X, y, A)
    second = FairretAdapter(epochs=2, random_state=19, fairness_coefficient=0.1).fit(X, y, A)
    assert np.array_equal(first.predict_event_probability(X), second.predict_event_probability(X))

    bad_y = y.copy()
    bad_y[A == 3] = 1
    with pytest.raises(NotSupportedError, match="support"):
        FairretAdapter(epochs=2, random_state=19, fairness_coefficient=0.1).fit(X, bad_y, A)


def test_dp_variant_and_survey_weights_are_explicit():
    X, y, A = _data()
    dp = FairretAdapter(
        epochs=2, random_state=7, fairness_coefficient=0.1, fairness_variant="DP"
    ).fit(X, y, A)
    assert dp.loss_instance_ is not None
    with pytest.raises(NotSupportedError, match="survey weights"):
        FairretAdapter(epochs=2).fit(X, y, A, sample_weight=np.ones(len(y)))


def test_default_fairret_is_regularized_and_plain_baseline_rejects_regularization():
    X, y, A = _data()
    fair = FairretAdapter(epochs=1).fit(X, y, A)
    assert fair.fairness_coefficient == pytest.approx(1.0)
    assert fair.loss_instance_ is not None
    with pytest.raises(ValueError, match="cannot use fairness"):
        UnmitigatedMLPAdapter(fairness_coefficient=0.1)


def test_failed_refit_invalidates_old_model_and_unknown_input_dim_refits_width():
    X, y, A = _data()
    clf = TorchMLPClassifier(epochs=1, random_state=3)
    clf.fit(X[:, :3], y)
    assert clf.get_params()["input_dim"] is None
    assert clf.n_features_in_ == 3
    with pytest.raises(ValueError, match="both classes"):
        clf.fit(X[:, :3], np.ones(len(y)))
    with pytest.raises(RuntimeError, match="fitted"):
        clf.predict(X[:, :3])
    clf.fit(X, y)
    assert clf.get_params()["input_dim"] is None
    assert clf.n_features_in_ == 5
    assert clf.predict_proba(X).shape == (28, 2)


def test_mlp_factory_is_lazy_and_sklearn_clone_keeps_constructor_params():
    from sklearn.base import clone

    estimator = make_estimator("MLP")
    before = estimator.get_params()
    cloned = clone(estimator)
    assert cloned.get_params() == before
    X, y, _ = _data()
    estimator.fit(X, y)
    assert estimator.get_params()["input_dim"] is None
    assert estimator.n_features_in_ == 5
