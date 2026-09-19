"""Unit tests for Benchmark V1 Method Adapters."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from nhis_fairbias.benchmark.adapters import (
    BaseMethodAdapter,
    ExponentiatedGradientAdapter,
    FairBiasAdapter,
    LFRAdapter,
    NotSupportedError,
    ReweighingAdapter,
    ThresholdOptimizerAdapter,
    UnmitigatedAdapter,
)


@pytest.fixture
def binary_data():
    X, y = make_classification(n_samples=200, n_features=8, random_state=42)
    # Sensitive attribute: binary 1 and 2
    A = np.random.default_rng(42).choice([1, 2], size=len(y))
    return X, y, A


@pytest.fixture
def multigroup_data():
    X, y = make_classification(n_samples=200, n_features=8, random_state=42)
    # Sensitive attribute: 7 groups (1..7)
    A = np.random.default_rng(42).choice(np.arange(1, 8), size=len(y))
    return X, y, A


def test_unmitigated_adapter(binary_data):
    X, y, A = binary_data
    adapter = UnmitigatedAdapter()
    adapter.fit(X, y, A)
    preds = adapter.predict(X)
    probs = adapter.predict_decision_proba(X)
    assert len(preds) == len(y)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_reweighing_adapter(binary_data, multigroup_data):
    # Binary
    X, y, A = binary_data
    adapter = ReweighingAdapter()
    adapter.fit(X, y, A)
    probs = adapter.predict_decision_proba(X)
    assert len(probs) == len(y)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # Multi-group support
    X_m, y_m, A_m = multigroup_data
    adapter_m = ReweighingAdapter()
    adapter_m.fit(X_m, y_m, A_m)
    probs_m = adapter_m.predict_decision_proba(X_m)
    assert len(probs_m) == len(y_m)


def test_lfr_adapter_binary(binary_data):
    X, y, A = binary_data
    adapter = LFRAdapter(k=3)
    adapter.fit(X, y, A)
    probs = adapter.predict_decision_proba(X, A=A)
    assert len(probs) == len(y)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_lfr_adapter_multigroup_not_supported(multigroup_data):
    X, y, A = multigroup_data
    adapter = LFRAdapter(k=3)
    with pytest.raises(NotSupportedError, match="Arm 002 is NOT_SUPPORTED"):
        adapter.fit(X, y, A)


def test_fairlearn_eg_dp_adapter(binary_data):
    X, y, A = binary_data
    adapter = ExponentiatedGradientAdapter(constraint_type="demographic_parity", eps=0.05, max_iter=10)
    adapter.fit(X, y, A)
    q = adapter.predict_decision_proba(X)
    assert len(q) == len(y)
    assert np.all((q >= 0.0) & (q <= 1.0))


def test_fairlearn_eg_eo_adapter(binary_data):
    X, y, A = binary_data
    adapter = ExponentiatedGradientAdapter(constraint_type="equalized_odds", eps=0.05, max_iter=10)
    adapter.fit(X, y, A)
    q = adapter.predict_decision_proba(X)
    assert len(q) == len(y)
    assert np.all((q >= 0.0) & (q <= 1.0))


def test_fairlearn_threshold_optimizer(binary_data):
    X, y, A = binary_data
    adapter = ThresholdOptimizerAdapter()
    adapter.fit_base(X[:100], y[:100])
    adapter.calibrate(X[100:], y[100:], A[100:])
    q = adapter.predict_decision_proba(X, A=A)
    assert len(q) == len(y)
    assert np.all((q >= 0.0) & (q <= 1.0))


def test_fairbias_adapter(binary_data, multigroup_data):
    X, y, A = binary_data
    adapter = FairBiasAdapter(max_iterations=2)
    with pytest.raises(TypeError, match="semantic|X_semantic"):
        adapter.fit(X, y, A)

    X_m, y_m, A_m = multigroup_data
    adapter_m = FairBiasAdapter(max_iterations=2)
    with pytest.raises(TypeError, match="semantic|X_semantic"):
        adapter_m.fit(X_m, y_m, A_m)


def test_fairbias_adapter_semantic_arms():
    import pandas as pd
    from nhis_fairbias.benchmark.data_contracts import ARM_SPECS

    # Create small synthetic semantic dataframe matching arm_003
    spec = ARM_SPECS["arm_003"]
    cols = list(spec["features"])
    N = 28
    rng = np.random.default_rng(13)
    df = pd.DataFrame({c: rng.integers(1, 3, size=N) for c in cols})
    df["agep_a"] = rng.uniform(20, 80, size=N)
    y = np.tile([0, 1], N // 2)
    A = np.repeat([1, 2], N // 2)

    adapter = FairBiasAdapter(arm_id="arm_003", epsilon_ratio=100, max_iterations=0)
    X_dummy = np.zeros((N, 10))
    adapter.fit(X_dummy, y, A, X_semantic=df)

    assert adapter.fitted_ is True
    assert adapter.converged_ is True
    assert adapter.changed_dict_ == {}  # legal high-epsilon F-only no-op
    assert adapter.fit_manifest_["source_partition"] == "F"

    probs = adapter.predict_decision_proba(X_dummy, X_semantic=df)
    assert len(probs) == N
    assert np.all((probs >= 0.0) & (probs <= 1.0))
