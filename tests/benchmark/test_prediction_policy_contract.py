"""Synthetic contract tests for frozen p/q prediction policies."""

import numpy as np
import pytest

from nhis_fairbias.benchmark.predictions import (
    FrozenDecisionPolicy,
    PredictionBundle,
    PredictionContractError,
)


class StubAdapter:
    def __init__(self, values, output_type="event_probability_p", requires_a=False):
        self.values = np.asarray(values)
        self.output_type = output_type
        self.requires_sensitive_at_predict = requires_a
        self.name = "STUB"
        self.calls = 0

    def predict_decision_proba(self, X, A=None):
        self.calls += 1
        if self.requires_sensitive_at_predict and A is None:
            raise ValueError("A required")
        return self.values[: len(X)]


def test_counterexample_uses_hard_ba_not_probability_average():
    adapter = StubAdapter([0.9, 0.1])
    policy = FrozenDecisionPolicy().fit_calibration(
        adapter, np.zeros((2, 1)), np.array([1, 0])
    )
    bundle = policy.predict(np.zeros((2, 1)))
    assert policy.threshold == pytest.approx(0.5)
    assert np.array_equal(bundle.yhat, [1, 0])
    assert np.array_equal(bundle.q_decision, bundle.yhat)
    assert bundle.p_event.tolist() == [0.9, 0.1]


def test_fractional_q_is_preserved_and_cannot_be_event_risk():
    adapter = StubAdapter([0.25, 0.75], output_type="decision_probability_q", requires_a=True)
    policy = FrozenDecisionPolicy().fit_calibration(
        adapter, np.zeros((2, 1)), np.array([1, 0]), A_C=np.array([1, 2])
    )
    bundle = policy.predict(np.zeros((2, 1)), A=np.array([1, 2]))
    assert bundle.p_event is None
    assert np.allclose(bundle.q_decision, [0.25, 0.75])
    assert bundle.yhat is None
    with pytest.raises(PredictionContractError, match="p_event"):
        bundle.risk_probability()


def test_contract_rejects_shape_bounds_and_missing_calibration_support():
    with pytest.raises(PredictionContractError, match="both classes"):
        FrozenDecisionPolicy().fit_calibration(
            StubAdapter([0.1, 0.2]), np.zeros((2, 1)), np.array([1, 1])
        )
    with pytest.raises(PredictionContractError, match="one-dimensional"):
        FrozenDecisionPolicy().fit_calibration(
            StubAdapter([[0.1], [0.2]]), np.zeros((2, 1)), np.array([1, 0])
        )
    with pytest.raises(PredictionContractError, match=r"\[0, 1\]"):
        FrozenDecisionPolicy().fit_calibration(
            StubAdapter([1.1, 0.2]), np.zeros((2, 1)), np.array([1, 0])
        )
    with pytest.raises(PredictionContractError, match="A_C"):
        FrozenDecisionPolicy().fit_calibration(
            StubAdapter([0.1, 0.2], output_type="decision_probability_q", requires_a=True),
            np.zeros((2, 1)), np.array([1, 0]),
        )


def test_tied_probabilities_boundaries_and_threshold_freeze():
    # With all scores tied, BA ties at t=0 and t=1.  The prescribed tie rule
    # chooses the larger threshold, yielding all-negative under >=.
    adapter = StubAdapter([0.5, 0.5, 0.5, 0.5])
    policy = FrozenDecisionPolicy().fit_calibration(
        adapter, np.zeros((4, 1)), np.array([1, 0, 1, 0])
    )
    assert policy.threshold == pytest.approx(1.0)
    assert np.array_equal(policy.predict(np.zeros((4, 1))).yhat, [0, 0, 0, 0])

    # C fixes the threshold: later labels are not read and cannot change it.
    adapter.values = np.asarray([0.9, 0.1, 0.9, 0.1])
    assert policy.threshold == pytest.approx(1.0)
    assert policy.predict(np.zeros((2, 1))).p_event.tolist() == [0.9, 0.1]

    # Exact endpoints retain the >= rule and still admit an all-negative
    # boundary above 1 when p contains 1.
    endpoint = StubAdapter([0.0, 1.0])
    endpoint_policy = FrozenDecisionPolicy().fit_calibration(
        endpoint, np.zeros((2, 1)), np.array([0, 1])
    )
    assert endpoint_policy.threshold == pytest.approx(0.5)
    assert np.array_equal(endpoint_policy.predict(np.zeros((2, 1))).yhat, [0, 1])


def test_prediction_bundle_copies_and_validates_deterministic_values():
    p = np.array([0.2, 0.8])
    q = np.array([0.0, 1.0])
    yhat = np.array([0.0, 1.0])
    bundle = PredictionBundle(p, q, yhat)
    p[0] = 1.0
    assert bundle.p_event[0] == pytest.approx(0.2)
    with pytest.raises(PredictionContractError, match="equal"):
        PredictionBundle([0.1], [0.0, 1.0])
    with pytest.raises(PredictionContractError, match="equal q_decision"):
        PredictionBundle([0.1, 0.2], [0.0, 1.0], [1.0, 0.0])
