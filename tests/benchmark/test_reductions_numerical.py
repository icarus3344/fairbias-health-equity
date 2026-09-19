"""Synthetic-only tests for bounded Fairlearn reductions PMF recovery."""

import numpy as np
import pytest

from nhis_fairbias.benchmark.adapters.adapter_reductions_numerical import (
    NumericalRecoveryError,
    NumericallyRecoveredExponentiatedGradientAdapter,
    recover_convex_mixture_q,
)
from nhis_fairbias.benchmark.adapters.adapter_reductions import ExponentiatedGradientAdapter
from nhis_fairbias.benchmark.predictions import FrozenDecisionPolicy, PredictionContractError


def test_repairs_only_endpoint_machine_roundoff_and_preserves_fractional_q():
    pmf = np.asarray([[-2.0e-16, 1.0 + 2.0e-16], [0.25, 0.75]])
    q = recover_convex_mixture_q(pmf, np.asarray([0.1, 0.9]))
    assert np.array_equal(q, np.asarray([1.0, 0.75]))


@pytest.mark.parametrize(
    "pmf, weights",
    [
        (np.asarray([[0.0, 1.0 + 1.0e-6]]), np.asarray([1.0])),
        (np.asarray([[0.0, 1.0]]), np.asarray([0.5, 0.500001])),
        (np.asarray([[0.0, 1.0]]), np.asarray([np.nan])),
        (np.asarray([[0.0, 1.0]]), np.asarray([-1.0e-16, 1.0 + 1.0e-16])),
    ],
)
def test_rejects_materially_invalid_or_nonfinite_mixture_inputs(pmf, weights):
    with pytest.raises(NumericalRecoveryError):
        recover_convex_mixture_q(pmf, weights)


def test_rejects_nonfinite_and_bad_row_sum_without_arbitrary_clipping():
    with pytest.raises(NumericalRecoveryError):
        recover_convex_mixture_q(np.asarray([[np.nan, 1.0]]), np.asarray([1.0]))
    with pytest.raises(NumericalRecoveryError):
        recover_convex_mixture_q(np.asarray([[0.2, 0.8 + 1.0e-6]]), np.asarray([1.0]))


@pytest.mark.parametrize(
    "pmf",
    [np.asarray([0.0, 1.0]), np.asarray([[0.0, 1.0, 0.0]]), np.asarray([[0.0 + 1.0j, 1.0 - 1.0j]])],
)
def test_rejects_nonfinite_complex_or_wrong_shape_pmf(pmf):
    with pytest.raises(NumericalRecoveryError):
        recover_convex_mixture_q(pmf, np.asarray([1.0]))


class _FakeFittedReductionsModel:
    weights_ = np.asarray([0.1, 0.9])

    def _pmf_predict(self, X):
        assert len(X) == 2
        return np.asarray([[-2.0e-16, 1.0 + 2.0e-16], [0.25, 0.75]])


def test_opt_in_adapter_recovers_frozen_policy_q_where_original_contract_fails():
    X = np.zeros((2, 1))
    y = np.asarray([1, 0])

    original = ExponentiatedGradientAdapter.__new__(ExponentiatedGradientAdapter)
    original.model = _FakeFittedReductionsModel()
    original.output_type = "decision_probability_q"
    with pytest.raises(PredictionContractError, match=r"must lie in \[0, 1\]"):
        FrozenDecisionPolicy().fit_calibration(original, X, y)

    recovered = NumericallyRecoveredExponentiatedGradientAdapter.__new__(
        NumericallyRecoveredExponentiatedGradientAdapter
    )
    recovered.model = _FakeFittedReductionsModel()
    recovered.output_type = "decision_probability_q"
    policy = FrozenDecisionPolicy().fit_calibration(recovered, X, y)
    assert np.array_equal(policy.predict(X).q_decision, np.asarray([1.0, 0.75]))
