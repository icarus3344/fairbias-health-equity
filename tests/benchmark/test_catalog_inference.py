from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from nhis_fairbias.benchmark.adapters.adapter_reductions import ExponentiatedGradientAdapter
from nhis_fairbias.benchmark.adapters.adapter_reductions_numerical import NumericallyRecoveredExponentiatedGradientAdapter
from nhis_fairbias.benchmark.catalog_inference import EG_BOUNDED_PMF_POLICY, predict_frozen_bundle
from nhis_fairbias.benchmark.predictions import FrozenDecisionPolicy, PredictionBundle, PredictionContractError


def _fitted_model(adapter):
    X = np.array([[-2.0], [-1.0], [-.2], [.1], [.8], [1.2], [2.0], [2.5]])
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    A = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    adapter.fit(X, y, A)
    policy = FrozenDecisionPolicy().fit_calibration(adapter, X, y, A)
    return {"policy": policy}, X[:4], A[:4]


def test_original_and_recovered_eg_use_bounded_pmf_without_fitting():
    for adapter in (ExponentiatedGradientAdapter(max_iter=10, random_state=3),
                    NumericallyRecoveredExponentiatedGradientAdapter(max_iter=10, random_state=3)):
        model, X, A = _fitted_model(adapter)
        before = adapter.model.weights_.copy() if hasattr(adapter.model.weights_, "copy") else adapter.model.weights_
        bundle = predict_frozen_bundle(model, X, A, method="EG_EO",
                                       expected_output_type="decision_probability_q",
                                       inference_policy=EG_BOUNDED_PMF_POLICY)
        assert bundle.p_event is None
        assert bundle.q_decision.shape == (4,)
        assert np.array_equal(adapter.model.weights_, before)


def test_eg_requires_explicit_policy_and_rejects_threshold_or_wrong_type():
    model, X, A = _fitted_model(ExponentiatedGradientAdapter(max_iter=5))
    with pytest.raises(PredictionContractError):
        predict_frozen_bundle(model, X, A, method="EG_EO", expected_output_type="decision_probability_q", inference_policy="FROZEN_POLICY_PREDICT_V1")
    model["policy"]._threshold = .5
    with pytest.raises(PredictionContractError):
        predict_frozen_bundle(model, X, A, method="EG_EO", expected_output_type="decision_probability_q", inference_policy=EG_BOUNDED_PMF_POLICY)


@pytest.mark.parametrize("bad_method", ["UNKNOWN", "EG_EO"])
def test_inference_rejects_bad_method_inputs_and_unfitted_policy(bad_method):
    model, X, A = _fitted_model(ExponentiatedGradientAdapter(max_iter=5))
    if bad_method == "EG_EO":
        model["policy"]._adapter.constraint_type = "demographic_parity"
    with pytest.raises(PredictionContractError):
        predict_frozen_bundle(model, X, A[:2], method=bad_method,
                              expected_output_type="decision_probability_q",
                              inference_policy=EG_BOUNDED_PMF_POLICY)
    with pytest.raises(PredictionContractError):
        predict_frozen_bundle({"policy": FrozenDecisionPolicy()}, X, A,
                              method="UNMITIGATED", expected_output_type="event_probability_p",
                              inference_policy="FROZEN_POLICY_PREDICT_V1")


def test_eg_rejects_wrong_pmf_length_and_nonfinite_values():
    model, X, A = _fitted_model(ExponentiatedGradientAdapter(max_iter=5))
    fitted = model["policy"]._adapter.model
    fitted._pmf_predict = lambda query: np.zeros((len(query) + 1, 2))
    with pytest.raises(PredictionContractError):
        predict_frozen_bundle(model, X, A, method="EG_EO", expected_output_type="decision_probability_q", inference_policy=EG_BOUNDED_PMF_POLICY)


def test_non_eg_uses_frozen_policy_and_validates_semantics():
    policy = FrozenDecisionPolicy()
    policy._adapter = SimpleNamespace(output_type="event_probability_p")
    policy.predict = lambda X, A: PredictionBundle(np.full(len(X), .4), np.full(len(X), .2), None)
    X = np.zeros((3, 1))
    bundle = predict_frozen_bundle({"policy": policy}, X, np.zeros(3), method="UNMITIGATED",
                                   expected_output_type="event_probability_p", inference_policy="FROZEN_POLICY_PREDICT_V1")
    assert np.all(bundle.p_event == .4)
    with pytest.raises(PredictionContractError):
        predict_frozen_bundle({"policy": policy}, X, np.zeros(3), method="UNMITIGATED",
                              expected_output_type="decision_probability_q", inference_policy="FROZEN_POLICY_PREDICT_V1")
