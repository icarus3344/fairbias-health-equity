"""Independent fault injection at the actual frozen-inference boundary."""
from types import SimpleNamespace

import numpy as np
import pytest

from nhis_fairbias.benchmark.catalog_inference import (
    predict_frozen_bundle, predict_untouched_base_p,
)
from nhis_fairbias.benchmark.predictions import FrozenDecisionPolicy, PredictionBundle, PredictionContractError
from nhis_fairbias.benchmark.adapters.adapter_reductions import ExponentiatedGradientAdapter


@pytest.fixture
def eg():
    x = np.arange(16, dtype=float).reshape(8, 2)
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    a = np.tile([0, 1], 4)
    adapter = ExponentiatedGradientAdapter(max_iter=5, random_state=3).fit(x, y, a)
    return {"policy": FrozenDecisionPolicy().fit_calibration(adapter, x, y, a)}, x[:4], a[:4]


def eg_predict(model, x, a):
    return predict_frozen_bundle(model, x, a, method="EG_EO", expected_output_type="decision_probability_q",
                                 inference_policy="EG_BOUNDED_PMF_V1")


def event_model(output):
    policy = FrozenDecisionPolicy()
    policy._adapter = SimpleNamespace(output_type="event_probability_p")
    policy.predict = lambda x, a: output
    return {"policy": policy}


@pytest.mark.parametrize("n", [0, 2, 4])
def test_non_eg_wrong_length_is_rejected(n):
    model = event_model(PredictionBundle(np.full(n, .4), np.zeros(n)))
    with pytest.raises(PredictionContractError, match="length"):
        predict_frozen_bundle(model, np.ones((3, 2)), np.zeros(3), method="UNMITIGATED",
            expected_output_type="event_probability_p", inference_policy="FROZEN_POLICY_PREDICT_V1")


def test_eg_constraint_checked_with_other_inputs_valid(eg):
    model, x, a = eg
    model["policy"]._adapter.constraint_type = "demographic_parity"
    with pytest.raises(PredictionContractError, match="constraint"):
        eg_predict(model, x, a)


@pytest.mark.parametrize("n", [0, 3, 5])
def test_eg_length_checked_with_valid_probability_rows(eg, n):
    model, x, a = eg
    model["policy"]._adapter.model._pmf_predict = lambda x: np.tile([.75, .25], (n, 1))
    with pytest.raises(PredictionContractError, match="length"):
        eg_predict(model, x, a)


@pytest.mark.parametrize("value", [np.nan, np.inf, -1e-6, 1.000001, .2+.1j])
def test_invalid_pmf_cannot_be_clipped_into_validity(eg, value):
    model, x, a = eg
    model["policy"]._adapter.model._pmf_predict = lambda x: np.tile([1-value, value], (len(x), 1))
    with pytest.raises(PredictionContractError, match="PMF"):
        eg_predict(model, x, a)


def test_endpoint_repair_preserves_fraction_and_never_fits(eg, monkeypatch):
    model, x, a = eg
    adapter = model["policy"]._adapter
    eps = np.finfo(float).eps
    pmf = np.array([[1+eps, -eps], [.75, .25], [0., 1.], [1., 0.]])
    monkeypatch.setattr(adapter.model, "_pmf_predict", lambda x: pmf)
    def forbidden(*args, **kwargs):
        raise AssertionError("Fitting during frozen inference")
    monkeypatch.setattr(adapter, "fit", forbidden)
    monkeypatch.setattr(adapter.model, "fit", forbidden)
    monkeypatch.setattr(FrozenDecisionPolicy, "fit_calibration", forbidden)
    original_pmf = pmf.copy()
    original_weights = adapter.model.weights_.copy()
    bundle = eg_predict(model, x, a)
    np.testing.assert_array_equal(bundle.q_decision, [0., .25, 1., 0.])
    np.testing.assert_array_equal(adapter.model.weights_, original_weights)
    np.testing.assert_array_equal(pmf, original_pmf)
    assert bundle.p_event is None and bundle.yhat is None


def test_mutated_existing_bundle_is_revalidated():
    bundle = PredictionBundle(np.array([.4]*3), np.zeros(3))
    bundle.q_decision.setflags(write=True)
    bundle.q_decision[0] = np.nan
    with pytest.raises(PredictionContractError, match="non-finite"):
        predict_frozen_bundle(event_model(bundle), np.ones((3, 2)), np.zeros(3), method="UNMITIGATED",
            expected_output_type="event_probability_p", inference_policy="FROZEN_POLICY_PREDICT_V1")


@pytest.mark.parametrize("values", [[.2, .3], [.2]*4, [np.nan]*3, [1.1]*3, [.2+.1j]*3, [".2"]*3])
def test_untouched_base_risk_is_strictly_validated(values):
    model = event_model(None)
    model["policy"]._adapter.predict_event_probability = lambda x, A: np.array(values)
    with pytest.raises(PredictionContractError):
        predict_untouched_base_p(model, np.ones((3, 2)), np.zeros(3))


def test_untouched_base_does_not_expose_model_array():
    p = np.array([.1, .2, .3])
    model = event_model(None)
    model["policy"]._adapter.predict_event_probability = lambda x, A: p
    actual = predict_untouched_base_p(model, np.ones((3, 2)), np.zeros(3))
    np.testing.assert_array_equal(actual, p)
    assert not actual.flags.writeable and not np.shares_memory(actual, p)
