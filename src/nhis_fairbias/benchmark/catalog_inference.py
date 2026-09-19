"""Pure prediction from an already-fitted frozen benchmark artifact."""
from __future__ import annotations

from typing import Any

import numpy as np

from .adapters.adapter_reductions import ExponentiatedGradientAdapter
from .adapters.adapter_reductions_numerical import (
    NumericallyRecoveredExponentiatedGradientAdapter,
    recover_convex_mixture_q,
)
from .predictions import FrozenDecisionPolicy, PredictionBundle, PredictionContractError

Q_METHODS = {"EG_DP", "EG_EO", "TO_EO", "OXONFAIR_EO"}
P_METHODS = {"UNMITIGATED", "FAIRBIAS_BM", "REWEIGHING", "LFR_RECONSTRUCTED",
             "FAIRGBM_EO", "FAIRRET_EO", "FRAPPE_EO", "FAIRBIAS_BM_AE", "FAIRBIAS_JOINT"}

EG_BOUNDED_PMF_POLICY = "EG_BOUNDED_PMF_V1"
_EG_TYPES = (ExponentiatedGradientAdapter, NumericallyRecoveredExponentiatedGradientAdapter)


def _validate_expected(bundle: PredictionBundle, expected: str) -> PredictionBundle:
    if not isinstance(bundle, PredictionBundle):
        raise PredictionContractError("frozen policy must return PredictionBundle")
    if expected == "event_probability_p" and bundle.p_event is None:
        raise PredictionContractError("expected event-risk p but policy returned q only")
    if expected == "decision_probability_q" and bundle.p_event is not None:
        raise PredictionContractError("expected decision q but policy returned event-risk p")
    # Reconstruct to validate lengths/ranges again and detach any caller-mutated
    # write flags or array contents from a previously returned bundle.
    return PredictionBundle(bundle.p_event, bundle.q_decision, bundle.yhat, bundle.provenance)


def _validate_inputs(X: Any, A: Any) -> int:
    arr = np.asarray(X)
    if arr.ndim != 2 or min(arr.shape) == 0:
        raise PredictionContractError("X must be a non-empty two-dimensional table")
    n = int(arr.shape[0])
    if A is not None:
        sensitive = np.asarray(A)
        if sensitive.ndim != 1 or len(sensitive) != n:
            raise PredictionContractError("A must be one-dimensional and match X length")
    return n


def predict_frozen_bundle(model: dict[str, Any], X: Any, A: Any = None, *,
                          method: str, expected_output_type: str,
                          inference_policy: str) -> PredictionBundle:
    """Predict from a serialized fitted model without fitting or mutation.

    EG artifacts use one explicit bounded-PMF path for both the original and
    numerical-recovery adapters. Other frozen policies use their existing
    ``predict`` method and must return a validated :class:`PredictionBundle`.
    """
    if expected_output_type not in {"event_probability_p", "decision_probability_q"}:
        raise PredictionContractError("unsupported expected_output_type")
    if not isinstance(method, str) or (method not in Q_METHODS | P_METHODS
            and not method.startswith("FAIRBIAS_GEOMETRY_")):
        raise PredictionContractError("unsupported catalog method")
    inferred_output = "decision_probability_q" if method in Q_METHODS else "event_probability_p"
    if method.startswith("FAIRBIAS_GEOMETRY_"):
        inferred_output = "event_probability_p"
    if expected_output_type != inferred_output:
        raise PredictionContractError("method/output contract mismatch")
    n = _validate_inputs(X, A)
    if not isinstance(model, dict) or "policy" not in model:
        raise PredictionContractError("model artifact must contain policy")
    policy = model["policy"]
    if type(policy) is not FrozenDecisionPolicy or not policy.is_calibrated:
        raise PredictionContractError("model policy must be a calibrated FrozenDecisionPolicy")
    if getattr(policy._adapter, "output_type", None) != expected_output_type:
        raise PredictionContractError("frozen adapter/output contract mismatch")
    if method in {"EG_DP", "EG_EO"}:
        if inference_policy != EG_BOUNDED_PMF_POLICY:
            raise PredictionContractError("EG requires explicit bounded-PMF inference policy")
        adapter = getattr(policy, "_adapter", None)
        if type(adapter) not in _EG_TYPES:
            raise PredictionContractError("EG policy adapter is not a fitted supported EG adapter")
        if getattr(policy, "threshold", None) is not None:
            raise PredictionContractError("EG inference forbids calibration thresholds")
        if getattr(adapter, "output_type", None) != "decision_probability_q":
            raise PredictionContractError("EG adapter output type is not q")
        expected_constraint = "demographic_parity" if method == "EG_DP" else "equalized_odds"
        if str(getattr(adapter, "constraint_type", "")).lower() not in {expected_constraint, "dp" if method == "EG_DP" else "eo"}:
            raise PredictionContractError("EG method/constraint mismatch")
        fitted_model = getattr(adapter, "model", None)
        weights = getattr(fitted_model, "weights_", None)
        pmf_predict = getattr(fitted_model, "_pmf_predict", None)
        if weights is None or not callable(pmf_predict):
            raise PredictionContractError("EG artifact is not fitted with PMF and weights")
        pmf = pmf_predict(X)
        try:
            q = recover_convex_mixture_q(pmf, weights)
        except (ValueError, TypeError) as exc:
            raise PredictionContractError("EG PMF recovery failed validation") from exc
        if len(q) != n:
            raise PredictionContractError("EG q length does not match X")
        return _validate_expected(PredictionBundle(None, q, None, {"inference_policy": inference_policy, "method": method}), expected_output_type)
    if inference_policy != "FROZEN_POLICY_PREDICT_V1":
        raise PredictionContractError("non-EG inference requires explicit frozen-policy schema")
    predict = getattr(policy, "predict", None)
    if not callable(predict):
        raise PredictionContractError("frozen policy has no predict method")
    bundle = _validate_expected(predict(X, A), expected_output_type)
    if bundle.n_samples != n:
        raise PredictionContractError("frozen prediction length does not match X")
    return bundle


def predict_untouched_base_p(model: dict[str, Any], X: Any, A: Any = None) -> np.ndarray:
    """Return untouched-base event risk through an existing fitted interface."""
    policy = model.get("policy") if isinstance(model, dict) else None
    if type(policy) is not FrozenDecisionPolicy or not policy.is_calibrated:
        raise PredictionContractError("base risk requires a calibrated frozen policy")
    adapter = policy._adapter
    predict = getattr(adapter, "predict_event_probability", None)
    if not callable(predict):
        raise PredictionContractError("fitted adapter has no untouched-base p interface")
    n = _validate_inputs(X, A)
    raw = np.asarray(predict(X, A=A))
    if raw.ndim != 1 or len(raw) != n or not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw) or not np.all(np.isfinite(raw)) or np.any((raw < 0) | (raw > 1)):
        raise PredictionContractError("untouched-base p is invalid")
    result = np.array(raw, dtype=float, copy=True)
    result.setflags(write=False)
    return result
