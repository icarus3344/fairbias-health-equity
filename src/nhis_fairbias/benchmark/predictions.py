"""Prediction bundles and frozen C-set decision policies.

The benchmark has two deliberately different probability meanings:
``p_event`` is an event-risk probability, while ``q_decision`` is the
probability that a policy makes a positive decision.  This module keeps those
meanings separate at the adapter boundary and never re-fits an adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional

import numpy as np


class PredictionContractError(ValueError):
    """Raised when an adapter prediction cannot satisfy the benchmark contract."""


def _vector(value: Any, name: str, *, binary: bool = False) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise PredictionContractError(f"{name} must be a one-dimensional vector; got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
        raise PredictionContractError(f"{name} must be a real numeric vector")
    arr = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise PredictionContractError(f"{name} contains non-finite values")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise PredictionContractError(f"{name} must lie in [0, 1]")
    if binary and not np.all((arr == 0.0) | (arr == 1.0)):
        raise PredictionContractError(f"{name} must contain only 0/1 values")
    return arr


@dataclass(frozen=True)
class PredictionBundle:
    """Immutable-ish, validated prediction values with explicit semantics.

    Arrays are copied and marked read-only during construction.  ``q_decision``
    is always present; ``p_event`` is absent for methods whose native output is
    a decision probability.  A decision-probability-only bundle therefore
    cannot be accidentally consumed as an event-risk prediction.
    """

    p_event: Optional[np.ndarray]
    q_decision: np.ndarray
    yhat: Optional[np.ndarray] = None
    provenance: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        p = None if self.p_event is None else _vector(self.p_event, "p_event")
        q = _vector(self.q_decision, "q_decision")
        yhat = None if self.yhat is None else _vector(self.yhat, "yhat", binary=True)
        n = len(q)
        if p is not None and len(p) != n:
            raise PredictionContractError("p_event and q_decision must have equal length")
        if yhat is not None:
            if len(yhat) != n:
                raise PredictionContractError("yhat and q_decision must have equal length")
            if not np.array_equal(yhat, q):
                raise PredictionContractError("for deterministic policies yhat must equal q_decision")

        # frozen dataclasses do not freeze numpy fields, so replace each
        # reference with a private read-only copy.
        if p is not None:
            p = np.array(p, copy=True)
            p.setflags(write=False)
            object.__setattr__(self, "p_event", p)
        q = np.array(q, copy=True)
        q.setflags(write=False)
        object.__setattr__(self, "q_decision", q)
        if yhat is not None:
            yhat = np.array(yhat, copy=True)
            yhat.setflags(write=False)
            object.__setattr__(self, "yhat", yhat)
        if not isinstance(self.provenance, Mapping):
            raise PredictionContractError("provenance must be a mapping")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def n_samples(self) -> int:
        return len(self.q_decision)

    @property
    def has_event_risk(self) -> bool:
        return self.p_event is not None

    def risk_probability(self) -> np.ndarray:
        """Return p, refusing to reinterpret q as event risk."""
        if self.p_event is None:
            raise PredictionContractError(
                "q_decision is not an event-risk probability; p_event is required"
            )
        return self.p_event

    def __reduce__(self):
        return (type(self), (self.p_event, self.q_decision, self.yhat, dict(self.provenance)))


def _validate_binary_labels(y: Any, expected_length: int, name: str) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1 or len(arr) != expected_length:
        raise PredictionContractError(f"{name} must be a vector of length {expected_length}")
    try:
        finite = np.isfinite(arr)
    except TypeError as exc:
        raise PredictionContractError(f"{name} must contain finite binary labels") from exc
    if not np.all(finite) or not np.all((arr == 0) | (arr == 1)):
        raise PredictionContractError(f"{name} must contain finite binary labels")
    return np.asarray(arr, dtype=int)


def _best_balanced_accuracy_threshold(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Find the C-set BA threshold in O(n log n), including policy boundaries."""
    if not np.any(y == 0) or not np.any(y == 1):
        raise PredictionContractError("calibration labels must contain both classes")

    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    ys = y[order]
    unique = np.unique(ps)
    candidates = [0.0, 1.0]
    if len(unique) > 1:
        candidates.extend(((unique[:-1] + unique[1:]) / 2.0).tolist())
    # With >=, 1.0 is not all-negative when p contains an exact 1.  Include
    # the smallest representable threshold above that endpoint in that case.
    if ps[-1] >= 1.0:
        candidates.append(float(np.nextafter(1.0, np.inf)))

    total_pos = int(np.sum(ys == 1))
    total_neg = int(np.sum(ys == 0))
    pos_suffix = np.cumsum((ys == 1)[::-1])[::-1]
    neg_suffix = np.cumsum((ys == 0)[::-1])[::-1]
    scored = []
    for threshold in sorted(set(float(t) for t in candidates)):
        idx = int(np.searchsorted(ps, threshold, side="left"))
        tp = int(pos_suffix[idx]) if idx < len(ps) else 0
        fp = int(neg_suffix[idx]) if idx < len(ps) else 0
        tn = total_neg - fp
        ba = 0.5 * (tp / total_pos + tn / total_neg)
        scored.append((ba, threshold))

    # Maximise BA; exact ties use the specified closest-to-.5, then larger-t
    # rule.  Scores are rational counts, so equality is stable here.
    best_ba, best_t = sorted(
        scored, key=lambda item: (-item[0], abs(item[1] - 0.5), -item[1])
    )[0]
    return float(best_t), float(best_ba)


class FrozenDecisionPolicy:
    """Adapter-facing frozen prediction policy for the F/C contract."""

    def __init__(self) -> None:
        self._adapter: Any = None
        self._threshold: Optional[float] = None
        self._calibration: Mapping[str, Any] = MappingProxyType({})

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_calibration"] = dict(self._calibration)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._calibration = MappingProxyType(dict(state["_calibration"]))

    @property
    def threshold(self) -> Optional[float]:
        return self._threshold

    @property
    def is_calibrated(self) -> bool:
        return self._adapter is not None

    def fit_calibration(
        self,
        adapter: Any,
        X_C: Any,
        y_C: Any,
        A_C: Optional[Any] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "FrozenDecisionPolicy":
        """Freeze a policy from C predictions without fitting the adapter."""
        self._adapter = None
        self._threshold = None
        self._calibration = MappingProxyType({})
        output_type = getattr(adapter, "output_type", None)
        if output_type not in ("event_probability_p", "decision_probability_q"):
            raise PredictionContractError(f"unsupported adapter output_type: {output_type!r}")
        n = len(X_C)
        y = _validate_binary_labels(y_C, n, "y_C")
        requires_a = bool(getattr(adapter, "requires_sensitive_at_predict", False))
        if requires_a and A_C is None:
            raise PredictionContractError("adapter requires A_C at prediction time")
        if A_C is not None and len(A_C) != n:
            raise PredictionContractError("A_C length must match X_C")
        raw = adapter.predict_decision_proba(X_C, A=A_C)
        values = _vector(raw, "adapter decision output")
        if len(values) != n:
            raise PredictionContractError("adapter decision output length does not match X_C")

        threshold = None
        calibration: dict[str, Any] = {
            "adapter_name": getattr(adapter, "name", type(adapter).__name__),
            "adapter_output_type": output_type,
            "n_calibration": n,
        }
        if output_type == "event_probability_p":
            threshold, ba = _best_balanced_accuracy_threshold(values, y)
            calibration.update({"threshold": threshold, "calibration_balanced_accuracy": ba})
        else:
            calibration["threshold"] = None
            calibration["q_preserved"] = True
        if metadata is not None:
            calibration["metadata"] = dict(metadata)

        self._adapter = adapter
        self._threshold = threshold
        self._calibration = MappingProxyType(calibration)
        return self

    def predict(self, X: Any, A: Optional[Any] = None) -> PredictionBundle:
        if self._adapter is None:
            raise RuntimeError("FrozenDecisionPolicy must be calibrated before prediction")
        requires_a = bool(getattr(self._adapter, "requires_sensitive_at_predict", False))
        if requires_a and A is None:
            raise PredictionContractError("adapter requires A at prediction time")
        if A is not None and len(A) != len(X):
            raise PredictionContractError("A length must match X")
        output_type = getattr(self._adapter, "output_type", None)
        raw = _vector(
            self._adapter.predict_decision_proba(X, A=A), "adapter decision output"
        )
        if len(raw) != len(X):
            raise PredictionContractError("adapter decision output length does not match X")
        provenance = dict(self._calibration)
        provenance["prediction_n"] = len(raw)
        if output_type == "event_probability_p":
            assert self._threshold is not None
            yhat = (raw >= self._threshold).astype(float)
            provenance["decision_rule"] = ">= frozen global C threshold"
            return PredictionBundle(raw, yhat, yhat, provenance)
        if output_type == "decision_probability_q":
            provenance["decision_rule"] = "native adapter decision probability"
            return PredictionBundle(None, raw, None, provenance)
        raise PredictionContractError(f"unsupported adapter output_type: {output_type!r}")
