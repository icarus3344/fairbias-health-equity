"""Opt-in numerical recovery for Fairlearn reductions decision mixtures.

Fairlearn's reductions PMF is a two-column ``[1-q, q]`` array formed from a
weighted sum of base predictions.  At an endpoint, floating-point summation
can produce a value a few ulps outside ``[0, 1]`` even when the weights are a
valid convex mixture.  This module repairs only that bounded roundoff case;
it does not clip arbitrary adapter outputs.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .adapter_reductions import ExponentiatedGradientAdapter


class NumericalRecoveryError(ValueError):
    """Raised when a reduction PMF is not a numerically valid convex mixture."""


def recover_convex_mixture_q(
    pmf: Any,
    weights: Any,
) -> np.ndarray:
    """Return q from a Fairlearn PMF, repairing only endpoint roundoff.

    ``weights`` must be finite, nonnegative, and sum to one within the same
    roundoff envelope.  Both PMF columns must be finite, in the envelope, and
    sum to one row-wise.  Values outside the envelope are rejected rather than
    clipped.  Fractional q values are returned unchanged (subject only to a
    copy), so this helper preserves decision-probability ``q`` semantics and
    never creates an event-risk ``p`` value.
    """
    raw_w = np.asarray(weights)
    if np.iscomplexobj(raw_w):
        raise NumericalRecoveryError("reduction weights must be real")
    w = np.asarray(raw_w, dtype=float)
    if w.ndim != 1 or w.size == 0 or not np.all(np.isfinite(w)):
        raise NumericalRecoveryError("reduction weights must be a finite non-empty vector")

    # A dot product of k terms accumulates O(k*eps) error.  Keep the bound
    # deliberately small and scale-free; this is not a general probability
    # clipping tolerance.
    eps = np.finfo(float).eps
    bound = 8.0 * eps * max(1, w.size)
    if np.any(w < 0.0) or not np.isclose(np.sum(w), 1.0, rtol=0.0, atol=bound):
        raise NumericalRecoveryError("reduction weights are not a valid convex mixture")

    raw_arr = np.asarray(pmf)
    if np.iscomplexobj(raw_arr):
        raise NumericalRecoveryError("reduction PMF must be real")
    arr = np.asarray(raw_arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or not np.all(np.isfinite(arr)):
        raise NumericalRecoveryError("reduction PMF must be a finite (n, 2) array")
    if np.any(arr < -bound) or np.any(arr > 1.0 + bound):
        raise NumericalRecoveryError("reduction PMF is materially outside [0, 1]")
    if not np.all(np.isclose(np.sum(arr, axis=1), 1.0, rtol=0.0, atol=bound)):
        raise NumericalRecoveryError("reduction PMF rows must sum to one")

    q = np.array(arr[:, 1], copy=True)
    # Endpoint repair only.  Interior fractional probabilities remain exact.
    q[(q < 0.0) & (q >= -bound)] = 0.0
    q[(q > 1.0) & (q <= 1.0 + bound)] = 1.0
    return q


class NumericallyRecoveredExponentiatedGradientAdapter(ExponentiatedGradientAdapter):
    """Explicit opt-in EG adapter using bounded PMF roundoff recovery."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.name = f"{self.name}_NUMERICAL_RECOVERY"
        self.upstream_implementation = (
            "fairlearn.reductions.ExponentiatedGradient + bounded PMF roundoff recovery"
        )

    def predict_decision_proba(
        self, X: np.ndarray, A: Optional[np.ndarray] = None
    ) -> np.ndarray:
        pmf = self.model._pmf_predict(X)
        weights = getattr(self.model, "weights_", None)
        if weights is None:
            raise NumericalRecoveryError("fitted reductions model has no mixture weights")
        if hasattr(weights, "to_numpy"):
            weights = weights.to_numpy()
        return recover_convex_mixture_q(pmf, weights)
