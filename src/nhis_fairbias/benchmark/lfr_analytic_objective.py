"""Piecewise analytic derivative of the *unchanged* AIF360 LFR objective.

Reference: aif360.algorithms.preprocessing.lfr_helpers.helpers (AIF360 0.6.1).
This uses Euclidean distances, not squared distances; group reconstruction
losses are separately averaged, label loss is pooled, and fairness is the mean
absolute membership difference. No smoothing, sample reweighting, feature
scaling, clipping change, or regularization is introduced.

The objective is not everywhere differentiable. At zero distance we choose a
zero norm derivative; at exact absolute-value and clip corners we choose zero.
These are explicit primitive derivative conventions, not a claim of a unique
classical gradient (nor a convergence guarantee for nonsmooth L-BFGS-B).
Auxiliary arrays are O(n*k + n*d + k*d), without an n*n or n*k*d tensor.

Derivative on the smooth region (group g has n_g rows and d features):
    M_g = softmax(S_g), S_g = -cdist(X_g, V), R_g = M_g V, q_g = M_g w.
    H_g = 2 A_x (R_g - X_g)/(n_g d).
    t_g = A_y/N (-y_g/q_g + (1-y_g)/(1-q_g)), times the clip indicator.
    B_g = H_g V.T + t_g w.T +/- A_z sign(mean(M_u)-mean(M_p))/(k n_g).
    C_g = M_g * (B_g - rowsum(M_g * B_g)).
    dL/dw = sum_g M_g.T t_g.
    dL/dV_j = sum_g [M_g.T H_g]_j
              + sum_i C_g[i,j] (X_g[i]-V_j)/||X_g[i]-V_j||.
These follow the MSE, pooled cross-entropy, absolute-value, row-softmax and
Euclidean-norm chain rules directly. The implementation below computes exactly
these expressions, using zero conventions only where those rules are undefined.
"""

from __future__ import annotations

from typing import MutableMapping, Any

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax


OBJECTIVE_ID = "aif360_lfr_euclidean_group_mse_pooled_clipped_ce_mean_abs_v1"
GRADIENT_ID = "analytic_piecewise_zero_at_norm_abs_clip_corners_v1"


def lfr_objective_and_gradient(
    parameters, x_unprivileged, x_privileged, y_unprivileged, y_privileged,
    k=10, A_x=0.01, A_y=0.1, A_z=0.5, print_interval=250, verbose=0,
    *, diagnostics: MutableMapping[str, Any] | None = None,
):
    """Return upstream-equivalent loss and its piecewise analytic derivative.

    Argument order matches ``LFR_optim_objective``. Printing is intentionally
    omitted; a caller can record the returned aggregate diagnostics. Inputs
    must contain both nonempty sensitive groups and binary labels.
    """
    p = np.asarray(parameters, dtype=float)
    groups = [np.asarray(x_unprivileged, dtype=float),
              np.asarray(x_privileged, dtype=float)]
    labels = [np.asarray(y_unprivileged, dtype=float).reshape(-1, 1),
              np.asarray(y_privileged, dtype=float).reshape(-1, 1)]
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("LFR k must be a positive integer")
    if any(x.ndim != 2 or min(x.shape) == 0 for x in groups):
        raise ValueError("LFR needs two nonempty two-dimensional feature groups")
    d = groups[0].shape[1]
    if groups[1].shape[1] != d or p.shape != (k * (d + 1),):
        raise ValueError("LFR feature/parameter dimensions disagree")
    if any(len(y) != len(x) or not np.isin(y, [0, 1]).all()
           for x, y in zip(groups, labels)):
        raise ValueError("LFR needs aligned binary labels")
    if not all(np.isfinite(v).all() for v in [p, *groups, *labels]):
        raise ValueError("LFR objective inputs must be finite")
    if not np.isfinite([A_x, A_y, A_z]).all() or min(A_x, A_y, A_z) < 0:
        raise ValueError("LFR objective weights must be finite and nonnegative")

    w, prototypes = p[:k], p[k:].reshape(k, d)
    eps = np.finfo(float).eps
    states = []
    for x in groups:
        distance = cdist(x, prototypes)
        membership = softmax(-distance, axis=1)
        x_hat = membership @ prototypes
        raw_y_hat = membership @ w.reshape(-1, 1)
        y_hat = np.clip(raw_y_hat, eps, 1.0 - eps)
        states.append((distance, membership, x_hat, raw_y_hat, y_hat))

    # Match upstream operation order, including group-specific MSE divisors.
    x_loss = (np.mean((states[0][2] - groups[0]) ** 2)
              + np.mean((states[1][2] - groups[1]) ** 2))
    delta = np.mean(states[0][1], axis=0) - np.mean(states[1][1], axis=0)
    z_loss = np.mean(abs(delta))
    y_hat = np.concatenate([states[0][4], states[1][4]], axis=0)
    y = np.concatenate(labels, axis=0)
    y_loss = -np.mean(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat))
    loss = A_x * x_loss + A_y * y_loss + A_z * z_loss

    grad_w = np.zeros(k)
    grad_prototypes = np.zeros_like(prototypes)
    fairness_derivative = A_z * np.sign(delta) / k
    zero_distances = clip_corners = clipped_values = 0
    for index, (x, group_y, state) in enumerate(zip(groups, labels, states)):
        distance, membership, x_hat, raw_q, q = state
        grad_x_hat = 2.0 * A_x * (x_hat - x) / x.size
        interior = (raw_q > eps) & (raw_q < 1.0 - eps)
        grad_q = A_y * (-group_y / q + (1.0 - group_y) / (1.0 - q)) / len(y)
        grad_q *= interior
        grad_w += (membership.T @ grad_q).ravel()
        grad_membership = grad_x_hat @ prototypes.T + grad_q @ w[None, :]
        group_sign = 1.0 if index == 0 else -1.0
        grad_membership += group_sign * fairness_derivative[None, :] / len(x)
        grad_scores = membership * (
            grad_membership - np.sum(membership * grad_membership, axis=1, keepdims=True)
        )
        distance_factor = np.divide(
            grad_scores, distance, out=np.zeros_like(grad_scores), where=distance != 0
        )
        grad_prototypes += membership.T @ grad_x_hat
        # d(-||x-v||)/dv=(x-v)/||x-v||, evaluated without a 3D tensor.
        grad_prototypes += (distance_factor.T @ x
                            - np.sum(distance_factor, axis=0)[:, None] * prototypes)
        zero_distances += int(np.count_nonzero(distance == 0))
        clip_corners += int(np.count_nonzero((raw_q == eps) | (raw_q == 1.0 - eps)))
        clipped_values += int(np.count_nonzero(~interior))

    gradient = np.concatenate([grad_w, grad_prototypes.ravel()])
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite LFR objective or analytic derivative")
    if diagnostics is not None:
        diagnostics.update({
            "reconstruction_loss": float(x_loss), "label_loss": float(y_loss),
            "fairness_loss": float(z_loss), "zero_distance_pairs": zero_distances,
            "absolute_value_corners": int(np.count_nonzero(delta == 0)),
            "clip_corners": clip_corners, "clipped_probabilities": clipped_values,
            "primitive_corner_present": bool(zero_distances or clip_corners or np.any(delta == 0)),
        })
    return float(loss), gradient
