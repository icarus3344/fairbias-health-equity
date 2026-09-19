"""Survey-weighted evaluation metrics for NHIS Benchmark V1.

Implements:
1. Survey-weighted confusion matrix:
   - TPR_W, TNR_W, FPR_W, Balanced Accuracy (BA_W)
2. Handling of decision probability policy q:
   - Supports expectation E_w[q] for randomized classifiers (EG, TO)
   - Supports hard predictions y_pred in {0, 1}
3. Demographic Parity (DP) Gap:
   - Extreme range max_a SR(a) - min_a SR(a) across expected groups
4. Equalized Odds (EO) Gap:
   - max( max_a TPR(a) - min_a TPR(a), max_a FPR(a) - min_a FPR(a) )
5. Estimability contract:
   - Returns float or np.nan / 'NOT_ESTIMABLE' on empty support
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


def kish_effective_sample_size(weights: np.ndarray) -> float:
    """Return Kish's effective sample size for finite nonnegative weights."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be a finite, non-negative 1-dimensional array")
    positive = w[w > 0]
    if len(positive) == 0:
        return float("nan")
    scale = float(np.max(positive))
    scaled = positive / scale
    total = float(np.sum(scaled))
    denom = float(np.sum(scaled * scaled))
    if total <= 0 or denom <= 0 or not np.isfinite(total) or not np.isfinite(denom):
        return float("nan")
    return float(total * total / denom)


def compute_weighted_confusion_metrics(
    y_true: np.ndarray,
    y_decision: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, float]:
    """Compute survey-weighted TPR, TNR, FPR, selection rate, and balanced accuracy.

    y_decision can be binary predictions in {0, 1} or decision probabilities q in [0, 1].
    """
    y_raw = np.asarray(y_true)
    q = np.asarray(y_decision)
    w = np.asarray(weights)

    if y_raw.ndim != 1 or q.ndim != 1 or w.ndim != 1:
        raise ValueError("y_true, y_decision, and weights must be 1-dimensional")
    if np.iscomplexobj(y_raw) or np.iscomplexobj(q) or np.iscomplexobj(w):
        raise ValueError("complex inputs are not supported")
    try:
        y_num = y_raw.astype(float)
        q = q.astype(float)
        w = w.astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("inputs must be numeric") from exc
    if not np.all(np.isfinite(y_num)) or not np.all(np.isin(y_num, [0.0, 1.0])):
        raise ValueError("y_true must be finite binary values")
    y_true = y_num.astype(int)

    if len(y_true) != len(q) or len(y_true) != len(w):
        raise ValueError("Arrays y_true, y_decision, weights must have identical length.")
    if not np.all(np.isfinite(q)) or np.any(q < 0) or np.any(q > 1):
        raise ValueError("y_decision must be finite and lie in [0, 1]")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and non-negative")

    # Ratios are invariant to a common scale. Normalize BEFORE any sum so
    # finite individual weights cannot overflow totals or weighted products.
    if len(w) and np.max(w) > 0:
        w = w / np.max(w)
    total_w = float(np.sum(w))
    if total_w <= 0.0:
        return {
            "selection_rate": np.nan,
            "tpr": np.nan,
            "tnr": np.nan,
            "fpr": np.nan,
            "balanced_accuracy": np.nan,
            "ppv": np.nan, "npv": np.nan, "accuracy": np.nan,
        }

    # Selection rate
    sel_rate = float(np.sum(w * q) / total_w)

    # Positive class (y = 1)
    pos_mask = (y_true == 1)
    pos_w = float(np.sum(w[pos_mask]))
    tpr = float(np.sum(w[pos_mask] * q[pos_mask]) / pos_w) if pos_w > 0 else np.nan

    # Negative class (y = 0)
    neg_mask = (y_true == 0)
    neg_w = float(np.sum(w[neg_mask]))
    fpr = float(np.sum(w[neg_mask] * q[neg_mask]) / neg_w) if neg_w > 0 else np.nan
    tnr = 1.0 - fpr if not np.isnan(fpr) else np.nan

    # Balanced accuracy
    if not np.isnan(tpr) and not np.isnan(tnr):
        ba = 0.5 * (tpr + tnr)
    else:
        ba = np.nan
    predicted_positive = float(np.dot(w, q))
    predicted_negative = float(np.dot(w, 1-q))
    ppv = float(np.dot(w * y_true, q) / predicted_positive) if predicted_positive > 0 else np.nan
    npv = float(np.dot(w * (1-y_true), 1-q) / predicted_negative) if predicted_negative > 0 else np.nan

    return {
        "selection_rate": sel_rate,
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fpr,
        "balanced_accuracy": ba,
        "ppv": ppv, "npv": npv,
        "accuracy": float(np.dot(w, y_true*q + (1-y_true)*(1-q)) / total_w),
    }


def compute_survey_fairness_metrics(
    y_true: np.ndarray,
    y_decision: np.ndarray,
    A: np.ndarray,
    weights: np.ndarray,
    expected_groups: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Compute overall survey-weighted performance and multi-group fairness metrics.

    Calculates:
    - Overall: balanced_accuracy, selection_rate, tpr, fpr
    - Per-group metrics for all groups in expected_groups
    - Demographic Parity gap: max(SR) - min(SR)
    - Equalized Odds gap: max(max(TPR)-min(TPR), max(FPR)-min(FPR))
    """
    y_raw, q_raw, a_raw, w_raw = map(np.asarray, (y_true, y_decision, A, weights))
    if any(x.ndim != 1 for x in (y_raw, q_raw, a_raw, w_raw)):
        raise ValueError("y_true, y_decision, A, and weights must be 1-dimensional")
    if len({len(y_raw), len(q_raw), len(a_raw), len(w_raw)}) != 1:
        raise ValueError("y_true, y_decision, A, and weights must have identical lengths")
    if any(np.iscomplexobj(x) for x in (y_raw, q_raw, a_raw, w_raw)):
        raise ValueError("complex inputs are not supported")
    try:
        y_num, q, a_num, w = (y_raw.astype(float), q_raw.astype(float), a_raw.astype(float), w_raw.astype(float))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("metric inputs must be numeric") from exc
    if not np.all(np.isfinite(y_num)) or not np.all(np.isin(y_num, [0.0, 1.0])):
        raise ValueError("y_true must be finite binary values")
    if not np.all(np.isfinite(a_num)) or not np.all(np.equal(a_num, np.floor(a_num))):
        raise ValueError("A must contain finite integer group labels")
    y_true, A = y_num.astype(int), a_num.astype(int)
    if not set(A[w > 0]).issubset(set(expected_groups) if expected_groups is not None else set(A)):
        raise ValueError("Positive-weight observations contain an unexpected protected group")

    if expected_groups is None:
        expected_groups = sorted(np.unique(A))
    expected_groups = tuple(expected_groups)
    if len(expected_groups) == 0 or len(set(expected_groups)) != len(expected_groups):
        raise ValueError("expected_groups must be a non-empty sequence of unique labels")
    if any(isinstance(g, (bool, np.bool_)) or not isinstance(g, (int, np.integer)) for g in expected_groups):
        raise ValueError("expected_groups must contain integer labels")

    # Overall metrics
    overall = compute_weighted_confusion_metrics(y_true, q, w)

    # Group-level metrics
    group_metrics: Dict[int, Dict[str, float]] = {}
    group_srs: List[float] = []
    group_tprs: List[float] = []
    group_fprs: List[float] = []

    for g in expected_groups:
        g_mask = (A == g)
        if np.any(g_mask):
            gm = compute_weighted_confusion_metrics(y_true[g_mask], q[g_mask], w[g_mask])
            group_metrics[g] = gm
            if not np.isnan(gm["selection_rate"]):
                group_srs.append(gm["selection_rate"])
            if not np.isnan(gm["tpr"]):
                group_tprs.append(gm["tpr"])
            if not np.isnan(gm["fpr"]):
                group_fprs.append(gm["fpr"])
        else:
            group_metrics[g] = {
                "selection_rate": np.nan,
                "tpr": np.nan,
                "tnr": np.nan,
                "fpr": np.nan,
                "balanced_accuracy": np.nan,
                "ppv": np.nan, "npv": np.nan, "accuracy": np.nan,
            }

    # Gaps across groups
    # The registered domain is the estimand: do not silently replace it with
    # the subset of groups that happen to survive a replicate.
    all_sr_estimable = len(group_srs) == len(expected_groups)
    if all_sr_estimable and len(group_srs) >= 2:
        dp_gap = float(max(group_srs) - min(group_srs))
    else:
        dp_gap = np.nan

    all_eo_estimable = len(group_tprs) == len(expected_groups) and len(group_fprs) == len(expected_groups)
    if all_eo_estimable and len(group_tprs) >= 2 and len(group_fprs) >= 2:
        tpr_gap = float(max(group_tprs) - min(group_tprs))
        fpr_gap = float(max(group_fprs) - min(group_fprs))
        eo_gap = float(max(tpr_gap, fpr_gap))
    else:
        tpr_gap = np.nan
        fpr_gap = np.nan
        eo_gap = np.nan

    return {
        "balanced_accuracy": overall["balanced_accuracy"],
        "selection_rate": overall["selection_rate"],
        "tpr": overall["tpr"],
        "fpr": overall["fpr"],
        "ppv": overall["ppv"], "npv": overall["npv"], "accuracy": overall["accuracy"],
        "dp_gap": dp_gap,
        "eo_gap": eo_gap,
        "tpr_gap": tpr_gap,
        "fpr_gap": fpr_gap,
        "group_metrics": group_metrics,
        "effective_sample_size": kish_effective_sample_size(w),
        "group_effective_sample_size": {
            int(g): kish_effective_sample_size(w[A == g]) for g in expected_groups
        },
        "group_support": {int(g): {"n": int(np.sum(A == g)),
            "events": int(np.sum((A == g) & (y_true == 1))),
            "nonevents": int(np.sum((A == g) & (y_true == 0)))} for g in expected_groups},
    }
