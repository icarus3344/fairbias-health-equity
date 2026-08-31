"""Survey-weighted evaluation metrics, capacity-aware allocations, and subgroup fairness disparities."""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def weighted_auroc(
    y_true: Sequence[float] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Compute survey-weighted Area Under the ROC Curve."""
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_prob, dtype=float).ravel()
    w = np.asarray(sample_weight, dtype=float).ravel() if sample_weight is not None else None

    if len(np.unique(y_t)) < 2:
        return 0.5

    try:
        return float(roc_auc_score(y_t, y_p, sample_weight=w))
    except Exception:
        return 0.5


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Compute numerical integration via trapezoid rule with backward compatibility for numpy < 2.0 and numpy >= 2.0."""
    trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trap_fn is not None:
        return float(trap_fn(y, x))
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    return float(np.sum((x_arr[1:] - x_arr[:-1]) * (y_arr[1:] + y_arr[:-1]) / 2.0))


def weighted_auprc(
    y_true: Sequence[float] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Compute survey-weighted Area Under the Precision-Recall Curve (Weighted AUPRC)."""
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_prob, dtype=float).ravel()
    w = np.asarray(sample_weight, dtype=float).ravel() if sample_weight is not None else np.ones_like(y_t)

    if len(y_t) == 0 or (y_t == 1.0).sum() == 0:
        return 0.0

    # Sort descending by predicted probability
    desc_idx = np.argsort(-y_p)
    y_sorted = y_t[desc_idx]
    w_sorted = w[desc_idx]

    w_pos = w_sorted * (y_sorted == 1.0).astype(float)
    cum_tp = np.cumsum(w_pos)
    cum_total = np.cumsum(w_sorted)
    tot_pos = cum_tp[-1]

    if tot_pos <= 0:
        return 0.0

    recalls = np.divide(cum_tp, tot_pos, out=np.zeros_like(cum_tp), where=tot_pos > 0)
    precisions = np.divide(cum_tp, cum_total, out=np.zeros_like(cum_tp), where=cum_total > 0)

    # Trapezoidal approximation of PR curve with initial point (0, precisions[0] if len > 0 else 0)
    first_prec = precisions[0] if len(precisions) > 0 else 0.0
    rec_with_0 = np.concatenate(([0.0], recalls))
    prec_with_0 = np.concatenate(([first_prec], precisions))
    # Numerical integration using composite trapezoid rule
    return _trapezoid(prec_with_0, rec_with_0)


def weighted_brier_score(
    y_true: Sequence[float] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Compute survey-weighted Brier Score."""
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_prob, dtype=float).ravel()
    w = np.asarray(sample_weight, dtype=float).ravel() if sample_weight is not None else np.ones_like(y_t)

    tot_w = np.sum(w)
    if tot_w <= 0:
        return 0.0

    return float(np.sum(w * ((y_p - y_t) ** 2)) / tot_w)


def weighted_calibration_stats(
    y_true: Sequence[float] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    sample_weight: Sequence[float] | np.ndarray | None = None,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute survey-weighted calibration intercept, slope, and Expected Calibration Error (ECE)."""
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_prob, dtype=float).ravel()
    w = np.asarray(sample_weight, dtype=float).ravel() if sample_weight is not None else np.ones_like(y_t)

    tot_w = np.sum(w)
    if tot_w <= 0 or len(np.unique(y_t)) < 2:
        return {"intercept": 0.0, "slope": 1.0, "ece": 0.0}

    # Survey-weighted logistic calibration regression with light L2 regularization: logit(y) = a + b * logit(p)
    from sklearn.linear_model import LogisticRegression
    eps = 1e-7
    logit_p = np.log(np.clip(y_p, eps, 1 - eps) / (1 - np.clip(y_p, eps, 1 - eps))).reshape(-1, 1)

    cal_reg = LogisticRegression(penalty="l2", C=10.0, solver="lbfgs", max_iter=2000)
    try:
        cal_reg.fit(logit_p, y_t, sample_weight=w)
        slope = float(cal_reg.coef_[0][0])
        intercept = float(cal_reg.intercept_[0])
    except Exception:
        slope = 1.0
        intercept = 0.0

    # ECE computation across risk deciles
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        low, high = bin_edges[b], bin_edges[b + 1]
        in_bin = (y_p >= low) & (y_p <= high if b == n_bins - 1 else y_p < high)
        bin_w = w[in_bin].sum()
        if bin_w > 0:
            obs_rate = (w[in_bin] * y_t[in_bin]).sum() / bin_w
            pred_rate = (w[in_bin] * y_p[in_bin]).sum() / bin_w
            ece += (bin_w / tot_w) * abs(obs_rate - pred_rate)

    return {
        "intercept": intercept,
        "slope": slope,
        "ece": float(ece),
    }


def capacity_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    sample_weight: Sequence[float] | np.ndarray | None = None,
    capacity_fractions: Sequence[float] = (0.05, 0.10, 0.20),
) -> dict[str, float]:
    """Compute survey-weighted Recall and Precision (PPV) at fixed population capacity fractions."""
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_prob, dtype=float).ravel()
    w = np.asarray(sample_weight, dtype=float).ravel() if sample_weight is not None else np.ones_like(y_t)

    tot_w = np.sum(w)
    tot_pos_w = np.sum(w * (y_t == 1.0).astype(float))

    if tot_w <= 0 or tot_pos_w <= 0:
        return {f"recall_at_{int(c*100)}pct": 0.0 for c in capacity_fractions} | {
            f"precision_at_{int(c*100)}pct": 0.0 for c in capacity_fractions
        }

    desc_idx = np.argsort(-y_p)
    y_sorted = y_t[desc_idx]
    w_sorted = w[desc_idx]

    cum_w = np.cumsum(w_sorted)
    cum_pos_w = np.cumsum(w_sorted * (y_sorted == 1.0).astype(float))

    results: dict[str, float] = {}
    for cap in capacity_fractions:
        target_w = cap * tot_w
        idx = np.searchsorted(cum_w, target_w)
        idx = min(idx, len(y_sorted) - 1)

        cap_selected_pos_w = cum_pos_w[idx]
        cap_selected_tot_w = cum_w[idx]

        recall = float(cap_selected_pos_w / tot_pos_w) if tot_pos_w > 0 else 0.0
        precision = float(cap_selected_pos_w / cap_selected_tot_w) if cap_selected_tot_w > 0 else 0.0

        pct_label = int(cap * 100)
        results[f"recall_at_{pct_label}pct"] = recall
        results[f"precision_at_{pct_label}pct"] = precision

    return results


def fixed_threshold_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    threshold: float,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute binary classification metrics at a fixed decision threshold."""
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_prob, dtype=float).ravel()
    w = np.asarray(sample_weight, dtype=float).ravel() if sample_weight is not None else np.ones_like(y_t)

    pred_pos = (y_p >= threshold).astype(float)
    pred_neg = 1.0 - pred_pos
    true_pos = (y_t == 1.0).astype(float)
    true_neg = 1.0 - true_pos

    tp = np.sum(w * true_pos * pred_pos)
    fp = np.sum(w * true_neg * pred_pos)
    fn = np.sum(w * true_pos * pred_neg)
    tn = np.sum(w * true_neg * pred_neg)

    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    selection_rate = float((tp + fp) / np.sum(w)) if np.sum(w) > 0 else 0.0

    return {
        "tpr": tpr,
        "tnr": tnr,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "selection_rate": selection_rate,
    }


def subgroup_audit_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    audit_series: Sequence[Any] | pd.Series,
    sample_weight: Sequence[float] | np.ndarray | None = None,
    capacity: float = 0.10,
    min_n: int = 100,
    min_pos: int = 20,
    min_neg: int = 20,
    min_kish: float = 50.0,
) -> dict[str, Any]:
    """Compute subgroup error rates and capacity-aware disparities with strict suppression enforcement."""
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_prob, dtype=float).ravel()
    groups = pd.Series(np.asarray(audit_series)).reset_index(drop=True)
    w = np.asarray(sample_weight, dtype=float).ravel() if sample_weight is not None else np.ones_like(y_t)

    # Find overall threshold at target capacity (e.g. 10%)
    from meps_fairness.evaluation.calibration import find_weighted_capacity_threshold
    threshold = find_weighted_capacity_threshold(y_p, w, target_capacity=capacity)

    subgroups: dict[Any, dict[str, Any]] = {}
    valid_subgroup_tprs: dict[Any, float] = {}

    for g in sorted(groups.unique()):
        mask = (groups == g).values
        n_g = int(mask.sum())
        y_g = y_t[mask]
        p_g = y_p[mask]
        w_g = w[mask]

        pos_g = int((y_g == 1.0).sum())
        neg_g = int((y_g == 0.0).sum())
        sum_w_g = float(w_g.sum())
        sum_w_sq = float((w_g ** 2).sum())
        kish_g = float((sum_w_g ** 2) / sum_w_sq) if sum_w_sq > 0 else 0.0

        # Check suppression criteria
        is_suppressed = (n_g < min_n) or (pos_g < min_pos) or (neg_g < min_neg) or (kish_g < min_kish)

        if is_suppressed:
            subgroups[str(g)] = {
                "sample_n": n_g,
                "positives": pos_g,
                "negatives": neg_g,
                "sum_weights": sum_w_g,
                "kish_effective_n": kish_g,
                "status": "Suppressed (Insufficient Sample/Power)",
                "tpr_at_capacity": None,
                "ppv_at_capacity": None,
                "selection_rate": None,
            }
        else:
            fixed = fixed_threshold_metrics(y_g, p_g, threshold, w_g)
            subgroups[str(g)] = {
                "sample_n": n_g,
                "positives": pos_g,
                "negatives": neg_g,
                "sum_weights": sum_w_g,
                "kish_effective_n": kish_g,
                "status": "Unsuppressed",
                "tpr_at_capacity": fixed["tpr"],
                "ppv_at_capacity": fixed["ppv"],
                "selection_rate": fixed["selection_rate"],
            }
            valid_subgroup_tprs[str(g)] = fixed["tpr"]

    # Compute maximum absolute pairwise TPR gap across valid unsuppressed subgroups
    pairwise_gaps: list[float] = []
    subgroup_keys = list(valid_subgroup_tprs.keys())
    for i in range(len(subgroup_keys)):
        for j in range(i + 1, len(subgroup_keys)):
            gap = abs(valid_subgroup_tprs[subgroup_keys[i]] - valid_subgroup_tprs[subgroup_keys[j]])
            pairwise_gaps.append(gap)

    max_tpr_gap = float(max(pairwise_gaps)) if pairwise_gaps else None

    return {
        "capacity_threshold": threshold,
        "subgroups": subgroups,
        "max_tpr_gap": max_tpr_gap,
        "unsuppressed_subgroups_count": len(valid_subgroup_tprs),
    }


def primary_fairness_endpoint(
    y_true: Sequence[float] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    audit_df: pd.DataFrame,
    sample_weight: Sequence[float] | np.ndarray | None = None,
    capacity: float = 0.10,
) -> dict[str, Any]:
    """Compute primary fairness endpoint: maximum across primary audit dimensions of max pairwise TPR gap."""
    race_audit = subgroup_audit_metrics(
        y_true, y_prob, audit_df["RACETHX"], sample_weight, capacity=capacity
    )
    sex_audit = subgroup_audit_metrics(
        y_true, y_prob, audit_df["SEX"], sample_weight, capacity=capacity
    )

    max_tpr_gap_race = race_audit["max_tpr_gap"]
    max_tpr_gap_sex = sex_audit["max_tpr_gap"]
    valid_gaps = [g for g in (max_tpr_gap_race, max_tpr_gap_sex) if g is not None]
    primary_fairness_gap = float(max(valid_gaps)) if valid_gaps else None

    return {
        "primary_fairness_max_tpr_gap": primary_fairness_gap,
        "race_ethnicity": race_audit,
        "sex": sex_audit,
    }

