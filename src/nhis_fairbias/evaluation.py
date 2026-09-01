"""Substantive evaluation metrics, group-level rates, and fairness gaps for NHIS FairBias.

Guarantees:
1. Utility metrics:
   - AUROC (predicted continuous probabilities, never hard labels)
   - AUPRC / average precision (predicted continuous probabilities)
   - Balanced accuracy
   - F1
   - Accuracy
   - Undefined metrics return None with explicit diagnostic reasons (never silently zero).
2. Group-level metrics:
   - For each protected group: n, outcome counts, predicted counts, prevalence,
     selection rate, TPR, FPR, PPV.
   - Undefined conditional rates remain None/NaN with explicit denominator records.
   - Never silently sets undefined TPR/FPR/PPV to zero.
3. Primary application fairness gaps:
   - demographic_parity_gap = max_g(selection_rate_g) - min_g(selection_rate_g)
   - equal_opportunity_gap = max_g(TPR_g) - min_g(TPR_g)
   - fpr_gap = max_g(FPR_g) - min_g(FPR_g)
   - equalized_odds_max_gap = max(equal_opportunity_gap, fpr_gap)
   - Records defined groups for each gap.
4. Multicategory extension:
   - For K categories (e.g. HISPALLP_A K=7), emits all K*(K-1)/2 unordered pairs (e.g. exactly 21 pairs).
   - Reports pairwise differences for selection rate, TPR, FPR, PPV where defined.
   - Labeled as 'empirical_multicategory_extension_21_pairs'.
"""

from __future__ import annotations

import copy
from itertools import combinations
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def compute_utility_metrics(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    y_prob: Union[Sequence[float], np.ndarray, pd.Series],
) -> Dict[str, Any]:
    """
    Compute utility metrics: AUROC, AUPRC, balanced accuracy, F1, accuracy.

    Probabilities are required for AUROC and AUPRC.
    Mathematically undefined metrics evaluate to None with diagnostic reasons.
    """
    y_t = np.asarray(y_true, dtype=int).ravel()
    y_p = np.asarray(y_pred, dtype=int).ravel()
    y_s = np.asarray(y_prob, dtype=float).ravel()

    n = len(y_t)
    if len(y_p) != n or len(y_s) != n:
        raise ValueError(
            f"Dimension mismatch in evaluation inputs: y_true={n}, y_pred={len(y_p)}, y_prob={len(y_s)}"
        )

    results: Dict[str, Optional[float]] = {
        "auroc": None,
        "auprc": None,
        "balanced_accuracy": None,
        "f1": None,
        "accuracy": None,
    }
    undefined_reasons: Dict[str, str] = {}

    if n == 0:
        for k in results:
            undefined_reasons[k] = "Empty sample array (N = 0)"
        return {
            **results,
            "undefined_reasons": undefined_reasons,
        }

    # Accuracy
    try:
        results["accuracy"] = float(accuracy_score(y_t, y_p))
    except Exception as exc:
        results["accuracy"] = None
        undefined_reasons["accuracy"] = f"Calculation error: {exc}"

    # Unique classes in ground truth
    unique_classes = np.unique(y_t)
    n_classes = len(unique_classes)

    # AUROC
    if n_classes < 2:
        results["auroc"] = None
        undefined_reasons["auroc"] = (
            f"AUROC undefined: ground truth contains only 1 class ({unique_classes.tolist()})"
        )
    elif np.isnan(y_s).any() or np.isinf(y_s).any():
        results["auroc"] = None
        undefined_reasons["auroc"] = "AUROC undefined: predicted probabilities contain NaN or Inf"
    else:
        try:
            results["auroc"] = float(roc_auc_score(y_t, y_s))
        except Exception as exc:
            results["auroc"] = None
            undefined_reasons["auroc"] = f"AUROC calculation error: {exc}"

    # AUPRC / Average Precision
    pos_count = int(np.sum(y_t == 1))
    if pos_count == 0:
        results["auprc"] = None
        undefined_reasons["auprc"] = (
            "AUPRC undefined: no positive ground-truth instances present (pos_count = 0)"
        )
    elif np.isnan(y_s).any() or np.isinf(y_s).any():
        results["auprc"] = None
        undefined_reasons["auprc"] = "AUPRC undefined: predicted probabilities contain NaN or Inf"
    else:
        try:
            results["auprc"] = float(average_precision_score(y_t, y_s))
        except Exception as exc:
            results["auprc"] = None
            undefined_reasons["auprc"] = f"AUPRC calculation error: {exc}"

    # Balanced Accuracy
    if n_classes < 2:
        results["balanced_accuracy"] = None
        undefined_reasons["balanced_accuracy"] = (
            f"Balanced accuracy undefined: ground truth contains only 1 class ({unique_classes.tolist()})"
        )
    else:
        try:
            results["balanced_accuracy"] = float(balanced_accuracy_score(y_t, y_p))
        except Exception as exc:
            results["balanced_accuracy"] = None
            undefined_reasons["balanced_accuracy"] = f"Balanced accuracy calculation error: {exc}"

    # F1 Score
    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    denom_f1 = 2 * tp + fp + fn
    if denom_f1 == 0:
        results["f1"] = None
        undefined_reasons["f1"] = (
            "F1 undefined: denominator 2*TP + FP + FN is zero (no positive labels or predictions)"
        )
    else:
        results["f1"] = float(2 * tp / denom_f1)

    return {
        **results,
        "undefined_reasons": undefined_reasons,
    }


def compute_group_metrics(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    o_group: Union[Sequence[Any], np.ndarray, pd.Series],
) -> List[Dict[str, Any]]:
    """
    Compute raw group-level metrics for every protected attribute category:
    n, outcome counts, predicted counts, prevalence, selection rate, TPR, FPR, PPV.

    Undefined conditional rates are returned as None with explicit denominator records.
    """
    y_t = np.asarray(y_true, dtype=int).ravel()
    y_p = np.asarray(y_pred, dtype=int).ravel()
    o_g = np.asarray(o_group).ravel()

    n = len(y_t)
    if len(y_p) != n or len(o_g) != n:
        raise ValueError(
            f"Dimension mismatch in group metrics inputs: y_true={n}, y_pred={len(y_p)}, o_group={len(o_g)}"
        )

    unique_groups = sorted(list(pd.Series(o_g).dropna().unique()), key=lambda x: str(x))
    group_rows: List[Dict[str, Any]] = []

    for grp in unique_groups:
        mask = (o_g == grp)
        n_g = int(np.sum(mask))

        if n_g == 0:
            continue

        y_t_g = y_t[mask]
        y_p_g = y_p[mask]

        pos_count = int(np.sum(y_t_g == 1))
        neg_count = int(np.sum(y_t_g == 0))
        pred_pos_count = int(np.sum(y_p_g == 1))
        pred_neg_count = int(np.sum(y_p_g == 0))

        tp = int(np.sum((y_t_g == 1) & (y_p_g == 1)))
        fp = int(np.sum((y_t_g == 0) & (y_p_g == 1)))
        tn = int(np.sum((y_t_g == 0) & (y_p_g == 0)))
        fn = int(np.sum((y_t_g == 1) & (y_p_g == 0)))

        prevalence = float(pos_count / n_g) if n_g > 0 else None
        selection_rate = float(pred_pos_count / n_g) if n_g > 0 else None

        tpr_denom = tp + fn  # pos_count
        fpr_denom = fp + tn  # neg_count
        ppv_denom = tp + fp  # pred_pos_count

        tpr = float(tp / tpr_denom) if tpr_denom > 0 else None
        fpr = float(fp / fpr_denom) if fpr_denom > 0 else None
        ppv = float(tp / ppv_denom) if ppv_denom > 0 else None

        undefined_reasons: Dict[str, str] = {}
        if tpr is None:
            undefined_reasons["tpr"] = f"Undefined: outcome-positive denominator (TP + FN) = {tpr_denom}"
        if fpr is None:
            undefined_reasons["fpr"] = f"Undefined: outcome-negative denominator (FP + TN) = {fpr_denom}"
        if ppv is None:
            undefined_reasons["ppv"] = f"Undefined: predicted-positive denominator (TP + FP) = {ppv_denom}"

        # Standard representation for row
        row: Dict[str, Any] = {
            "group": int(grp) if isinstance(grp, (int, np.integer)) else str(grp),
            "n": n_g,
            "outcome_positive_count": pos_count,
            "outcome_negative_count": neg_count,
            "predicted_positive_count": pred_pos_count,
            "predicted_negative_count": pred_neg_count,
            "tp_count": tp,
            "fp_count": fp,
            "tn_count": tn,
            "fn_count": fn,
            "prevalence": prevalence,
            "selection_rate": selection_rate,
            "tpr": tpr,
            "fpr": fpr,
            "ppv": ppv,
            "tpr_denominator": tpr_denom,
            "fpr_denominator": fpr_denom,
            "ppv_denominator": ppv_denom,
            "undefined_reasons": undefined_reasons,
        }
        group_rows.append(row)

    return group_rows


def compute_fairness_gaps(group_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute primary application fairness gaps:
    - demographic_parity_gap = max_g(selection_rate_g) - min_g(selection_rate_g)
    - equal_opportunity_gap = max_g(TPR_g) - min_g(TPR_g)
    - fpr_gap = max_g(FPR_g) - min_g(FPR_g)
    - equalized_odds_max_gap = max(equal_opportunity_gap, fpr_gap)

    Records defined groups and PPV group values.
    """
    sr_defined = [
        (g["group"], g["selection_rate"])
        for g in group_metrics
        if g.get("selection_rate") is not None
    ]
    tpr_defined = [
        (g["group"], g["tpr"])
        for g in group_metrics
        if g.get("tpr") is not None
    ]
    fpr_defined = [
        (g["group"], g["fpr"])
        for g in group_metrics
        if g.get("fpr") is not None
    ]
    ppv_defined = [
        (g["group"], g["ppv"])
        for g in group_metrics
        if g.get("ppv") is not None
    ]

    dp_gap = None
    if len(sr_defined) >= 2:
        sr_vals = [v for _, v in sr_defined]
        dp_gap = float(max(sr_vals) - min(sr_vals))

    eq_opp_gap = None
    if len(tpr_defined) >= 2:
        tpr_vals = [v for _, v in tpr_defined]
        eq_opp_gap = float(max(tpr_vals) - min(tpr_vals))

    fpr_gap = None
    if len(fpr_defined) >= 2:
        fpr_vals = [v for _, v in fpr_defined]
        fpr_gap = float(max(fpr_vals) - min(fpr_vals))

    eq_odds_max = None
    if eq_opp_gap is not None and fpr_gap is not None:
        eq_odds_max = float(max(eq_opp_gap, fpr_gap))
    elif eq_opp_gap is not None:
        eq_odds_max = float(eq_opp_gap)
    elif fpr_gap is not None:
        eq_odds_max = float(fpr_gap)

    return {
        "demographic_parity_gap": dp_gap,
        "equal_opportunity_gap": eq_opp_gap,
        "fpr_gap": fpr_gap,
        "equalized_odds_max_gap": eq_odds_max,
        "group_ppv_values": {str(g["group"]): g.get("ppv") for g in group_metrics},
        "defined_groups": {
            "selection_rate": [g for g, _ in sr_defined],
            "tpr": [g for g, _ in tpr_defined],
            "fpr": [g for g, _ in fpr_defined],
            "ppv": [g for g, _ in ppv_defined],
        },
    }


def compute_multicategory_pairwise_differences(
    group_metrics: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute all unordered pairwise differences across protected categories.

    For K categories, emits exactly K*(K-1)/2 pairs.
    For K=7 (e.g. HISPALLP_A), emits exactly 21 unordered group-pair differences for:
    - selection rate
    - TPR
    - FPR
    - PPV where defined

    Labeled explicitly as 'empirical_multicategory_extension_21_pairs'.
    """
    group_map = {g["group"]: g for g in group_metrics}
    groups = sorted(list(group_map.keys()), key=lambda x: str(x))
    k = len(groups)
    expected_pairs = k * (k - 1) // 2

    pair_records: List[Dict[str, Any]] = []

    for g1, g2 in combinations(groups, 2):
        d1 = group_map[g1]
        d2 = group_map[g2]

        sr1, sr2 = d1.get("selection_rate"), d2.get("selection_rate")
        tpr1, tpr2 = d1.get("tpr"), d2.get("tpr")
        fpr1, fpr2 = d1.get("fpr"), d2.get("fpr")
        ppv1, ppv2 = d1.get("ppv"), d2.get("ppv")

        sr_diff = float(abs(sr1 - sr2)) if (sr1 is not None and sr2 is not None) else None
        tpr_diff = float(abs(tpr1 - tpr2)) if (tpr1 is not None and tpr2 is not None) else None
        fpr_diff = float(abs(fpr1 - fpr2)) if (fpr1 is not None and fpr2 is not None) else None
        ppv_diff = float(abs(ppv1 - ppv2)) if (ppv1 is not None and ppv2 is not None) else None

        pair_records.append({
            "pair": [g1, g2],
            "pair_key": f"{g1}_vs_{g2}",
            "selection_rate_diff": sr_diff,
            "tpr_diff": tpr_diff,
            "fpr_diff": fpr_diff,
            "ppv_diff": ppv_diff,
        })

    def _max_valid(vals: List[Optional[float]]) -> Optional[float]:
        valid = [v for v in vals if v is not None]
        return float(max(valid)) if valid else None

    max_pair_gaps = {
        "selection_rate_max_pair_diff": _max_valid([p["selection_rate_diff"] for p in pair_records]),
        "tpr_max_pair_diff": _max_valid([p["tpr_diff"] for p in pair_records]),
        "fpr_max_pair_diff": _max_valid([p["fpr_diff"] for p in pair_records]),
        "ppv_max_pair_diff": _max_valid([p["ppv_diff"] for p in pair_records]),
    }

    return {
        "formulation_label": "empirical_multicategory_extension_21_pairs",
        "num_groups": k,
        "num_pairs": len(pair_records),
        "expected_pairs_for_k": expected_pairs,
        "max_pair_gaps": max_pair_gaps,
        "pairs": pair_records,
    }


def evaluate_predictions(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    y_prob: Union[Sequence[float], np.ndarray, pd.Series],
    o_group: Union[Sequence[Any], np.ndarray, pd.Series],
) -> Dict[str, Any]:
    """
    Combined evaluation returning utility metrics, group-level metrics, and fairness gaps.
    """
    utility = compute_utility_metrics(y_true, y_pred, y_prob)
    group_rows = compute_group_metrics(y_true, y_pred, o_group)
    fairness_gaps = compute_fairness_gaps(group_rows)

    result: Dict[str, Any] = {
        "utility": utility,
        "fairness_gaps": fairness_gaps,
        "group_metrics": group_rows,
    }

    if len(group_rows) > 2:
        result["multicategory_pairwise"] = compute_multicategory_pairwise_differences(group_rows)

    return result


def compute_evaluation_comparison(
    baseline_eval: Dict[str, Any],
    fairbias_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute paired delta comparison: FairBias minus Baseline.
    Positive delta means FairBias is higher; negative means FairBias is lower.
    """
    base_util = baseline_eval.get("utility", {})
    fb_util = fairbias_eval.get("utility", {})

    utility_deltas: Dict[str, Optional[float]] = {}
    for metric_name in ("auroc", "auprc", "balanced_accuracy", "f1", "accuracy"):
        bv = base_util.get(metric_name)
        fv = fb_util.get(metric_name)
        if bv is not None and fv is not None:
            utility_deltas[metric_name] = float(fv - bv)
        else:
            utility_deltas[metric_name] = None

    base_gaps = baseline_eval.get("fairness_gaps", {})
    fb_gaps = fairbias_eval.get("fairness_gaps", {})

    gap_deltas: Dict[str, Optional[float]] = {}
    for gap_name in ("demographic_parity_gap", "equal_opportunity_gap", "fpr_gap", "equalized_odds_max_gap"):
        bg = base_gaps.get(gap_name)
        fg = fb_gaps.get(gap_name)
        if bg is not None and fg is not None:
            gap_deltas[gap_name] = float(fg - bg)
        else:
            gap_deltas[gap_name] = None

    # Group-level rate deltas
    base_grp_map = {str(g["group"]): g for g in baseline_eval.get("group_metrics", [])}
    fb_grp_map = {str(g["group"]): g for g in fairbias_eval.get("group_metrics", [])}

    group_deltas: Dict[str, Dict[str, Optional[float]]] = {}
    all_groups = sorted(list(set(base_grp_map.keys()) | set(fb_grp_map.keys())))

    for grp in all_groups:
        bg = base_grp_map.get(grp, {})
        fg = fb_grp_map.get(grp, {})
        group_deltas[grp] = {}
        for r_name in ("selection_rate", "tpr", "fpr", "ppv"):
            b_val = bg.get(r_name)
            f_val = fg.get(r_name)
            if b_val is not None and f_val is not None:
                group_deltas[grp][r_name] = float(f_val - b_val)
            else:
                group_deltas[grp][r_name] = None

    return {
        "utility_deltas": utility_deltas,
        "fairness_gap_deltas": gap_deltas,
        "group_deltas": group_deltas,
        "baseline_summary": {
            "utility": {k: base_util.get(k) for k in ("auroc", "auprc", "balanced_accuracy", "f1", "accuracy")},
            "fairness_gaps": {k: base_gaps.get(k) for k in ("demographic_parity_gap", "equal_opportunity_gap", "fpr_gap", "equalized_odds_max_gap")},
        },
        "fairbias_summary": {
            "utility": {k: fb_util.get(k) for k in ("auroc", "auprc", "balanced_accuracy", "f1", "accuracy")},
            "fairness_gaps": {k: fb_gaps.get(k) for k in ("demographic_parity_gap", "equal_opportunity_gap", "fpr_gap", "equalized_odds_max_gap")},
        },
    }
