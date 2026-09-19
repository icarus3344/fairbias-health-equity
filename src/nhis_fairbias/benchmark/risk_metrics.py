"""Risk metrics use event probability p, never a randomized policy's q."""
from typing import Any, Optional, Sequence
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from .predictions import PredictionBundle, PredictionContractError


def compute_risk_metrics(y: Any, prediction: PredictionBundle, weights: Any,
                         A: Optional[Any] = None, expected_groups: Optional[Sequence[int]] = None) -> dict:
    if not isinstance(prediction, PredictionBundle):
        raise PredictionContractError("Risk evaluation requires an explicit PredictionBundle")
    if prediction.p_event is None:
        return {"status": "NOT_AVAILABLE", "reason": "Method supplies decision probability only",
                "average_precision": None, "auroc": None, "brier": None, "calibration": None}
    labels = np.asarray(y)
    w = np.asarray(weights, dtype=float)
    p = prediction.risk_probability()
    if labels.ndim != 1 or w.ndim != 1 or len(labels) != len(p) or len(w) != len(p):
        raise ValueError("Risk labels, weights and p must be aligned vectors")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Risk labels must be binary and weights finite/nonnegative")
    positive = w > 0
    if not positive.any():
        return {"status": "NOT_ESTIMABLE", "average_precision": None, "auroc": None,
                "brier": None, "calibration": None}
    w = w / w.max()
    two_classes = len(np.unique(labels[positive])) == 2
    result = {
        "status": "VALID" if two_classes else "RANKING_NOT_ESTIMABLE",
        "average_precision": float(average_precision_score(labels, p, sample_weight=w)) if two_classes else None,
        "auroc": float(roc_auc_score(labels, p, sample_weight=w)) if two_classes else None,
        "brier": float(np.average((labels - p) ** 2, weights=w)),
        "calibration_bins": "fixed_deciles_0_to_1",
    }
    # Fixed bins are a reporting convention, not fitted probability calibration.
    bins = np.minimum((p * 10).astype(int), 9)
    curve = []
    for k in range(10):
        mask = (bins == k) & positive
        weight = float(w[mask].sum())
        curve.append({"lower": k / 10, "upper": (k + 1) / 10, "n": int(mask.sum()),
            "weight_fraction": weight / float(w.sum()),
            "predicted_mean": float(np.average(p[mask], weights=w[mask])) if weight else None,
            "observed_rate": float(np.average(labels[mask], weights=w[mask])) if weight else None})
    result["calibration"] = curve
    if A is not None:
        groups = np.asarray(A)
        if groups.ndim != 1 or len(groups) != len(p):
            raise ValueError("Risk group labels are not aligned")
        expected = list(expected_groups) if expected_groups is not None else np.unique(groups[positive]).tolist()
        result["group_metrics"] = {}
        for g in expected:
            mask = groups == g
            sub = PredictionBundle(p[mask], prediction.q_decision[mask])
            result["group_metrics"][str(g)] = compute_risk_metrics(labels[mask], sub, w[mask])
    return result
