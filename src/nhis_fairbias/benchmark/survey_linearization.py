"""Additive Taylor linearization for complete annual survey designs.

This module is an approximate public-use-design covariance calculation.  It
keeps all annual rows and PSUs, including PSUs whose domain contribution is
zero, and is deliberately separate from the rescaled-PSU bootstrap engine.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from scipy import stats


@dataclasses.dataclass(frozen=True)
class LinearizedMetric:
    """One metric and its approximate simultaneous interval."""

    point_estimate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    status: str = "VALID"


def _domain(values: Optional[np.ndarray], n: int) -> np.ndarray:
    if values is None:
        return np.ones(n, dtype=bool)
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) != n or np.iscomplexobj(raw):
        raise ValueError("domain_mask must be a 1-dimensional 0/1 or boolean mask")
    if raw.dtype.kind == "b":
        return raw.copy()
    try:
        x = raw.astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("domain_mask must contain only 0/1 or boolean values") from exc
    if not np.all(np.isfinite(x)) or not np.all(np.isin(x, [0.0, 1.0])):
        raise ValueError("domain_mask must contain only 0/1 or boolean values")
    return x.astype(bool)


def _validate_groups(expected_groups: Sequence[int]) -> tuple[int, ...]:
    try:
        groups = tuple(expected_groups)
    except TypeError as exc:
        raise ValueError("expected_groups must be a nonempty sequence") from exc
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("expected_groups must contain unique group identifiers")
    out = []
    for group in groups:
        if isinstance(group, (bool, np.bool_)) or not np.isscalar(group):
            raise ValueError("expected_groups must contain scalar integer identifiers")
        try:
            value = float(group)
        except (TypeError, ValueError):
            raise ValueError("expected_groups must contain scalar integer identifiers")
        if not np.isfinite(value) or value != np.floor(value):
            raise ValueError("expected_groups must contain scalar integer identifiers")
        out.append(int(value))
    return tuple(out)


def _validate_design(y, predictions, A, strata, psus, weights, expected_groups, domain_mask):
    y, A, strata, psus, weights = map(np.asarray, (y, A, strata, psus, weights))
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    n = len(y)
    arrays = (A, strata, psus, weights)
    if any(x.ndim != 1 or len(x) != n for x in arrays):
        raise ValueError("survey arrays must be aligned one-dimensional arrays")
    if np.iscomplexobj(y) or not np.all(np.isfinite(y)) or not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must be finite binary values")
    for name, x in (("A", A), ("strata", strata), ("psus", psus)):
        if np.iscomplexobj(x) or not np.all(np.isfinite(x)) or not np.all(np.equal(x, np.floor(x))):
            raise ValueError(f"{name} must contain finite integer identifiers")
    raw_weights = np.asarray(weights)
    if np.iscomplexobj(raw_weights):
        raise ValueError("base survey weights must be real")
    weights = np.asarray(raw_weights, dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("base survey weights must be finite and strictly positive")
    groups = _validate_groups(expected_groups)
    domain = _domain(domain_mask, n)
    if not set(np.asarray(A, dtype=int)[domain]).issubset(set(groups)):
        raise ValueError("eligible A contains identifiers outside expected_groups")
    if not predictions:
        raise ValueError("predictions must contain at least one method")
    pred_out: Dict[str, np.ndarray] = {}
    for name, q in predictions.items():
        q = np.asarray(q)
        if q.ndim != 1 or len(q) != n or np.iscomplexobj(q):
            raise ValueError(f"prediction vector {name!r} must be aligned and real")
        q = q.astype(float)
        if not np.all(np.isfinite(q)) or np.any(q < 0) or np.any(q > 1):
            raise ValueError(f"prediction vector {name!r} must be finite in [0, 1]")
        pred_out[str(name)] = q
    if not np.any(domain):
        raise ValueError("domain_mask must contain at least one eligible row")
    return (y.astype(int), pred_out, A.astype(int), strata.astype(int), psus.astype(int), weights, groups, domain)


def _rate_and_score(q, denominator, weights, domain):
    mask = domain & denominator
    total = float(weights[mask].sum())
    if total <= 0:
        return np.nan, None
    rate = float(np.dot(weights[mask], q[mask]) / total)
    score = np.zeros(len(q), dtype=float)
    score[mask] = weights[mask] * (q[mask] - rate) / total
    return rate, score


def _psu_covariance(scores: np.ndarray, strata: np.ndarray, psus: np.ndarray) -> np.ndarray:
    """Covariance from centered PSU influence totals, retaining zero PSUs."""
    keys = [(h, p) for h in sorted(np.unique(strata))
            for p in sorted(np.unique(psus[strata == h]))]
    totals = np.zeros((len(keys), scores.shape[1]), dtype=float)
    for i, (h, p) in enumerate(keys):
        totals[i] = scores[(strata == h) & (psus == p)].sum(axis=0)
    covariance = np.zeros((scores.shape[1], scores.shape[1]), dtype=float)
    for h in sorted(np.unique(strata)):
        idx = [i for i, (hh, _) in enumerate(keys) if hh == h]
        centered = totals[idx] - totals[idx].mean(axis=0, keepdims=True)
        n_h = len(idx)
        covariance += (n_h / float(n_h - 1)) * centered.T @ centered
    return (covariance + covariance.T) / 2.0


def linearized_survey_inference(
    y_true: np.ndarray,
    predictions_dict: Mapping[str, np.ndarray],
    A: np.ndarray,
    strata: np.ndarray,
    psus: np.ndarray,
    weights: np.ndarray,
    expected_groups: Sequence[int],
    *,
    domain_mask: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    reference_method: str = "FAIRBIAS_BM",
) -> Dict[str, Any]:
    """Estimate annual domain metrics and Taylor covariance.

    Rate intervals and DP/EO projected intervals are asymptotic simultaneous
    Bonferroni-t bands.  They are not exact finite-sample survey guarantees.
    """
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must lie strictly between 0 and 1")
    y, predictions, A, strata, psus, weights, groups, domain = _validate_design(
        y_true, predictions_dict, A, strata, psus, weights, expected_groups, domain_mask
    )
    strata_values = sorted(np.unique(strata).tolist())
    psu_counts = {h: len(np.unique(psus[strata == h])) for h in strata_values}
    if any(count <= 1 for count in psu_counts.values()):
        return {"status": "DESIGN_NOT_ESTIMABLE", "reason": "complete design contains singleton stratum", "methods": {}}
    df = int(sum(count - 1 for count in psu_counts.values()))
    # Coordinate order is stable and shared by every method, enabling paired
    # contrasts without reusing independently calculated covariance matrices.
    labels = ["tpr", "fpr"] + [f"selection_g{g}" for g in groups]
    labels += [f"tpr_g{g}" for g in groups] + [f"fpr_g{g}" for g in groups]
    m_rates = len(labels)
    # The caller may project or average bands from different seed models.
    # Joint coverage therefore covers model x rate coordinates, not just
    # the coordinates within each independently considered model.
    tcrit = float(stats.t.ppf(1.0 - float(alpha) / (2.0 * m_rates * len(predictions)), df=df))
    methods: Dict[str, Any] = {}
    joint: Dict[str, Any] = {}
    for name, q in predictions.items():
        rates = []
        scores = []
        denominators = [y == 1, y == 0]
        denominators += [A == g for g in groups]
        denominators += [(A == g) & (y == 1) for g in groups]
        denominators += [(A == g) & (y == 0) for g in groups]
        for denominator in denominators:
            rate, score = _rate_and_score(q, denominator, weights, domain)
            rates.append(rate)
            scores.append(np.zeros(len(y)) if score is None else score)
        scores_matrix = np.column_stack(scores)
        covariance = _psu_covariance(scores_matrix, strata, psus)
        joint[name] = {"labels": labels, "covariance": covariance}
        rate_results: Dict[str, LinearizedMetric] = {}
        valid = True
        for i, label in enumerate(labels):
            se = float(np.sqrt(max(covariance[i, i], 0.0))) if np.isfinite(rates[i]) else np.nan
            if not np.isfinite(rates[i]):
                valid = False
                rate_results[label] = LinearizedMetric(np.nan, np.nan, np.nan, np.nan, "NOT_ESTIMABLE")
            else:
                half = tcrit * se
                rate_results[label] = LinearizedMetric(rates[i], se, max(0.0, rates[i] - half), min(1.0, rates[i] + half))
        def gap_interval(indices):
            if any(not np.isfinite(rates[i]) for i in indices):
                return (np.nan, np.nan, "NOT_ESTIMABLE")
            if len(indices) <= 1:
                return (0.0, 0.0, "VALID")
            lows = [rate_results[labels[i]].ci_lower for i in indices]
            highs = [rate_results[labels[i]].ci_upper for i in indices]
            return (float(np.clip(max(0.0, max(lows) - min(highs)), 0, 1)),
                    float(np.clip(max(highs) - min(lows), 0, 1)), "VALID")
        dp = gap_interval(list(range(2, 2 + len(groups))))
        eo_tpr = gap_interval(list(range(2 + len(groups), 2 + 2 * len(groups))))
        eo_fpr = gap_interval(list(range(2 + 2 * len(groups), m_rates)))
        if eo_tpr[2] == "VALID" and eo_fpr[2] == "VALID":
            eo = (max(eo_tpr[0], eo_fpr[0]), min(1.0, max(eo_tpr[1], eo_fpr[1])), "VALID")
        else:
            eo = (np.nan, np.nan, "NOT_ESTIMABLE")
        ba_c = np.zeros(m_rates); ba_c[0] = .5; ba_c[1] = -.5
        ba_valid = np.all(np.isfinite(rates[:2]))
        ba = .5 * rates[0] - .5 * rates[1] + .5 if ba_valid else np.nan
        ba_var = float(ba_c @ covariance @ ba_c) if ba_valid else np.nan
        ba_se = float(np.sqrt(max(ba_var, 0.0))) if np.isfinite(ba_var) else np.nan
        ba_tcrit = float(stats.t.ppf(1 - alpha / 2, df))
        ba_half = ba_tcrit * ba_se if np.isfinite(ba_se) else np.nan
        methods[name] = {"status": "VALID" if valid else "NOT_ESTIMABLE", "rates": rate_results,
                         "balanced_accuracy": LinearizedMetric(ba, ba_se,
                             max(0.0, ba - ba_half) if np.isfinite(ba_half) else np.nan,
                             min(1.0, ba + ba_half) if np.isfinite(ba_half) else np.nan,
                             "VALID" if ba_valid else "NOT_ESTIMABLE"),
                         "dp_gap_interval": dp, "eo_gap_interval": eo,
                         "degrees_of_freedom": df, "t_critical": tcrit}
    paired: Dict[str, Any] = {}
    ref = next((name for name in predictions if name.lower() == str(reference_method).lower()), None)
    if ref is not None:
        for base in predictions:
            if base == ref:
                continue
            c = np.zeros(m_rates); c[0] = .5; c[1] = -.5
            # Cross covariance is obtained from the shared row influence scores;
            # reconstructing it here keeps paired BA exact for identical methods.
            qdiff = predictions[ref] - predictions[base]
            r_tpr, s_tpr = _rate_and_score(qdiff, y == 1, weights, domain)
            r_fpr, s_fpr = _rate_and_score(qdiff, y == 0, weights, domain)
            diff_scores = np.column_stack([np.zeros(len(y)), np.zeros(len(y))])
            diff_scores[:, 0] = 0 if s_tpr is None else s_tpr
            diff_scores[:, 1] = 0 if s_fpr is None else s_fpr
            diff_cov = _psu_covariance(diff_scores, strata, psus)
            diff_ba = .5 * r_tpr - .5 * r_fpr if r_tpr == r_tpr and r_fpr == r_fpr else np.nan
            diff_se = float(np.sqrt(max(float(np.array([.5, -.5]) @ diff_cov @ np.array([.5, -.5])), 0.0))) if np.isfinite(diff_ba) else np.nan
            half = float(stats.t.ppf(1 - alpha / 2, df)) * diff_se if np.isfinite(diff_se) else np.nan
            ref_eo = methods[ref]["eo_gap_interval"]
            base_eo = methods[base]["eo_gap_interval"]
            if ref_eo[2] == "VALID" and base_eo[2] == "VALID" and np.array_equal(predictions[ref], predictions[base]):
                paired_eo = (0.0, 0.0, "VALID")
            elif ref_eo[2] == "VALID" and base_eo[2] == "VALID":
                paired_eo = (float(np.clip(ref_eo[0] - base_eo[1], -1.0, 1.0)),
                             float(np.clip(ref_eo[1] - base_eo[0], -1.0, 1.0)), "VALID")
            else:
                paired_eo = (np.nan, np.nan, "NOT_ESTIMABLE")
            paired[base] = {"reference_method": ref, "balanced_accuracy": LinearizedMetric(diff_ba, diff_se,
                diff_ba - half if np.isfinite(half) else np.nan, diff_ba + half if np.isfinite(half) else np.nan,
                "VALID" if np.isfinite(diff_ba) else "NOT_ESTIMABLE"),
                "eo_gap_interval": paired_eo,
                "status": "VALID" if np.isfinite(diff_ba) else "NOT_ESTIMABLE"}
    return {"status": "VALID", "methods": methods, "paired": paired, "joint_covariance": joint,
            "degrees_of_freedom": df, "domain_n": int(domain.sum()), "domain_weight_sum": float(np.dot(weights, domain)),
            "interval_label": "Rate and DP/EO intervals are asymptotic simultaneous Bonferroni-t projections; BA intervals are individual t intervals at supplied alpha",
            "alpha": float(alpha)}
