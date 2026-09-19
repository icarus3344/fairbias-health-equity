"""Design-preserving rescaled PSU bootstrap inference for NHIS benchmarks."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from .metrics import compute_survey_fairness_metrics


class DesignNotEstimableError(ValueError):
    """The complete annual design cannot support the requested variance estimate."""


@dataclasses.dataclass(frozen=True)
class MetricInferenceResult:
    metric_name: str
    point_estimate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    df: int
    replicates_valid: int
    replicates_total: int
    status: str = "VALID"
    valid_fraction: float = 1.0


@dataclasses.dataclass(frozen=True)
class PairedContrastResult:
    metric_name: str
    baseline_name: str
    comparison_name: str
    point_contrast: float
    std_error: float
    ci_lower: float
    ci_upper: float
    df: int
    p_value: float
    replicates_valid: int = 0
    replicates_total: int = 0
    status: str = "VALID"


class RescaledPSUBootstrapEngine:
    """Generate shared Rao-Wu-style rescaled PSU replicate weights."""

    def __init__(self, strata: np.ndarray, psus: np.ndarray, weights: np.ndarray, *, seed: int = 20260914):
        s_raw, p_raw = np.asarray(strata), np.asarray(psus)
        self.weights = np.asarray(weights, dtype=float)
        if s_raw.ndim != 1 or p_raw.ndim != 1 or self.weights.ndim != 1:
            raise ValueError("strata, psus, and weights must be 1-dimensional")
        if not (len(s_raw) == len(p_raw) == len(self.weights)):
            raise ValueError("Lengths of strata, psus, and weights must match.")
        if not np.all(np.isfinite(s_raw)) or not np.all(np.isfinite(p_raw)):
            raise ValueError("strata and psus must be finite")
        if not np.all(np.equal(s_raw, np.floor(s_raw))) or not np.all(np.equal(p_raw, np.floor(p_raw))):
            raise ValueError("strata and psus must contain integer identifiers")
        if not np.all(np.isfinite(self.weights)) or np.any(self.weights <= 0):
            raise ValueError("base survey weights must be finite and strictly positive")
        self.strata, self.psus = s_raw.astype(int), p_raw.astype(int)
        self.seed = int(seed)
        self.unique_strata = sorted(np.unique(self.strata).tolist())
        self.H = len(self.unique_strata)
        self.stratum_psu_map: Dict[int, List[int]] = {}
        self.psu_row_indices: Dict[Tuple[int, int], np.ndarray] = {}
        for h in self.unique_strata:
            psu_list = sorted(np.unique(self.psus[self.strata == h]).tolist())
            self.stratum_psu_map[h] = psu_list
            for p in psu_list:
                self.psu_row_indices[(h, p)] = np.where((self.strata == h) & (self.psus == p))[0]
        self.total_psus = sum(len(v) for v in self.stratum_psu_map.values())
        self.df = self.total_psus - self.H
        if any(len(v) <= 1 for v in self.stratum_psu_map.values()) or self.df <= 0:
            raise DesignNotEstimableError("complete design contains singleton stratum or nonpositive df")

    def generate_replicate_weights(self, B: int) -> np.ndarray:
        if int(B) <= 0:
            raise ValueError("B must be positive")
        rng = np.random.default_rng(self.seed)
        out = np.zeros((int(B), len(self.weights)), dtype=float)
        for b in range(int(B)):
            rep = self.weights.copy()
            for h, psu_list in self.stratum_psu_map.items():
                n_h = len(psu_list)
                draw = rng.choice(n_h, size=n_h - 1, replace=True)
                counts = np.bincount(draw, minlength=n_h)
                factor = n_h / float(n_h - 1)
                for idx, p in enumerate(psu_list):
                    rows = self.psu_row_indices[(h, p)]
                    rep[rows] = self.weights[rows] * factor * counts[idx]
            out[b] = rep
        return out

    def iter_replicate_weights(self, B: int):
        """Yield replicate vectors one at a time to bound production memory."""
        if int(B) <= 0:
            raise ValueError("B must be positive")
        rng = np.random.default_rng(self.seed)
        for _ in range(int(B)):
            rep = self.weights.copy()
            for h, psu_list in self.stratum_psu_map.items():
                n_h = len(psu_list)
                counts = np.bincount(rng.choice(n_h, size=n_h - 1, replace=True), minlength=n_h)
                factor = n_h / float(n_h - 1)
                for idx, p in enumerate(psu_list):
                    rows = self.psu_row_indices[(h, p)]
                    rep[rows] = self.weights[rows] * factor * counts[idx]
            yield rep


def _not_estimable_result(name: str, point: float, df: int, valid: int, total: int) -> MetricInferenceResult:
    return MetricInferenceResult(name, point, np.nan, np.nan, np.nan, df, valid, total, "NOT_ESTIMABLE", valid / float(total))


def evaluate_with_survey_bootstrap(
    y_true: np.ndarray, predictions_dict: Dict[str, np.ndarray], A: np.ndarray,
    strata: np.ndarray, psus: np.ndarray, weights: np.ndarray,
    expected_groups: Sequence[int], *, B: int = 2000, seed: int = 20260914,
    alpha: float = 0.05, domain_mask: Optional[np.ndarray] = None,
    reference_method: str = "FAIRBIAS_BM",
) -> Dict[str, Any]:
    """Evaluate full annual design arrays with optional domain restriction."""
    y, a, s, p, w = map(np.asarray, (y_true, A, strata, psus, weights))
    n = len(y)
    if any(x.ndim != 1 or len(x) != n for x in (a, s, p, w)):
        raise ValueError("all survey arrays must be aligned 1-dimensional arrays")
    if isinstance(B, bool) or not isinstance(B, (int, np.integer)) or int(B) < 2:
        raise ValueError("B must be an integer >= 2")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must lie strictly between 0 and 1")
    if domain_mask is None:
        domain = np.ones(n, dtype=bool)
    else:
        raw_domain = np.asarray(domain_mask)
        if raw_domain.ndim != 1 or len(raw_domain) != n or np.iscomplexobj(raw_domain):
            raise ValueError("domain_mask must be a 1-dimensional 0/1 or boolean mask")
        if raw_domain.dtype.kind == "b":
            domain = raw_domain.copy()
        else:
            try:
                dm = raw_domain.astype(float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("domain_mask must contain only 0/1 or boolean values") from exc
            if not np.all(np.isfinite(dm)) or not np.all(np.isin(dm, [0.0, 1.0])):
                raise ValueError("domain_mask must contain only 0/1 or boolean values")
            domain = dm.astype(bool)
    if domain.ndim != 1 or len(domain) != n or not np.any(domain):
        return {"status": "DESIGN_NOT_ESTIMABLE", "reason": "invalid or empty domain", "point_estimates": {}, "method_inference": {}, "paired_contrasts": {}}
    try:
        engine = RescaledPSUBootstrapEngine(s, p, w, seed=seed)
    except DesignNotEstimableError as exc:
        return {"status": "DESIGN_NOT_ESTIMABLE", "reason": str(exc), "degrees_of_freedom": 0, "point_estimates": {}, "method_inference": {}, "paired_contrasts": {}}
    reps = engine.iter_replicate_weights(B)
    domain_w = w * domain
    point = {name: compute_survey_fairness_metrics(y, np.asarray(pred), a, domain_w, expected_groups=expected_groups) for name, pred in predictions_dict.items()}
    keys = ("balanced_accuracy", "dp_gap", "eo_gap")
    rep_metrics = {name: {k: [] for k in keys} for name in predictions_dict}
    for rep in reps:
        w_b = rep * domain
        for name, pred in predictions_dict.items():
            m = compute_survey_fairness_metrics(y, np.asarray(pred), a, w_b, expected_groups=expected_groups)
            for k in keys:
                rep_metrics[name][k].append(m[k])
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=engine.df))
    method = {}
    for name in predictions_dict:
        method[name] = {}
        for k in keys:
            vals = np.asarray(rep_metrics[name][k], dtype=float)
            valid = vals[np.isfinite(vals)]
            b_valid = len(valid)
            pt = float(point[name][k])
            if b_valid > 1 and b_valid / float(B) >= 0.95 and np.isfinite(pt):
                se = float(np.std(valid, ddof=1))
                result = MetricInferenceResult(k, pt, se, pt - t_crit * se, pt + t_crit * se, engine.df, b_valid, B, "VALID", b_valid / float(B))
            else:
                result = _not_estimable_result(k, pt, engine.df, b_valid, B)
            method[name][k] = result
    paired = {}
    fair = reference_method if reference_method in predictions_dict else next((name for name in predictions_dict if name.lower() == reference_method.lower()), None)
    if fair is not None:
        for base in predictions_dict:
            if base == fair:
                continue
            paired[base] = {}
            for k in ("balanced_accuracy", "eo_gap"):
                fb, bb = np.asarray(rep_metrics[fair][k]), np.asarray(rep_metrics[base][k])
                mask = np.isfinite(fb) & np.isfinite(bb)
                diffs = fb[mask] - bb[mask]
                pt = float(point[fair][k] - point[base][k])
                if len(diffs) > 1 and len(diffs) / float(B) >= 0.95 and np.isfinite(pt):
                    se = float(np.std(diffs, ddof=1)); ci_l, ci_u = pt - t_crit * se, pt + t_crit * se
                    if se > 0:
                        pval = float(2 * (1 - stats.t.cdf(abs(pt / se), df=engine.df)))
                        status = "VALID"
                    elif pt == 0:
                        pval = 1.0
                        status = "VALID"
                    else:
                        pval = np.nan
                        status = "NOT_ESTIMABLE"
                else:
                    se = ci_l = ci_u = pval = np.nan; status = "NOT_ESTIMABLE"
                paired[base][k] = PairedContrastResult(k, base, fair, pt, se, ci_l, ci_u, engine.df, pval, len(diffs), B, status)
    return {"status": "VALID", "point_estimates": point, "method_inference": method, "paired_contrasts": paired,
            "degrees_of_freedom": engine.df, "design": {"strata": engine.H, "psus": engine.total_psus,
            "domain_n": int(domain.sum()), "domain_weight_sum": float(domain_w.sum())}}
