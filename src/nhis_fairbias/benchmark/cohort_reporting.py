"""Aggregate-only cohort and survey-design reporting.

The functions here consume already materialized ``PartitionDataset``-like
objects.  They never load data, fit preprocessing, or return record keys.
Raw semantic missingness is reported only when the partition explicitly
provides an aggregate ``metadata['raw_feature_missingness']`` payload.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def _vector(partition: Any, name: str) -> np.ndarray:
    value = np.asarray(getattr(partition, name))
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return value


def _design_coverage(partition: Any, n: int) -> dict[str, Any]:
    design = getattr(partition, "annual_design", None)
    if design is not None:
        strata = np.asarray(design.strata)
        psus = np.asarray(design.psus)
        design_rows = len(strata)
    else:
        strata = _vector(partition, "PSTRAT")
        psus = _vector(partition, "PPSU")
        design_rows = n
    if strata.ndim != 1 or psus.ndim != 1 or len(strata) != len(psus):
        raise ValueError("strata and PSU vectors must be aligned")
    keys = {(int(h), int(p)) for h, p in zip(strata, psus)}
    domain_strata = _vector(partition, "PSTRAT")
    domain_psus = _vector(partition, "PPSU")
    domain_keys = {(int(h), int(p)) for h, p in zip(domain_strata, domain_psus)}
    return {
        "design_rows": int(design_rows),
        "strata_count": int(len(np.unique(strata))),
        "psu_count": int(len(keys)),
        "strata_psu_coverage": [
            {"stratum": int(h), "psu_count": int(sum(k[0] == h for k in keys))}
            for h in sorted({k[0] for k in keys})
        ],
        "domain_rows": int(n),
        "domain_strata_count": int(len(np.unique(domain_strata))),
        "domain_psu_count": int(len(domain_keys)),
        "domain_strata_psu_coverage": [
            {"stratum": int(h), "psu_count": int(sum(k[0] == h for k in domain_keys))}
            for h in sorted({k[0] for k in domain_keys})
        ],
    }


def _stats(y: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    count = int(mask.sum())
    if count == 0:
        return {"n": 0, "event_count": 0, "weight_sum": 0.0, "weighted_event_rate": None, "kish_ess": None,
                "status": "NOT_ESTIMABLE"}
    w = weights[mask]
    yy = y[mask]
    scale = float(np.max(w))
    wn = w / scale
    scaled_total = float(np.sum(wn))
    scaled_sumsq = float(np.dot(wn, wn))
    with np.errstate(over="ignore", invalid="ignore"):
        total = float(scale * scaled_total)
    total_ok = np.isfinite(total) and total > 0
    return {
        "n": count,
        "event_count": int(np.sum(yy == 1)),
        "weight_sum": total if total_ok else None,
        "weighted_event_rate": float(np.dot(wn, yy) / scaled_total) if scaled_total > 0 else None,
        "kish_ess": float(scaled_total * scaled_total / scaled_sumsq) if scaled_total > 0 and scaled_sumsq > 0 else None,
        "status": "VALID" if total_ok else "WEIGHT_SUM_OVERFLOW",
    }


def _missingness(partition: Any, feature_names: Sequence[str], fitted_preprocessor: Any = None) -> dict[str, Any]:
    frame = getattr(partition, "X_semantic", None)
    semantic = {}
    if isinstance(frame, pd.DataFrame):
        denominator = len(frame)
        for feature in feature_names:
            if feature not in frame.columns:
                semantic[feature] = {"status": "UNAVAILABLE"}
                continue
            series = frame[feature]
            missing = int(series.isna().sum())
            structural = {}
            for code in (-1, -2):
                try:
                    structural[str(code)] = int((series == code).fillna(False).sum())
                except (TypeError, ValueError):
                    structural[str(code)] = 0
            semantic[feature] = {"status": "VALID", "semantic_missing_count": missing,
                                 "semantic_missing_fraction": missing / denominator if denominator else None,
                                 "structural_code_counts": structural}
            if fitted_preprocessor is not None and feature in getattr(fitted_preprocessor, "cat_vocabularies_", {}):
                vocab = set(fitted_preprocessor.cat_vocabularies_[feature])
                strings = series.map(lambda value: "MISSING" if pd.isna(value) else str(value))
                unknown = int((~strings.isin(vocab)).sum())
                semantic[feature]["unknown_count"] = unknown
                semantic[feature]["unknown_fraction"] = unknown / denominator if denominator else None
    metadata = getattr(partition, "metadata", {}) or {}
    raw = metadata.get("raw_feature_missingness")
    if raw is None:
        return {"status": "SEMANTIC_AVAILABLE", "semantic_layer": semantic,
                "raw_cdc": {"status": "UNAVAILABLE", "reason": "raw semantic missingness aggregate was not attached to this partition"}}
    if not isinstance(raw, Mapping):
        raise ValueError("metadata.raw_feature_missingness must be a mapping")
    result = {}
    for feature in feature_names:
        value = raw.get(feature)
        if not isinstance(value, Mapping) or "missing_count" not in value or "denominator" not in value:
            result[feature] = {"status": "UNAVAILABLE"}
            continue
        missing = float(value["missing_count"])
        denominator = float(value["denominator"])
        if not np.isfinite(missing) or not np.isfinite(denominator) or denominator <= 0 or missing < 0 or missing > denominator:
            raise ValueError(f"invalid raw missingness aggregate for {feature}")
        result[feature] = {"status": "VALID", "missing_count": int(missing), "denominator": int(denominator),
                           "missing_fraction": missing / denominator}
    return {"status": "SEMANTIC_AVAILABLE", "semantic_layer": semantic, "raw_cdc": {"status": "VALID", "features": result}}


def summarize_partition(partition: Any, expected_groups: Optional[Sequence[int]] = None, *, fitted_preprocessor: Any = None) -> dict[str, Any]:
    """Return aggregate statistics for one F/C/S/T partition."""
    y, a, w = _vector(partition, "y"), _vector(partition, "A"), _vector(partition, "WTFA_A")
    if not (len(y) == len(a) == len(w)):
        raise ValueError("y, A, and WTFA_A must have equal length")
    if np.iscomplexobj(y) or not np.all(np.isfinite(y)) or not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must be finite binary values")
    if np.iscomplexobj(w) or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("WTFA_A must be finite and strictly positive")
    groups = tuple(expected_groups if expected_groups is not None else sorted(np.unique(a).tolist()))
    if len(set(groups)) != len(groups):
        raise ValueError("expected_groups must be unique")
    features = tuple(getattr(partition, "feature_names", ()))
    result = {
        "role": str(getattr(partition, "role", "UNKNOWN")),
        "year": int(getattr(partition, "year")),
        "arm_id": str(getattr(partition, "arm_id", "UNKNOWN")),
        "overall": _stats(y, w, np.ones(len(y), dtype=bool)),
        "groups": {str(g): _stats(y, w, a == g) for g in groups},
        "design": _design_coverage(partition, len(y)),
        "feature_missingness": _missingness(partition, features, fitted_preprocessor),
    }
    return result


def summarize_cohort(partitions: Mapping[str, Any], expected_groups: Optional[Sequence[int]] = None, *, fitted_preprocessor: Any = None) -> dict[str, Any]:
    """Summarize supplied F/C/S/T partitions without redefining their design."""
    if not partitions:
        raise ValueError("partitions must be nonempty")
    return {"partitions": {str(role): summarize_partition(partition, expected_groups, fitted_preprocessor=fitted_preprocessor)
                            for role, partition in partitions.items()}}


__all__ = ["summarize_partition", "summarize_cohort"]


def summarize_eligibility(cohort: pd.DataFrame, arm_specs: Optional[Mapping[str, Mapping[str, Any]]] = None) -> dict[str, Any]:
    """Summarize annual item eligibility for every registered arm.

    ``cohort`` must already be a harmonized DataFrame.  This function does not
    load data or apply longitudinal attrition rules: exclusions are item-level
    outcome/protected-attribute eligibility in each survey year.
    """
    if not isinstance(cohort, pd.DataFrame):
        raise TypeError("cohort must be an already loaded harmonized DataFrame")
    if arm_specs is None:
        from .data_contracts import ARM_SPECS
        arm_specs = ARM_SPECS
    required = {"year", "WTFA_A", "MEDDL12M_A"}
    required.update(spec["protected_attribute"] for spec in arm_specs.values())
    missing = required - set(cohort.columns)
    if missing:
        raise ValueError(f"cohort lacks eligibility columns: {sorted(missing)}")
    weights = pd.to_numeric(cohort["WTFA_A"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("WTFA_A must be finite and nonnegative for eligibility reporting")

    def weighted(mask: np.ndarray) -> dict[str, Any]:
        selected = weights[mask]
        scale = float(np.max(selected)) if selected.size else 0.0
        scaled = float(np.sum(selected / scale)) if scale > 0 else 0.0
        with np.errstate(over="ignore", invalid="ignore"):
            total = float(scale * scaled) if scale > 0 else 0.0
        annual = weights[year_mask]
        annual_scale = float(annual.max()) if annual.size else 0.
        denominator = float(np.sum(annual / annual_scale)) if annual_scale > 0 else 0.
        fraction = float(np.sum(selected / annual_scale) / denominator) if denominator > 0 else None
        return {"n": int(mask.sum()), "weight_sum": total if np.isfinite(total) else None,
                "weight_fraction_of_annual": fraction,
                "status": "VALID" if np.isfinite(total) else "WEIGHT_SUM_OVERFLOW"}

    years = sorted(pd.unique(cohort["year"]).tolist())
    report: dict[str, Any] = {"eligibility_semantics": "annual item eligibility; not longitudinal loss-to-follow-up", "years": {}}
    for year in years:
        year_mask = cohort["year"].to_numpy() == year
        report["years"][str(year)] = {}
        for arm_id, spec in arm_specs.items():
            y = cohort.loc[year_mask, "MEDDL12M_A"].to_numpy()
            a = cohort.loc[year_mask, spec["protected_attribute"]].to_numpy()
            y_valid_local = pd.to_numeric(pd.Series(y), errors="coerce").isin([0, 1]).to_numpy()
            a_valid_local = pd.to_numeric(pd.Series(a), errors="coerce").isin(spec["expected_categories"]).to_numpy()
            y_valid = np.zeros(len(cohort), dtype=bool); y_valid[year_mask] = y_valid_local
            a_valid = np.zeros(len(cohort), dtype=bool); a_valid[year_mask] = a_valid_local
            both = y_valid & a_valid
            only_y = (~y_valid) & a_valid
            only_a = y_valid & (~a_valid)
            both_invalid = (~y_valid) & (~a_valid)
            report["years"][str(year)][arm_id] = {
                "protected_attribute": spec["protected_attribute"],
                "full_annual": weighted(year_mask),
                "outcome_Y": {"valid": weighted(year_mask & y_valid), "invalid": weighted(year_mask & ~y_valid)},
                "protected_A": {"valid": weighted(year_mask & a_valid), "invalid": weighted(year_mask & ~a_valid)},
                "joint": {"eligible": weighted(year_mask & both), "excluded": weighted(year_mask & ~both)},
                "exclusion_reasons": {"both": weighted(year_mask & both_invalid),
                                      "onlyY": weighted(year_mask & only_y),
                                      "onlyA": weighted(year_mask & only_a)},
            }
    return report


__all__.append("summarize_eligibility")
