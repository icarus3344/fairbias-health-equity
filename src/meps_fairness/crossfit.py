"""Grouped out-of-fold robustness analysis for the authorized Panel 26 cohort.

This module is deliberately additive to the accepted MEPS pipeline.  It keeps
all participant-level arrays in memory and returns aggregate summaries to the
runner, which is responsible for writing only aggregate artifacts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
from collections import Counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from meps_fairness.data.cohort import extract_meps_cohort
from meps_fairness.data.preprocess import MEPSPreprocessor, check_survey_design_quality
from meps_fairness.evaluation.metrics import (
    weighted_auprc,
    weighted_auroc,
    weighted_brier_score,
    weighted_calibration_stats,
)
from meps_fairness.pilot import (
    CAPACITY_FRACTION,
    EVIDENCE_SCOPE_NOTE,
    EXPECTED_ELIGIBLE_RECORD_COUNT,
    EXPECTED_POSITIVE_EVENTS,
    MIN_KISH,
    MIN_N,
    MIN_NEG,
    MIN_POS,
    PANEL_NUMBER,
    PRIMARY_AUDIT_DIMENSIONS,
    RESULT_LABEL,
    SEED,
    default_model_suite,
    fit_validation_calibrator_and_freeze_threshold,
    rank_models_by_validation,
)


ANALYSIS_SUBTYPE = "PANEL26_GROUPED_OOF_ROBUSTNESS"
FAIRNESS_LABEL = "EXPLORATORY_CROSSFITTED_SINGLE_PANEL_FAIRNESS"
WEIGHT_MODE_LEGACY = "LEGACY_RAW_LONGWT"
WEIGHT_MODE_MEAN1 = "MEAN1_NORMALIZED_LONGWT"
WEIGHT_MODES: tuple[str, ...] = (WEIGHT_MODE_LEGACY, WEIGHT_MODE_MEAN1)
OUTER_FOLDS = 5
INNER_FOLDS = 4
WEIGHT_SCALE_SHIFT_THRESHOLD = 0.01
CALIBRATED_AUROC_INTERPRETATION_NOTE = (
    "Fold-specific Platt transformations may alter cross-fold score ordering after "
    "concatenation, so differences between aggregated raw and calibrated OOF AUROC "
    "must NOT be interpreted as calibration improving discrimination."
)
CANDIDATE_MODEL_NAMES: tuple[str, ...] = (
    "WeightedGradientBoostingClassifier",
    "WeightedLogisticClassifier",
    "WeightedRandomForestClassifier",
)


def _python_scalar(value: Any) -> Any:
    """Convert a NumPy scalar to a plain Python scalar for stable bookkeeping."""
    return value.item() if isinstance(value, np.generic) else value


def _float_array(name: str, values: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_group_inputs(
    groups: Sequence[Any] | np.ndarray | pd.Series,
    y: Sequence[float] | np.ndarray | pd.Series,
    longwt: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_array = np.asarray(groups).reshape(-1)
    y_array = _float_array("y", y)
    weight_array = _float_array("LONGWT", longwt)
    if not (len(group_array) == len(y_array) == len(weight_array)):
        raise ValueError("groups, y, and LONGWT must have equal lengths")
    if len(group_array) == 0:
        raise ValueError("Grouped fold assignment requires at least one row")
    if pd.isna(group_array).any():
        raise ValueError("DUID/group values must be non-missing")
    if not np.isin(y_array, [0.0, 1.0]).all():
        raise ValueError("y must contain only binary 0/1 values")
    if (weight_array <= 0).any():
        raise ValueError("LONGWT must be strictly positive")
    return group_array, y_array, weight_array


@dataclasses.dataclass(frozen=True)
class GroupedFoldAssignment:
    """A deterministic row-level view of a group-to-fold assignment."""

    n_splits: int
    seed: int
    row_fold_ids: np.ndarray
    group_to_fold: dict[Any, int]
    assignment_hash: str


def _group_stratification_table(
    groups: np.ndarray,
    y: np.ndarray,
    longwt: np.ndarray,
) -> pd.DataFrame:
    """Build group-level outcome-presence and total-weight strata."""
    frame = pd.DataFrame({"group": groups, "y": y, "longwt": longwt})
    stats = (
        frame.groupby("group", sort=False, dropna=False)
        .agg(
            record_count=("y", "size"),
            has_positive=("y", "max"),
            total_longwt=("longwt", "sum"),
        )
        .reset_index()
    )
    stats["group_sort_key"] = stats["group"].map(str)
    stats = stats.sort_values("group_sort_key", kind="mergesort").reset_index(drop=True)
    if (stats["total_longwt"] <= 0).any() or (~np.isfinite(stats["total_longwt"])).any():
        raise ValueError("Every group must have finite positive total LONGWT")

    if len(stats) == 1:
        stats["weight_quintile"] = 0
    else:
        # Ranking before qcut gives a deterministic quintile even when several
        # households have identical total survey weights.
        ranks = stats["total_longwt"].rank(method="first")
        bins = pd.qcut(ranks, q=min(5, len(stats)), labels=False, duplicates="drop")
        stats["weight_quintile"] = (
            pd.Series(bins, index=stats.index).fillna(0).astype(int)
        )
    stats["stratum"] = (
        stats["has_positive"].astype(int).astype(str)
        + "_"
        + stats["weight_quintile"].astype(int).astype(str)
    )
    return stats


def assign_grouped_folds(
    groups: Sequence[Any] | np.ndarray | pd.Series,
    y: Sequence[float] | np.ndarray | pd.Series,
    longwt: Sequence[float] | np.ndarray | pd.Series,
    n_splits: int = OUTER_FOLDS,
    seed: int = SEED,
) -> GroupedFoldAssignment:
    """Assign every group to exactly one deterministic stratified fold.

    Stratification is approximate and occurs at group level using outcome
    presence and total LONGWT quintile.  Assignment is round-robin within each
    shuffled stratum, so it never falls back to row-wise K-fold splitting.
    """
    if int(n_splits) < 2:
        raise ValueError("n_splits must be at least 2")
    group_array, y_array, weight_array = _validate_group_inputs(groups, y, longwt)
    stats = _group_stratification_table(group_array, y_array, weight_array)
    rng = np.random.RandomState(int(seed))
    group_to_fold: dict[Any, int] = {}

    for stratum in sorted(stats["stratum"].unique().tolist()):
        stratum_groups = [
            _python_scalar(value)
            for value in stats.loc[stats["stratum"] == stratum, "group"].tolist()
        ]
        shuffled = np.asarray(stratum_groups, dtype=object)
        rng.shuffle(shuffled)
        for offset, group in enumerate(shuffled.tolist()):
            group_key = _python_scalar(group)
            if group_key in group_to_fold:
                raise RuntimeError("A group was assigned more than once")
            group_to_fold[group_key] = int(offset % int(n_splits))

    if len(group_to_fold) != int(stats["group"].nunique()):
        raise RuntimeError("Grouped fold assignment did not cover every group")
    try:
        row_fold_ids = np.asarray(
            [group_to_fold[_python_scalar(group)] for group in group_array],
            dtype=int,
        )
    except KeyError as exc:
        raise RuntimeError("A row group is missing from the fold assignment") from exc

    if np.any(row_fold_ids < 0) or np.any(row_fold_ids >= int(n_splits)):
        raise RuntimeError("Fold assignment contains an invalid fold ID")

    assignment_record = [
        {"group": str(group), "fold": int(group_to_fold[group])}
        for group in sorted(group_to_fold, key=str)
    ]
    assignment_json = json.dumps(assignment_record, sort_keys=True, separators=(",", ":"))
    assignment_hash = hashlib.sha256(assignment_json.encode("utf-8")).hexdigest()
    return GroupedFoldAssignment(
        n_splits=int(n_splits),
        seed=int(seed),
        row_fold_ids=row_fold_ids,
        group_to_fold=group_to_fold,
        assignment_hash=assignment_hash,
    )


def assign_outer_folds(
    cohort: Any,
    n_splits: int = OUTER_FOLDS,
    seed: int = SEED,
) -> GroupedFoldAssignment:
    """Assign the supplied cohort's DUIDs to the outer OOF folds."""
    return assign_grouped_folds(
        cohort.design["DUID"],
        cohort.y,
        cohort.design["LONGWT"],
        n_splits=n_splits,
        seed=seed,
    )


def fitting_weights(
    longwt: Sequence[float] | np.ndarray | pd.Series,
    weight_mode: str,
) -> tuple[np.ndarray, float]:
    """Return local fitting weights and the raw mean used for normalization."""
    raw = _float_array("LONGWT", longwt)
    if (raw <= 0).any():
        raise ValueError("LONGWT must be strictly positive")
    if weight_mode == WEIGHT_MODE_LEGACY:
        return raw.copy(), 1.0
    if weight_mode == WEIGHT_MODE_MEAN1:
        factor = float(raw.mean())
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Cannot normalize LONGWT with a non-positive/non-finite mean")
        return raw / factor, factor
    raise ValueError(f"Unknown weight mode: {weight_mode}")


def _aligned_arrays(
    y: Sequence[float] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray,
    longwt: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_array = _float_array("y", y)
    probability_array = _float_array("probabilities", probabilities)
    weight_array = _float_array("LONGWT", longwt)
    if not (len(y_array) == len(probability_array) == len(weight_array)):
        raise ValueError("y, probabilities, and LONGWT must have equal lengths")
    if len(y_array) == 0:
        raise ValueError("Metrics require at least one row")
    if not np.isin(y_array, [0.0, 1.0]).all():
        raise ValueError("y must contain only binary 0/1 values")
    if (weight_array <= 0).any():
        raise ValueError("LONGWT must be strictly positive")
    return y_array, probability_array, weight_array


def weighted_mean(
    values: Sequence[float] | np.ndarray,
    longwt: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    values_array = _float_array("values", values)
    weights_array = _float_array("LONGWT", longwt)
    if len(values_array) != len(weights_array):
        raise ValueError("values and LONGWT must have equal lengths")
    total_weight = float(weights_array.sum())
    if total_weight <= 0:
        raise ValueError("LONGWT must have a positive sum")
    return float(np.sum(values_array * weights_array) / total_weight)


def binary_decision_metrics(
    y: Sequence[float] | np.ndarray | pd.Series,
    decisions: Sequence[float] | np.ndarray | pd.Series,
    longwt: Sequence[float] | np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute weighted operational metrics from already-frozen binary decisions."""
    y_array = _float_array("y", y)
    decision_array = _float_array("decisions", decisions)
    weight_array = _float_array("LONGWT", longwt)
    if not (len(y_array) == len(decision_array) == len(weight_array)):
        raise ValueError("y, decisions, and LONGWT must have equal lengths")
    if len(y_array) == 0:
        raise ValueError("Decision metrics require at least one row")
    if not np.isin(y_array, [0.0, 1.0]).all():
        raise ValueError("y must contain only binary 0/1 values")
    if not np.isin(decision_array, [0.0, 1.0]).all():
        raise ValueError("decisions must contain only binary 0/1 values")
    if (weight_array <= 0).any():
        raise ValueError("LONGWT must be strictly positive")

    positive = y_array == 1.0
    negative = ~positive
    predicted_positive = decision_array == 1.0
    predicted_negative = ~predicted_positive
    tp = float(np.sum(weight_array * positive * predicted_positive))
    fp = float(np.sum(weight_array * negative * predicted_positive))
    fn = float(np.sum(weight_array * positive * predicted_negative))
    tn = float(np.sum(weight_array * negative * predicted_negative))
    tpr = tp / (tp + fn) if tp + fn > 0 else 0.0
    tnr = tn / (tn + fp) if tn + fp > 0 else 0.0
    ppv = tp / (tp + fp) if tp + fp > 0 else 0.0
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) if 2.0 * tp + fp + fn > 0 else 0.0
    total_weight = float(weight_array.sum())
    return {
        "tpr": float(tpr),
        "tnr": float(tnr),
        "fpr": float(1.0 - tnr),
        "ppv": float(ppv),
        "f1": float(f1),
        "selection_rate": float((tp + fp) / total_weight),
    }


def _kish_effective_n(longwt: np.ndarray) -> float:
    total_weight = float(longwt.sum())
    sum_squared = float(np.sum(longwt**2))
    return float(total_weight**2 / sum_squared) if sum_squared > 0 else 0.0


def audit_oof_binary_decisions(
    y: Sequence[float] | np.ndarray | pd.Series,
    decisions: Sequence[float] | np.ndarray | pd.Series,
    audit_df: pd.DataFrame,
    longwt: Sequence[float] | np.ndarray | pd.Series,
    dimensions: Sequence[str] = PRIMARY_AUDIT_DIMENSIONS,
    min_n: int = MIN_N,
    min_pos: int = MIN_POS,
    min_neg: int = MIN_NEG,
    min_kish: float = MIN_KISH,
) -> dict[str, Any]:
    """Audit precomputed OOF decisions under the unchanged suppression rules.

    This helper intentionally accepts decisions rather than probabilities or a
    threshold.  It therefore cannot derive a global threshold from the OOF
    sample, and it preserves each row's fold-specific decision boundary.
    """
    y_array = _float_array("y", y)
    decision_array = _float_array("decisions", decisions)
    weight_array = _float_array("LONGWT", longwt)
    if not (len(y_array) == len(decision_array) == len(weight_array) == len(audit_df)):
        raise ValueError("y, decisions, LONGWT, and audit_df must have equal lengths")
    if not np.isin(y_array, [0.0, 1.0]).all():
        raise ValueError("y must contain only binary 0/1 values")
    if not np.isin(decision_array, [0.0, 1.0]).all():
        raise ValueError("decisions must contain only binary 0/1 values")
    if (weight_array <= 0).any():
        raise ValueError("LONGWT must be strictly positive")
    if min_n < 1 or min_pos < 0 or min_neg < 0 or min_kish < 0:
        raise ValueError("Suppression thresholds must be non-negative and min_n must be positive")

    audit = audit_df.reset_index(drop=True)
    dimension_results: dict[str, Any] = {}
    for dimension in dimensions:
        if dimension not in audit.columns:
            raise ValueError(f"Missing required audit dimension: {dimension}")
        groups = audit[dimension]
        if groups.isna().any():
            raise ValueError(f"Audit dimension {dimension} contains missing values")
        subgroup_results: dict[str, Any] = {}
        valid_tprs: dict[str, float] = {}
        group_values = sorted(groups.unique().tolist(), key=str)
        for group_value in group_values:
            mask = (groups == group_value).to_numpy(dtype=bool)
            y_group = y_array[mask]
            decisions_group = decision_array[mask]
            weights_group = weight_array[mask]
            sample_n = int(mask.sum())
            positive_events = int((y_group == 1.0).sum())
            negative_events = int((y_group == 0.0).sum())
            kish = _kish_effective_n(weights_group)
            suppression_reasons: list[str] = []
            if sample_n < min_n:
                suppression_reasons.append(f"sample_n<{min_n}")
            if positive_events < min_pos:
                suppression_reasons.append(f"positive_events<{min_pos}")
            if negative_events < min_neg:
                suppression_reasons.append(f"negative_events<{min_neg}")
            if kish < min_kish:
                suppression_reasons.append(f"kish_effective_n<{min_kish:g}")

            group_key = str(group_value)
            common = {
                "group": group_key,
                "sample_n": sample_n,
                "positive_events": positive_events,
                "negative_events": negative_events,
                "kish_effective_n": float(kish),
            }
            if suppression_reasons:
                subgroup_results[group_key] = {
                    **common,
                    "status": "Suppressed (Insufficient Sample/Power)",
                    "suppression_reasons": suppression_reasons,
                    "tpr": None,
                    "fpr": None,
                    "ppv": None,
                    "selection_rate": None,
                }
                continue

            metrics = binary_decision_metrics(y_group, decisions_group, weights_group)
            subgroup_results[group_key] = {
                **common,
                "status": "Unsuppressed",
                "suppression_reasons": [],
                "tpr": metrics["tpr"],
                "fpr": metrics["fpr"],
                "ppv": metrics["ppv"],
                "selection_rate": metrics["selection_rate"],
            }
            valid_tprs[group_key] = metrics["tpr"]

        pairwise_gaps = [
            abs(valid_tprs[left] - valid_tprs[right])
            for left, right in itertools.combinations(sorted(valid_tprs), 2)
        ]
        partial_gap = float(max(pairwise_gaps)) if pairwise_gaps else None
        unsuppressed_count = len(valid_tprs)
        group_count = len(group_values)
        if unsuppressed_count >= 2 and unsuppressed_count == group_count:
            endpoint = "FULLY_ESTIMABLE"
            full_gap = partial_gap
            partial_gap = None
        elif unsuppressed_count >= 2:
            endpoint = "PARTIALLY_ESTIMABLE_UNSUPPRESSED_GROUPS_ONLY"
            full_gap = None
        else:
            endpoint = "NOT_ESTIMABLE_SUPPRESSED"
            full_gap = None
            partial_gap = None
        dimension_results[dimension] = {
            "dimension": dimension,
            "subgroups": subgroup_results,
            "group_count": group_count,
            "unsuppressed_subgroups_count": unsuppressed_count,
            "suppressed_subgroups_count": group_count - unsuppressed_count,
            "full_max_pairwise_tpr_gap": full_gap,
            "partial_max_pairwise_tpr_gap_unsuppressed_groups": partial_gap,
            # Keep the historical keys as aliases for the fully estimable
            # endpoint only.  A partial endpoint must never expose its
            # available-group gap through a field that could be read as the
            # prespecified full maximum.
            "max_pairwise_tpr_gap": full_gap,
            "max_tpr_gap": full_gap,
            "endpoint": endpoint,
            "estimability_status": endpoint,
        }

    all_primary_dimensions_fully_estimable = bool(dimension_results) and all(
        record["endpoint"] == "FULLY_ESTIMABLE"
        for record in dimension_results.values()
    )
    if all_primary_dimensions_fully_estimable:
        primary_gap = float(
            max(
                record["full_max_pairwise_tpr_gap"]
                for record in dimension_results.values()
                if record["full_max_pairwise_tpr_gap"] is not None
            )
        )
        endpoint = "PRIMARY_FAIRNESS_ENDPOINT = FULLY_ESTIMABLE"
    else:
        primary_gap = None
        endpoint = "PRIMARY_FAIRNESS_ENDPOINT = NOT_FULLY_ESTIMABLE_DUE_TO_SUPPRESSION"
    return {
        "fairness_label": FAIRNESS_LABEL,
        "decision_source": "concatenated OOF binary decisions at fold-specific frozen thresholds",
        "threshold_source": "inner-validation-derived threshold within each outer fold",
        "global_threshold_used": False,
        "dimensions": dimension_results,
        "primary_fairness_max_tpr_gap": primary_gap,
        "primary_fairness_endpoint": endpoint,
        "primary_fairness_endpoint_status": endpoint,
        "primary_dimensions_fully_estimable": all_primary_dimensions_fully_estimable,
    }


def _probability_metrics(
    y: Sequence[float] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray,
    longwt: Sequence[float] | np.ndarray | pd.Series,
) -> dict[str, Any]:
    y_array, probability_array, weight_array = _aligned_arrays(y, probabilities, longwt)
    auroc = float(weighted_auroc(y_array, probability_array, weight_array))
    auprc = float(weighted_auprc(y_array, probability_array, weight_array))
    prevalence = weighted_mean(y_array, weight_array)
    return {
        "weighted_auroc": auroc,
        "weighted_auprc": auprc,
        "weighted_brier": float(weighted_brier_score(y_array, probability_array, weight_array)),
        "weighted_brier_score": float(weighted_brier_score(y_array, probability_array, weight_array)),
        "weighted_observed_prevalence": prevalence,
        "weighted_observed_event_prevalence": prevalence,
        "predicted_mean_risk": weighted_mean(probability_array, weight_array),
        "auprc_weighted_prevalence_ratio": float(auprc / prevalence) if prevalence > 0 else None,
    }


def calibration_bins(
    y: Sequence[float] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray,
    longwt: Sequence[float] | np.ndarray | pd.Series,
    probability_type: str,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Create aggregate equal-width calibration bins using original LONGWT."""
    y_array, probability_array, weight_array = _aligned_arrays(y, probabilities, longwt)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for bin_index in range(n_bins):
        lower = float(edges[bin_index])
        upper = float(edges[bin_index + 1])
        mask = (probability_array >= lower) & (
            probability_array <= upper if bin_index == n_bins - 1 else probability_array < upper
        )
        if not mask.any():
            rows.append(
                {
                    "probability_type": probability_type,
                    "bin": bin_index + 1,
                    "lower_edge": lower,
                    "upper_edge": upper,
                    "sample_n": 0,
                    "weighted_n": 0.0,
                    "weighted_predicted_mean": None,
                    "weighted_observed_event_prevalence": None,
                }
            )
            continue
        bin_weights = weight_array[mask]
        rows.append(
            {
                "probability_type": probability_type,
                "bin": bin_index + 1,
                "lower_edge": lower,
                "upper_edge": upper,
                "sample_n": int(mask.sum()),
                "weighted_n": float(bin_weights.sum()),
                "weighted_predicted_mean": weighted_mean(probability_array[mask], bin_weights),
                "weighted_observed_event_prevalence": weighted_mean(y_array[mask], bin_weights),
            }
        )
    return rows


@dataclasses.dataclass
class GroupedOOFResult:
    """In-memory OOF arrays plus aggregate-ready fold records.

    The arrays in this object are intentionally not serializable by the run
    writer.  They exist only long enough to calculate aggregate artifacts.
    """

    weight_mode: str
    seed: int
    n_outer_folds: int
    outer_assignment_hash: str
    inner_assignment_hashes: dict[int, str]
    y: np.ndarray
    longwt: np.ndarray
    audit: pd.DataFrame
    outer_fold_ids: np.ndarray
    prediction_counts: np.ndarray
    candidate_raw_oof: dict[str, np.ndarray]
    selected_raw_oof: np.ndarray
    selected_calibrated_oof: np.ndarray
    selected_decisions_oof: np.ndarray
    selected_model_oof: np.ndarray
    thresholds_oof: np.ndarray
    outer_fold_summaries: list[dict[str, Any]]
    validation_metrics: list[dict[str, Any]]
    candidate_fold_metrics: list[dict[str, Any]]
    selected_fold_metrics: list[dict[str, Any]]
    normalization_records: list[dict[str, Any]]
    fold_partitions: list[dict[str, Any]]


def _validate_candidate_specifications(specifications: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = [dict(specification) for specification in specifications]
    names = [str(item.get("model_name")) for item in normalized]
    if len(normalized) != len(CANDIDATE_MODEL_NAMES) or set(names) != set(CANDIDATE_MODEL_NAMES):
        raise ValueError("The robustness analysis requires exactly the reviewed three candidate models")
    if len(set(names)) != len(names):
        raise ValueError("Candidate model names must be unique")
    for item, name in zip(normalized, names):
        if "model" not in item:
            raise ValueError(f"Candidate specification {name} has no model")
        item["model_name"] = name
        item.setdefault("configuration", {})
    return normalized


def _safe_probability_vector(name: str, values: Sequence[float] | np.ndarray) -> np.ndarray:
    probabilities = _float_array(name, values)
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError(f"{name} must lie in [0, 1]")
    return probabilities


def run_grouped_oof(
    cohort: Any,
    weight_mode: str,
    seed: int = SEED,
    n_outer_folds: int = OUTER_FOLDS,
    preprocessor_factory: Callable[..., Any] = MEPSPreprocessor,
    model_suite_factory: Callable[[int], Sequence[Mapping[str, Any]]] = default_model_suite,
    calibrator_fit_fn: Callable[..., tuple[Any, np.ndarray, float]] = fit_validation_calibrator_and_freeze_threshold,
) -> GroupedOOFResult:
    """Run one deterministic grouped OOF pass for one fitting-weight mode."""
    if weight_mode not in WEIGHT_MODES:
        raise ValueError(f"Unknown weight mode: {weight_mode}")
    if int(n_outer_folds) != OUTER_FOLDS:
        raise ValueError("This robustness contract requires exactly five outer folds")

    y_full = _float_array("cohort.y", cohort.y)
    longwt_full = _float_array("cohort.LONGWT", cohort.design["LONGWT"])
    if not (len(cohort.X) == len(y_full) == len(cohort.design) == len(cohort.audit)):
        raise ValueError("Cohort components must have equal lengths")
    if cohort.design["DUID"].isna().any():
        raise ValueError("Cohort DUID values must be non-missing")
    for dimension in PRIMARY_AUDIT_DIMENSIONS:
        if dimension not in cohort.audit.columns:
            raise ValueError(f"Cohort is missing required audit dimension {dimension}")

    n_records = len(y_full)
    outer_assignment = assign_outer_folds(cohort, n_splits=OUTER_FOLDS, seed=seed)
    candidate_raw_oof = {
        name: np.full(n_records, np.nan, dtype=float) for name in CANDIDATE_MODEL_NAMES
    }
    selected_raw_oof = np.full(n_records, np.nan, dtype=float)
    selected_calibrated_oof = np.full(n_records, np.nan, dtype=float)
    selected_decisions_oof = np.full(n_records, np.nan, dtype=float)
    selected_model_oof = np.full(n_records, "", dtype=object)
    thresholds_oof = np.full(n_records, np.nan, dtype=float)
    outer_fold_ids = outer_assignment.row_fold_ids.copy()
    prediction_counts = np.zeros(n_records, dtype=int)
    inner_assignment_hashes: dict[int, str] = {}
    outer_fold_summaries: list[dict[str, Any]] = []
    validation_metrics: list[dict[str, Any]] = []
    candidate_fold_metrics: list[dict[str, Any]] = []
    selected_fold_metrics: list[dict[str, Any]] = []
    normalization_records: list[dict[str, Any]] = []
    fold_partitions: list[dict[str, Any]] = []

    cohort_groups = cohort.design["DUID"].to_numpy()
    for outer_fold in range(OUTER_FOLDS):
        outer_test_idx = np.flatnonzero(outer_assignment.row_fold_ids == outer_fold)
        development_idx = np.flatnonzero(outer_assignment.row_fold_ids != outer_fold)
        if len(outer_test_idx) == 0 or len(development_idx) == 0:
            raise RuntimeError("Every outer fold must have non-empty test and development partitions")

        inner_seed = int(seed) + 1000 + outer_fold
        inner_assignment = assign_grouped_folds(
            cohort_groups[development_idx],
            y_full[development_idx],
            longwt_full[development_idx],
            n_splits=INNER_FOLDS,
            seed=inner_seed,
        )
        inner_assignment_hashes[outer_fold] = inner_assignment.assignment_hash
        inner_validation_idx = development_idx[inner_assignment.row_fold_ids == 0]
        inner_train_idx = development_idx[inner_assignment.row_fold_ids != 0]
        if len(inner_train_idx) == 0 or len(inner_validation_idx) == 0:
            raise RuntimeError("Every outer development pool must have non-empty inner train/validation partitions")

        outer_test_duids = set(cohort_groups[outer_test_idx].tolist())
        inner_train_duids = set(cohort_groups[inner_train_idx].tolist())
        inner_validation_duids = set(cohort_groups[inner_validation_idx].tolist())
        if outer_test_duids & inner_train_duids or outer_test_duids & inner_validation_duids:
            raise RuntimeError("Outer-test DUIDs entered an inner fitting or validation partition")
        if inner_train_duids & inner_validation_duids:
            raise RuntimeError("Inner train and validation DUIDs overlap")

        preprocessor = preprocessor_factory(
            continuous_features=cohort.continuous_features,
            categorical_features=cohort.categorical_features,
        )
        X_inner_train = preprocessor.fit_transform(cohort.X.iloc[inner_train_idx].copy())
        X_inner_validation = preprocessor.transform(cohort.X.iloc[inner_validation_idx].copy())
        X_outer_test = preprocessor.transform(cohort.X.iloc[outer_test_idx].copy())

        y_train = y_full[inner_train_idx]
        y_validation = y_full[inner_validation_idx]
        y_outer_test = y_full[outer_test_idx]
        weight_train_raw = longwt_full[inner_train_idx]
        weight_validation_raw = longwt_full[inner_validation_idx]
        weight_outer_test = longwt_full[outer_test_idx]
        weight_train_fit, train_factor = fitting_weights(weight_train_raw, weight_mode)
        weight_validation_fit, validation_factor = fitting_weights(weight_validation_raw, weight_mode)
        normalization_record = {
            "weight_mode": weight_mode,
            "outer_fold": outer_fold,
            "inner_train_raw_weight_mean": float(weight_train_raw.mean()),
            "inner_validation_raw_weight_mean": float(weight_validation_raw.mean()),
            "inner_train_normalization_factor": float(train_factor),
            "inner_validation_normalization_factor": float(validation_factor),
            "inner_train_fit_weight_mean": float(weight_train_fit.mean()),
            "inner_validation_fit_weight_mean": float(weight_validation_fit.mean()),
            "metric_evaluation_weights": "original LONGWT",
        }
        if weight_mode == WEIGHT_MODE_MEAN1:
            if not np.isclose(weight_train_fit.mean(), 1.0) or not np.isclose(weight_validation_fit.mean(), 1.0):
                raise RuntimeError("Mean-1 weight mode did not produce mean-1 fitting weights")
        normalization_records.append(normalization_record)

        specifications = _validate_candidate_specifications(model_suite_factory(int(seed)))
        models_by_name: dict[str, Any] = {}
        outer_raw_by_name: dict[str, np.ndarray] = {}
        validation_raw_by_name: dict[str, np.ndarray] = {}
        validation_rows_for_fold: list[dict[str, Any]] = []
        for specification in specifications:
            model_name = str(specification["model_name"])
            model = specification["model"]
            model.fit(X_inner_train, y_train, sample_weight=weight_train_fit)
            validation_raw = _safe_probability_vector(
                f"{model_name} inner-validation probabilities",
                model.predict_proba(X_inner_validation),
            )
            outer_raw = _safe_probability_vector(
                f"{model_name} outer-test probabilities",
                model.predict_proba(X_outer_test),
            )
            if len(validation_raw) != len(inner_validation_idx) or len(outer_raw) != len(outer_test_idx):
                raise RuntimeError("Model probability output length does not match its partition")
            models_by_name[model_name] = model
            outer_raw_by_name[model_name] = outer_raw
            validation_raw_by_name[model_name] = validation_raw
            validation_rows_for_fold.append(
                {
                    "weight_mode": weight_mode,
                    "outer_fold": outer_fold,
                    "model_name": model_name,
                    "configuration": dict(specification["configuration"]),
                    "validation_weighted_auroc": float(weighted_auroc(y_validation, validation_raw, weight_validation_raw)),
                    "validation_weighted_auprc": float(weighted_auprc(y_validation, validation_raw, weight_validation_raw)),
                    "validation_weighted_brier": float(weighted_brier_score(y_validation, validation_raw, weight_validation_raw)),
                    "validation_predicted_mean_risk": weighted_mean(validation_raw, weight_validation_raw),
                    "validation_weighted_observed_prevalence": weighted_mean(y_validation, weight_validation_raw),
                }
            )

        ranked_validation = rank_models_by_validation(validation_rows_for_fold)
        selected = ranked_validation[0]
        selected_model_name = str(selected["model_name"])
        for validation_row in ranked_validation:
            validation_metrics.append(
                {
                    **validation_row,
                    "selected": bool(validation_row["model_name"] == selected_model_name),
                }
            )

        selected_validation_raw = _safe_probability_vector(
            "selected inner-validation probabilities",
            validation_raw_by_name[selected_model_name],
        )
        calibrator, selected_validation_calibrated, frozen_threshold = calibrator_fit_fn(
            selected_validation_raw,
            y_validation,
            weight_validation_fit,
            seed=int(seed),
            capacity_fraction=CAPACITY_FRACTION,
        )
        selected_validation_calibrated = _safe_probability_vector(
            "selected calibrated inner-validation probabilities", selected_validation_calibrated
        )
        frozen_threshold = float(frozen_threshold)
        if not np.isfinite(frozen_threshold) or not 0.0 <= frozen_threshold <= 1.0:
            raise RuntimeError("Fold-specific frozen threshold is not finite and within [0, 1]")

        # Model selection and calibration are complete before any outer-test
        # outcome is used.  Outer probabilities below are evaluation-only.
        for model_name in CANDIDATE_MODEL_NAMES:
            outer_probabilities = outer_raw_by_name[model_name]
            if np.isfinite(candidate_raw_oof[model_name][outer_test_idx]).any():
                raise RuntimeError("An outer OOF slot was assigned more than once")
            candidate_raw_oof[model_name][outer_test_idx] = outer_probabilities
            outer_metrics = _probability_metrics(y_outer_test, outer_probabilities, weight_outer_test)
            candidate_fold_metrics.append(
                {
                    "weight_mode": weight_mode,
                    "record_type": "candidate_model",
                    "outer_fold": outer_fold,
                    "model_name": model_name,
                    "selected_model": selected_model_name,
                    "outer_test_record_count": int(len(outer_test_idx)),
                    "outer_test_positive_events": int((y_outer_test == 1.0).sum()),
                    "raw_auroc": outer_metrics["weighted_auroc"],
                    "raw_auprc": outer_metrics["weighted_auprc"],
                    "raw_brier": outer_metrics["weighted_brier"],
                }
            )

        selected_outer_raw = outer_raw_by_name[selected_model_name]
        selected_outer_calibrated = _safe_probability_vector(
            "selected calibrated outer-test probabilities",
            calibrator.predict_proba(selected_outer_raw),
        )
        selected_outer_decisions = (selected_outer_calibrated >= frozen_threshold).astype(float)
        selected_raw_oof[outer_test_idx] = selected_outer_raw
        selected_calibrated_oof[outer_test_idx] = selected_outer_calibrated
        selected_decisions_oof[outer_test_idx] = selected_outer_decisions
        selected_model_oof[outer_test_idx] = selected_model_name
        thresholds_oof[outer_test_idx] = frozen_threshold
        prediction_counts[outer_test_idx] += 1

        outer_raw_metrics = _probability_metrics(y_outer_test, selected_outer_raw, weight_outer_test)
        outer_calibrated_metrics = _probability_metrics(
            y_outer_test, selected_outer_calibrated, weight_outer_test
        )
        selected_outer_operational = binary_decision_metrics(
            y_outer_test, selected_outer_decisions, weight_outer_test
        )
        selected_fold_metrics.append(
            {
                "weight_mode": weight_mode,
                "record_type": "selected_pipeline",
                "outer_fold": outer_fold,
                "model_name": selected_model_name,
                "selected_model": selected_model_name,
                "outer_test_record_count": int(len(outer_test_idx)),
                "outer_test_positive_events": int((y_outer_test == 1.0).sum()),
                "frozen_threshold": frozen_threshold,
                "raw_auroc": outer_raw_metrics["weighted_auroc"],
                "raw_auprc": outer_raw_metrics["weighted_auprc"],
                "calibrated_auroc": outer_calibrated_metrics["weighted_auroc"],
                "calibrated_auprc": outer_calibrated_metrics["weighted_auprc"],
                "calibrated_brier": outer_calibrated_metrics["weighted_brier"],
                "tpr": selected_outer_operational["tpr"],
                "tnr": selected_outer_operational["tnr"],
                "fpr": selected_outer_operational["fpr"],
                "ppv": selected_outer_operational["ppv"],
                "f1": selected_outer_operational["f1"],
                "selection_rate": selected_outer_operational["selection_rate"],
            }
        )

        validation_decisions = (selected_validation_calibrated >= frozen_threshold).astype(float)
        outer_fold_summaries.append(
            {
                "weight_mode": weight_mode,
                "outer_fold": outer_fold,
                "outer_test_record_count": int(len(outer_test_idx)),
                "outer_test_positive_events": int((y_outer_test == 1.0).sum()),
                "outer_test_duid_count": int(len(outer_test_duids)),
                "outer_test_weighted_prevalence": weighted_mean(y_outer_test, weight_outer_test),
                "development_record_count": int(len(development_idx)),
                "inner_train_record_count": int(len(inner_train_idx)),
                "inner_validation_record_count": int(len(inner_validation_idx)),
                "inner_train_duid_count": int(len(inner_train_duids)),
                "inner_validation_duid_count": int(len(inner_validation_duids)),
                "selected_model": selected_model_name,
                "frozen_threshold": frozen_threshold,
                "validation_weighted_selection_rate": binary_decision_metrics(
                    y_validation, validation_decisions, weight_validation_raw
                )["selection_rate"],
                **normalization_record,
            }
        )
        fold_partitions.append(
            {
                "outer_fold": outer_fold,
                "outer_test_indices": outer_test_idx.copy(),
                "development_indices": development_idx.copy(),
                "inner_train_indices": inner_train_idx.copy(),
                "inner_validation_indices": inner_validation_idx.copy(),
                "outer_test_duids": tuple(sorted(outer_test_duids, key=str)),
                "inner_train_duids": tuple(sorted(inner_train_duids, key=str)),
                "inner_validation_duids": tuple(sorted(inner_validation_duids, key=str)),
            }
        )

    if not np.all(prediction_counts == 1):
        raise RuntimeError("Every eligible record must receive exactly one outer-test prediction")
    if any(not np.isfinite(values).all() for values in candidate_raw_oof.values()):
        raise RuntimeError("At least one candidate model lacks a finite OOF probability")
    if not np.isfinite(selected_raw_oof).all() or not np.isfinite(selected_calibrated_oof).all():
        raise RuntimeError("Selected pipeline lacks a finite OOF probability")
    if not np.isin(selected_decisions_oof, [0.0, 1.0]).all():
        raise RuntimeError("Selected pipeline lacks a binary OOF decision")
    if (selected_model_oof == "").any() or not np.isfinite(thresholds_oof).all():
        raise RuntimeError("Selected pipeline OOF metadata is incomplete")

    return GroupedOOFResult(
        weight_mode=weight_mode,
        seed=int(seed),
        n_outer_folds=OUTER_FOLDS,
        outer_assignment_hash=outer_assignment.assignment_hash,
        inner_assignment_hashes=inner_assignment_hashes,
        y=y_full.copy(),
        longwt=longwt_full.copy(),
        audit=cohort.audit[list(PRIMARY_AUDIT_DIMENSIONS)].copy().reset_index(drop=True),
        outer_fold_ids=outer_fold_ids,
        prediction_counts=prediction_counts,
        candidate_raw_oof=candidate_raw_oof,
        selected_raw_oof=selected_raw_oof,
        selected_calibrated_oof=selected_calibrated_oof,
        selected_decisions_oof=selected_decisions_oof,
        selected_model_oof=selected_model_oof,
        thresholds_oof=thresholds_oof,
        outer_fold_summaries=outer_fold_summaries,
        validation_metrics=validation_metrics,
        candidate_fold_metrics=candidate_fold_metrics,
        selected_fold_metrics=selected_fold_metrics,
        normalization_records=normalization_records,
        fold_partitions=fold_partitions,
    )


def summarize_grouped_oof(result: GroupedOOFResult) -> dict[str, Any]:
    """Reduce one in-memory OOF result to aggregate-only records."""
    candidate_metrics: list[dict[str, Any]] = []
    for model_name in CANDIDATE_MODEL_NAMES:
        metrics = _probability_metrics(result.y, result.candidate_raw_oof[model_name], result.longwt)
        candidate_metrics.append(
            {
                "result_label": RESULT_LABEL,
                "analysis_subtype": ANALYSIS_SUBTYPE,
                "weight_mode": result.weight_mode,
                "model_name": model_name,
                "weighted_auroc": metrics["weighted_auroc"],
                "weighted_auprc": metrics["weighted_auprc"],
                "weighted_brier": metrics["weighted_brier"],
                "weighted_observed_prevalence": metrics["weighted_observed_prevalence"],
                "auprc_weighted_prevalence_ratio": metrics["auprc_weighted_prevalence_ratio"],
            }
        )

    selected_raw_metrics = _probability_metrics(
        result.y, result.selected_raw_oof, result.longwt
    )
    selected_calibrated_metrics = _probability_metrics(
        result.y, result.selected_calibrated_oof, result.longwt
    )
    calibration_stats = weighted_calibration_stats(
        result.y, result.selected_calibrated_oof, result.longwt
    )
    selected_operational = binary_decision_metrics(
        result.y, result.selected_decisions_oof, result.longwt
    )
    fairness = audit_oof_binary_decisions(
        result.y,
        result.selected_decisions_oof,
        result.audit,
        result.longwt,
    )
    calibration_bin_rows = [
        {
            "result_label": RESULT_LABEL,
            "analysis_subtype": ANALYSIS_SUBTYPE,
            "weight_mode": result.weight_mode,
            **row,
        }
        for probability_type, probabilities in (
            ("raw", result.selected_raw_oof),
            ("calibrated", result.selected_calibrated_oof),
        )
        for row in calibration_bins(result.y, probabilities, result.longwt, probability_type)
    ]

    outer_lookup = {
        (int(row["outer_fold"]), str(row["model_name"])): row
        for row in result.candidate_fold_metrics
    }
    wins = Counter(str(row["selected_model"]) for row in result.outer_fold_summaries)
    stability_rows: list[dict[str, Any]] = []
    for validation_row in result.validation_metrics:
        fold = int(validation_row["outer_fold"])
        model_name = str(validation_row["model_name"])
        outer_row = outer_lookup[(fold, model_name)]
        stability_rows.append(
            {
                "result_label": RESULT_LABEL,
                "analysis_subtype": ANALYSIS_SUBTYPE,
                "weight_mode": result.weight_mode,
                "record_type": "fold_validation",
                "outer_fold": fold,
                "model_name": model_name,
                "selected": bool(validation_row["selected"]),
                "folds_won": None,
                "validation_weighted_auprc": validation_row["validation_weighted_auprc"],
                "validation_weighted_auroc": validation_row["validation_weighted_auroc"],
                "outer_fold_raw_auprc": outer_row["raw_auprc"],
                "selected_model_outer_fold_auprc": (
                    outer_row["raw_auprc"] if validation_row["selected"] else None
                ),
            }
        )
    for model_name in CANDIDATE_MODEL_NAMES:
        stability_rows.append(
            {
                "result_label": RESULT_LABEL,
                "analysis_subtype": ANALYSIS_SUBTYPE,
                "weight_mode": result.weight_mode,
                "record_type": "model_frequency",
                "outer_fold": None,
                "model_name": model_name,
                "selected": None,
                "folds_won": int(wins.get(model_name, 0)),
                "validation_weighted_auprc": None,
                "validation_weighted_auroc": None,
                "outer_fold_raw_auprc": None,
                "selected_model_outer_fold_auprc": None,
            }
        )

    selected_pipeline = {
        "raw": {
            "weighted_auroc": selected_raw_metrics["weighted_auroc"],
            "weighted_auprc": selected_raw_metrics["weighted_auprc"],
            "weighted_brier": selected_raw_metrics["weighted_brier"],
            "weighted_observed_prevalence": selected_raw_metrics["weighted_observed_prevalence"],
            "predicted_mean_risk": selected_raw_metrics["predicted_mean_risk"],
        },
        "calibrated": {
            "weighted_auroc": selected_calibrated_metrics["weighted_auroc"],
            "weighted_auprc": selected_calibrated_metrics["weighted_auprc"],
            "weighted_brier": selected_calibrated_metrics["weighted_brier"],
            "calibration_intercept": float(calibration_stats["intercept"]),
            "calibration_slope": float(calibration_stats["slope"]),
            "ece": float(calibration_stats["ece"]),
            "predicted_mean_risk": selected_calibrated_metrics["predicted_mean_risk"],
            "weighted_observed_prevalence": selected_calibrated_metrics["weighted_observed_prevalence"],
            "auroc_interpretation_note": CALIBRATED_AUROC_INTERPRETATION_NOTE,
        },
        "operational": selected_operational,
        "threshold_provenance": {
            "fold_specific": True,
            "global_oof_threshold_derived": False,
            "thresholds_by_outer_fold": [
                {
                    "outer_fold": int(row["outer_fold"]),
                    "selected_model": row["selected_model"],
                    "frozen_threshold": row["frozen_threshold"],
                }
                for row in sorted(result.outer_fold_summaries, key=lambda item: int(item["outer_fold"]))
            ],
        },
    }
    return {
        "result_label": RESULT_LABEL,
        "analysis_subtype": ANALYSIS_SUBTYPE,
        "weight_mode": result.weight_mode,
        "outer_assignment_hash": result.outer_assignment_hash,
        "inner_assignment_hashes": {
            str(key): value for key, value in sorted(result.inner_assignment_hashes.items())
        },
        "outer_folds": result.outer_fold_summaries,
        "candidate_model_oof_metrics": candidate_metrics,
        "selected_pipeline_oof": selected_pipeline,
        "model_selection_frequency": {name: int(wins.get(name, 0)) for name in CANDIDATE_MODEL_NAMES},
        "model_selection_stability": stability_rows,
        "fold_metrics": [*result.candidate_fold_metrics, *result.selected_fold_metrics],
        "fairness": fairness,
        "calibration_bins": calibration_bin_rows,
        "normalization_factors": result.normalization_records,
        "oof_coverage": {
            "eligible_record_count": int(len(result.y)),
            "outer_prediction_count_min": int(result.prediction_counts.min()),
            "outer_prediction_count_max": int(result.prediction_counts.max()),
            "candidate_model_oof_counts": {
                name: int(np.isfinite(probabilities).sum())
                for name, probabilities in result.candidate_raw_oof.items()
            },
            "selected_raw_oof_count": int(np.isfinite(result.selected_raw_oof).sum()),
            "selected_calibrated_oof_count": int(np.isfinite(result.selected_calibrated_oof).sum()),
            "selected_decision_count": int(np.isin(result.selected_decisions_oof, [0.0, 1.0]).sum()),
        },
    }


def compare_weight_modes(
    mode_summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the requested descriptive raw-versus-mean-1 comparison."""
    if set(mode_summaries) != set(WEIGHT_MODES):
        raise ValueError("Both required weight modes must be available")
    legacy = mode_summaries[WEIGHT_MODE_LEGACY]
    normalized = mode_summaries[WEIGHT_MODE_MEAN1]
    rows: list[dict[str, Any]] = []

    selected_auprc_difference = abs(
        float(normalized["selected_pipeline_oof"]["raw"]["weighted_auprc"])
        - float(legacy["selected_pipeline_oof"]["raw"]["weighted_auprc"])
    )
    selected_pipeline_classification = (
        "SELECTED_PIPELINE_WEIGHT_SCALE_SENSITIVITY_MATERIAL"
        if selected_auprc_difference >= WEIGHT_SCALE_SHIFT_THRESHOLD
        else "SELECTED_PIPELINE_WEIGHT_SCALE_SENSITIVITY_SMALL"
    )

    candidate_auprc_differences: dict[str, float] = {}
    for model_name in CANDIDATE_MODEL_NAMES:
        legacy_row = next(
            row
            for row in legacy["candidate_model_oof_metrics"]
            if row["model_name"] == model_name
        )
        normalized_row = next(
            row
            for row in normalized["candidate_model_oof_metrics"]
            if row["model_name"] == model_name
        )
        candidate_auprc_differences[model_name] = abs(
            float(normalized_row["weighted_auprc"])
            - float(legacy_row["weighted_auprc"])
        )
    maximum_candidate_model_name = max(
        candidate_auprc_differences,
        key=lambda model_name: (candidate_auprc_differences[model_name], str(model_name)),
    )
    maximum_candidate_model_shift = candidate_auprc_differences[maximum_candidate_model_name]
    candidate_model_classification = (
        "CANDIDATE_MODEL_WEIGHT_SCALE_SENSITIVITY_MATERIAL"
        if maximum_candidate_model_shift >= WEIGHT_SCALE_SHIFT_THRESHOLD
        else "CANDIDATE_MODEL_WEIGHT_SCALE_SENSITIVITY_SMALL"
    )

    def add_numeric(
        metric: str,
        legacy_value: float,
        normalized_value: float,
        model_name: str | None = None,
        classification: str = "",
    ) -> None:
        difference = abs(float(normalized_value) - float(legacy_value))
        rows.append(
            {
                "result_label": RESULT_LABEL,
                "analysis_subtype": ANALYSIS_SUBTYPE,
                "metric": metric,
                "model_name": model_name,
                "legacy_raw_longwt": float(legacy_value),
                "mean1_normalized_longwt": float(normalized_value),
                "absolute_difference": difference,
                "changed": bool(difference > 0.0),
                "classification": classification,
            }
        )

    add_numeric(
        "selected_pipeline_oof_auroc",
        legacy["selected_pipeline_oof"]["raw"]["weighted_auroc"],
        normalized["selected_pipeline_oof"]["raw"]["weighted_auroc"],
    )
    add_numeric(
        "selected_pipeline_oof_auprc",
        legacy["selected_pipeline_oof"]["raw"]["weighted_auprc"],
        normalized["selected_pipeline_oof"]["raw"]["weighted_auprc"],
        classification=selected_pipeline_classification,
    )
    add_numeric(
        "selected_pipeline_calibrated_brier",
        legacy["selected_pipeline_oof"]["calibrated"]["weighted_brier"],
        normalized["selected_pipeline_oof"]["calibrated"]["weighted_brier"],
    )
    add_numeric(
        "selected_pipeline_weighted_selection_rate",
        legacy["selected_pipeline_oof"]["operational"]["selection_rate"],
        normalized["selected_pipeline_oof"]["operational"]["selection_rate"],
    )
    for model_name in CANDIDATE_MODEL_NAMES:
        legacy_row = next(row for row in legacy["candidate_model_oof_metrics"] if row["model_name"] == model_name)
        normalized_row = next(
            row for row in normalized["candidate_model_oof_metrics"] if row["model_name"] == model_name
        )
        add_numeric(
            "candidate_model_oof_auprc",
            legacy_row["weighted_auprc"],
            normalized_row["weighted_auprc"],
            model_name=model_name,
            classification=(
                "CANDIDATE_MODEL_WEIGHT_SCALE_SENSITIVITY_MATERIAL"
                if candidate_auprc_differences[model_name] >= WEIGHT_SCALE_SHIFT_THRESHOLD
                else "CANDIDATE_MODEL_WEIGHT_SCALE_SENSITIVITY_SMALL"
            ),
        )
        add_numeric(
            "model_selection_frequency",
            legacy["model_selection_frequency"][model_name],
            normalized["model_selection_frequency"][model_name],
            model_name=model_name,
        )

    def add_endpoint(metric: str, legacy_value: str, normalized_value: str, model_name: str | None = None) -> None:
        rows.append(
            {
                "result_label": RESULT_LABEL,
                "analysis_subtype": ANALYSIS_SUBTYPE,
                "metric": metric,
                "model_name": model_name,
                "legacy_raw_longwt": legacy_value,
                "mean1_normalized_longwt": normalized_value,
                "absolute_difference": None,
                "changed": bool(legacy_value != normalized_value),
                "classification": "ENDPOINT_AVAILABILITY_CHANGED" if legacy_value != normalized_value else "",
            }
        )

    for dimension in PRIMARY_AUDIT_DIMENSIONS:
        add_endpoint(
            f"fairness_{dimension}_endpoint_availability",
            str(legacy["fairness"]["dimensions"][dimension]["endpoint"]),
            str(normalized["fairness"]["dimensions"][dimension]["endpoint"]),
        )
    add_endpoint(
        "primary_fairness_endpoint_availability",
        str(legacy["fairness"]["primary_fairness_endpoint"]),
        str(normalized["fairness"]["primary_fairness_endpoint"]),
    )

    sensitivity = {
        "performance_shift_metric": "selected_pipeline_oof_auprc",
        "selected_pipeline_shift_metric": "selected_pipeline_oof_auprc",
        "selected_pipeline_oof_auprc_absolute_difference": selected_auprc_difference,
        "material_shift_threshold": WEIGHT_SCALE_SHIFT_THRESHOLD,
        "selected_pipeline_classification": selected_pipeline_classification,
        "maximum_candidate_model_auprc_absolute_difference": maximum_candidate_model_shift,
        "maximum_candidate_model_oof_auprc_absolute_difference": maximum_candidate_model_shift,
        "maximum_candidate_model_name": maximum_candidate_model_name,
        "candidate_model_classification": candidate_model_classification,
        "candidate_model_auprc_absolute_differences": candidate_auprc_differences,
        "candidate_model_auprc_classifications": {
            model_name: (
                "CANDIDATE_MODEL_WEIGHT_SCALE_SENSITIVITY_MATERIAL"
                if difference >= WEIGHT_SCALE_SHIFT_THRESHOLD
                else "CANDIDATE_MODEL_WEIGHT_SCALE_SENSITIVITY_SMALL"
            )
            for model_name, difference in candidate_auprc_differences.items()
        },
        "formal_hypothesis_testing": False,
        "fairness_endpoint_availability": {
            dimension: {
                WEIGHT_MODE_LEGACY: legacy["fairness"]["dimensions"][dimension]["endpoint"],
                WEIGHT_MODE_MEAN1: normalized["fairness"]["dimensions"][dimension]["endpoint"],
            }
            for dimension in PRIMARY_AUDIT_DIMENSIONS
        },
        "primary_fairness_endpoint_availability": {
            WEIGHT_MODE_LEGACY: legacy["fairness"]["primary_fairness_endpoint"],
            WEIGHT_MODE_MEAN1: normalized["fairness"]["primary_fairness_endpoint"],
        },
    }
    for row in rows:
        if row["metric"] == "selected_pipeline_oof_auprc":
            row["classification"] = selected_pipeline_classification
    return rows, sensitivity


def _validate_panel26_input(raw_df: pd.DataFrame) -> None:
    if not isinstance(raw_df, pd.DataFrame):
        raise TypeError("raw_df must be a pandas DataFrame")
    if "PANEL" not in raw_df.columns:
        raise ValueError("Panel 26 input requires the PANEL column")
    panel_values = raw_df["PANEL"].dropna().unique().tolist()
    if set(panel_values) != {PANEL_NUMBER}:
        raise ValueError("Robustness input must contain only Panel 26 records")


def run_panel26_oof_robustness(
    raw_df: pd.DataFrame,
    seed: int = SEED,
    source_path: str = "data/interim/meps/h244/h244.dta",
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Extract the authorized cohort and run both local fitting-weight modes."""
    _validate_panel26_input(raw_df)
    cohort = extract_meps_cohort(raw_df, panel_number=PANEL_NUMBER, allow_target=True)
    if cohort.panel != PANEL_NUMBER:
        raise ValueError("Extracted cohort is not Panel 26")
    if int(cohort.eligible_record_count) != EXPECTED_ELIGIBLE_RECORD_COUNT:
        raise ValueError("Panel 26 eligible cohort count does not match the authorized stop condition")
    positive_events = int((cohort.y == 1.0).sum())
    if positive_events != EXPECTED_POSITIVE_EVENTS:
        raise ValueError("Panel 26 positive event count does not match the authorized stop condition")
    design_quality = check_survey_design_quality(cohort.design)
    if design_quality.get("status") != "PASSED":
        raise ValueError(f"Survey design quality check failed: {design_quality}")

    mode_summaries: dict[str, dict[str, Any]] = {}
    for weight_mode in WEIGHT_MODES:
        in_memory_result = run_grouped_oof(cohort, weight_mode=weight_mode, seed=int(seed))
        mode_summaries[weight_mode] = summarize_grouped_oof(in_memory_result)
        # Explicitly release row-level arrays before starting the next mode;
        # no caller-facing artifact contains them.
        del in_memory_result

    sensitivity_rows, sensitivity_summary = compare_weight_modes(mode_summaries)
    legacy_summary = mode_summaries[WEIGHT_MODE_LEGACY]
    original_single_split = {
        "selected_model": "WeightedRandomForestClassifier",
        "test_auroc": 0.57292,
        "test_auprc": 0.05129,
        "test_selection_rate": 0.07753,
        "fairness": "NOT_ESTIMABLE_SUPPRESSED",
    }
    legacy_oof = legacy_summary["selected_pipeline_oof"]
    comparison = {
        "original_accepted_preliminary_untouched_test": original_single_split,
        "legacy_raw_longwt_oof": {
            "selected_pipeline_raw_auroc": legacy_oof["raw"]["weighted_auroc"],
            "selected_pipeline_raw_auprc": legacy_oof["raw"]["weighted_auprc"],
            "selected_pipeline_weighted_selection_rate": legacy_oof["operational"]["selection_rate"],
        },
        "legacy_oof_minus_original": {
            "auroc": legacy_oof["raw"]["weighted_auroc"] - original_single_split["test_auroc"],
            "auprc": legacy_oof["raw"]["weighted_auprc"] - original_single_split["test_auprc"],
            "selection_rate": legacy_oof["operational"]["selection_rate"] - original_single_split["test_selection_rate"],
        },
        "interpretation_boundary": "Descriptive comparison only; OOF development evidence is not temporal or final validation.",
    }
    cohort_summary = {
        "panel": PANEL_NUMBER,
        "raw_record_count": int(cohort.raw_record_count),
        "eligible_record_count": int(cohort.eligible_record_count),
        "positive_events": positive_events,
        "negative_events": int((cohort.y == 0.0).sum()),
        "duid_count": int(cohort.design["DUID"].nunique()),
        "weighted_observed_prevalence": weighted_mean(
            cohort.y.to_numpy(dtype=float), cohort.design["LONGWT"].to_numpy(dtype=float)
        ),
        "predictor_count": int(cohort.X.shape[1]),
        "continuous_feature_count": int(len(cohort.continuous_features)),
        "categorical_feature_count": int(len(cohort.categorical_features)),
        "survey_design_quality": design_quality,
    }
    primary_outer_folds = [
        {
            key: value
            for key, value in row.items()
            if key
            in {
                "outer_fold",
                "outer_test_record_count",
                "outer_test_positive_events",
                "outer_test_duid_count",
                "outer_test_weighted_prevalence",
            }
        }
        for row in legacy_summary["outer_folds"]
    ]
    top_summary = {
        "result_label": RESULT_LABEL,
        "analysis_subtype": ANALYSIS_SUBTYPE,
        "evidence_scope_note": EVIDENCE_SCOPE_NOTE,
        "fairness_label": FAIRNESS_LABEL,
        "calibrated_auroc_interpretation_note": CALIBRATED_AUROC_INTERPRETATION_NOTE,
        "panel": PANEL_NUMBER,
        "seed": int(seed),
        "source": {"path": source_path, "sha256": source_sha256},
        "protocol": {
            "outer_folds": OUTER_FOLDS,
            "outer_assignment_unit": "DUID",
            "outer_stratification": ["DUID outcome presence", "DUID total LONGWT quintile"],
            "inner_development_split": "DUID-grouped 75% inner train / 25% inner validation",
            "preprocessing_fit_partition": "inner train only",
            "candidate_models": list(CANDIDATE_MODEL_NAMES),
            "model_selection": "inner-validation weighted AUPRC, then weighted AUROC, then alphabetical model name",
            "hyperparameter_search": False,
            "platt_calibration_partition": "inner validation only",
            "calibrated_auroc_interpretation_note": CALIBRATED_AUROC_INTERPRETATION_NOTE,
            "capacity_threshold": "one 10% weighted-capacity threshold per outer fold, derived from inner validation only",
            "reported_metric_weights": "original LONGWT",
            "row_level_predictions_written": False,
        },
        "cohort": cohort_summary,
        "outer_folds": primary_outer_folds,
        "modes": mode_summaries,
        "weight_scale_sensitivity": sensitivity_summary,
        "comparison_with_original_single_split": comparison,
        "output_contract": {
            "aggregate_artifacts_only": True,
            "participant_level_prediction_files": [],
            "panel27_accessed": False,
            "hc217_accessed": False,
            "other_meps_artifacts_accessed": False,
        },
    }
    return {
        "oof_summary": top_summary,
        "candidate_model_oof_metrics": [
            row
            for mode in WEIGHT_MODES
            for row in mode_summaries[mode]["candidate_model_oof_metrics"]
        ],
        "fold_metrics": [
            row
            for mode in WEIGHT_MODES
            for row in mode_summaries[mode]["fold_metrics"]
        ],
        "model_selection_stability": [
            row
            for mode in WEIGHT_MODES
            for row in mode_summaries[mode]["model_selection_stability"]
        ],
        "fairness_audit": [
            {
                "result_label": RESULT_LABEL,
                "analysis_subtype": ANALYSIS_SUBTYPE,
                "weight_mode": mode,
                "fairness_label": FAIRNESS_LABEL,
                "primary_fairness_endpoint": mode_summaries[mode]["fairness"]["primary_fairness_endpoint"],
                "dimension": dimension,
                "row_type": "group",
                "group": group,
                "status": group_record["status"],
                "sample_n": group_record["sample_n"],
                "positive_events": group_record["positive_events"],
                "negative_events": group_record["negative_events"],
                "kish_effective_n": group_record["kish_effective_n"],
                "tpr": group_record["tpr"],
                "fpr": group_record["fpr"],
                "ppv": group_record["ppv"],
                "selection_rate": group_record["selection_rate"],
                "full_max_pairwise_tpr_gap": dimension_record["full_max_pairwise_tpr_gap"],
                "partial_max_pairwise_tpr_gap_unsuppressed_groups": dimension_record[
                    "partial_max_pairwise_tpr_gap_unsuppressed_groups"
                ],
                "dimension_endpoint": dimension_record["endpoint"],
                "suppression_reasons": json.dumps(group_record["suppression_reasons"], sort_keys=True),
            }
            for mode in WEIGHT_MODES
            for dimension, dimension_record in mode_summaries[mode]["fairness"]["dimensions"].items()
            for group, group_record in dimension_record["subgroups"].items()
        ]
        + [
            {
                "result_label": RESULT_LABEL,
                "analysis_subtype": ANALYSIS_SUBTYPE,
                "weight_mode": mode,
                "fairness_label": FAIRNESS_LABEL,
                "primary_fairness_endpoint": mode_summaries[mode]["fairness"]["primary_fairness_endpoint"],
                "dimension": dimension,
                "row_type": "dimension_summary",
                "group": None,
                "status": dimension_record["endpoint"],
                "sample_n": None,
                "positive_events": None,
                "negative_events": None,
                "kish_effective_n": None,
                "tpr": None,
                "fpr": None,
                "ppv": None,
                "selection_rate": None,
                "full_max_pairwise_tpr_gap": dimension_record["full_max_pairwise_tpr_gap"],
                "partial_max_pairwise_tpr_gap_unsuppressed_groups": dimension_record[
                    "partial_max_pairwise_tpr_gap_unsuppressed_groups"
                ],
                "dimension_endpoint": dimension_record["endpoint"],
                "suppression_reasons": None,
            }
            for mode in WEIGHT_MODES
            for dimension, dimension_record in mode_summaries[mode]["fairness"]["dimensions"].items()
        ],
        "weight_scale_sensitivity": sensitivity_rows,
        "calibration_bins": [
            row
            for mode in WEIGHT_MODES
            for row in mode_summaries[mode]["calibration_bins"]
        ],
    }
