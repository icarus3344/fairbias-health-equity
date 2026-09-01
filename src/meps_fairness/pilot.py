"""Single-panel HC-244 / Panel 26 preliminary development pilot.

This module deliberately keeps the pilot separate from the accepted pipeline.
It uses the existing cohort, split, preprocessing, model, calibration, and
metric implementations, but changes the evaluation boundary so that the
existing ``cal`` partition is treated as an untouched final development test
partition.  The validation partition is used for model selection, Platt
calibration, and threshold freezing.

The functions in this module do not load data.  The companion script owns the
HC-244-only input preflight and Stata loading boundary.
"""

from __future__ import annotations

import itertools
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from meps_fairness.data.cohort import extract_meps_cohort
from meps_fairness.data.preprocess import MEPSPreprocessor, check_survey_design_quality
from meps_fairness.data.split import PartitionData, split_panel26_duid_grouped
from meps_fairness.evaluation.calibration import SurveyWeightedPlattCalibrator
from meps_fairness.evaluation.metrics import (
    fixed_threshold_metrics,
    weighted_auprc,
    weighted_auroc,
    weighted_brier_score,
    weighted_calibration_stats,
)
from meps_fairness.models.baseline import (
    WeightedGradientBoostingClassifier,
    WeightedLogisticClassifier,
    WeightedRandomForestClassifier,
)


RESULT_LABEL = "PRELIMINARY_SINGLE_PANEL_DEVELOPMENT_ONLY"
PANEL_NUMBER = 26
SEED = 20260828
CAPACITY_FRACTION = 0.10
EXPECTED_ELIGIBLE_RECORD_COUNT = 2882
EXPECTED_POSITIVE_EVENTS = 136
MIN_N = 100
MIN_POS = 20
MIN_NEG = 20
MIN_KISH = 50.0
PRIMARY_AUDIT_DIMENSIONS: tuple[str, ...] = ("RACETHX", "SEX")

LIMITATION_STATEMENTS: tuple[str, ...] = (
    "NOT multi-panel evidence",
    "NOT population-pooled evidence",
    "NOT temporal holdout evidence",
    "NOT Panel 27 evidence",
    "NOT final fairness evidence",
    "NOT publication-ready inference",
)
EVIDENCE_SCOPE_NOTE = "; ".join(LIMITATION_STATEMENTS) + "."


def evidence_scope() -> dict[str, Any]:
    """Return the immutable claim boundary copied into every result artifact."""
    return {
        "result_label": RESULT_LABEL,
        "limitations": list(LIMITATION_STATEMENTS),
        "scope_note": EVIDENCE_SCOPE_NOTE,
    }

def _array(name: str, values: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _aligned_arrays(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_prob: Sequence[float] | np.ndarray | pd.Series,
    sample_weight: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = _array("y_true", y_true)
    p = _array("y_prob", y_prob)
    w = _array("sample_weight", sample_weight)
    if not (len(y) == len(p) == len(w)):
        raise ValueError("y_true, y_prob, and sample_weight must have equal lengths")
    if len(y) == 0:
        raise ValueError("Pilot metrics require at least one record")
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("y_true must contain only binary 0/1 values")
    if (w <= 0).any():
        raise ValueError("sample_weight must be strictly positive")
    return y, p, w


def weighted_mean(values: Sequence[float] | np.ndarray, weights: Sequence[float] | np.ndarray) -> float:
    """Compute a finite positive-weight mean for aggregate reporting."""
    values_arr = _array("values", values)
    weights_arr = _array("weights", weights)
    if len(values_arr) != len(weights_arr):
        raise ValueError("values and weights must have equal lengths")
    total_weight = float(weights_arr.sum())
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive value")
    return float(np.sum(values_arr * weights_arr) / total_weight)


def _partition_summary(partition: PartitionData) -> dict[str, Any]:
    y = partition.y.to_numpy(dtype=float)
    w = partition.design["LONGWT"].to_numpy(dtype=float)
    return {
        "name": partition.name,
        "record_count": int(partition.record_count),
        "duid_count": int(partition.duid_count),
        "positive_events": int((y == 1.0).sum()),
        "negative_events": int((y == 0.0).sum()),
        "total_weight": float(w.sum()),
        "kish_effective_n": float(partition.kish_effective_n),
        "weighted_event_prevalence": weighted_mean(y, w),
    }


def default_model_suite(seed: int = SEED) -> list[dict[str, Any]]:
    """Construct the three existing conservative weighted baselines.

    The returned order is alphabetical by model name so that a tie-break is
    deterministic and explicit.  No hyperparameter search is performed.
    """
    return [
        {
            "model_name": "WeightedGradientBoostingClassifier",
            "model": WeightedGradientBoostingClassifier(
                max_iter=100,
                max_depth=4,
                min_samples_leaf=20,
                random_state=seed,
            ),
            "configuration": {
                "max_iter": 100,
                "max_depth": 4,
                "min_samples_leaf": 20,
                "random_state": seed,
            },
        },
        {
            "model_name": "WeightedLogisticClassifier",
            "model": WeightedLogisticClassifier(
                C=1.0,
                max_iter=10000,
                random_state=seed,
            ),
            "configuration": {
                "C": 1.0,
                "max_iter": 10000,
                "random_state": seed,
            },
        },
        {
            "model_name": "WeightedRandomForestClassifier",
            "model": WeightedRandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=10,
                random_state=seed,
            ),
            "configuration": {
                "n_estimators": 100,
                "max_depth": 6,
                "min_samples_leaf": 10,
                "random_state": seed,
            },
        },
    ]


def rank_models_by_validation(
    validation_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rank models using validation AUPRC, AUROC, then alphabetical name."""
    if not validation_results:
        raise ValueError("At least one validation result is required")

    normalized: list[dict[str, Any]] = []
    for row in validation_results:
        for required in (
            "model_name",
            "validation_weighted_auprc",
            "validation_weighted_auroc",
        ):
            if required not in row:
                raise ValueError(f"Validation result is missing {required}")
        auprc = float(row["validation_weighted_auprc"])
        auroc = float(row["validation_weighted_auroc"])
        if not (np.isfinite(auprc) and np.isfinite(auroc)):
            raise ValueError("Validation selection metrics must be finite")
        item = dict(row)
        item["model_name"] = str(item["model_name"])
        item["validation_weighted_auprc"] = auprc
        item["validation_weighted_auroc"] = auroc
        normalized.append(item)

    ranked = sorted(
        normalized,
        key=lambda item: (
            -item["validation_weighted_auprc"],
            -item["validation_weighted_auroc"],
            item["model_name"],
        ),
    )
    for rank, item in enumerate(ranked, start=1):
        item["validation_selection_rank"] = rank
    return ranked


def select_model_by_validation(
    validation_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select exactly one model from validation results only."""
    return rank_models_by_validation(validation_results)[0]


def fit_validation_calibrator_and_freeze_threshold(
    validation_raw_probs: Sequence[float] | np.ndarray,
    validation_y: Sequence[float] | np.ndarray | pd.Series,
    validation_weights: Sequence[float] | np.ndarray | pd.Series,
    seed: int = SEED,
    capacity_fraction: float = CAPACITY_FRACTION,
) -> tuple[SurveyWeightedPlattCalibrator, np.ndarray, float]:
    """Fit Platt scaling on validation and freeze its validation threshold.

    No test argument exists in this function by design.  This makes the
    threshold provenance mechanically separate from final-test probabilities
    and outcomes.
    """
    raw = _array("validation_raw_probs", validation_raw_probs)
    y, _, weights = _aligned_arrays(validation_y, raw, validation_weights)
    if len(np.unique(y)) < 2:
        raise ValueError("Validation partition must contain both outcome classes for Platt calibration")
    if not (0.0 < float(capacity_fraction) < 1.0):
        raise ValueError("capacity_fraction must be strictly between 0 and 1")

    calibrator = SurveyWeightedPlattCalibrator(random_state=seed)
    calibrator.fit(
        raw,
        y,
        sample_weight=weights,
        freeze_capacity_fraction=capacity_fraction,
    )
    validation_calibrated_probs = calibrator.predict_proba(raw)
    threshold = calibrator.frozen_threshold_10pct_
    if threshold is None or not np.isfinite(float(threshold)):
        raise ValueError("Validation calibrator did not produce a finite frozen threshold")
    threshold_float = float(threshold)
    if not 0.0 <= threshold_float <= 1.0:
        raise ValueError("Frozen calibrated threshold must be within [0, 1]")
    return calibrator, validation_calibrated_probs, threshold_float


def fixed_threshold_metrics_with_fpr(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_prob: Sequence[float] | np.ndarray,
    threshold: float,
    sample_weight: Sequence[float] | np.ndarray | pd.Series,
) -> dict[str, float]:
    """Return existing fixed-threshold metrics plus FPR and naming aliases."""
    y, p, weights = _aligned_arrays(y_true, y_prob, sample_weight)
    threshold_float = float(threshold)
    if not np.isfinite(threshold_float) or not 0.0 <= threshold_float <= 1.0:
        raise ValueError("threshold must be finite and within [0, 1]")
    base = fixed_threshold_metrics(y, p, threshold_float, weights)
    fpr = max(0.0, min(1.0, 1.0 - float(base["tnr"])))
    return {
        "tpr": float(base["tpr"]),
        "recall": float(base["tpr"]),
        "fpr": fpr,
        "tnr": float(base["tnr"]),
        "ppv": float(base["ppv"]),
        "precision": float(base["ppv"]),
        "f1": float(base["f1"]),
        "selection_rate": float(base["selection_rate"]),
    }


def _kish_effective_n(weights: np.ndarray) -> float:
    sum_weights = float(weights.sum())
    sum_weights_squared = float(np.sum(weights**2))
    if sum_weights_squared <= 0:
        return 0.0
    return float((sum_weights**2) / sum_weights_squared)


def fixed_threshold_subgroup_audit(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_prob: Sequence[float] | np.ndarray,
    audit_df: pd.DataFrame,
    threshold: float,
    sample_weight: Sequence[float] | np.ndarray | pd.Series,
    dimensions: Sequence[str] = PRIMARY_AUDIT_DIMENSIONS,
    min_n: int = MIN_N,
    min_pos: int = MIN_POS,
    min_neg: int = MIN_NEG,
    min_kish: float = MIN_KISH,
) -> dict[str, Any]:
    """Audit subgroups at a supplied threshold without recomputing capacity.

    This is intentionally isolated from the existing capacity helper, whose
    historical contract derives a fresh threshold from the supplied data.
    """
    y, p, weights = _aligned_arrays(y_true, y_prob, sample_weight)
    if len(audit_df) != len(y):
        raise ValueError("audit_df and metric arrays must have equal lengths")
    threshold_float = float(threshold)
    if not np.isfinite(threshold_float) or not 0.0 <= threshold_float <= 1.0:
        raise ValueError("threshold must be finite and within [0, 1]")
    if min_n < 1 or min_pos < 0 or min_neg < 0 or min_kish < 0:
        raise ValueError("Suppression thresholds must be non-negative and min_n must be positive")

    dimension_results: dict[str, Any] = {}
    dimension_gaps: list[float] = []
    for dimension in dimensions:
        if dimension not in audit_df.columns:
            raise ValueError(f"Missing required audit dimension: {dimension}")
        groups = audit_df[dimension].reset_index(drop=True)
        if groups.isna().any():
            raise ValueError(f"Audit dimension {dimension} contains missing values")

        subgroup_results: dict[str, Any] = {}
        valid_tprs: dict[str, float] = {}
        group_values = sorted(groups.unique().tolist(), key=lambda value: str(value))
        for group_value in group_values:
            mask = (groups == group_value).to_numpy(dtype=bool)
            y_group = y[mask]
            p_group = p[mask]
            w_group = weights[mask]
            sample_n = int(mask.sum())
            positive_events = int((y_group == 1.0).sum())
            negative_events = int((y_group == 0.0).sum())
            kish = _kish_effective_n(w_group)
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
                "positives": positive_events,
                "negatives": negative_events,
                "kish_effective_n": kish,
                "threshold": threshold_float,
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

            fixed = fixed_threshold_metrics_with_fpr(y_group, p_group, threshold_float, w_group)
            subgroup_results[group_key] = {
                **common,
                "status": "Unsuppressed",
                "suppression_reasons": [],
                "tpr": fixed["tpr"],
                "fpr": fixed["fpr"],
                "ppv": fixed["ppv"],
                "selection_rate": fixed["selection_rate"],
            }
            valid_tprs[group_key] = fixed["tpr"]

        pairwise_gaps = [
            abs(valid_tprs[left] - valid_tprs[right])
            for left, right in itertools.combinations(sorted(valid_tprs), 2)
        ]
        max_gap = float(max(pairwise_gaps)) if pairwise_gaps else None
        if max_gap is not None:
            dimension_gaps.append(max_gap)
        dimension_results[dimension] = {
            "dimension": dimension,
            "threshold": threshold_float,
            "subgroups": subgroup_results,
            "unsuppressed_subgroups_count": len(valid_tprs),
            "max_pairwise_tpr_gap": max_gap,
            "max_tpr_gap": max_gap,
        }

    if dimension_gaps:
        primary_gap: float | None = float(max(dimension_gaps))
        endpoint_label = "ESTIMABLE"
    else:
        primary_gap = None
        endpoint_label = "PRIMARY_FAIRNESS_ENDPOINT = NOT_ESTIMABLE_SUPPRESSED"

    return {
        "result_label": RESULT_LABEL,
        "threshold_source": "validation_calibrated_probabilities",
        "threshold_recomputed_on_test": False,
        "threshold": threshold_float,
        "dimensions": dimension_results,
        "primary_fairness_max_tpr_gap": primary_gap,
        "primary_fairness_endpoint": endpoint_label,
        "primary_fairness_endpoint_status": endpoint_label,
    }


def _probability_metrics(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray | pd.Series,
) -> dict[str, Any]:
    y, p, w = _aligned_arrays(y_true, probabilities, weights)
    return {
        "weighted_auroc": weighted_auroc(y, p, w),
        "weighted_auprc": weighted_auprc(y, p, w),
        "weighted_brier_score": weighted_brier_score(y, p, w),
        "calibration": weighted_calibration_stats(y, p, w),
        "predicted_mean_risk": weighted_mean(p, w),
        "weighted_observed_event_prevalence": weighted_mean(y, w),
    }


def _calibration_bins(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray | pd.Series,
    probability_type: str,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    y, p, w = _aligned_arrays(y_true, probabilities, weights)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for bin_index in range(n_bins):
        lower = float(edges[bin_index])
        upper = float(edges[bin_index + 1])
        mask = (p >= lower) & (p <= upper if bin_index == n_bins - 1 else p < upper)
        if not mask.any():
            rows.append(
                {
                    "result_label": RESULT_LABEL,
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
        w_bin = w[mask]
        y_bin = y[mask]
        p_bin = p[mask]
        rows.append(
            {
                "result_label": RESULT_LABEL,
                "probability_type": probability_type,
                "bin": bin_index + 1,
                "lower_edge": lower,
                "upper_edge": upper,
                "sample_n": int(mask.sum()),
                "weighted_n": float(w_bin.sum()),
                "weighted_predicted_mean": weighted_mean(p_bin, w_bin),
                "weighted_observed_event_prevalence": weighted_mean(y_bin, w_bin),
            }
        )
    return rows


def _validate_panel26_frame(raw_df: pd.DataFrame) -> None:
    if not isinstance(raw_df, pd.DataFrame):
        raise TypeError("raw_df must be a pandas DataFrame")
    if "PANEL" not in raw_df.columns:
        raise ValueError("Panel 26 pilot requires the PANEL column")
    panel_values = raw_df["PANEL"].dropna().unique().tolist()
    if set(panel_values) != {PANEL_NUMBER}:
        raise ValueError("Pilot input must contain only Panel 26 records")


def _validate_panel26_stop_condition(cohort: Any) -> None:
    observed_eligible_record_count = int(cohort.eligible_record_count)
    observed_positive_events = int((cohort.y == 1.0).sum())
    if observed_eligible_record_count != EXPECTED_ELIGIBLE_RECORD_COUNT:
        raise ValueError(
            "Panel 26 pilot stop condition failed: expected "
            f"{EXPECTED_ELIGIBLE_RECORD_COUNT} eligible records, observed "
            f"{observed_eligible_record_count}"
        )
    if observed_positive_events != EXPECTED_POSITIVE_EVENTS:
        raise ValueError(
            "Panel 26 pilot stop condition failed: expected "
            f"{EXPECTED_POSITIVE_EVENTS} positive events, observed "
            f"{observed_positive_events}"
        )


def run_panel26_pilot(
    raw_df: pd.DataFrame,
    seed: int = SEED,
    source_path: str = "data/interim/meps/h244/h244.dta",
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Run the authorized single-panel pilot in memory and return aggregates.

    The caller must perform the repository/data preflight before loading the
    source dataframe.  This function accepts only a single Panel 26 frame and
    never accepts or loads another panel.
    """
    _validate_panel26_frame(raw_df)
    seed = int(seed)

    cohort = extract_meps_cohort(raw_df, panel_number=PANEL_NUMBER, allow_target=True)
    if cohort.panel != PANEL_NUMBER:
        raise ValueError("Extracted cohort is not Panel 26")
    _validate_panel26_stop_condition(cohort)
    if cohort.eligible_record_count == 0:
        raise ValueError("Panel 26 pilot cohort is empty")
    if len(np.unique(cohort.y.to_numpy(dtype=float))) < 2:
        raise ValueError("Panel 26 pilot cohort must contain both outcome classes")

    split = split_panel26_duid_grouped(cohort, seed=seed)
    split.validate_zero_leakage()
    if len(np.unique(split.train.y.to_numpy(dtype=float))) < 2:
        raise ValueError("Training partition must contain both outcome classes")
    if len(np.unique(split.val.y.to_numpy(dtype=float))) < 2:
        raise ValueError("Validation partition must contain both outcome classes")
    if split.cal.record_count == 0:
        raise ValueError("Final development test partition is empty")

    design_quality = check_survey_design_quality(cohort.design)
    if design_quality.get("status") != "PASSED":
        raise ValueError(f"Survey design quality check failed: {design_quality}")

    # The preprocessor is fitted exactly once on train.  Validation and the
    # existing cal partition are transformed with that fitted object.
    preprocessor = MEPSPreprocessor(
        continuous_features=cohort.continuous_features,
        categorical_features=cohort.categorical_features,
    )
    X_train = preprocessor.fit_transform(split.train.X)
    X_val = preprocessor.transform(split.val.X)
    X_test = preprocessor.transform(split.cal.X)

    train_y = split.train.y.to_numpy(dtype=float)
    val_y = split.val.y.to_numpy(dtype=float)
    test_y = split.cal.y.to_numpy(dtype=float)
    train_w = split.train.design["LONGWT"].to_numpy(dtype=float)
    val_w = split.val.design["LONGWT"].to_numpy(dtype=float)
    test_w = split.cal.design["LONGWT"].to_numpy(dtype=float)

    validation_results: list[dict[str, Any]] = []
    models_by_name: dict[str, Any] = {}
    validation_raw_by_name: dict[str, np.ndarray] = {}
    model_configurations: list[dict[str, Any]] = []

    # Every model is fit on train only.  The validation AUPRC is the sole
    # primary selection score; AUROC and alphabetical name are tie-breakers.
    for specification in default_model_suite(seed):
        model_name = str(specification["model_name"])
        model = specification["model"]
        model.fit(X_train, train_y, sample_weight=train_w)
        val_raw = model.predict_proba(X_val)
        models_by_name[model_name] = model
        validation_raw_by_name[model_name] = val_raw
        model_configurations.append(
            {"model_name": model_name, "configuration": dict(specification["configuration"])}
        )
        validation_results.append(
            {
                "result_label": RESULT_LABEL,
                "model_name": model_name,
                "configuration": dict(specification["configuration"]),
                "validation_weighted_auroc": weighted_auroc(val_y, val_raw, val_w),
                "validation_weighted_auprc": weighted_auprc(val_y, val_raw, val_w),
                "validation_weighted_brier_score": weighted_brier_score(val_y, val_raw, val_w),
                "validation_predicted_mean_risk": weighted_mean(val_raw, val_w),
                "validation_weighted_observed_event_prevalence": weighted_mean(val_y, val_w),
            }
        )

    ranked_validation_results = rank_models_by_validation(validation_results)
    selected = ranked_validation_results[0]
    selected_model_name = str(selected["model_name"])
    selected_model = models_by_name[selected_model_name]
    selected_val_raw = validation_raw_by_name[selected_model_name]

    calibrator, selected_val_calibrated, frozen_threshold = fit_validation_calibrator_and_freeze_threshold(
        selected_val_raw,
        val_y,
        val_w,
        seed=seed,
        capacity_fraction=CAPACITY_FRACTION,
    )
    selected_test_raw = selected_model.predict_proba(X_test)
    selected_test_calibrated = calibrator.predict_proba(selected_test_raw)

    validation_raw_metrics = _probability_metrics(val_y, selected_val_raw, val_w)
    validation_calibrated_metrics = _probability_metrics(val_y, selected_val_calibrated, val_w)
    test_raw_metrics = _probability_metrics(test_y, selected_test_raw, test_w)
    test_calibrated_metrics = _probability_metrics(test_y, selected_test_calibrated, test_w)
    validation_operational = fixed_threshold_metrics_with_fpr(
        val_y, selected_val_calibrated, frozen_threshold, val_w
    )
    test_operational = fixed_threshold_metrics_with_fpr(
        test_y, selected_test_calibrated, frozen_threshold, test_w
    )

    fairness = fixed_threshold_subgroup_audit(
        test_y,
        selected_test_calibrated,
        split.cal.audit,
        frozen_threshold,
        test_w,
        dimensions=PRIMARY_AUDIT_DIMENSIONS,
    )

    cohort_summary = {
        **evidence_scope(),
        "panel": PANEL_NUMBER,
        "source": {
            "path": source_path,
            "sha256": source_sha256,
        },
        "raw_record_count": int(cohort.raw_record_count),
        "eligible_record_count": int(cohort.eligible_record_count),
        "positive_events": int((cohort.y == 1.0).sum()),
        "negative_events": int((cohort.y == 0.0).sum()),
        "weighted_event_prevalence": weighted_mean(
            cohort.y.to_numpy(dtype=float), cohort.design["LONGWT"].to_numpy(dtype=float)
        ),
        "predictor_count": int(cohort.X.shape[1]),
        "continuous_feature_count": int(len(cohort.continuous_features)),
        "categorical_feature_count": int(len(cohort.categorical_features)),
        "survey_design_quality": design_quality,
        "split": {
            "seed": seed,
            "split_assignment_hash": split.split_assignment_hash,
            "method": "DUID-grouped stratified 60/20/20 split",
            "partition_roles": {
                "train": "model_fitting_only",
                "validation": "model_selection_plus_platt_calibration_plus_threshold_freezing",
                "cal": "untouched_final_development_test",
            },
            "partitions": [
                _partition_summary(split.train),
                _partition_summary(split.val),
                _partition_summary(split.cal),
            ],
        },
    }

    test_metrics = {
        **evidence_scope(),
        "panel": PANEL_NUMBER,
        "partition": "cal",
        "partition_role": "untouched_final_development_test",
        "selected_model": selected_model_name,
        "threshold": {
            "value": frozen_threshold,
            "capacity_fraction": CAPACITY_FRACTION,
            "source": "validation_calibrated_probabilities",
            "frozen_before_final_test_evaluation": True,
            "recomputed_from_final_test": False,
            "final_test_outcomes_used_for_threshold": False,
            "final_test_probabilities_used_for_threshold": False,
        },
        "validation": {
            "selected_model_raw": validation_raw_metrics,
            "selected_model_calibrated": validation_calibrated_metrics,
            "frozen_threshold_weighted_selection_rate": validation_operational["selection_rate"],
            "operational_at_frozen_threshold": validation_operational,
        },
        "test": {
            "raw": test_raw_metrics,
            "calibrated": test_calibrated_metrics,
            "operational_at_frozen_validation_threshold": test_operational,
            "weighted_selection_rate_at_frozen_threshold": test_operational["selection_rate"],
        },
    }

    pilot_summary = {
        **evidence_scope(),
        "panel": PANEL_NUMBER,
        "seed": seed,
        "source": {
            "path": source_path,
            "sha256": source_sha256,
        },
        "protocol": {
            "train_role": "model fitting only",
            "validation_role": "model selection, Platt calibration, and 10% weighted-capacity threshold freezing",
            "cal_role": "untouched final development test",
            "refit_on_train_plus_validation": False,
            "hyperparameter_search": False,
            "selection_metric": "validation weighted AUPRC",
            "tie_break": ["validation weighted AUROC", "alphabetical model name"],
            "fairness_threshold_source": "same frozen validation threshold",
        },
        "selected_model": selected_model_name,
        "selected_model_validation_rank": int(selected["validation_selection_rank"]),
        "model_configurations": model_configurations,
        "validation_model_comparison": ranked_validation_results,
        "frozen_threshold": frozen_threshold,
        "validation_weighted_selection_rate": validation_operational["selection_rate"],
        "test_weighted_selection_rate": test_operational["selection_rate"],
        "primary_fairness_endpoint": fairness["primary_fairness_endpoint"],
        "primary_fairness_max_tpr_gap": fairness["primary_fairness_max_tpr_gap"],
    }

    calibration_bins = [
        {
            **row,
            "partition": "cal",
            "partition_role": "untouched_final_development_test",
        }
        for probability_type, probabilities in (
            ("raw", selected_test_raw),
            ("calibrated", selected_test_calibrated),
        )
        for row in _calibration_bins(test_y, probabilities, test_w, probability_type)
    ]

    return {
        "pilot_summary": pilot_summary,
        "cohort_summary": cohort_summary,
        "model_validation_comparison": ranked_validation_results,
        "test_metrics": test_metrics,
        "fairness_audit": {
            **evidence_scope(),
            "panel": PANEL_NUMBER,
            "partition": "cal",
            "partition_role": "untouched_final_development_test",
            "selected_model": selected_model_name,
            **fairness,
        },
        "calibration_bins": calibration_bins,
    }
