"""Survey-weighted evaluation, calibration, fairness metrics, and design-aware inference."""

from __future__ import annotations

from meps_fairness.evaluation.calibration import (
    SurveyWeightedPlattCalibrator,
    find_weighted_capacity_threshold,
)
from meps_fairness.evaluation.inference import (
    BootstrapInferenceResult,
    stratified_psu_bootstrap_inference,
)
from meps_fairness.evaluation.metrics import (
    capacity_metrics,
    fixed_threshold_metrics,
    primary_fairness_endpoint,
    subgroup_audit_metrics,
    weighted_auprc,
    weighted_auroc,
    weighted_brier_score,
    weighted_calibration_stats,
)

__all__ = [
    "SurveyWeightedPlattCalibrator",
    "find_weighted_capacity_threshold",
    "weighted_auroc",
    "weighted_auprc",
    "weighted_brier_score",
    "weighted_calibration_stats",
    "capacity_metrics",
    "fixed_threshold_metrics",
    "subgroup_audit_metrics",
    "primary_fairness_endpoint",
    "BootstrapInferenceResult",
    "stratified_psu_bootstrap_inference",
]
