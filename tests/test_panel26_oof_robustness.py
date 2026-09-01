"""Focused synthetic tests for the Panel 26 grouped OOF contract."""

from __future__ import annotations

import inspect
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meps_fairness.crossfit import (  # noqa: E402
    ANALYSIS_SUBTYPE,
    CALIBRATED_AUROC_INTERPRETATION_NOTE,
    CANDIDATE_MODEL_NAMES,
    FAIRNESS_LABEL,
    GroupedOOFResult,
    WEIGHT_MODE_MEAN1,
    WEIGHT_MODE_LEGACY,
    audit_oof_binary_decisions,
    assign_grouped_folds,
    compare_weight_modes,
    run_grouped_oof,
    summarize_grouped_oof,
)
from meps_fairness.data.cohort import CohortData  # noqa: E402
from meps_fairness.evaluation.metrics import weighted_auprc  # noqa: E402
from scripts.run_panel26_oof_robustness import (  # noqa: E402
    ARTIFACT_NAMES,
    _write_aggregate_artifacts,
)


def _synthetic_cohort(n_groups: int = 40, rows_per_group: int = 4) -> CohortData:
    group_numbers = np.arange(1000, 1000 + n_groups)
    group_positive = (np.arange(n_groups) % 5 == 0).astype(float)
    duids = np.repeat(group_numbers, rows_per_group)
    y = np.repeat(group_positive, rows_per_group)
    row_index = np.arange(len(duids))
    x = np.where(y == 1.0, 0.78 + (row_index % 4) * 0.02, 0.18 + (row_index % 4) * 0.02)
    longwt = 1.0 + (row_index % 9) * 0.75
    X = pd.DataFrame({"x": x}, index=row_index)
    design = pd.DataFrame(
        {
            "DUID": duids,
            "LONGWT": longwt,
            "VARSTR": 1 + row_index % 4,
            "VARPSU": 10 + row_index % 8,
        },
        index=row_index,
    )
    audit = pd.DataFrame(
        {
            "RACETHX": 1 + (duids - 1000) % 2,
            "SEX": 1 + (duids - 1000) % 2,
        },
        index=row_index,
    )
    y_series = pd.Series(y, index=row_index, name="y")
    return CohortData(
        panel=26,
        raw_record_count=len(duids),
        eligible_record_count=len(duids),
        X=X,
        y=y_series,
        y_prolonged=y_series.copy(),
        y_yearend=y_series.copy(),
        design=design,
        audit=audit,
        continuous_features=("x",),
        categorical_features=(),
    )


class _RecordingPreprocessor:
    def __init__(self, instances: list[_RecordingPreprocessor], **_: object) -> None:
        self.instances = instances
        self.fit_indices: tuple[int, ...] = ()
        instances.append(self)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit_indices = tuple(int(index) for index in X.index)
        return X[["x"]].copy()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[["x"]].copy()


class _ScoreModel:
    def __init__(self, model_name: str, offset: float, seen: list[_ScoreModel]) -> None:
        self.model_name = model_name
        self.offset = offset
        self.fit_weight_mean: float | None = None
        self.fit_indices: tuple[int, ...] = ()
        seen.append(self)

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray) -> _ScoreModel:
        self.fit_indices = tuple(int(index) for index in X.index)
        self.fit_weight_mean = float(np.mean(sample_weight))
        assert len(X) == len(y) == len(sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        values = X["x"].to_numpy(dtype=float) + self.offset
        return np.clip(values, 0.001, 0.999)


def _fake_model_suite(seen: list[_ScoreModel]):
    offsets = {
        "WeightedGradientBoostingClassifier": 0.00,
        "WeightedLogisticClassifier": -0.02,
        "WeightedRandomForestClassifier": 0.02,
    }

    def factory(_: int):
        return [
            {
                "model_name": name,
                "model": _ScoreModel(name, offsets[name], seen),
                "configuration": {"synthetic": True, "offset": offsets[name]},
            }
            for name in CANDIDATE_MODEL_NAMES
        ]

    return factory


class _IdentityCalibrator:
    def predict_proba(self, raw_probabilities: np.ndarray) -> np.ndarray:
        return np.asarray(raw_probabilities, dtype=float)


def _spy_calibrator(calls: list[dict[str, object]]):
    def fit_fn(
        validation_raw_probs: np.ndarray,
        validation_y: np.ndarray,
        validation_weights: np.ndarray,
        seed: int,
        capacity_fraction: float,
    ) -> tuple[_IdentityCalibrator, np.ndarray, float]:
        calls.append(
            {
                "raw": np.asarray(validation_raw_probs, dtype=float).copy(),
                "y": np.asarray(validation_y, dtype=float).copy(),
                "weights": np.asarray(validation_weights, dtype=float).copy(),
                "seed": seed,
                "capacity_fraction": capacity_fraction,
            }
        )
        threshold = 0.20 + 0.05 * (len(calls) - 1)
        return _IdentityCalibrator(), np.asarray(validation_raw_probs, dtype=float), threshold

    return fit_fn


def _run_synthetic_mean1() -> tuple[GroupedOOFResult, list[_RecordingPreprocessor], list[_ScoreModel], list[dict[str, object]]]:
    cohort = _synthetic_cohort()
    preprocessors: list[_RecordingPreprocessor] = []
    models: list[_ScoreModel] = []
    calibrator_calls: list[dict[str, object]] = []
    result = run_grouped_oof(
        cohort,
        weight_mode=WEIGHT_MODE_MEAN1,
        seed=20260828,
        preprocessor_factory=lambda **kwargs: _RecordingPreprocessor(preprocessors, **kwargs),
        model_suite_factory=_fake_model_suite(models),
        calibrator_fit_fn=_spy_calibrator(calibrator_calls),
    )
    return result, preprocessors, models, calibrator_calls


def test_group_assignment_is_deterministic_and_one_fold_per_duid() -> None:
    cohort = _synthetic_cohort()
    assignment_a = assign_grouped_folds(
        cohort.design["DUID"], cohort.y, cohort.design["LONGWT"], n_splits=5, seed=20260828
    )
    assignment_b = assign_grouped_folds(
        cohort.design["DUID"], cohort.y, cohort.design["LONGWT"], n_splits=5, seed=20260828
    )
    assert assignment_a.assignment_hash == assignment_b.assignment_hash
    assert np.array_equal(assignment_a.row_fold_ids, assignment_b.row_fold_ids)
    assert set(assignment_a.group_to_fold) == set(cohort.design["DUID"].unique())
    for duid in cohort.design["DUID"].unique():
        assert len(set(assignment_a.row_fold_ids[cohort.design["DUID"].to_numpy() == duid])) == 1


def test_oof_boundaries_coverage_preprocessing_calibration_thresholds_and_mean1_weights() -> None:
    result, preprocessors, models, calibrator_calls = _run_synthetic_mean1()
    assert result.n_outer_folds == 5
    assert np.array_equal(result.prediction_counts, np.ones(len(result.y), dtype=int))
    assert all(np.isfinite(values).all() for values in result.candidate_raw_oof.values())
    assert len(preprocessors) == 5
    assert len(calibrator_calls) == 5
    assert len(models) == 15

    all_outer_test_indices: set[int] = set()
    for partition, preprocessor in zip(result.fold_partitions, preprocessors):
        outer_test = set(int(index) for index in partition["outer_test_indices"])
        inner_train = set(int(index) for index in partition["inner_train_indices"])
        inner_validation = set(int(index) for index in partition["inner_validation_indices"])
        all_outer_test_indices.update(outer_test)
        assert not outer_test & inner_train
        assert not outer_test & inner_validation
        assert not inner_train & inner_validation
        assert not outer_test & set(preprocessor.fit_indices)
    assert all_outer_test_indices == set(range(len(result.y)))

    assert all(np.isclose(float(call["weights"].mean()), 1.0) for call in calibrator_calls)
    assert all(model.fit_weight_mean is not None and np.isclose(model.fit_weight_mean, 1.0) for model in models)
    assert np.allclose(
        result.selected_decisions_oof,
        (result.selected_calibrated_oof >= result.thresholds_oof).astype(float),
    )
    assert len(np.unique(result.thresholds_oof)) == 5

    summary = summarize_grouped_oof(result)
    assert summary["selected_pipeline_oof"]["threshold_provenance"]["global_oof_threshold_derived"] is False
    assert len(summary["selected_pipeline_oof"]["threshold_provenance"]["thresholds_by_outer_fold"]) == 5
    assert (
        summary["selected_pipeline_oof"]["calibrated"]["auroc_interpretation_note"]
        == CALIBRATED_AUROC_INTERPRETATION_NOTE
    )
    assert "threshold" not in inspect.signature(audit_oof_binary_decisions).parameters


def test_model_selection_and_metrics_use_inner_validation_and_original_longwt() -> None:
    result, _, _, _ = _run_synthetic_mean1()
    summary = summarize_grouped_oof(result)
    for outer_fold in range(5):
        validation_rows = [
            row for row in result.validation_metrics if int(row["outer_fold"]) == outer_fold
        ]
        expected = sorted(
            validation_rows,
            key=lambda row: (
                -float(row["validation_weighted_auprc"]),
                -float(row["validation_weighted_auroc"]),
                str(row["model_name"]),
            ),
        )[0]["model_name"]
        actual = result.outer_fold_summaries[outer_fold]["selected_model"]
        assert actual == expected

    for row in summary["candidate_model_oof_metrics"]:
        if row["weight_mode"] != WEIGHT_MODE_MEAN1:
            continue
        expected_auprc = weighted_auprc(
            result.y,
            result.candidate_raw_oof[row["model_name"]],
            result.longwt,
        )
        assert np.isclose(float(row["weighted_auprc"]), expected_auprc)


def test_fairness_dimension_is_fully_estimable_when_all_groups_pass() -> None:
    y = np.array([1.0, 0.0, 0.0, 1.0])
    decisions = np.array([1.0, 0.0, 0.0, 0.0])
    audit = pd.DataFrame(
        {
            "RACETHX": [1, 1, 2, 2],
            "SEX": [1, 1, 2, 2],
        }
    )
    result = audit_oof_binary_decisions(
        y,
        decisions,
        audit,
        np.ones(len(y)),
        min_n=2,
        min_pos=1,
        min_neg=1,
        min_kish=0,
    )
    for dimension in ("RACETHX", "SEX"):
        dimension_result = result["dimensions"][dimension]
        assert dimension_result["endpoint"] == "FULLY_ESTIMABLE"
        assert dimension_result["full_max_pairwise_tpr_gap"] == 1.0
        assert dimension_result["partial_max_pairwise_tpr_gap_unsuppressed_groups"] is None
    assert result["primary_fairness_endpoint"] == "PRIMARY_FAIRNESS_ENDPOINT = FULLY_ESTIMABLE"
    assert result["primary_fairness_max_tpr_gap"] == 1.0


def test_fairness_partial_dimension_hides_full_gap_and_reports_unsuppressed_gap() -> None:
    y = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    decisions = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    audit = pd.DataFrame(
        {
            "RACETHX": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
            "SEX": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        }
    )
    result = audit_oof_binary_decisions(
        y,
        decisions,
        audit,
        np.ones(len(y)),
        min_n=4,
        min_pos=1,
        min_neg=1,
        min_kish=0,
    )
    race = result["dimensions"]["RACETHX"]
    assert race["endpoint"] == "PARTIALLY_ESTIMABLE_UNSUPPRESSED_GROUPS_ONLY"
    assert race["full_max_pairwise_tpr_gap"] is None
    assert race["partial_max_pairwise_tpr_gap_unsuppressed_groups"] == 1.0
    assert race["subgroups"]["3"]["status"].startswith("Suppressed")
    sex = result["dimensions"]["SEX"]
    assert sex["endpoint"] == "FULLY_ESTIMABLE"
    assert result["primary_fairness_endpoint"] == (
        "PRIMARY_FAIRNESS_ENDPOINT = NOT_FULLY_ESTIMABLE_DUE_TO_SUPPRESSION"
    )
    assert result["primary_fairness_max_tpr_gap"] is None


def test_fairness_dimension_is_not_estimable_with_fewer_than_two_available_groups() -> None:
    y = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    decisions = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    audit = pd.DataFrame({"RACETHX": [1, 1, 1, 1, 2, 2]})
    result = audit_oof_binary_decisions(
        y,
        decisions,
        audit,
        np.ones(len(y)),
        dimensions=("RACETHX",),
        min_n=4,
        min_pos=1,
        min_neg=1,
        min_kish=0,
    )
    race = result["dimensions"]["RACETHX"]
    assert race["endpoint"] == "NOT_ESTIMABLE_SUPPRESSED"
    assert race["unsuppressed_subgroups_count"] == 1
    assert race["full_max_pairwise_tpr_gap"] is None
    assert race["partial_max_pairwise_tpr_gap_unsuppressed_groups"] is None
    assert result["primary_fairness_endpoint"] == (
        "PRIMARY_FAIRNESS_ENDPOINT = NOT_FULLY_ESTIMABLE_DUE_TO_SUPPRESSION"
    )


def test_fairness_consumes_binary_decisions_and_preserves_suppression() -> None:
    n = 50
    y = np.array([1.0] * 5 + [0.0] * 45)
    decisions = np.array([1.0] * 25 + [0.0] * 25)
    longwt = np.ones(n)
    audit = pd.DataFrame({"RACETHX": [1] * n, "SEX": [1] * n})
    result = audit_oof_binary_decisions(y, decisions, audit, longwt)
    subgroup = result["dimensions"]["SEX"]["subgroups"]["1"]
    assert subgroup["status"] == "Suppressed (Insufficient Sample/Power)"
    assert subgroup["tpr"] is None
    assert subgroup["fpr"] is None
    assert result["primary_fairness_endpoint"] == (
        "PRIMARY_FAIRNESS_ENDPOINT = NOT_FULLY_ESTIMABLE_DUE_TO_SUPPRESSION"
    )
    with pytest.raises(ValueError, match="decisions must contain only binary"):
        audit_oof_binary_decisions(y, np.linspace(0.01, 0.99, n), audit, longwt)


def test_weight_scale_classifications_are_independent_for_selected_and_candidates() -> None:
    def mode_summary(selected_auprc: float, candidate_auprc: dict[str, float]) -> dict[str, object]:
        return {
            "selected_pipeline_oof": {
                "raw": {"weighted_auroc": 0.5, "weighted_auprc": selected_auprc},
                "calibrated": {"weighted_brier": 0.1},
                "operational": {"selection_rate": 0.1},
            },
            "candidate_model_oof_metrics": [
                {"model_name": name, "weighted_auprc": value}
                for name, value in candidate_auprc.items()
            ],
            "model_selection_frequency": {name: 0 for name in CANDIDATE_MODEL_NAMES},
            "fairness": {
                "dimensions": {
                    "RACETHX": {"endpoint": "FULLY_ESTIMABLE"},
                    "SEX": {"endpoint": "FULLY_ESTIMABLE"},
                },
                "primary_fairness_endpoint": "PRIMARY_FAIRNESS_ENDPOINT = FULLY_ESTIMABLE",
            },
        }

    rows, summary = compare_weight_modes(
        {
            WEIGHT_MODE_LEGACY: mode_summary(
                0.100,
                {
                    "WeightedGradientBoostingClassifier": 0.200,
                    "WeightedLogisticClassifier": 0.300,
                    "WeightedRandomForestClassifier": 0.400,
                },
            ),
            WEIGHT_MODE_MEAN1: mode_summary(
                0.105,
                {
                    "WeightedGradientBoostingClassifier": 0.201,
                    "WeightedLogisticClassifier": 0.310,
                    "WeightedRandomForestClassifier": 0.402,
                },
            ),
        }
    )
    assert summary["selected_pipeline_classification"] == (
        "SELECTED_PIPELINE_WEIGHT_SCALE_SENSITIVITY_SMALL"
    )
    assert summary["candidate_model_classification"] == (
        "CANDIDATE_MODEL_WEIGHT_SCALE_SENSITIVITY_MATERIAL"
    )
    assert np.isclose(summary["maximum_candidate_model_auprc_absolute_difference"], 0.01)
    assert summary["maximum_candidate_model_name"] == "WeightedLogisticClassifier"
    selected_row = next(row for row in rows if row["metric"] == "selected_pipeline_oof_auprc")
    logistic_row = next(
        row
        for row in rows
        if row["metric"] == "candidate_model_oof_auprc"
        and row["model_name"] == "WeightedLogisticClassifier"
    )
    assert selected_row["classification"] == "SELECTED_PIPELINE_WEIGHT_SCALE_SENSITIVITY_SMALL"
    assert logistic_row["classification"] == "CANDIDATE_MODEL_WEIGHT_SCALE_SENSITIVITY_MATERIAL"


def test_output_writer_has_only_the_authorized_aggregate_artifact_names(tmp_path: pathlib.Path) -> None:
    assert set(ARTIFACT_NAMES) == {
        "oof_summary.json",
        "candidate_model_oof_metrics.csv",
        "fold_metrics.csv",
        "model_selection_stability.csv",
        "fairness_audit.csv",
        "weight_scale_sensitivity.csv",
        "calibration_bins.csv",
        "RUN_README.md",
        "execution_manifest.json",
    }
    assert not any("prediction" in name.lower() for name in ARTIFACT_NAMES)

    minimal_mode = {
        "selected_pipeline_oof": {
            "raw": {"weighted_auroc": 0.5, "weighted_auprc": 0.1},
            "calibrated": {"weighted_brier": 0.1},
            "operational": {"selection_rate": 0.1},
        },
        "model_selection_frequency": {name: 0 for name in CANDIDATE_MODEL_NAMES},
        "fairness": {
            "primary_fairness_endpoint": "PRIMARY_FAIRNESS_ENDPOINT = NOT_FULLY_ESTIMABLE_DUE_TO_SUPPRESSION",
            "dimensions": {
                "RACETHX": {
                    "endpoint": "NOT_ESTIMABLE_SUPPRESSED",
                    "full_max_pairwise_tpr_gap": None,
                    "partial_max_pairwise_tpr_gap_unsuppressed_groups": None,
                },
                "SEX": {
                    "endpoint": "NOT_ESTIMABLE_SUPPRESSED",
                    "full_max_pairwise_tpr_gap": None,
                    "partial_max_pairwise_tpr_gap_unsuppressed_groups": None,
                },
            },
        },
    }
    aggregate = {
        "oof_summary": {
            "seed": 20260828,
            "protocol": {"outer_folds": 5},
            "calibrated_auroc_interpretation_note": CALIBRATED_AUROC_INTERPRETATION_NOTE,
            "cohort": {"eligible_record_count": 0, "positive_events": 0, "duid_count": 0},
            "modes": {WEIGHT_MODE_LEGACY: minimal_mode, WEIGHT_MODE_MEAN1: minimal_mode},
            "weight_scale_sensitivity": {
                "selected_pipeline_oof_auprc_absolute_difference": 0.0,
                "selected_pipeline_classification": "SELECTED_PIPELINE_WEIGHT_SCALE_SENSITIVITY_SMALL",
                "maximum_candidate_model_auprc_absolute_difference": 0.0,
                "maximum_candidate_model_name": "WeightedGradientBoostingClassifier",
                "candidate_model_classification": "CANDIDATE_MODEL_WEIGHT_SCALE_SENSITIVITY_SMALL",
            },
        },
        "candidate_model_oof_metrics": [],
        "fold_metrics": [],
        "model_selection_stability": [],
        "fairness_audit": [],
        "weight_scale_sensitivity": [],
        "calibration_bins": [],
    }
    preflight = {
        "branch": "research/meps-hc252-longitudinal",
        "head": "59b90eb3545ae4948475b8eb45f6f739d855b8bc",
        "hc244_sha256": "synthetic",
    }
    _write_aggregate_artifacts(tmp_path, aggregate, "synthetic_run", preflight, 0.1)
    assert {path.name for path in tmp_path.iterdir()} == set(ARTIFACT_NAMES) - {"execution_manifest.json"}


def test_new_implementation_has_no_unauthorized_data_file_path() -> None:
    source_text = "\n".join(
        [
            (SRC_ROOT / "meps_fairness" / "crossfit.py").read_text(encoding="utf-8").lower(),
            (REPO_ROOT / "scripts" / "run_panel26_oof_robustness.py").read_text(encoding="utf-8").lower(),
        ]
    )
    assert "data/interim/meps/h252" not in source_text
    assert "h252.dta" not in source_text
    assert "data/interim/meps/h225" not in source_text
    assert "data/interim/meps/h234" not in source_text
    assert "data/interim/meps/h217" not in source_text
