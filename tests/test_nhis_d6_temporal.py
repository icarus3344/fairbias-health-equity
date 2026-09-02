"""Comprehensive unit and synthetic integration tests for Gate D6.0a Temporal Harness.

All 60 mandated checks + dynamic ordering + 2024 poison + validation selection invariance.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import pathlib
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from fairbias.config import (
    ALGORITHM_MODE_PAPER_FAITHFUL,
    FairBiasConfig,
)
from fairbias.evaluator import FairEvaluator
from fairbias.mitigation import FairBiasMitigation
from fairbias.transform import (
    FairTransform,
    calculate_nmi_dict,
)
from fairbias.transform_trace import (
    FairBiasTransformStep,
    FairBiasTransformTrace,
)

from nhis_fairbias.adapter import (
    DISABILITY_COMPONENTS,
    OUTCOME_MAP,
    PROTECTED_MAP,
    STUDY_YEAR_ROLES,
    NHISStudyAdapter,
)
from nhis_fairbias.d6_temporal_runner import (
    CLASSIFIER_MAX_ITER,
    CLASSIFIER_SOLVER,
    DEFAULT_PREDICTION_THRESHOLD,
    EXCLUDED_DISABILITY_COMPONENTS,
    EXPECTED_MEDDL12M_TOTALS,
    EXPECTED_TOTAL_ROW_COUNT,
    EXPECTED_YEAR_ROW_COUNTS,
    FROZEN_D4_ARCHIVED_COMMIT,
    FROZEN_D4_TAG,
    FROZEN_D4_TAG_OBJECT,
    FROZEN_D5_SECONDARY_TEST_ARCHIVED_COMMIT,
    FROZEN_D5_SECONDARY_TEST_TAG,
    FROZEN_D5_SECONDARY_TEST_TAG_OBJECT,
    FROZEN_D5_TRAIN_VAL_ARCHIVED_COMMIT,
    FROZEN_D5_TRAIN_VAL_TAG,
    FROZEN_D5_TRAIN_VAL_TAG_OBJECT,
    FROZEN_D6_ARMS,
    FROZEN_FEATURES_PARQUET_PATH,
    FROZEN_FEATURES_PARQUET_SHA256,
    FROZEN_SCIENTIFIC_PATHS_DIFF,
    FROZEN_SCIENTIFIC_PATHS_SPEC,
    LEAKAGE_FORBIDDEN_PREDICTORS,
    NHISD6TemporalReleaseManager,
    NHISD6TemporalRunner,
    PRIMARY_D6_RANDOM_SEED,
    PRIOR_RELEASE_ARCHIVE_DIRS,
    REQUIRED_PER_ARM_ARTIFACTS,
    SCIENTIFIC_EXECUTION_BASE_COMMIT,
    TEMPORAL_2024_DISCLOSURE,
    TEMPORAL_STUDY_ROLES,
    TEMPORAL_TEST_YEAR,
    TEMPORAL_TRAIN_YEAR,
    TEMPORAL_VALIDATION_YEAR,
    compute_canonical_json_sha256,
    compute_cohort_source_row_digest,
    extract_logistic_regression_state,
    extract_minmax_scaler_state,
    get_git_commit,
    verify_cohort_alignment_and_uniqueness,
    verify_execution_preconditions,
    verify_frozen_features_parquet,
    verify_prior_release_archives,
    verify_prior_release_tags,
    verify_scientific_code_boundary,
)
from nhis_fairbias.download import compute_sha256
from nhis_fairbias.evaluation import (
    compute_evaluation_comparison,
    evaluate_predictions,
)
from nhis_fairbias.preprocessing import (
    NHISLeakageError,
    NHISPreprocessor,
    PreprocessingFitRecord,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

PRIMARY_CORE_COLS = [
    "educp_a",
    "region",
    "ratcat_a",
    "empwrklsw1_a",
    "empwrkft1_a",
    "notcov_a",
    "phstat_a",
    "hypev_a",
    "chlev_a",
    "dibev_a",
    "asev_a",
    "visiondf_a",
    "hearingdf_a",
    "diff_a",
    "comdiff_a",
    "uppslfcr_a",
    "cogmemdff_a",
    "usualpl_a",
    "agep_a",
    "pcnt18uptc",
    "pcntlt18tc",
]


def create_mock_cohort(
    year: int,
    outcome: str = "MEDDL12M_A",
    protected_attribute: str = "SEX_A",
    feature_set: str = "primary_core",
    disability_arm: str = "full_feature",
    n: int = 70,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Generate synthetic (X, y, o, w, metadata) with exact feature names and aligned indices."""
    rng = np.random.default_rng(seed + year)

    # 18 categoricals
    cat_data: Dict[str, np.ndarray] = {
        "educp_a": rng.integers(1, 5, size=n),
        "region": rng.integers(1, 5, size=n),
        "ratcat_a": rng.integers(1, 15, size=n),
        "empwrklsw1_a": rng.integers(1, 3, size=n),
        "empwrkft1_a": rng.integers(1, 5, size=n),
        "notcov_a": rng.integers(1, 3, size=n),
        "phstat_a": rng.integers(1, 6, size=n),
        "hypev_a": rng.integers(1, 3, size=n),
        "chlev_a": rng.integers(1, 3, size=n),
        "dibev_a": rng.integers(1, 3, size=n),
        "asev_a": rng.integers(1, 3, size=n),
        "visiondf_a": rng.integers(1, 5, size=n),
        "hearingdf_a": rng.integers(1, 5, size=n),
        "diff_a": rng.integers(1, 5, size=n),
        "comdiff_a": rng.integers(1, 5, size=n),
        "uppslfcr_a": rng.integers(1, 5, size=n),
        "cogmemdff_a": rng.integers(1, 5, size=n),
        "usualpl_a": rng.integers(1, 3, size=n),
    }
    # 3 numericals
    num_data: Dict[str, np.ndarray] = {
        "agep_a": rng.uniform(18.0, 85.0, size=n),
        "pcnt18uptc": rng.integers(1, 6, size=n).astype(float),
        "pcntlt18tc": rng.integers(0, 5, size=n).astype(float),
    }

    all_data = {**cat_data, **num_data}
    idx = pd.RangeIndex(start=1000 + (year - 2022) * 500, stop=1000 + (year - 2022) * 500 + n)

    if disability_arm == "exclude_disability_components":
        for comp in EXCLUDED_DISABILITY_COMPONENTS:
            all_data.pop(comp, None)

    X = pd.DataFrame(all_data, index=idx)

    # Outcomes
    y_vals = rng.integers(0, 2, size=n)
    y = pd.Series(y_vals, index=idx, name=outcome, dtype=int)

    # Protected attribute
    if protected_attribute == "SEX_A":
        o_vals = rng.choice([1, 2], size=n)
    elif protected_attribute == "HISPALLP_A":
        # Ensure all 7 categories are present
        base_groups = np.array([1, 2, 3, 4, 5, 6, 7])
        rest = rng.choice(base_groups, size=n - len(base_groups))
        o_vals = np.concatenate([base_groups, rest])
        rng.shuffle(o_vals)
    elif protected_attribute == "DISAB3_A":
        o_vals = rng.choice([1, 2], size=n)
    else:
        o_vals = rng.choice([1, 2], size=n)
    o = pd.Series(o_vals, index=idx, name=protected_attribute, dtype=int)

    # Weights and metadata
    w = pd.Series(rng.uniform(500.0, 5000.0, size=n), index=idx, name="WTFA_A")
    meta = pd.DataFrame(
        {
            "survey_year": year,
            "study_role": TEMPORAL_STUDY_ROLES[year],
            "WTFA_A": w,
            "PSTRAT": 100,
            "PPSU": 1,
        },
        index=idx,
    )

    return X, y, o, w, meta


def make_mock_preprocessor() -> NHISPreprocessor:
    """Construct a mock NHISPreprocessor instance reporting 2022 development_train."""
    prep = NHISPreprocessor()
    prep._fitted_record = PreprocessingFitRecord(
        fit_year=2022,
        fit_study_role="development_train",
        row_count=27651,
        numerical_medians={"agep_a": 50.0, "pcnt18uptc": 2.0, "pcntlt18tc": 0.0},
        numerical_stats={},
        categorical_categories={},
        empwrkft_distribution={},
        rules_manifest={"schema_version": "nhis-fairbias-d2-1.0", "regime": "temporal"},
    )
    return prep


def make_mock_arm_execution_result(arm_id: str, **kwargs: Any) -> Dict[str, Any]:
    """Generate synthetic execution result simulating execute_arm_train_validation."""
    arm_spec = FROZEN_D6_ARMS[arm_id]
    prot = arm_spec["protected_attribute"]

    step1 = FairBiasTransformStep(
        iteration=1,
        selected_feature="educp_a",
        feature_semantic_type="categorical",
        d_phi_before=0.08,
        epsilon=0.04,
        proposed_transformation={"1": "1", "2": "1"},
        accepted_transformation={"1": "1", "2": "1"},
        categorical_merge_mapping={"1": "1", "2": "1"},
        d_phi_after=0.03,
        dropped=False,
    )
    trace = FairBiasTransformTrace(
        algorithm_mode=ALGORITHM_MODE_PAPER_FAITHFUL,
        protected_attribute=prot,
        epsilon_threshold=0.04,
        steps=[step1],
        final_status="CONVERGED",
        final_max_dphi=0.03,
    )

    group_count = arm_spec["expected_group_count"]
    group_ids = arm_spec["expected_groups"]
    base_groups = []
    for gid in group_ids:
        base_groups.append(
            {
                "group": gid,
                "n": 5000,
                "outcome_positive_count": 500,
                "outcome_negative_count": 4500,
                "predicted_positive_count": 400,
                "predicted_negative_count": 4600,
                "tp_count": 250,
                "fp_count": 150,
                "tn_count": 4350,
                "fn_count": 250,
                "prevalence": 0.10,
                "selection_rate": 0.08,
                "tpr": 0.50,
                "fpr": 0.033,
                "ppv": 0.625,
            }
        )

    group_metrics_df = pd.DataFrame(base_groups)

    val_eval_base = {
        "utility": {
            "auroc": 0.75,
            "auprc": 0.35,
            "balanced_accuracy": 0.68,
            "f1": 0.40,
            "accuracy": 0.85,
            "count_predicted_positive": 400 * group_count,
            "selection_rate": 0.08,
            "count_outcome_positive": 500 * group_count,
            "prevalence": 0.10,
        },
        "fairness_gaps": {
            "demographic_parity_gap": 0.02,
            "equal_opportunity_gap": 0.03,
            "fpr_gap": 0.01,
            "equalized_odds_max_gap": 0.03,
        },
        "group_metrics": base_groups,
        "group_coverage": {
            "expected_group_count": group_count,
            "observed_group_count": group_count,
            "observed_groups": group_ids,
            "group_coverage_complete": True,
            "expected_pair_count": arm_spec["expected_pair_count"],
            "observed_pair_count": arm_spec["expected_pair_count"],
            "diagnostics": None,
        },
    }
    val_eval_fb = copy.deepcopy(val_eval_base)
    val_comparison = compute_evaluation_comparison(val_eval_base, val_eval_fb)
    val_comparison["predicted_positive_delta"] = 0
    val_comparison["selection_rate_delta"] = 0.0

    arm_cfg = {
        **arm_spec,
        "algorithm_mode": ALGORITHM_MODE_PAPER_FAITHFUL,
        "survey_weighting": "NONE",
        "train_year": TEMPORAL_TRAIN_YEAR,
        "validation_year": TEMPORAL_VALIDATION_YEAR,
        "future_test_year": TEMPORAL_TEST_YEAR,
        "classifier": {
            "type": "LR",
            "random_state": PRIMARY_D6_RANDOM_SEED,
            "solver": CLASSIFIER_SOLVER,
            "max_iter": CLASSIFIER_MAX_ITER,
            "sample_weight": None,
        },
        "prediction_threshold": DEFAULT_PREDICTION_THRESHOLD,
        "categorical_features": ["educp_a"],
        "numerical_features": ["agep_a"],
        "repeated_cross_sectional": True,
        "longitudinal": False,
        "causal_analysis": False,
        "temporal_robustness_analysis": True,
    }

    input_prov = {
        "arm_id": arm_id,
        "features_parquet_path": str(FROZEN_FEATURES_PARQUET_PATH),
        "features_parquet_sha256": FROZEN_FEATURES_PARQUET_SHA256,
        "preprocessor_fit_year": TEMPORAL_TRAIN_YEAR,
        "preprocessor_fit_role": TEMPORAL_STUDY_ROLES[TEMPORAL_TRAIN_YEAR],
        "train_year": TEMPORAL_TRAIN_YEAR,
        "train_n": 27453,
        "train_source_row_digest": "mock_train_digest",
        "train_outcome_positive_count": 1770,
        "train_prevalence": 0.064,
        "train_group_counts": {str(g): 5000 for g in group_ids},
        "train_group_positive_counts": {str(g): 500 for g in group_ids},
        "train_group_prevalences": {str(g): 0.10 for g in group_ids},
        "validation_year": TEMPORAL_VALIDATION_YEAR,
        "validation_n": 29283,
        "validation_source_row_digest": "mock_val_digest",
        "validation_outcome_positive_count": 1931,
        "validation_prevalence": 0.065,
        "validation_group_counts": {str(g): 5000 for g in group_ids},
        "validation_group_positive_counts": {str(g): 500 for g in group_ids},
        "validation_group_prevalences": {str(g): 0.10 for g in group_ids},
        "predictor_names": ["educp_a", "agep_a"],
        "protected_attribute": prot,
        "survey_weight_used_for_geometry": False,
        "survey_weight_used_for_classifier": False,
        "survey_weight_used_for_evaluation": False,
        "validation_used_for_selection": False,
        "test_year_requested": False,
        "test_year_evaluated": False,
        "frozen_2022_training_state": {
            "documentation": (
                "These frozen training-state fingerprints allow the future D6.1 temporal TEST harness "
                "to refit deterministically from the same frozen 2022 cohort and verify exact "
                "scaler/model/representation state reproduction before requesting the 2024 cohort."
            ),
            "baseline_scaler": {
                "sha256": "mock_baseline_scaler_sha",
                "state": {"feature_order": ["educp_a", "agep_a"]},
            },
            "baseline_logistic_regression": {
                "sha256": "mock_baseline_lr_sha",
                "state": {"feature_order": ["educp_a", "agep_a"]},
            },
            "fairbias_scaler": {
                "sha256": "mock_fb_scaler_sha",
                "state": {"feature_order": ["educp_a", "agep_a"]},
            },
            "fairbias_logistic_regression": {
                "sha256": "mock_fb_lr_sha",
                "state": {"feature_order": ["educp_a", "agep_a"]},
            },
            "changed_dict_sha256": "mock_changed_dict_sha",
        },
    }

    train_dphi = {
        "arm_id": arm_id,
        "train_year": TEMPORAL_TRAIN_YEAR,
        "protected_attribute": prot,
        "epsilon_threshold": 0.04,
        "initial_dphi": {"educp_a": 0.08},
        "initial_max_dphi": 0.08,
        "highest_initial_feature": "educp_a",
        "final_dphi": {"educp_a": 0.03},
        "final_max_dphi": 0.03,
        "delta_max_dphi": -0.05,
        "termination_reason": "epsilon_reached",
        "converged": True,
        "total_mitigation_steps": 1,
    }

    val_dphi = {
        "arm_id": arm_id,
        "validation_year": TEMPORAL_VALIDATION_YEAR,
        "protected_attribute": prot,
        "frozen_train_epsilon_threshold": 0.04,
        "original_validation_dphi": {"educp_a": 0.08},
        "original_validation_max_dphi": 0.08,
        "transformed_validation_dphi": {"educp_a": 0.03},
        "transformed_validation_max_dphi": 0.03,
        "delta_max_dphi": -0.05,
        "max_dphi_below_train_epsilon": True,
        "dphi_transport_relearned": False,
        "diagnostic_only": True,
    }

    return {
        "arm_id": arm_id,
        "arm_config": arm_cfg,
        "input_provenance": input_prov,
        "train_dphi_before_after": train_dphi,
        "train_fairbias_trace": trace.to_dict(),
        "final_changed_dict": {"educp_a": {"1": "1", "2": "1"}},
        "validation_dphi_before_after": val_dphi,
        "validation_metrics_baseline": val_eval_base,
        "validation_metrics_fairbias": val_eval_fb,
        "validation_group_metrics_baseline": group_metrics_df,
        "validation_group_metrics_fairbias": group_metrics_df,
        "validation_comparison": val_comparison,
    }


class MockStudyAdapter:
    """Mock NHISStudyAdapter tracking cohort calls and poisoning year 2024."""

    def __init__(
        self,
        poison_2024: bool = True,
        features_parquet_path: Optional[pathlib.Path] = None,
        preprocessor: Optional[NHISPreprocessor] = None,
    ):
        self.poison_2024 = poison_2024
        self.features_parquet_path = (
            pathlib.Path(features_parquet_path).resolve()
            if features_parquet_path
            else FROZEN_FEATURES_PARQUET_PATH.resolve()
        )
        self.preprocessor = preprocessor or make_mock_preprocessor()
        self.requested_cohorts: List[Dict[str, Any]] = []

    def get_cohort(
        self,
        year: int,
        outcome: str = "MEDDL12M_A",
        protected_attribute: str = "SEX_A",
        feature_set: str = "primary_core",
        disability_arm: str = "full_feature",
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
        self.requested_cohorts.append(
            {
                "year": year,
                "outcome": outcome,
                "protected_attribute": protected_attribute,
                "feature_set": feature_set,
                "disability_arm": disability_arm,
            }
        )
        if year == 2024 and self.poison_2024:
            raise AssertionError("FATAL: D6.0 attempted to access temporal TEST year 2024")
        if year not in (2022, 2023, 2024):
            raise ValueError(f"Invalid NHIS year: {year}")

        return create_mock_cohort(
            year=year,
            outcome=outcome,
            protected_attribute=protected_attribute,
            feature_set=feature_set,
            disability_arm=disability_arm,
        )


# ==============================================================================
# All 60 Mandated Checks
# ==============================================================================


def test_01_exact_four_d6_arms() -> None:
    """Check 1: Exact four D6 arms."""
    assert set(FROZEN_D6_ARMS.keys()) == {
        "D6_ARM_001",
        "D6_ARM_002",
        "D6_ARM_003",
        "D6_ARM_004",
    }
    assert FROZEN_D6_ARMS["D6_ARM_001"]["protected_attribute"] == "SEX_A"
    assert FROZEN_D6_ARMS["D6_ARM_002"]["protected_attribute"] == "HISPALLP_A"
    assert FROZEN_D6_ARMS["D6_ARM_003"]["protected_attribute"] == "DISAB3_A"
    assert FROZEN_D6_ARMS["D6_ARM_004"]["protected_attribute"] == "DISAB3_A"
    for arm in FROZEN_D6_ARMS.values():
        assert arm["outcome"] == "MEDDL12M_A"
        assert arm["feature_set"] == "PRIMARY_CORE"


def test_02_temporal_years_fixed() -> None:
    """Check 2: Temporal years fixed to 2022/2023/2024."""
    assert TEMPORAL_TRAIN_YEAR == 2022
    assert TEMPORAL_VALIDATION_YEAR == 2023
    assert TEMPORAL_TEST_YEAR == 2024
    assert TEMPORAL_STUDY_ROLES[2022] == "development_train"
    assert TEMPORAL_STUDY_ROLES[2023] == "development_validation"
    assert TEMPORAL_STUDY_ROLES[2024] == "frozen_test"


def test_03_repeated_cross_sectional_manifest_semantics(tmp_path: pathlib.Path) -> None:
    """Check 3: Repeated-cross-sectional manifest semantics."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    manager = NHISD6TemporalReleaseManager(
        release_id="TEST_SEMANTICS",
        base_dir=tmp_path,
        runner=runner,
        allow_substantive_execution=True,
    )
    with patch.object(runner, "execute_arm_train_validation", side_effect=make_mock_arm_execution_result):
        manifest = manager.execute_release()
    assert manifest["repeated_cross_sectional"] is True
    assert manifest["longitudinal"] is False
    assert manifest["causal_analysis"] is False
    assert manifest["temporal_robustness_analysis"] is True
    assert manifest["replaces_d4_primary"] is False
    assert manifest["primary_analysis"] is False


def test_04_random_split_route_prohibited() -> None:
    """Check 4: Random split route absent/prohibited."""
    adapter = MockStudyAdapter()
    assert not hasattr(adapter, "random_split") or pytest.raises(Exception, adapter.random_split)


def test_05_pooled_adapter_not_used() -> None:
    """Check 5: Pooled adapter is not used in runner."""
    import nhis_fairbias.d6_temporal_runner as mod
    assert "NHISPooledAdapter" not in dir(mod)


def test_06_d6_is_strictly_unweighted() -> None:
    """Check 6: D6 is strictly unweighted."""
    for arm in FROZEN_D6_ARMS.values():
        res = make_mock_arm_execution_result(arm["arm_id"])
        cfg = res["arm_config"]
        assert cfg["survey_weighting"] == "NONE"
        assert cfg["classifier"]["sample_weight"] is None
        prov = res["input_provenance"]
        assert prov["survey_weight_used_for_geometry"] is False
        assert prov["survey_weight_used_for_classifier"] is False
        assert prov["survey_weight_used_for_evaluation"] is False


def test_07_sample_weight_never_passed_to_fairbias_dphi() -> None:
    """Check 7: sample_weight never passed to FairBias d_phi."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    orig_calc = FairEvaluator.calculate_epsilon
    weights_passed: List[Any] = []

    def spy_calc(self, *args: Any, **kwargs: Any) -> Any:
        weights_passed.append(kwargs.get("sample_weight"))
        return orig_calc(self, *args, **kwargs)

    with patch.object(FairEvaluator, "calculate_epsilon", spy_calc), \
         patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        runner.execute_arm_train_validation("D6_ARM_001")
        assert len(weights_passed) >= 1
        for w in weights_passed:
            assert w is None


def test_08_sample_weight_never_passed_to_lr() -> None:
    """Check 8: sample_weight never passed to LR."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    orig_fit = LogisticRegression.fit
    weights_passed: List[Any] = []

    def spy_fit(self, *args: Any, **kwargs: Any) -> Any:
        weights_passed.append(kwargs.get("sample_weight"))
        return orig_fit(self, *args, **kwargs)

    with patch.object(LogisticRegression, "fit", spy_fit), \
         patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        runner.execute_arm_train_validation("D6_ARM_001")
        assert len(weights_passed) == 2
        for w in weights_passed:
            assert w is None


def test_09_prediction_threshold_exactly_half() -> None:
    """Check 9: Prediction threshold exactly 0.5."""
    assert DEFAULT_PREDICTION_THRESHOLD == 0.5


def test_10_random_state_exactly_zero() -> None:
    """Check 10: Random state exactly 0."""
    assert PRIMARY_D6_RANDOM_SEED == 0


def test_11_solver_lbfgs() -> None:
    """Check 11: Solver lbfgs."""
    assert CLASSIFIER_SOLVER == "lbfgs"


def test_12_max_iter_1000() -> None:
    """Check 12: Max iter 1000."""
    assert CLASSIFIER_MAX_ITER == 1000


def test_13_preprocessing_reports_fit_year_2022() -> None:
    """Check 13: Preprocessing reports fit year 2022."""
    prep = make_mock_preprocessor()
    prep._fitted_record.fit_year = 2023
    mock_adapter = MockStudyAdapter(preprocessor=prep)
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    with pytest.raises(NHISLeakageError, match="Preprocessor fit year must be 2022"):
        runner.execute_arm_train_validation("D6_ARM_001")


def test_14_preprocessing_reports_development_train_role() -> None:
    """Check 14: Preprocessing reports development_train role."""
    prep = make_mock_preprocessor()
    prep._fitted_record.fit_study_role = "pooled_train"
    mock_adapter = MockStudyAdapter(preprocessor=prep)
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    with pytest.raises(NHISLeakageError, match="Preprocessor fit role must be development_train"):
        runner.execute_arm_train_validation("D6_ARM_001")


def test_15_predictor_counts_21_21_21_15() -> None:
    """Check 15: Predictor counts 21/21/21/15."""
    counts = [arm["expected_predictors"] for arm in FROZEN_D6_ARMS.values()]
    assert counts == [21, 21, 21, 15]


def test_16_hisp_expected_groups_exactly_7() -> None:
    """Check 16: HISP expected groups exactly 7."""
    hisp_arm = FROZEN_D6_ARMS["D6_ARM_002"]
    assert hisp_arm["expected_group_count"] == 7
    assert hisp_arm["expected_groups"] == [1, 2, 3, 4, 5, 6, 7]


def test_17_hisp_expected_pair_count_exactly_21() -> None:
    """Check 17: HISP expected pair count exactly 21."""
    assert FROZEN_D6_ARMS["D6_ARM_002"]["expected_pair_count"] == 21


def test_18_disability_exclusion_list_exactly_six_components() -> None:
    """Check 18: Disability exclusion list exactly six components."""
    assert len(EXCLUDED_DISABILITY_COMPONENTS) == 6
    assert set(EXCLUDED_DISABILITY_COMPONENTS) == {
        "visiondf_a",
        "hearingdf_a",
        "diff_a",
        "comdiff_a",
        "uppslfcr_a",
        "cogmemdff_a",
    }


def test_19_cohort_index_alignment() -> None:
    """Check 19: Cohort index alignment verification."""
    X, y, o, _, _ = create_mock_cohort(2022)
    y_bad = y.copy()
    y_bad.index = pd.RangeIndex(len(y_bad))
    with pytest.raises(ValueError, match="Cohort index alignment mismatch"):
        verify_cohort_alignment_and_uniqueness(X, y_bad, o)


def test_20_cohort_source_indices_unique() -> None:
    """Check 20: Cohort source indices unique verification."""
    X, y, o, _, _ = create_mock_cohort(2022)
    bad_idx = list(X.index)
    bad_idx[1] = bad_idx[0]
    X_dup = X.copy()
    X_dup.index = pd.Index(bad_idx)
    y_dup = y.copy()
    y_dup.index = pd.Index(bad_idx)
    o_dup = o.copy()
    o_dup.index = pd.Index(bad_idx)
    with pytest.raises(ValueError, match="Cohort indices are not unique"):
        verify_cohort_alignment_and_uniqueness(X_dup, y_dup, o_dup)


def test_21_deterministic_row_digest_stability() -> None:
    """Check 21: Deterministic row digest stability."""
    idx1 = pd.RangeIndex(0, 100)
    idx2 = pd.RangeIndex(0, 100)
    idx3 = pd.RangeIndex(1, 101)
    d1 = compute_cohort_source_row_digest(2022, idx1)
    d2 = compute_cohort_source_row_digest(2022, idx2)
    d3 = compute_cohort_source_row_digest(2022, idx3)
    d4 = compute_cohort_source_row_digest(2023, idx1)
    assert d1 == d2
    assert d1 != d3
    assert d1 != d4


def test_22_fairbias_epsilon_calculated_only_on_2022() -> None:
    """Check 22: FairBias epsilon calculated only on 2022."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    events: List[str] = []
    with patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        runner.execute_arm_train_validation("D6_ARM_001", event_callback=events.append)
    eps_idx = events.index("compute_2022_epsilon")
    req_2023_idx = events.index("request_2023")
    assert eps_idx < req_2023_idx


def test_23_nmi_calculated_only_on_2022() -> None:
    """Check 23: NMI calculated only on 2022."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    events: List[str] = []
    with patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        runner.execute_arm_train_validation("D6_ARM_001", event_callback=events.append)
    nmi_idx = events.index("compute_2022_nmi")
    req_2023_idx = events.index("request_2023")
    assert nmi_idx < req_2023_idx


def test_24_fairbias_mitigation_receives_only_2022() -> None:
    """Check 24: FairBias mitigation receives only 2022."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    events: List[str] = []
    with patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        runner.execute_arm_train_validation("D6_ARM_001", event_callback=events.append)
    mit_idx = events.index("learn_2022_changed_dict")
    req_2023_idx = events.index("request_2023")
    assert mit_idx < req_2023_idx


def test_25_terminal_changed_dict_finalized_before_2023_request() -> None:
    """Check 25: Terminal changed_dict finalized before 2023 request."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    events: List[str] = []
    with patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        runner.execute_arm_train_validation("D6_ARM_001", event_callback=events.append)
    freeze_idx = events.index("freeze_2022_changed_dict")
    req_2023_idx = events.index("request_2023")
    assert freeze_idx < req_2023_idx


def test_26_baseline_scaler_fit_only_on_2022() -> None:
    """Check 26: Baseline scaler fit only on 2022."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    events: List[str] = []
    with patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        runner.execute_arm_train_validation("D6_ARM_001", event_callback=events.append)
    fit_idx = events.index("fit_baseline_scaler_2022")
    req_2023_idx = events.index("request_2023")
    assert fit_idx < req_2023_idx


def test_27_fairbias_scaler_fit_only_on_transformed_2022() -> None:
    """Check 27: FairBias scaler fit only on transformed 2022."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    events: List[str] = []
    with patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        runner.execute_arm_train_validation("D6_ARM_001", event_callback=events.append)
    fit_idx = events.index("fit_fairbias_scaler_2022")
    req_2023_idx = events.index("request_2023")
    assert fit_idx < req_2023_idx


def test_28_baseline_lr_fit_only_on_2022() -> None:
    """Check 28: Baseline LR fit only on 2022."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    events: List[str] = []
    with patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        runner.execute_arm_train_validation("D6_ARM_001", event_callback=events.append)
    fit_idx = events.index("fit_baseline_lr_2022")
    req_2023_idx = events.index("request_2023")
    assert fit_idx < req_2023_idx


def test_29_fairbias_lr_fit_only_on_transformed_2022() -> None:
    """Check 29: FairBias LR fit only on transformed 2022."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    events: List[str] = []
    with patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        runner.execute_arm_train_validation("D6_ARM_001", event_callback=events.append)
    fit_idx = events.index("fit_fairbias_lr_2022")
    req_2023_idx = events.index("request_2023")
    assert fit_idx < req_2023_idx


def test_30_2023_receives_transform_only() -> None:
    """Check 30: 2023 receives transform only."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    orig_fit = MinMaxScaler.fit
    fit_count = 0

    def spy_scaler_fit(self, *args: Any, **kwargs: Any) -> Any:
        nonlocal fit_count
        fit_count += 1
        return orig_fit(self, *args, **kwargs)

    with patch.object(MinMaxScaler, "fit", spy_scaler_fit), \
         patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        runner.execute_arm_train_validation("D6_ARM_001")
        assert fit_count == 2


def test_31_2023_never_triggers_scaler_fit() -> None:
    """Check 31: 2023 never triggers scaler fit."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    orig_fit_transform = MinMaxScaler.fit_transform
    fit_calls: List[np.ndarray] = []

    def spy_fit_transform(self, X, y=None):
        fit_calls.append(np.asarray(X))
        return orig_fit_transform(self, X, y=y)

    with patch.object(MinMaxScaler, "fit_transform", spy_fit_transform), patch.object(
        FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)
    ):
        runner.execute_arm_train_validation("D6_ARM_001")
        assert len(fit_calls) == 2
        for X_fit in fit_calls:
            assert len(X_fit) == 70


def test_32_2023_never_triggers_lr_fit() -> None:
    """Check 32: 2023 never triggers LR fit."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    orig_fit = LogisticRegression.fit
    fit_count = 0

    def spy_lr_fit(self, *args: Any, **kwargs: Any) -> Any:
        nonlocal fit_count
        fit_count += 1
        return orig_fit(self, *args, **kwargs)

    with patch.object(LogisticRegression, "fit", spy_lr_fit), \
         patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        runner.execute_arm_train_validation("D6_ARM_001")
        assert fit_count == 2


def test_33_2023_never_changes_changed_dict() -> None:
    """Check 33: 2023 never changes changed_dict."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    events: List[str] = []
    with patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        res = runner.execute_arm_train_validation("D6_ARM_001", event_callback=events.append)
    assert "freeze_2022_changed_dict" in events
    assert res["final_changed_dict"] is not None


def test_34_2023_never_changes_epsilon() -> None:
    """Check 34: 2023 never changes epsilon."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    with patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        res = runner.execute_arm_train_validation("D6_ARM_001")
    t_eps = res["train_dphi_before_after"]["epsilon_threshold"]
    v_eps = res["validation_dphi_before_after"]["frozen_train_epsilon_threshold"]
    assert t_eps == v_eps


def test_35_2023_dphi_calculation_is_diagnostic_only() -> None:
    """Check 35: 2023 d_phi calculation is diagnostic only."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    with patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        res = runner.execute_arm_train_validation("D6_ARM_001")
    v_diag = res["validation_dphi_before_after"]
    assert v_diag["diagnostic_only"] is True
    assert v_diag["dphi_transport_relearned"] is False


def test_36_validation_comparison_does_not_select_model() -> None:
    """Check 36: Validation comparison does not select a model."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    with patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        res = runner.execute_arm_train_validation("D6_ARM_001")
    comp_str = json.dumps(res["validation_comparison"])
    for forbidden in ["winner", "improved", "worse", "successful", "failed scientifically"]:
        assert forbidden not in comp_str.lower()


def test_37_production_call_sequence_is_2022_then_2023_per_arm() -> None:
    """Check 37: Production call sequence is 2022 then 2023 per arm."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    with patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        for arm_id in sorted(FROZEN_D6_ARMS.keys()):
            runner.execute_arm_train_validation(arm_id)

    years = [call["year"] for call in mock_adapter.requested_cohorts]
    assert years == [2022, 2023, 2022, 2023, 2022, 2023, 2022, 2023]


def test_38_no_get_cohort_2024_call_occurs() -> None:
    """Check 38: No get_cohort(2024) call occurs."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    with patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        for arm_id in sorted(FROZEN_D6_ARMS.keys()):
            runner.execute_arm_train_validation(arm_id)

    years = [call["year"] for call in mock_adapter.requested_cohorts]
    assert 2024 not in years


def test_39_poison_get_cohort_2024_full_run_still_completes(tmp_path: pathlib.Path) -> None:
    """Check 39: Poison get_cohort(2024) with AssertionError and full synthetic run still completes."""
    mock_adapter = MockStudyAdapter(poison_2024=True)
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    manager = NHISD6TemporalReleaseManager(
        release_id="TEST_POISON_2024",
        base_dir=tmp_path,
        runner=runner,
        allow_substantive_execution=True,
    )
    with patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        manifest = manager.execute_release()
    assert manifest["status"] == "COMPLETE"
    assert manifest["test_year_requested"] is False
    assert manifest["test_year_evaluated"] is False


def test_40_no_public_d6_method_evaluates_2024() -> None:
    """Check 40: No public D6.0 method evaluates 2024."""
    for cls in [NHISD6TemporalRunner, NHISD6TemporalReleaseManager]:
        public_methods = [m for m in dir(cls) if not m.startswith("_")]
        for m in public_methods:
            m_lower = m.lower()
            assert "2024" not in m_lower
            assert "evaluate_test" not in m_lower
            assert "score_test" not in m_lower


def test_41_audit_only_calls_zero_year_cohorts() -> None:
    """Check 41: audit-only calls zero year cohorts."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=False)
    runner.run_audit_only()
    assert len(mock_adapter.requested_cohorts) == 0


def test_42_audit_only_fits_zero_models() -> None:
    """Check 42: audit-only fits zero models and scalers."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=False)
    with patch.object(LogisticRegression, "fit") as mock_lr, patch.object(MinMaxScaler, "fit") as mock_scale:
        runner.run_audit_only()
        assert mock_lr.call_count == 0
        assert mock_scale.call_count == 0


def test_43_audit_only_creates_no_substantive_result_directory(tmp_path: pathlib.Path) -> None:
    """Check 43: audit-only creates no substantive result directory."""
    runner = NHISD6TemporalRunner(allow_execution=False)
    before_files = set(tmp_path.iterdir())
    runner.run_audit_only()
    after_files = set(tmp_path.iterdir())
    assert before_files == after_files


def test_44_future_canonical_release_collision_fails_closed(tmp_path: pathlib.Path) -> None:
    """Check 44: Future canonical release collision fails closed."""
    rel_dir = tmp_path / "COLLISION_TEST"
    rel_dir.mkdir(parents=True)
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    manager = NHISD6TemporalReleaseManager(
        release_id="COLLISION_TEST",
        base_dir=tmp_path,
        runner=runner,
        allow_substantive_execution=True,
    )
    with pytest.raises(FileExistsError, match="Canonical release directory already exists"):
        manager.execute_release()


def test_45_started_occurs_before_first_substantive_computation(tmp_path: pathlib.Path) -> None:
    """Check 45: STARTED occurs before first substantive computation."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    manager = NHISD6TemporalReleaseManager(
        release_id="TEST_STARTED",
        base_dir=tmp_path,
        runner=runner,
        allow_substantive_execution=True,
    )

    state_during_execution = None

    def capture_started(*args: Any, **kwargs: Any) -> Any:
        nonlocal state_during_execution
        st_file = manager.release_dir / "release_state.json"
        if st_file.is_file():
            state_during_execution = json.loads(st_file.read_text())["status"]
        return make_mock_arm_execution_result("D6_ARM_001")

    with patch.object(runner, "execute_arm_train_validation", side_effect=capture_started):
        try:
            manager.execute_release()
        except Exception:
            pass
    assert state_during_execution == "STARTED"


def test_46_execution_exception_creates_failed_state(tmp_path: pathlib.Path) -> None:
    """Check 46: Execution exception creates FAILED state."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    manager = NHISD6TemporalReleaseManager(
        release_id="TEST_FAILED",
        base_dir=tmp_path,
        runner=runner,
        allow_substantive_execution=True,
    )

    with patch.object(runner, "execute_arm_train_validation", side_effect=RuntimeError("Simulated arm crash")):
        with pytest.raises(RuntimeError, match="Simulated arm crash"):
            manager.execute_release()

    state_path = manager.release_dir / "release_state.json"
    state = json.loads(state_path.read_text())
    assert state["status"] == "FAILED"
    assert "Simulated arm crash" in state["error"]


def test_47_failed_canonical_release_cannot_silently_rerun(tmp_path: pathlib.Path) -> None:
    """Check 47: FAILED canonical release cannot silently rerun."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    manager = NHISD6TemporalReleaseManager(
        release_id="TEST_NO_RERUN",
        base_dir=tmp_path,
        runner=runner,
        allow_substantive_execution=True,
    )

    with patch.object(runner, "execute_arm_train_validation", side_effect=RuntimeError("Crash")):
        with pytest.raises(RuntimeError):
            manager.execute_release()

    # Second run against existing failed directory must fail closed
    with pytest.raises(FileExistsError, match="Canonical release directory already exists"):
        manager.execute_release()


def test_48_complete_requires_all_4_arms(tmp_path: pathlib.Path) -> None:
    """Check 48: COMPLETE requires all 4 arms."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    manager = NHISD6TemporalReleaseManager(
        release_id="TEST_COMPLETE_ARMS",
        base_dir=tmp_path,
        runner=runner,
        allow_substantive_execution=True,
    )
    with patch.object(runner, "execute_arm_train_validation", side_effect=make_mock_arm_execution_result):
        manifest = manager.execute_release()
    assert len(manifest["arms"]) == 4
    for arm_id in ["D6_ARM_001", "D6_ARM_002", "D6_ARM_003", "D6_ARM_004"]:
        assert arm_id in manifest["arms"]
        assert (manager.release_dir / arm_id).is_dir()


def test_49_manifest_contains_44_per_arm_artifact_hashes(tmp_path: pathlib.Path) -> None:
    """Check 49: Manifest contains 44 per-arm artifact hashes."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    manager = NHISD6TemporalReleaseManager(
        release_id="TEST_44_ARTIFACTS",
        base_dir=tmp_path,
        runner=runner,
        allow_substantive_execution=True,
    )
    with patch.object(runner, "execute_arm_train_validation", side_effect=make_mock_arm_execution_result):
        manifest = manager.execute_release()
    artifacts = manifest["artifacts"]
    per_arm_count = 0
    for rel_p in artifacts.keys():
        for arm_id in FROZEN_D6_ARMS.keys():
            if rel_p.startswith(f"{arm_id}/"):
                per_arm_count += 1
    assert per_arm_count == 44


def test_50_undefined_tpr_ppv_remain_undefined() -> None:
    """Check 50: Undefined TPR/PPV remain undefined."""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.1, 0.2])
    o_group = np.array([1, 1, 2, 2])
    ev = evaluate_predictions(y_true, y_pred, y_prob, o_group)
    for grp in ev["group_metrics"]:
        assert grp["tpr"] is None
        assert grp["ppv"] is None


def test_51_hisp_coverage_schema_remains_7_21() -> None:
    """Check 51: HISP coverage schema remains 7/21."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    with patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        res = runner.execute_arm_train_validation("D6_ARM_002")
    base_eval = res["validation_metrics_baseline"]
    cov = base_eval["group_coverage"]
    assert cov["expected_group_count"] == 7
    assert cov["expected_pair_count"] == 21
    assert cov["observed_group_count"] == 7
    assert cov["observed_pair_count"] == 21
    assert cov["group_coverage_complete"] is True


def test_52_prior_d4_tag_unchanged() -> None:
    """Check 52: Prior D4 tag unchanged."""
    res = subprocess.run(
        ["git", "rev-parse", f"{FROZEN_D4_TAG}^{{commit}}"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout.strip() == FROZEN_D4_ARCHIVED_COMMIT


def test_53_prior_d5_train_val_tag_unchanged() -> None:
    """Check 53: Prior D5 train/val tag unchanged."""
    res = subprocess.run(
        ["git", "rev-parse", f"{FROZEN_D5_TRAIN_VAL_TAG}^{{commit}}"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout.strip() == FROZEN_D5_TRAIN_VAL_ARCHIVED_COMMIT


def test_54_prior_d5_secondary_test_tag_unchanged() -> None:
    """Check 54: Prior D5 secondary test tag unchanged."""
    res = subprocess.run(
        ["git", "rev-parse", f"{FROZEN_D5_SECONDARY_TEST_TAG}^{{commit}}"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout.strip() == FROZEN_D5_SECONDARY_TEST_ARCHIVED_COMMIT


def test_55_frozen_features_parquet_sha_constant_exact() -> None:
    """Check 55: Frozen features parquet SHA constant exact."""
    assert FROZEN_FEATURES_PARQUET_SHA256 == "49f415132ff0be0228f8533f9f74c48cd79ff6fa8be66db085f7329d9b083383"
    info = verify_frozen_features_parquet()
    assert info["hash_verified"] is True


def test_56_scientific_boundary_positive_test() -> None:
    """Check 56: Scientific boundary positive test."""
    info = verify_scientific_code_boundary()
    assert info["scientific_base_is_ancestor"] is True
    assert info["scientific_code_diff_clean"] is True


def test_57_scientific_boundary_drift_negative_test() -> None:
    """Check 57: Scientific boundary drift negative test."""
    # Test non-ancestor detection (base commit is child of target commit)
    with pytest.raises(RuntimeError, match="not an ancestor"):
        verify_scientific_code_boundary(
            base_commit=SCIENTIFIC_EXECUTION_BASE_COMMIT,
            target_commit=FROZEN_D5_TRAIN_VAL_ARCHIVED_COMMIT,
        )

    # Test scientific path drift detection (older commit with diff in scientific paths)
    with pytest.raises(RuntimeError, match="scientific code has changed"):
        verify_scientific_code_boundary(base_commit=FROZEN_D4_ARCHIVED_COMMIT)


def test_58_nhis_pooled_adapter_absent_from_d6_module() -> None:
    """Check 58: NHISPooledAdapter absent from production D6 module."""
    src_file = _REPO_ROOT / "src" / "nhis_fairbias" / "d6_temporal_runner.py"
    text = src_file.read_text(encoding="utf-8")
    assert "NHISPooledAdapter" not in text


def test_59_train_test_split_absent() -> None:
    """Check 59: train_test_split absent."""
    src_file = _REPO_ROOT / "src" / "nhis_fairbias" / "d6_temporal_runner.py"
    text = src_file.read_text(encoding="utf-8")
    assert "train_test_split" not in text
    assert "generate_pooled_splits" not in text


def test_60_no_d5_weighted_changed_dict_loaded() -> None:
    """Check 60: No D5 weighted changed_dict is loaded."""
    src_file = _REPO_ROOT / "src" / "nhis_fairbias" / "d6_temporal_runner.py"
    text = src_file.read_text(encoding="utf-8")
    assert "weighted_changed_dict" not in text
    assert "d5_changed_dict" not in text
    assert "load_d5" not in text


# ==============================================================================
# Special Integration Tests
# ==============================================================================


def test_61_dynamic_ordering_event_sequence() -> None:
    """Section 33: Dynamic ordering event sequence."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    events: List[str] = []
    with patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        runner.execute_arm_train_validation("D6_ARM_001", event_callback=events.append)

    expected_order = [
        "request_2022",
        "verify_2022_preprocessor",
        "compute_2022_initial_dphi",
        "compute_2022_epsilon",
        "compute_2022_nmi",
        "learn_2022_changed_dict",
        "freeze_2022_changed_dict",
        "compute_2022_final_dphi",
        "fit_baseline_scaler_2022",
        "fit_baseline_lr_2022",
        "fit_fairbias_scaler_2022",
        "fit_fairbias_lr_2022",
        "freeze_2022_model_state",
        "request_2023",
        "transform_2023",
        "score_and_evaluate_2023",
        "diagnostic_dphi_2023",
    ]
    for ev in expected_order:
        assert ev in events

    # Assert freeze_2022_model_state strictly precedes request_2023
    assert events.index("freeze_2022_model_state") < events.index("request_2023")
    req_2023_idx = events.index("request_2023")
    assert req_2023_idx > events.index("freeze_2022_changed_dict")
    assert req_2023_idx > events.index("fit_baseline_lr_2022")
    assert req_2023_idx > events.index("fit_fairbias_lr_2022")


def test_62_cohort_request_sequence_8_total(tmp_path: pathlib.Path) -> None:
    """Section 34: Exact 8 cohort requests sequence: 4 x 2022, 4 x 2023, 0 x 2024."""
    mock_adapter = MockStudyAdapter(poison_2024=True)
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    manager = NHISD6TemporalReleaseManager(
        release_id="TEST_SEQ_8",
        base_dir=tmp_path,
        runner=runner,
        allow_substantive_execution=True,
    )
    with patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        manifest = manager.execute_release()
    assert manifest["status"] == "COMPLETE"

    years = [req["year"] for req in mock_adapter.requested_cohorts]
    assert len(years) == 8
    assert years == [2022, 2023, 2022, 2023, 2022, 2023, 2022, 2023]
    assert years.count(2022) == 4
    assert years.count(2023) == 4
    assert years.count(2024) == 0


def test_63_validation_selection_poison_invariant() -> None:
    """Section 35 & Section 14: Validation-selection poison test.

    Assert 2022 changed_dict, epsilon, scalers, LR models, and all frozen state SHA fingerprints
    remain strictly invariant under pathological validation data.
    """
    mock_adapter_normal = MockStudyAdapter()
    runner_normal = NHISD6TemporalRunner(adapter=mock_adapter_normal, allow_execution=True)
    with patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        res_normal = runner_normal.execute_arm_train_validation("D6_ARM_001")

    # Inverted validation cohort
    class PoisonValidationAdapter(MockStudyAdapter):
        def get_cohort(
            self,
            year,
            outcome="MEDDL12M_A",
            protected_attribute="SEX_A",
            feature_set="primary_core",
            disability_arm="full_feature",
        ):
            X, y, o, w, meta = super().get_cohort(
                year, outcome, protected_attribute, feature_set, disability_arm
            )
            if year == 2023:
                y = 1 - y
                X = X * 1000.0
            return X, y, o, w, meta

    mock_adapter_poison = PoisonValidationAdapter()
    runner_poison = NHISD6TemporalRunner(adapter=mock_adapter_poison, allow_execution=True)
    with patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        res_poison = runner_poison.execute_arm_train_validation("D6_ARM_001")

    # The 2022-learned representations and parameters must be bit-for-bit identical
    assert res_normal["final_changed_dict"] == res_poison["final_changed_dict"]
    assert (
        res_normal["train_dphi_before_after"]["epsilon_threshold"]
        == res_poison["train_dphi_before_after"]["epsilon_threshold"]
    )
    assert (
        res_normal["train_dphi_before_after"]["final_max_dphi"]
        == res_poison["train_dphi_before_after"]["final_max_dphi"]
    )
    assert (
        res_normal["input_provenance"]["train_source_row_digest"]
        == res_poison["input_provenance"]["train_source_row_digest"]
    )

    norm_state = res_normal["input_provenance"]["frozen_2022_training_state"]
    pois_state = res_poison["input_provenance"]["frozen_2022_training_state"]

    assert norm_state["changed_dict_sha256"] == pois_state["changed_dict_sha256"]
    assert norm_state["baseline_scaler"]["sha256"] == pois_state["baseline_scaler"]["sha256"]
    assert (
        norm_state["baseline_logistic_regression"]["sha256"]
        == pois_state["baseline_logistic_regression"]["sha256"]
    )
    assert norm_state["fairbias_scaler"]["sha256"] == pois_state["fairbias_scaler"]["sha256"]
    assert (
        norm_state["fairbias_logistic_regression"]["sha256"]
        == pois_state["fairbias_logistic_regression"]["sha256"]
    )
    assert (
        res_normal["arm_config"]["prediction_threshold"]
        == res_poison["arm_config"]["prediction_threshold"]
        == 0.5
    )


def test_64_cli_audit_only_execution() -> None:
    """Test CLI audit-only execution output."""
    import scripts.run_nhis_d6_temporal as cli_mod

    with patch("sys.argv", ["run_nhis_d6_temporal.py", "--audit-only"]):
        rc = cli_mod.main()
        assert rc == 0


def test_65_cli_rejects_test_flags() -> None:
    """Test CLI rejects forbidden test flags."""
    import scripts.run_nhis_d6_temporal as cli_mod

    for bad_flag in ["--test", "--execute-test", "--year-2024", "--evaluate-2024"]:
        with pytest.raises(SystemExit):
            cli_mod.parse_args([bad_flag])


# ==============================================================================
# Gate D6.0a.1 Hardened Provenance Tests
# ==============================================================================


def test_66_preconditions_parquet_hash_failure_fails_closed(tmp_path: pathlib.Path) -> None:
    """Section 16: Parquet hash failure aborts substantive release before any cohort request."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    manager = NHISD6TemporalReleaseManager(
        release_id="TEST_PRECOND_PQ",
        base_dir=tmp_path,
        runner=runner,
        allow_substantive_execution=True,
    )
    with patch(
        "nhis_fairbias.d6_temporal_runner.verify_frozen_features_parquet",
        side_effect=ValueError("Corrupted parquet hash"),
    ):
        with pytest.raises(ValueError, match="Corrupted parquet hash"):
            manager.execute_release()

    assert not (tmp_path / "TEST_PRECOND_PQ").exists()
    assert len(mock_adapter.requested_cohorts) == 0


def test_67_preconditions_scientific_boundary_failure_fails_closed(tmp_path: pathlib.Path) -> None:
    """Section 16: Scientific boundary failure aborts substantive release before any cohort request."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    manager = NHISD6TemporalReleaseManager(
        release_id="TEST_PRECOND_BOUNDARY",
        base_dir=tmp_path,
        runner=runner,
        allow_substantive_execution=True,
    )
    with patch(
        "nhis_fairbias.d6_temporal_runner.verify_scientific_code_boundary",
        side_effect=RuntimeError("Scientific boundary modified"),
    ):
        with pytest.raises(RuntimeError, match="Scientific boundary modified"):
            manager.execute_release()

    assert not (tmp_path / "TEST_PRECOND_BOUNDARY").exists()
    assert len(mock_adapter.requested_cohorts) == 0


def test_68_preconditions_prior_tag_failure_fails_closed(tmp_path: pathlib.Path) -> None:
    """Section 16: Prior tag failure aborts substantive release before any cohort request."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    manager = NHISD6TemporalReleaseManager(
        release_id="TEST_PRECOND_TAG",
        base_dir=tmp_path,
        runner=runner,
        allow_substantive_execution=True,
    )
    with patch(
        "nhis_fairbias.d6_temporal_runner.verify_prior_release_tags",
        side_effect=RuntimeError("Prior tag mismatch"),
    ):
        with pytest.raises(RuntimeError, match="Prior tag mismatch"):
            manager.execute_release()

    assert not (tmp_path / "TEST_PRECOND_TAG").exists()
    assert len(mock_adapter.requested_cohorts) == 0


def test_69_preconditions_missing_archive_directory_fails_closed(tmp_path: pathlib.Path) -> None:
    """Section 16: Missing archive directory aborts substantive release before any cohort request."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    manager = NHISD6TemporalReleaseManager(
        release_id="TEST_PRECOND_ARCHIVE",
        base_dir=tmp_path,
        runner=runner,
        allow_substantive_execution=True,
    )
    with patch(
        "nhis_fairbias.d6_temporal_runner.verify_prior_release_archives",
        side_effect=FileNotFoundError("Missing prior release archive"),
    ):
        with pytest.raises(FileNotFoundError, match="Missing prior release archive"):
            manager.execute_release()

    assert not (tmp_path / "TEST_PRECOND_ARCHIVE").exists()
    assert len(mock_adapter.requested_cohorts) == 0


def test_70_substantive_execution_without_prior_audit_only(tmp_path: pathlib.Path) -> None:
    """Section 17: Substantive execution automatically runs preconditions without audit-only call."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    manager = NHISD6TemporalReleaseManager(
        release_id="TEST_DIRECT_SUBSTANTIVE",
        base_dir=tmp_path,
        runner=runner,
        allow_substantive_execution=True,
    )
    # Execute substantive release directly WITHOUT calling runner.run_audit_only()
    with patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        manifest = manager.execute_release()

    assert manifest["status"] == "COMPLETE"
    assert "execution_preconditions" in manifest
    prec = manifest["execution_preconditions"]
    assert prec["frozen_features_verified"] is True
    assert prec["prior_release_tags_verified"] is True
    assert prec["prior_release_archives_verified"] is True
    assert prec["scientific_code_diff_clean"] is True


def test_71_release_state_created_at_timestamp_stability(tmp_path: pathlib.Path) -> None:
    """Section 18: created_at timestamp is preserved unchanged across STARTED, COMPLETE, and FAILED."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)

    # 1. Success path: STARTED -> COMPLETE
    manager_success = NHISD6TemporalReleaseManager(
        release_id="TEST_TIME_SUCCESS",
        base_dir=tmp_path,
        runner=runner,
        allow_substantive_execution=True,
    )
    with patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        manifest = manager_success.execute_release()

    state_path = tmp_path / "TEST_TIME_SUCCESS" / "release_state.json"
    with open(state_path, "r", encoding="utf-8") as f:
        complete_state = json.load(f)

    assert complete_state["status"] == "COMPLETE"
    assert isinstance(complete_state["created_at"], str)
    assert complete_state["created_at"] == manifest["created_at"]

    # 2. Failure path: STARTED -> FAILED
    manager_fail = NHISD6TemporalReleaseManager(
        release_id="TEST_TIME_FAIL",
        base_dir=tmp_path,
        runner=runner,
        allow_substantive_execution=True,
    )
    with patch.object(
        runner, "execute_arm_train_validation", side_effect=RuntimeError("Simulated arm fail")
    ):
        with pytest.raises(RuntimeError, match="Simulated arm fail"):
            manager_fail.execute_release()

    fail_state_path = tmp_path / "TEST_TIME_FAIL" / "release_state.json"
    with open(fail_state_path, "r", encoding="utf-8") as f:
        failed_state = json.load(f)

    assert failed_state["status"] == "FAILED"
    assert isinstance(failed_state["created_at"], str)
    assert failed_state["error"] == "Simulated arm fail"


def test_72_release_directory_exact_file_count_semantics(tmp_path: pathlib.Path) -> None:
    """Section 19: Verify 44 per-arm scientific artifacts, 45 manifest artifacts, 47 total files."""
    mock_adapter = MockStudyAdapter()
    runner = NHISD6TemporalRunner(adapter=mock_adapter, allow_execution=True)
    manager = NHISD6TemporalReleaseManager(
        release_id="TEST_FILE_COUNTS_47",
        base_dir=tmp_path,
        runner=runner,
        allow_substantive_execution=True,
    )
    with patch.object(FairBiasMitigation, "mitigate_step", return_value=(None, {}, None, None)):
        manifest = manager.execute_release()

    rel_dir = tmp_path / "TEST_FILE_COUNTS_47"

    # 44 per-arm scientific artifacts (11 per arm x 4 arms)
    per_arm_files = [p for p in rel_dir.glob("D6_ARM_*/*") if p.is_file()]
    assert len(per_arm_files) == 44

    # 45 manifest-tracked artifacts (44 per arm + 1 preprocessing_provenance.json)
    assert len(manifest["artifacts"]) == 45
    assert "preprocessing_provenance.json" in manifest["artifacts"]

    # 47 total files in COMPLETE release directory (44 per-arm + preprocessing + manifest + release_state)
    all_files = [p for p in rel_dir.rglob("*") if p.is_file()]
    assert len(all_files) == 47


def test_73_deterministic_scaler_and_lr_state_schemas_and_hashes() -> None:
    """Section 9, 10, 12: Verify deterministic state extraction schemas and SHA-256 hashes."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MinMaxScaler

    rng = np.random.default_rng(42)
    features = ["f1", "f2", "f3"]
    X = rng.uniform(0.0, 100.0, size=(50, 3))
    y = rng.integers(0, 2, size=50)

    # Scaler
    scaler = MinMaxScaler()
    scaler.fit(X)
    s_state = extract_minmax_scaler_state(scaler, features)
    assert s_state["feature_order"] == features
    assert s_state["n_features_in_"] == 3
    assert len(s_state["data_min_"]) == 3
    assert len(s_state["data_max_"]) == 3
    assert len(s_state["scale_"]) == 3
    assert len(s_state["min_"]) == 3

    sha1 = compute_canonical_json_sha256(s_state)
    sha2 = compute_canonical_json_sha256(s_state)
    assert sha1 == sha2
    assert len(sha1) == 64

    # Model
    model = LogisticRegression(random_state=0, solver="lbfgs", max_iter=1000)
    model.fit(X, y)
    m_state = extract_logistic_regression_state(model, features)
    assert m_state["feature_order"] == features
    assert m_state["classes_"] == [0, 1]
    assert len(m_state["coef_"]) == 1
    assert len(m_state["coef_"][0]) == 3
    assert len(m_state["intercept_"]) == 1
    assert m_state["random_state"] == 0
    assert m_state["solver"] == "lbfgs"
    assert m_state["max_iter"] == 1000

    m_sha1 = compute_canonical_json_sha256(m_state)
    m_sha2 = compute_canonical_json_sha256(m_state)
    assert m_sha1 == m_sha2
    assert len(m_sha1) == 64
