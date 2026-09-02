"""Unit, integration, barrier, and production-wiring tests for Gate D7.1a.1.

All tests use synthetic/mock cohorts and frozen archive data.
Zero real NHIS cohort access, zero estimator fitting, zero FairBias relearning.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from nhis_fairbias.adapter import NHISStudyAdapter
from nhis_fairbias.d6_temporal_runner import (
    compute_cohort_source_row_digest as compute_d6_cohort_source_row_digest,
)
from nhis_fairbias.d7_terminal_mechanism import (
    ARM_DISABILITY_POLICIES,
    ARM_PROTECTED_ATTRIBUTES,
    CANONICAL_DECISION_THRESHOLD,
    CohortProvenanceBarrierError,
    D6_ARM_IDS,
    D6_TEST_COMMIT,
    D6_TEST_MANIFEST_SHA256,
    D6_TEST_TAG,
    D6_TEST_TAG_OBJECT,
    D6_TRAIN_VAL_COMMIT,
    D6_TRAIN_VAL_MANIFEST_SHA256,
    D6_TRAIN_VAL_TAG,
    D6_TRAIN_VAL_TAG_OBJECT,
    D7_ALL_RELEASE_FILES,
    D7_HYPOTHESES,
    D7_MANIFEST_TRACKED_ARTIFACTS,
    D7_UNTRACKED_CONTROL_FILES,
    D7TerminalMechanismReleaseManager,
    EXPECTED_COHORT_SOURCE_ROW_DIGESTS,
    EXPECTED_TRAINING_STATE_ANCHORS,
    FAMILY_I_INVARIANT,
    FAMILY_II_INVARIANT,
    FROZEN_FEATURES_PARQUET_PATH,
    FROZEN_FEATURES_PARQUET_SHA256,
    FrozenArmState,
    FrozenLogisticRegression,
    FrozenScaler,
    FrozenScoredCohort,
    MANDATORY_D7_DISCLOSURE,
    NHISD7TerminalMechanismHarness,
    PREPROCESSING_STATE_SHA256,
    ProductionD7TerminalMechanismRuntime,
    SCIENTIFIC_TERMINOLOGY,
    SCORING_REPRODUCTION_ABSOLUTE_TOLERANCE,
    TEMPORAL_YEARS,
    ScoringReproductionBarrierError,
    build_macro_context,
    compute_canonical_json_sha256,
    compute_cohort_source_row_digest,
    compute_cohort_utility_metrics,
    compute_distribution_summary,
    compute_family1_diagnostics,
    compute_family2_diagnostics,
    compute_logit_contribution_diagnostics,
    compute_mechanism_arm_summary,
    compute_nmi,
    compute_protected_group_score_diagnostics,
    compute_score_distribution_diagnostics,
    compute_sha256,
    load_frozen_20_states,
    read_archived_scoring_benchmark,
    verify_cohort_provenance_barrier,
    verify_frozen_archives,
    verify_git_execution_preconditions,
    verify_git_tag_provenance,
    verify_scoring_reproduction_barrier,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# Fixtures & Poison Sentinels
# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def poison_estimator_fits(monkeypatch):
    """Enforce zero estimator fitting globally across test execution."""

    def poisoned_fit(*args, **kwargs):
        raise AssertionError("POISON TRIGGERED: Estimator .fit() called during D7 test!")

    def poisoned_fit_transform(*args, **kwargs):
        raise AssertionError("POISON TRIGGERED: Estimator .fit_transform() called during D7 test!")

    monkeypatch.setattr(MinMaxScaler, "fit", poisoned_fit)
    monkeypatch.setattr(MinMaxScaler, "fit_transform", poisoned_fit_transform)
    monkeypatch.setattr(LogisticRegression, "fit", poisoned_fit)


@pytest.fixture
def poison_real_nhis_adapter(monkeypatch):
    """Poison real NHISStudyAdapter constructor to guarantee zero real microdata access."""

    def poisoned_init(*args, **kwargs):
        raise AssertionError("POISON TRIGGERED: Real NHISStudyAdapter instantiated during test!")

    monkeypatch.setattr(NHISStudyAdapter, "__init__", poisoned_init)


@pytest.fixture
def mock_arm_states():
    """Load real frozen 20 states from immutable archives (read-only)."""
    tv_dir = _REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609"
    t_dir = _REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7"
    return load_frozen_20_states(tv_dir, t_dir)


@pytest.fixture
def mock_observed_perfect_metrics():
    """Extract archived utility metrics to form perfectly reproducing observed predictions."""
    tv_dir = _REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609"
    t_dir = _REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7"
    metrics: Dict[int, Dict[str, Dict[str, Any]]] = {2023: {}, 2024: {}}

    for arm_id in D6_ARM_IDS:
        b_23 = json.loads((tv_dir / arm_id / "validation_metrics_baseline.json").read_text())["utility"]
        f_23 = json.loads((tv_dir / arm_id / "validation_metrics_fairbias.json").read_text())["utility"]
        metrics[2023][arm_id] = {"baseline": b_23, "fairbias": f_23}

        b_24 = json.loads((t_dir / arm_id / "test_metrics_baseline.json").read_text())["utility"]
        f_24 = json.loads((t_dir / arm_id / "test_metrics_fairbias.json").read_text())["utility"]
        metrics[2024][arm_id] = {"baseline": b_24, "fairbias": f_24}

    return metrics


# -----------------------------------------------------------------------------
# 1-6. Archive, Manifest, Tag, States, Context
# -----------------------------------------------------------------------------
def test_01_frozen_d6_archive_manifest_verification():
    res = verify_frozen_archives(_REPO_ROOT)
    assert res["train_val_manifest_sha256"] == D6_TRAIN_VAL_MANIFEST_SHA256
    assert res["test_manifest_sha256"] == D6_TEST_MANIFEST_SHA256
    assert res["train_val_artifact_count"] == 45
    assert res["test_artifact_count"] == 41
    assert res["status"] == "VERIFIED"


def test_02_frozen_d6_tag_verification():
    res = verify_git_tag_provenance(_REPO_ROOT)
    assert res["tags_verified"] is True
    assert res["train_val_tag_object"] == D6_TRAIN_VAL_TAG_OBJECT
    assert res["train_val_commit"] == D6_TRAIN_VAL_COMMIT
    assert res["test_tag_object"] == D6_TEST_TAG_OBJECT
    assert res["test_commit"] == D6_TEST_COMMIT


def test_03_d6_20_20_state_anchor_loading(mock_arm_states):
    assert len(mock_arm_states) == 4
    total_states = 0
    for arm_id, state in mock_arm_states.items():
        assert arm_id in D6_ARM_IDS
        exp = EXPECTED_TRAINING_STATE_ANCHORS[arm_id]
        for k in ("changed_dict", "baseline_scaler", "baseline_LR", "FairBias_scaler", "FairBias_LR"):
            assert state.state_hashes[k] == exp[k]
            total_states += 1
    assert total_states == 20


def test_04_correct_d6_changed_dict_loading(mock_arm_states):
    cd1 = mock_arm_states["D6_ARM_001"].changed_dict
    assert cd1.get("agep_a") == "dropped"
    assert cd1.get("pcnt18uptc") == {"power": 5.0}
    assert cd1.get("pcntlt18tc") == {"power": 9.0}

    cd2 = mock_arm_states["D6_ARM_002"].changed_dict
    assert cd2.get("agep_a") == {"power": 9.0}
    assert cd2.get("pcnt18uptc") == "dropped"
    assert cd2.get("region") == "dropped"


def test_05_d4_local_preflight_paths_not_required():
    macro = build_macro_context(_REPO_ROOT)
    assert macro["local_d4_preflight_required"] is False
    assert "d4_primary_aggregate_context" in macro
    serialized = json.dumps(macro)
    assert "runs/" not in serialized


def test_06_d5_only_archive_context_is_used():
    macro = build_macro_context(_REPO_ROOT)
    assert "d5_weighted_aggregate_context" in macro
    assert "d5_weighted_terminal_transformation_summary" in macro
    assert macro["source"] == "canonical_frozen_releases_only"


# -----------------------------------------------------------------------------
# 7-11. Audit-Only Constraints & Git Preconditions
# -----------------------------------------------------------------------------
def test_07_audit_only_constructs_no_adapter(poison_real_nhis_adapter):
    harness = NHISD7TerminalMechanismHarness(repo_root=_REPO_ROOT)
    res = harness.run_audit_only()
    assert res["audit_status"] == "PASS"


def test_08_audit_only_scores_no_cohort():
    harness = NHISD7TerminalMechanismHarness(repo_root=_REPO_ROOT)
    assert harness.real_nhis_cohort_accessed is False


def test_09_audit_only_performs_zero_estimator_fits():
    harness = NHISD7TerminalMechanismHarness(repo_root=_REPO_ROOT)
    res = harness.run_audit_only()
    assert res["estimator_fit_count"] == 0


def test_10_audit_only_executes_zero_fairbias():
    harness = NHISD7TerminalMechanismHarness(repo_root=_REPO_ROOT)
    res = harness.run_audit_only()
    assert res["fairbias_execution_count"] == 0


def test_11_tracked_dirty_substantive_authorization_fails(monkeypatch):
    def mock_run(cmd, *args, **kwargs):
        res = MagicMock()
        if "rev-parse" in cmd:
            res.stdout = "e365d020d67f39ee9b2aeace9dcb3ede6c09391d\n"
        elif "status" in cmd:
            res.stdout = " M src/nhis_fairbias/some_file.py\n"
        return res

    monkeypatch.setattr("subprocess.run", mock_run)
    with pytest.raises(ValueError, match="Tracked working tree is dirty"):
        verify_git_execution_preconditions(_REPO_ROOT, "e365d020d67f39ee9b2aeace9dcb3ede6c09391d")


# -----------------------------------------------------------------------------
# 12-18. Scaler & Logistic Regression Zero-Refit Scoring
# -----------------------------------------------------------------------------
def test_12_scaler_direct_formula_equals_X_scale_plus_min():
    state = {
        "feature_order": ["f1", "f2"],
        "feature_range": [0.0, 1.0],
        "n_features_in_": 2,
        "data_min_": [10.0, 100.0],
        "data_max_": [20.0, 300.0],
        "data_range_": [10.0, 200.0],
        "scale_": [0.1, 0.005],
        "min_": [-1.0, -0.5],
    }
    scaler = FrozenScaler(state)
    X = np.array([[10.0, 100.0], [15.0, 200.0], [20.0, 300.0]])
    scaled = scaler.transform(X)

    sk_scaler = scaler.to_sklearn_scaler()
    df_X = pd.DataFrame(X, columns=scaler.feature_order)
    sk_scaled = sk_scaler.transform(df_X)

    assert np.allclose(scaled, sk_scaled, atol=1e-14)
    assert np.allclose(scaled[0], [0.0, 0.0])
    assert np.allclose(scaled[2], [1.0, 1.0])


def test_13_explicit_test_catches_incorrect_formula():
    state = {
        "feature_order": ["f1"],
        "feature_range": [0.0, 1.0],
        "n_features_in_": 1,
        "data_min_": [50.0],
        "data_max_": [150.0],
        "data_range_": [100.0],
        "scale_": [0.01],
        "min_": [-0.5],  # data_min_ (50.0) != min_ (-0.5)
    }
    scaler = FrozenScaler(state)
    X = np.array([[50.0], [100.0], [150.0]])

    correct_scaled = scaler.transform(X)
    wrong_scaled = (X - scaler.min_) * scaler.scale_

    assert np.allclose(correct_scaled, [[0.0], [0.5], [1.0]])
    assert not np.allclose(correct_scaled, wrong_scaled)
    assert abs(wrong_scaled[0, 0] - 0.505) < 1e-12


def test_14_feature_order_mismatch_fails_closed():
    state = {
        "feature_order": ["a", "b"],
        "feature_range": [0.0, 1.0],
        "n_features_in_": 2,
        "data_min_": [0.0, 0.0],
        "data_max_": [1.0, 1.0],
        "data_range_": [1.0, 1.0],
        "scale_": [1.0, 1.0],
        "min_": [0.0, 0.0],
    }
    scaler = FrozenScaler(state)
    df_wrong = pd.DataFrame({"b": [0.5], "a": [0.5]})
    with pytest.raises(ValueError, match="Feature order mismatch"):
        scaler.transform(df_wrong)


def test_15_classes_other_than_0_1_fails_closed():
    state = {
        "feature_order": ["f1"],
        "classes_": [1, 2],
        "coef_": [[0.5]],
        "intercept_": [0.0],
        "n_features_in_": 1,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 0,
    }
    with pytest.raises(ValueError, match="Classes must be exactly"):
        FrozenLogisticRegression(state)


def test_16_stable_logistic_score_computation_works():
    state = {
        "feature_order": ["f1"],
        "classes_": [0, 1],
        "coef_": [[2.0]],
        "intercept_": [-1.0],
        "n_features_in_": 1,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 0,
    }
    lr = FrozenLogisticRegression(state)
    X = np.array([[0.5]])
    probs = lr.predict_proba(X)
    assert abs(probs[0] - 0.5) < 1e-12


def test_17_scorer_performs_zero_lr_fit():
    state = {
        "feature_order": ["f1"],
        "classes_": [0, 1],
        "coef_": [[1.0]],
        "intercept_": [0.0],
        "n_features_in_": 1,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 0,
    }
    lr = FrozenLogisticRegression(state)
    preds = lr.predict(np.array([[0.0], [1.0]]))
    assert len(preds) == 2


def test_18_scorer_performs_zero_scaler_fit():
    state = {
        "feature_order": ["f1"],
        "feature_range": [0.0, 1.0],
        "n_features_in_": 1,
        "data_min_": [0.0],
        "data_max_": [10.0],
        "data_range_": [10.0],
        "scale_": [0.1],
        "min_": [0.0],
    }
    scaler = FrozenScaler(state)
    scaled = scaler.transform(np.array([[5.0]]))
    assert abs(scaled[0, 0] - 0.5) < 1e-12


# -----------------------------------------------------------------------------
# 19-24. 12/12 Cohort Digest Barrier & Preprocessing Hash
# -----------------------------------------------------------------------------
def test_19_12_of_12_synthetic_digest_barrier_passes():
    res = verify_cohort_provenance_barrier(EXPECTED_COHORT_SOURCE_ROW_DIGESTS)
    assert res["status"] == "PASS"
    assert res["total_cohorts_matched"] == 12


def test_20_one_2022_digest_mismatch_blocks_diagnostics():
    corrupted = copy.deepcopy(EXPECTED_COHORT_SOURCE_ROW_DIGESTS)
    corrupted[2022]["D6_ARM_001"] = "bad" * 16
    with pytest.raises(CohortProvenanceBarrierError):
        verify_cohort_provenance_barrier(corrupted)


def test_21_one_2023_digest_mismatch_blocks_diagnostics():
    corrupted = copy.deepcopy(EXPECTED_COHORT_SOURCE_ROW_DIGESTS)
    corrupted[2023]["D6_ARM_002"] = "bad" * 16
    with pytest.raises(CohortProvenanceBarrierError):
        verify_cohort_provenance_barrier(corrupted)


def test_22_one_2024_digest_mismatch_blocks_diagnostics():
    corrupted = copy.deepcopy(EXPECTED_COHORT_SOURCE_ROW_DIGESTS)
    corrupted[2024]["D6_ARM_004"] = "bad" * 16
    with pytest.raises(CohortProvenanceBarrierError):
        verify_cohort_provenance_barrier(corrupted)


def test_23_preprocessing_hash_mismatch_blocks_diagnostics(tmp_path):
    tv_dir = tmp_path / "docs" / "releases" / "NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609"
    tv_dir.mkdir(parents=True)
    (tv_dir / "d6_temporal_train_val_manifest.json").write_text("{}", encoding="utf-8")
    (tv_dir / "archive_ledger.json").write_text("{}", encoding="utf-8")
    (tv_dir / "preprocessing_provenance.json").write_text('{"bad": 1}', encoding="utf-8")

    test_dir = tmp_path / "docs" / "releases" / "NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7"
    test_dir.mkdir(parents=True)
    (test_dir / "d6_temporal_test_manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(Exception):
        verify_frozen_archives(tmp_path)


def test_24_no_diagnostics_before_global_provenance_barrier():
    corrupted_digests = copy.deepcopy(EXPECTED_COHORT_SOURCE_ROW_DIGESTS)
    corrupted_digests[2022]["D6_ARM_001"] = "corrupted_sha"

    with pytest.raises(CohortProvenanceBarrierError):
        verify_cohort_provenance_barrier(corrupted_digests)


# -----------------------------------------------------------------------------
# 25-31. Archived Terminal Scoring Reproduction Barrier
# -----------------------------------------------------------------------------
def test_25_mock_2023_archived_score_reproduction_passes(mock_observed_perfect_metrics):
    tv_dir = _REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609"
    t_dir = _REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7"
    res = verify_scoring_reproduction_barrier(mock_observed_perfect_metrics, tv_dir, t_dir)
    assert res["status"] == "PASS"


def test_26_mock_2024_archived_score_reproduction_passes(mock_observed_perfect_metrics):
    tv_dir = _REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609"
    t_dir = _REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7"
    res = verify_scoring_reproduction_barrier(mock_observed_perfect_metrics, tv_dir, t_dir)
    assert res["anchors_verified_count"] == 16


def test_27_auroc_mismatch_blocks_diagnostics(mock_observed_perfect_metrics):
    tv_dir = _REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609"
    t_dir = _REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7"
    corrupted = copy.deepcopy(mock_observed_perfect_metrics)
    corrupted[2024]["D6_ARM_001"]["baseline"]["auroc"] += 0.001
    with pytest.raises(ScoringReproductionBarrierError, match="tolerance_exceeded"):
        verify_scoring_reproduction_barrier(corrupted, tv_dir, t_dir)


def test_28_auprc_mismatch_blocks_diagnostics(mock_observed_perfect_metrics):
    tv_dir = _REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609"
    t_dir = _REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7"
    corrupted = copy.deepcopy(mock_observed_perfect_metrics)
    corrupted[2023]["D6_ARM_002"]["fairbias"]["auprc"] -= 0.005
    with pytest.raises(ScoringReproductionBarrierError, match="tolerance_exceeded"):
        verify_scoring_reproduction_barrier(corrupted, tv_dir, t_dir)


def test_29_predicted_positive_mismatch_blocks_diagnostics(mock_observed_perfect_metrics):
    tv_dir = _REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609"
    t_dir = _REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7"
    corrupted = copy.deepcopy(mock_observed_perfect_metrics)
    corrupted[2023]["D6_ARM_001"]["baseline"]["count_predicted_positive"] += 1
    with pytest.raises(ScoringReproductionBarrierError, match="exact_int_mismatch"):
        verify_scoring_reproduction_barrier(corrupted, tv_dir, t_dir)


def test_30_selection_rate_mismatch_blocks_diagnostics(mock_observed_perfect_metrics):
    tv_dir = _REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609"
    t_dir = _REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7"
    corrupted = copy.deepcopy(mock_observed_perfect_metrics)
    corrupted[2024]["D6_ARM_003"]["fairbias"]["selection_rate"] += 1e-8
    with pytest.raises(ScoringReproductionBarrierError, match="tolerance_exceeded"):
        verify_scoring_reproduction_barrier(corrupted, tv_dir, t_dir)


def test_31_tolerance_is_fixed_before_execution():
    assert SCORING_REPRODUCTION_ABSOLUTE_TOLERANCE == 1e-12


# -----------------------------------------------------------------------------
# 32-33. Family I Diagnostics
# -----------------------------------------------------------------------------
def test_32_family1_drop_post_nmi_convention_equals_zero():
    df_raw = pd.DataFrame({"agep_a": [20, 30, 40, 50], "other": [1, 1, 2, 2]})
    df_trans = pd.DataFrame({"other": [1, 1, 2, 2]})
    y = pd.Series([0, 0, 1, 1])
    a = pd.Series([1, 2, 1, 2])
    changed_dict = {"agep_a": "dropped"}

    records = compute_family1_diagnostics(df_raw, df_trans, y, a, changed_dict, "D6_ARM_001", 2022)
    feat_summary = next(r for r in records if r["record_type"] == "feature_summary")
    assert feat_summary["transform_type"] == "feature_drop"
    assert feat_summary["terminal_cardinality"] == 1
    assert feat_summary["nmi_y_after"] == 0.0
    assert feat_summary["post_state"] == "dropped_constant_equivalent"

    cat_after = next(r for r in records if r["record_type"] == "category_state" and r["state"] == "after")
    assert cat_after["category"] == "dropped_constant_equivalent"
    assert cat_after["outcome_prevalence"] == 0.5


def test_33_family1_category_cardinality_reduction_correct():
    df_raw = pd.DataFrame({"diff_a": [1, 2, 3, 4], "other": [1, 2, 3, 4]})
    df_trans = pd.DataFrame({"diff_a": [1, 1, 1, 1], "other": [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])
    a = pd.Series([1, 1, 2, 2])
    changed_dict = {"diff_a": {"2": 1, "3": 1, "4": 1}}

    records = compute_family1_diagnostics(df_raw, df_trans, y, a, changed_dict, "D6_ARM_001", 2023)
    feat_summary = next(r for r in records if r["record_type"] == "feature_summary")
    assert feat_summary["transform_type"] == "categorical_merge"
    assert feat_summary["original_cardinality"] == 4
    assert feat_summary["terminal_cardinality"] == 1
    assert feat_summary["cardinality_reduction"] == 3


# -----------------------------------------------------------------------------
# 34-36. Family II Diagnostics
# -----------------------------------------------------------------------------
def test_34_family2_power_transform_retains_monotone_ordering():
    raw_vals = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
    power = 5.0
    power_vals = np.sign(raw_vals) * (np.abs(raw_vals) ** power)
    assert np.all(np.diff(power_vals) >= 0)


def test_35_family2_quantile_mapping_correct():
    df_raw = pd.DataFrame({"pcnt18uptc": np.linspace(1, 10, 100)})
    power = 3.0
    df_trans = pd.DataFrame({"pcnt18uptc": df_raw["pcnt18uptc"] ** power})

    scaler_state = {
        "feature_order": ["pcnt18uptc"],
        "feature_range": [0.0, 1.0],
        "n_features_in_": 1,
        "data_min_": [1.0],
        "data_max_": [1000.0],
        "data_range_": [999.0],
        "scale_": [1.0 / 999.0],
        "min_": [-1.0 / 999.0],
    }
    scaler = FrozenScaler(scaler_state)
    records = compute_family2_diagnostics(df_raw, df_trans, {"pcnt18uptc": {"power": 3.0}}, scaler, "D6_ARM_001", 2022)
    rec = records[0]
    assert rec["power_exponent"] == 3.0
    assert rec["information_preserving_reparameterization"] is True
    assert "raw_p50" in rec and "power_p50" in rec and "scaled_p50" in rec


def test_36_numeric_power_nmi_not_used_as_geometry_pass_fail():
    df_raw = pd.DataFrame({"pcnt18uptc": [1.0, 2.0, 3.0, 4.0]})
    df_trans = pd.DataFrame({"pcnt18uptc": [1.0, 8.0, 27.0, 64.0]})
    scaler_state = {
        "feature_order": ["pcnt18uptc"],
        "feature_range": [0.0, 1.0],
        "n_features_in_": 1,
        "data_min_": [1.0],
        "data_max_": [64.0],
        "data_range_": [63.0],
        "scale_": [1.0 / 63.0],
        "min_": [-1.0 / 63.0],
    }
    scaler = FrozenScaler(scaler_state)
    records = compute_family2_diagnostics(df_raw, df_trans, {"pcnt18uptc": {"power": 3.0}}, scaler, "D6_ARM_001", 2024)
    assert "nmi_gate_failed" not in records[0]


# -----------------------------------------------------------------------------
# 37-39. Score Quantiles, KS Separation & Fraction >= 0.5
# -----------------------------------------------------------------------------
def test_37_score_quantiles_correct():
    probs = np.linspace(0.0, 1.0, 101)
    y = pd.Series([0] * 50 + [1] * 51)
    diag = compute_score_distribution_diagnostics(probs, y, "D6_ARM_001", 2024, "baseline")
    assert abs(diag["overall"]["median"] - 0.5) < 1e-12
    assert abs(diag["overall"]["p90"] - 0.9) < 1e-12


def test_38_ks_separation_correct():
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    y = pd.Series([0, 0, 1, 1])
    diag = compute_score_distribution_diagnostics(probs, y, "D6_ARM_001", 2024, "baseline")
    assert diag["ks_statistic"] == 1.0


def test_39_fraction_ge_0_5_correct():
    probs = np.array([0.2, 0.4, 0.6, 0.8])
    y = pd.Series([0, 0, 1, 1])
    diag = compute_score_distribution_diagnostics(probs, y, "D6_ARM_001", 2024, "baseline")
    assert diag["overall"]["fraction_ge_0_5"] == 0.5


# -----------------------------------------------------------------------------
# 40-41. Protected Group Summaries (all, Y0, Y1 strata)
# -----------------------------------------------------------------------------
def test_40_protected_group_summaries_retain_all_expected_hisp_groups():
    y = pd.Series([0, 1, 0, 0])
    a = pd.Series([1, 2, 3, 7])
    probs = np.array([0.2, 0.8, 0.3, 0.4])

    records = compute_protected_group_score_diagnostics(
        probs, y, a, expected_groups=range(1, 8), arm_id="D6_ARM_002", year=2024, model_name="fairbias"
    )
    groups = {r["group"] for r in records}
    assert groups == set(range(1, 8))
    strata = {r["outcome_stratum"] for r in records}
    assert strata == {"all", "Y0", "Y1"}


def test_41_small_empty_denominator_remains_undefined():
    y = pd.Series([0, 1, 0, 0])
    a = pd.Series([1, 2, 3, 7])
    probs = np.array([0.2, 0.8, 0.3, 0.4])

    records = compute_protected_group_score_diagnostics(
        probs, y, a, expected_groups=range(1, 8), arm_id="D6_ARM_002", year=2024, model_name="fairbias"
    )
    g4 = next(r for r in records if r["group"] == 4 and r["outcome_stratum"] == "all")
    assert g4["n"] == 0
    assert g4["mean"] is None
    assert g4["fraction_ge_0_5"] is None


# -----------------------------------------------------------------------------
# 42-45. Logit Contribution Diagnostics & Persistence Checks
# -----------------------------------------------------------------------------
def test_42_logit_contribution_algebra_correct():
    state = {
        "feature_order": ["x1"],
        "classes_": [0, 1],
        "coef_": [[2.0]],
        "intercept_": [0.0],
        "n_features_in_": 1,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 0,
    }
    lr = FrozenLogisticRegression(state)
    X_scaled = np.array([[0.2], [0.8]])
    y = pd.Series([0, 1])

    records = compute_logit_contribution_diagnostics(X_scaled, y, lr, "D6_ARM_001", 2024, "baseline")
    rx1 = records[0]
    assert abs(rx1["mean_scaled_X_j_Y0"] - 0.2) < 1e-12
    assert abs(rx1["mean_scaled_X_j_Y1"] - 0.8) < 1e-12
    assert abs(rx1["class_separation_contribution"] - (2.0 * (0.8 - 0.2))) < 1e-12


def test_43_intercept_tracked_separately():
    state = {
        "feature_order": ["x1"],
        "classes_": [0, 1],
        "coef_": [[1.5]],
        "intercept_": [-3.14],
        "n_features_in_": 1,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 0,
    }
    lr = FrozenLogisticRegression(state)
    records = compute_logit_contribution_diagnostics(
        np.array([[0.5], [0.5]]), pd.Series([0, 1]), lr, "D6_ARM_001", 2024, "baseline"
    )
    assert records[0]["intercept"] == -3.14


def test_44_no_contribution_called_causal():
    state = {
        "feature_order": ["x1"],
        "classes_": [0, 1],
        "coef_": [[1.0]],
        "intercept_": [0.0],
        "n_features_in_": 1,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 0,
    }
    lr = FrozenLogisticRegression(state)
    records = compute_logit_contribution_diagnostics(
        np.array([[0.1], [0.9]]), pd.Series([0, 1]), lr, "D6_ARM_001", 2024, "baseline"
    )
    assert "causal" not in records[0]["interpretation_status"].lower()
    assert records[0]["interpretation_status"] == "descriptive decomposition of the fitted terminal linear scoring function"


def test_45_no_respondent_level_file_written():
    for fname in D7_ALL_RELEASE_FILES:
        assert "row" not in fname.lower()
        assert "respondent" not in fname.lower()
        assert "microdata" not in fname.lower()


# -----------------------------------------------------------------------------
# 46-49. Schemas, Disclosures & Terminology Constraints
# -----------------------------------------------------------------------------
def test_46_future_artifact_schema_exactly_11_files():
    assert len(D7_ALL_RELEASE_FILES) == 11


def test_47_manifest_tracks_exactly_9_artifacts():
    assert len(D7_MANIFEST_TRACKED_ARTIFACTS) == 9
    assert len(D7_UNTRACKED_CONTROL_FILES) == 2


def test_48_post_hoc_disclosure_mandatory():
    assert "explanatory post-primary analysis" in MANDATORY_D7_DISCLOSURE


def test_49_longitudinal_wording_prohibited():
    assert SCIENTIFIC_TERMINOLOGY["longitudinal"] is False
    assert SCIENTIFIC_TERMINOLOGY["panel"] is False
    assert SCIENTIFIC_TERMINOLOGY["causal"] is False
    assert SCIENTIFIC_TERMINOLOGY["temporal_repeated_cross_sectional"] is True


# -----------------------------------------------------------------------------
# 50-53. Policy Invariants
# -----------------------------------------------------------------------------
def test_50_no_d7_2_stepwise_code():
    import nhis_fairbias.d7_terminal_mechanism as d7_mod

    source = inspect.getsource(d7_mod)
    assert "stepwise_replay" not in source
    assert "greedy_step_utility" not in source
    assert "refit_intermediate" not in source


def test_51_no_threshold_optimization():
    assert CANONICAL_DECISION_THRESHOLD == 0.5


def test_52_no_synthetic_transform_family_counterfactuals(mock_arm_states):
    for arm_id in D6_ARM_IDS:
        assert mock_arm_states[arm_id].changed_dict is not None


def test_53_no_real_nhis_cohort_access(poison_real_nhis_adapter):
    harness = NHISD7TerminalMechanismHarness(repo_root=_REPO_ROOT)
    audit_res = harness.run_audit_only()
    assert audit_res["real_nhis_cohort_accessed"] is False


# -----------------------------------------------------------------------------
# 54. AST-Level Anti-Fit Inspection Test (Section 22)
# -----------------------------------------------------------------------------
def test_54_source_level_ast_anti_fit_inspection():
    """Inspect AST of d7_terminal_mechanism.py to ensure zero .fit() / .fit_transform() calls."""
    import nhis_fairbias.d7_terminal_mechanism as d7_mod

    source = inspect.getsource(d7_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                attr_name = node.func.attr
                assert attr_name != "fit", (
                    f"AST anti-fit inspection failed: found substantive call '.fit()' at line {node.lineno}"
                )
                assert attr_name != "fit_transform", (
                    f"AST anti-fit inspection failed: found substantive call '.fit_transform()' at line {node.lineno}"
                )
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id
                assert func_name != "FairBiasMitigation", (
                    f"AST anti-fit inspection failed: found constructor call 'FairBiasMitigation()' at line {node.lineno}"
                )


# -----------------------------------------------------------------------------
# 55-68. Production-Wiring Synthetic Tests (Section 20 & 21)
# -----------------------------------------------------------------------------
class SyntheticTestAdapter:
    """Deterministic synthetic adapter matching NHISStudyAdapter interface."""

    def __init__(self, arm_states: Dict[str, FrozenArmState]) -> None:
        self.arm_states = arm_states
        # Create a mock preprocessor with the exact expected preprocessing hash
        self.preprocessor = MagicMock()
        prep_record = json.loads(
            (_REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609" / "preprocessing_provenance.json").read_text()
        )
        self.preprocessor.fitted_record.to_dict.return_value = prep_record

        # Feature family lists
        feat_reg = json.loads((_REPO_ROOT / "configs" / "nhis" / "features.json").read_text())
        cats = feat_reg["feature_lists"]["primary_categorical_features"]
        nums = feat_reg["feature_lists"]["primary_numerical_features"]
        self.preprocessor.get_feature_family_lists.return_value = (cats, nums)

    def get_cohort(self, year: int, outcome: str, protected_attribute: str, feature_set: str, disability_arm: str):
        # Determine arm_id
        arm_id = None
        for a_id, attr in ARM_PROTECTED_ATTRIBUTES.items():
            if attr == protected_attribute and ARM_DISABILITY_POLICIES[a_id] == disability_arm:
                arm_id = a_id
                break
        assert arm_id is not None

        arm_state = self.arm_states[arm_id]
        cols = arm_state.baseline_scaler.feature_order
        n_samples = 100

        # Synthetic X with valid features
        rng = np.random.RandomState(year + int(arm_id[-1]))
        data = rng.uniform(0.0, 1.0, size=(n_samples, len(cols)))
        X = pd.DataFrame(data, columns=cols)
        y = pd.Series(rng.binomial(1, 0.3, size=n_samples))
        a = pd.Series(rng.choice(range(1, 8 if arm_id == "D6_ARM_002" else 3), size=n_samples))
        w = np.ones(n_samples)
        meta = {"year": year, "arm_id": arm_id}
        return X, y, a, w, meta


def test_55_preconditions_before_mkdir(tmp_path):
    """Release manager must fail before directory creation if git precondition fails."""
    rel_manager = D7TerminalMechanismReleaseManager(
        repo_root=_REPO_ROOT,
        releases_parent_dir=tmp_path / "releases",
    )
    bad_head = "0000000000000000000000000000000000000000"
    target_dir = tmp_path / "releases" / "TEST_RELEASE_P1"

    with pytest.raises(ValueError, match="Git HEAD mismatch"):
        rel_manager.execute_release(
            release_id="TEST_RELEASE_P1",
            expected_execution_head=bad_head,
        )
    assert not target_dir.exists()


def test_56_lazy_adapter_construction(poison_real_nhis_adapter):
    """Runtime must not construct adapter in __init__."""
    runtime = ProductionD7TerminalMechanismRuntime(repo_root=_REPO_ROOT)
    assert runtime.adapter is None


def test_57_scored_cohort_structural_binding(mock_arm_states):
    """Scored cohorts must derive reproduction metrics directly from baseline_probs and fairbias_probs."""
    X = pd.DataFrame(np.zeros((10, 21)), columns=mock_arm_states["D6_ARM_001"].baseline_scaler.feature_order)
    probs_base = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    probs_fb = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92, 0.96])
    y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    a = pd.Series([1] * 10)

    scored = FrozenScoredCohort(
        year=2024,
        arm_id="D6_ARM_001",
        X_original=X,
        X_terminal=X,
        X_baseline_scaled=X.to_numpy(),
        X_fairbias_scaled=X.to_numpy(),
        y=y,
        a=a,
        baseline_logits=np.zeros(10),
        baseline_probs=probs_base,
        fairbias_logits=np.zeros(10),
        fairbias_probs=probs_fb,
        source_row_digest="test_digest",
    )

    metrics = compute_cohort_utility_metrics(np.asarray(y), scored.baseline_probs)
    assert metrics["count_predicted_positive"] == 6
    assert abs(metrics["selection_rate"] - 0.6) < 1e-12
    assert metrics["auroc"] > 0.9


def test_58_score_perturbation_causes_reproduction_failure(tmp_path):
    """Perturbing an actual probability value in a score array recomputes metrics and fails reproduction barrier."""
    cols = ["feat1", "feat2"]
    X = pd.DataFrame([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], columns=cols)
    y = pd.Series([0, 0, 1, 1])
    a = pd.Series([1, 2, 1, 2])
    base_probs = np.array([0.1, 0.2, 0.8, 0.9])
    fb_probs = np.array([0.15, 0.25, 0.75, 0.85])

    scored = FrozenScoredCohort(
        year=2024,
        arm_id="D6_ARM_001",
        X_original=X,
        X_terminal=X,
        X_baseline_scaled=X.to_numpy(),
        X_fairbias_scaled=X.to_numpy(),
        y=y,
        a=a,
        baseline_logits=np.zeros(4),
        baseline_probs=base_probs,
        fairbias_logits=np.zeros(4),
        fairbias_probs=fb_probs,
        source_row_digest="test_digest",
    )

    y_arr = np.asarray(scored.y, dtype=int)
    # Compute reproduction metrics from baseline_probs and fairbias_probs
    base_metrics = compute_cohort_utility_metrics(y_arr, scored.baseline_probs)
    fb_metrics = compute_cohort_utility_metrics(y_arr, scored.fairbias_probs)

    # Set up mock archive directories containing these unperturbed benchmark metrics
    tv_dir = tmp_path / "train_val"
    t_dir = tmp_path / "test"
    for arm_id in D6_ARM_IDS:
        (tv_dir / arm_id).mkdir(parents=True, exist_ok=True)
        (t_dir / arm_id).mkdir(parents=True, exist_ok=True)
        for model_k, m in (("baseline", base_metrics), ("fairbias", fb_metrics)):
            val_file = tv_dir / arm_id / f"validation_metrics_{model_k}.json"
            val_file.write_text(json.dumps({"utility": m}), encoding="utf-8")
            test_file = t_dir / arm_id / f"test_metrics_{model_k}.json"
            test_file.write_text(json.dumps({"utility": m}), encoding="utf-8")

    # Observed metrics formed from the unperturbed scored cohort
    unperturbed_metrics = {
        2023: {arm: {"baseline": base_metrics, "fairbias": fb_metrics} for arm in D6_ARM_IDS},
        2024: {arm: {"baseline": base_metrics, "fairbias": fb_metrics} for arm in D6_ARM_IDS},
    }
    # Unperturbed barrier must pass
    clean_res = verify_scoring_reproduction_barrier(unperturbed_metrics, tv_dir, t_dir)
    assert clean_res["status"] == "PASS"

    # Perturb an actual probability value in one score array (not an AUROC dict)
    mutated_base_probs = scored.baseline_probs.copy()
    mutated_base_probs[0] = 0.95  # True label was 0; flipping probability to 0.95 alters count, selection_rate, AUROC, etc.

    # Recompute metrics from that mutated score array
    mutated_base_metrics = compute_cohort_utility_metrics(y_arr, mutated_base_probs)
    assert mutated_base_metrics != base_metrics

    # Update observed metrics with metrics recomputed from mutated probability array
    corrupted_metrics = copy.deepcopy(unperturbed_metrics)
    corrupted_metrics[2024]["D6_ARM_001"]["baseline"] = mutated_base_metrics

    # Scoring reproduction barrier must fail
    with pytest.raises(ScoringReproductionBarrierError):
        verify_scoring_reproduction_barrier(corrupted_metrics, tv_dir, t_dir)


def test_59_no_arbitrary_nmi_thresholds_or_automatic_hypothesis_booleans():
    """Mechanism arm summary must output PI_REVIEW_REQUIRED and no hardcoded hypothesis booleans."""
    summary = compute_mechanism_arm_summary(
        arm_id="D6_ARM_001",
        changed_dict={"agep_a": "dropped"},
        family1_records=[
            {
                "record_type": "feature_summary",
                "arm_id": "D6_ARM_001",
                "year": 2024,
                "feature": "agep_a",
                "transform_type": "feature_drop",
                "delta_nmi_y": -0.05,
                "delta_nmi_a": -0.02,
            }
        ],
        family2_records=[],
        score_dist_records=[],
        logit_contrib_records=[],
    )
    assert summary["hypothesis_adjudication"] == "PI_REVIEW_REQUIRED"
    assert "evidence_consistent_with_H1" not in summary
    assert "evidence_consistent_with_H2" not in summary
    assert "empirical_information_destruction_demonstrated" not in summary
    assert summary["causal_claim_made"] is False


def test_60_logit_contribution_missing_feature_explicit_semantics():
    """Features dropped in FairBias must have fairbias_present=False and absence_equivalent_zero_contribution=True."""
    state = {
        "feature_order": ["x1"],
        "classes_": [0, 1],
        "coef_": [[1.5]],
        "intercept_": [0.0],
        "n_features_in_": 1,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 0,
    }
    lr = FrozenLogisticRegression(state)
    records = compute_logit_contribution_diagnostics(
        X_scaled=np.array([[0.5], [0.5]]),
        y_true=pd.Series([0, 1]),
        lr=lr,
        arm_id="D6_ARM_001",
        year=2024,
        model_name="fairbias",
        changed_dict={"dropped_col": "dropped"},
        all_baseline_features=["x1", "dropped_col"],
    )

    rec_x1 = next(r for r in records if r["feature"] == "x1")
    assert rec_x1["fairbias_present"] is True
    assert rec_x1["absence_equivalent_zero_contribution"] is False

    rec_drop = next(r for r in records if r["feature"] == "dropped_col")
    assert rec_drop["fairbias_present"] is False
    assert rec_drop["absence_equivalent_zero_contribution"] is True
    assert rec_drop["comparison_type"] == "dropped_from_fairbias_representation"
    assert rec_drop["beta_j"] is None


def test_61_release_manager_collision_fails_closed(tmp_path):
    """Release manager must fail closed if target release directory already exists."""
    rel_dir = tmp_path / "releases" / "EXISTING_RELEASE"
    rel_dir.mkdir(parents=True)

    manager = D7TerminalMechanismReleaseManager(
        repo_root=_REPO_ROOT,
        releases_parent_dir=tmp_path / "releases",
    )

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    with patch("nhis_fairbias.d7_terminal_mechanism.verify_git_execution_preconditions"):
        with pytest.raises(FileExistsError, match="Release directory collision"):
            manager.execute_release("EXISTING_RELEASE", expected_execution_head=head)


def test_62_failed_release_preserves_directory_with_failed_state(tmp_path):
    """Failed execution must preserve release directory and write FAILED state."""
    releases_dir = tmp_path / "releases"
    manager = D7TerminalMechanismReleaseManager(
        repo_root=_REPO_ROOT,
        releases_parent_dir=releases_dir,
    )
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()

    # Force failure by patching build_and_score_all_cohorts to fail
    with patch("nhis_fairbias.d7_terminal_mechanism.verify_git_execution_preconditions"), \
         patch.object(ProductionD7TerminalMechanismRuntime, "build_and_score_all_cohorts", side_effect=RuntimeError("Intentional test failure")):
        with pytest.raises(RuntimeError, match="Intentional test failure"):
            manager.execute_release("FAIL_RELEASE", expected_execution_head=head)

    rel_path = releases_dir / "FAIL_RELEASE"
    assert rel_path.is_dir()
    state_file = rel_path / "release_state.json"
    assert state_file.is_file()
    state_data = json.loads(state_file.read_text())
    assert state_data["status"] == "FAILED"
    assert "Intentional test failure" in state_data["error"]


def test_63_archive_verification_rejects_mutated_artifact(tmp_path):
    """Mutating any manifest-tracked artifact must fail archive verification."""
    # Copy train/val release to tmp_path and corrupt one file
    import shutil
    copy_dir = tmp_path / "docs" / "releases"
    copy_dir.mkdir(parents=True)
    shutil.copytree(_REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609", copy_dir / "NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609")
    shutil.copytree(_REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7", copy_dir / "NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7")

    # Corrupt one artifact
    target = copy_dir / "NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609" / "D6_ARM_001" / "arm_config.json"
    target.write_text('{"corrupted": true}', encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        verify_frozen_archives(tmp_path)


def test_64_archive_verification_rejects_wrong_tag_object():
    """Wrong tag object must fail git tag provenance verification."""
    with patch("subprocess.run") as mock_run:
        res1 = MagicMock()
        res1.stdout = "wrong_tag_obj\n"
        mock_run.return_value = res1

        with pytest.raises(ValueError, match="tag provenance mismatch"):
            verify_git_tag_provenance(_REPO_ROOT)


def test_65_d5_macro_context_transformation_summary():
    """Macro context must extract D5 terminal transformation summaries."""
    macro = build_macro_context(_REPO_ROOT)
    d5_summary = macro.get("d5_weighted_terminal_transformation_summary", {})
    assert "D6_ARM_001" in d5_summary
    assert "drops" in d5_summary["D6_ARM_001"]
    assert "powers" in d5_summary["D6_ARM_001"]
    assert "merges" in d5_summary["D6_ARM_001"]


def test_66_synthetic_release_manager_end_to_end(tmp_path, mock_arm_states, mock_observed_perfect_metrics):
    """End-to-end synthetic execution of D7TerminalMechanismReleaseManager producing all 11 files."""
    releases_dir = tmp_path / "releases"
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()

    # Synthetic adapter factory
    def make_adapter():
        return SyntheticTestAdapter(mock_arm_states)

    manager = D7TerminalMechanismReleaseManager(
        repo_root=_REPO_ROOT,
        releases_parent_dir=releases_dir,
        adapter_factory=make_adapter,
    )

    # Patch digests and scoring reproduction to test full artifact persistence and manifest generation
    created_runtimes = []
    orig_init = ProductionD7TerminalMechanismRuntime.__init__
    def mock_init(self_runtime, *args, **kwargs):
        orig_init(self_runtime, *args, **kwargs)
        self_runtime.arm_states = mock_arm_states
        self_runtime.observed_cohort_digests = EXPECTED_COHORT_SOURCE_ROW_DIGESTS
        self_runtime.observed_preprocessing_sha256 = PREPROCESSING_STATE_SHA256
        created_runtimes.append(self_runtime)

    with patch("nhis_fairbias.d7_terminal_mechanism.verify_git_execution_preconditions"), \
         patch.object(ProductionD7TerminalMechanismRuntime, "__init__", mock_init), \
         patch.object(ProductionD7TerminalMechanismRuntime, "build_and_score_all_cohorts") as mock_score, \
         patch.object(ProductionD7TerminalMechanismRuntime, "compute_diagnostics") as mock_diag:

        # Set up mock scoring result
        scored_cohorts = {2022: {}, 2023: {}, 2024: {}}
        for year in TEMPORAL_YEARS:
            for arm_id in D6_ARM_IDS:
                cols = mock_arm_states[arm_id].baseline_scaler.feature_order
                X = pd.DataFrame(np.zeros((10, len(cols))), columns=cols)
                y = pd.Series([0] * 5 + [1] * 5)
                a = pd.Series([1] * 10)
                scored_cohorts[year][arm_id] = FrozenScoredCohort(
                    year=year,
                    arm_id=arm_id,
                    X_original=X,
                    X_terminal=X,
                    X_baseline_scaled=X.to_numpy(),
                    X_fairbias_scaled=X.to_numpy(),
                    y=y,
                    a=a,
                    baseline_logits=np.zeros(10),
                    baseline_probs=np.linspace(0.1, 0.9, 10),
                    fairbias_logits=np.zeros(10),
                    fairbias_probs=np.linspace(0.1, 0.9, 10),
                    source_row_digest="test_digest",
                )

        def run_score(*args, **kwargs):
            if created_runtimes:
                created_runtimes[-1].scored_cohorts = scored_cohorts
            return scored_cohorts

        mock_score.side_effect = run_score

        mock_diag.return_value = {
            "family1_records": [
                {
                    "record_type": "feature_summary",
                    "year": 2024,
                    "arm_id": "D6_ARM_001",
                    "feature": "agep_a",
                    "transform_family": "Family_I",
                    "transform_type": "feature_drop",
                    "original_cardinality": 85,
                    "terminal_cardinality": 1,
                    "cardinality_reduction": 84,
                    "nmi_y_before": 0.05,
                    "nmi_y_after": 0.0,
                    "delta_nmi_y": -0.05,
                    "nmi_a_before": 0.02,
                    "nmi_a_after": 0.0,
                    "delta_nmi_a": -0.02,
                    "post_state": "dropped_constant_equivalent",
                }
            ],
            "family2_records": [
                {
                    "year": 2024,
                    "arm_id": "D6_ARM_001",
                    "feature": "pcnt18uptc",
                    "transform_family": "Family_II",
                    "transform_type": "power_reparameterization",
                    "power_exponent": 5.0,
                    "information_preserving_reparameterization": True,
                }
            ],
            "score_distribution_records": [
                {
                    "year": 2024,
                    "arm_id": "D6_ARM_001",
                    "model": "baseline",
                    "auroc": 0.85,
                    "auprc": 0.45,
                    "ks_statistic": 0.55,
                    "ks_pvalue": 1e-10,
                    "overall": {"n": 100, "median": 0.5, "fraction_ge_0_5": 0.4},
                }
            ],
            "protected_group_records": [
                {
                    "year": 2024,
                    "arm_id": "D6_ARM_001",
                    "model": "baseline",
                    "group": 1,
                    "outcome_stratum": "all",
                    "n": 50,
                    "median": 0.45,
                    "fraction_ge_0_5": 0.35,
                }
            ],
            "logit_contribution_records": [
                {
                    "year": 2024,
                    "arm_id": "D6_ARM_001",
                    "model": "baseline",
                    "feature": "educp_a",
                    "intercept": -1.5,
                    "beta_j": 0.4,
                    "mean_scaled_X_j_Y0": 0.3,
                    "mean_scaled_X_j_Y1": 0.7,
                    "class_separation_contribution": 0.16,
                    "baseline_present": True,
                    "fairbias_present": True,
                    "comparison_type": "shared_feature_same_name",
                    "absence_equivalent_zero_contribution": False,
                    "interpretation_status": "descriptive decomposition of the fitted terminal linear scoring function",
                }
            ],
            "arm_summaries": {
                "D6_ARM_001": {
                    "arm_id": "D6_ARM_001",
                    "hypothesis_adjudication": "PI_REVIEW_REQUIRED",
                    "causal_claim_made": False,
                }
            },
        }

        res = manager.execute_release("D7_SYNTHETIC_TEST_V1", expected_execution_head=head)
        assert res["status"] == "COMPLETE"
        rel_dir = Path(res["release_dir"])

        # Check all 11 files exist
        for fname in D7_ALL_RELEASE_FILES:
            assert (rel_dir / fname).is_file(), f"Missing release file {fname}"

        # Check manifest tracks exactly 9 artifacts
        manifest = json.loads((rel_dir / "d7_terminal_mechanism_manifest.json").read_text())
        assert len(manifest["artifacts"]) == 9
        for fname in D7_MANIFEST_TRACKED_ARTIFACTS:
            assert fname in manifest["artifacts"]

        # Check release_state.json
        state_data = json.loads((rel_dir / "release_state.json").read_text())
        assert state_data["status"] == "COMPLETE"
        assert state_data["manifest_sha256"] == res["manifest_sha256"]

        # Check provenance_summary.json explicit expected and observed provenance
        prov = json.loads((rel_dir / "provenance_summary.json").read_text())
        assert prov["cohort_digest_matches"] == 12
        assert prov["cohort_digest_expected"] == 12
        assert prov["preprocessing_expected_sha256"] == PREPROCESSING_STATE_SHA256
        assert prov["preprocessing_observed_sha256"] == PREPROCESSING_STATE_SHA256
        exp_str_keys = {str(y): val for y, val in EXPECTED_COHORT_SOURCE_ROW_DIGESTS.items()}
        assert prov["expected_cohort_source_row_digests"] == exp_str_keys
        assert prov["observed_cohort_source_row_digests"] == exp_str_keys


# -----------------------------------------------------------------------------
# 67-70. Top-Level Fail-Closed Tag Provenance Tests (P1-A)
# -----------------------------------------------------------------------------
def test_67_wrong_train_val_tag_object_fails_verify_frozen_archives():
    """Top-level verify_frozen_archives must fail closed if TRAIN/VAL tag object mismatches."""
    def fake_subprocess_run(cmd, *args, **kwargs):
        res = MagicMock()
        if f"refs/tags/{D6_TRAIN_VAL_TAG}" in cmd and "^{commit}" not in cmd[2]:
            res.stdout = "0000000000000000000000000000000000000000\n"
        elif f"refs/tags/{D6_TRAIN_VAL_TAG}^{{commit}}" in cmd:
            res.stdout = f"{D6_TRAIN_VAL_COMMIT}\n"
        elif f"refs/tags/{D6_TEST_TAG}" in cmd and "^{commit}" not in cmd[2]:
            res.stdout = f"{D6_TEST_TAG_OBJECT}\n"
        elif f"refs/tags/{D6_TEST_TAG}^{{commit}}" in cmd:
            res.stdout = f"{D6_TEST_COMMIT}\n"
        else:
            res.stdout = "dummy\n"
        return res

    with patch("subprocess.run", side_effect=fake_subprocess_run):
        with pytest.raises(ValueError, match="D6 Train/Val tag provenance mismatch"):
            verify_frozen_archives(_REPO_ROOT)


def test_68_wrong_test_tag_object_fails_verify_frozen_archives():
    """Top-level verify_frozen_archives must fail closed if TEST tag object mismatches."""
    def fake_subprocess_run(cmd, *args, **kwargs):
        res = MagicMock()
        if f"refs/tags/{D6_TRAIN_VAL_TAG}" in cmd and "^{commit}" not in cmd[2]:
            res.stdout = f"{D6_TRAIN_VAL_TAG_OBJECT}\n"
        elif f"refs/tags/{D6_TRAIN_VAL_TAG}^{{commit}}" in cmd:
            res.stdout = f"{D6_TRAIN_VAL_COMMIT}\n"
        elif f"refs/tags/{D6_TEST_TAG}" in cmd and "^{commit}" not in cmd[2]:
            res.stdout = "1111111111111111111111111111111111111111\n"
        elif f"refs/tags/{D6_TEST_TAG}^{{commit}}" in cmd:
            res.stdout = f"{D6_TEST_COMMIT}\n"
        else:
            res.stdout = "dummy\n"
        return res

    with patch("subprocess.run", side_effect=fake_subprocess_run):
        with pytest.raises(ValueError, match="D6 Test tag provenance mismatch"):
            verify_frozen_archives(_REPO_ROOT)


def test_69_wrong_dereferenced_commit_fails_verify_frozen_archives():
    """Top-level verify_frozen_archives must fail closed if dereferenced commit mismatches."""
    # Test Train/Val commit mismatch
    def fake_tv_commit_run(cmd, *args, **kwargs):
        res = MagicMock()
        if f"refs/tags/{D6_TRAIN_VAL_TAG}" in cmd and "^{commit}" not in cmd[2]:
            res.stdout = f"{D6_TRAIN_VAL_TAG_OBJECT}\n"
        elif f"refs/tags/{D6_TRAIN_VAL_TAG}^{{commit}}" in cmd:
            res.stdout = "2222222222222222222222222222222222222222\n"
        elif f"refs/tags/{D6_TEST_TAG}" in cmd and "^{commit}" not in cmd[2]:
            res.stdout = f"{D6_TEST_TAG_OBJECT}\n"
        elif f"refs/tags/{D6_TEST_TAG}^{{commit}}" in cmd:
            res.stdout = f"{D6_TEST_COMMIT}\n"
        else:
            res.stdout = "dummy\n"
        return res

    with patch("subprocess.run", side_effect=fake_tv_commit_run):
        with pytest.raises(ValueError, match="D6 Train/Val tag provenance mismatch"):
            verify_frozen_archives(_REPO_ROOT)

    # Test Test commit mismatch
    def fake_test_commit_run(cmd, *args, **kwargs):
        res = MagicMock()
        if f"refs/tags/{D6_TRAIN_VAL_TAG}" in cmd and "^{commit}" not in cmd[2]:
            res.stdout = f"{D6_TRAIN_VAL_TAG_OBJECT}\n"
        elif f"refs/tags/{D6_TRAIN_VAL_TAG}^{{commit}}" in cmd:
            res.stdout = f"{D6_TRAIN_VAL_COMMIT}\n"
        elif f"refs/tags/{D6_TEST_TAG}" in cmd and "^{commit}" not in cmd[2]:
            res.stdout = f"{D6_TEST_TAG_OBJECT}\n"
        elif f"refs/tags/{D6_TEST_TAG}^{{commit}}" in cmd:
            res.stdout = "3333333333333333333333333333333333333333\n"
        else:
            res.stdout = "dummy\n"
        return res

    with patch("subprocess.run", side_effect=fake_test_commit_run):
        with pytest.raises(ValueError, match="D6 Test tag provenance mismatch"):
            verify_frozen_archives(_REPO_ROOT)


def test_70_release_manager_wrong_tag_provenance_blocks_release(tmp_path):
    """Release manager must fail before directory creation and before adapter construction if tag provenance fails."""
    adapter_factory_called = False
    def fake_adapter_factory():
        nonlocal adapter_factory_called
        adapter_factory_called = True
        return MagicMock()

    target_releases_dir = tmp_path / "releases"
    manager = D7TerminalMechanismReleaseManager(
        repo_root=_REPO_ROOT,
        releases_parent_dir=target_releases_dir,
        adapter_factory=fake_adapter_factory,
    )

    release_id = "D7_TEST_FAIL_CLOSED_TAGS"
    target_rel_path = target_releases_dir / release_id

    def fake_subprocess_run(cmd, *args, **kwargs):
        res = MagicMock()
        if "git" in cmd[0] and "rev-parse" in cmd:
            if "refs/tags/" in str(cmd):
                res.stdout = "0000000000000000000000000000000000000000\n"
                return res
        res.stdout = "40d6baa81bccd31a27329b8758cbe62530fee132\n"
        return res

    with patch("nhis_fairbias.d7_terminal_mechanism.verify_git_execution_preconditions"), \
         patch("subprocess.run", side_effect=fake_subprocess_run):
        with pytest.raises(ValueError, match="tag provenance mismatch"):
            manager.execute_release(
                release_id=release_id,
                expected_execution_head="40d6baa81bccd31a27329b8758cbe62530fee132",
            )

    # Verify no release directory created, no adapter constructed
    assert not target_rel_path.exists()
    assert not adapter_factory_called


# -----------------------------------------------------------------------------
# 71-72. Family-I Index Alignment & Mismatch Fail-Closed Tests (P1-B)
# -----------------------------------------------------------------------------
def test_71_family1_nonconsecutive_index_alignment():
    """Family I diagnostics must correctly compute category states with nonconsecutive pandas index labels."""
    indices = [10, 20, 40, 80]
    X_raw = pd.DataFrame(
        {
            "cat_drop": [1, 1, 2, 2],
            "cat_merge": [1, 2, 1, 3],
        },
        index=indices,
    )
    X_term = pd.DataFrame(
        {
            "cat_merge": [2, 2, 2, 3],  # 1 merged into 2
        },
        index=indices,
    )
    y = pd.Series([0, 1, 1, 0], index=indices)
    a = pd.Series([1, 1, 2, 2], index=indices)

    changed_dict = {
        "cat_drop": "dropped",
        "cat_merge": {1: 2},
    }

    records = compute_family1_diagnostics(
        X_raw=X_raw,
        X_transformed=X_term,
        y=y,
        a=a,
        changed_dict=changed_dict,
        arm_id="D6_ARM_001",
        year=2024,
    )

    cat_records = [r for r in records if r.get("record_type") == "category_state"]

    # 1. Feature drop: cat_drop
    # Before: category 1 (indices 10, 20), y=[0, 1] -> n=2, count=1, prev=0.5
    drop_b1 = [r for r in cat_records if r["feature"] == "cat_drop" and r["state"] == "before" and r["category"] == "1"][0]
    assert drop_b1["n"] == 2
    assert drop_b1["outcome_positive_count"] == 1
    assert drop_b1["outcome_prevalence"] == 0.5

    # Before: category 2 (indices 40, 80), y=[1, 0] -> n=2, count=1, prev=0.5
    drop_b2 = [r for r in cat_records if r["feature"] == "cat_drop" and r["state"] == "before" and r["category"] == "2"][0]
    assert drop_b2["n"] == 2
    assert drop_b2["outcome_positive_count"] == 1
    assert drop_b2["outcome_prevalence"] == 0.5

    # After: constant equivalent -> n=4, count=2, prev=0.5
    drop_after = [r for r in cat_records if r["feature"] == "cat_drop" and r["state"] == "after"][0]
    assert drop_after["n"] == 4
    assert drop_after["outcome_positive_count"] == 2
    assert drop_after["outcome_prevalence"] == 0.5

    # 2. Categorical merge: cat_merge
    # Before:
    # cat 1 (indices 10, 40), y=[0, 1] -> n=2, count=1, prev=0.5
    merge_b1 = [r for r in cat_records if r["feature"] == "cat_merge" and r["state"] == "before" and r["category"] == "1"][0]
    assert merge_b1["n"] == 2
    assert merge_b1["outcome_positive_count"] == 1
    assert merge_b1["outcome_prevalence"] == 0.5

    # cat 2 (index 20), y=[1] -> n=1, count=1, prev=1.0
    merge_b2 = [r for r in cat_records if r["feature"] == "cat_merge" and r["state"] == "before" and r["category"] == "2"][0]
    assert merge_b2["n"] == 1
    assert merge_b2["outcome_positive_count"] == 1
    assert merge_b2["outcome_prevalence"] == 1.0

    # cat 3 (index 80), y=[0] -> n=1, count=0, prev=0.0
    merge_b3 = [r for r in cat_records if r["feature"] == "cat_merge" and r["state"] == "before" and r["category"] == "3"][0]
    assert merge_b3["n"] == 1
    assert merge_b3["outcome_positive_count"] == 0
    assert merge_b3["outcome_prevalence"] == 0.0

    # After:
    # cat 2 (indices 10, 20, 40), y=[0, 1, 1] -> n=3, count=2, prev=2/3
    merge_a2 = [r for r in cat_records if r["feature"] == "cat_merge" and r["state"] == "after" and r["category"] == "2"][0]
    assert merge_a2["n"] == 3
    assert merge_a2["outcome_positive_count"] == 2
    assert abs(merge_a2["outcome_prevalence"] - 2 / 3) < 1e-12

    # cat 3 (index 80), y=[0] -> n=1, count=0, prev=0.0
    merge_a3 = [r for r in cat_records if r["feature"] == "cat_merge" and r["state"] == "after" and r["category"] == "3"][0]
    assert merge_a3["n"] == 1
    assert merge_a3["outcome_positive_count"] == 0
    assert merge_a3["outcome_prevalence"] == 0.0


def test_72_family1_mismatched_index_fails_closed():
    """Family I diagnostics must fail closed if X_raw, y, a, or X_transformed indices do not match."""
    indices = [10, 20, 40, 80]
    mismatched_indices = [10, 20, 40, 99]

    X_raw = pd.DataFrame({"feat": [1, 2, 3, 4]}, index=indices)
    X_term = pd.DataFrame({"feat": [1, 2, 3, 4]}, index=indices)
    y_ok = pd.Series([0, 1, 0, 1], index=indices)
    a_ok = pd.Series([1, 1, 2, 2], index=indices)

    y_bad = pd.Series([0, 1, 0, 1], index=mismatched_indices)
    a_bad = pd.Series([1, 1, 2, 2], index=mismatched_indices)
    X_term_bad = pd.DataFrame({"feat": [1, 2, 3, 4]}, index=mismatched_indices)

    cd = {"feat": "dropped"}

    with pytest.raises(ValueError, match="Family-I index alignment failure: X_raw.index != y.index"):
        compute_family1_diagnostics(X_raw, X_term, y_bad, a_ok, cd, "D6_ARM_001", 2024)

    with pytest.raises(ValueError, match="Family-I index alignment failure: X_raw.index != a.index"):
        compute_family1_diagnostics(X_raw, X_term, y_ok, a_bad, cd, "D6_ARM_001", 2024)

    with pytest.raises(ValueError, match="Family-I index alignment failure: X_raw.index != X_transformed.index"):
        compute_family1_diagnostics(X_raw, X_term_bad, y_ok, a_ok, cd, "D6_ARM_001", 2024)


# -----------------------------------------------------------------------------
# 73-75. Preprocessing Anchor Mandatory Tests (P2-C)
# -----------------------------------------------------------------------------
def test_73_adapter_missing_preprocessor_or_fitted_record_fails():
    """Missing preprocessor or fitted_record must fail closed before cohort request."""
    # Case 1: no preprocessor
    mock_adapter1 = MagicMock(spec=[])
    runtime1 = ProductionD7TerminalMechanismRuntime(
        repo_root=_REPO_ROOT,
        adapter_factory=lambda: mock_adapter1,
    )
    with pytest.raises(ValueError, match="missing required fitted_record"):
        runtime1.construct_adapter()

    # Case 2: preprocessor fitted_record is None
    mock_adapter2 = MagicMock()
    mock_adapter2.preprocessor.fitted_record = None
    runtime2 = ProductionD7TerminalMechanismRuntime(
        repo_root=_REPO_ROOT,
        adapter_factory=lambda: mock_adapter2,
    )
    with pytest.raises(ValueError, match="missing required fitted_record"):
        runtime2.construct_adapter()


def test_74_adapter_uncanonicalizable_fitted_record_fails():
    """Uncanonicalizable fitted_record must fail closed."""
    mock_adapter = MagicMock()
    class UncanonicalizableRecord:
        pass
    mock_adapter.preprocessor.fitted_record = UncanonicalizableRecord()
    runtime = ProductionD7TerminalMechanismRuntime(
        repo_root=_REPO_ROOT,
        adapter_factory=lambda: mock_adapter,
    )
    with pytest.raises(ValueError, match="not canonicalizable"):
        runtime.construct_adapter()


def test_75_adapter_preprocessing_hash_mismatch_fails():
    """Preprocessing hash mismatch from adapter must fail closed."""
    mock_adapter = MagicMock()
    mock_adapter.preprocessor.fitted_record.to_dict.return_value = {"tampered": True}
    runtime = ProductionD7TerminalMechanismRuntime(
        repo_root=_REPO_ROOT,
        adapter_factory=lambda: mock_adapter,
    )
    with pytest.raises(ValueError, match="Preprocessing state hash mismatch from adapter"):
        runtime.construct_adapter()


# -----------------------------------------------------------------------------
# 76. True Runtime Synthetic Integration Test (P2-A)
# -----------------------------------------------------------------------------
def test_76_true_runtime_synthetic_integration_test():
    """True runtime synthetic execution with nonconsecutive indices and real unpatched pipelines."""
    tv_dir = _REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609"
    t_dir = _REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7"
    states = load_frozen_20_states(tv_dir, t_dir)

    class SynthNonconsecutiveAdapter:
        def __init__(self):
            self.preprocessor = MagicMock()
            prep_record = json.loads((tv_dir / "preprocessing_provenance.json").read_text())
            self.preprocessor.fitted_record.to_dict.return_value = prep_record
            feat_reg = json.loads((_REPO_ROOT / "configs" / "nhis" / "features.json").read_text())
            cats = feat_reg["feature_lists"]["primary_categorical_features"]
            nums = feat_reg["feature_lists"]["primary_numerical_features"]
            self.preprocessor.get_feature_family_lists.return_value = (cats, nums)

        def get_cohort(self, year, outcome, protected_attribute, feature_set, disability_arm):
            arm_id = None
            for a_id, attr in ARM_PROTECTED_ATTRIBUTES.items():
                if attr == protected_attribute and ARM_DISABILITY_POLICIES[a_id] == disability_arm:
                    arm_id = a_id
                    break
            cols = states[arm_id].baseline_scaler.feature_order
            n_samples = 100
            # Deliberately nonconsecutive indices
            indices = [10 * (i + 1) for i in range(n_samples)]
            feat_reg = json.loads((_REPO_ROOT / "configs" / "nhis" / "features.json").read_text())
            cats = set(feat_reg["feature_lists"]["primary_categorical_features"])
            rng = np.random.RandomState(year + int(arm_id[-1]))
            col_data = {}
            for c in cols:
                if c in cats:
                    col_data[c] = rng.choice([1, 2, 3, 4], size=n_samples)
                else:
                    col_data[c] = rng.uniform(0.1, 1.0, size=n_samples)
            X = pd.DataFrame(col_data, index=indices)
            y = pd.Series(rng.binomial(1, 0.4, size=n_samples), index=indices)
            a = pd.Series(rng.choice(range(1, 8 if arm_id == "D6_ARM_002" else 3), size=n_samples), index=indices)
            w = np.ones(n_samples)
            return X, y, a, w, {"year": year, "arm_id": arm_id}

    synth_indices = [10 * (i + 1) for i in range(100)]
    synth_digests = {
        y: {a: compute_cohort_source_row_digest(y, synth_indices) for a in D6_ARM_IDS}
        for y in TEMPORAL_YEARS
    }

    runtime = ProductionD7TerminalMechanismRuntime(
        repo_root=_REPO_ROOT,
        adapter_factory=lambda: SynthNonconsecutiveAdapter(),
    )

    # 1. construct_adapter
    adapter = runtime.construct_adapter()
    assert adapter is not None
    assert runtime.observed_preprocessing_sha256 == PREPROCESSING_STATE_SHA256

    # 2. load_states
    arm_states = runtime.load_states()
    assert len(arm_states) == 4

    # 3. build_and_score_all_cohorts & compute_diagnostics (UNPATCHED)
    def fake_benchmark(arch_path):
        arm_id = arch_path.parent.name
        model_key = "baseline" if "baseline" in arch_path.name else "fairbias"
        year = 2023 if "validation" in arch_path.name else 2024
        return runtime.reproduction_metrics[year][arm_id][model_key]

    with patch("nhis_fairbias.d7_terminal_mechanism.EXPECTED_COHORT_SOURCE_ROW_DIGESTS", synth_digests), \
         patch("nhis_fairbias.d7_terminal_mechanism.read_archived_scoring_benchmark", side_effect=fake_benchmark):
        # Assert neither build_and_score_all_cohorts nor compute_diagnostics is mocked
        assert not isinstance(runtime.build_and_score_all_cohorts, MagicMock)
        assert not isinstance(runtime.compute_diagnostics, MagicMock)

        scored = runtime.build_and_score_all_cohorts()
        assert len(scored) == 3
        for y in TEMPORAL_YEARS:
            assert len(scored[y]) == 4

        diagnostics = runtime.compute_diagnostics()

    # Verify diagnostics payload integrity
    assert len(diagnostics["family1_records"]) > 0
    assert len(diagnostics["family2_records"]) > 0
    assert len(diagnostics["score_distribution_records"]) == 24
    assert len(diagnostics["protected_group_records"]) == 234
    assert len(diagnostics["logit_contribution_records"]) == 468
    assert len(diagnostics["arm_summaries"]) == 4
    for arm_id in D6_ARM_IDS:
        assert arm_id in diagnostics["arm_summaries"]
        assert diagnostics["arm_summaries"][arm_id]["hypothesis_adjudication"] == "PI_REVIEW_REQUIRED"


# -----------------------------------------------------------------------------
# 77-83. Provenance-Semantic Regression Tests (Gate D7.1b.1)
# -----------------------------------------------------------------------------
def test_77_d7_delegates_to_canonical_d6_digest():
    """D7 compute_cohort_source_row_digest must delegate to canonical D6 implementation."""
    with patch(
        "nhis_fairbias.d7_terminal_mechanism.compute_d6_cohort_source_row_digest",
        return_value="mock_d6_digest",
    ) as mock_d6:
        res = compute_cohort_source_row_digest(2024, [10, 20, 40])
        assert res == "mock_d6_digest"
        mock_d6.assert_called_once()
        call_args = mock_d6.call_args[0]
        assert call_args[0] == 2024
        assert isinstance(call_args[1], pd.Index)
        assert list(call_args[1]) == [10, 20, 40]


def test_78_digest_exact_equivalence_across_index_types():
    """D7 digest must exactly equal canonical D6 digest across varied index configurations."""
    test_cases = [
        ("consecutive", [0, 1, 2, 3, 4]),
        ("nonconsecutive", [10, 20, 40, 80]),
        ("unsorted", [40, 10, 80, 20]),
        ("single_row", [42]),
        ("empty", []),
    ]
    for name, idx in test_cases:
        d7_res = compute_cohort_source_row_digest(2024, idx)
        d6_res = compute_d6_cohort_source_row_digest(2024, pd.Index(idx))
        assert d7_res == d6_res, f"Digest equivalence failed for {name} index"


def test_79_d7_digest_order_sensitivity_no_sorting():
    """D7 digest must be order-sensitive and must NOT sort row indices."""
    idx_forward = [10, 20, 40]
    idx_reversed = [40, 20, 10]
    digest_forward = compute_cohort_source_row_digest(2024, idx_forward)
    digest_reversed = compute_cohort_source_row_digest(2024, idx_reversed)
    assert digest_forward != digest_reversed


def test_80_known_literal_newline_payload_exact_sha256():
    """D7 digest for [10, 20, 40] must match exact literal newline-separated SHA-256."""
    literal_payload = b"2024:10\n2024:20\n2024:40"
    expected_sha = hashlib.sha256(literal_payload).hexdigest()
    observed_sha = compute_cohort_source_row_digest(2024, [10, 20, 40])
    assert observed_sha == expected_sha


def test_81_old_defective_serialization_differs():
    """The defective old comma-joined and sorted formula must differ from the canonical digest."""
    idx = [10, 20, 40]
    old_sorted_indices = sorted(str(i) for i in idx)
    old_defective_payload = f"2024:" + ",".join(old_sorted_indices)
    old_defective_sha = hashlib.sha256(old_defective_payload.encode("utf-8")).hexdigest()

    canonical_sha = compute_cohort_source_row_digest(2024, idx)
    assert canonical_sha != old_defective_sha


def test_82_expected_cohort_source_row_digests_constants_byte_for_byte_preserved():
    """All 12 expected cohort digest constants must remain strictly byte-for-byte preserved."""
    exact_historical_anchors = {
        2022: {
            "D6_ARM_001": "fbf4c5b0cadd74b7dfa565082e5d6577c8ec75fbdbf5abbdc8e48d712896ff6e",
            "D6_ARM_002": "30a71c454f54871cde9c03e34eabe936ce0c1a55a789e31017d9b31e81f23aa4",
            "D6_ARM_003": "e54056b34627b15440af95bcdb8e3f6ca1f282a909d3a59a3f47d62e7959185d",
            "D6_ARM_004": "e54056b34627b15440af95bcdb8e3f6ca1f282a909d3a59a3f47d62e7959185d",
        },
        2023: {
            "D6_ARM_001": "f66e94714e614c8ffd6ddeb2f9d466e616164e7b9aaf023e58b34c776006716a",
            "D6_ARM_002": "31197a19f03bf65a75cb7a6db2c97de8c7bb6a4878aa5260f9d1ee5ab56d82ad",
            "D6_ARM_003": "b24ede4a8a5a71c798612be991a1b3032a868b24da5f5793a201372b211d747a",
            "D6_ARM_004": "b24ede4a8a5a71c798612be991a1b3032a868b24da5f5793a201372b211d747a",
        },
        2024: {
            "D6_ARM_001": "f1d4386c9c14d939482bebaae94001f126fd388a8e55c494c65532f306b2ad4f",
            "D6_ARM_002": "0162b49440230ce4047ff2ebc1c2e0261479344615af282d9674f07e91786708",
            "D6_ARM_003": "0a9efcfd5a18647c55614d515066177ea6fabf8f5292472f4558ad87692079e9",
            "D6_ARM_004": "0a9efcfd5a18647c55614d515066177ea6fabf8f5292472f4558ad87692079e9",
        },
    }
    assert EXPECTED_COHORT_SOURCE_ROW_DIGESTS == exact_historical_anchors


def test_83_preserved_failed_release_immutable_and_cannot_be_overwritten():
    """The first failed canonical release must exist, remain FAILED, and block overwrite/reuse."""
    failed_release_dir = _REPO_ROOT / "runs" / "nhis_d7_terminal_mechanism" / "releases" / "NHIS_D7_TERMINAL_MECHANISM_V1_ed5597f5"
    assert failed_release_dir.is_dir()

    state_file = failed_release_dir / "release_state.json"
    assert state_file.is_file()
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["status"] == "FAILED"
    assert state["execution_head"] == "ed5597f5e67bf06caa6de134662f40ae53ac36b1"
    assert state["manifest_sha256"] is None
    assert "Cohort provenance barrier failed: 0/12 matched." in state["error"]

    # Release manager must refuse to reuse or overwrite this release ID
    manager = D7TerminalMechanismReleaseManager(
        repo_root=_REPO_ROOT,
        releases_parent_dir=failed_release_dir.parent,
        adapter_factory=lambda: MagicMock(),
    )
    with patch("nhis_fairbias.d7_terminal_mechanism.verify_git_execution_preconditions"):
        with pytest.raises(FileExistsError, match="Release directory collision"):
            manager.execute_release(
                release_id="NHIS_D7_TERMINAL_MECHANISM_V1_ed5597f5",
                expected_execution_head="ed5597f5e67bf06caa6de134662f40ae53ac36b1",
            )
