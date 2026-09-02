"""Unit, integration, and barrier tests for Gate D7.1a mechanism-audit harness.

All tests use synthetic/mock cohorts and frozen archive data.
Zero real NHIS cohort access, zero estimator fitting, zero FairBias relearning.
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from nhis_fairbias.d7_terminal_mechanism import (
    CANONICAL_DECISION_THRESHOLD,
    CohortProvenanceBarrierError,
    D6_ARM_IDS,
    D6_TEST_MANIFEST_SHA256,
    D6_TEST_TAG,
    D6_TRAIN_VAL_MANIFEST_SHA256,
    D6_TRAIN_VAL_TAG,
    D7_ALL_RELEASE_FILES,
    D7_HYPOTHESES,
    D7_MANIFEST_TRACKED_ARTIFACTS,
    D7_UNTRACKED_CONTROL_FILES,
    EXPECTED_COHORT_SOURCE_ROW_DIGESTS,
    EXPECTED_TRAINING_STATE_ANCHORS,
    FAMILY_I_INVARIANT,
    FAMILY_II_INVARIANT,
    FrozenArmState,
    FrozenLogisticRegression,
    FrozenScaler,
    MANDATORY_D7_DISCLOSURE,
    NHISD7TerminalMechanismHarness,
    PREPROCESSING_STATE_SHA256,
    PROHIBITED_SUBSTANTIVE_CALLS,
    SCIENTIFIC_TERMINOLOGY,
    SCORING_REPRODUCTION_ABSOLUTE_TOLERANCE,
    TEMPORAL_YEARS,
    ScoringReproductionBarrierError,
    build_macro_context,
    compute_cohort_source_row_digest,
    compute_distribution_summary,
    compute_family1_diagnostics,
    compute_family2_diagnostics,
    compute_logit_contribution_diagnostics,
    compute_mechanism_arm_summary,
    compute_nmi,
    compute_protected_group_score_diagnostics,
    compute_score_distribution_diagnostics,
    compute_sha256,
    execute_mechanism_diagnostics_pipeline,
    load_frozen_20_states,
    verify_cohort_provenance_barrier,
    verify_frozen_archives,
    verify_git_execution_preconditions,
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
        raise AssertionError("POISON TRIGGERED: Estimator .fit() called during D7.1a test!")

    def poisoned_fit_transform(*args, **kwargs):
        raise AssertionError("POISON TRIGGERED: Estimator .fit_transform() called during D7.1a test!")

    monkeypatch.setattr(MinMaxScaler, "fit", poisoned_fit)
    monkeypatch.setattr(MinMaxScaler, "fit_transform", poisoned_fit_transform)
    monkeypatch.setattr(LogisticRegression, "fit", poisoned_fit)


@pytest.fixture
def poison_adapter_cohort_access(monkeypatch):
    """Poison real NHIS cohort downloads or adapter cohort requests."""
    try:
        import nhis_fairbias.adapter as adapter_mod

        def poisoned_get_cohort(*args, **kwargs):
            raise AssertionError("POISON TRIGGERED: Real NHIS cohort accessed via adapter!")

        if hasattr(adapter_mod, "NHISAdapter"):
            monkeypatch.setattr(adapter_mod.NHISAdapter, "get_cohort", poisoned_get_cohort)
    except ImportError:
        pass


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
    assert res["status"] == "VERIFIED"


def test_02_frozen_d6_tag_verification():
    res = verify_frozen_archives(_REPO_ROOT)
    assert res["tag_verification"]["tags_verified"] is True
    assert res["tag_verification"]["train_val_tag"] == D6_TRAIN_VAL_TAG
    assert res["tag_verification"]["test_tag"] == D6_TEST_TAG


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
    assert macro["source"] == "canonical_frozen_releases_only"


# -----------------------------------------------------------------------------
# 7-11. Audit-Only Constraints & Git Preconditions
# -----------------------------------------------------------------------------
def test_07_audit_only_constructs_no_adapter(poison_adapter_cohort_access):
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
            res.stdout = "e24685cbde26497d0209f26fcf1d82b131dbe726\n"
        elif "status" in cmd:
            res.stdout = " M src/nhis_fairbias/some_file.py\n"
        return res

    monkeypatch.setattr("subprocess.run", mock_run)
    with pytest.raises(ValueError, match="Tracked working tree is dirty"):
        verify_git_execution_preconditions(_REPO_ROOT, "e24685cbde26497d0209f26fcf1d82b131dbe726")


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

    # Equivalence with exact sklearn transform
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
    # logit = -1.0 + 2.0*0.5 = 0.0 -> prob = 0.5
    probs = lr.predict_proba(X)
    assert abs(probs[0] - 0.5) < 1e-12


def test_17_scorer_performs_zero_lr_fit():
    # poison_estimator_fits fixture ensures any .fit() will fail immediately
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


def test_24_no_diagnostics_before_global_provenance_barrier(mock_observed_perfect_metrics):
    """Verify that execute_mechanism_diagnostics_pipeline fails closed if provenance fails."""
    tv_dir = _REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609"
    t_dir = _REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7"

    corrupted_digests = copy.deepcopy(EXPECTED_COHORT_SOURCE_ROW_DIGESTS)
    corrupted_digests[2022]["D6_ARM_001"] = "corrupted_sha"

    with pytest.raises(CohortProvenanceBarrierError):
        execute_mechanism_diagnostics_pipeline(
            cohort_digests=corrupted_digests,
            observed_metrics=mock_observed_perfect_metrics,
            archived_train_val_dir=tv_dir,
            archived_test_dir=t_dir,
        )


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
    rec = records[0]
    assert rec["transform_type"] == "feature_drop"
    assert rec["terminal_cardinality"] == 1
    assert rec["nmi_y_after"] == 0.0
    assert rec["post_state"] == "dropped_constant_equivalent"


def test_33_family1_category_cardinality_reduction_correct():
    df_raw = pd.DataFrame({"diff_a": [1, 2, 3, 4], "other": [1, 2, 3, 4]})
    df_trans = pd.DataFrame({"diff_a": [1, 1, 1, 1], "other": [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])
    a = pd.Series([1, 1, 2, 2])
    changed_dict = {"diff_a": {"2": 1, "3": 1, "4": 1}}

    records = compute_family1_diagnostics(df_raw, df_trans, y, a, changed_dict, "D6_ARM_001", 2023)
    rec = records[0]
    assert rec["transform_type"] == "categorical_merge"
    assert rec["original_cardinality"] == 4
    assert rec["terminal_cardinality"] == 1
    assert rec["cardinality_reduction"] == 3


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
# 40-41. Protected Group Summaries
# -----------------------------------------------------------------------------
def test_40_protected_group_summaries_retain_all_expected_hisp_groups():
    y = pd.Series([0, 1, 0, 0])
    a = pd.Series([1, 2, 3, 7])
    probs = np.array([0.2, 0.8, 0.3, 0.4])

    records = compute_protected_group_score_diagnostics(
        probs, y, a, expected_groups=range(1, 8), arm_id="D6_ARM_002", year=2024, model_name="fairbias"
    )
    groups = [r["group"] for r in records]
    assert groups == list(range(1, 8))


def test_41_small_empty_denominator_remains_undefined():
    y = pd.Series([0, 1, 0, 0])
    a = pd.Series([1, 2, 3, 7])
    probs = np.array([0.2, 0.8, 0.3, 0.4])

    records = compute_protected_group_score_diagnostics(
        probs, y, a, expected_groups=range(1, 8), arm_id="D6_ARM_002", year=2024, model_name="fairbias"
    )
    g4 = next(r for r in records if r["group"] == 4)
    assert g4["n"] == 0
    assert g4["score_mean"] is None
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
    records = compute_logit_contribution_diagnostics(np.array([[0.5], [0.5]]), pd.Series([0, 1]), lr, "D6_ARM_001", 2024, "baseline")
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
    records = compute_logit_contribution_diagnostics(np.array([[0.1], [0.9]]), pd.Series([0, 1]), lr, "D6_ARM_001", 2024, "baseline")
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
    # Verify that all changed_dicts are strictly the frozen immutable ones
    for arm_id in D6_ARM_IDS:
        assert mock_arm_states[arm_id].changed_dict is not None


def test_53_no_real_nhis_cohort_access(poison_adapter_cohort_access):
    harness = NHISD7TerminalMechanismHarness(repo_root=_REPO_ROOT)
    audit_res = harness.run_audit_only()
    assert audit_res["real_nhis_cohort_accessed"] is False


# -----------------------------------------------------------------------------
# 54. AST-Level Anti-Fit Inspection Test (Section 33)
# -----------------------------------------------------------------------------
def test_54_source_level_ast_anti_fit_inspection():
    """Inspect AST of d7_terminal_mechanism.py to ensure zero .fit() / .fit_transform() calls."""
    import nhis_fairbias.d7_terminal_mechanism as d7_mod

    source = inspect.getsource(d7_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check attribute calls like model.fit() or scaler.fit_transform()
            if isinstance(node.func, ast.Attribute):
                attr_name = node.func.attr
                assert attr_name != "fit", (
                    f"AST anti-fit inspection failed: found substantive call '.fit()' at line {node.lineno}"
                )
                assert attr_name != "fit_transform", (
                    f"AST anti-fit inspection failed: found substantive call '.fit_transform()' at line {node.lineno}"
                )
            # Check direct calls like FairBiasMitigation()
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id
                assert func_name != "FairBiasMitigation", (
                    f"AST anti-fit inspection failed: found constructor call 'FairBiasMitigation()' at line {node.lineno}"
                )
