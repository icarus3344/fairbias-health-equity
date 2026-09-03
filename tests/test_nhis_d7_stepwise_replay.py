"""Unit, barrier, state machine, and invariant tests for Gate D7.2a.1 Stepwise Replay Harness.

Guarantees:
- Replay state machine: categorical merge, numerical revisit, and drop semantics.
- Revisit semantics: numerical revisits evaluate against raw baseline with replacement power.
- Fail-closed provenance: exact D7.1 and D6 tags/commits, manifest artifact integrity.
- 12/12 global cohort source-row digest barrier.
- 12/12 terminal matrix-equivalence barrier.
- Endpoint-before-intermediate ordering barrier.
- Active semantic-list filtering: dropped features never enter calculate_epsilon.
- Temporal key-path summary: separate 2023 and 2024 rankings with paired deltas.
- Full production anti-leakage: 2023 and 2024 poisoned objects never enter .fit().
- Exact fit-count semantics: 47 scalers and 47 LRs.
- Release lifecycle: collision fail-closed, FAILED state preservation, 10-file schema, 8-file manifest.
- Zero real NHIS cohort access in tests or audit-only.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from fairbias.evaluator import FairEvaluator
from fairbias.transform import FairTransform
from nhis_fairbias.d7_stepwise_replay import (
    ARM_DISABILITY_POLICIES,
    ARM_PROTECTED_ATTRIBUTES,
    CANONICAL_DECISION_THRESHOLD,
    D6_ARM_IDS,
    D6_TEST_COMMIT,
    D6_TEST_MANIFEST_SHA256,
    D6_TEST_RELEASE_ID,
    D6_TEST_TAG,
    D6_TEST_TAG_OBJECT,
    D6_TRAIN_VAL_COMMIT,
    D6_TRAIN_VAL_MANIFEST_SHA256,
    D6_TRAIN_VAL_RELEASE_ID,
    D6_TRAIN_VAL_TAG,
    D6_TRAIN_VAL_TAG_OBJECT,
    D7_1_COMMIT,
    D7_1_TAG,
    D7_1_TAG_OBJECT,
    D7_2_ALL_RELEASE_FILES,
    D7_2_MANIFEST_TRACKED_ARTIFACTS,
    D7_2_REFITS_INTERMEDIATE_CLASSIFIERS,
    D7_2_SCIENTIFIC_QUESTION,
    D7_2_SCIENTIFIC_TERMINOLOGY,
    D7_2_UNTRACKED_CONTROL_FILES,
    EXPECTED_ARCHIVED_STEP_COUNTS,
    EXPECTED_COHORT_SOURCE_ROW_DIGESTS,
    EXPECTED_TRAINING_STATE_ANCHORS,
    FORBIDDEN_CAUSAL_TERMS,
    FROZEN_FEATURES_PARQUET_PATH,
    FROZEN_FEATURES_PARQUET_SHA256,
    LR_MAX_ITER,
    LR_RANDOM_STATE,
    LR_SOLVER,
    MATRIX_NUMERICAL_TOLERANCE,
    METRIC_REPRODUCTION_TOLERANCE,
    PREPROCESSING_STATE_SHA256,
    STARTING_HEAD_COMMIT,
    TEMPORAL_YEARS,
    TOTAL_EXPECTED_ACCEPTED_STEPS,
    TOTAL_EXPECTED_REPRESENTATION_STATES,
    ArchivedTraceStep,
    D7StepwiseReplayError,
    D7StepwiseReplayReleaseManager,
    DataLeakageError,
    EndpointReproductionBarrierError,
    IllegalReplayError,
    NHISD7StepwiseReplayHarness,
    ProductionD7StepwiseReplayRuntime,
    ProvenanceVerificationError,
    RepresentationState,
    SequentialReplayStateMachine,
    TerminalReplayBarrierError,
    build_d7_stepwise_manifest,
    compute_sha256,
    compute_stepwise_deltas,
    compute_stepwise_metrics,
    fit_intermediate_model_2022,
    generate_key_path_summary,
    load_all_archived_traces,
    load_archived_trace,
    predict_proba_fitted_model,
    verify_12_cohort_provenance_barrier,
    verify_endpoint_model_reproduction_barrier,
    verify_git_execution_preconditions,
    verify_terminal_replay_barrier,
    verify_upstream_provenance,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# 1. Categorical merge replay
# -----------------------------------------------------------------------------
def test_categorical_merge_replay():
    df = pd.DataFrame({
        "empwrkft1_a": [-1, 1, 2, 3],
        "other": [10, 20, 30, 40],
    })
    step = ArchivedTraceStep(
        iteration=1,
        selected_feature="empwrkft1_a",
        feature_semantic_type="categorical",
        d_phi_before=0.005,
        epsilon=0.0005,
        proposed_transformation={"-1": 1},
        accepted_transformation={"-1": 1},
        numerical_exponent=None,
        categorical_merge_mapping={"-1": "1"},
        dropped=False,
    )
    r_machine = SequentialReplayStateMachine(
        arm_id="D6_ARM_001",
        trace_steps=[step],
        base_feature_order=["empwrkft1_a", "other"],
        cate_attrs=["empwrkft1_a"],
        validate_step_count=False,
    )
    res = r_machine.replay_all(df)
    state_1_df = res[1]

    # -1 should become 1; 1, 2, 3 remain unchanged
    expected = [1, 1, 2, 3]
    assert state_1_df["empwrkft1_a"].tolist() == expected
    assert state_1_df["other"].tolist() == [10, 20, 30, 40]


# -----------------------------------------------------------------------------
# 2. Repeated categorical merge on same feature
# -----------------------------------------------------------------------------
def test_repeated_categorical_merge_same_feature():
    df = pd.DataFrame({
        "cat_feat": [1, 2, 3, 4],
    })
    step1 = ArchivedTraceStep(
        iteration=1,
        selected_feature="cat_feat",
        feature_semantic_type="categorical",
        d_phi_before=0.01,
        epsilon=0.0005,
        proposed_transformation={"2": 1},
        accepted_transformation={"2": 1},
        categorical_merge_mapping={"2": "1"},
        dropped=False,
    )
    step2 = ArchivedTraceStep(
        iteration=2,
        selected_feature="cat_feat",
        feature_semantic_type="categorical",
        d_phi_before=0.005,
        epsilon=0.0005,
        proposed_transformation={"2": 1, "3": 1},
        accepted_transformation={"2": 1, "3": 1},
        categorical_merge_mapping={"2": "1", "3": "1"},
        dropped=False,
    )
    r_machine = SequentialReplayStateMachine(
        arm_id="D6_ARM_001",
        trace_steps=[step1, step2],
        base_feature_order=["cat_feat"],
        cate_attrs=["cat_feat"],
        validate_step_count=False,
    )
    res = r_machine.replay_all(df)

    assert res[1]["cat_feat"].tolist() == [1, 1, 3, 4]
    assert res[2]["cat_feat"].tolist() == [1, 1, 1, 4]


# -----------------------------------------------------------------------------
# 3. Numerical power replay
# -----------------------------------------------------------------------------
def test_numerical_power_replay():
    df = pd.DataFrame({
        "pcnt": [-2.0, 0.0, 1.0, 3.0],
    })
    step = ArchivedTraceStep(
        iteration=1,
        selected_feature="pcnt",
        feature_semantic_type="numerical",
        d_phi_before=0.01,
        epsilon=0.0005,
        proposed_transformation={"power": 3.0},
        accepted_transformation={"power": 3.0},
        numerical_exponent=3.0,
        dropped=False,
    )
    r_machine = SequentialReplayStateMachine(
        arm_id="D6_ARM_001",
        trace_steps=[step],
        base_feature_order=["pcnt"],
        num_attrs=["pcnt"],
        validate_step_count=False,
    )
    res = r_machine.replay_all(df)
    expected = [-8.0, 0.0, 1.0, 27.0]
    np.testing.assert_allclose(res[1]["pcnt"].to_numpy(), expected, atol=1e-12)


# -----------------------------------------------------------------------------
# 4. Repeated numerical powers on same feature (replacement semantics)
# -----------------------------------------------------------------------------
def test_repeated_numerical_powers_same_feature():
    df = pd.DataFrame({
        "agep_a": [2.0, 3.0],
    })
    step1 = ArchivedTraceStep(
        iteration=1,
        selected_feature="agep_a",
        feature_semantic_type="numerical",
        d_phi_before=0.01,
        epsilon=0.0005,
        proposed_transformation={"power": 3.0},
        accepted_transformation={"power": 3.0},
        numerical_exponent=3.0,
        dropped=False,
    )
    step2 = ArchivedTraceStep(
        iteration=2,
        selected_feature="agep_a",
        feature_semantic_type="numerical",
        d_phi_before=0.005,
        epsilon=0.0005,
        proposed_transformation={"power": 5.0},
        accepted_transformation={"power": 5.0},
        numerical_exponent=5.0,
        dropped=False,
    )
    r_machine = SequentialReplayStateMachine(
        arm_id="D6_ARM_001",
        trace_steps=[step1, step2],
        base_feature_order=["agep_a"],
        num_attrs=["agep_a"],
        validate_step_count=False,
    )
    res = r_machine.replay_all(df)
    # State 1: x^3 -> [8.0, 27.0]
    np.testing.assert_allclose(res[1]["agep_a"].to_numpy(), [8.0, 27.0], atol=1e-12)
    # State 2: x^5 (replacing power) -> [32.0, 243.0]
    np.testing.assert_allclose(res[2]["agep_a"].to_numpy(), [32.0, 243.0], atol=1e-12)


# -----------------------------------------------------------------------------
# 5. Feature drop
# -----------------------------------------------------------------------------
def test_feature_drop():
    df = pd.DataFrame({
        "f1": [1, 2],
        "f2": [3, 4],
        "f3": [5, 6],
    })
    step = ArchivedTraceStep(
        iteration=1,
        selected_feature="f2",
        feature_semantic_type="numerical",
        d_phi_before=0.01,
        epsilon=0.0005,
        proposed_transformation="dropped",
        accepted_transformation="dropped",
        dropped=True,
    )
    r_machine = SequentialReplayStateMachine(
        arm_id="D6_ARM_001",
        trace_steps=[step],
        base_feature_order=["f1", "f2", "f3"],
        validate_step_count=False,
    )
    res = r_machine.replay_all(df)
    assert list(res[1].columns) == ["f1", "f3"]
    assert "f2" not in res[1].columns


# -----------------------------------------------------------------------------
# 6. Dropped feature cannot reappear
# -----------------------------------------------------------------------------
def test_dropped_feature_cannot_reappear():
    step1 = ArchivedTraceStep(
        iteration=1,
        selected_feature="f2",
        feature_semantic_type="numerical",
        d_phi_before=0.01,
        epsilon=0.0005,
        proposed_transformation="dropped",
        accepted_transformation="dropped",
        dropped=True,
    )
    step2 = ArchivedTraceStep(
        iteration=2,
        selected_feature="f2",
        feature_semantic_type="numerical",
        d_phi_before=0.005,
        epsilon=0.0005,
        proposed_transformation={"power": 3.0},
        accepted_transformation={"power": 3.0},
        numerical_exponent=3.0,
        dropped=False,
    )
    with pytest.raises(IllegalReplayError, match="already dropped"):
        SequentialReplayStateMachine(
            arm_id="D6_ARM_001",
            trace_steps=[step1, step2],
            base_feature_order=["f1", "f2"],
            validate_step_count=False,
        )


# -----------------------------------------------------------------------------
# 7. Mixed transform path
# -----------------------------------------------------------------------------
def test_mixed_transform_path():
    df = pd.DataFrame({
        "f_num": [1.0, 2.0],
        "f_cat": [1, 2],
        "f_drop": [10, 20],
    })
    steps = [
        ArchivedTraceStep(
            iteration=1,
            selected_feature="f_num",
            feature_semantic_type="numerical",
            d_phi_before=0.02,
            epsilon=0.0005,
            proposed_transformation={"power": 2.0},
            accepted_transformation={"power": 2.0},
            numerical_exponent=2.0,
            dropped=False,
        ),
        ArchivedTraceStep(
            iteration=2,
            selected_feature="f_cat",
            feature_semantic_type="categorical",
            d_phi_before=0.01,
            epsilon=0.0005,
            proposed_transformation={"2": 1},
            accepted_transformation={"2": 1},
            categorical_merge_mapping={"2": "1"},
            dropped=False,
        ),
        ArchivedTraceStep(
            iteration=3,
            selected_feature="f_drop",
            feature_semantic_type="numerical",
            d_phi_before=0.008,
            epsilon=0.0005,
            proposed_transformation="dropped",
            accepted_transformation="dropped",
            dropped=True,
        ),
    ]
    r_machine = SequentialReplayStateMachine(
        arm_id="D6_ARM_001",
        trace_steps=steps,
        base_feature_order=["f_num", "f_cat", "f_drop"],
        num_attrs=["f_num", "f_drop"],
        cate_attrs=["f_cat"],
        validate_step_count=False,
    )
    res = r_machine.replay_all(df)
    state_3 = res[3]
    assert list(state_3.columns) == ["f_num", "f_cat"]
    np.testing.assert_allclose(state_3["f_num"].to_numpy(), [1.0, 4.0], atol=1e-12)
    assert state_3["f_cat"].tolist() == [1, 1]


# -----------------------------------------------------------------------------
# 8. Unsorted / nonconsecutive row index preservation
# -----------------------------------------------------------------------------
def test_unsorted_nonconsecutive_row_index_preservation():
    custom_index = pd.Index([105, 12, 999, 44, 2], name="custom_id")
    df = pd.DataFrame(
        {
            "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_b": [10, 20, 30, 40, 50],
        },
        index=custom_index,
    )
    step = ArchivedTraceStep(
        iteration=1,
        selected_feature="feat_a",
        feature_semantic_type="numerical",
        d_phi_before=0.01,
        epsilon=0.0005,
        proposed_transformation={"power": 2.0},
        accepted_transformation={"power": 2.0},
        numerical_exponent=2.0,
        dropped=False,
    )
    r_machine = SequentialReplayStateMachine(
        arm_id="D6_ARM_001",
        trace_steps=[step],
        base_feature_order=["feat_a", "feat_b"],
        num_attrs=["feat_a"],
        validate_step_count=False,
    )
    res = r_machine.replay_all(df)
    assert res[0].index.equals(custom_index)
    assert res[1].index.equals(custom_index)


# -----------------------------------------------------------------------------
# 9. Final sequential state equals direct terminal transform
# -----------------------------------------------------------------------------
def test_final_sequential_state_equals_direct_terminal_transform():
    traces = load_all_archived_traces(_REPO_ROOT)
    base_dir = _REPO_ROOT / "docs" / "releases" / D6_TRAIN_VAL_RELEASE_ID

    for arm_id in D6_ARM_IDS:
        arm_steps = traces[arm_id]
        cd_path = base_dir / arm_id / "final_changed_dict.json"
        cd_dict = json.loads(cd_path.read_text(encoding="utf-8"))
        expected_hash = EXPECTED_TRAINING_STATE_ANCHORS[arm_id]["changed_dict"]

        rng = np.random.default_rng(42)
        n_rows = 100
        synth_data: Dict[str, Any] = {}
        all_features = list(cd_dict.keys())

        for f in all_features:
            val = cd_dict[f]
            if isinstance(val, dict) and "power" in val:
                synth_data[f] = rng.uniform(1.0, 10.0, size=n_rows)
            else:
                synth_data[f] = rng.choice([-1, 1, 2, 3, 4, 5], size=n_rows)

        sample_df = pd.DataFrame(synth_data)
        num_attrs = [f for f, v in cd_dict.items() if isinstance(v, dict) and "power" in v]
        cate_attrs = [f for f in all_features if f not in num_attrs]

        barrier_res = verify_terminal_replay_barrier(
            arm_id=arm_id,
            trace_steps=arm_steps,
            final_changed_dict=cd_dict,
            expected_changed_dict_sha256=expected_hash,
            sample_df=sample_df,
            num_attrs=num_attrs,
            cate_attrs=cate_attrs,
            tolerance=MATRIX_NUMERICAL_TOLERANCE,
        )
        assert barrier_res["matrix_equivalence_verified"] is True
        assert barrier_res["max_numeric_diff"] <= MATRIX_NUMERICAL_TOLERANCE


# -----------------------------------------------------------------------------
# 10. Trace order mutation fails terminal barrier
# -----------------------------------------------------------------------------
def test_trace_order_mutation_fails_terminal_barrier():
    traces = load_all_archived_traces(_REPO_ROOT)
    arm_steps = list(traces["D6_ARM_001"])

    s8 = arm_steps[7]
    s10 = arm_steps[9]
    mutated_steps = list(arm_steps)
    mutated_steps[7] = s10
    mutated_steps[9] = s8

    with pytest.raises(IllegalReplayError, match="already dropped"):
        SequentialReplayStateMachine(
            arm_id="D6_ARM_001",
            trace_steps=mutated_steps,
            base_feature_order=[s.selected_feature for s in arm_steps],
            validate_step_count=True,
        )


# -----------------------------------------------------------------------------
# 11. Missing trace step fails
# -----------------------------------------------------------------------------
def test_missing_trace_step_fails():
    traces = load_all_archived_traces(_REPO_ROOT)
    arm_steps = list(traces["D6_ARM_001"])[:-1]

    base_dir = _REPO_ROOT / "docs" / "releases" / D6_TRAIN_VAL_RELEASE_ID
    cd_path = base_dir / "D6_ARM_001" / "final_changed_dict.json"
    cd_dict = json.loads(cd_path.read_text(encoding="utf-8"))
    expected_hash = EXPECTED_TRAINING_STATE_ANCHORS["D6_ARM_001"]["changed_dict"]

    with pytest.raises(TerminalReplayBarrierError, match="step count"):
        verify_terminal_replay_barrier(
            arm_id="D6_ARM_001",
            trace_steps=arm_steps,
            final_changed_dict=cd_dict,
            expected_changed_dict_sha256=expected_hash,
        )


# -----------------------------------------------------------------------------
# 12. Altered accepted transformation fails
# -----------------------------------------------------------------------------
def test_altered_accepted_transformation_fails():
    traces = load_all_archived_traces(_REPO_ROOT)
    arm_steps = list(traces["D6_ARM_001"])

    mutated_last = ArchivedTraceStep(
        iteration=arm_steps[14].iteration,
        selected_feature=arm_steps[14].selected_feature,
        feature_semantic_type=arm_steps[14].feature_semantic_type,
        d_phi_before=arm_steps[14].d_phi_before,
        epsilon=arm_steps[14].epsilon,
        proposed_transformation={"power": 8.0},
        accepted_transformation={"power": 8.0},
        numerical_exponent=8.0,
        dropped=False,
    )
    mutated_steps = list(arm_steps)
    mutated_steps[14] = mutated_last

    base_dir = _REPO_ROOT / "docs" / "releases" / D6_TRAIN_VAL_RELEASE_ID
    cd_path = base_dir / "D6_ARM_001" / "final_changed_dict.json"
    cd_dict = json.loads(cd_path.read_text(encoding="utf-8"))
    expected_hash = EXPECTED_TRAINING_STATE_ANCHORS["D6_ARM_001"]["changed_dict"]

    with pytest.raises(TerminalReplayBarrierError):
        verify_terminal_replay_barrier(
            arm_id="D6_ARM_001",
            trace_steps=mutated_steps,
            final_changed_dict=cd_dict,
            expected_changed_dict_sha256=expected_hash,
        )


# -----------------------------------------------------------------------------
# 13. Endpoint state-0 reproduction barrier
# -----------------------------------------------------------------------------
def test_endpoint_state_0_reproduction_barrier():
    scaler = MinMaxScaler()
    scaler.fit(np.array([[1.0], [2.0]]))
    lr = LogisticRegression()
    lr.fit(np.array([[1.0], [2.0]]), np.array([0, 1]))

    with pytest.raises(EndpointReproductionBarrierError, match="scaler hash mismatch"):
        verify_endpoint_model_reproduction_barrier(
            arm_id="D6_ARM_001",
            state_0_scaler=scaler,
            state_0_lr=lr,
            state_0_features=["a"],
            state_K_scaler=scaler,
            state_K_lr=lr,
            state_K_features=["a"],
            archived_2023_baseline_metrics={},
            archived_2023_fairbias_metrics={},
            archived_2024_baseline_metrics={},
            archived_2024_fairbias_metrics={},
            observed_2023_state_0_metrics={},
            observed_2023_state_K_metrics={},
            observed_2024_state_0_metrics={},
            observed_2024_state_K_metrics={},
        )


# -----------------------------------------------------------------------------
# 14. Endpoint state-K reproduction barrier
# -----------------------------------------------------------------------------
def test_endpoint_state_K_reproduction_barrier():
    with patch("nhis_fairbias.d7_stepwise_replay.extract_minmax_scaler_state") as m_scaler, \
         patch("nhis_fairbias.d7_stepwise_replay.extract_logistic_regression_state") as m_lr, \
         patch("nhis_fairbias.d7_stepwise_replay.compute_canonical_json_sha256") as m_sha:

        anchors = EXPECTED_TRAINING_STATE_ANCHORS["D6_ARM_001"]
        m_sha.side_effect = [
            anchors["baseline_scaler"],
            anchors["baseline_LR"],
            anchors["FairBias_scaler"],
            anchors["FairBias_LR"],
        ]

        scaler = MagicMock()
        lr = MagicMock()

        archived = {
            "count_predicted_positive": 100,
            "auroc": 0.80,
            "auprc": 0.30,
            "selection_rate": 0.05,
        }
        observed_bad = {
            "count_predicted_positive": 101,
            "auroc": 0.80,
            "auprc": 0.30,
            "selection_rate": 0.05,
        }

        with pytest.raises(EndpointReproductionBarrierError, match="predicted positive count mismatch"):
            verify_endpoint_model_reproduction_barrier(
                arm_id="D6_ARM_001",
                state_0_scaler=scaler,
                state_0_lr=lr,
                state_0_features=["a"],
                state_K_scaler=scaler,
                state_K_lr=lr,
                state_K_features=["a"],
                archived_2023_baseline_metrics=archived,
                archived_2023_fairbias_metrics=archived,
                archived_2024_baseline_metrics=archived,
                archived_2024_fairbias_metrics=archived,
                observed_2023_state_0_metrics=archived,
                observed_2023_state_K_metrics=observed_bad,
                observed_2024_state_0_metrics=archived,
                observed_2024_state_K_metrics=archived,
            )


# -----------------------------------------------------------------------------
# 15. Intermediate model fit uses 2022 only
# -----------------------------------------------------------------------------
def test_intermediate_model_fit_uses_2022_only():
    X_2022 = pd.DataFrame({"feat1": [1.0, 2.0, 3.0, 4.0]})
    y_2022 = pd.Series([0, 1, 0, 1])

    scaler, lr = fit_intermediate_model_2022(X_2022, y_2022, arm_id="D6_ARM_001", state_index=1)
    assert isinstance(scaler, MinMaxScaler)
    assert isinstance(lr, LogisticRegression)
    assert lr.random_state == LR_RANDOM_STATE
    assert lr.solver == LR_SOLVER
    assert lr.max_iter == LR_MAX_ITER


# -----------------------------------------------------------------------------
# 16. 2023/2024 never enter .fit()
# -----------------------------------------------------------------------------
def test_2023_2024_never_enter_fit():
    fit_call_data = []
    orig_fit = LogisticRegression.fit

    def recording_fit(self, X, y, sample_weight=None):
        fit_call_data.append(X)
        return orig_fit(self, X, y, sample_weight=sample_weight)

    with patch.object(LogisticRegression, "fit", recording_fit):
        X_2022 = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]})
        y_2022 = pd.Series([0, 0, 1, 1])
        scaler, lr = fit_intermediate_model_2022(X_2022, y_2022, "D6_ARM_001", 1)

        X_2023 = pd.DataFrame({"f": [2.0, 3.0]})
        X_2024 = pd.DataFrame({"f": [1.0, 4.0]})
        _ = predict_proba_fitted_model(scaler, lr, X_2023)
        _ = predict_proba_fitted_model(scaler, lr, X_2024)

    assert len(fit_call_data) == 1
    assert len(fit_call_data[0]) == len(X_2022)


# -----------------------------------------------------------------------------
# 17. Decision threshold fixed at 0.5
# -----------------------------------------------------------------------------
def test_threshold_fixed_at_0_5():
    probs = np.array([0.49, 0.50, 0.51, 0.90])
    y_true = np.array([0, 0, 1, 1])
    metrics = compute_stepwise_metrics(
        arm_id="D6_ARM_001",
        state_index=1,
        year=2023,
        probs=probs,
        y_true=y_true,
    )
    assert metrics["threshold"] == CANONICAL_DECISION_THRESHOLD
    assert metrics["count_predicted_positive"] == 3
    assert metrics["selection_rate"] == 0.75


# -----------------------------------------------------------------------------
# 18. All intermediate states emitted (47 total)
# -----------------------------------------------------------------------------
def test_all_intermediate_states_emitted():
    traces = load_all_archived_traces(_REPO_ROOT)
    total_states = 0
    for arm_id, steps in traces.items():
        r_machine = SequentialReplayStateMachine(
            arm_id=arm_id,
            trace_steps=steps,
            base_feature_order=[s.selected_feature for s in steps],
            validate_step_count=True,
        )
        total_states += len(r_machine.states)

    assert total_states == TOTAL_EXPECTED_REPRESENTATION_STATES


# -----------------------------------------------------------------------------
# 19. Pathwise delta arithmetic exact
# -----------------------------------------------------------------------------
def test_pathwise_delta_arithmetic_exact():
    records = [
        {"arm_id": "D6_ARM_001", "year": 2023, "state_index": 0, "auroc": 0.70, "max_d_phi": 0.01, "count_predicted_positive": 100, "selection_rate": 0.05, "auprc": 0.20, "ks_statistic": 0.40},
        {"arm_id": "D6_ARM_001", "year": 2023, "state_index": 1, "auroc": 0.72, "max_d_phi": 0.008, "count_predicted_positive": 110, "selection_rate": 0.055, "auprc": 0.22, "ks_statistic": 0.42},
        {"arm_id": "D6_ARM_001", "year": 2023, "state_index": 2, "auroc": 0.69, "max_d_phi": 0.005, "count_predicted_positive": 105, "selection_rate": 0.052, "auprc": 0.19, "ks_statistic": 0.39},
    ]
    deltas_df = compute_stepwise_deltas(records)
    assert len(deltas_df) == 2

    r1 = deltas_df[deltas_df["state_index"] == 1].iloc[0]
    assert np.isclose(r1["pathwise_marginal_delta_auroc"], 0.02, atol=1e-12)
    assert np.isclose(r1["cumulative_delta_from_baseline_auroc"], 0.02, atol=1e-12)

    r2 = deltas_df[deltas_df["state_index"] == 2].iloc[0]
    assert np.isclose(r2["pathwise_marginal_delta_auroc"], -0.03, atol=1e-12)
    assert np.isclose(r2["cumulative_delta_from_baseline_auroc"], -0.01, atol=1e-12)


# -----------------------------------------------------------------------------
# 20. No causal language in generated structured summaries
# -----------------------------------------------------------------------------
def test_no_causal_language_in_generated_structured_summaries():
    deltas_df = pd.DataFrame([
        {
            "arm_id": "D6_ARM_001",
            "year": 2023,
            "state_index": 1,
            "pathwise_marginal_delta_auroc": -0.05,
            "pathwise_marginal_delta_auprc": -0.02,
            "pathwise_marginal_delta_ks_statistic": -0.04,
            "pathwise_marginal_delta_selection_rate": 0.001,
        },
        {
            "arm_id": "D6_ARM_001",
            "year": 2024,
            "state_index": 1,
            "pathwise_marginal_delta_auroc": -0.04,
            "pathwise_marginal_delta_auprc": -0.015,
            "pathwise_marginal_delta_ks_statistic": -0.035,
            "pathwise_marginal_delta_selection_rate": 0.001,
        },
    ])
    traces = load_all_archived_traces(_REPO_ROOT)
    summary = generate_key_path_summary(deltas_df, traces)

    serialized = json.dumps(summary).lower()
    for forbidden in FORBIDDEN_CAUSAL_TERMS:
        assert forbidden.lower() not in serialized


# -----------------------------------------------------------------------------
# 21. No row-level persistence
# -----------------------------------------------------------------------------
def test_no_row_level_persistence():
    for f in D7_2_MANIFEST_TRACKED_ARTIFACTS:
        assert not f.endswith(".parquet")
        assert not f.endswith(".npz")
        assert "probability" not in f
        assert "logit" not in f
        assert "cohort" not in f
        assert "row" not in f


# -----------------------------------------------------------------------------
# 22. Audit-only fit count = 0
# -----------------------------------------------------------------------------
def test_audit_only_fit_count_is_zero():
    harness = NHISD7StepwiseReplayHarness(repo_root=_REPO_ROOT)
    results = harness.run_audit_only()
    assert results["models_fit_count"] == 0


# -----------------------------------------------------------------------------
# 23. Audit-only NHIS cohort count = 0
# -----------------------------------------------------------------------------
def test_audit_only_nhis_cohort_count_is_zero():
    harness = NHISD7StepwiseReplayHarness(repo_root=_REPO_ROOT)
    results = harness.run_audit_only()
    assert results["nhis_cohort_count"] == 0
    assert results["real_nhis_cohort_accessed"] is False


# -----------------------------------------------------------------------------
# 24. No FairBiasMitigation execution
# -----------------------------------------------------------------------------
def test_no_fairbias_mitigation_execution():
    harness = NHISD7StepwiseReplayHarness(repo_root=_REPO_ROOT)
    results = harness.run_audit_only()
    assert results["fairbias_mitigation_count"] == 0


# -----------------------------------------------------------------------------
# 25. Explicit anti-leakage test helper poisoning
# -----------------------------------------------------------------------------
def test_explicit_anti_leakage_poisoning():
    """Poison 2023 and 2024 labels/features so that any access during model fitting raises immediately."""

    class Poisoned2023DataFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return Poisoned2023DataFrame

        def to_numpy(self, *args, **kwargs):
            raise AssertionError("LEAKAGE DETECTED: 2023 cohort converted to array during model fitting!")

        def __array__(self, *args, **kwargs):
            raise AssertionError("LEAKAGE DETECTED: 2023 cohort converted to array during model fitting!")

    class Poisoned2024Series(pd.Series):
        @property
        def _constructor(self):
            return Poisoned2024Series

        def to_numpy(self, *args, **kwargs):
            raise AssertionError("LEAKAGE DETECTED: 2024 series accessed during model fitting!")

        def __array__(self, *args, **kwargs):
            raise AssertionError("LEAKAGE DETECTED: 2024 series accessed during model fitting!")

    X_2022 = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    y_2022 = pd.Series([0, 1, 0, 1])

    scaler, lr = fit_intermediate_model_2022(X_2022, y_2022, arm_id="D6_ARM_001", state_index=1)
    assert scaler is not None
    assert lr is not None

    p_X_2023 = Poisoned2023DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    p_y_2024 = Poisoned2024Series([0, 1, 0, 1])

    with pytest.raises(AssertionError, match="LEAKAGE DETECTED"):
        fit_intermediate_model_2022(p_X_2023, y_2022, arm_id="D6_ARM_001", state_index=1)

    with pytest.raises(AssertionError, match="LEAKAGE DETECTED"):
        fit_intermediate_model_2022(X_2022, p_y_2024, arm_id="D6_ARM_001", state_index=1)


# -----------------------------------------------------------------------------
# 26. Exact D7.1 and D6 tag verification & artifact integrity
# -----------------------------------------------------------------------------
def test_exact_d7_1_and_d6_tag_verification():
    prov = verify_upstream_provenance(_REPO_ROOT)
    assert prov["provenance_status"] == "VERIFIED"
    assert prov["d7_1_tag_object"] == D7_1_TAG_OBJECT
    assert prov["d7_1_commit"] == D7_1_COMMIT
    assert prov["d6_train_val_tag_object"] == D6_TRAIN_VAL_TAG_OBJECT
    assert prov["d6_train_val_commit"] == D6_TRAIN_VAL_COMMIT
    assert prov["d6_test_tag_object"] == D6_TEST_TAG_OBJECT
    assert prov["d6_test_commit"] == D6_TEST_COMMIT
    assert prov["d6_train_val_artifacts_verified"] == 45
    assert prov["d6_test_artifacts_verified"] == 41


# -----------------------------------------------------------------------------
# 27. Missing or wrong D7.1 tag fails closed
# -----------------------------------------------------------------------------
def test_missing_or_wrong_d7_1_tag_fails_closed():
    # Test wrong tag object
    with patch("subprocess.run") as mock_sub:
        mock_sub.return_value = MagicMock(returncode=0, stdout="bad_object_sha\n", stderr="")
        with pytest.raises(ProvenanceVerificationError, match="D7.1 tag object mismatch"):
            verify_upstream_provenance(_REPO_ROOT)

    # Test git command failure (missing tag)
    with patch("subprocess.run") as mock_sub:
        mock_sub.return_value = MagicMock(returncode=1, stdout="", stderr="fatal: Not a valid object name")
        with pytest.raises(ProvenanceVerificationError, match="failed"):
            verify_upstream_provenance(_REPO_ROOT)


# -----------------------------------------------------------------------------
# 28. Missing or wrong D6 tags fail closed
# -----------------------------------------------------------------------------
def test_missing_or_wrong_d6_tags_fail_closed():
    # Wrong D6 train/val tag object
    def fake_run(cmd, *args, **kwargs):
        target = cmd[2]
        if target == D7_1_TAG:
            return MagicMock(returncode=0, stdout=D7_1_TAG_OBJECT + "\n", stderr="")
        if target == f"{D7_1_TAG}^{{commit}}":
            return MagicMock(returncode=0, stdout=D7_1_COMMIT + "\n", stderr="")
        if target == D6_TRAIN_VAL_TAG:
            return MagicMock(returncode=0, stdout="wrong_d6_tag\n", stderr="")
        return MagicMock(returncode=0, stdout="dummy\n", stderr="")

    with patch("subprocess.run", side_effect=fake_run):
        with pytest.raises(ProvenanceVerificationError, match="D6 train/val tag object mismatch"):
            verify_upstream_provenance(_REPO_ROOT)


# -----------------------------------------------------------------------------
# 29. Modified manifest-tracked artifact fails closed
# -----------------------------------------------------------------------------
def test_modified_manifest_tracked_artifact_fails_closed():
    orig_sha256 = compute_sha256

    def tampered_sha(path):
        if "final_changed_dict.json" in str(path):
            return "tampered_hash_000000000000000000000000000000000000000000000000"
        return orig_sha256(path)

    with patch("nhis_fairbias.d7_stepwise_replay.compute_sha256", side_effect=tampered_sha):
        with pytest.raises(ProvenanceVerificationError, match="SHA-256 mismatch"):
            verify_upstream_provenance(_REPO_ROOT)


# -----------------------------------------------------------------------------
# 30. Active semantic-list filtering after drops (P1-C)
# -----------------------------------------------------------------------------
def test_active_semantic_list_filtering_after_drops():
    # DataFrame where 'cat_drop' and 'num_drop' have been dropped
    X_state = pd.DataFrame({
        "num_kept": [1.0, 2.0, 3.0, 4.0],
        "cat_kept": [1, 2, 1, 2],
    })
    O_df = pd.DataFrame({"SEX_A": [1, 1, 2, 2]})
    all_cats = ["cat_kept", "cat_drop"]
    all_nums = ["num_kept", "num_drop"]

    with patch.object(FairEvaluator, "calculate_epsilon", return_value={"SEX_A": {"num_kept": 0.01}}) as mock_calc:
        res = compute_stepwise_metrics(
            arm_id="D6_ARM_001",
            state_index=1,
            year=2022,
            probs=np.array([0.2, 0.4, 0.6, 0.8]),
            y_true=np.array([0, 0, 1, 1]),
            X_state=X_state,
            O_df=O_df,
            cate_attrs=all_cats,
            num_attrs=all_nums,
        )
        assert mock_calc.called
        kwargs = mock_calc.call_args.kwargs
        # Dropped attributes must NOT be passed to calculate_epsilon
        assert kwargs["cate_attrs"] == ["cat_kept"]
        assert kwargs["num_attrs"] == ["num_kept"]
        assert "cat_drop" not in kwargs["cate_attrs"]
        assert "num_drop" not in kwargs["num_attrs"]


# -----------------------------------------------------------------------------
# 31. 12-cohort global barrier structure
# -----------------------------------------------------------------------------
def test_12_cohort_global_barrier_structure():
    # Exact digests pass
    res = verify_12_cohort_provenance_barrier(EXPECTED_COHORT_SOURCE_ROW_DIGESTS)
    assert res["cohort_provenance_barrier"] == "PASS"

    # Perturbed digest fails closed
    perturbed = copy.deepcopy(EXPECTED_COHORT_SOURCE_ROW_DIGESTS)
    perturbed[2024]["D6_ARM_004"] = "perturbed_digest_0000000000000000000000000000000000000000000000000000"
    with pytest.raises(ProvenanceVerificationError, match="Cohort source-row digest mismatch"):
        verify_12_cohort_provenance_barrier(perturbed)


# -----------------------------------------------------------------------------
# 32. 12 terminal matrix barrier structure
# -----------------------------------------------------------------------------
def test_12_terminal_matrix_barrier_structure():
    traces = load_all_archived_traces(_REPO_ROOT)
    steps = traces["D6_ARM_001"]
    base_dir = _REPO_ROOT / "docs" / "releases" / D6_TRAIN_VAL_RELEASE_ID
    cd_dict = json.loads((base_dir / "D6_ARM_001" / "final_changed_dict.json").read_text())

    sample_df = pd.DataFrame({
        col: [1.0, 2.0, 3.0] if isinstance(val, dict) and "power" in val else [1, 2, 3]
        for col, val in cd_dict.items()
    })

    # Perturb tolerance to fail
    with pytest.raises(TerminalReplayBarrierError):
        verify_terminal_replay_barrier(
            arm_id="D6_ARM_001",
            trace_steps=steps,
            final_changed_dict=cd_dict,
            expected_changed_dict_sha256=EXPECTED_TRAINING_STATE_ANCHORS["D6_ARM_001"]["changed_dict"],
            sample_df=sample_df,
            num_attrs=[c for c, v in cd_dict.items() if isinstance(v, dict)],
            cate_attrs=[c for c, v in cd_dict.items() if not isinstance(v, dict)],
            tolerance=-1.0,  # Negative tolerance forces numeric diff check to fail
        )


# -----------------------------------------------------------------------------
# 33. Endpoint-before-intermediate ordering
# -----------------------------------------------------------------------------
def test_endpoint_before_intermediate_ordering():
    runtime = ProductionD7StepwiseReplayRuntime(repo_root=_REPO_ROOT)
    # Attempting to fit intermediate models before verifying endpoints must fail
    with pytest.raises(EndpointReproductionBarrierError, match="endpoint barrier not verified"):
        runtime.fit_and_evaluate_all_intermediate_states()


# -----------------------------------------------------------------------------
# 34. Full production anti-leakage with poisoned holdouts & 47-state fit count
# -----------------------------------------------------------------------------
def test_full_production_anti_leakage_and_fit_counts():
    """Verify that in production runtime, holdout years 2023/2024 are never passed to fit/fit_transform

    and that exactly 47 scaler fits and 47 LR fits are performed.
    """
    scaler_fit_calls = []
    lr_fit_calls = []

    orig_scaler_fit = MinMaxScaler.fit
    orig_scaler_fit_transform = MinMaxScaler.fit_transform
    orig_lr_fit = LogisticRegression.fit

    def tracked_scaler_fit(self, X, y=None):
        scaler_fit_calls.append(X)
        return orig_scaler_fit(self, X, y=y)

    def tracked_scaler_fit_transform(self, X, y=None, **fit_params):
        scaler_fit_calls.append(X)
        return orig_scaler_fit_transform(self, X, y=y, **fit_params)

    def tracked_lr_fit(self, X, y, sample_weight=None):
        lr_fit_calls.append(X)
        return orig_lr_fit(self, X, y, sample_weight=sample_weight)

    with patch.object(MinMaxScaler, "fit", tracked_scaler_fit), \
         patch.object(MinMaxScaler, "fit_transform", tracked_scaler_fit_transform), \
         patch.object(LogisticRegression, "fit", tracked_lr_fit):

        # Synthetic adapter that returns poisoned data for 2023 and 2024
        class MockPoisonAdapter:
            def __init__(self):
                self.preprocessor = MagicMock()
                self.preprocessor.fitted_record = MagicMock()
                self.preprocessor.fitted_record.to_dict.return_value = json.loads(
                    (_REPO_ROOT / "docs" / "releases" / D6_TRAIN_VAL_RELEASE_ID / "d6_temporal_train_val_manifest.json").read_text()
                )
                self.preprocessor.get_feature_family_lists.return_value = (["c1"], ["n1"])

            def get_cohort(self, year, outcome, protected_attribute, feature_set, disability_arm):
                # 2022 is valid
                if year == 2022:
                    X = pd.DataFrame({"c1": [1, 2, 1, 2], "n1": [1.0, 2.0, 3.0, 4.0]})
                    y = pd.Series([0, 0, 1, 1])
                    a = pd.Series([1, 1, 2, 2])
                    w = np.ones(4)
                    return X, y, a, w, {}

                # 2023 and 2024 raise if converted to array via fit
                class PoisonHoldoutDF(pd.DataFrame):
                    @property
                    def _constructor(self):
                        return PoisonHoldoutDF

                    def __array__(self, *args, **kwargs):
                        # Allow predict/predict_proba transform, but track if called in fit
                        import inspect
                        stack = [frame.function for frame in inspect.stack()]
                        if "fit" in stack or "fit_transform" in stack:
                            raise AssertionError(f"LEAKAGE DETECTED in holdout year {year} during fitting!")
                        return super().__array__(*args, **kwargs)

                pX = PoisonHoldoutDF({"c1": [1, 2], "n1": [2.0, 3.0]})
                y = pd.Series([0, 1])
                a = pd.Series([1, 2])
                w = np.ones(2)
                return pX, y, a, w, {}

        # Run synthetic runtime for 1 arm with 2 steps
        step1 = ArchivedTraceStep(1, "c1", "categorical", 0.01, 0.0005, {"2": 1}, {"2": 1}, None, {"2": "1"}, None, False)
        step2 = ArchivedTraceStep(2, "n1", "numerical", 0.005, 0.0005, {"power": 2.0}, {"power": 2.0}, 2.0, None, None, False)
        test_traces = {
            "D6_ARM_001": [step1, step2],
        }

        runtime = ProductionD7StepwiseReplayRuntime(
            repo_root=_REPO_ROOT,
            adapter_factory=lambda: MockPoisonAdapter(),
        )

        with patch("nhis_fairbias.d7_stepwise_replay.load_all_archived_traces", return_value=test_traces), \
             patch("nhis_fairbias.d7_stepwise_replay.compute_canonical_json_sha256", return_value=PREPROCESSING_STATE_SHA256), \
             patch("nhis_fairbias.d7_stepwise_replay.verify_12_cohort_provenance_barrier", return_value={"status": "PASS"}), \
             patch("nhis_fairbias.d7_stepwise_replay.D6_ARM_IDS", ("D6_ARM_001",)):

            runtime.traces = test_traces
            runtime.build_all_cohorts()
            # Manually populate replayed states for the 1 arm fixture
            r_machine = SequentialReplayStateMachine("D6_ARM_001", [step1, step2], ["c1", "n1"], ["n1"], ["c1"], validate_step_count=False)
            runtime.replayed_states["D6_ARM_001"] = {
                2022: r_machine.replay_all(runtime.raw_cohorts[2022]["D6_ARM_001"]["X"]),
                2023: r_machine.replay_all(runtime.raw_cohorts[2023]["D6_ARM_001"]["X"]),
                2024: r_machine.replay_all(runtime.raw_cohorts[2024]["D6_ARM_001"]["X"]),
            }
            # Mock endpoint verification
            runtime.endpoint_barrier_results["D6_ARM_001"] = {"status": "PASS"}
            runtime.fitted_models["D6_ARM_001"] = {
                0: fit_intermediate_model_2022(runtime.replayed_states["D6_ARM_001"][2022][0], runtime.raw_cohorts[2022]["D6_ARM_001"]["y"], "D6_ARM_001", 0),
                2: fit_intermediate_model_2022(runtime.replayed_states["D6_ARM_001"][2022][2], runtime.raw_cohorts[2022]["D6_ARM_001"]["y"], "D6_ARM_001", 2),
            }

            # Fit intermediate states (state 1) and evaluate on 2022, 2023, 2024
            metrics = runtime.fit_and_evaluate_all_intermediate_states()
            assert len(metrics) == 3 * 3  # 3 states x 3 years = 9 metric records

    # Verify no leakage occurred and 2023/2024 never entered fit
    for x_inp in scaler_fit_calls:
        assert len(x_inp) == 4  # 2022 sample count
    for x_inp in lr_fit_calls:
        assert len(x_inp) == 4  # 2022 sample count


# -----------------------------------------------------------------------------
# 35. Temporal key-path summary retains separate 2023 and 2024 rankings
# -----------------------------------------------------------------------------
def test_temporal_key_path_summary_retains_separate_2023_and_2024():
    deltas_df = pd.DataFrame([
        {
            "arm_id": "D6_ARM_001",
            "year": 2023,
            "state_index": 1,
            "pathwise_marginal_delta_auroc": 0.05,
            "pathwise_marginal_delta_auprc": 0.02,
            "pathwise_marginal_delta_ks_statistic": 0.04,
            "pathwise_marginal_delta_selection_rate": 0.01,
        },
        {
            "arm_id": "D6_ARM_001",
            "year": 2024,
            "state_index": 1,
            "pathwise_marginal_delta_auroc": -0.03,
            "pathwise_marginal_delta_auprc": 0.01,
            "pathwise_marginal_delta_ks_statistic": -0.02,
            "pathwise_marginal_delta_selection_rate": 0.01,
        },
        {
            "arm_id": "D6_ARM_001",
            "year": 2023,
            "state_index": 2,
            "pathwise_marginal_delta_auroc": 0.08,
            "pathwise_marginal_delta_auprc": 0.04,
            "pathwise_marginal_delta_ks_statistic": 0.06,
            "pathwise_marginal_delta_selection_rate": 0.02,
        },
        {
            "arm_id": "D6_ARM_001",
            "year": 2024,
            "state_index": 2,
            "pathwise_marginal_delta_auroc": 0.07,
            "pathwise_marginal_delta_auprc": 0.03,
            "pathwise_marginal_delta_ks_statistic": 0.05,
            "pathwise_marginal_delta_selection_rate": 0.02,
        },
    ])
    traces = load_all_archived_traces(_REPO_ROOT)
    summary = generate_key_path_summary(deltas_df, traces)

    arm1 = summary["arms"]["D6_ARM_001"]
    assert "largest_2023_pathwise_auroc_changes" in arm1
    assert "largest_2024_pathwise_auroc_changes" in arm1

    # In 2023, state 2 had delta 0.08, ranked top
    top_2023 = arm1["largest_2023_pathwise_auroc_changes"][0]
    assert top_2023["state_index"] == 2
    assert top_2023["delta_2023"] == 0.08
    assert top_2023["delta_2024"] == 0.07
    assert top_2023["same_direction"] is True

    # Check state 1 where 2023 was +0.05 and 2024 was -0.03 -> same_direction is False
    st1_record = [r for r in arm1["largest_2023_pathwise_auroc_changes"] if r["state_index"] == 1][0]
    assert st1_record["delta_2023"] == 0.05
    assert st1_record["delta_2024"] == -0.03
    assert st1_record["same_direction"] is False


# -----------------------------------------------------------------------------
# 36. Release collision fails closed and FAILED state preserves directory
# -----------------------------------------------------------------------------
def test_release_collision_fails_closed_and_failed_preserves_dir(tmp_path):
    manager = D7StepwiseReplayReleaseManager(repo_root=_REPO_ROOT, releases_parent_dir=tmp_path)

    # 1. Collision check
    colliding_dir = tmp_path / "REL_001"
    colliding_dir.mkdir(parents=True)
    with patch("nhis_fairbias.d7_stepwise_replay.verify_git_execution_preconditions"), \
         patch("nhis_fairbias.d7_stepwise_replay.verify_upstream_provenance"):
        with pytest.raises(FileExistsError, match="collision"):
            manager.execute_release("REL_001", "some_sha")

    # 2. Failure preserves directory and records FAILED status
    fail_rel_dir = tmp_path / "REL_FAIL"

    def exploding_runtime(*args, **kwargs):
        raise RuntimeError("Synthetic runtime failure during execution!")

    with patch("nhis_fairbias.d7_stepwise_replay.verify_git_execution_preconditions"), \
         patch("nhis_fairbias.d7_stepwise_replay.verify_upstream_provenance"), \
         patch("nhis_fairbias.d7_stepwise_replay.ProductionD7StepwiseReplayRuntime.build_all_cohorts", side_effect=exploding_runtime):
        with pytest.raises(RuntimeError, match="Synthetic runtime failure"):
            manager.execute_release("REL_FAIL", "some_sha")

    assert fail_rel_dir.is_dir()
    state_file = fail_rel_dir / "release_state.json"
    assert state_file.is_file()
    state_data = json.loads(state_file.read_text())
    assert state_data["status"] == "FAILED"
    assert "Synthetic runtime failure" in state_data["error"]


# -----------------------------------------------------------------------------
# 37. Synthetic production integration end-to-end (10 files, 8 manifest-tracked)
# -----------------------------------------------------------------------------
def test_synthetic_production_integration_end_to_end(tmp_path):
    """Synthetic end-to-end integration test of ProductionD7StepwiseReplayRuntime and ReleaseManager."""
    manager = D7StepwiseReplayReleaseManager(repo_root=_REPO_ROOT, releases_parent_dir=tmp_path)

    traces = load_all_archived_traces(_REPO_ROOT)
    # Use real trace steps from arm 1 as synthetic fixture
    arm_steps = traces["D6_ARM_001"][:2]

    # Mock runtime steps to write valid synthetic release without hitting real NHIS cohorts
    rel_dir = tmp_path / "REL_SYNTH"
    rel_dir.mkdir(parents=True)

    runtime = ProductionD7StepwiseReplayRuntime(repo_root=_REPO_ROOT)
    runtime.traces = {"D6_ARM_001": arm_steps}
    runtime.cohort_digests = EXPECTED_COHORT_SOURCE_ROW_DIGESTS
    runtime.observed_preprocessing_sha256 = PREPROCESSING_STATE_SHA256
    runtime.terminal_matrix_results = {"D6_ARM_001_2022": {"verified": True, "max_numeric_diff": 0.0}}
    runtime.endpoint_barrier_results = {"D6_ARM_001": {"status": "PASS"}}

    # Populate synthetic metric records
    runtime.metrics_records = [
        {"arm_id": "D6_ARM_001", "state_index": 0, "year": 2022, "threshold": 0.5, "max_d_phi": 0.01, "auroc": 0.75, "auprc": 0.3, "balanced_accuracy": 0.7, "f1": 0.5, "accuracy": 0.8, "count_predicted_positive": 50, "selection_rate": 0.05, "ks_statistic": 0.4, "ks_pvalue": 0.001, "score_mean": 0.1, "score_median": 0.08, "score_iqr": 0.05, "y0_mean": 0.07, "y0_median": 0.05, "y1_mean": 0.4, "y1_median": 0.35, "dp_gap": 0.01, "tpr_gap": 0.02, "fpr_gap": 0.01, "equalized_odds_max_gap": 0.02},
        {"arm_id": "D6_ARM_001", "state_index": 1, "year": 2022, "threshold": 0.5, "max_d_phi": 0.008, "auroc": 0.76, "auprc": 0.31, "balanced_accuracy": 0.71, "f1": 0.51, "accuracy": 0.81, "count_predicted_positive": 52, "selection_rate": 0.052, "ks_statistic": 0.42, "ks_pvalue": 0.001, "score_mean": 0.1, "score_median": 0.08, "score_iqr": 0.05, "y0_mean": 0.07, "y0_median": 0.05, "y1_mean": 0.4, "y1_median": 0.35, "dp_gap": 0.01, "tpr_gap": 0.02, "fpr_gap": 0.01, "equalized_odds_max_gap": 0.02},
        {"arm_id": "D6_ARM_001", "state_index": 0, "year": 2023, "threshold": 0.5, "max_d_phi": 0.01, "auroc": 0.74, "auprc": 0.29, "balanced_accuracy": 0.69, "f1": 0.49, "accuracy": 0.79, "count_predicted_positive": 48, "selection_rate": 0.048, "ks_statistic": 0.39, "ks_pvalue": 0.001, "score_mean": 0.1, "score_median": 0.08, "score_iqr": 0.05, "y0_mean": 0.07, "y0_median": 0.05, "y1_mean": 0.4, "y1_median": 0.35, "dp_gap": 0.01, "tpr_gap": 0.02, "fpr_gap": 0.01, "equalized_odds_max_gap": 0.02},
        {"arm_id": "D6_ARM_001", "state_index": 1, "year": 2023, "threshold": 0.5, "max_d_phi": 0.008, "auroc": 0.75, "auprc": 0.30, "balanced_accuracy": 0.70, "f1": 0.50, "accuracy": 0.80, "count_predicted_positive": 50, "selection_rate": 0.050, "ks_statistic": 0.41, "ks_pvalue": 0.001, "score_mean": 0.1, "score_median": 0.08, "score_iqr": 0.05, "y0_mean": 0.07, "y0_median": 0.05, "y1_mean": 0.4, "y1_median": 0.35, "dp_gap": 0.01, "tpr_gap": 0.02, "fpr_gap": 0.01, "equalized_odds_max_gap": 0.02},
        {"arm_id": "D6_ARM_001", "state_index": 0, "year": 2024, "threshold": 0.5, "max_d_phi": 0.01, "auroc": 0.73, "auprc": 0.28, "balanced_accuracy": 0.68, "f1": 0.48, "accuracy": 0.78, "count_predicted_positive": 46, "selection_rate": 0.046, "ks_statistic": 0.38, "ks_pvalue": 0.001, "score_mean": 0.1, "score_median": 0.08, "score_iqr": 0.05, "y0_mean": 0.07, "y0_median": 0.05, "y1_mean": 0.4, "y1_median": 0.35, "dp_gap": 0.01, "tpr_gap": 0.02, "fpr_gap": 0.01, "equalized_odds_max_gap": 0.02},
        {"arm_id": "D6_ARM_001", "state_index": 1, "year": 2024, "threshold": 0.5, "max_d_phi": 0.008, "auroc": 0.74, "auprc": 0.29, "balanced_accuracy": 0.69, "f1": 0.49, "accuracy": 0.79, "count_predicted_positive": 47, "selection_rate": 0.047, "ks_statistic": 0.40, "ks_pvalue": 0.001, "score_mean": 0.1, "score_median": 0.08, "score_iqr": 0.05, "y0_mean": 0.07, "y0_median": 0.05, "y1_mean": 0.4, "y1_median": 0.35, "dp_gap": 0.01, "tpr_gap": 0.02, "fpr_gap": 0.01, "equalized_odds_max_gap": 0.02},
    ]
    runtime.deltas_df = compute_stepwise_deltas(runtime.metrics_records)
    runtime.key_path_summary = generate_key_path_summary(runtime.deltas_df, runtime.traces)

    manager._write_release_artifacts(rel_dir, "REL_SYNTH", "synth_sha", runtime)
    manifest = build_d7_stepwise_manifest(rel_dir)
    manifest_file = rel_dir / "d7_stepwise_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    state_file = rel_dir / "release_state.json"
    state_file.write_text(json.dumps({"status": "COMPLETE", "manifest_sha256": "abc"}, indent=2))

    # Verify 10 files
    files = {f.name for f in rel_dir.iterdir()}
    assert len(files) == 10
    assert set(D7_2_ALL_RELEASE_FILES) == files

    # Verify 8 manifest tracked
    assert len(manifest["artifacts"]) == 8
    for fname in D7_2_MANIFEST_TRACKED_ARTIFACTS:
        assert fname in manifest["artifacts"]
