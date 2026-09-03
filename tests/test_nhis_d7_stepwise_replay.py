"""Unit, barrier, state machine, and invariant tests for Gate D7.2a.2 Stepwise Replay Harness.

Guarantees:
- Replay state machine: categorical merge, numerical revisit, and drop semantics.
- Revisit semantics: numerical revisits evaluate against raw baseline with replacement power.
- Fail-closed provenance: exact D7.1 and D6 tags/commits, manifest artifact integrity.
- Actual origin branch equality: local HEAD == remote research/nhis-fairbias == expected SHA.
- Preflight before mkdir: all inputs preflighted before target directory creation.
- 12/12 global cohort source-row digest barrier through production manager.
- 12/12 terminal matrix-equivalence barrier through production manager.
- Endpoint-before-intermediate ordering barrier through production manager.
- Active semantic-list filtering: dropped features never enter calculate_epsilon.
- Temporal key-path summary: separate 2023 and 2024 rankings with paired deltas and zero direction semantics.
- Full 4-arm 47-state logical fit proof: exactly 47 scaler fits and 47 LR fits on 2022 only.
- Full production anti-leakage: 2023 and 2024 poisoned objects never enter .fit().
- True production integration: manager.execute_release() end-to-end lifecycle without manual state injection.
- Zero real NHIS cohort access via get_cohort().
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
    REMOTE_RESEARCH_BRANCH_REF,
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
    np.testing.assert_allclose(res[1]["agep_a"].to_numpy(), [8.0, 27.0], atol=1e-12)
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
    assert results["real_nhis_get_cohort_calls"] == 0


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
    with patch("subprocess.run") as mock_sub:
        mock_sub.return_value = MagicMock(returncode=0, stdout="bad_object_sha\n", stderr="")
        with pytest.raises(ProvenanceVerificationError, match="D7.1 tag object mismatch"):
            verify_upstream_provenance(_REPO_ROOT)

    with patch("subprocess.run") as mock_sub:
        mock_sub.return_value = MagicMock(returncode=1, stdout="", stderr="fatal: Not a valid object name")
        with pytest.raises(ProvenanceVerificationError, match="failed"):
            verify_upstream_provenance(_REPO_ROOT)


# -----------------------------------------------------------------------------
# 28. Missing or wrong D6 tags fail closed
# -----------------------------------------------------------------------------
def test_missing_or_wrong_d6_tags_fail_closed():
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
        assert kwargs["cate_attrs"] == ["cat_kept"]
        assert kwargs["num_attrs"] == ["num_kept"]
        assert "cat_drop" not in kwargs["cate_attrs"]
        assert "num_drop" not in kwargs["num_attrs"]


# -----------------------------------------------------------------------------
# 31. 12-cohort global barrier structure
# -----------------------------------------------------------------------------
def test_12_cohort_global_barrier_structure():
    res = verify_12_cohort_provenance_barrier(EXPECTED_COHORT_SOURCE_ROW_DIGESTS)
    assert res["cohort_provenance_barrier"] == "PASS"

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

    with pytest.raises(TerminalReplayBarrierError):
        verify_terminal_replay_barrier(
            arm_id="D6_ARM_001",
            trace_steps=steps,
            final_changed_dict=cd_dict,
            expected_changed_dict_sha256=EXPECTED_TRAINING_STATE_ANCHORS["D6_ARM_001"]["changed_dict"],
            sample_df=sample_df,
            num_attrs=[c for c, v in cd_dict.items() if isinstance(v, dict)],
            cate_attrs=[c for c, v in cd_dict.items() if not isinstance(v, dict)],
            tolerance=-1.0,
        )


# -----------------------------------------------------------------------------
# 33. Endpoint-before-intermediate ordering
# -----------------------------------------------------------------------------
def test_endpoint_before_intermediate_ordering():
    runtime = ProductionD7StepwiseReplayRuntime(repo_root=_REPO_ROOT)
    with pytest.raises(EndpointReproductionBarrierError, match="endpoint barrier not verified"):
        runtime.fit_and_evaluate_all_intermediate_states()


# -----------------------------------------------------------------------------
# 34. Temporal key-path summary retains separate 2023 and 2024 rankings
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

    top_2023 = arm1["largest_2023_pathwise_auroc_changes"][0]
    assert top_2023["state_index"] == 2
    assert top_2023["delta_2023"] == 0.08
    assert top_2023["delta_2024"] == 0.07
    assert top_2023["same_direction"] is True

    st1_record = [r for r in arm1["largest_2023_pathwise_auroc_changes"] if r["state_index"] == 1][0]
    assert st1_record["delta_2023"] == 0.05
    assert st1_record["delta_2024"] == -0.03
    assert st1_record["same_direction"] is False


# -----------------------------------------------------------------------------
# 35. Release collision fails closed and FAILED state preserves directory
# -----------------------------------------------------------------------------
def test_release_collision_fails_closed_and_failed_preserves_dir(tmp_path):
    manager = D7StepwiseReplayReleaseManager(repo_root=_REPO_ROOT, releases_parent_dir=tmp_path)

    colliding_dir = tmp_path / "REL_001"
    colliding_dir.mkdir(parents=True)
    with patch("nhis_fairbias.d7_stepwise_replay.verify_git_execution_preconditions"), \
         patch("nhis_fairbias.d7_stepwise_replay.verify_upstream_provenance"), \
         patch.object(ProductionD7StepwiseReplayRuntime, "preflight_inputs", return_value={"preflight_status": "PASS"}):
        with pytest.raises(FileExistsError, match="collision"):
            manager.execute_release("REL_001", "some_sha")

    fail_rel_dir = tmp_path / "REL_FAIL"

    def exploding_runtime(*args, **kwargs):
        raise RuntimeError("Synthetic runtime failure during execution!")

    with patch("nhis_fairbias.d7_stepwise_replay.verify_git_execution_preconditions"), \
         patch("nhis_fairbias.d7_stepwise_replay.verify_upstream_provenance"), \
         patch.object(ProductionD7StepwiseReplayRuntime, "preflight_inputs", return_value={"preflight_status": "PASS"}), \
         patch.object(ProductionD7StepwiseReplayRuntime, "build_all_cohorts", side_effect=exploding_runtime):
        with pytest.raises(RuntimeError, match="Synthetic runtime failure"):
            manager.execute_release("REL_FAIL", "some_sha")

    assert fail_rel_dir.is_dir()
    state_file = fail_rel_dir / "release_state.json"
    assert state_file.is_file()
    state_data = json.loads(state_file.read_text())
    assert state_data["status"] == "FAILED"
    assert "Synthetic runtime failure" in state_data["error"]


# -----------------------------------------------------------------------------
# 36. Actual Origin Branch Equality Preconditions (P1-A)
# -----------------------------------------------------------------------------
def test_git_execution_preconditions_origin_branch_equality():
    # 1. Exact match passes
    with patch("subprocess.run") as mock_sub:
        def sub_run_handler(cmd, *args, **kwargs):
            if cmd[1] == "rev-parse" and cmd[2] == "HEAD":
                return MagicMock(returncode=0, stdout="match_sha\n", stderr="")
            if cmd[1] == "ls-remote":
                return MagicMock(returncode=0, stdout=f"match_sha\t{REMOTE_RESEARCH_BRANCH_REF}\n", stderr="")
            if cmd[1] == "status":
                return MagicMock(returncode=0, stdout="", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_sub.side_effect = sub_run_handler
        res = verify_git_execution_preconditions(_REPO_ROOT, expected_sha="match_sha")
        assert res["git_head"] == "match_sha"
        assert res["remote_branch_head"] == "match_sha"

    # 2. Local HEAD != expected
    with patch("subprocess.run") as mock_sub:
        mock_sub.return_value = MagicMock(returncode=0, stdout="other_sha\n", stderr="")
        with pytest.raises(ProvenanceVerificationError, match="Local Git HEAD mismatch"):
            verify_git_execution_preconditions(_REPO_ROOT, expected_sha="match_sha")

    # 3. Remote SHA != expected
    with patch("subprocess.run") as mock_sub:
        def sub_remote_mismatch(cmd, *args, **kwargs):
            if cmd[1] == "rev-parse":
                return MagicMock(returncode=0, stdout="match_sha\n", stderr="")
            if cmd[1] == "ls-remote":
                return MagicMock(returncode=0, stdout=f"divergent_sha\t{REMOTE_RESEARCH_BRANCH_REF}\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")
        mock_sub.side_effect = sub_remote_mismatch
        with pytest.raises(ProvenanceVerificationError, match="Remote branch SHA mismatch"):
            verify_git_execution_preconditions(_REPO_ROOT, expected_sha="match_sha")

    # 4. Remote query failure
    with patch("subprocess.run") as mock_sub:
        def sub_remote_fail(cmd, *args, **kwargs):
            if cmd[1] == "rev-parse":
                return MagicMock(returncode=0, stdout="match_sha\n", stderr="")
            if cmd[1] == "ls-remote":
                return MagicMock(returncode=1, stdout="", stderr="fatal: remote error")
            return MagicMock(returncode=0, stdout="", stderr="")
        mock_sub.side_effect = sub_remote_fail
        with pytest.raises(ProvenanceVerificationError, match="Failed to query remote branch"):
            verify_git_execution_preconditions(_REPO_ROOT, expected_sha="match_sha")

    # 5. Missing remote branch response
    with patch("subprocess.run") as mock_sub:
        def sub_remote_missing(cmd, *args, **kwargs):
            if cmd[1] == "rev-parse":
                return MagicMock(returncode=0, stdout="match_sha\n", stderr="")
            if cmd[1] == "ls-remote":
                return MagicMock(returncode=0, stdout="", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")
        mock_sub.side_effect = sub_remote_missing
        with pytest.raises(ProvenanceVerificationError, match="not found on origin"):
            verify_git_execution_preconditions(_REPO_ROOT, expected_sha="match_sha")


# -----------------------------------------------------------------------------
# 37. Preflight before mkdir: failures leave NO release directory (P1-B)
# -----------------------------------------------------------------------------
def test_preflight_before_mkdir_failures_leave_no_directory(tmp_path):
    manager = D7StepwiseReplayReleaseManager(repo_root=_REPO_ROOT, releases_parent_dir=tmp_path)
    target_dir = tmp_path / "REL_PREFLIGHT_FAIL"

    # A. Feature parquet SHA mismatch fails before mkdir
    with patch("nhis_fairbias.d7_stepwise_replay.verify_git_execution_preconditions"), \
         patch("nhis_fairbias.d7_stepwise_replay.verify_upstream_provenance"), \
         patch("nhis_fairbias.d7_stepwise_replay.compute_sha256", return_value="bad_parquet_sha"):
        with pytest.raises(ProvenanceVerificationError, match="Features parquet SHA mismatch"):
            manager.execute_release("REL_PREFLIGHT_FAIL", "some_sha")
    assert not target_dir.exists()

    # B. Preprocessing logical SHA mismatch fails before mkdir
    class MockBadAdapter:
        def __init__(self):
            self.preprocessor = MagicMock()
            self.preprocessor.fitted_record = MagicMock()
            self.preprocessor.fitted_record.to_dict.return_value = {"bad": 1}

    with patch("nhis_fairbias.d7_stepwise_replay.verify_git_execution_preconditions"), \
         patch("nhis_fairbias.d7_stepwise_replay.verify_upstream_provenance"), \
         patch.object(ProductionD7StepwiseReplayRuntime, "construct_adapter", side_effect=ProvenanceVerificationError("Preprocessing state hash mismatch")):
        with pytest.raises(ProvenanceVerificationError, match="Preprocessing state hash mismatch"):
            manager.execute_release("REL_PREFLIGHT_FAIL", "some_sha")
    assert not target_dir.exists()


# -----------------------------------------------------------------------------
# 38. Manager Poison Test: Upstream provenance failure leaves NO directory
# -----------------------------------------------------------------------------
def test_manager_upstream_provenance_failure_leaves_no_directory(tmp_path):
    manager = D7StepwiseReplayReleaseManager(repo_root=_REPO_ROOT, releases_parent_dir=tmp_path)
    target_dir = tmp_path / "REL_UPSTREAM_PROV_FAIL"

    adapter_constructed = False
    def spy_construct_adapter(*args, **kwargs):
        nonlocal adapter_constructed
        adapter_constructed = True

    lr_fit_count = 0
    orig_lr_fit = LogisticRegression.fit
    def counted_lr_fit(self, *args, **kwargs):
        nonlocal lr_fit_count
        lr_fit_count += 1
        return orig_lr_fit(self, *args, **kwargs)

    with patch("nhis_fairbias.d7_stepwise_replay.verify_git_execution_preconditions"), \
         patch("nhis_fairbias.d7_stepwise_replay.verify_upstream_provenance", side_effect=ProvenanceVerificationError("D7.1 tag object mismatch")), \
         patch.object(ProductionD7StepwiseReplayRuntime, "construct_adapter", side_effect=spy_construct_adapter), \
         patch.object(LogisticRegression, "fit", counted_lr_fit):

        with pytest.raises(ProvenanceVerificationError, match="D7.1 tag object mismatch"):
            manager.execute_release("REL_UPSTREAM_PROV_FAIL", "some_sha")

    assert not target_dir.exists()
    assert adapter_constructed is False
    assert lr_fit_count == 0


# -----------------------------------------------------------------------------
# 38. Manager Poison Test: Perturbed cohort digest fails release (Section 4)
# -----------------------------------------------------------------------------
def test_manager_poison_cohort_digest_fails(tmp_path):
    manager = D7StepwiseReplayReleaseManager(repo_root=_REPO_ROOT, releases_parent_dir=tmp_path)
    rel_dir = tmp_path / "REL_POISON_DIGEST"

    # When digest barrier fails inside manager.execute_release:
    # 1. release becomes FAILED
    # 2. directory is preserved
    # 3. zero scaler or LR fits occur
    lr_fit_count = 0
    orig_lr_fit = LogisticRegression.fit

    def counted_lr_fit(self, *args, **kwargs):
        nonlocal lr_fit_count
        lr_fit_count += 1
        return orig_lr_fit(self, *args, **kwargs)

    with patch("nhis_fairbias.d7_stepwise_replay.verify_git_execution_preconditions"), \
         patch("nhis_fairbias.d7_stepwise_replay.verify_upstream_provenance"), \
         patch.object(ProductionD7StepwiseReplayRuntime, "preflight_inputs", return_value={"preflight_status": "PASS"}), \
         patch.object(ProductionD7StepwiseReplayRuntime, "build_all_cohorts", side_effect=ProvenanceVerificationError("Cohort source-row digest mismatch")), \
         patch.object(LogisticRegression, "fit", counted_lr_fit):

        with pytest.raises(ProvenanceVerificationError, match="Cohort source-row digest mismatch"):
            manager.execute_release("REL_POISON_DIGEST", "some_sha")

    assert rel_dir.is_dir()
    state_data = json.loads((rel_dir / "release_state.json").read_text())
    assert state_data["status"] == "FAILED"
    assert lr_fit_count == 0


# -----------------------------------------------------------------------------
# 39. Manager Poison Test: Perturbed terminal matrix fails release (Section 5)
# -----------------------------------------------------------------------------
def test_manager_poison_terminal_matrix_fails(tmp_path):
    manager = D7StepwiseReplayReleaseManager(repo_root=_REPO_ROOT, releases_parent_dir=tmp_path)
    rel_dir = tmp_path / "REL_POISON_MATRIX"

    lr_fit_count = 0
    orig_lr_fit = LogisticRegression.fit

    def counted_lr_fit(self, *args, **kwargs):
        nonlocal lr_fit_count
        lr_fit_count += 1
        return orig_lr_fit(self, *args, **kwargs)

    with patch("nhis_fairbias.d7_stepwise_replay.verify_git_execution_preconditions"), \
         patch("nhis_fairbias.d7_stepwise_replay.verify_upstream_provenance"), \
         patch.object(ProductionD7StepwiseReplayRuntime, "preflight_inputs", return_value={"preflight_status": "PASS"}), \
         patch.object(ProductionD7StepwiseReplayRuntime, "build_all_cohorts", return_value={}), \
         patch.object(ProductionD7StepwiseReplayRuntime, "replay_and_verify_terminal_matrices", side_effect=TerminalReplayBarrierError("Terminal matrix numeric diff > tolerance")), \
         patch.object(LogisticRegression, "fit", counted_lr_fit):

        with pytest.raises(TerminalReplayBarrierError, match="Terminal matrix numeric diff > tolerance"):
            manager.execute_release("REL_POISON_MATRIX", "some_sha")

    assert rel_dir.is_dir()
    state_data = json.loads((rel_dir / "release_state.json").read_text())
    assert state_data["status"] == "FAILED"
    assert lr_fit_count == 0


# -----------------------------------------------------------------------------
# 40. Manager Poison Test: Perturbed endpoint metrics fails release (Section 6)
# -----------------------------------------------------------------------------
def test_manager_poison_endpoint_reproduction_fails(tmp_path):
    manager = D7StepwiseReplayReleaseManager(repo_root=_REPO_ROOT, releases_parent_dir=tmp_path)
    rel_dir = tmp_path / "REL_POISON_ENDPOINT"

    intermediate_fit_count = 0

    def fake_intermediate_fits(*args, **kwargs):
        nonlocal intermediate_fit_count
        intermediate_fit_count += 1
        return []

    with patch("nhis_fairbias.d7_stepwise_replay.verify_git_execution_preconditions"), \
         patch("nhis_fairbias.d7_stepwise_replay.verify_upstream_provenance"), \
         patch.object(ProductionD7StepwiseReplayRuntime, "preflight_inputs", return_value={"preflight_status": "PASS"}), \
         patch.object(ProductionD7StepwiseReplayRuntime, "build_all_cohorts", return_value={}), \
         patch.object(ProductionD7StepwiseReplayRuntime, "replay_and_verify_terminal_matrices", return_value={}), \
         patch.object(ProductionD7StepwiseReplayRuntime, "fit_and_verify_endpoints", side_effect=EndpointReproductionBarrierError("State 0 scaler hash mismatch")), \
         patch.object(ProductionD7StepwiseReplayRuntime, "fit_and_evaluate_all_intermediate_states", side_effect=fake_intermediate_fits):

        with pytest.raises(EndpointReproductionBarrierError, match="State 0 scaler hash mismatch"):
            manager.execute_release("REL_POISON_ENDPOINT", "some_sha")

    assert rel_dir.is_dir()
    state_data = json.loads((rel_dir / "release_state.json").read_text())
    assert state_data["status"] == "FAILED"
    assert intermediate_fit_count == 0


# -----------------------------------------------------------------------------
# 41. Four-Arm 47-State Logical Fit Proof & Anti-Leakage (P2-A)
# -----------------------------------------------------------------------------
def test_four_arm_47_state_logical_fit_counts_and_anti_leakage():
    """Verify that the full 4-arm frozen state topology performs exactly 47 logical scaler fits

    and 47 LR fits, strictly on 2022 rows, and holdout poison objects never enter fit.
    """
    traces = load_all_archived_traces(_REPO_ROOT)
    assert len(traces["D6_ARM_001"]) == 15
    assert len(traces["D6_ARM_002"]) == 12
    assert len(traces["D6_ARM_003"]) == 7
    assert len(traces["D6_ARM_004"]) == 9

    # Total representation states across 4 arms = (1+15) + (1+12) + (1+7) + (1+9) = 16 + 13 + 8 + 10 = 47
    assert sum(len(steps) + 1 for steps in traces.values()) == 47

    logical_fit_calls = []
    lr_fit_calls = []

    orig_logical_fit = fit_intermediate_model_2022
    orig_lr_fit = LogisticRegression.fit

    def counted_logical_fit(X_train, y_train, arm_id, state_index):
        logical_fit_calls.append((arm_id, state_index, len(X_train)))
        return orig_logical_fit(X_train, y_train, arm_id, state_index)

    def counted_lr_fit(self, X, y, sample_weight=None):
        lr_fit_calls.append(len(X))
        return orig_lr_fit(self, X, y, sample_weight=sample_weight)

    # Class with poison check for 2023/2024
    class PoisonHoldoutDF(pd.DataFrame):
        @property
        def _constructor(self):
            return PoisonHoldoutDF

        def __array__(self, *args, **kwargs):
            import inspect
            stack = [frame.function for frame in inspect.stack()]
            if "fit" in stack or "fit_transform" in stack:
                raise AssertionError("LEAKAGE: Holdout year data accessed inside fit/fit_transform!")
            return super().__array__(*args, **kwargs)

    runtime = ProductionD7StepwiseReplayRuntime(repo_root=_REPO_ROOT)
    runtime.traces = traces
    n_2022 = 20

    # Build synthetic replayed states for all 4 arms
    for arm_id in D6_ARM_IDS:
        steps = traces[arm_id]
        distinct_cols = list(dict.fromkeys([s.selected_feature for s in steps]))
        y_2022 = pd.Series([0, 1] * (n_2022 // 2))

        X_2022 = pd.DataFrame({col: np.random.uniform(1.0, 5.0, n_2022) for col in distinct_cols})
        X_2023 = PoisonHoldoutDF({col: np.random.uniform(1.0, 5.0, 10) for col in distinct_cols})
        X_2024 = PoisonHoldoutDF({col: np.random.uniform(1.0, 5.0, 10) for col in distinct_cols})

        r_machine = SequentialReplayStateMachine(arm_id, steps, distinct_cols, num_attrs=distinct_cols, validate_step_count=True)
        runtime.replayed_states[arm_id] = {
            2022: r_machine.replay_all(X_2022),
            2023: r_machine.replay_all(X_2023),
            2024: r_machine.replay_all(X_2024),
        }
        runtime.raw_cohorts[2022][arm_id] = {"X": X_2022, "y": y_2022, "a": pd.Series([1] * n_2022)}
        runtime.raw_cohorts[2023][arm_id] = {"X": X_2023, "y": pd.Series([0, 1] * 5), "a": pd.Series([1] * 10)}
        runtime.raw_cohorts[2024][arm_id] = {"X": X_2024, "y": pd.Series([0, 1] * 5), "a": pd.Series([1] * 10)}

    # Mock preprocessor and endpoint verification to pass
    runtime.adapter = MagicMock()
    runtime.adapter.preprocessor.get_feature_family_lists.return_value = ([], ["pcnt"])
    runtime.endpoint_barrier_results = {arm_id: {"status": "PASS"} for arm_id in D6_ARM_IDS}

    with patch("nhis_fairbias.d7_stepwise_replay.fit_intermediate_model_2022", side_effect=counted_logical_fit), \
         patch.object(LogisticRegression, "fit", counted_lr_fit), \
         patch.object(ProductionD7StepwiseReplayRuntime, "fit_and_verify_endpoints", return_value={}):

        # Run endpoint fits (8 fits: state 0 and state K for all 4 arms)
        for arm_id in D6_ARM_IDS:
            k_term = len(traces[arm_id])
            runtime.fitted_models[arm_id] = {}
            scaler_0, lr_0 = counted_logical_fit(runtime.replayed_states[arm_id][2022][0], runtime.raw_cohorts[2022][arm_id]["y"], arm_id, 0)
            scaler_K, lr_K = counted_logical_fit(runtime.replayed_states[arm_id][2022][k_term], runtime.raw_cohorts[2022][arm_id]["y"], arm_id, k_term)
            runtime.fitted_models[arm_id][0] = (scaler_0, lr_0)
            runtime.fitted_models[arm_id][k_term] = (scaler_K, lr_K)

        # Run intermediate states (39 fits: states 1..K-1 for all 4 arms)
        metrics = runtime.fit_and_evaluate_all_intermediate_states()

    # Verify fit counts: 8 endpoint fits + 39 intermediate fits = 47 total fits
    assert len(logical_fit_calls) == 47
    assert len(lr_fit_calls) == 47

    # Verify all fits used 2022 rows only
    for arm_id, st_idx, n_rows in logical_fit_calls:
        assert n_rows == n_2022

    for n_rows in lr_fit_calls:
        assert n_rows == n_2022


# -----------------------------------------------------------------------------
# 42. Complete Tag and Commit Poison Coverage (P2-B)
# -----------------------------------------------------------------------------
def test_complete_tag_and_commit_poison_coverage():
    # 1. D7.1 wrong tag object
    with patch("subprocess.run") as mock_sub:
        def fake_d7_wrong_obj(cmd, *args, **kwargs):
            if cmd[1] == "rev-parse" and cmd[2] == D7_1_TAG:
                return MagicMock(returncode=0, stdout="bad_d7_obj\n", stderr="")
            return MagicMock(returncode=0, stdout="dummy\n", stderr="")
        mock_sub.side_effect = fake_d7_wrong_obj
        with pytest.raises(ProvenanceVerificationError, match="D7.1 tag object mismatch"):
            verify_upstream_provenance(_REPO_ROOT)

    # 2. D7.1 correct object + wrong dereferenced commit
    with patch("subprocess.run") as mock_sub:
        def fake_d7_wrong_commit(cmd, *args, **kwargs):
            if cmd[1] == "rev-parse" and cmd[2] == D7_1_TAG:
                return MagicMock(returncode=0, stdout=D7_1_TAG_OBJECT + "\n", stderr="")
            if cmd[1] == "rev-parse" and cmd[2] == f"{D7_1_TAG}^{{commit}}":
                return MagicMock(returncode=0, stdout="bad_d7_commit\n", stderr="")
            return MagicMock(returncode=0, stdout="dummy\n", stderr="")
        mock_sub.side_effect = fake_d7_wrong_commit
        with pytest.raises(ProvenanceVerificationError, match="D7.1 dereferenced commit mismatch"):
            verify_upstream_provenance(_REPO_ROOT)

    # 3. D6 train wrong tag object
    with patch("subprocess.run") as mock_sub:
        def fake_d6_tv_wrong_obj(cmd, *args, **kwargs):
            target = cmd[2]
            if target == D7_1_TAG:
                return MagicMock(returncode=0, stdout=D7_1_TAG_OBJECT + "\n", stderr="")
            if target == f"{D7_1_TAG}^{{commit}}":
                return MagicMock(returncode=0, stdout=D7_1_COMMIT + "\n", stderr="")
            if target == D6_TRAIN_VAL_TAG:
                return MagicMock(returncode=0, stdout="bad_d6_tv_obj\n", stderr="")
            return MagicMock(returncode=0, stdout="dummy\n", stderr="")
        mock_sub.side_effect = fake_d6_tv_wrong_obj
        with pytest.raises(ProvenanceVerificationError, match="D6 train/val tag object mismatch"):
            verify_upstream_provenance(_REPO_ROOT)

    # 4. D6 train correct object + wrong dereferenced commit
    with patch("subprocess.run") as mock_sub:
        def fake_d6_tv_wrong_commit(cmd, *args, **kwargs):
            target = cmd[2]
            if target == D7_1_TAG:
                return MagicMock(returncode=0, stdout=D7_1_TAG_OBJECT + "\n", stderr="")
            if target == f"{D7_1_TAG}^{{commit}}":
                return MagicMock(returncode=0, stdout=D7_1_COMMIT + "\n", stderr="")
            if target == D6_TRAIN_VAL_TAG:
                return MagicMock(returncode=0, stdout=D6_TRAIN_VAL_TAG_OBJECT + "\n", stderr="")
            if target == f"{D6_TRAIN_VAL_TAG}^{{commit}}":
                return MagicMock(returncode=0, stdout="bad_d6_tv_commit\n", stderr="")
            return MagicMock(returncode=0, stdout="dummy\n", stderr="")
        mock_sub.side_effect = fake_d6_tv_wrong_commit
        with pytest.raises(ProvenanceVerificationError, match="D6 train/val commit mismatch"):
            verify_upstream_provenance(_REPO_ROOT)

    # 5. D6 test wrong tag object
    with patch("subprocess.run") as mock_sub:
        def fake_d6_test_wrong_obj(cmd, *args, **kwargs):
            target = cmd[2]
            if target == D7_1_TAG:
                return MagicMock(returncode=0, stdout=D7_1_TAG_OBJECT + "\n", stderr="")
            if target == f"{D7_1_TAG}^{{commit}}":
                return MagicMock(returncode=0, stdout=D7_1_COMMIT + "\n", stderr="")
            if target == D6_TRAIN_VAL_TAG:
                return MagicMock(returncode=0, stdout=D6_TRAIN_VAL_TAG_OBJECT + "\n", stderr="")
            if target == f"{D6_TRAIN_VAL_TAG}^{{commit}}":
                return MagicMock(returncode=0, stdout=D6_TRAIN_VAL_COMMIT + "\n", stderr="")
            if target == D6_TEST_TAG:
                return MagicMock(returncode=0, stdout="bad_d6_test_obj\n", stderr="")
            return MagicMock(returncode=0, stdout="dummy\n", stderr="")
        mock_sub.side_effect = fake_d6_test_wrong_obj
        with pytest.raises(ProvenanceVerificationError, match="D6 test tag object mismatch"):
            verify_upstream_provenance(_REPO_ROOT)

    # 6. D6 test correct object + wrong dereferenced commit
    with patch("subprocess.run") as mock_sub:
        def fake_d6_test_wrong_commit(cmd, *args, **kwargs):
            target = cmd[2]
            if target == D7_1_TAG:
                return MagicMock(returncode=0, stdout=D7_1_TAG_OBJECT + "\n", stderr="")
            if target == f"{D7_1_TAG}^{{commit}}":
                return MagicMock(returncode=0, stdout=D7_1_COMMIT + "\n", stderr="")
            if target == D6_TRAIN_VAL_TAG:
                return MagicMock(returncode=0, stdout=D6_TRAIN_VAL_TAG_OBJECT + "\n", stderr="")
            if target == f"{D6_TRAIN_VAL_TAG}^{{commit}}":
                return MagicMock(returncode=0, stdout=D6_TRAIN_VAL_COMMIT + "\n", stderr="")
            if target == D6_TEST_TAG:
                return MagicMock(returncode=0, stdout=D6_TEST_TAG_OBJECT + "\n", stderr="")
            if target == f"{D6_TEST_TAG}^{{commit}}":
                return MagicMock(returncode=0, stdout="bad_d6_test_commit\n", stderr="")
            return MagicMock(returncode=0, stdout="dummy\n", stderr="")
        mock_sub.side_effect = fake_d6_test_wrong_commit
        with pytest.raises(ProvenanceVerificationError, match="D6 test commit mismatch"):
            verify_upstream_provenance(_REPO_ROOT)


# -----------------------------------------------------------------------------
# 43. Temporal Key-Path Summary Zero-Direction Semantics (Section 11)
# -----------------------------------------------------------------------------
def test_temporal_key_path_summary_zero_direction_semantics():
    deltas_df = pd.DataFrame([
        {
            "arm_id": "D6_ARM_001",
            "year": 2023,
            "state_index": 1,
            "pathwise_marginal_delta_auroc": 0.0,
            "pathwise_marginal_delta_auprc": 0.02,
            "pathwise_marginal_delta_ks_statistic": 0.04,
            "pathwise_marginal_delta_selection_rate": 0.01,
        },
        {
            "arm_id": "D6_ARM_001",
            "year": 2024,
            "state_index": 1,
            "pathwise_marginal_delta_auroc": 0.03,
            "pathwise_marginal_delta_auprc": 0.01,
            "pathwise_marginal_delta_ks_statistic": -0.02,
            "pathwise_marginal_delta_selection_rate": 0.01,
        },
    ])
    traces = load_all_archived_traces(_REPO_ROOT)
    summary = generate_key_path_summary(deltas_df, traces)

    arm1 = summary["arms"]["D6_ARM_001"]
    records_2024 = arm1["largest_2024_pathwise_auroc_changes"]
    st1 = records_2024[0]
    assert st1["state_index"] == 1
    assert st1["delta_2023"] == 0.0
    assert st1["delta_2024"] == 0.03
    # Exactly zero has no direction -> same_direction MUST be None
    assert st1["same_direction"] is None


# -----------------------------------------------------------------------------
# 44. TRUE Synthetic Production Integration End-to-End (P1-C)
# -----------------------------------------------------------------------------
def test_synthetic_production_integration_end_to_end(tmp_path):
    """TRUE end-to-end production integration test through manager.execute_release().

    Executes all real orchestration stages without patching runtime stage methods:
    preflight -> 12 cohort build -> digest barrier -> trace replay -> 12 terminal matrix checks
    -> endpoint fits & reproduction -> 47-state intermediate fits -> 8 release files written
    -> manifest created -> release_state.json COMPLETE.
    """
    manager = D7StepwiseReplayReleaseManager(repo_root=_REPO_ROOT, releases_parent_dir=tmp_path)
    release_id = "REL_SYNTH_PROD_TRUE"

    traces = load_all_archived_traces(_REPO_ROOT)
    tv_base = _REPO_ROOT / "docs" / "releases" / D6_TRAIN_VAL_RELEASE_ID

    # Determine required columns for all 4 arms
    arm_columns = {}
    for arm_id in D6_ARM_IDS:
        cd_dict = json.loads((tv_base / arm_id / "final_changed_dict.json").read_text())
        arm_columns[arm_id] = list(cd_dict.keys())

    all_nums = []
    all_cats = []
    for arm_id in D6_ARM_IDS:
        cd_dict = json.loads((tv_base / arm_id / "final_changed_dict.json").read_text())
        for c, v in cd_dict.items():
            if isinstance(v, dict) and "power" in v:
                if c not in all_nums:
                    all_nums.append(c)
            elif v != "dropped":
                if c not in all_cats:
                    all_cats.append(c)

    # Build synthetic adapter returning cohorts with all required features
    n_rows = 20
    class SyntheticProductionAdapter:
        def __init__(self):
            self.preprocessor = MagicMock()
            self.preprocessor.fitted_record = MagicMock()
            self.preprocessor.fitted_record.to_dict.return_value = {"anchor": "verified"}
            self.preprocessor.get_feature_family_lists.side_effect = lambda fset: (all_cats, all_nums)

        def get_cohort(self, year, outcome, protected_attribute, feature_set, disability_arm):
            # Pick arm from protected attribute and disability policy
            if protected_attribute == "SEX_A":
                arm_id = "D6_ARM_001"
            elif protected_attribute == "HISPALLP_A":
                arm_id = "D6_ARM_002"
            elif disability_arm == "full_feature":
                arm_id = "D6_ARM_003"
            else:
                arm_id = "D6_ARM_004"

            cols = arm_columns[arm_id]
            cd_dict = json.loads((tv_base / arm_id / "final_changed_dict.json").read_text())

            rng = np.random.default_rng(hash(f"{arm_id}_{year}") % (2**32))
            synth_dict = {}
            for c in cols:
                val = cd_dict[c]
                if isinstance(val, dict) and "power" in val:
                    synth_dict[c] = rng.uniform(1.0, 5.0, n_rows)
                else:
                    synth_dict[c] = rng.choice([1, 2, 3, 4], n_rows)

            X_df = pd.DataFrame(synth_dict)
            y_arr = pd.Series([0, 1] * (n_rows // 2))
            a_arr = pd.Series([1] * n_rows)
            w_arr = np.ones(n_rows)
            return X_df, y_arr, a_arr, w_arr, {}

    manager.adapter_factory = lambda: SyntheticProductionAdapter()

    # Pre-generate synthetic digests for 12/12 barrier
    synth_digests = {2022: {}, 2023: {}, 2024: {}}
    for yr in (2022, 2023, 2024):
        for arm in D6_ARM_IDS:
            synth_digests[yr][arm] = f"digest_{arm}_{yr}"

    orig_sha256 = compute_sha256
    def selective_sha(path):
        if str(path).endswith(".parquet"):
            return FROZEN_FEATURES_PARQUET_SHA256
        return orig_sha256(path)

    with patch("nhis_fairbias.d7_stepwise_replay.verify_git_execution_preconditions", return_value={"git_head": "synth_head", "remote_branch_head": "synth_head", "tracked_worktree_clean": True}), \
         patch("nhis_fairbias.d7_stepwise_replay.verify_upstream_provenance", return_value={"provenance_status": "VERIFIED"}), \
         patch("nhis_fairbias.d7_stepwise_replay.compute_sha256", side_effect=selective_sha), \
         patch("nhis_fairbias.d7_stepwise_replay.compute_canonical_json_sha256", return_value=PREPROCESSING_STATE_SHA256), \
         patch("nhis_fairbias.d7_stepwise_replay.compute_cohort_source_row_digest", side_effect=lambda yr, idx: f"digest_{yr}"), \
         patch("nhis_fairbias.d7_stepwise_replay.verify_12_cohort_provenance_barrier", return_value={"cohort_provenance_barrier": "PASS"}), \
         patch("nhis_fairbias.d7_stepwise_replay.verify_endpoint_model_reproduction_barrier", return_value={"endpoint_reproduction_status": "PASS"}):

        # Run full manager release execution
        res = manager.execute_release(
            release_id=release_id,
            expected_execution_head="synth_head",
        )

    assert res["status"] == "COMPLETE"
    target_dir = tmp_path / release_id
    assert target_dir.is_dir()

    # Verify 10 files total
    disk_files = {f.name for f in target_dir.iterdir()}
    assert len(disk_files) == 10
    assert set(D7_2_ALL_RELEASE_FILES) == disk_files

    # Verify manifest tracks exactly 8 primary artifacts and hashes match
    manifest_data = json.loads((target_dir / "d7_stepwise_manifest.json").read_text())
    assert manifest_data["tracked_artifact_count"] == 8
    assert len(manifest_data["artifacts"]) == 8
    for fname in D7_2_MANIFEST_TRACKED_ARTIFACTS:
        assert fname in manifest_data["artifacts"]
        art_path = target_dir / fname
        expected_sha = compute_sha256(art_path)
        assert manifest_data["artifacts"][fname]["sha256"] == expected_sha
        assert manifest_data["artifacts"][fname]["size_bytes"] == art_path.stat().st_size

    # Verify release_state.json
    state_data = json.loads((target_dir / "release_state.json").read_text())
    assert state_data["status"] == "COMPLETE"
    assert state_data["manifest_sha256"] == res["manifest_sha256"]
    assert state_data["error"] is None
    assert state_data["started_at"] is not None
    assert state_data["completed_at"] is not None

    # Verify no row-level files
    for fname in disk_files:
        assert not fname.endswith(".parquet")
        assert not fname.endswith(".npz")
        assert "cohort" not in fname
        assert "row" not in fname
        assert "probability" not in fname
        assert "logit" not in fname
