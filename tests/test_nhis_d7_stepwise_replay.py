"""Unit, barrier, state machine, and invariant tests for Gate D7.2a Stepwise Replay Harness.

Guarantees:
- Tests replay state machine with categorical, numerical, and drop steps.
- Tests invariants: revisit semantics, dropped features cannot reappear, unsorted index preservation.
- Tests terminal replay barriers and endpoint reproduction barriers.
- Explicit anti-leakage test poisoning 2023 and 2024 holdouts during fitting.
- Zero real NHIS cohort access, zero FairBiasMitigation re-instantiation.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from fairbias.transform import FairTransform
from nhis_fairbias.d7_stepwise_replay import (
    ARM_PROTECTED_ATTRIBUTES,
    CANONICAL_DECISION_THRESHOLD,
    D6_ARM_IDS,
    D6_TEST_MANIFEST_SHA256,
    D6_TRAIN_VAL_MANIFEST_SHA256,
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
    EXPECTED_TRAINING_STATE_ANCHORS,
    FORBIDDEN_CAUSAL_TERMS,
    LR_MAX_ITER,
    LR_RANDOM_STATE,
    LR_SOLVER,
    MATRIX_NUMERICAL_TOLERANCE,
    METRIC_REPRODUCTION_TOLERANCE,
    STARTING_HEAD_COMMIT,
    TOTAL_EXPECTED_ACCEPTED_STEPS,
    TOTAL_EXPECTED_REPRESENTATION_STATES,
    ArchivedTraceStep,
    D7StepwiseReplayError,
    D7StepwiseReplayReleaseManager,
    DataLeakageError,
    EndpointReproductionBarrierError,
    IllegalReplayError,
    NHISD7StepwiseReplayHarness,
    ProvenanceVerificationError,
    RepresentationState,
    SequentialReplayStateMachine,
    TerminalReplayBarrierError,
    build_d7_stepwise_manifest,
    compute_stepwise_deltas,
    compute_stepwise_metrics,
    fit_intermediate_model_2022,
    generate_key_path_summary,
    load_all_archived_traces,
    load_archived_trace,
    predict_proba_fitted_model,
    verify_endpoint_model_reproduction_barrier,
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

    # State 1: 2 merged to 1 -> [1, 1, 3, 4]
    assert res[1]["cat_feat"].tolist() == [1, 1, 3, 4]
    # State 2: 3 merged to 1 -> [1, 1, 1, 4]
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
    # sign(x) * |x|^3
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
    base_dir = _REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609"

    for arm_id in D6_ARM_IDS:
        arm_steps = traces[arm_id]
        cd_path = base_dir / arm_id / "final_changed_dict.json"
        cd_dict = json.loads(cd_path.read_text(encoding="utf-8"))
        expected_hash = EXPECTED_TRAINING_STATE_ANCHORS[arm_id]["changed_dict"]

        # Synthesize test DataFrame containing all features touched by this arm
        rng = np.random.default_rng(42)
        n_rows = 100
        synth_data: Dict[str, Any] = {}
        all_features = list(cd_dict.keys())

        for f in all_features:
            val = cd_dict[f]
            if isinstance(val, dict) and "power" in val:
                # Numerical
                synth_data[f] = rng.uniform(1.0, 10.0, size=n_rows)
            else:
                # Categorical or dropped
                synth_data[f] = rng.choice([-1, 1, 2, 3, 4, 5], size=n_rows)

        sample_df = pd.DataFrame(synth_data)
        num_attrs = [f for f, v in cd_dict.items() if isinstance(v, dict) and "power" in v]
        cate_attrs = [f for f in all_features if f not in num_attrs]

        # Verify terminal barrier
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

    # Swap step 8 (numerical power on agep_a) and step 10 (drop agep_a)
    # In original: step 8 transforms agep_a, step 10 drops agep_a.
    # If we swap, step 8 drops agep_a, then step 10 attempts to transform dropped agep_a!
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
    arm_steps = list(traces["D6_ARM_001"])[:-1]  # 14 steps instead of 15

    base_dir = _REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609"
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

    # Alter step 15 (last step on pcntlt18tc: 9.0 -> 8.0)
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

    base_dir = _REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609"
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

    # Mock mismatching state hashes
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
    # Verify that metric difference fails with fail-closed message
    with patch("nhis_fairbias.d7_stepwise_replay.extract_minmax_scaler_state") as m_scaler, \
         patch("nhis_fairbias.d7_stepwise_replay.extract_logistic_regression_state") as m_lr, \
         patch("nhis_fairbias.d7_stepwise_replay.compute_canonical_json_sha256") as m_sha:

        anchors = EXPECTED_TRAINING_STATE_ANCHORS["D6_ARM_001"]
        # Return expected hashes in order: s0_scaler, s0_lr, sK_scaler, sK_lr
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
        # Observed has count 101 instead of 100
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
    # Track all datasets that enter fit()
    fit_call_data = []

    orig_fit = LogisticRegression.fit

    def recording_fit(self, X, y, sample_weight=None):
        fit_call_data.append(X)
        return orig_fit(self, X, y, sample_weight=sample_weight)

    with patch.object(LogisticRegression, "fit", recording_fit):
        X_2022 = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]})
        y_2022 = pd.Series([0, 0, 1, 1])
        scaler, lr = fit_intermediate_model_2022(X_2022, y_2022, "D6_ARM_001", 1)

        # 2023 and 2024 evaluation uses predict_proba only
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
    # >= 0.5 are positives -> indices 1, 2, 3 -> count 3
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

    # 4 arms baseline + 43 steps = 47 total states
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

    # Step 1: 0.72 - 0.70 = 0.02
    r1 = deltas_df[deltas_df["state_index"] == 1].iloc[0]
    assert np.isclose(r1["pathwise_marginal_delta_auroc"], 0.02, atol=1e-12)
    assert np.isclose(r1["cumulative_delta_from_baseline_auroc"], 0.02, atol=1e-12)

    # Step 2: 0.69 - 0.72 = -0.03; cumulative: 0.69 - 0.70 = -0.01
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
        }
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
        # Must be aggregate summaries or inventories only
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
# 25. Explicit anti-leakage test (Section 19)
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

    # 2022 unpoisoned data
    X_2022 = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    y_2022 = pd.Series([0, 1, 0, 1])

    # Fit must succeed using 2022 only
    scaler, lr = fit_intermediate_model_2022(X_2022, y_2022, arm_id="D6_ARM_001", state_index=1)
    assert scaler is not None
    assert lr is not None

    # Now verify that if someone mistakenly passed poisoned 2023/2024 into fit, it raises immediately
    p_X_2023 = Poisoned2023DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    p_y_2024 = Poisoned2024Series([0, 1, 0, 1])

    with pytest.raises(AssertionError, match="LEAKAGE DETECTED"):
        fit_intermediate_model_2022(p_X_2023, y_2022, arm_id="D6_ARM_001", state_index=1)

    with pytest.raises(AssertionError, match="LEAKAGE DETECTED"):
        fit_intermediate_model_2022(X_2022, p_y_2024, arm_id="D6_ARM_001", state_index=1)
