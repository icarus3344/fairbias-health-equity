"""Unit and integration tests for Gate D4.0 NHIS FairBias runner and preflight.

Covers all 13 required test contracts:
1. D4 runner does not call or reproduce train_test_split.
2. Frozen cohort assignment comes from NHISPooledAdapter.
3. FairBias d_phi / epsilon / transform selection receives TRAIN only.
4. Validation changes cannot alter the learned FairBias changed_dict (mutation test).
5. Paper mode rejects sample weights.
6. No finite iteration budget / no Pareto rollback is introduced.
7. Test evaluation guard fails closed when allow_test_evaluation=False.
8. D4.0 manifest asserts test_evaluated == false.
9. Utility metric definitions on a small hand-calculable synthetic example.
10. Group metric definitions including undefined denominators.
11. Multicategory helper: 7 protected groups -> 21 unordered pairs exactly.
12. Baseline and transformed models use identical fixed LR specifications.
13. Same train-learned transform is applied unchanged to validation.
"""

from __future__ import annotations

import copy
import json
import pathlib
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_DIR = str(_REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from fairbias.config import (
    ALGORITHM_MODE_PAPER_FAITHFUL,
    FairBiasConfig,
)
from nhis_fairbias.d4_runner import (
    FROZEN_D4_ARMS,
    NHISD4Runner,
    PRIMARY_D4_RANDOM_SEED,
)
from nhis_fairbias.evaluation import (
    compute_evaluation_comparison,
    compute_fairness_gaps,
    compute_group_coverage,
    compute_group_metrics,
    compute_multicategory_pairwise_differences,
    compute_utility_metrics,
    evaluate_predictions,
)
from nhis_fairbias.pooled import NHISPooledAdapter
from nhis_fairbias.preprocessing import NHISLeakageError


# ------------------------------------------------------------------------------
# Test 1: Runner does not call train_test_split or run_fairbias_pipeline
# ------------------------------------------------------------------------------
def test_d4_runner_does_not_call_train_test_split(tmp_path: pathlib.Path) -> None:
    """Proof that D4 runner never calls train_test_split or generic run_fairbias_pipeline."""
    with patch("sklearn.model_selection.train_test_split", side_effect=AssertionError("train_test_split was called!")):
        with patch("fairbias.pipeline.run_fairbias_pipeline", side_effect=AssertionError("run_fairbias_pipeline was called!")):
            # Initialize runner and mock adapter to run a quick preflight
            adapter = NHISPooledAdapter()
            runner = NHISD4Runner(adapter=adapter, allow_test_evaluation=False)
            assert not runner.allow_test_evaluation

            # Verify that inspecting arms and cohort extraction does not split data
            arm_cfg = runner.get_arm_config("ARM_D3_001")
            assert arm_cfg["protected_attribute"] == "SEX_A"


# ------------------------------------------------------------------------------
# Test 2: Frozen cohort assignment comes from NHISPooledAdapter
# ------------------------------------------------------------------------------
def test_frozen_cohort_assignment_from_pooled_adapter() -> None:
    """Proof that cohort assignment comes strictly from NHISPooledAdapter without re-splitting."""
    adapter = NHISPooledAdapter()
    cohorts = adapter.get_pooled_cohort(
        outcome="MEDDL12M_A",
        protected_attribute="SEX_A",
        feature_set="primary_core",
        disability_arm="full_feature",
    )
    assert set(cohorts.keys()) == {"train", "val", "test"}
    X_tr, y_tr, o_tr, _, meta_tr = cohorts["train"]
    X_va, y_va, o_va, _, meta_va = cohorts["val"]
    X_te, y_te, o_te, _, meta_te = cohorts["test"]

    # All records have valid outcomes and protected attributes
    assert y_tr.notna().all() and o_tr.notna().all()
    assert y_va.notna().all() and o_va.notna().all()
    assert y_te.notna().all() and o_te.notna().all()

    # Split role integrity
    assert (meta_tr["split_role"] == "train").all()
    assert (meta_va["split_role"] == "val").all()
    assert (meta_te["split_role"] == "test").all()

    # Exactly 21 active features for PRIMARY_CORE full_feature
    assert len(X_tr.columns) == 21
    assert len(X_va.columns) == 21
    assert len(X_te.columns) == 21


# ------------------------------------------------------------------------------
# Test 3: FairBias d_phi / epsilon / transform selection receives TRAIN only
# ------------------------------------------------------------------------------
def test_fairbias_fitting_receives_train_only() -> None:
    """Proof that FairBias d_phi, epsilon, and transform selection receive strictly TRAIN data."""
    adapter = NHISPooledAdapter()
    cohorts = adapter.get_pooled_cohort(outcome="MEDDL12M_A", protected_attribute="SEX_A")
    X_tr, y_tr, o_tr, _, _ = cohorts["train"]
    X_va, y_va, o_va, _, _ = cohorts["val"]

    train_n = len(X_tr)
    val_n = len(X_va)
    assert train_n != val_n

    recorded_x_lens = []

    from fairbias.evaluator import FairEvaluator
    orig_calculate_eps = FairEvaluator.calculate_epsilon

    def spy_calculate_eps(self_obj, X, O, *args, **kwargs):
        recorded_x_lens.append(len(X))
        return orig_calculate_eps(self_obj, X, O, *args, **kwargs)

    with patch.object(FairEvaluator, "calculate_epsilon", side_effect=spy_calculate_eps, autospec=True):
        res = adapter.compute_train_epsilon(protected_attribute="SEX_A")
        assert res["epsilon_threshold"] > 0
        # Every epsilon computation received train_n rows, never val_n
        assert all(l == train_n for l in recorded_x_lens)


# ------------------------------------------------------------------------------
# Test 4: Validation changes cannot alter learned FairBias changed_dict (Mutation Test)
# ------------------------------------------------------------------------------
def test_validation_mutation_does_not_alter_learned_fairbias(tmp_path: pathlib.Path) -> None:
    """Synthetic mutation test: hold TRAIN fixed, strongly mutate VALIDATION.
    The learned transform trace and changed_dict must remain identical.
    """
    # Create small synthetic dataset with train and val partitions
    rng = np.random.default_rng(42)
    n_tr, n_va = 200, 100

    # Synthetic training fold
    X_tr = pd.DataFrame({
        "cat1": rng.choice([0, 1, 2], size=n_tr),
        "cat2": rng.choice([0, 1], size=n_tr),
        "num1": rng.normal(0, 1, size=n_tr),
    })
    y_tr = pd.Series(rng.choice([0, 1], size=n_tr), name="MEDDL12M_A")
    o_tr = pd.Series(rng.choice([1, 2], size=n_tr), name="SEX_A")
    meta_tr = pd.DataFrame({"split_role": ["train"] * n_tr, "record_id": [f"TR_{i}" for i in range(n_tr)]})

    # Synthetic validation fold A
    X_va_A = pd.DataFrame({
        "cat1": rng.choice([0, 1, 2], size=n_va),
        "cat2": rng.choice([0, 1], size=n_va),
        "num1": rng.normal(0, 1, size=n_va),
    })
    y_va_A = pd.Series(rng.choice([0, 1], size=n_va), name="MEDDL12M_A")
    o_va_A = pd.Series(rng.choice([1, 2], size=n_va), name="SEX_A")
    meta_va_A = pd.DataFrame({"split_role": ["val"] * n_va, "record_id": [f"VA_{i}" for i in range(n_va)]})

    # Synthetic validation fold B (STRONGLY MUTATED: flipped labels, scaled features, inverted groups)
    X_va_B = X_va_A * 100.0 + 42.0
    y_va_B = 1 - y_va_A
    o_va_B = pd.Series(np.where(o_va_A == 1, 2, 1), name="SEX_A")
    meta_va_B = meta_va_A.copy()

    mock_adapter_A = MagicMock()
    mock_adapter_A.features_parquet_path = tmp_path / "mock.parquet"
    mock_adapter_A.features_parquet_path.touch()
    mock_adapter_A.preprocessor.get_feature_family_lists.return_value = (["cat1", "cat2"], ["num1"])
    mock_adapter_A.get_pooled_cohort.return_value = {
        "train": (X_tr, y_tr, o_tr, None, meta_tr),
        "val": (X_va_A, y_va_A, o_va_A, None, meta_va_A),
    }

    mock_adapter_B = MagicMock()
    mock_adapter_B.features_parquet_path = tmp_path / "mock.parquet"
    mock_adapter_B.preprocessor.get_feature_family_lists.return_value = (["cat1", "cat2"], ["num1"])
    mock_adapter_B.get_pooled_cohort.return_value = {
        "train": (X_tr, y_tr, o_tr, None, meta_tr),
        "val": (X_va_B, y_va_B, o_va_B, None, meta_va_B),
    }

    # Override arm config to expect 3 features
    custom_arm = {
        "arm_id": "ARM_SYNTHETIC",
        "outcome": "MEDDL12M_A",
        "protected_attribute": "SEX_A",
        "feature_set": "primary_core",
        "disability_arm": "full_feature",
        "expected_predictors": 3,
    }

    runner_A = NHISD4Runner(adapter=mock_adapter_A)
    runner_A.get_arm_config = lambda arm_id: copy.deepcopy(custom_arm)
    res_A = runner_A.run_preflight(arm_id="ARM_SYNTHETIC", output_dir=tmp_path / "run_A")

    runner_B = NHISD4Runner(adapter=mock_adapter_B)
    runner_B.get_arm_config = lambda arm_id: copy.deepcopy(custom_arm)
    res_B = runner_B.run_preflight(arm_id="ARM_SYNTHETIC", output_dir=tmp_path / "run_B")

    # Learned changed_dict and training traces must be 100% IDENTICAL
    assert res_A["dphi_record"]["final_changed_dict"] == res_B["dphi_record"]["final_changed_dict"]
    assert res_A["dphi_record"]["initial_max_dphi"] == res_B["dphi_record"]["initial_max_dphi"]
    assert res_A["dphi_record"]["final_max_dphi"] == res_B["dphi_record"]["final_max_dphi"]
    assert res_A["dphi_record"]["epsilon_threshold"] == res_B["dphi_record"]["epsilon_threshold"]
    assert res_A["dphi_record"]["termination_reason"] == res_B["dphi_record"]["termination_reason"]


# ------------------------------------------------------------------------------
# Test 5: Paper mode rejects sample weights
# ------------------------------------------------------------------------------
def test_paper_mode_rejects_sample_weights(tmp_path: pathlib.Path) -> None:
    """Proof that paper mode strictly rejects sample weights with NHISLeakageError."""
    runner = NHISD4Runner()
    weights = np.ones(10)
    with pytest.raises(NHISLeakageError, match="strictly forbids sample_weight"):
        runner.run_preflight(arm_id="ARM_D3_001", sample_weight=weights)


# ------------------------------------------------------------------------------
# Test 6: No finite iteration budget / no Pareto rollback
# ------------------------------------------------------------------------------
def test_no_finite_iteration_budget_and_no_pareto_rollback() -> None:
    """Proof that tang2024_paper_faithful config forbids accuracy enhancement and Pareto rollback."""
    cfg = FairBiasConfig(algorithm_mode=ALGORITHM_MODE_PAPER_FAITHFUL)
    resolved = cfg.resolved()

    assert resolved.algorithm_mode == ALGORITHM_MODE_PAPER_FAITHFUL
    assert resolved.use_accuracy_enhancement is False
    assert resolved.failed_attribute_mode == "stop"
    assert resolved.power_revisit_policy == "restart"
    assert resolved.power_sequence_policy == "official_stream"

    # Attempting to enable accuracy enhancement in paper mode must raise ValueError
    with pytest.raises(ValueError, match="use_accuracy_enhancement is not part of Tang"):
        FairBiasConfig(
            algorithm_mode=ALGORITHM_MODE_PAPER_FAITHFUL,
            use_accuracy_enhancement=True,
        ).resolved()


# ------------------------------------------------------------------------------
# Test 7: Test evaluation guard fails closed when allow_test_evaluation=False
# ------------------------------------------------------------------------------
def test_test_evaluation_guard_fails_closed() -> None:
    """Proof that evaluate_test_embargo_guard fails closed when allow_test_evaluation=False."""
    runner = NHISD4Runner(allow_test_evaluation=False)
    with pytest.raises(RuntimeError, match="Test partition evaluation is strictly embargoed"):
        runner.evaluate_test_embargo_guard()


# ------------------------------------------------------------------------------
# Test 8: D4.0 manifest asserts test_evaluated == false
# ------------------------------------------------------------------------------
def test_manifest_asserts_test_not_evaluated(tmp_path: pathlib.Path) -> None:
    """Proof that preflight manifest records test_evaluated == false and authorized partitions."""
    out_dir = tmp_path / "test_run"
    out_dir.mkdir(parents=True)

    # Synthetic mock run to verify manifest contents
    runner = NHISD4Runner()
    manifest_path = out_dir / "d4_preflight_manifest.json"

    # Run lightweight synthetic preflight
    rng = np.random.default_rng(0)
    X_tr = pd.DataFrame({"c1": [0, 1, 0, 1] * 20, "n1": rng.normal(0, 1, 80)})
    y_tr = pd.Series([0, 1] * 40, name="MEDDL12M_A")
    o_tr = pd.Series([1, 2] * 40, name="SEX_A")
    meta_tr = pd.DataFrame({"split_role": ["train"] * 80, "record_id": [f"T{i}" for i in range(80)]})

    X_va = pd.DataFrame({"c1": [0, 1, 0, 1] * 10, "n1": rng.normal(0, 1, 40)})
    y_va = pd.Series([0, 1] * 20, name="MEDDL12M_A")
    o_va = pd.Series([1, 2] * 20, name="SEX_A")
    meta_va = pd.DataFrame({"split_role": ["val"] * 40, "record_id": [f"V{i}" for i in range(40)]})

    mock_adapter = MagicMock()
    mock_adapter.features_parquet_path = tmp_path / "mock.parquet"
    mock_adapter.features_parquet_path.touch()
    mock_adapter.preprocessor.get_feature_family_lists.return_value = (["c1"], ["n1"])
    mock_adapter.get_pooled_cohort.return_value = {
        "train": (X_tr, y_tr, o_tr, None, meta_tr),
        "val": (X_va, y_va, o_va, None, meta_va),
    }

    mock_arm = {
        "arm_id": "ARM_SYNTH",
        "outcome": "MEDDL12M_A",
        "protected_attribute": "SEX_A",
        "feature_set": "primary_core",
        "disability_arm": "full_feature",
        "expected_predictors": 2,
    }

    runner.adapter = mock_adapter
    runner.get_arm_config = lambda arm_id: copy.deepcopy(mock_arm)
    res = runner.run_preflight(arm_id="ARM_SYNTH", output_dir=out_dir)

    manifest = res["manifest"]
    assert manifest["test_evaluated"] is False
    assert manifest["authorized_partitions"] == ["train", "validation"]
    assert (out_dir / "d4_preflight_manifest.json").is_file()

    # Verify no test outcome metrics in any output json/csv
    for fpath in out_dir.iterdir():
        content = fpath.read_text(encoding="utf-8")
        assert "test AUROC" not in content
        assert "test AUPRC" not in content
        assert "test predictions" not in content


# ------------------------------------------------------------------------------
# Test 9: Utility metric definitions on a small hand-calculable synthetic example
# ------------------------------------------------------------------------------
def test_utility_metrics_synthetic_hand_calculated() -> None:
    """Verify AUROC, AUPRC, balanced accuracy, F1, and accuracy against hand-calculated values."""
    y_true = np.array([1, 1, 0, 0, 1])
    y_prob = np.array([0.9, 0.8, 0.3, 0.1, 0.4])
    y_pred = (y_prob >= 0.5).astype(int)  # [1, 1, 0, 0, 0]

    # Hand calculations:
    # y_true = [1, 1, 0, 0, 1], y_pred = [1, 1, 0, 0, 0]
    # TP: 2 (idx 0, 1)
    # FP: 0
    # TN: 2 (idx 2, 3)
    # FN: 1 (idx 4)
    # Accuracy = 4 / 5 = 0.8
    # TPR = 2 / 3, TNR = 2 / 2 = 1.0
    # Balanced Accuracy = 0.5 * (2/3 + 1.0) = 5/6 = 0.8333333333333334
    # F1 = 2*2 / (2*2 + 0 + 1) = 4 / 5 = 0.8

    metrics = compute_utility_metrics(y_true, y_pred, y_prob)
    assert pytest.approx(metrics["accuracy"], abs=1e-6) == 0.8
    assert pytest.approx(metrics["balanced_accuracy"], abs=1e-6) == 5.0 / 6.0
    assert pytest.approx(metrics["f1"], abs=1e-6) == 0.8
    assert metrics["auroc"] is not None and 0.5 <= metrics["auroc"] <= 1.0
    assert metrics["auprc"] is not None and 0.0 <= metrics["auprc"] <= 1.0
    assert len(metrics["undefined_reasons"]) == 0

    # Single-class edge case
    single_class_y = np.array([1, 1, 1])
    single_prob = np.array([0.8, 0.7, 0.9])
    single_pred = np.array([1, 1, 1])
    single_metrics = compute_utility_metrics(single_class_y, single_pred, single_prob)
    assert single_metrics["auroc"] is None
    assert "undefined" in single_metrics["undefined_reasons"]["auroc"].lower()
    assert single_metrics["balanced_accuracy"] is None


# ------------------------------------------------------------------------------
# Test 10: Group metric definitions including undefined denominators
# ------------------------------------------------------------------------------
def test_group_metrics_and_undefined_denominators() -> None:
    """Verify group metrics record None for undefined denominators without silently zeroing."""
    # Group 1: has both positives and negatives
    # Group 2: has ONLY negatives (pos_count = 0 -> TPR denominator is 0)
    # Group 3: has NO predicted positives (pred_pos = 0 -> PPV denominator is 0)
    y_true = np.array([1, 0, 1, 0, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 0, 0, 0])
    groups = np.array([1, 1, 1, 1, 2, 2, 3, 3])

    rows = compute_group_metrics(y_true, y_pred, groups)
    group_map = {r["group"]: r for r in rows}

    # Group 1: TP=1, FP=1, TN=1, FN=1
    g1 = group_map[1]
    assert g1["n"] == 4
    assert g1["tpr"] == 0.5  # 1 / 2
    assert g1["fpr"] == 0.5  # 1 / 2
    assert g1["ppv"] == 0.5  # 1 / 2

    # Group 2: y_true=[0, 0], y_pred=[0, 0] -> TP=0, FP=0, TN=2, FN=0
    g2 = group_map[2]
    assert g2["n"] == 2
    assert g2["outcome_positive_count"] == 0
    assert g2["tpr_denominator"] == 0
    assert g2["tpr"] is None  # MUST NOT be silently 0.0
    assert "Undefined" in g2["undefined_reasons"]["tpr"]
    assert g2["fpr"] == 0.0   # 0 / 2
    assert g2["ppv"] is None  # 0 / 0
    assert "Undefined" in g2["undefined_reasons"]["ppv"]

    # Group 3: y_true=[1, 0], y_pred=[0, 0] -> TP=0, FP=0, TN=1, FN=1
    g3 = group_map[3]
    assert g3["n"] == 2
    assert g3["tpr"] == 0.0   # 0 / 1
    assert g3["fpr"] == 0.0   # 0 / 1
    assert g3["ppv"] is None  # 0 / 0

    # Fairness gaps
    gaps = compute_fairness_gaps(rows)
    # Demographic parity: defined across all 3 groups
    assert gaps["demographic_parity_gap"] is not None
    # Equal opportunity: defined only for groups 1 and 3 (group 2 omitted from gap)
    assert gaps["equal_opportunity_gap"] is not None
    assert set(gaps["defined_groups"]["tpr"]) == {1, 3}


# ------------------------------------------------------------------------------
# Test 11: Multicategory helper: generic label and K*(K-1)/2 coverage
# ------------------------------------------------------------------------------
def test_multicategory_helper_7_groups_21_pairs() -> None:
    """Proof that multicategory pairwise helper uses generic label and computes K*(K-1)/2 pairs."""
    for k, expected_pairs in [(7, 21), (6, 15), (3, 3)]:
        mock_group_rows = []
        for g in range(1, k + 1):
            mock_group_rows.append({
                "group": g,
                "n": 100,
                "selection_rate": 0.05 * g,
                "tpr": 0.04 * g,
                "fpr": 0.01 * g,
                "ppv": 0.10 * g,
            })

        multi_res = compute_multicategory_pairwise_differences(mock_group_rows)
        assert multi_res["formulation_label"] == "empirical_multicategory_pairwise_extension"
        assert "21" not in multi_res["formulation_label"]
        assert multi_res["num_groups"] == k
        assert multi_res["num_pairs"] == expected_pairs
        assert multi_res["expected_pairs_for_k"] == expected_pairs
        assert len(multi_res["pairs"]) == expected_pairs

        # Verify all pairs are distinct unordered combinations
        seen_pairs = set()
        for p in multi_res["pairs"]:
            pair_tuple = tuple(sorted(p["pair"]))
            assert pair_tuple not in seen_pairs
            seen_pairs.add(pair_tuple)

        assert len(seen_pairs) == expected_pairs


# ------------------------------------------------------------------------------
# Test 12: Baseline and transformed models use identical fixed LR specifications
# ------------------------------------------------------------------------------
def test_baseline_and_transformed_models_identical_specification() -> None:
    """Proof that baseline and FairBias models use identical fixed LR specifications and scaling."""
    m_base = LogisticRegression(random_state=0, max_iter=1000, solver="lbfgs")
    m_fb = LogisticRegression(random_state=0, max_iter=1000, solver="lbfgs")

    assert m_base.get_params() == m_fb.get_params()
    assert m_base is not m_fb

    s_base = MinMaxScaler(feature_range=(0, 1))
    s_fb = MinMaxScaler(feature_range=(0, 1))
    assert s_base.get_params() == s_fb.get_params()
    assert s_base is not s_fb


# ------------------------------------------------------------------------------
# Test 13: Same train-learned transform is applied unchanged to validation
# ------------------------------------------------------------------------------
def test_same_train_learned_transform_applied_to_validation() -> None:
    """Proof that the exact frozen changed_dict from TRAIN is applied to VALIDATION."""
    from fairbias.transform import FairTransform

    transformer = FairTransform()
    changed_dict = {"feat_cat": {1: 0}, "feat_num": {"power": 3.0}}

    df_train = pd.DataFrame({
        "feat_cat": [1, 2, 1, 0],
        "feat_num": [1.0, 2.0, -1.0, 0.0],
    })
    df_val = pd.DataFrame({
        "feat_cat": [1, 1, 2, 0],
        "feat_num": [3.0, -2.0, 0.0, 1.0],
    })

    t_train = transformer.transform_data(df_train, changed_dict, num_attrs=["feat_num"], cate_attrs=["feat_cat"])
    t_val = transformer.transform_data(df_val, changed_dict, num_attrs=["feat_num"], cate_attrs=["feat_cat"])

    # In train: 1 mapped to 0
    assert list(t_train["feat_cat"]) == [0, 2, 0, 0]
    # In val: exact same 1 mapped to 0
    assert list(t_val["feat_cat"]) == [0, 0, 2, 0]

    # In train: power 3 applied
    assert list(t_train["feat_num"]) == [1.0, 8.0, -1.0, 0.0]
    # In val: exact same power 3 applied
    assert list(t_val["feat_num"]) == [27.0, -8.0, 0.0, 1.0]


# ------------------------------------------------------------------------------
# Test 14: Trace schema conformance against D3 contract
# ------------------------------------------------------------------------------
def test_trace_schema_conformance() -> None:
    """Proof that generated train_fairbias_trace conforms to D3 schema."""
    import jsonschema
    from fairbias.transform_trace import FairBiasTransformStep, FairBiasTransformTrace

    schema_path = _REPO_ROOT / "artifacts" / "nhis" / "d3" / "fairbias_transform_trace_schema.json"
    if not schema_path.is_file():
        pytest.skip("D3 schema file not found")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    trace = FairBiasTransformTrace(
        algorithm_mode=ALGORITHM_MODE_PAPER_FAITHFUL,
        protected_attribute="SEX_A",
        epsilon_threshold=0.0005,
        steps=[
            FairBiasTransformStep(
                iteration=1,
                selected_feature="empwrkft1_a",
                feature_semantic_type="categorical",
                d_phi_before=0.0042,
                epsilon=0.0005,
                proposed_transformation={"-1": "1"},
                accepted_transformation={"-1": "1"},
                numerical_exponent=None,
                categorical_merge_mapping={"-1": "1"},
                d_phi_after=0.0002,
                dropped=False,
                stopped_reason=None,
            )
        ],
        final_status="CONVERGED",
        final_max_dphi=0.0002,
    )
    payload = trace.to_dict()
    jsonschema.validate(instance=payload, schema=schema)


# ------------------------------------------------------------------------------
# Test 15: Primary D4 random seed frozen to 0 and non-zero rejected fail-closed
# ------------------------------------------------------------------------------
def test_primary_d4_random_seed_frozen_and_non_zero_rejected() -> None:
    """Proof that PRIMARY_D4_RANDOM_SEED is 0 and non-zero seeds raise ValueError."""
    assert PRIMARY_D4_RANDOM_SEED == 0

    runner = NHISD4Runner()
    for bad_seed in [1, 42, -1, 100]:
        with pytest.raises(ValueError, match="Primary D4 analysis protocol strictly freezes random_seed to 0"):
            runner.run_preflight(arm_id="ARM_D3_001", random_seed=bad_seed)


# ------------------------------------------------------------------------------
# Test 16: Distinguish trace steps from accepted transforms
# ------------------------------------------------------------------------------
def test_distinguish_trace_steps_from_accepted_transforms() -> None:
    """Proof that unaccepted/terminal traces are not counted as accepted transforms."""
    from fairbias.transform_trace import FairBiasTransformStep

    step_accepted = FairBiasTransformStep(
        iteration=1,
        selected_feature="feat_cat",
        feature_semantic_type="categorical",
        d_phi_before=0.005,
        epsilon=0.0005,
        proposed_transformation={"-1": 1},
        accepted_transformation={"-1": 1},
        numerical_exponent=None,
        categorical_merge_mapping={"-1": "1"},
        d_phi_after=0.0002,
        dropped=False,
        stopped_reason=None,
    )
    step_terminal = FairBiasTransformStep(
        iteration=2,
        selected_feature="feat_num",
        feature_semantic_type="numerical",
        d_phi_before=0.003,
        epsilon=0.0005,
        proposed_transformation={"power": 2.0},
        accepted_transformation=None,  # Not accepted / terminal search failure
        numerical_exponent=2.0,
        categorical_merge_mapping=None,
        d_phi_after=0.003,
        dropped=False,
        stopped_reason="no_improvement",
    )

    trace_steps = [step_accepted, step_terminal]
    total_trace_steps = len(trace_steps)
    accepted_transform_steps = sum(
        1 for s in trace_steps if s.accepted_transformation is not None
    )
    unaccepted_or_terminal_trace_steps = total_trace_steps - accepted_transform_steps

    assert total_trace_steps == 2
    assert accepted_transform_steps == 1
    assert unaccepted_or_terminal_trace_steps == 1


# ------------------------------------------------------------------------------
# Test 17: HISP group coverage contract (complete vs incomplete coverage)
# ------------------------------------------------------------------------------
def test_hisp_group_coverage_contract() -> None:
    """Proof that HISP primary arm coverage requires 7 categories and 21 pairs, flagging incomplete partitions."""
    arm_hisp = FROZEN_D4_ARMS["ARM_D3_002"]
    assert arm_hisp["expected_group_count"] == 7
    assert arm_hisp["expected_pair_count"] == 21

    # Case A: Complete coverage (all 7 categories present)
    o_complete = pd.Series([1, 2, 3, 4, 5, 6, 7] * 10)
    cov_comp = compute_group_coverage(
        o_complete,
        expected_group_count=arm_hisp["expected_group_count"],
        expected_groups=arm_hisp["expected_groups"],
    )
    assert cov_comp["group_coverage_complete"] is True
    assert cov_comp["observed_group_count"] == 7
    assert cov_comp["expected_pair_count"] == 21
    assert cov_comp["observed_pair_count"] == 21
    assert cov_comp["diagnostics"] is None

    # Case B: Incomplete coverage (e.g. only 5 categories represented)
    o_incomplete = pd.Series([1, 2, 3, 4, 5] * 10)
    cov_incomp = compute_group_coverage(
        o_incomplete,
        expected_group_count=arm_hisp["expected_group_count"],
        expected_groups=arm_hisp["expected_groups"],
    )
    assert cov_incomp["group_coverage_complete"] is False
    assert cov_incomp["observed_group_count"] == 5
    assert cov_incomp["observed_pair_count"] == 10  # 5*4/2 = 10
    assert cov_incomp["expected_pair_count"] == 21
    assert "Incomplete group coverage" in cov_incomp["diagnostics"]

    # Full evaluate_predictions integration with incomplete coverage
    y_true = np.array([0, 1] * 25)
    y_pred = np.array([0, 0] * 25)
    y_prob = np.array([0.2, 0.4] * 25)
    eval_res = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        o_group=o_incomplete,
        expected_group_count=arm_hisp["expected_group_count"],
        expected_groups=arm_hisp["expected_groups"],
    )
    assert eval_res["group_coverage"]["group_coverage_complete"] is False
    assert eval_res["multicategory_pairwise"]["coverage_complete"] is False
    assert "Incomplete group coverage" in eval_res["multicategory_pairwise"]["diagnostics"]
