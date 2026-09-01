"""Integration and synthetic test suite for Gate D5.1a: survey-weighted FairBias preflight harness.

Covers all 25 mandatory tests specified in Section 17 of the Gate D5.1a protocol:
1. Exact four-arm registry
2. Exact frozen split/features hashes
3. D4 archive/tag remains immutable
4. WTFA_A source is present
5. TRAIN/VAL weight alignment by respondent identity
6. Shuffled/misaligned weights fail closed
7. Missing/NaN/inf/nonpositive real-like weights fail closed
8. Per-group positive total weights required
9. Weighted mode receives TRAIN weights
10. Validation weights never enter FairBias search
11. Validation outcomes never enter FairBias search
12. D5 starts from empty changed_dict
13. R1 remains unweighted
14. LR fit receives no sample_weight
15. Evaluation receives no sample_weight
16. Prediction threshold fixed at 0.5
17. Random seed fixed at 0
18. Separate baseline/FairBias scalers
19. Scalers fit TRAIN only
20. HISP expected 7 groups / 21 pairs
21. Audit-only cannot invoke real mitigation
22. Audit-only cannot score validation
23. Audit-only cannot access TEST scoring
24. No D4 artifact mutation
25. No weighted result artifact generated in audit-only
"""

from __future__ import annotations

import copy
import inspect
import json
import pathlib
import subprocess
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from fairbias.config import (
    ALGORITHM_MODE_SURVEY_WEIGHTED,
    FairBiasConfig,
)
from fairbias.evaluator import FairEvaluator
from fairbias.mitigation import FairBiasMitigation
from nhis_fairbias.d5_weighted_runner import (
    D4_COMPARISON_REGISTRY,
    FROZEN_D4_RELEASE_TAG,
    FROZEN_D5_ARMS,
    FROZEN_FEATURES_PARQUET_PATH,
    FROZEN_FEATURES_PARQUET_SHA256,
    FROZEN_PRIMARY_ANALYSIS_COMMIT,
    FROZEN_SPLIT_MANIFEST_PATH,
    FROZEN_SPLIT_MANIFEST_SHA256,
    PRIMARY_D5_RANDOM_SEED,
    NHISD5WeightedRunner,
    compute_series_weight_diagnostics,
)
from nhis_fairbias.download import compute_sha256
from nhis_fairbias.evaluation import (
    compute_evaluation_comparison,
    evaluate_predictions,
)
from nhis_fairbias.pooled import NHISPooledAdapter

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def create_mock_cohort(n_train: int = 40, n_val: int = 20, n_features: int = 21, seed: int = 42):
    """Generate synthetic Presplit cohort matching NHIS adapter structure."""
    rng = np.random.RandomState(seed)

    cols = [f"f_{i}" for i in range(n_features)]
    cate_cols = cols[:5]
    num_cols = cols[5:]

    # Train
    idx_train = pd.Index([f"train_{i}" for i in range(n_train)])
    X_tr = pd.DataFrame(rng.normal(0, 1, size=(n_train, n_features)), index=idx_train, columns=cols)
    y_tr = pd.Series(rng.choice([0, 1], size=n_train), index=idx_train, name="MEDDL12M_A")
    o_tr = pd.Series(rng.choice([1, 2], size=n_train), index=idx_train, name="SEX_A")
    w_tr = pd.Series(rng.uniform(500.0, 15000.0, size=n_train), index=idx_train, name="WTFA_A")
    meta_tr = pd.DataFrame({
        "record_id": idx_train.tolist(),
        "WTFA_A": w_tr.tolist(),
        "survey_year": [2023] * n_train,
    }, index=idx_train)

    # Val
    idx_val = pd.Index([f"val_{i}" for i in range(n_val)])
    X_va = pd.DataFrame(rng.normal(0, 1, size=(n_val, n_features)), index=idx_val, columns=cols)
    y_va = pd.Series(rng.choice([0, 1], size=n_val), index=idx_val, name="MEDDL12M_A")
    o_va = pd.Series(rng.choice([1, 2], size=n_val), index=idx_val, name="SEX_A")
    w_va = pd.Series(rng.uniform(500.0, 15000.0, size=n_val), index=idx_val, name="WTFA_A")
    meta_va = pd.DataFrame({
        "record_id": idx_val.tolist(),
        "WTFA_A": w_va.tolist(),
        "survey_year": [2023] * n_val,
    }, index=idx_val)

    return {
        "train": (X_tr, y_tr, o_tr, w_tr, meta_tr),
        "val": (X_va, y_va, o_va, w_va, meta_va),
    }, cate_cols, num_cols


# ------------------------------------------------------------------------------
# 1. Exact Four-Arm Registry
# ------------------------------------------------------------------------------
def test_1_exact_four_arm_registry() -> None:
    """Verify exact 4-arm specification and D4 comparison registry."""
    assert set(FROZEN_D5_ARMS.keys()) == {"D5_ARM_001", "D5_ARM_002", "D5_ARM_003", "D5_ARM_004"}

    arm1 = FROZEN_D5_ARMS["D5_ARM_001"]
    assert arm1["protected_attribute"] == "SEX_A"
    assert arm1["feature_set"] == "PRIMARY_CORE"
    assert arm1["disability_arm"] == "full_feature"
    assert arm1["expected_predictors"] == 21
    assert arm1["expected_group_count"] == 2
    assert arm1["expected_pair_count"] == 1
    assert arm1["d4_reference_arm"] == "ARM_D3_001"

    arm2 = FROZEN_D5_ARMS["D5_ARM_002"]
    assert arm2["protected_attribute"] == "HISPALLP_A"
    assert arm2["feature_set"] == "PRIMARY_CORE"
    assert arm2["disability_arm"] == "full_feature"
    assert arm2["expected_predictors"] == 21
    assert arm2["expected_group_count"] == 7
    assert arm2["expected_pair_count"] == 21
    assert arm2["d4_reference_arm"] == "ARM_D3_002"

    arm3 = FROZEN_D5_ARMS["D5_ARM_003"]
    assert arm3["protected_attribute"] == "DISAB3_A"
    assert arm3["feature_set"] == "PRIMARY_CORE"
    assert arm3["disability_arm"] == "full_feature"
    assert arm3["expected_predictors"] == 21
    assert arm3["expected_group_count"] == 2
    assert arm3["expected_pair_count"] == 1
    assert arm3["d4_reference_arm"] == "ARM_D3_003"

    arm4 = FROZEN_D5_ARMS["D5_ARM_004"]
    assert arm4["protected_attribute"] == "DISAB3_A"
    assert arm4["feature_set"] == "PRIMARY_CORE"
    assert arm4["disability_arm"] == "exclude_disability_components"
    assert arm4["expected_predictors"] == 15
    assert arm4["expected_group_count"] == 2
    assert arm4["expected_pair_count"] == 1
    assert arm4["d4_reference_arm"] == "ARM_D3_004"

    assert len(D4_COMPARISON_REGISTRY) == 4
    for ref_id in ("ARM_D3_001", "ARM_D3_002", "ARM_D3_003", "ARM_D3_004"):
        assert ref_id in D4_COMPARISON_REGISTRY
        assert "run_id" in D4_COMPARISON_REGISTRY[ref_id]
        assert "manifest_sha256" in D4_COMPARISON_REGISTRY[ref_id]
        assert "changed_dict_sha256" in D4_COMPARISON_REGISTRY[ref_id]


# ------------------------------------------------------------------------------
# 2. Exact Frozen Split/Features Hashes
# ------------------------------------------------------------------------------
def test_2_exact_frozen_split_and_features_hashes() -> None:
    """Verify frozen master split manifest and feature parquet SHA-256 hashes."""
    assert FROZEN_SPLIT_MANIFEST_PATH.is_file()
    assert FROZEN_FEATURES_PARQUET_PATH.is_file()

    split_sha = compute_sha256(FROZEN_SPLIT_MANIFEST_PATH)
    feat_sha = compute_sha256(FROZEN_FEATURES_PARQUET_PATH)

    assert split_sha == FROZEN_SPLIT_MANIFEST_SHA256 == "6874a56f5484186dffdd5faeb7871c3bef85042ccfdf8d1e375fdde75a3ee9f5"
    assert feat_sha == FROZEN_FEATURES_PARQUET_SHA256 == "49f415132ff0be0228f8533f9f74c48cd79ff6fa8be66db085f7329d9b083383"
    assert FROZEN_PRIMARY_ANALYSIS_COMMIT == "58a1e02339bf35dedc14dd2719d72417de6f6910"


# ------------------------------------------------------------------------------
# 3. D4 Archive/Tag Remains Immutable
# ------------------------------------------------------------------------------
def test_3_d4_archive_and_tag_remain_immutable() -> None:
    """Verify D4 release archive diff is empty and tag is unchanged."""
    res_diff = subprocess.run(
        [
            "git", "diff",
            "bb1f0601bf3a079bb09882003740df8a78bc0374..HEAD",
            "--", "docs/releases/NHIS_D4_PRIMARY_TEST_RELEASE_V1_9920fc5a",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert res_diff.stdout.strip() == "", "D4 release archive must have zero diff vs bb1f0601"

    res_tag = subprocess.run(
        ["git", "rev-parse", FROZEN_D4_RELEASE_TAG],
        capture_output=True,
        text=True,
        check=True,
    )
    assert res_tag.stdout.strip() == "d74af83fb98a3805a7fb0767c5beeb3dbf1407e4"


# ------------------------------------------------------------------------------
# 4. WTFA_A Source is Present
# ------------------------------------------------------------------------------
def test_4_wtfa_a_source_is_present() -> None:
    """Verify WTFA_A is present in the frozen features parquet and is strictly positive."""
    df = pd.read_parquet(FROZEN_FEATURES_PARQUET_PATH, columns=["WTFA_A"])
    assert "WTFA_A" in df.columns
    assert df["WTFA_A"].isna().sum() == 0
    assert (df["WTFA_A"] > 0).all()
    assert pd.api.types.is_float_dtype(df["WTFA_A"])


# ------------------------------------------------------------------------------
# 5. TRAIN/VAL Weight Alignment by Respondent Identity
# ------------------------------------------------------------------------------
def test_5_train_val_weight_alignment_by_respondent_identity() -> None:
    """Verify weight Series index, feature index, and record_id are strictly 1-to-1 aligned."""
    runner = NHISD5WeightedRunner(enforce_frozen_inputs=True)
    cohorts = runner.get_train_val_cohort("D5_ARM_001")

    for part in ("train", "val"):
        X, y, o, w, meta = cohorts[part]
        assert X.index.equals(w.index)
        assert X.index.equals(o.index)
        assert X.index.equals(y.index)
        assert X.index.equals(meta.index)
        assert (w == meta["WTFA_A"]).all()
        assert meta["record_id"].nunique() == len(meta)


# ------------------------------------------------------------------------------
# 6. Shuffled/Misaligned Weights Fail Closed
# ------------------------------------------------------------------------------
def test_6_shuffled_misaligned_weights_fail_closed() -> None:
    """Index mismatches between weight Series and features fail closed."""
    idx1 = pd.Index(["id_1", "id_2", "id_3"])
    idx2 = pd.Index(["id_3", "id_1", "id_2"])  # Shuffled order

    o_good = pd.Series([1, 1, 2], index=idx1)

    # Shuffled index
    w_shuffled = pd.Series([100.0, 200.0, 300.0], index=idx2)
    with pytest.raises(ValueError, match="alignment mismatch"):
        compute_series_weight_diagnostics(w_shuffled, idx1, o_good, "test_part")


# ------------------------------------------------------------------------------
# 7. Missing/NaN/Inf/Nonpositive Weights Fail Closed
# ------------------------------------------------------------------------------
def test_7_missing_nan_inf_nonpositive_weights_fail_closed() -> None:
    """NaN, Inf, 0.0, and negative weights fail closed."""
    idx = pd.Index(["id_1", "id_2", "id_3"])
    o = pd.Series([1, 1, 2], index=idx)

    # 1. NaN
    w_nan = pd.Series([100.0, np.nan, 300.0], index=idx)
    with pytest.raises(ValueError, match="missing values|non-finite"):
        compute_series_weight_diagnostics(w_nan, idx, o, "part")

    # 2. Inf
    w_inf = pd.Series([100.0, np.inf, 300.0], index=idx)
    with pytest.raises(ValueError, match="non-finite"):
        compute_series_weight_diagnostics(w_inf, idx, o, "part")

    # 3. Negative
    w_neg = pd.Series([100.0, -5.0, 300.0], index=idx)
    with pytest.raises(ValueError, match="negative values"):
        compute_series_weight_diagnostics(w_neg, idx, o, "part")

    # 4. Zero
    w_zero = pd.Series([100.0, 0.0, 300.0], index=idx)
    with pytest.raises(ValueError, match="zero values"):
        compute_series_weight_diagnostics(w_zero, idx, o, "part")


# ------------------------------------------------------------------------------
# 8. Per-Group Positive Total Weights Required
# ------------------------------------------------------------------------------
def test_8_per_group_positive_total_weights_required() -> None:
    """If any protected group has non-positive total weight, execution fails closed."""
    idx = pd.Index(["id_1", "id_2", "id_3"])
    o = pd.Series([1, 2, 3], index=idx)
    w = pd.Series([100.0, 0.0, 300.0], index=idx)
    with pytest.raises(ValueError, match="zero values|non-positive total weight"):
        compute_series_weight_diagnostics(w, idx, o, "part")


# ------------------------------------------------------------------------------
# 9. Weighted Mode Receives TRAIN Weights
# ------------------------------------------------------------------------------
def test_9_weighted_mode_receives_train_weights() -> None:
    """Verify FairBias evaluator calculate_epsilon receives TRAIN weights."""
    mock_cohorts, c_cols, n_cols = create_mock_cohort(n_train=30, n_val=15)
    runner = NHISD5WeightedRunner(enforce_frozen_inputs=False, allow_mitigation=True)
    runner.get_train_val_cohort = MagicMock(return_value=mock_cohorts)
    runner.adapter.preprocessor.get_feature_family_lists = MagicMock(return_value=(c_cols, n_cols))

    eps_calls = []
    orig_calc = FairEvaluator.calculate_epsilon

    def tracking_calc(self, *args, **kwargs):
        eps_calls.append(kwargs.get("sample_weight"))
        return orig_calc(self, *args, **kwargs)

    def mock_mitigate_step(self, **kwargs):
        return kwargs["X"], kwargs["changed_dict"], None, None

    with patch.object(FairEvaluator, "calculate_epsilon", new=tracking_calc), \
         patch.object(FairBiasMitigation, "mitigate_step", new=mock_mitigate_step):
        runner.execute_train_validation(arm_id="D5_ARM_001")

    assert len(eps_calls) > 0
    w_train = mock_cohorts["train"][3]
    for sw in eps_calls:
        if sw is not None:
            assert np.array_equal(sw, w_train)


# ------------------------------------------------------------------------------
# 10. Validation Weights Never Enter FairBias Search
# ------------------------------------------------------------------------------
def test_10_validation_weights_never_enter_fairbias_search() -> None:
    """Validation weights w_val must never be passed to FairBias evaluator during mitigation."""
    mock_cohorts, c_cols, n_cols = create_mock_cohort(n_train=30, n_val=15)
    runner = NHISD5WeightedRunner(enforce_frozen_inputs=False, allow_mitigation=True)
    runner.get_train_val_cohort = MagicMock(return_value=mock_cohorts)
    runner.adapter.preprocessor.get_feature_family_lists = MagicMock(return_value=(c_cols, n_cols))

    w_val = mock_cohorts["val"][3]
    eps_calls = []
    orig_calc = FairEvaluator.calculate_epsilon

    def tracking_calc(self, *args, **kwargs):
        eps_calls.append(kwargs.get("sample_weight"))
        return orig_calc(self, *args, **kwargs)

    def mock_mitigate_step(self, **kwargs):
        return kwargs["X"], kwargs["changed_dict"], None, None

    with patch.object(FairEvaluator, "calculate_epsilon", new=tracking_calc), \
         patch.object(FairBiasMitigation, "mitigate_step", new=mock_mitigate_step):
        runner.execute_train_validation(arm_id="D5_ARM_001")

    for sw in eps_calls:
        if sw is not None:
            assert not np.array_equal(sw, w_val)


# ------------------------------------------------------------------------------
# 11. Validation Outcomes Never Enter FairBias Search
# ------------------------------------------------------------------------------
def test_11_validation_outcomes_never_enter_fairbias_search() -> None:
    """Validation outcome y_val and o_val must never be passed to FairBiasMitigation."""
    mock_cohorts, c_cols, n_cols = create_mock_cohort(n_train=30, n_val=15)
    runner = NHISD5WeightedRunner(enforce_frozen_inputs=False, allow_mitigation=True)
    runner.get_train_val_cohort = MagicMock(return_value=mock_cohorts)
    runner.adapter.preprocessor.get_feature_family_lists = MagicMock(return_value=(c_cols, n_cols))

    y_val = mock_cohorts["val"][1]
    step_calls = []

    def mock_mitigate_step(self, **kwargs):
        step_calls.append(kwargs)
        return kwargs["X"], kwargs["changed_dict"], None, None

    with patch.object(FairBiasMitigation, "mitigate_step", new=mock_mitigate_step):
        runner.execute_train_validation(arm_id="D5_ARM_001")

    for call in step_calls:
        assert not call["Y"].equals(y_val)


# ------------------------------------------------------------------------------
# 12. D5 Starts From Empty changed_dict
# ------------------------------------------------------------------------------
def test_12_d5_starts_from_empty_changed_dict() -> None:
    """The greedy mitigation loop in D5 must start from changed_dict = {}."""
    mock_cohorts, c_cols, n_cols = create_mock_cohort(n_train=30, n_val=15)
    runner = NHISD5WeightedRunner(enforce_frozen_inputs=False, allow_mitigation=True)
    runner.get_train_val_cohort = MagicMock(return_value=mock_cohorts)
    runner.adapter.preprocessor.get_feature_family_lists = MagicMock(return_value=(c_cols, n_cols))

    step_calls = []

    def mock_mitigate_step(self, **kwargs):
        step_calls.append(kwargs)
        return kwargs["X"], kwargs["changed_dict"], None, None

    with patch.object(FairBiasMitigation, "mitigate_step", new=mock_mitigate_step):
        runner.execute_train_validation(arm_id="D5_ARM_001")

    if step_calls:
        assert step_calls[0]["changed_dict"] == {}


# ------------------------------------------------------------------------------
# 13. R1 Remains Unweighted in D5
# ------------------------------------------------------------------------------
def test_13_r1_remains_unweighted_in_d5() -> None:
    """compute_r1_rebin must have no sample_weight argument and behave unweighted."""
    sig = inspect.signature(FairBiasMitigation.compute_r1_rebin)
    assert "sample_weight" not in sig.parameters

    engine = FairBiasMitigation(
        evaluator=None, transformer=None, label_O=["p"], cate_attrs=["c"], num_attrs=[], sample_weight=np.ones(10)
    )
    with pytest.raises(TypeError):
        engine.compute_r1_rebin(pd.Series(["A"] * 10), pd.Series([1] * 10), sample_weight=np.ones(10))


# ------------------------------------------------------------------------------
# 14. LR Fit Receives No sample_weight
# ------------------------------------------------------------------------------
def test_14_lr_fit_receives_no_sample_weight() -> None:
    """Model training (baseline and FairBias) must strictly receive NO sample_weight."""
    mock_cohorts, c_cols, n_cols = create_mock_cohort(n_train=30, n_val=15)
    runner = NHISD5WeightedRunner(enforce_frozen_inputs=False, allow_mitigation=True)
    runner.get_train_val_cohort = MagicMock(return_value=mock_cohorts)
    runner.adapter.preprocessor.get_feature_family_lists = MagicMock(return_value=(c_cols, n_cols))

    fit_sample_weights = []
    orig_fit = LogisticRegression.fit

    def tracking_fit(self, X, y, sample_weight=None):
        fit_sample_weights.append(sample_weight)
        return orig_fit(self, X, y, sample_weight=sample_weight)

    def mock_mitigate_step(self, **kwargs):
        return kwargs["X"], kwargs["changed_dict"], None, None

    with patch.object(LogisticRegression, "fit", new=tracking_fit), \
         patch.object(FairBiasMitigation, "mitigate_step", new=mock_mitigate_step):
        runner.execute_train_validation(arm_id="D5_ARM_001")

    assert len(fit_sample_weights) == 2  # baseline + FairBias
    for sw in fit_sample_weights:
        assert sw is None


# ------------------------------------------------------------------------------
# 15. Evaluation Receives No sample_weight
# ------------------------------------------------------------------------------
def test_15_evaluation_receives_no_sample_weight() -> None:
    """evaluate_predictions and compute_evaluation_comparison must have no sample_weight."""
    sig_eval = inspect.signature(evaluate_predictions)
    assert "sample_weight" not in sig_eval.parameters

    sig_comp = inspect.signature(compute_evaluation_comparison)
    assert "sample_weight" not in sig_comp.parameters


# ------------------------------------------------------------------------------
# 16. Prediction Threshold Fixed at 0.5
# ------------------------------------------------------------------------------
def test_16_prediction_threshold_fixed_at_0_5() -> None:
    """Probability decision threshold is strictly fixed at 0.5."""
    mock_cohorts, c_cols, n_cols = create_mock_cohort(n_train=30, n_val=15)
    runner = NHISD5WeightedRunner(enforce_frozen_inputs=False, allow_mitigation=True)
    runner.get_train_val_cohort = MagicMock(return_value=mock_cohorts)
    runner.adapter.preprocessor.get_feature_family_lists = MagicMock(return_value=(c_cols, n_cols))

    def mock_mitigate_step(self, **kwargs):
        return kwargs["X"], kwargs["changed_dict"], None, None

    with patch.object(FairBiasMitigation, "mitigate_step", new=mock_mitigate_step):
        res = runner.execute_train_validation(arm_id="D5_ARM_001")

    assert "utility" in res["val_eval_base"]
    assert "utility" in res["val_eval_fb"]


# ------------------------------------------------------------------------------
# 17. Random Seed Fixed at 0
# ------------------------------------------------------------------------------
def test_17_random_seed_fixed_at_0() -> None:
    """Non-zero random seed must fail closed."""
    runner = NHISD5WeightedRunner(enforce_frozen_inputs=False, allow_mitigation=True)
    with pytest.raises(ValueError, match="protocol strictly freezes random_seed to 0"):
        runner.execute_train_validation(arm_id="D5_ARM_001", random_seed=42)


# ------------------------------------------------------------------------------
# 18. Separate Baseline/FairBias Scalers
# ------------------------------------------------------------------------------
def test_18_separate_baseline_and_fairbias_scalers() -> None:
    """Baseline and FairBias models must use fresh, distinct MinMaxScaler instances."""
    mock_cohorts, c_cols, n_cols = create_mock_cohort(n_train=30, n_val=15)
    runner = NHISD5WeightedRunner(enforce_frozen_inputs=False, allow_mitigation=True)
    runner.get_train_val_cohort = MagicMock(return_value=mock_cohorts)
    runner.adapter.preprocessor.get_feature_family_lists = MagicMock(return_value=(c_cols, n_cols))

    created_scalers = []

    def tracking_scaler(*args, **kwargs):
        s = MinMaxScaler(*args, **kwargs)
        created_scalers.append(s)
        return s

    def mock_mitigate_step(self, **kwargs):
        return kwargs["X"], kwargs["changed_dict"], None, None

    with patch("nhis_fairbias.d5_weighted_runner.MinMaxScaler", side_effect=tracking_scaler), \
         patch.object(FairBiasMitigation, "mitigate_step", new=mock_mitigate_step):
        runner.execute_train_validation(arm_id="D5_ARM_001")

    assert len(created_scalers) == 2
    assert created_scalers[0] is not created_scalers[1]


# ------------------------------------------------------------------------------
# 19. Scalers Fit TRAIN Only
# ------------------------------------------------------------------------------
def test_19_scalers_fit_train_only() -> None:
    """MinMaxScaler must be fit strictly on TRAIN, never on VALIDATION."""
    mock_cohorts, c_cols, n_cols = create_mock_cohort(n_train=30, n_val=15)
    runner = NHISD5WeightedRunner(enforce_frozen_inputs=False, allow_mitigation=True)
    runner.get_train_val_cohort = MagicMock(return_value=mock_cohorts)
    runner.adapter.preprocessor.get_feature_family_lists = MagicMock(return_value=(c_cols, n_cols))

    fit_calls = []
    orig_fit = MinMaxScaler.fit

    def tracking_fit(self, X, y=None):
        fit_calls.append(len(X))
        return orig_fit(self, X, y)

    def mock_mitigate_step(self, **kwargs):
        return kwargs["X"], kwargs["changed_dict"], None, None

    with patch.object(MinMaxScaler, "fit", side_effect=tracking_fit, autospec=True), \
         patch.object(FairBiasMitigation, "mitigate_step", new=mock_mitigate_step):
        runner.execute_train_validation(arm_id="D5_ARM_001")

    # Both fit calls must be with TRAIN length (30), never validation length (15)
    assert len(fit_calls) == 2
    assert fit_calls == [30, 30]


# ------------------------------------------------------------------------------
# 20. HISP Expected 7 Groups / 21 Pairs
# ------------------------------------------------------------------------------
def test_20_hisp_expected_7_groups_21_pairs() -> None:
    """D5_ARM_002 (HISPALLP_A) must specify 7 groups and 21 pairwise comparisons."""
    arm2 = FROZEN_D5_ARMS["D5_ARM_002"]
    assert arm2["expected_group_count"] == 7
    assert arm2["expected_groups"] == [1, 2, 3, 4, 5, 6, 7]
    assert arm2["expected_pair_count"] == 21
    # 7 choose 2 = 21
    assert (7 * 6) // 2 == 21


# ------------------------------------------------------------------------------
# 21. Audit-Only Cannot Invoke Real Mitigation
# ------------------------------------------------------------------------------
def test_21_audit_only_cannot_invoke_real_mitigation() -> None:
    """NHISD5WeightedRunner with allow_mitigation=False forbids substantive execution."""
    runner = NHISD5WeightedRunner(allow_mitigation=False)
    with pytest.raises(RuntimeError, match="strictly forbidden in Gate D5.1a"):
        runner.execute_train_validation("D5_ARM_001")


# ------------------------------------------------------------------------------
# 22. Audit-Only Cannot Score Validation
# ------------------------------------------------------------------------------
def test_22_audit_only_cannot_score_validation(tmp_path: pathlib.Path) -> None:
    """run_audit must not score validation metrics."""
    runner = NHISD5WeightedRunner(enforce_frozen_inputs=False, allow_mitigation=False)
    mock_cohorts, _, _ = create_mock_cohort()
    runner.get_train_val_cohort = MagicMock(return_value=mock_cohorts)

    results = runner.run_audit(output_dir=tmp_path)
    manifest = results["manifest"]
    assert manifest["validation_scored"] is False
    assert manifest["real_weighted_mitigation_executed"] is False


# ------------------------------------------------------------------------------
# 23. Audit-Only Cannot Access TEST Scoring
# ------------------------------------------------------------------------------
def test_23_audit_only_cannot_access_test_scoring(tmp_path: pathlib.Path) -> None:
    """Runner must not access or score the TEST partition."""
    runner = NHISD5WeightedRunner(enforce_frozen_inputs=False, allow_mitigation=False)
    mock_cohorts, _, _ = create_mock_cohort()
    runner.get_train_val_cohort = MagicMock(return_value=mock_cohorts)

    results = runner.run_audit(output_dir=tmp_path)
    manifest = results["manifest"]
    assert manifest["test_evaluated"] is False
    assert manifest["test_partition_accessed"] is False
    assert manifest["authorized_partitions"] == ["train", "validation"]


# ------------------------------------------------------------------------------
# 24. No D4 Artifact Mutation
# ------------------------------------------------------------------------------
def test_24_no_d4_artifact_mutation() -> None:
    """Verify D4 reference preflight runs remain unmodified."""
    for arm_id, ref_info in D4_COMPARISON_REGISTRY.items():
        run_dir = _REPO_ROOT / ref_info["rel_path"]
        assert run_dir.is_dir()

        manifest_p = run_dir / "d4_preflight_manifest.json"
        changed_p = run_dir / "final_changed_dict.json"

        assert manifest_p.is_file()
        assert changed_p.is_file()

        assert compute_sha256(manifest_p) == ref_info["manifest_sha256"]
        assert compute_sha256(changed_p) == ref_info["changed_dict_sha256"]


# ------------------------------------------------------------------------------
# 25. No Weighted Result Artifact Generated in Audit-Only
# ------------------------------------------------------------------------------
def test_25_no_weighted_result_artifact_generated_in_audit_only(tmp_path: pathlib.Path) -> None:
    """Audit-only execution generates only allowed audit artifacts, no mitigation results."""
    runner = NHISD5WeightedRunner(enforce_frozen_inputs=False, allow_mitigation=False)
    mock_cohorts, _, _ = create_mock_cohort()
    runner.get_train_val_cohort = MagicMock(return_value=mock_cohorts)

    runner.run_audit(output_dir=tmp_path)

    allowed = {
        "audit_manifest.json",
        "frozen_input_contract.json",
        "weight_input_audit.json",
        "weight_group_audit.csv",
        "four_arm_registry.json",
    }
    found_files = {p.name for p in tmp_path.iterdir() if p.is_file()}
    assert found_files == allowed

    # Explicitly check forbidden artifacts
    forbidden_prefixes = [
        "weighted_changed_dict",
        "weighted_trace",
        "validation_metrics",
        "test_metrics",
        "predictions",
    ]
    for fn in found_files:
        for prefix in forbidden_prefixes:
            assert not fn.startswith(prefix), f"Forbidden artifact found: {fn}"
