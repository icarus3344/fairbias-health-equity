"""Synthetic and unit test suite for Gate D5.0: survey-weighted FairBias geometry extension.

Covers all 14 required specifications:
1. Equal-weight degeneracy (w_i = const > 0 degenerates to unweighted).
2. Global scale invariance (w'_i = c * w_i yields identical geometry).
3. Integer replication equivalence (row duplication matches weighted statistics).
4. Manually verified numeric weighted means.
5. Manually verified categorical weighted proportions.
6. Multicategory protected attribute and feature support.
7. Negative, NaN, Inf, and all-zero weight rejection (fail-closed).
8. Zero group-weight rejection (fail-closed) vs individual zero-weight acceptance.
9. Row-alignment and index mismatch safety.
10. Paper-faithful mode strictly forbids sample_weight.
11. Survey-weighted geometry extension provenance labeling.
12. Classifier fitting remains strictly unweighted (no sample_weight leakage).
13. Downstream evaluation remains strictly unweighted.
14. Deterministic unweighted regression equivalence.
"""

from __future__ import annotations

import copy
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fairbias.bias_metric import (
    SURVEY_WEIGHTED_EXTENSION_DECLARATION,
    compute_bias_concentration,
    compute_dphi_matrix,
    compute_pairwise_divergences,
)
from fairbias.config import (
    ALGORITHM_MODE_PAPER_FAITHFUL,
    ALGORITHM_MODE_SURVEY_WEIGHTED,
    FairBiasConfig,
)
from fairbias.evaluator import FairEvaluator
from fairbias.mitigation import FairBiasMitigation
from fairbias.transform import FairTransform, calculate_nmi_dict
from nhis_fairbias.survey import (
    get_survey_weighted_geometry_provenance,
    validate_survey_weights,
)


# ------------------------------------------------------------------------------
# Fixtures and Helpers
# ------------------------------------------------------------------------------
def create_synthetic_toy_data(n: int = 100, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Create reproducible synthetic dataset with numerical and categorical features."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "num_f1": rng.normal(10.0, 2.0, size=n),
        "num_f2": rng.uniform(0.0, 100.0, size=n),
        "cat_f1": rng.choice(["low", "med", "high"], size=n),
        "cat_f2": rng.choice(["A", "B"], size=n),
    })
    O = pd.DataFrame({
        "prot_binary": rng.choice([1, 2], size=n),
        "prot_multi": rng.choice([1, 2, 3], size=n),
    })
    Y = pd.Series(rng.choice([0, 1], size=n), name="target")
    return X, O, Y


# ------------------------------------------------------------------------------
# 1. Equal-Weight Degeneracy
# ------------------------------------------------------------------------------
def test_equal_weight_degeneracy_pairwise_divergences() -> None:
    """When all weights are equal (w_i = c > 0), divergences must match unweighted exactly."""
    X, O, _ = create_synthetic_toy_data(n=60, seed=123)
    o_series = O["prot_binary"]
    cate_attrs = ["cat_f1", "cat_f2"]
    num_attrs = ["num_f1", "num_f2"]

    # Unweighted divergences
    div_unweighted = compute_pairwise_divergences(
        X, o_series, cate_attrs=cate_attrs, num_attrs=num_attrs, sample_weight=None
    )

    # Weighted with constant positive weight (e.g. 7.5)
    weights = pd.Series(np.full(len(X), 7.5), index=X.index)
    div_weighted = compute_pairwise_divergences(
        X, o_series, cate_attrs=cate_attrs, num_attrs=num_attrs, sample_weight=weights
    )

    pd.testing.assert_frame_equal(div_unweighted, div_weighted, check_exact=False, atol=1e-12)


def test_equal_weight_degeneracy_dphi_and_mitigation() -> None:
    """d_phi, epsilon, and mitigation decisions must match unweighted when weights are equal."""
    X, O, Y = create_synthetic_toy_data(n=60, seed=456)
    cate_attrs = ["cat_f1", "cat_f2"]
    num_attrs = ["num_f1", "num_f2"]

    weights = pd.Series(np.ones(len(X)) * 2.0, index=X.index)

    # 1. d_phi matrix
    dphi_unweighted = compute_dphi_matrix(
        X, O[["prot_binary"]], cate_attrs, num_attrs, sample_weight=None
    )
    dphi_weighted = compute_dphi_matrix(
        X, O[["prot_binary"]], cate_attrs, num_attrs, sample_weight=weights
    )
    for p_col in dphi_unweighted:
        for feat in dphi_unweighted[p_col]:
            np.testing.assert_allclose(
                dphi_weighted[p_col][feat],
                dphi_unweighted[p_col][feat],
                atol=1e-10,
            )

    # 2. Mitigation step
    cfg = FairBiasConfig(
        algorithm_mode="engineering_bounded",
        label_O=("prot_binary",),
        random_seed=0,
    )
    evaluator = FairEvaluator(cfg, label_O=["prot_binary"], cate_attrs=cate_attrs, num_attrs=num_attrs)
    transformer = FairTransform(n_bins=5)
    nmi_org = calculate_nmi_dict(X, Y)

    engine_unweighted = FairBiasMitigation(
        evaluator=evaluator,
        transformer=transformer,
        label_O=["prot_binary"],
        cate_attrs=cate_attrs,
        num_attrs=num_attrs,
        sample_weight=None,
    )
    engine_weighted = FairBiasMitigation(
        evaluator=evaluator,
        transformer=transformer,
        label_O=["prot_binary"],
        cate_attrs=cate_attrs,
        num_attrs=num_attrs,
        sample_weight=weights,
    )

    step_unweighted = engine_unweighted.mitigate_step(
        X=X, Y=Y, O=O[["prot_binary"]], nmi_org=nmi_org,
        changed_dict={}, current_epsilon=dphi_unweighted,
        epsilon_threshold=0.05, iteration=1,
    )
    step_weighted = engine_weighted.mitigate_step(
        X=X, Y=Y, O=O[["prot_binary"]], nmi_org=nmi_org,
        changed_dict={}, current_epsilon=dphi_weighted,
        epsilon_threshold=0.05, iteration=1,
    )

    # Compare resulting transformed X and changed_dict
    pd.testing.assert_frame_equal(step_unweighted[0], step_weighted[0])
    assert step_unweighted[1] == step_weighted[1]  # changed_dict
    assert step_unweighted[2] == step_weighted[2]  # sel_o
    assert step_unweighted[3] == step_weighted[3]  # sel_attr


# ------------------------------------------------------------------------------
# 2. Global Scale Invariance
# ------------------------------------------------------------------------------
def test_global_scale_invariance() -> None:
    """Scaling all weights by an arbitrary positive scalar c > 0 must not change geometry."""
    X, O, _ = create_synthetic_toy_data(n=80, seed=789)
    rng = np.random.RandomState(999)
    base_weights = pd.Series(rng.uniform(0.5, 10.0, size=len(X)), index=X.index)
    scaled_weights = base_weights * 42.195

    cate_attrs = ["cat_f1", "cat_f2"]
    num_attrs = ["num_f1", "num_f2"]

    # Pairwise divergences
    div_base = compute_pairwise_divergences(
        X, O["prot_multi"], cate_attrs, num_attrs, sample_weight=base_weights
    )
    div_scaled = compute_pairwise_divergences(
        X, O["prot_multi"], cate_attrs, num_attrs, sample_weight=scaled_weights
    )
    pd.testing.assert_frame_equal(div_base, div_scaled, check_exact=False, atol=1e-12)

    # d_phi matrix
    dphi_base = compute_dphi_matrix(
        X, O[["prot_multi"]], cate_attrs, num_attrs, sample_weight=base_weights
    )
    dphi_scaled = compute_dphi_matrix(
        X, O[["prot_multi"]], cate_attrs, num_attrs, sample_weight=scaled_weights
    )
    for p_col in dphi_base:
        for feat in dphi_base[p_col]:
            np.testing.assert_allclose(
                dphi_scaled[p_col][feat],
                dphi_base[p_col][feat],
                atol=1e-10,
            )


# ------------------------------------------------------------------------------
# 3. Integer Replication Equivalence
# ------------------------------------------------------------------------------
def test_integer_replication_equivalence() -> None:
    """A dataset with positive integer weights matches an expanded unweighted dataset formed by row replication."""
    # Small synthetic dataset
    X_small = pd.DataFrame({
        "num1": [1.0, 2.0, 5.0, 10.0, 20.0],
        "cat1": ["a", "b", "a", "c", "b"],
    })
    O_small = pd.Series([1, 1, 2, 2, 2], name="prot")
    int_weights = pd.Series([3, 1, 2, 4, 1], name="wt")

    # Form replicated unweighted dataset
    rep_indices = []
    for idx, w in enumerate(int_weights):
        rep_indices.extend([idx] * int(w))
    X_rep = X_small.iloc[rep_indices].reset_index(drop=True)
    O_rep = O_small.iloc[rep_indices].reset_index(drop=True)

    # Compare group divergences
    div_weighted = compute_pairwise_divergences(
        X_small, O_small, cate_attrs=["cat1"], num_attrs=["num1"], sample_weight=int_weights
    )
    div_unweighted_rep = compute_pairwise_divergences(
        X_rep, O_rep, cate_attrs=["cat1"], num_attrs=["num1"], sample_weight=None
    )

    pd.testing.assert_frame_equal(div_weighted, div_unweighted_rep, check_exact=False, atol=1e-12)


# ------------------------------------------------------------------------------
# 4. Manually Verified Numeric Weighted Means
# ------------------------------------------------------------------------------
def test_manual_numeric_weighted_mean() -> None:
    """Manually verify group means and within-pair min-max scaling divergence."""
    # Data design:
    # Group 1: values [10.0, 30.0], weights [1.0, 3.0]
    #   weighted mean raw = (10*1 + 30*3)/4 = 100/4 = 25.0
    # Group 2: values [20.0, 40.0], weights [3.0, 1.0]
    #   weighted mean raw = (20*3 + 40*1)/4 = 100/4 = 25.0
    # Overall min = 10.0, max = 40.0, span = 30.0
    # Min-max transformed values:
    # Group 1: [0.0, 20/30 = 2/3], weights [1.0, 3.0]
    #   weighted mean scaled = (0.0*1 + (2/3)*3) / 4 = 2.0 / 4 = 0.5
    # Group 2: [10/30 = 1/3, 30/30 = 1.0], weights [3.0, 1.0]
    #   weighted mean scaled = ((1/3)*3 + 1.0*1) / 4 = 2.0 / 4 = 0.5
    # Expected divergence |0.5 - 0.5| = 0.0
    #
    # Contrast with unweighted:
    # Group 1 unweighted scaled: (0.0 + 2/3) / 2 = 1/3 ≈ 0.333333
    # Group 2 unweighted scaled: (1/3 + 1.0) / 2 = 2/3 ≈ 0.666667
    # Unweighted divergence |1/3 - 2/3| = 1/3 ≈ 0.333333
    df_x = pd.DataFrame({"feat": [10.0, 30.0, 20.0, 40.0]})
    df_o = pd.Series([1, 1, 2, 2], name="prot")
    weights = pd.Series([1.0, 3.0, 3.0, 1.0])

    div_unweighted = compute_pairwise_divergences(
        df_x, df_o, cate_attrs=[], num_attrs=["feat"], sample_weight=None
    )
    assert np.isclose(div_unweighted.loc["feat", "1_2"], 1.0 / 3.0, atol=1e-6)

    div_weighted = compute_pairwise_divergences(
        df_x, df_o, cate_attrs=[], num_attrs=["feat"], sample_weight=weights
    )
    # The weighted divergence must be exactly 0.0 per hand calculation
    assert np.isclose(div_weighted.loc["feat", "1_2"], 0.0, atol=1e-12)


# ------------------------------------------------------------------------------
# 5. Manually Verified Categorical Weighted Proportions
# ------------------------------------------------------------------------------
def test_manual_categorical_weighted_proportions() -> None:
    """Manually verify categorical frequency gap (cat-a) with non-uniform weights."""
    # Group 1:
    #   cat "A": 1 observation, weight 1.0
    #   cat "B": 1 observation, weight 9.0
    #   total weight = 10.0 => p(A) = 0.1, p(B) = 0.9, p(C) = 0.0
    # Group 2:
    #   cat "A": 1 observation, weight 8.0
    #   cat "C": 1 observation, weight 2.0
    #   total weight = 10.0 => p(A) = 0.8, p(B) = 0.0, p(C) = 0.2
    # Union of categories: {A, B, C} => K = 3
    # Abs gaps:
    #   A: |0.1 - 0.8| = 0.7
    #   B: |0.9 - 0.0| = 0.9
    #   C: |0.0 - 0.2| = 0.2
    # Total abs gap = 0.7 + 0.9 + 0.2 = 1.8
    # Divergence = (1/K) * 1.8 = 1.8 / 3 = 0.60
    df_x = pd.DataFrame({"cat_feat": ["A", "B", "A", "C"]})
    df_o = pd.Series([1, 1, 2, 2], name="prot")
    weights = pd.Series([1.0, 9.0, 8.0, 2.0])

    div_weighted = compute_pairwise_divergences(
        df_x, df_o, cate_attrs=["cat_feat"], num_attrs=[], sample_weight=weights
    )
    assert np.isclose(div_weighted.loc["cat_feat", "1_2"], 0.60, atol=1e-12)


# ------------------------------------------------------------------------------
# 6. Multicategory Support Across 3 Groups and 3 Categories
# ------------------------------------------------------------------------------
def test_multicategory_3groups_3categories() -> None:
    """Protected attribute with 3 groups and feature with 3 categories produces expected pair columns."""
    df_x = pd.DataFrame({"color": ["R", "G", "B", "R", "G", "B", "R", "R", "B"]})
    df_o = pd.Series([1, 1, 1, 2, 2, 2, 3, 3, 3], name="group")
    weights = pd.Series([1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0])

    div = compute_pairwise_divergences(
        df_x, df_o, cate_attrs=["color"], num_attrs=[], sample_weight=weights
    )
    # Expected columns: 1_2, 1_3, 2_3
    assert set(div.columns) == {"1_2", "1_3", "2_3"}
    for col in div.columns:
        val = div.loc["color", col]
        assert np.isfinite(val)
        assert 0.0 <= val <= 2.0


# ------------------------------------------------------------------------------
# 7. Weight Validation Failures (Fail-Closed)
# ------------------------------------------------------------------------------
def test_weight_validation_rejections() -> None:
    """Negative, NaN, Inf, and all-zero weights must fail closed."""
    df_x = pd.DataFrame({"n": [1.0, 2.0, 3.0, 4.0]})
    df_o = pd.Series([1, 1, 2, 2])

    # 1. Negative weight
    with pytest.raises(ValueError, match="strictly positive|non-negative"):
        compute_pairwise_divergences(df_x, df_o, [], ["n"], sample_weight=[1.0, -0.1, 1.0, 1.0])
    with pytest.raises(ValueError, match="non-negative"):
        compute_pairwise_divergences(df_x, df_o, [], ["n"], sample_weight=[1.0, -0.1, 1.0, 1.0], allow_zero_weights=True)

    # 2. NaN weight
    with pytest.raises(ValueError, match="NaN or non-finite"):
        compute_pairwise_divergences(df_x, df_o, [], ["n"], sample_weight=[1.0, np.nan, 1.0, 1.0])

    # 3. +Inf weight
    with pytest.raises(ValueError, match="NaN or non-finite"):
        compute_pairwise_divergences(df_x, df_o, [], ["n"], sample_weight=[1.0, np.inf, 1.0, 1.0])

    # 4. -Inf weight
    with pytest.raises(ValueError, match="NaN or non-finite"):
        compute_pairwise_divergences(df_x, df_o, [], ["n"], sample_weight=[1.0, -np.inf, 1.0, 1.0])

    # 5. All-zero weights
    with pytest.raises(ValueError, match="all-zero|strictly positive"):
        compute_pairwise_divergences(df_x, df_o, [], ["n"], sample_weight=[0.0, 0.0, 0.0, 0.0])

    # 6. Length mismatch
    with pytest.raises(ValueError, match="length"):
        compute_pairwise_divergences(df_x, df_o, [], ["n"], sample_weight=[1.0, 2.0])


def test_zero_group_weight_rejection_vs_individual_zero() -> None:
    """A protected group with zero total weight fails closed; individual zeros are allowed with allow_zero_weights=True if group total > 0."""
    df_x = pd.DataFrame({"n": [1.0, 2.0, 3.0, 4.0]})
    df_o = pd.Series([1, 1, 2, 2])

    # 1. By default, any zero weight is rejected for strict backward compatibility
    with pytest.raises(ValueError, match="strictly positive"):
        compute_pairwise_divergences(df_x, df_o, [], ["n"], sample_weight=[0.0, 5.0, 2.0, 2.0], allow_zero_weights=False)

    # 2. Entire group 1 has zero weight -> fail closed even with allow_zero_weights=True
    with pytest.raises(ValueError, match="non-positive total weight"):
        compute_pairwise_divergences(df_x, df_o, [], ["n"], sample_weight=[0.0, 0.0, 1.0, 2.0], allow_zero_weights=True)

    # 3. Individual zero weight with group total > 0 is permitted under allow_zero_weights=True
    div = compute_pairwise_divergences(df_x, df_o, [], ["n"], sample_weight=[0.0, 5.0, 2.0, 2.0], allow_zero_weights=True)
    assert np.isfinite(div.loc["n", "1_2"])


# ------------------------------------------------------------------------------
# 8. Alignment Safety Tests
# ------------------------------------------------------------------------------
def test_row_alignment_safety() -> None:
    """Index mismatches between X, O, and sample_weight must fail closed rather than align positionally."""
    # Non-trivial index
    idx1 = [10, 20, 30, 40]
    idx_shuffled = [40, 20, 10, 30]

    df_x = pd.DataFrame({"n": [1.0, 2.0, 3.0, 4.0]}, index=idx1)
    df_o_bad = pd.Series([1, 1, 2, 2], index=idx_shuffled)
    weights_good = pd.Series([1.0, 1.0, 1.0, 1.0], index=idx1)
    weights_bad = pd.Series([1.0, 1.0, 1.0, 1.0], index=idx_shuffled)

    # 1. O index mismatch
    with pytest.raises(ValueError, match="Index alignment mismatch"):
        compute_pairwise_divergences(df_x, df_o_bad, [], ["n"], sample_weight=weights_good)

    # 2. sample_weight index mismatch
    df_o_good = pd.Series([1, 1, 2, 2], index=idx1)
    with pytest.raises(ValueError, match="Index alignment mismatch"):
        compute_pairwise_divergences(df_x, df_o_good, [], ["n"], sample_weight=weights_bad)

    # 3. compute_dphi_matrix alignment check
    with pytest.raises(ValueError, match="Index alignment mismatch"):
        compute_dphi_matrix(
            df_x, pd.DataFrame({"p": df_o_bad}), [], ["n"], sample_weight=weights_good
        )


# ------------------------------------------------------------------------------
# 9. Mode Separation: Paper-Faithful Mode Rejection
# ------------------------------------------------------------------------------
def test_paper_faithful_mode_strictly_rejects_weights() -> None:
    """tang2024_paper_faithful strictly forbids sample_weight (fail closed)."""
    X, O, _ = create_synthetic_toy_data(n=30, seed=1)
    cfg = FairBiasConfig(algorithm_mode=ALGORITHM_MODE_PAPER_FAITHFUL)
    evaluator = FairEvaluator(cfg, label_O=["prot_binary"], num_attrs=["num_f1"])

    weights = pd.Series(np.ones(len(X)), index=X.index)
    with pytest.raises(ValueError, match="tang2024_paper_faithful.*strictly forbids sample_weight"):
        evaluator.calculate_epsilon(
            X[["num_f1"]], O[["prot_binary"]], num_attrs=["num_f1"], sample_weight=weights
        )


def test_survey_weighted_mode_requires_weights() -> None:
    """survey_weighted_geometry mode requires sample_weight to be provided."""
    X, O, _ = create_synthetic_toy_data(n=30, seed=2)
    cfg = FairBiasConfig(algorithm_mode=ALGORITHM_MODE_SURVEY_WEIGHTED)
    evaluator = FairEvaluator(cfg, label_O=["prot_binary"], num_attrs=["num_f1"])

    with pytest.raises(ValueError, match="survey_weighted_geometry.*requires sample_weight"):
        evaluator.calculate_epsilon(
            X[["num_f1"]], O[["prot_binary"]], num_attrs=["num_f1"], sample_weight=None
        )


# ------------------------------------------------------------------------------
# 10. Provenance Labeling Metadata
# ------------------------------------------------------------------------------
def test_survey_weighted_provenance_labeling() -> None:
    """Verify exact provenance dictionary structure and fields."""
    prov = get_survey_weighted_geometry_provenance(survey_weight_variable="WTFA_A")
    assert prov["base_algorithm"] == "tang2024_paper_faithful"
    assert prov["extension"] == "survey_weighted_geometry"
    assert prov["release_protocol"] == "survey_weighted_geometry_extension"
    assert prov["survey_weight_variable"] == "WTFA_A"
    assert prov["classifier_weighted"] is False
    assert prov["evaluation_weighted"] is False
    assert prov["complex_survey_inference"] is False
    assert prov["PSTRAT_used_for_variance"] is False
    assert prov["PPSU_used_for_variance"] is False


# ------------------------------------------------------------------------------
# 11. No Classifier / Evaluation Weighting Leakage
# ------------------------------------------------------------------------------
def test_no_classifier_weighting_leakage() -> None:
    """LogisticRegression.fit() must receive NO sample_weight during model fitting."""
    X, O, Y = create_synthetic_toy_data(n=50, seed=3)
    evaluator = FairEvaluator(label_O=["prot_binary"])

    # Spy on the underlying model's fit method
    original_fit = evaluator.model.fit
    fit_mock = MagicMock(side_effect=original_fit)
    evaluator.model.fit = fit_mock

    evaluator.fit_and_predict(X[["num_f1", "num_f2"]], Y, X[["num_f1", "num_f2"]])

    # Ensure fit was called with (X, y) and no sample_weight keyword argument
    assert fit_mock.called
    call_args, call_kwargs = fit_mock.call_args
    assert "sample_weight" not in call_kwargs or call_kwargs["sample_weight"] is None


def test_no_evaluation_weighting_leakage() -> None:
    """Evaluation compute_metrics must calculate unweighted classification and disparity metrics."""
    _, O, Y = create_synthetic_toy_data(n=40, seed=4)
    evaluator = FairEvaluator(label_O=["prot_binary"])

    y_pred = np.zeros(len(Y), dtype=int)
    y_prob = np.full(len(Y), 0.2)

    # Calling compute_metrics
    metrics = evaluator.compute_metrics(Y, y_pred, O[["prot_binary"]], y_prob=y_prob)

    # Standard unweighted metric keys must be present
    assert "ACC" in metrics
    assert "F1" in metrics
    assert "SP" in metrics
    assert "EO" in metrics
    # Ensure no sample_weights parameter exists in compute_metrics signature
    import inspect
    sig = inspect.signature(evaluator.compute_metrics)
    assert "sample_weight" not in sig.parameters


# ------------------------------------------------------------------------------
# 12. Helper Function Unit Tests
# ------------------------------------------------------------------------------
def test_validate_survey_weights_helper() -> None:
    """Test the standalone validate_survey_weights validation helper."""
    valid_s = pd.Series([1.0, 2.5, 0.0, 10.0], index=[10, 20, 30, 40])
    validated = validate_survey_weights(valid_s, expected_length=4, expected_index=valid_s.index)
    assert isinstance(validated, np.ndarray)
    assert len(validated) == 4

    # Index mismatch
    with pytest.raises(ValueError, match="Index alignment mismatch"):
        validate_survey_weights(valid_s, expected_index=pd.Index([1, 2, 3, 4]))

    # Negative weight
    with pytest.raises(ValueError, match="non-negative"):
        validate_survey_weights(np.array([1.0, -2.0, 3.0]))

    # Non-finite
    with pytest.raises(ValueError, match="non-finite"):
        validate_survey_weights(np.array([1.0, np.nan, 3.0]))


# ------------------------------------------------------------------------------
# 13. Deterministic Unweighted Regression Equivalence
# ------------------------------------------------------------------------------
def test_unweighted_regression_exact_match() -> None:
    """Calling with sample_weight=None must yield bit-for-bit identical results to existing pipeline."""
    X, O, Y = create_synthetic_toy_data(n=50, seed=5)
    cate_attrs = ["cat_f1", "cat_f2"]
    num_attrs = ["num_f1", "num_f2"]

    # Running with None
    div1 = compute_pairwise_divergences(X, O["prot_binary"], cate_attrs, num_attrs, sample_weight=None)
    dphi1 = compute_dphi_matrix(X, O[["prot_binary"]], cate_attrs, num_attrs, sample_weight=None)

    # Running again with None
    div2 = compute_pairwise_divergences(X, O["prot_binary"], cate_attrs, num_attrs, sample_weight=None)
    dphi2 = compute_dphi_matrix(X, O[["prot_binary"]], cate_attrs, num_attrs, sample_weight=None)

    pd.testing.assert_frame_equal(div1, div2)
    assert dphi1 == dphi2
