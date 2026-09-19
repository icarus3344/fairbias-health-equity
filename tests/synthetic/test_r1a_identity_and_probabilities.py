from decimal import Decimal
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from fairbias.application_metrics import compute_application_group_fairness
from fairbias.config import FairBiasConfig
from fairbias.enhancement_contracts import (
    CandidateEvaluationResult,
    EnhancementStatus,
    EvaluationPartition,
    evaluate_candidate_utility,
)
from fairbias.evaluator import FairEvaluator
from fairbias.prediction_contracts import (
    ProbabilityValidationError,
    validate_and_extract_positive_probabilities,
)
from fairbias.transform import FairTransform
from nhis_fairbias.d8_enhancement_runner import evaluate_representation


def test_identity_1_numeric_year_and_id_normalization():
    """IDENTITY-1: Equivalent numeric years (2023 and 2023.0) and numeric IDs normalize consistently."""
    df_a = pd.DataFrame({"year": [2023, 2023], "f1": [1.0, 2.0]}, index=[1, 2])
    df_b = pd.DataFrame({"year": [2023.0, 2023.0], "f1": [1.0, 2.0]}, index=[1.0, 2.0])

    ids_a = EvaluationPartition._extract_record_ids(df_a, source_override="study")
    ids_b = EvaluationPartition._extract_record_ids(df_b, source_override="study")

    assert ids_a == ids_b == ["src=5:study|yr=4:2023|id=1:1", "src=5:study|yr=4:2023|id=1:2"]

    # String ID with leading zeros preserved
    df_str = pd.DataFrame({"year": [2023]}, index=["00123"])
    ids_str = EvaluationPartition._extract_record_ids(df_str, source_override="study")
    assert ids_str == ["src=5:study|yr=4:2023|id=5:00123"], "Leading zeros in string record ID must be preserved"


def test_identity_2_invalid_year_rejection():
    """IDENTITY-2: Non-integer years (2023.9), boolean, NaN/Inf, missing year rejected."""
    df_float = pd.DataFrame({"year": [2023.9], "f1": [1.0]}, index=[1])
    with pytest.raises(ValueError, match="Non-integer year"):
        EvaluationPartition._extract_record_ids(df_float, "study")

    df_bool = pd.DataFrame({"year": [True], "f1": [1.0]}, index=[1])
    with pytest.raises(TypeError, match="Boolean value"):
        EvaluationPartition._extract_record_ids(df_bool, "study")

    df_nan = pd.DataFrame({"year": [float("nan")], "f1": [1.0]}, index=[1])
    with pytest.raises(ValueError, match="Year cannot be None, NaN, or missing"):
        EvaluationPartition._extract_record_ids(df_nan, "study")

    df_inf = pd.DataFrame({"year": [float("inf")], "f1": [1.0]}, index=[1])
    with pytest.raises(ValueError, match="Non-finite year"):
        EvaluationPartition._extract_record_ids(df_inf, "study")


def test_identity_3_no_substring_source_matching():
    """IDENTITY-3: Explicit source/year does not use substring matching."""
    df = pd.DataFrame({"year": [2023]}, index=[10])
    ids = EvaluationPartition._extract_record_ids(df, source_override="nhis_2023")
    assert ids == ["src=9:nhis_2023|yr=4:2023|id=2:10"]


def test_identity_4_duplicate_and_overlap_rejection():
    """IDENTITY-4: Internal duplicate record IDs and cross-partition overlap rejected with ValueError."""
    # Internal duplicates
    df_dup = pd.DataFrame({"f1": [1.0, 2.0]}, index=[1, 1])
    y_dup = pd.Series([0, 1], index=[1, 1])
    df_other = pd.DataFrame({"f1": [3.0, 4.0]}, index=[2, 3])
    y_other = pd.Series([0, 1], index=[2, 3])

    with pytest.raises(ValueError, match="duplicate record IDs detected"):
        EvaluationPartition(fit_X=df_dup, fit_y=y_dup, selection_X=df_other, selection_y=y_other)

    # Cross-partition overlap
    df_fit = pd.DataFrame({"f1": [1.0, 2.0]}, index=[1, 2])
    y_fit = pd.Series([0, 1], index=[1, 2])
    df_sel = pd.DataFrame({"f1": [3.0, 4.0]}, index=[2, 3])  # index 2 shared
    y_sel = pd.Series([0, 1], index=[2, 3])

    with pytest.raises(ValueError, match="share 1 record IDs"):
        EvaluationPartition(fit_X=df_fit, fit_y=y_fit, selection_X=df_sel, selection_y=y_sel)


def test_identity_5_different_source_same_id():
    """IDENTITY-5: Different sources with the same ID are distinguished and do not collide."""
    df_fit = pd.DataFrame({"f1": [1.0, 2.0]}, index=[1, 2])
    y_fit = pd.Series([0, 1], index=[1, 2])
    df_sel = pd.DataFrame({"f1": [3.0, 4.0]}, index=[1, 2])
    y_sel = pd.Series([0, 1], index=[1, 2])

    # With different sources, index [1, 2] is namespaced to src_a:1 vs src_b:1
    part = EvaluationPartition(
        fit_X=df_fit,
        fit_y=y_fit,
        selection_X=df_sel,
        selection_y=y_sel,
        fit_source="src_a",
        selection_source="src_b",
    )
    assert part is not None


def test_identity_6_feature_equivalence_not_leakage():
    """IDENTITY-6: Identical feature values across different record IDs do NOT trigger leakage; generic core accepts valid years outside 2022-2024."""
    # Identical features, disjoint indices
    df_fit = pd.DataFrame({"f1": [1.0, 2.0], "year": [2018, 2019]}, index=[1, 2])
    y_fit = pd.Series([0, 1], index=[1, 2])
    df_sel = pd.DataFrame({"f1": [1.0, 2.0], "year": [2018, 2019]}, index=[3, 4])
    y_sel = pd.Series([0, 1], index=[3, 4])

    part = EvaluationPartition(
        fit_X=df_fit,
        fit_y=y_fit,
        selection_X=df_sel,
        selection_y=y_sel,
        fit_source="generic_study",
        selection_source="generic_study",
    )
    assert part is not None


class MockModelWithReversedClasses:
    def __init__(self):
        self.classes_ = np.array([1, 0])  # Positive class 1 is at column 0!

    def predict_proba(self, X):
        n = len(X)
        # Column 0 is P(Y=1), Column 1 is P(Y=0)
        p1 = np.linspace(0.6, 0.9, n)
        p0 = 1.0 - p1
        return np.column_stack([p1, p0])


def test_proba_1_classes_order_positive_column():
    """PROBA-1: classes_=[1, 0] correctly extracts the positive class probability column (at index 0)."""
    model = MockModelWithReversedClasses()
    X = np.ones((5, 2))
    probs = validate_and_extract_positive_probabilities(model, X, expected_classes=(0, 1), pos_label=1)
    assert np.all(probs >= 0.6), "Must extract column 0 corresponding to class 1, not column 1"


def test_proba_2_valid_two_column_computation():
    """PROBA-2: Legitimate two-column probabilities compute AUROC successfully."""
    X_fit = pd.DataFrame({"f1": [0.1, 0.2, 0.8, 0.9]}, index=[1, 2, 3, 4])
    y_fit = pd.Series([0, 0, 1, 1], index=[1, 2, 3, 4])
    X_sel = pd.DataFrame({"f1": [0.15, 0.85]}, index=[5, 6])
    y_sel = pd.Series([0, 1], index=[5, 6])

    partition = EvaluationPartition(fit_X=X_fit, fit_y=y_fit, selection_X=X_sel, selection_y=y_sel)
    transformer = FairTransform()
    evaluator = FairEvaluator(config=FairBiasConfig())

    res = evaluate_candidate_utility(
        partition=partition,
        changed_dict={},
        num_attrs=["f1"],
        cate_attrs=[],
        transformer=transformer,
        evaluator=evaluator,
    )
    assert res.is_valid
    assert res.utility_score == 1.0


class MockCorruptModel:
    def __init__(self, mode: str):
        self.mode = mode
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        n = len(X)
        if self.mode == "negative":
            return np.column_stack([np.full(n, 1.5), np.full(n, -0.5)])
        if self.mode == "greater_than_one":
            return np.column_stack([np.full(n, 2.0), np.full(n, -1.0)])
        if self.mode == "nan":
            return np.column_stack([np.full(n, np.nan), np.full(n, 0.5)])
        if self.mode == "inf":
            return np.column_stack([np.full(n, np.inf), np.full(n, 0.0)])
        if self.mode == "bad_row_sums":
            return np.column_stack([np.full(n, 0.7), np.full(n, 0.7)])
        if self.mode == "1d":
            return np.full(n, 0.5)
        if self.mode == "3_classes":
            return np.full((n, 3), 1 / 3)
        return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])


def test_proba_3_negative_and_excess_proba_rejected():
    """PROBA-3: Negative probabilities (-1.0) or probabilities > 1.0 rejected."""
    X = np.ones((4, 2))
    model_neg = MockCorruptModel("negative")
    with pytest.raises(ProbabilityValidationError, match="outside \\[0.0, 1.0\\]"):
        validate_and_extract_positive_probabilities(model_neg, X)

    model_excess = MockCorruptModel("greater_than_one")
    with pytest.raises(ProbabilityValidationError, match="outside \\[0.0, 1.0\\]"):
        validate_and_extract_positive_probabilities(model_excess, X)


def test_proba_4_non_finite_proba_rejected():
    """PROBA-4: Inf/NaN in probabilities rejected."""
    X = np.ones((4, 2))
    model_nan = MockCorruptModel("nan")
    with pytest.raises(ProbabilityValidationError, match="NaN, Inf, or non-finite"):
        validate_and_extract_positive_probabilities(model_nan, X)

    model_inf = MockCorruptModel("inf")
    with pytest.raises(ProbabilityValidationError, match="NaN, Inf, or non-finite"):
        validate_and_extract_positive_probabilities(model_inf, X)


def test_proba_5_invalid_row_sums_and_shape_rejected():
    """PROBA-5: Invalid row sums or wrong dimension rejected."""
    X = np.ones((4, 2))
    model_sum = MockCorruptModel("bad_row_sums")
    with pytest.raises(ProbabilityValidationError, match="row sums deviate from 1.0"):
        validate_and_extract_positive_probabilities(model_sum, X)

    model_1d = MockCorruptModel("1d")
    with pytest.raises(ProbabilityValidationError, match="shape"):
        validate_and_extract_positive_probabilities(model_1d, X)


def test_proba_6_candidate_and_final_paths_fail_closed():
    """PROBA-6: Single class, unknown labels, and missing predict_proba rejected in both candidate utility evaluation and final paths."""
    X_fit = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4]}, index=[1, 2, 3, 4])
    y_fit = pd.Series([0, 0, 0, 0], index=[1, 2, 3, 4])  # Single class!
    X_sel = pd.DataFrame({"f1": [0.5, 0.6]}, index=[5, 6])
    y_sel = pd.Series([0, 0], index=[5, 6])

    partition = EvaluationPartition(fit_X=X_fit, fit_y=y_fit, selection_X=X_sel, selection_y=y_sel)
    transformer = FairTransform()
    evaluator = FairEvaluator(config=FairBiasConfig())

    res = evaluate_candidate_utility(
        partition=partition,
        changed_dict={},
        num_attrs=["f1"],
        cate_attrs=[],
        transformer=transformer,
        evaluator=evaluator,
    )
    assert not res.is_valid, "Single class target must fail candidate utility evaluation"
    assert res.utility_score is None

    # Final evaluation path rejection
    with pytest.raises((ValueError, RuntimeError)):
        evaluator.fit_and_predict(X_fit, y_fit, X_sel)


def test_identity_7_delimiter_framing_collision_prevention():
    """IDENTITY-7: Delimiter framing prevents collisions between 'a:b'+'c' and 'a'+'b:c'."""
    df_a = pd.DataFrame({"year": [2022]}, index=["c"])
    ids_a = EvaluationPartition._extract_record_ids(df_a, source_override="a:b")

    df_b = pd.DataFrame({"year": [2022]}, index=["b:c"])
    ids_b = EvaluationPartition._extract_record_ids(df_b, source_override="a")

    assert ids_a != ids_b, "Length-framed IDs must not collide when delimiters appear in components"
    assert "src=3:a:b" in ids_a[0]
    assert "src=1:a" in ids_b[0]


def test_identity_8_empty_id_and_source_rejection():
    """IDENTITY-8: None, empty string, or whitespace-only record IDs and sources are rejected."""
    df_empty_id = pd.DataFrame({"year": [2022]}, index=[""])
    with pytest.raises(ValueError, match="Empty string rejected as record ID"):
        EvaluationPartition._extract_record_ids(df_empty_id, source_override="test_src")

    df_ws_id = pd.DataFrame({"year": [2022]}, index=["   "])
    with pytest.raises(ValueError, match="Empty string rejected as record ID"):
        EvaluationPartition._extract_record_ids(df_ws_id, source_override="test_src")

    df_valid = pd.DataFrame({"year": [2022]}, index=["id1"])
    with pytest.raises(ValueError, match="Source identifier cannot be empty string"):
        EvaluationPartition._extract_record_ids(df_valid, source_override="")


def test_proba_7_row_sum_tol_and_real_lr():
    """PROBA-7: row_sum_tol contract validation and real LogisticRegression extraction."""
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    y = np.array([0, 0, 1, 1])

    clf = LogisticRegression(random_state=42).fit(X, y)

    # Valid extraction with real scikit-learn LogisticRegression
    probs = validate_and_extract_positive_probabilities(clf, X, expected_classes=(0, 1), pos_label=1)
    assert len(probs) == 4
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # Test invalid row_sum_tol
    with pytest.raises(ValueError, match="row_sum_tol must be"):
        validate_and_extract_positive_probabilities(clf, X, row_sum_tol=-0.01)

    with pytest.raises(ValueError, match="row_sum_tol must be"):
        validate_and_extract_positive_probabilities(clf, X, row_sum_tol=0.6)

    with pytest.raises(ValueError, match="row_sum_tol must be"):
        validate_and_extract_positive_probabilities(clf, X, row_sum_tol=float("nan"))


def test_identity_9_equivalent_multiyear_normalization():
    """IDENTITY-9: Multiple year columns with equivalent representations (2023 and '2023.0') normalize without conflict."""
    df_my = pd.DataFrame({"year": [2023], "source_year": ["2023.0"]}, index=[1])
    ids = EvaluationPartition._extract_record_ids(df_my, "s")
    assert len(ids) == 1
    assert "yr=4:2023" in ids[0]


def test_identity_10_source_override_validation():
    """IDENTITY-10: NaN, Inf, and boolean source identifiers are strictly rejected."""
    df = pd.DataFrame({"year": [2022]}, index=[1])
    with pytest.raises(ValueError, match="Non-finite source identifier"):
        EvaluationPartition._extract_record_ids(df, float("nan"))
    with pytest.raises(ValueError, match="Non-finite source identifier"):
        EvaluationPartition._extract_record_ids(df, float("inf"))
    with pytest.raises(TypeError, match="Boolean value"):
        EvaluationPartition._extract_record_ids(df, True)


def test_identity_11_none_id_rejected():
    """IDENTITY-11: _normalize_id strictly rejects None with ValueError."""
    with pytest.raises(ValueError, match="Record ID cannot be None, NaN, or missing"):
        EvaluationPartition._normalize_id(None)


def test_identity_12_sanitized_leakage_error_message():
    """IDENTITY-12: Data leakage errors report counts and do not leak raw set/dict/tuple dumps of record IDs."""
    df_fit = pd.DataFrame({"year": [2022], "f1": [1.0]}, index=[1])
    y_fit = pd.Series([0], index=[1])
    with pytest.raises(ValueError) as excinfo:
        EvaluationPartition(
            fit_X=df_fit,
            fit_y=y_fit,
            selection_X=df_fit,
            selection_y=y_fit,
            fit_source=None,
            selection_source="study",
        )
    msg = str(excinfo.value)
    assert "Data leakage detected" in msg
    assert "share 1 record IDs" in msg
    assert "{" not in msg, "Raw set/dict dump of records must not appear in exception message"


def test_t_d8_terminal_schema_and_expected_groups():
    """T-D8: compute_application_group_fairness -> compute_group_fairness_gaps -> real LR evaluate_representation.

    Must return complete schema without KeyError: 'expected_groups', including expected_groups,
    expected_groups_source, primary_result_eligible, and primary_estimand_declared.
    """
    fit = pd.DataFrame({"x": [0.0, 0.1, 0.2, 0.8, 0.9, 1.0]})
    yf = np.array([0, 0, 0, 1, 1, 1])
    val = pd.DataFrame({"x": [0.1, 0.9, 0.2, 0.8]})
    yv = np.array([0, 1, 0, 1])
    gf = np.array(["a", "b", "a", "b", "a", "b"])
    gv = np.array(["a", "a", "b", "b"])
    e = FairEvaluator(FairBiasConfig(mds_fixed_components=2))
    res = evaluate_representation(
        LogisticRegression(random_state=42),
        MinMaxScaler(),
        fit,
        yf,
        val,
        yv,
        val,
        yv,
        gf,
        gv,
        gv,
        {},
        e,
        FairTransform(),
        [],
        ["x"],
        "protected",
        expected_groups=["a", "b", "missing"],
    )
    val_res = res["validation"]
    assert "expected_groups" in val_res, "Validation output must contain expected_groups"
    assert "expected_groups_source" in val_res, "Validation output must contain expected_groups_source"
    assert val_res["expected_groups"] == ["a", "b", "missing"]
    assert "equalized_odds_gap" in val_res
    assert "is_primary_estimand" in val_res
    assert "primary_result_eligible" in val_res
    assert "primary_estimand_declared" in val_res
    assert val_res["primary_estimand_declared"] is True
    assert val_res["primary_result_eligible"] is False, "Missing group in validation partition makes primary result ineligible"


def test_t_group_universe_and_non_finite_rejection():
    """T-GROUP: Unexpected group rejection, full declaration, missing group null with reason, non-finite rejection."""
    y = np.array([1, 0, 1, 0, 1, 0])
    p = np.array([1, 0, 1, 0, 0, 1])
    g = np.array(["a", "a", "b", "b", "c", "c"])

    # a/b/c 3 groups, only declaring a/b must be rejected (cannot silently exclude group c)
    with pytest.raises(ValueError, match="Observed unexpected groups not declared"):
        compute_application_group_fairness(y, p, g, expected_groups=["a", "b"])

    # Full declaration a/b/c yields EO=1.0 and primary_result_eligible=True
    full = compute_application_group_fairness(y, p, g, expected_groups=["a", "b", "c"])
    assert full["equalized_odds_gap"] == 1.0
    assert full["is_primary_estimand"] is True
    assert full["primary_result_eligible"] is True
    assert full["primary_estimand_declared"] is True

    # Missing group d yields EO=None, primary_result_eligible=False, status indicates missing
    missing = compute_application_group_fairness(y, p, g, expected_groups=["a", "b", "c", "d"])
    assert missing["equalized_odds_gap"] is None
    assert missing["equalized_odds_estimable"] is False
    assert "MISSING_EXPECTED_GROUPS" in str(missing["equalized_odds_status"])
    assert missing["primary_result_eligible"] is False
    assert missing["is_primary_estimand"] is False
    assert missing["primary_estimand_declared"] is True

    # String as expected_groups must be rejected with TypeError (cannot treat 'abc' as ['a', 'b', 'c'])
    with pytest.raises(TypeError, match="expected_groups must be a non-string sequence"):
        compute_application_group_fairness(y, p, g, expected_groups="abc")

    # Decimal Infinity in protected attribute must be rejected with ValueError
    with pytest.raises(ValueError, match="non-finite"):
        compute_application_group_fairness(
            np.array([1, 0, 1, 0]),
            np.array([1, 0, 1, 1]),
            np.array([0, 0, Decimal("Infinity"), Decimal("Infinity")], dtype=object),
            [0, Decimal("Infinity")],
        )


def test_r4_07_non_scalar_group_types_rejected():
    """R4-07: Non-scalar group types (frozenset, containers, arbitrary objects) must be rejected."""
    y = np.array([1, 0, 1, 0])
    p = np.array([1, 0, 1, 0])

    # 1. frozenset in expected_groups rejected with TypeError
    with pytest.raises(TypeError, match="group.*scalar"):
        compute_application_group_fairness(
            y, p, np.array(["a", "a", "b", "b"]),
            expected_groups=[frozenset([1]), frozenset([2])],
        )

    # 2. frozenset in protected_vals rejected with TypeError
    with pytest.raises(TypeError, match="group.*scalar"):
        compute_application_group_fairness(
            y, p,
            np.array([frozenset([1]), frozenset([1]), frozenset([2]), frozenset([2])], dtype=object),
            expected_groups=[frozenset([1]), frozenset([2])],
        )

    # 3. Arbitrary container (list/dict) as group elements rejected with TypeError
    with pytest.raises(TypeError, match="group.*scalar"):
        compute_application_group_fairness(
            y, p, np.array(["a", "a", "b", "b"]),
            expected_groups=[[1], [2]],
        )

    class CustomGroupObj:
        def __init__(self, val):
            self.val = val

    with pytest.raises(TypeError, match="group.*scalar"):
        compute_application_group_fairness(
            y, p,
            np.array([CustomGroupObj(1), CustomGroupObj(1), CustomGroupObj(2), CustomGroupObj(2)], dtype=object),
            expected_groups=[CustomGroupObj(1), CustomGroupObj(2)],
        )

    # 4. Valid scalar groups produce strictly JSON-serializable outputs
    res = compute_application_group_fairness(y, p, np.array([0, 0, 1, 1]), expected_groups=[0, 1])
    dumped = json.dumps(res)
    assert dumped is not None

    # 5. Decimal values (finite, underflow, infinite) explicitly rejected to prevent lossy conversion and collision (R6-05)
    from decimal import Decimal
    y_bin = np.array([0, 1, 0, 1])
    p_bin = np.array([0, 1, 0, 1])

    # 5a. Finite Decimals rejected with expected_groups
    with pytest.raises(TypeError, match="Decimal or non-standard numeric type"):
        compute_application_group_fairness(
            y_bin, p_bin,
            np.array([Decimal("1"), Decimal("1"), Decimal("2"), Decimal("2")], dtype=object),
            expected_groups=[Decimal("1"), Decimal("2")],
        )

    # 5b. Finite Decimals rejected without expected_groups
    with pytest.raises(TypeError, match="Decimal or non-standard numeric type"):
        compute_application_group_fairness(
            y_bin, p_bin,
            np.array([Decimal("1.25"), Decimal("1.25"), Decimal("2.25"), Decimal("2.25")], dtype=object),
        )

    # 5c. Underflow Decimals 1e-1000 / 2e-1000 rejected both with and without expected_groups
    with pytest.raises(TypeError, match="Decimal or non-standard numeric type"):
        compute_application_group_fairness(
            y_bin, p_bin,
            np.array([Decimal("1e-1000"), Decimal("1e-1000"), Decimal("2e-1000"), Decimal("2e-1000")], dtype=object),
            expected_groups=[Decimal("1e-1000"), Decimal("2e-1000")],
        )
    with pytest.raises(TypeError, match="Decimal or non-standard numeric type"):
        compute_application_group_fairness(
            y_bin, p_bin,
            np.array([Decimal("1e-1000"), Decimal("1e-1000"), Decimal("2e-1000"), Decimal("2e-1000")], dtype=object),
        )

    # 5d. Infinite Decimal rejected (non-finite ValueError)
    with pytest.raises((TypeError, ValueError), match="(Decimal|non-finite)"):
        compute_application_group_fairness(
            y_bin, p_bin,
            np.array([Decimal("Infinity"), Decimal("Infinity"), 1, 1], dtype=object),
            expected_groups=[Decimal("Infinity"), 1],
        )

    # 6. JSON key string collision: mixed numeric and string object array [1, '1'] (R6-05)
    # 6a. Rejected with expected_groups=[1, '1']
    with pytest.raises(ValueError, match="collision under JSON string conversion"):
        compute_application_group_fairness(
            y_bin, p_bin,
            np.array([1, 1, "1", "1"], dtype=object),
            expected_groups=[1, "1"],
        )

    # 6b. Rejected without expected_groups (inferred unique_in_data)
    with pytest.raises(ValueError, match="collision under JSON string conversion"):
        compute_application_group_fairness(
            y_bin, p_bin,
            np.array([1, 1, "1", "1"], dtype=object),
        )

    # 7. Common string, integer, and float groups roundtrip losslessly without loss of identity
    for valid_groups in ([0, 1], ["group_a", "group_b"], [10.5, 20.5]):
        g0, g1 = valid_groups[0], valid_groups[1]
        prot_valid = np.array([g0, g0, g1, g1], dtype=object)
        res_valid = compute_application_group_fairness(
            y_bin, p_bin, prot_valid, expected_groups=[g0, g1]
        )
        assert res_valid["is_primary_estimand"] is True
        dumped_str = json.dumps(res_valid)
        loaded_dict = json.loads(dumped_str)
        assert len(loaded_dict["group_selection_rates"]) == 2
        assert len(loaded_dict["group_tprs"]) == 2
        assert len(loaded_dict["group_fprs"]) == 2
        assert loaded_dict["demographic_parity_difference"] == res_valid["demographic_parity_difference"]

    # 8. Arbitrary custom object with is_finite method rejected with TypeError
    class CustomWithIsFinite:
        def is_finite(self):
            return True

    with pytest.raises(TypeError, match="Decimal or non-standard numeric type"):
        compute_application_group_fairness(
            y, p, np.array([CustomWithIsFinite(), CustomWithIsFinite(), 1, 1], dtype=object),
            expected_groups=[CustomWithIsFinite(), 1],
        )

    # 9. Frozenset / sequence as group scalar rejected with TypeError
    with pytest.raises(TypeError, match="non-scalar type"):
        compute_application_group_fairness(
            y, p, np.array([frozenset([1]), frozenset([1]), 2, 2], dtype=object),
            expected_groups=[frozenset([1]), 2],
        )


def test_t_json_key_collision_prevention():
    """R7-06 / R8-01: Prevents group key collisions under JSON serialization and preserves distinct strings."""
    y = np.array([0, 1, 0, 1])
    p = np.array([0, 0, 1, 1])

    # 1. Collision between boolean True and string 'true' rejected across both expected_groups and data
    with pytest.raises(ValueError, match="Group identifier collision under JSON string conversion"):
        compute_application_group_fairness(
            y, p, np.array([True, True, "true", "true"], dtype=object),
            expected_groups=[True, "true"],
        )
    with pytest.raises(ValueError, match="Group identifier collision under JSON string conversion"):
        compute_application_group_fairness(
            y, p, np.array([True, True, "true", "true"], dtype=object),
        )

    # 2. Collision between integer 1 and string '1' rejected
    with pytest.raises(ValueError, match="Group identifier collision under JSON string conversion"):
        compute_application_group_fairness(
            y, p, np.array([1, 1, "1", "1"], dtype=object),
            expected_groups=[1, "1"],
        )
    with pytest.raises(ValueError, match="Group identifier collision under JSON string conversion"):
        compute_application_group_fairness(
            y, p, np.array([1, 1, "1", "1"], dtype=object),
        )

    # 3. Case-distinct strings 'True' and 'true' are distinct valid JSON keys and must NOT collide (R8-01)
    res_str_case = compute_application_group_fairness(
        y, p, np.array(["True", "True", "true", "true"]),
        expected_groups=["True", "true"],
    )
    assert res_str_case["dp_estimable"] is True
    assert set(res_str_case["group_selection_rates"].keys()) == {"True", "true"}
    s_dump = json.dumps(res_str_case)
    s_load = json.loads(s_dump)
    assert len(s_load["group_selection_rates"]) == 2
    assert "True" in s_load["group_selection_rates"]
    assert "true" in s_load["group_selection_rates"]

    # 4. Data-inferred case-distinct strings also roundtrip cleanly (R8-01)
    res_inferred_case = compute_application_group_fairness(
        y, p, np.array(["True", "True", "true", "true"]),
    )
    assert res_inferred_case["dp_estimable"] is True
    assert set(res_inferred_case["group_selection_rates"].keys()) == {"True", "true"}

    # 5. Strings with spaces are preserved intact (R8-01)
    res_spaces = compute_application_group_fairness(
        y, p, np.array([" group 1 ", " group 1 ", "group 2", "group 2"]),
        expected_groups=[" group 1 ", "group 2"],
    )
    assert " group 1 " in res_spaces["group_selection_rates"]


def test_t_numpy_integer_group_json_roundtrip():
    """R7-06: Numpy integer scalars are normalized to standard Python int and roundtrip losslessly through JSON."""
    y = np.array([0, 1, 0, 1])
    p = np.array([0, 0, 1, 1])

    # 1. Expected groups with numpy integers
    res_expected = compute_application_group_fairness(
        y, p, np.array([0, 0, 1, 1]),
        expected_groups=[np.int64(0), np.int64(1)],
    )
    assert res_expected["dp_estimable"] is True
    assert res_expected["primary_result_eligible"] is True
    for k in res_expected["group_selection_rates"].keys():
        assert isinstance(k, (int, np.integer))
    dump_expected = json.dumps(res_expected)
    loaded_expected = json.loads(dump_expected)
    assert len(loaded_expected["group_selection_rates"]) == 2
    assert loaded_expected["demographic_parity_difference"] == res_expected["demographic_parity_difference"]

    # 2. Data-inferred groups with numpy integers
    res_inferred = compute_application_group_fairness(
        y, p, np.array([np.int64(0), np.int64(0), np.int64(1), np.int64(1)]),
    )
    assert res_inferred["dp_estimable"] is True
    dump_inferred = json.dumps(res_inferred)
    loaded_inferred = json.loads(dump_inferred)
    assert len(loaded_inferred["group_selection_rates"]) == 2






