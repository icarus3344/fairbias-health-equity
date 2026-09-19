"""Synthetic tests for frozen benchmark preprocessing missingness semantics."""

import numpy as np
import pandas as pd
import pytest

from nhis_fairbias.benchmark.preprocessing import BenchmarkPreprocessor


def test_f_all_missing_numeric_is_dropped_and_c_value_cannot_reintroduce_it():
    pre = BenchmarkPreprocessor(["num", "code"], numerical_features=["num"])
    f = pd.DataFrame({"num": [np.nan, None, np.nan], "code": [-1, -2, -1]})
    pre.fit(f)
    assert pre.dropped_features_ == ["num"]
    assert "num" not in pre.transformed_feature_names_

    c1 = pd.DataFrame({"num": [999.0, 1.0], "code": [-1, -2]})
    c2 = pd.DataFrame({"num": [-999.0, np.nan], "code": [-1, -2]})
    assert np.array_equal(pre.transform(c1), pre.transform(c2))


def test_actual_missing_is_missing_even_if_absent_on_f_and_unseen_is_unknown():
    pre = BenchmarkPreprocessor(["code"], numerical_features=[])
    pre.fit(pd.DataFrame({"code": [-1, -2, -1]}))
    transformed = pre.transform(pd.DataFrame({"code": [np.nan, None, -3]}))
    names = list(pre.transformed_feature_names_)
    missing_idx = names.index("code_MISSING")
    unknown_idx = names.index("code_UNKNOWN")
    assert transformed[0, missing_idx] == 1.0
    assert transformed[1, missing_idx] == 1.0
    assert transformed[2, unknown_idx] == 1.0
    # Existing semantic negative codes remain distinct substantive levels.
    assert "code_-1" in names and "code_-2" in names


def test_nonfinite_numeric_values_fail_in_fit_and_transform():
    pre = BenchmarkPreprocessor(["num"], numerical_features=["num"])
    with pytest.raises(ValueError, match="non-finite"):
        pre.fit(pd.DataFrame({"num": [1.0, np.inf]}))

    pre.fit(pd.DataFrame({"num": [1.0, 2.0]}))
    with pytest.raises(ValueError, match="non-finite"):
        pre.transform(pd.DataFrame({"num": [np.nan, -np.inf]}))


def test_refit_clears_previous_vocabularies_and_dropped_state():
    pre = BenchmarkPreprocessor(["num", "code"], numerical_features=["num"])
    pre.fit(pd.DataFrame({"num": [1.0, 2.0], "code": ["old", "old"]}))
    assert pre.dropped_features_ == []
    assert "code_old" in pre.transformed_feature_names_

    pre.fit(pd.DataFrame({"num": [np.nan, np.nan], "code": ["new", "new"]}))
    assert pre.dropped_features_ == ["num"]
    assert "code_old" not in pre.transformed_feature_names_
    assert "code_new" in pre.transformed_feature_names_
    assert pre.transform(pd.DataFrame({"num": [4.0], "code": ["old"]})).shape[1] == 3


def test_duplicate_and_missing_columns_are_rejected():
    with pytest.raises(ValueError, match="feature_names.*duplicates"):
        BenchmarkPreprocessor(["x", "x"])

    pre = BenchmarkPreprocessor(["x"])
    with pytest.raises(ValueError, match="Missing required"):
        pre.fit(pd.DataFrame({"y": [1]}))
    with pytest.raises(ValueError, match="duplicate feature columns"):
        pre.fit(pd.DataFrame([[1, 2]], columns=["x", "x"]))


def test_numerical_features_empty_preserves_transformed_numerical_as_category():
    pre = BenchmarkPreprocessor(["agep_a"], numerical_features=[])
    output = pre.fit_transform(pd.DataFrame({"agep_a": ["20", "30"]}))
    assert output.shape == (2, 4)  # 2 substantive levels plus MISSING/UNKNOWN
    assert not pre.num_features
    assert "agep_a_20" in pre.transformed_feature_names_
    assert pre.transform(pd.DataFrame({"agep_a": ["40"]}))[0, -1] == 1.0

