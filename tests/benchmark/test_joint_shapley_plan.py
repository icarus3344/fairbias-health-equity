import numpy as np
import pandas as pd
import pytest

from fairbias.bias_metric import compute_shapley_distance_matrix
from nhis_fairbias.benchmark.joint_shapley_plan import (
    build_shapley_h1_plan,
    make_planned_shapley_distance_matrix,
)


def _table(seed=17, n_features=7, n_pairs=6):
    rng = np.random.default_rng(seed)
    names = [f"f{i}" for i in range(n_features)]
    values = rng.random((n_features, n_pairs))
    return pd.DataFrame(values, index=names, columns=[f"g{i}" for i in range(n_pairs)])


@pytest.mark.parametrize("aggregation", ["mean_pair", "author_max_pair"])
@pytest.mark.parametrize("feature_order", ["schema", "shuffled"])
def test_h1_plan_is_bitwise_identical_for_binary_and_multigroup(aggregation, feature_order):
    original = compute_shapley_distance_matrix
    planned = make_planned_shapley_distance_matrix(original)
    table = _table(n_features=6, n_pairs=1 if feature_order == "schema" else 6)
    features = list(table.index)
    if feature_order == "shuffled":
        features = [features[i] for i in [3, 0, 5, 2, 1, 4]]
    expected, expected_nodes = original(table, features, h_order=1, multigroup_aggregation=aggregation)
    got, got_nodes = planned(table, features, h_order=1, multigroup_aggregation=aggregation)
    assert got_nodes == expected_nodes
    assert np.array_equal(got, expected)


def test_h0_and_unsupported_h_fall_back_without_changing_original():
    table = _table()
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return compute_shapley_distance_matrix(*args, **kwargs)

    planned = make_planned_shapley_distance_matrix(original)
    for h in (0, 2, 99):
        expected = original(table, list(table.index), h_order=h)
        got = planned(table, list(table.index), h_order=h)
        assert np.array_equal(got[0], expected[0])
        assert got[1] == expected[1]
    assert len(calls) == 6


@pytest.mark.parametrize(
    "table, features, kwargs",
    [
        (pd.DataFrame(), [], {}),
        (_table().rename(index={"f0": "f1"}), list(_table().index), {}),
        (_table().assign(f_bad=np.nan), list(_table().index) + ["f_bad"], {}),
        (_table(), list(_table().index) + ["f0"], {}),
        (_table(), list(_table().index), {"multigroup_aggregation": "bad"}),
    ],
)
def test_invalid_empty_missing_inputs_delegate_to_original(table, features, kwargs):
    planned = make_planned_shapley_distance_matrix(compute_shapley_distance_matrix)
    try:
        expected = compute_shapley_distance_matrix(table, features, h_order=1, **kwargs)
    except Exception as exc:
        with pytest.raises(type(exc), match=str(exc)):
            planned(table, features, h_order=1, **kwargs)
    else:
        got = planned(table, features, h_order=1, **kwargs)
        assert np.array_equal(got[0], expected[0])
        assert got[1] == expected[1]


def test_plan_cache_is_bounded_and_context_is_schema_order_sensitive():
    planned = make_planned_shapley_distance_matrix(compute_shapley_distance_matrix, max_cached_plans=2)
    table = _table(n_features=5)
    for order in (list(table.index), list(reversed(table.index)), ["f0", "f2", "f1", "f3", "f4"]):
        planned(table, order, h_order=1)
    assert len(planned.plan_cache) <= 2
    assert build_shapley_h1_plan(list(table.index), list(table.index)).features == tuple(table.index)

