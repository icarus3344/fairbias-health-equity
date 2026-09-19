"""Synthetic-only contracts for the opt-in exact NumPy MDS kernel."""
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.manifold import MDS, _mds
from sklearn.metrics import pairwise

from fairbias import bias_metric
from nhis_fairbias.benchmark.adapters.geometry_audit import audited_mds
from nhis_fairbias.benchmark.joint_budget_optimization import exact_geometry_acceleration
from nhis_fairbias.benchmark.joint_mds_numpy import (
    _DEPENDENCY_HASHES,
    make_numpy_mds_distances,
    numpy_mds_acceleration,
)


@pytest.mark.parametrize("n", [1, 2, 3, 17, 31, 129])
@pytest.mark.parametrize("dimensions", [1, 2, 7, 15, 64])
@pytest.mark.parametrize("scale", [0, 1e-140, 1e-5, 1, 1e50, 1e140])
def test_540_exact_distance_layout_scale_cases(n, dimensions, scale):
    """180 parameter combinations x C, Fortran and reversed layouts = 540."""
    x = np.random.default_rng(713).normal(size=(n, dimensions)) * scale
    stats = {}
    optimized = make_numpy_mds_distances(pairwise.euclidean_distances, stats)
    for array in (x, np.asfortranarray(x), x[::-1]):
        expected = pairwise.euclidean_distances(array)
        before = array.tobytes()
        assert optimized(array).tobytes() == expected.tobytes()
        assert array.tobytes() == before
    assert stats == {"fast_calls": 3, "fallback_calls": 0}


def test_fallback_keeps_public_validation_and_optional_argument_behavior():
    from scipy.sparse import csr_matrix

    class ArraySubclass(np.ndarray):
        pass

    x = np.random.default_rng(42).normal(size=(12, 4))
    cases = [
        (x.astype(np.float32), {}),
        (x.astype(object), {}),
        (x.tolist(), {}),
        (x.view(ArraySubclass), {}),
        (csr_matrix(x), {}),
        (np.ones((130, 2)), {}),
        (np.ones((2, 65)), {}),
        (np.empty((0, 4)), {}),
        (np.ones(4), {}),
        (np.array([[np.nan, 1.0]]), {}),
        (np.array([[np.inf, 1.0]]), {}),
        (x, {"Y": x.copy()}),
        (x, {"squared": True}),
        (x, {"squared": np.bool_(False)}),
        (x, {"X_norm_squared": np.einsum("ij,ij->i", x, x)}),
        (x, {"Y_norm_squared": np.einsum("ij,ij->i", x, x)}),
    ]
    stats = {}
    optimized = make_numpy_mds_distances(pairwise.euclidean_distances, stats)
    for value, kwargs in cases:
        try:
            expected = pairwise.euclidean_distances(value, **kwargs)
        except (TypeError, ValueError) as exc:
            with pytest.raises(type(exc)) as observed:
                optimized(value, **kwargs)
            assert str(observed.value) == str(exc)
        else:
            assert optimized(value, **kwargs).tobytes() == expected.tobytes()
    assert stats == {"fast_calls": 0, "fallback_calls": len(cases)}


def test_finiteness_is_checked_on_every_distance_request():
    stats = {}
    optimized = make_numpy_mds_distances(pairwise.euclidean_distances, stats)
    x = np.ones((3, 2))
    optimized(x)
    x[0, 0] = np.nan
    with pytest.raises(ValueError):
        optimized(x)
    assert stats == {"fast_calls": 1, "fallback_calls": 1}


@pytest.mark.parametrize("seed,dimensions", [(0, 1), (7, 2), (19, 5), (37, 15)])
def test_complete_audited_mds_coordinates_stress_iterations_are_exact(seed, dimensions):
    distance = pairwise.euclidean_distances(np.random.default_rng(71).normal(size=(22, 4)))

    def fit(records):
        with warnings.catch_warnings(), audited_mds(records, lambda: 1):
            warnings.simplefilter("ignore", FutureWarning)
            model = bias_metric.MDS(
                n_components=dimensions, random_state=seed, dissimilarity="precomputed",
                n_init=4, max_iter=10000, eps=1e-10, normalized_stress="auto",
            )
            return model.fit_transform(distance)

    before, after = [], []
    expected = fit(before)
    with numpy_mds_acceleration() as stats:
        actual = fit(after)
    assert expected.tobytes() == actual.tobytes()
    assert before == after
    assert stats["mds_distances"]["fast_calls"] > 0
    assert stats["mds_distances"]["fallback_calls"] == 0
    assert stats["geometry"]["enabled"] is False
    assert stats["logical_budget_counts_unchanged"] is True
    assert stats["version"] == "joint_numpy_mds_v1"
    assert len(stats["numpy_kernel_source_sha256"]) == 64
    assert set(_DEPENDENCY_HASHES) <= set(stats["sklearn_function_hashes"])


def test_mds_four_initializations_preserve_mutable_rng_and_normalized_stress():
    distance = pairwise.euclidean_distances(np.random.default_rng(21).normal(size=(12, 4)))
    before_rng = np.random.RandomState(7)
    after_rng = np.random.RandomState(7)

    def fit(rng):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            model = MDS(
                n_components=3, random_state=rng, dissimilarity="precomputed",
                n_init=4, max_iter=10000, eps=1e-10, normalized_stress=True,
            )
            points = model.fit_transform(distance)
        return points, model.stress_, model.n_iter_

    expected = fit(before_rng)
    with numpy_mds_acceleration():
        actual = fit(after_rng)
    assert expected[0].tobytes() == actual[0].tobytes()
    assert expected[1:] == actual[1:]
    assert pickle.dumps(before_rng.get_state()) == pickle.dumps(after_rng.get_state())


def test_iteration_cap_receipt_and_error_are_preserved():
    distance = pairwise.euclidean_distances(np.random.default_rng(15).normal(size=(12, 4)))

    def fit(records):
        with warnings.catch_warnings(), audited_mds(records, lambda: 3):
            warnings.simplefilter("ignore", FutureWarning)
            model = bias_metric.MDS(
                n_components=2, random_state=3, dissimilarity="precomputed",
                n_init=1, max_iter=1, eps=1e-10, normalized_stress="auto",
            )
            with pytest.raises(RuntimeError, match="MDS_ITERATION_CAP") as error:
                model.fit_transform(distance)
            return str(error.value)

    before, after = [], []
    expected = fit(before)
    with numpy_mds_acceleration():
        actual = fit(after)
    assert expected == actual
    assert before == after
    assert after[0]["status"] == "MDS_ITERATION_CAP"


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_context_restores_all_hooks_after_base_exceptions(exception):
    original = (_mds.euclidean_distances, bias_metric.compute_bias_concentration,
                bias_metric.compute_shapley_distance_matrix)
    with pytest.raises(exception):
        with numpy_mds_acceleration():
            raise exception("synthetic interruption")
    assert (_mds.euclidean_distances, bias_metric.compute_bias_concentration,
            bias_metric.compute_shapley_distance_matrix) == original
    with numpy_mds_acceleration():
        pass


@pytest.mark.parametrize("outer,inner", [
    (numpy_mds_acceleration, numpy_mds_acceleration),
    (numpy_mds_acceleration, exact_geometry_acceleration),
    (exact_geometry_acceleration, numpy_mds_acceleration),
])
def test_nested_contexts_are_rejected_without_changing_outer_hook(outer, inner):
    original = _mds.euclidean_distances
    with outer():
        installed = _mds.euclidean_distances
        with pytest.raises(RuntimeError):
            with inner():
                pass
        assert _mds.euclidean_distances is installed
    assert _mds.euclidean_distances is original


def test_unknown_sklearn_version_is_rejected_before_hooks(monkeypatch):
    import sklearn

    original = _mds.euclidean_distances
    monkeypatch.setattr(sklearn, "__version__", "unreviewed-version")
    with pytest.raises(RuntimeError, match="reviewed sklearn 1.9.1"):
        with numpy_mds_acceleration():
            pass
    assert _mds.euclidean_distances is original


@pytest.mark.parametrize("change_alias_too", [False, True])
def test_changed_transitive_dependency_is_rejected_before_hooks(monkeypatch, change_alias_too):
    from sklearn.utils import extmath

    original = _mds.euclidean_distances

    def changed_row_norms(*args, **kwargs):
        raise AssertionError("unreviewed helper must not execute")

    monkeypatch.setattr(extmath, "row_norms", changed_row_norms)
    if change_alias_too:
        monkeypatch.setattr(pairwise, "row_norms", changed_row_norms)
    with pytest.raises(RuntimeError, match="reviewed sklearn dependency"):
        with numpy_mds_acceleration():
            pass
    assert _mds.euclidean_distances is original


@pytest.mark.parametrize("mode", ["BM_AE", "JOINT"])
@pytest.mark.parametrize("geometry_budget", [1, 1000])
@pytest.mark.parametrize("fixture", ["numeric_drop", "categorical_merge"])
def test_real_adapter_commits_predictions_and_budget_termination_are_exact(mode, geometry_budget, fixture):
    from nhis_fairbias.benchmark.adapters.adapter_fairbias_ae import FairBiasAEAdapter

    if fixture == "numeric_drop":
        X = pd.DataFrame({
            "agep_a": np.linspace(20.0, 80.0, 48),
            "pcnt18uptc": np.linspace(1.0, 12.0, 48),
            "status": ["a"] * 24 + ["b"] * 24,
            "region": np.tile(["x", "y", "z"], 16),
        })
        y = np.tile([0, 1], 24)
        protected = pd.DataFrame({"A": np.tile([1, 2], 24)})
        split, epsilon_ratio = 24, 0.25
        expected_change = {"agep_a": "dropped", "pcnt18uptc": "dropped"}
    else:
        training = pd.DataFrame({
            "cat": ["a"] * 12 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4
                   + ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 12,
            "z": np.tile([0.0, 1.0], 24),
        })
        X = pd.concat([training, training], ignore_index=True)
        y = np.tile([0, 1], 48)
        protected = pd.DataFrame({"A": ([0] * 24 + [1] * 24) * 2})
        split, epsilon_ratio = 48, 0.75
        expected_change = {"cat": {"d": "a"}}

    def fit():
        model = FairBiasAEAdapter(
            mode=mode, epsilon_ratio=epsilon_ratio, max_outer_iterations=2,
            max_bm_steps=5, max_geometry_evaluations=geometry_budget, random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            try:
                model.fit_development(
                    X.iloc[:split], y[:split], protected.iloc[:split],
                    X.iloc[split:], y[split:], protected.iloc[split:],
                )
            except RuntimeError as exc:
                return model, str(exc)
        return model, None

    before, error_before = fit()
    with numpy_mds_acceleration() as stats:
        after, error_after = fit()
    assert error_before == error_after
    assert before.termination_reason_ == after.termination_reason_
    assert before._geometry_evaluations_ == after._geometry_evaluations_
    assert before._utility_evaluations_ == after._utility_evaluations_
    assert before.provenance_ == after.provenance_
    assert before.changed_dict_ == after.changed_dict_
    assert before.is_fitted_ == after.is_fitted_
    assert stats["mds_distances"]["fast_calls"] > 0
    if geometry_budget == 1:
        assert error_after is not None and "geometry evaluations" in error_after
        assert after.termination_reason_ == "BUDGET_EXHAUSTED"
        assert after.is_fitted_ is False
    else:
        assert error_after is None
        assert after.provenance_["bm_commits"] > 0
        assert after.changed_dict_ == expected_change
        assert before.candidate_traces_ == after.candidate_traces_
        assert before.predict(X.iloc[split:]).tobytes() == after.predict(X.iloc[split:]).tobytes()
        assert (before.predict_event_probability(X.iloc[split:]).tobytes()
                == after.predict_event_probability(X.iloc[split:]).tobytes())


def test_real_nondrop_bm_commit_and_geometry_trace_are_exact():
    from fairbias.config import ALGORITHM_MODE_PAPER_FAITHFUL, FairBiasConfig
    from fairbias.evaluator import FairEvaluator
    from fairbias.mitigation import FairBiasMitigation
    from fairbias.transform import FairTransform, calculate_nmi_dict

    X = pd.DataFrame({
        "cat": ["a"] * 12 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4
               + ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 12,
        "z": np.tile([0.0, 1.0], 24),
    })
    protected = pd.DataFrame({"A": [0] * 24 + [1] * 24})
    y = pd.Series(np.tile([0, 1], 24))

    def fit():
        evaluator = FairEvaluator(
            FairBiasConfig(
                algorithm_mode=ALGORITHM_MODE_PAPER_FAITHFUL,
                label_O=("A",), random_seed=0,
            ).resolved(),
            label_O=["A"], num_attrs=["z"], cate_attrs=["cat"],
        )
        engine = FairBiasMitigation(
            evaluator, FairTransform(), ["A"], ["cat"], ["z"],
            phi_threshold=100, power_sequence_policy="official_stream",
            power_revisit_policy="restart",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            epsilon = evaluator.calculate_epsilon(X, protected)
            result = engine.mitigate_step(
                X, y, protected, calculate_nmi_dict(X, y), {}, epsilon,
                np.mean(list(epsilon["A"].values())),
            )
        return result, engine.step_traces

    before, before_trace = fit()
    with numpy_mds_acceleration() as stats:
        after, after_trace = fit()
    assert before[1] == after[1] == {"cat": {"d": "a"}}
    assert before[0].equals(after[0])
    assert before[2:] == after[2:]
    assert before_trace == after_trace
    assert stats["geometry"]["enabled"] is False
    assert stats["mds_distances"]["fast_calls"] > 0
