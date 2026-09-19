import copy

import numpy as np
import pandas as pd
import pytest

from fairbias import bias_metric
from fairbias.bias_metric import compute_bias_concentration
from nhis_fairbias.benchmark.joint_budget_optimization import (
    exact_geometry_acceleration,
    make_checked_mds_distances,
    memoized_geometry,
)


def _geometry_inputs(seed=11, n=18):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"x0": rng.normal(size=n), "x1": rng.normal(size=n)},
                     index=pd.Index(np.arange(n), name="rid"))
    o = pd.DataFrame({"A": np.tile([0, 1], n // 2)}, index=x.index)
    return x, o


def _fake_geometry(X, O, random_state=0, **kwargs):
    _ = O, random_state, kwargs
    return {c: float(i + 1) for i, c in enumerate(X.columns)}


def test_memoized_geometry_identity_invalidation_and_result_copy():
    stats = {}
    calls = []

    def original(X, O, sample_weight=None, random_state=0, **kwargs):
        calls.append((X.copy(), O.copy(), None if sample_weight is None else np.array(sample_weight), random_state, kwargs))
        return {c: float(i + 1) for i, c in enumerate(X.columns)}

    wrapped = memoized_geometry(original, stats, max_entries=8)
    X, O = _geometry_inputs()
    w = np.ones(len(X))
    first = wrapped(X, O, sample_weight=w, random_state=3, mds_fixed_components=2)
    first["x0"] = 999.0
    second = wrapped(X, O, sample_weight=w, random_state=3, mds_fixed_components=2)
    assert second["x0"] == 1.0
    assert len(calls) == 1 and stats["hits"] == 1

    for changed in [
        X.assign(x0=X.x0 + 1),
        X.rename(index={0: 100}),
        X.astype({"x0": "float32"}),
        X.rename(columns={"x0": "renamed"}),
    ]:
        wrapped(changed, O, sample_weight=w, random_state=3, mds_fixed_components=2)
    wrapped(X, O, sample_weight=w * 2, random_state=3, mds_fixed_components=2)
    wrapped(X, O, sample_weight=w, random_state=4, mds_fixed_components=2)
    wrapped(X, O, sample_weight=w, random_state=3, mds_fixed_components=3)
    assert len(calls) == 8
    wrapped(X, O.assign(A=1 - O.A), sample_weight=w, random_state=3, mds_fixed_components=2)
    wrapped(X[["x1", "x0"]], O, sample_weight=w, random_state=3, mds_fixed_components=2)
    assert len(calls) == 10


def test_memoized_geometry_lru_no_failure_or_nonfinite_cache_and_rng_bypass():
    stats = {}
    calls = []

    def original(X, O, random_state=0, **kwargs):
        calls.append(random_state)
        if kwargs.get("fail"):
            raise RuntimeError("boom")
        if kwargs.get("nonfinite"):
            return {"x0": float("nan")}
        return {c: 1.0 for c in X.columns}

    wrapped = memoized_geometry(original, stats, max_entries=1)
    X, O = _geometry_inputs(n=4)
    with pytest.raises(RuntimeError, match="boom"):
        wrapped(X, O, random_state=1, fail=True)
    with pytest.raises(RuntimeError, match="boom"):
        wrapped(X, O, random_state=1, fail=True)
    wrapped(X, O, random_state=1, nonfinite=True)
    wrapped(X, O, random_state=1, nonfinite=True)
    rng = np.random.default_rng(1)
    wrapped(X, O, random_state=rng)
    wrapped(X, O, random_state=rng)
    wrapped(X, O, random_state=None)
    wrapped(X, O, random_state=None)
    wrapped(X, O, random_state=1)
    wrapped(X.assign(x0=X.x0 + 1), O, random_state=1)
    assert stats["hits"] == 0
    assert stats["bypasses"] == 4
    assert stats["evictions"] >= 1
    assert stats["entries"] <= 1


def test_exact_geometry_context_restores_on_exception_and_rejects_nesting():
    from sklearn.manifold import _mds
    original_distances = _mds.euclidean_distances
    original_geometry = bias_metric.compute_bias_concentration
    original_shapley = bias_metric.compute_shapley_distance_matrix
    with pytest.raises(KeyboardInterrupt):
        with exact_geometry_acceleration(use_plan=False, use_memo=True):
            assert bias_metric.compute_bias_concentration is not original_geometry
            raise KeyboardInterrupt()
    assert bias_metric.compute_bias_concentration is original_geometry
    assert bias_metric.compute_shapley_distance_matrix is original_shapley
    assert _mds.euclidean_distances is original_distances
    with exact_geometry_acceleration(use_plan=False):
        with pytest.raises(RuntimeError, match="not reentrant"):
            with exact_geometry_acceleration(use_plan=False):
                pass


@pytest.mark.parametrize("mode", ["BM_AE", "JOINT"])
def test_small_adapter_path_is_exactly_unchanged_with_scoped_acceleration(mode):
    from nhis_fairbias.benchmark.adapters.adapter_fairbias_ae import FairBiasAEAdapter

    X, O = _geometry_inputs(n=24)
    y = (X["x0"].to_numpy() > 0).astype(int)
    base = FairBiasAEAdapter(
        mode=mode, max_outer_iterations=1, max_utility_evaluations=20,
        max_geometry_evaluations=100, max_bm_steps=2,
        poly_exponents=(1 / 3, 3.0), random_state=1,
    )
    base.fit_development(X.iloc[:12], y[:12], O.iloc[:12], X.iloc[12:], y[12:], O.iloc[12:])
    with exact_geometry_acceleration(use_plan=False) as stats:
        accelerated = FairBiasAEAdapter(
            mode=mode, max_outer_iterations=1, max_utility_evaluations=20,
            max_geometry_evaluations=100, max_bm_steps=2,
            poly_exponents=(1 / 3, 3.0), random_state=1,
        )
        accelerated.fit_development(X.iloc[:12], y[:12], O.iloc[:12], X.iloc[12:], y[12:], O.iloc[12:])
    assert accelerated.changed_dict_ == base.changed_dict_
    assert accelerated.termination_reason_ == base.termination_reason_
    assert accelerated.candidate_traces_ == base.candidate_traces_
    assert accelerated.provenance_["geometry_evaluations"] == base.provenance_["geometry_evaluations"]
    assert np.array_equal(accelerated.predict(X.iloc[12:]), base.predict(X.iloc[12:]))
    assert accelerated.predict_event_probability(X.iloc[12:]).tobytes() == base.predict_event_probability(X.iloc[12:]).tobytes()
    assert stats["logical_budget_counts_unchanged"] is True


def test_checked_mds_distance_exact_and_fallback_contract():
    pytest.importorskip("sklearn")
    from sklearn.metrics.pairwise import _euclidean_distances, euclidean_distances

    stats = {}
    checked = make_checked_mds_distances(euclidean_distances, _euclidean_distances, stats)
    rng = np.random.default_rng(9)
    x = rng.normal(size=(12, 4)).astype(np.float64)
    assert np.array_equal(checked(x), euclidean_distances(x))
    assert stats.get("fast_calls", 0) == 1
    for bad in [
        rng.normal(size=(12, 4)).astype(np.float32),
        np.empty((0, 4), dtype=np.float64),
        np.array([[1.0, np.nan]], dtype=np.float64),
        np.array([[1.0, np.inf]], dtype=np.float64),
        np.array([[1.0, 2.0]]),
    ]:
        try:
            expected = euclidean_distances(bad)
        except ValueError:
            with pytest.raises(ValueError):
                checked(bad)
        else:
            assert np.array_equal(checked(bad), expected)
    assert np.array_equal(checked(x, x, squared=True), euclidean_distances(x, x, squared=True))
    from scipy.sparse import csr_matrix
    assert np.array_equal(checked(csr_matrix(x)), euclidean_distances(csr_matrix(x)))
    assert stats.get("fallback_calls", 0) >= 6


@pytest.mark.parametrize("seed,dimensions", [(0, 1), (7, 2), (19, 5), (37, 15)])
def test_actual_audited_mds_points_stress_iterations_exact(seed, dimensions):
    from sklearn.metrics.pairwise import euclidean_distances
    from nhis_fairbias.benchmark.adapters.geometry_audit import audited_mds
    d = euclidean_distances(np.random.default_rng(71).normal(size=(22, 4)))

    def fit(records):
        with audited_mds(records, lambda: 1):
            model = bias_metric.MDS(n_components=dimensions, random_state=seed,
                                    dissimilarity="precomputed", n_init=4,
                                    max_iter=10000, eps=1e-10, normalized_stress="auto")
            return model.fit_transform(d)
    before, after = [], []
    a = fit(before)
    with exact_geometry_acceleration() as stats:
        b = fit(after)
    assert np.array_equal(a, b)
    assert before == after  # includes stress, iterations, cap and convergence status
    assert stats["mds_distances"]["fast_calls"] > 0


@pytest.mark.parametrize("mode", ["BM_AE", "JOINT"])
def test_nonidentity_bm_path_and_geometry_budget_are_preserved(mode):
    from nhis_fairbias.benchmark.adapters.adapter_fairbias_ae import FairBiasAEAdapter
    n = 48
    X = pd.DataFrame({"agep_a": np.linspace(20., 80., n),
                      "pcnt18uptc": np.linspace(1., 12., n),
                      "status": ["a"] * 24 + ["b"] * 24,
                      "region": np.tile(["x", "y", "z"], 16)})
    y = np.tile([0, 1], 24)
    A = pd.DataFrame({"A": np.tile([1, 2], 24)})

    def fit(budget):
        adapter = FairBiasAEAdapter(mode=mode, epsilon_ratio=.25, max_outer_iterations=2,
                                   max_bm_steps=5, max_geometry_evaluations=budget, random_state=0)
        try:
            adapter.fit_development(X.iloc[:24], y[:24], A.iloc[:24], X.iloc[24:], y[24:], A.iloc[24:])
        except RuntimeError as exc:
            return adapter, str(exc)
        return adapter, None
    for budget in (1, 1000):
        a, error_a = fit(budget)
        with exact_geometry_acceleration() as stats:
            b, error_b = fit(budget)
        assert error_a == error_b
        assert a.termination_reason_ == b.termination_reason_
        assert a._geometry_evaluations_ == b._geometry_evaluations_
        if error_a is None:
            assert a.provenance_["bm_commits"] > 0
            assert a.candidate_traces_ == b.candidate_traces_
            assert a.changed_dict_ == b.changed_dict_
            assert np.array_equal(a.predict_event_probability(X.iloc[24:]), b.predict_event_probability(X.iloc[24:]))


def test_unreviewed_library_version_is_rejected_before_hooks(monkeypatch):
    import sklearn
    original = bias_metric.compute_bias_concentration
    monkeypatch.setattr(sklearn, "__version__", "future-unknown")
    with pytest.raises(RuntimeError, match="requires reviewed"):
        with exact_geometry_acceleration():
            pass
    assert bias_metric.compute_bias_concentration is original


def test_equal_geometry_with_different_pandas_blocks_hits_exact_memo():
    x = pd.DataFrame({"a": [1., 2., 3., 4.], "b": [4., 5., 6., 7.],
                      "c": ["one", "two", "one", "two"]})
    o = pd.Series([0, 1, 0, 1], name="A")
    changed = x.copy()
    changed["a"] = x["a"].copy()  # split numeric blocks without changing values
    changed["c"] = [v.encode().decode() for v in x["c"]]  # new string identities
    stats = {}
    wrapped = memoized_geometry(_fake_geometry, stats)
    assert wrapped(x, o) == wrapped(changed, o.copy())
    assert stats["hits"] == 1
    # Object scalar types and sub-ULP-scaled float changes are not conflated.
    changed["a"] = np.nextafter(changed["a"], np.inf)
    wrapped(changed, o)
    assert stats["misses"] == 2


def test_unusable_geometry_results_pass_through_without_cache_error():
    X, O = _geometry_inputs()
    for value in (None, "invalid", np.array([1., 2.])):
        def original(X, O, random_state=0):
            return {c: value for c in X.columns}
        stats = {}
        memo = memoized_geometry(original, stats)
        assert memo(X, O)["x0"] is value
        assert memo(X, O)["x0"] is value
        assert stats["entries"] == 0


def test_actual_nondrop_bm_commit_reuses_geometry_and_preserves_trace():
    from fairbias.config import FairBiasConfig, ALGORITHM_MODE_PAPER_FAITHFUL
    from fairbias.evaluator import FairEvaluator
    from fairbias.mitigation import FairBiasMitigation
    from fairbias.transform import FairTransform, calculate_nmi_dict
    X = pd.DataFrame({"cat": ["a"] * 12 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4
                             + ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 12,
                      "z": np.tile([0., 1.], 24)})
    O = pd.DataFrame({"A": [0] * 24 + [1] * 24})
    y = pd.Series(np.tile([0, 1], 24))
    def fit():
        evaluator = FairEvaluator(FairBiasConfig(algorithm_mode=ALGORITHM_MODE_PAPER_FAITHFUL,
                                  label_O=("A",), random_seed=0).resolved(),
                                  label_O=["A"], num_attrs=["z"], cate_attrs=["cat"])
        bm = FairBiasMitigation(evaluator, FairTransform(), ["A"], ["cat"], ["z"],
                               phi_threshold=100, power_sequence_policy="official_stream",
                               power_revisit_policy="restart")
        eps = evaluator.calculate_epsilon(X, O)
        result = bm.mitigate_step(X, y, O, calculate_nmi_dict(X, y), {}, eps,
                                 np.mean(list(eps["A"].values())))
        return result, bm.step_traces
    a, at = fit()
    with exact_geometry_acceleration(use_memo=True) as stats:
        b, bt = fit()
    assert a[1] == b[1] == {"cat": {"d": "a"}}
    assert a[0].equals(b[0]) and at == bt
    assert stats["geometry"]["hits"] >= 1


def test_default_does_not_pay_geometry_hash_overhead():
    original = bias_metric.compute_bias_concentration
    with exact_geometry_acceleration() as stats:
        assert bias_metric.compute_bias_concentration is original
    assert stats["geometry"] == {"enabled": False}
