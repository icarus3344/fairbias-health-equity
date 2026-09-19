"""Synthetic checks of the explicit two-attempt MDS iteration-budget policy."""
import inspect

import numpy as np
import pytest
from sklearn.manifold import MDS

from fairbias import bias_metric
from nhis_fairbias.benchmark.adapters.geometry_audit import audited_mds
from nhis_fairbias.benchmark.mds_budget_retry import mds_budget_retry


def distance():
    return np.array([[0., 1., 1.], [1., 0., np.sqrt(2.)], [1., np.sqrt(2.), 0.]])


def params(**kwargs):
    return dict(n_components=2, n_init=4, random_state=7, dissimilarity="precomputed",
                max_iter=kwargs.pop("max_iter", 3000), eps=kwargs.pop("eps", 1e-6),
                normalized_stress="auto", **kwargs)


def assert_rng_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_no_cap_preserves_outputs_parameters_and_global_rng_bitwise():
    D = distance()
    baseline = MDS(**params())
    expected = baseline.fit_transform(D)
    before = np.random.get_state()
    audit = []
    with mds_budget_retry() as stats:
        with audited_mds(audit, lambda: 1):
            actual_model = bias_metric.MDS(**params())
            actual = actual_model.fit_transform(D)
    assert actual.tobytes() == expected.tobytes()
    assert actual_model.stress_ == baseline.stress_
    assert actual_model.n_iter_ == baseline.n_iter_
    assert actual_model.get_params() == baseline.get_params()
    assert_rng_equal(before, np.random.get_state())
    assert len(stats["attempts"]) == 1 and not stats["fits"][0]["retried"]
    assert stats["attempts"][0]["status"] == "CONVERGED_BEFORE_CAP"
    assert audit[0]["max_iter"] == 3000
    assert set(stats["source_hashes"]) == {"retry_module", "MDS_fit_transform", "MDS_fit", "smacof", "smacof_single"}
    assert all(len(value) == 64 for value in stats["source_hashes"].values())


def test_real_selected_cap_restarts_fresh_and_outer_audit_sees_actual_cap(monkeypatch):
    # With a loose synthetic tolerance, SMACOF stops after two iterations.
    # A cap of two is still rejected by the established conservative audit.
    # The same seed/tolerance with cap four supplies an unambiguous receipt.
    D = distance()
    first_params = params(max_iter=2, eps=1e6)
    expected_model = MDS(**dict(first_params, max_iter=4))
    expected = expected_model.fit_transform(D)
    real_fit = MDS.fit_transform
    seen = []

    def observed(self, X, y=None, init=None):
        seen.append((id(self), self.get_params(deep=False).copy(), init))
        return real_fit(self, X, y=y, init=init)

    monkeypatch.setattr(MDS, "fit_transform", observed)
    audit = []
    with mds_budget_retry() as stats:
        with audited_mds(audit, lambda: 35):
            fitted = bias_metric.MDS(**first_params)
            actual = fitted.fit_transform(D)
    assert len(seen) == 2 and seen[0][0] != seen[1][0]
    assert seen[0][2] is None and seen[1][2] is None
    assert seen[1][1] == dict(seen[0][1], max_iter=4)
    assert actual.tobytes() == expected.tobytes()
    assert fitted.stress_ == expected_model.stress_ and fitted.n_iter_ == expected_model.n_iter_
    assert fitted.get_params() == expected_model.get_params()
    assert [a["max_iter"] for a in stats["attempts"]] == [2, 4]
    assert [a["status"] for a in stats["attempts"]] == ["MDS_ITERATION_CAP", "CONVERGED_BEFORE_CAP"]
    assert stats["attempts"][0]["input_distance_sha256"] == stats["attempts"][1]["input_distance_sha256"]
    assert len(stats["attempts"][0]["input_distance_sha256"]) == 64
    np.testing.assert_array_equal(D, distance())
    assert stats["recoveries"] == 1 and stats["fits"][0]["complete"]
    assert audit[0]["geometry_call"] == 35 and audit[0]["max_iter"] == 4
    assert audit[0]["status"] == "CONVERGED_BEFORE_CAP"


def test_retry_still_capped_raises_original_outer_audit_failure():
    audit = []
    with mds_budget_retry() as stats:
        with audited_mds(audit, lambda: 96):
            fitted = bias_metric.MDS(**params(max_iter=1, eps=1e-10))
            with pytest.raises(RuntimeError, match="MDS_ITERATION_CAP"):
                fitted.fit_transform(distance())
    assert fitted.max_iter == 2
    assert len(stats["attempts"]) == 2 and stats["recoveries"] == 0
    assert all(a["status"] == "MDS_ITERATION_CAP" for a in stats["attempts"])
    assert audit[0]["max_iter"] == 2 and audit[0]["status"] == "MDS_ITERATION_CAP"
    assert not stats["fits"][0]["complete"]


@pytest.mark.parametrize("failure_kind", ["points", "stress"])
def test_nonfinite_first_attempt_never_retries(monkeypatch, failure_kind):
    real_fit = MDS.fit_transform
    calls = []

    def nonfinite(self, X, y=None, init=None):
        points = real_fit(self, X, y=y, init=init)
        calls.append(1)
        if failure_kind == "points":
            points[0, 0] = np.nan
        else:
            self.stress_ = np.nan
        return points

    monkeypatch.setattr(MDS, "fit_transform", nonfinite)
    with mds_budget_retry() as stats:
        with audited_mds([], lambda: 1):
            with pytest.raises(RuntimeError, match="nonfinite MDS"):
                bias_metric.MDS(**params(max_iter=1)).fit_transform(distance())
    assert len(calls) == len(stats["attempts"]) == 1
    assert not stats["fits"][0]["retried"]
    assert stats["attempts"][0]["status"] == "NONFINITE_MDS"
    if failure_kind == "stress":
        assert stats["attempts"][0]["stress"] is None


@pytest.mark.parametrize("seed_kind", ["mutable", "unset"])
def test_mutable_and_unset_rngs_have_exact_single_call_behavior(seed_kind):
    D = distance()
    seed1 = np.random.RandomState(41) if seed_kind == "mutable" else None
    seed2 = np.random.RandomState(41) if seed_kind == "mutable" else None
    p = params(max_iter=1)
    p["random_state"] = seed1
    global_before = np.random.get_state()
    expected_model = MDS(**p)
    expected = expected_model.fit_transform(D)
    global_after = np.random.get_state()
    np.random.set_state(global_before)
    p["random_state"] = seed2
    with mds_budget_retry() as stats:
        actual_model = bias_metric.MDS(**p)
        actual = actual_model.fit_transform(D)
    assert actual.tobytes() == expected.tobytes()
    assert_rng_equal(global_after, np.random.get_state())
    if seed_kind == "mutable":
        assert_rng_equal(seed1.get_state(), seed2.get_state())
    assert len(stats["attempts"]) == 1
    assert stats["fits"][0]["skip_reason"] == "RNG_NOT_FIXED_INTEGER"
    assert actual_model.max_iter == 1


def test_explicit_fit_initialization_bypasses_retry():
    init = np.array([[.1, .2], [.4, .6], [.8, .5]])
    expected = MDS(**params(max_iter=1)).fit_transform(distance(), init=init.copy())
    with mds_budget_retry() as stats:
        actual = bias_metric.MDS(**params(max_iter=1)).fit_transform(distance(), init=init.copy())
    assert actual.tobytes() == expected.tobytes()
    assert len(stats["attempts"]) == 1
    assert stats["fits"][0]["skip_reason"] == "EXPLICIT_FIT_INITIALIZATION"


@pytest.mark.parametrize("initialization", ["random", "classical_mds"])
def test_explicit_constructor_initialization_bypasses_retry(initialization):
    if "init" not in inspect.signature(MDS).parameters:
        pytest.skip("This sklearn version has no constructor initialization parameter")
    p = params(max_iter=1, init=initialization)
    expected = MDS(**p).fit_transform(distance())
    with mds_budget_retry() as stats:
        actual = bias_metric.MDS(**p).fit_transform(distance())
    assert actual.tobytes() == expected.tobytes()
    assert len(stats["attempts"]) == 1
    assert stats["fits"][0]["skip_reason"] == "EXPLICIT_CONSTRUCTOR_INITIALIZATION"


@pytest.mark.parametrize("failure", [ValueError("synthetic solver error"), KeyboardInterrupt(), SystemExit(5)])
def test_errors_and_interrupts_preserve_attempt_and_restore_hook(monkeypatch, failure):
    original = bias_metric.MDS

    def fail(self, X, y=None, init=None):
        raise failure

    monkeypatch.setattr(MDS, "fit_transform", fail)
    with pytest.raises(type(failure)):
        with mds_budget_retry() as stats:
            bias_metric.MDS(**params()).fit_transform(distance())
    assert bias_metric.MDS is original
    assert len(stats["attempts"]) == 1 and not stats["fits"][0]["complete"]
    assert stats["attempts"][0]["error_type"] == type(failure).__name__
    with mds_budget_retry():
        pass


def test_second_attempt_error_never_falls_back_to_initial_capped_result(monkeypatch):
    real_fit = MDS.fit_transform

    def second_fails(self, X, y=None, init=None):
        if self.max_iter == 4:
            raise ValueError("synthetic second attempt error")
        return real_fit(self, X, y=y, init=init)

    monkeypatch.setattr(MDS, "fit_transform", second_fails)
    with pytest.raises(ValueError, match="second attempt"):
        with mds_budget_retry() as stats:
            bias_metric.MDS(**params(max_iter=2, eps=1e6)).fit_transform(distance())
    assert [a["status"] for a in stats["attempts"]] == ["MDS_ITERATION_CAP", "ERROR"]
    assert stats["recoveries"] == 0 and not stats["fits"][0]["complete"]
    assert bias_metric.MDS is MDS


def test_resource_baseexception_is_interrupted_and_not_solver_error(monkeypatch):
    class PilotHardStop(BaseException):
        pass

    def fail(self, X, y=None, init=None):
        raise PilotHardStop()

    monkeypatch.setattr(MDS, "fit_transform", fail)
    with pytest.raises(PilotHardStop):
        with mds_budget_retry() as stats:
            bias_metric.MDS(**params()).fit_transform(distance())
    assert stats["attempts"][0]["status"] == "INTERRUPTED"
    assert stats["fits"][0]["status"] == "INTERRUPTED"
    assert bias_metric.MDS is MDS


def test_unknown_sklearn_version_is_rejected_before_hook(monkeypatch):
    import sklearn
    monkeypatch.setattr(sklearn, "__version__", "unreviewed")
    with pytest.raises(RuntimeError, match="reviewed sklearn 1.9.1"):
        with mds_budget_retry():
            pass
    assert bias_metric.MDS is MDS


def test_nesting_and_incorrect_audit_order_are_rejected_without_losing_outer_hook():
    original = bias_metric.MDS
    with mds_budget_retry():
        outer = bias_metric.MDS
        with pytest.raises(RuntimeError, match="not reentrant"):
            with mds_budget_retry():
                pass
        assert bias_metric.MDS is outer
    assert bias_metric.MDS is original
    with audited_mds([], lambda: 1):
        outer = bias_metric.MDS
        with pytest.raises(RuntimeError, match="outside audited_mds"):
            with mds_budget_retry():
                pass
        assert bias_metric.MDS is outer
    assert bias_metric.MDS is original


@pytest.mark.parametrize("initial_cap", [10000, 20000])
def test_final_and_elbow_cap_policy_has_only_one_double_budget(monkeypatch, initial_cap):
    real_fit = MDS.fit_transform

    def force_first_selected_cap(self, X, y=None, init=None):
        points = real_fit(self, X, y=y, init=init)
        if self.max_iter == initial_cap:
            self.n_iter_ = initial_cap
        return points

    monkeypatch.setattr(MDS, "fit_transform", force_first_selected_cap)
    with mds_budget_retry() as stats:
        with audited_mds([], lambda: 1):
            fitted = bias_metric.MDS(**params(max_iter=initial_cap, eps=1e6))
            fitted.fit(distance())  # Exercise MDS.fit -> overridden fit_transform.
    assert fitted.max_iter == 2 * initial_cap
    assert [a["max_iter"] for a in stats["attempts"]] == [initial_cap, 2 * initial_cap]
    assert len(stats["attempts"]) == 2
