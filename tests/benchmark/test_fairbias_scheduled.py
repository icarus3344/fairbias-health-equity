"""Synthetic controller probes plus a small, unmapped real-engine fit."""
import copy

import numpy as np
import pandas as pd
import pytest

import fairbias.enhancement as enhancement_module
from fairbias import bias_metric
from fairbias.enhancement_contracts import CandidateEvaluationResult
from nhis_fairbias.benchmark.adapters import adapter_fairbias_scheduled as scheduled


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def data():
    n = 24
    X = pd.DataFrame({"agep_a": np.linspace(.1, .9, n), "pcnt18uptc": np.linspace(.2, .8, n)})
    y = np.arange(n) % 2
    A = pd.DataFrame({"A": np.arange(n) % 2})
    return X.iloc[:12], y[:12], A.iloc[:12], X.iloc[12:], y[12:], A.iloc[12:]


def controller(monkeypatch, *, bm_stop=None, rescue=False, ae_action=None, geometry_action=None):
    seen = {"bm": 0, "ae": [], "geometry": 0}

    def geometry(self, X, O, *args, **kwargs):
        seen["geometry"] += 1
        if geometry_action:
            geometry_action(seen)
        power = round(np.log(float(X.iloc[0]["agep_a"])) / np.log(.1))
        return {"A": {"agep_a": max(0.0, 1.0 - .2 * (power - 1)), "pcnt18uptc": 0.0}}

    def bm(self, X, y, O, nmi, changed, eps, threshold, **kwargs):
        seen["bm"] += 1
        if bm_stop is not None and seen["bm"] > bm_stop:
            return X, changed, None, None
        result = copy.deepcopy(changed)
        result["agep_a"] = {"power": float(seen["bm"] + 1)}
        return X, result, "A", "agep_a"

    def ae(self, X, y, changed, O, threshold, eps, **kwargs):
        seen["ae"].append(seen["bm"])
        if ae_action:
            return ae_action(self, X, y, changed, O, threshold, eps, kwargs)
        if rescue and changed.get("agep_a", {}).get("power", 1) < 4:
            return X, {"agep_a": {"power": 4.0}}, "agep_a"
        return X, changed, None

    monkeypatch.setattr(scheduled.FairEvaluator, "calculate_epsilon", geometry)
    monkeypatch.setattr(scheduled.FairBiasMitigation, "mitigate_step", bm)
    monkeypatch.setattr(scheduled.FairAccuracyEnhancement, "enhance_step", ae)
    return seen


@pytest.mark.parametrize("interval,first_ae", [(1, 1), (3, 3)])
def test_schedule_interval_and_feasible_override(monkeypatch, interval, first_ae):
    seen = controller(monkeypatch)
    adapter = scheduled.ScheduledJointAdapter(ae_every_bm_commits=interval, monotonic=Clock())
    adapter.fit_development(*data())
    assert seen["ae"][0] == first_ae
    assert adapter._bm_commits == 3
    assert adapter.termination_reason_ == "SEARCH_EXHAUSTED"
    assert adapter.convergence_verified_
    assert adapter.changed_dict_ == {"agep_a": {"power": 4.0}}
    assert adapter.provenance_["paper_equivalent"] is False
    if interval == 3:
        assert sum(t["status"] == "DEFERRED" for t in adapter.candidate_traces_) == 2


def test_bm_no_progress_forces_rescue_before_interval(monkeypatch):
    seen = controller(monkeypatch, bm_stop=1, rescue=True)
    adapter = scheduled.ScheduledJointAdapter(ae_every_bm_commits=8, monotonic=Clock())
    adapter.fit_development(*data())
    assert seen["ae"][0] == 2
    assert adapter._bm_commits == 1 and adapter._ae_commits == 1
    assert adapter.termination_reason_ == "SEARCH_EXHAUSTED"


def test_commit_limit_returns_verified_incumbent_not_next_proposal(monkeypatch):
    controller(monkeypatch, bm_stop=1, rescue=True)
    adapter = scheduled.ScheduledJointAdapter(max_outer_iterations=1, monotonic=Clock())
    adapter.fit_development(*data())
    assert adapter.is_fitted_
    assert adapter.termination_reason_ == "FEASIBLE_BUDGET_LIMITED"
    assert adapter.budget_reason_ == "AE_COMMIT_LIMIT"
    assert adapter._ae_commits == 1 and not adapter.convergence_verified_
    assert adapter.changed_dict_ == {"agep_a": {"power": 4.0}}
    assert np.isfinite(adapter.predict_event_probability(data()[3])).all()


def test_no_feasible_incumbent_at_commit_budget_fails(monkeypatch):
    controller(monkeypatch)
    adapter = scheduled.ScheduledJointAdapter(max_bm_steps=1, monotonic=Clock())
    with pytest.raises(RuntimeError, match="NO_FEASIBLE_BUDGET_LIMITED"):
        adapter.fit_development(*data())
    assert adapter.model_ is None and not adapter.is_fitted_
    assert not adapter.incumbent_available_
    assert adapter.provenance_["bm_commits"] == 1


@pytest.mark.parametrize("limit,success", [(4, False), (5, True)])
def test_geometry_counter_allows_exact_bound_without_extra_operation(monkeypatch, limit, success):
    seen = controller(monkeypatch)
    adapter = scheduled.ScheduledJointAdapter(max_geometry_evaluations=limit, monotonic=Clock())
    adapter.fit_development(*data())
    assert seen["geometry"] == limit
    assert adapter._geometry_evaluations_ == limit
    assert adapter.convergence_verified_ is success
    assert adapter.termination_reason_ == ("SEARCH_EXHAUSTED" if success else "FEASIBLE_BUDGET_LIMITED")


def test_deadline_at_boundary_prevents_initial_geometry(monkeypatch):
    clock = Clock()
    seen = controller(monkeypatch)
    original = scheduled.ScheduledJointAdapter._prepare_development

    def prepared(self, *args):
        result = original(self, *args)
        clock.now = 2.0
        return result

    monkeypatch.setattr(scheduled.ScheduledJointAdapter, "_prepare_development", prepared)
    adapter = scheduled.ScheduledJointAdapter(search_seconds=2, monotonic=clock)
    with pytest.raises(RuntimeError, match="SEARCH_TIME_LIMIT: before geometry"):
        adapter.fit_development(*data())
    assert seen["geometry"] == 0


def test_deadline_after_refresh_does_not_commit_candidate(monkeypatch):
    clock = Clock()

    def late(seen):
        if seen["geometry"] == 2:
            clock.now = 2.0

    controller(monkeypatch, geometry_action=late)
    adapter = scheduled.ScheduledJointAdapter(search_seconds=2, monotonic=clock)
    with pytest.raises(RuntimeError, match="after geometry"):
        adapter.fit_development(*data())
    assert adapter._bm_commits == 0
    assert adapter._current_changed == {}
    assert adapter.candidate_traces_[0]["status"] == "PENDING_GEOMETRY_REFRESH"
    assert adapter.candidate_traces_[0]["committed"] is False


@pytest.mark.parametrize("stage", ["before", "after"])
def test_utility_deadline_before_and_after_operation(monkeypatch, stage):
    clock = Clock()
    calls = []

    def result(self, partition, changed):
        calls.append(1)
        self._utility_evaluations_ += 1
        clock.now = 2.0
        return CandidateEvaluationResult("VALID", .7, utility_metric="balanced_accuracy", model_fit_count=1)

    monkeypatch.setattr(scheduled.FairBiasAEAdapter, "_utility", result)
    adapter = scheduled.ScheduledJointAdapter(search_seconds=2, monotonic=clock)
    adapter._search_deadline, adapter._utility_evaluations_ = 2.0, 0
    if stage == "before":
        clock.now = 2.0
    with pytest.raises(scheduled.StopBudget, match=stage + " utility"):
        adapter._utility(None, {})
    assert len(calls) == (stage == "after")


def test_utility_counter_exact_bound(monkeypatch):
    adapter = scheduled.ScheduledJointAdapter(max_utility_evaluations=1, monotonic=Clock())
    adapter._search_deadline, adapter._utility_evaluations_ = 2.0, 1
    with pytest.raises(scheduled.StopBudget, match="UTILITY_EVALUATION_LIMIT"):
        adapter._utility(None, {})
    assert adapter._utility_evaluations_ == 1


@pytest.mark.parametrize("failure,expected", [
    (RuntimeError("BUDGET_EXHAUSTED: MDS_ITERATION_CAP; convergence not verified"), "FEASIBLE_GEOMETRY_INCOMPLETE"),
    (ValueError("unexpected numerical failure"), "EVALUATION_FAILED"),
])
def test_candidate_geometry_failure_escapes_legacy_ae_catch(monkeypatch, failure, expected):
    original_utility, original_mds = enhancement_module.evaluate_candidate_utility, bias_metric.MDS

    def late(seen):
        if seen["geometry"] >= 5:
            raise failure

    def evaluate_candidate(self, X, y, changed, O, threshold, eps, kwargs):
        self._is_fairness_acceptable(X, O, threshold, .4)
        return X, changed, None

    controller(monkeypatch, geometry_action=late, ae_action=evaluate_candidate)
    adapter = scheduled.ScheduledJointAdapter(monotonic=Clock())
    if expected == "EVALUATION_FAILED":
        with pytest.raises(ValueError, match="unexpected numerical failure"):
            adapter.fit_development(*data())
        assert not adapter.is_fitted_ and adapter.model_ is None
    else:
        adapter.fit_development(*data())
        assert adapter.is_fitted_ and adapter.changed_dict_ == {"agep_a": {"power": 4.0}}
    assert adapter.termination_reason_ == expected
    assert not adapter.convergence_verified_
    assert enhancement_module.evaluate_candidate_utility is original_utility
    assert bias_metric.MDS is original_mds


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit(7)])
def test_external_interrupt_never_returns_incumbent_and_restores_hooks(monkeypatch, failure):
    original_utility, original_mds = enhancement_module.evaluate_candidate_utility, bias_metric.MDS

    def interrupt(*args):
        raise failure

    controller(monkeypatch, ae_action=interrupt)
    adapter = scheduled.ScheduledJointAdapter(monotonic=Clock())
    with pytest.raises(type(failure)):
        adapter.fit_development(*data())
    assert adapter.termination_reason_ == "INTERRUPTED"
    assert not adapter.is_fitted_ and adapter.model_ is None
    assert enhancement_module.evaluate_candidate_utility is original_utility
    assert bias_metric.MDS is original_mds


def test_small_real_engines_preserve_fit_selection_and_prediction_contract():
    adapter = scheduled.ScheduledJointAdapter(
        epsilon_ratio=10, poly_exponents=(3.0,), max_outer_iterations=1,
        max_geometry_evaluations=40, max_utility_evaluations=12, random_state=2)
    inputs = data()
    original = [x.copy() for x in inputs]
    adapter.fit_development(*inputs)
    assert adapter.is_fitted_ and adapter.incumbent_available_
    assert adapter.termination_reason_ in {"SEARCH_EXHAUSTED", "FEASIBLE_BUDGET_LIMITED"}
    assert all(row["utility_metric"] == "balanced_accuracy" for row in adapter.provenance_["ae_audit"])
    assert all(row["accepted"] == row["committed"] for row in adapter.provenance_["ae_audit"])
    assert adapter.provenance_["mds_fits"]
    for before, after in zip(original, inputs):
        if isinstance(before, pd.DataFrame):
            pd.testing.assert_frame_equal(before, after)
        else:
            np.testing.assert_array_equal(before, after)
    p = adapter.predict_event_probability(inputs[3])
    assert p.shape == (12,) and np.isfinite(p).all()
    np.testing.assert_array_equal(p, adapter.predict_event_probability(inputs[3]))


@pytest.mark.parametrize("backbone", ["LR", "GBDT"])
def test_interval_one_real_engine_matches_registered_completed_trajectory(backbone):
    params = dict(epsilon_ratio=10, poly_exponents=(3.0,), max_outer_iterations=10,
                  max_geometry_evaluations=100, max_utility_evaluations=30,
                  backbone=backbone, random_state=2)
    original = scheduled.FairBiasAEAdapter(mode="JOINT", **params)
    extension = scheduled.ScheduledJointAdapter(ae_every_bm_commits=1, **params)
    inputs = data()
    original.fit_development(*inputs)
    extension.fit_development(*inputs)
    assert original.termination_reason_ == "STRICT_FEASIBLE_SEARCH_EXHAUSTED"
    assert extension.termination_reason_ == "SEARCH_EXHAUSTED"
    assert original.changed_dict_ == extension.changed_dict_
    original_commits = [(t["engine"], t["feature"]) for t in original.candidate_traces_ if t["committed"]]
    extension_commits = [(t["engine"], t["feature"]) for t in extension.candidate_traces_ if t["committed"]]
    assert original_commits == extension_commits
    np.testing.assert_array_equal(original.predict_event_probability(inputs[3]),
                                  extension.predict_event_probability(inputs[3]))


def test_final_refit_failure_leaves_no_fitted_model(monkeypatch):
    controller(monkeypatch)
    adapter = scheduled.ScheduledJointAdapter(monotonic=Clock())
    original = adapter._estimator

    def estimator():
        model = original()

        def failed(*args, **kwargs):
            raise ValueError("final refit failed")

        model.fit = failed
        return model

    monkeypatch.setattr(adapter, "_estimator", estimator)
    with pytest.raises(ValueError, match="final refit failed"):
        adapter.fit_development(*data())
    assert adapter.model_ is None and not adapter.is_fitted_
    assert adapter.termination_reason_ == "EVALUATION_FAILED"
    assert not adapter.convergence_verified_


def test_budget_limited_return_rechecks_partition_mutation(monkeypatch):
    def mutate(self, X, y, changed, O, threshold, eps, kwargs):
        kwargs["partition"].selection_y.iloc[0] = 1 - kwargs["partition"].selection_y.iloc[0]
        raise scheduled.StopBudget("INJECTED_BUDGET")

    controller(monkeypatch, ae_action=mutate)
    adapter = scheduled.ScheduledJointAdapter(monotonic=Clock())
    with pytest.raises(ValueError, match="mutated"):
        adapter.fit_development(*data())
    assert not adapter.is_fitted_ and adapter.model_ is None
    assert adapter.termination_reason_ == "EVALUATION_FAILED"


def test_wrapped_mds_cap_is_recognized_but_ordinary_mds_error_is_not():
    try:
        try:
            raise RuntimeError("BUDGET_EXHAUSTED: MDS_ITERATION_CAP")
        except RuntimeError as exc:
            raise RuntimeError("MDS embedding failed") from exc
    except RuntimeError as wrapped:
        assert scheduled._is_mds_cap(wrapped)
    assert not scheduled._is_mds_cap(RuntimeError("MDS embedding failed"))


def test_failed_utility_is_not_masked_as_time_budget(monkeypatch):
    clock = Clock()

    def failed(self, partition, changed):
        clock.now = 2.0
        return CandidateEvaluationResult("MODEL_FIT_FAILED", None, error_message="solver failure")

    monkeypatch.setattr(scheduled.FairBiasAEAdapter, "_utility", failed)
    adapter = scheduled.ScheduledJointAdapter(monotonic=clock)
    adapter._search_deadline, adapter._utility_evaluations_ = 2.0, 0
    with pytest.raises(scheduled._EvaluationFailure, match="solver failure"):
        adapter._utility(None, {})


@pytest.mark.parametrize("budget,expected", [(3, "NO_FEASIBLE_BUDGET_LIMITED"),
                                             (4, "FEASIBLE_BUDGET_LIMITED"),
                                             (100, "SEARCH_EXHAUSTED")])
def test_real_bm_category_merge_has_atomic_feasible_incumbent(budget, expected):
    frame = pd.DataFrame({
        "cat": ["a"] * 12 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4
               + ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 12,
        "z": np.tile([0., 1.], 24)})
    X = pd.concat([frame, frame], ignore_index=True)
    y = np.tile([0, 1], 48)
    A = pd.DataFrame({"A": ([0] * 24 + [1] * 24) * 2})
    adapter = scheduled.ScheduledJointAdapter(
        epsilon_ratio=.75, max_outer_iterations=1, max_bm_steps=5,
        max_geometry_evaluations=budget, search_seconds=60, random_state=0)
    args = (X.iloc[:48], y[:48], A.iloc[:48], X.iloc[48:], y[48:], A.iloc[48:])
    if expected.startswith("NO_FEASIBLE"):
        with pytest.raises(RuntimeError, match=expected):
            adapter.fit_development(*args)
        assert not adapter.is_fitted_ and adapter._bm_commits == 0
    else:
        adapter.fit_development(*args)
        assert adapter.is_fitted_ and adapter._bm_commits == 1
        assert adapter.changed_dict_ == {"cat": {"d": "a"}}
        assert np.isfinite(adapter.predict_event_probability(X.iloc[48:])).all()
    assert adapter.termination_reason_ == expected
    assert adapter.convergence_verified_ == (expected == "SEARCH_EXHAUSTED")
