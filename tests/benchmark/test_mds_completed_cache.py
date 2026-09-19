"""Synthetic-only exact replay and integrity tests for completed MDS fits."""
import copy
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from joblib import parallel_backend
from sklearn.manifold import MDS, _mds
from sklearn.metrics import pairwise
from threadpoolctl import threadpool_limits

from fairbias import bias_metric
from nhis_fairbias.benchmark.adapters.geometry_audit import audited_mds
from nhis_fairbias.benchmark.joint_mds_numpy import numpy_mds_acceleration
from nhis_fairbias.benchmark.mds_completed_cache import (
    MDSCacheIntegrityError, completed_mds_cache,
)


@pytest.fixture(autouse=True)
def serial_numerics():
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        yield


def distance(seed=5, n=9):
    result = pairwise.euclidean_distances(np.random.default_rng(seed).normal(size=(n, 4)))
    # Match FairBias's explicitly mirrored Shapley matrix construction.
    return (result + result.T) / 2


def fit(X, *, via_fit=False, **changes):
    params = dict(n_components=2, random_state=7, dissimilarity="precomputed",
                  n_init=1, max_iter=10000, eps=1e-8)
    params.update(changes)
    model = bias_metric.MDS(**params)
    if via_fit:
        assert model.fit(X) is model
    else:
        assert model.fit_transform(X) is model.embedding_
    return model


def assert_same_fit(expected, actual):
    assert set(expected.__dict__) == set(actual.__dict__)
    for name, value in expected.__dict__.items():
        other = actual.__dict__[name]
        assert type(value) is type(other), name
        if isinstance(value, np.ndarray):
            assert value.shape == other.shape and value.dtype == other.dtype
            assert value.tobytes() == other.tobytes(), name
        else:
            assert value == other, name


@pytest.mark.parametrize("seed,components,n_init,normalized,via_fit", [
    (0, 1, 1, "auto", False), (7, 2, 4, "auto", True),
    (21, 4, 1, True, True), (3, 7, 4, True, False),
])
def test_cold_ram_and_disk_hits_preserve_all_fitted_attributes_and_audit(
        tmp_path, seed, components, n_init, normalized, via_fit):
    X = distance()
    params = dict(random_state=seed, n_components=components, n_init=n_init,
                  normalized_stress=normalized, via_fit=via_fit)
    before, cold_audit, ram_audit, disk_audit = [], [], [], []
    with numpy_mds_acceleration(), audited_mds(before, lambda: 8):
        expected = fit(X, **params)
    with completed_mds_cache(tmp_path, "synthetic-family") as stats, numpy_mds_acceleration():
        with audited_mds(cold_audit, lambda: 8):
            cold = fit(X, **params)
        with audited_mds(ram_audit, lambda: 8):
            warm = fit(X.copy(), **params)
    with completed_mds_cache(tmp_path, "synthetic-family") as replay, numpy_mds_acceleration():
        with audited_mds(disk_audit, lambda: 8):
            disk = fit(X.copy(), **params)
    for model in (cold, warm, disk):
        assert_same_fit(expected, model)
    assert before == cold_audit == ram_audit == disk_audit
    assert stats["actual_fits"] == stats["writes"] == stats["ram_hits"] == 1
    assert replay["actual_fits"] == 0 and replay["disk_hits"] == 1
    assert replay["fits"][0]["origin"] == stats["fits"][0]["origin"]


def test_hits_do_not_share_mutable_arrays_or_change_external_rng(tmp_path):
    X = distance()
    np.random.seed(132)
    before_rng = pickle.dumps(np.random.get_state())
    with completed_mds_cache(tmp_path, "copies") as stats:
        expected = fit(X.copy())
        wanted = expected.embedding_.copy()
        expected.embedding_[:] = 123
        expected.dissimilarity_matrix_[:] = 999
        first_hit = fit(X.copy())
        assert first_hit.embedding_.tobytes() == wanted.tobytes()
        first_hit.embedding_[:] = -200
        second_hit = fit(X.copy())
        assert second_hit.embedding_.tobytes() == wanted.tobytes()
        assert second_hit.dissimilarity_matrix_.tobytes() == X.tobytes()
    assert stats["hits"] == 2
    assert pickle.dumps(np.random.get_state()) == before_rng


@pytest.mark.parametrize("rng_kind", ["mutable", "none", "boolean"])
def test_unsupported_rng_bypasses_and_preserves_state_progression(tmp_path, rng_kind):
    X = distance()
    if rng_kind == "mutable":
        before_rng, after_rng = np.random.RandomState(4), np.random.RandomState(4)
    else:
        before_rng = after_rng = None if rng_kind == "none" else True
    np.random.seed(441)
    expected = fit(X, random_state=before_rng)
    expected_global = pickle.dumps(np.random.get_state())
    np.random.seed(441)
    with completed_mds_cache(tmp_path, "rng") as stats:
        actual = fit(X, random_state=after_rng)
    assert actual.embedding_.tobytes() == expected.embedding_.tobytes()
    assert actual.stress_ == expected.stress_ and actual.n_iter_ == expected.n_iter_
    assert pickle.dumps(np.random.get_state()) == expected_global
    if rng_kind == "mutable":
        assert pickle.dumps(before_rng.get_state()) == pickle.dumps(after_rng.get_state())
    assert stats["bypasses"] == 1 and stats["writes"] == 0


def test_namespace_seed_input_and_all_relevant_parameters_change_identity(tmp_path):
    X = distance()
    with completed_mds_cache(tmp_path, "identity") as stats:
        fit(X)
        fit(X, random_state=8)
        fit(X, eps=1e-7)
        fit(X, max_iter=10001)
        fit(X, n_init=4)
        fit(X, n_components=3)
        changed = X.copy()
        changed[0, 1] = changed[1, 0] = changed[0, 1] + 1e-7
        fit(changed)
        fit(X)
    assert stats["misses"] == 7 and stats["hits"] == 1
    assert len({r["cache_key"] for r in stats["fits"]}) == 7
    with completed_mds_cache(tmp_path, "separate-family") as other:
        fit(X)
    assert other["misses"] == 1 and other["hits"] == 0


@pytest.mark.parametrize("kind", ["float32", "fortran", "subclass", "explicit_init", "nonprecomputed"])
def test_unsupported_inputs_use_original_fit(tmp_path, kind):
    X, kwargs, fit_init = distance(), {}, None
    if kind == "float32":
        X = X.astype(np.float32)
    elif kind == "fortran":
        X = np.asfortranarray(X)
    elif kind == "subclass":
        class Array(np.ndarray):
            pass
        X = X.view(Array)
    elif kind == "explicit_init":
        fit_init = np.random.RandomState(0).uniform(size=(len(X), 2))
    else:
        kwargs["dissimilarity"] = "deprecated"
    params = dict(random_state=3, n_init=1, max_iter=1000, dissimilarity="precomputed")
    params.update(kwargs)
    # The non-precomputed branch intentionally uses a square feature matrix.
    def one(cls):
        model = cls(**params)
        model.fit_transform(X, init=fit_init)
        return model
    expected = one(MDS)
    with completed_mds_cache(tmp_path, "bypass") as stats:
        actual = one(bias_metric.MDS)
    assert actual.embedding_.tobytes() == expected.embedding_.tobytes()
    assert stats["bypasses"] == 1 and stats["hits"] == stats["writes"] == 0


@pytest.mark.parametrize("kind", ["nan", "asymmetric"])
def test_invalid_inputs_keep_original_exception(tmp_path, kind):
    X = distance()
    X[0, 1] = np.nan if kind == "nan" else 900
    with pytest.raises(ValueError) as expected:
        fit(X)
    with completed_mds_cache(tmp_path, "invalid") as stats:
        with pytest.raises(ValueError) as actual:
            fit(X)
    assert str(actual.value).replace("CachedMDS", "MDS") == str(expected.value)
    assert stats["writes"] == 0


def test_multithread_blas_and_nonserial_joblib_bypass(tmp_path):
    X = distance()
    with completed_mds_cache(tmp_path, "threads") as stats:
        with threadpool_limits(limits=2):
            fit(X)
        with parallel_backend("threading", n_jobs=2):
            fit(X)
    assert stats["bypasses"] == 2 and stats["writes"] == 0


def test_warning_as_error_is_not_hidden_by_cache(tmp_path):
    X = distance()
    with completed_mds_cache(tmp_path, "warnings") as stats:
        fit(X)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            with pytest.raises(FutureWarning):
                fit(X)
    assert stats["writes"] == 1 and stats["bypasses"] == 1


def test_capped_selected_initialization_is_audited_and_never_cached(tmp_path):
    X = distance()
    records = []
    with completed_mds_cache(tmp_path, "cap") as stats, audited_mds(records, lambda: 2):
        for _ in range(2):
            with pytest.raises(RuntimeError, match="MDS_ITERATION_CAP"):
                fit(X, max_iter=1)
    assert stats["actual_fits"] == stats["uncacheable_results"] == 2
    assert stats["writes"] == stats["hits"] == 0
    assert len(records) == 2 and all(r["status"] == "MDS_ITERATION_CAP" for r in records)


@pytest.mark.parametrize("corruption", ["payload", "manifest", "missing", "shape"])
def test_corrupt_completed_entry_fails_closed(tmp_path, corruption):
    X = distance()
    with completed_mds_cache(tmp_path, "corrupt"):
        fit(X)
    manifest_path = next(tmp_path.rglob("manifest.json"))
    payload = manifest_path.parent / "payload.npz"
    if corruption == "payload":
        with payload.open("ab") as handle:
            handle.write(b"damage")
    elif corruption == "missing":
        manifest_path.unlink()
    else:
        manifest = json.loads(manifest_path.read_text())
        if corruption == "manifest":
            manifest["state"]["stress_"] += 0.1
        else:
            manifest["request"]["shape"] = [1, 1]
        manifest_path.write_text(json.dumps(manifest))
    with completed_mds_cache(tmp_path, "corrupt") as replay:
        with pytest.raises(MDSCacheIntegrityError):
            fit(X)
    assert replay["actual_fits"] == replay["hits"] == 0


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_error_during_fit_is_not_cached_and_context_restores(tmp_path, monkeypatch, error):
    X = distance()
    def fail(*args, **kwargs):
        raise error("synthetic failure")
    monkeypatch.setattr(MDS, "fit_transform", fail)
    original = bias_metric.MDS
    with pytest.raises(error):
        with completed_mds_cache(tmp_path, "failed") as stats:
            fit(X)
    assert bias_metric.MDS is original
    assert stats["actual_fits"] == 1 and stats["writes"] == 0
    assert not list(tmp_path.rglob("manifest.json"))


def test_completed_prefix_survives_interrupt_and_replays_in_new_context(tmp_path):
    X = distance()
    original = bias_metric.MDS
    with pytest.raises(KeyboardInterrupt):
        with completed_mds_cache(tmp_path, "prefix") as first:
            expected = fit(X)
            raise KeyboardInterrupt("after completed MDS")
    assert bias_metric.MDS is original and first["writes"] == 1
    with completed_mds_cache(tmp_path, "prefix") as replay:
        actual = fit(X)
        fit(distance(seed=8))
    assert_same_fit(expected, actual)
    assert replay["disk_hits"] == 1 and replay["actual_fits"] == 1


def test_nesting_old_retry_and_unknown_class_hooks_rejected(tmp_path):
    from nhis_fairbias.benchmark.mds_budget_retry import mds_budget_retry
    original = bias_metric.MDS
    with completed_mds_cache(tmp_path, "outer"):
        installed = bias_metric.MDS
        with pytest.raises(RuntimeError, match="not reentrant"):
            with completed_mds_cache(tmp_path, "inner"):
                pass
        with pytest.raises(RuntimeError, match="already replaced"):
            with mds_budget_retry():
                pass
        assert bias_metric.MDS is installed
    with mds_budget_retry():
        with pytest.raises(RuntimeError, match="original sklearn"):
            with completed_mds_cache(tmp_path, "invalid"):
                pass
    assert bias_metric.MDS is original


def test_actual_kernel_change_does_not_reuse_another_kernel_receipt(tmp_path):
    X = distance()
    with completed_mds_cache(tmp_path, "kernel") as stats:
        baseline = fit(X)
        with numpy_mds_acceleration():
            actual = fit(X)
            fit(X)
    assert_same_fit(baseline, actual)
    assert stats["misses"] == 2 and stats["hits"] == 1


def test_loaded_callable_change_fails_closed(tmp_path, monkeypatch):
    def replacement(*args, **kwargs):
        raise AssertionError("Changed numerical dependency must not execute")
    with completed_mds_cache(tmp_path, "identity"):
        monkeypatch.setattr(_mds, "_smacof_single", replacement)
        with pytest.raises(MDSCacheIntegrityError, match="source callable changed"):
            fit(distance())


def test_environment_identity_is_checked_again_on_every_request(tmp_path, monkeypatch):
    from nhis_fairbias.benchmark import mds_completed_cache as module
    original_machine = module.platform.machine
    X = distance()
    with completed_mds_cache(tmp_path, "architecture") as stats:
        fit(X)
        with monkeypatch.context() as patch:
            patch.setattr(module.platform, "machine", lambda: "different-architecture")
            fit(X)
        assert module.platform.machine is original_machine
        fit(X)
    assert len(stats["environments"]) == 2
    assert stats["misses"] == 2 and stats["ram_hits"] == 1


def test_library_enumeration_order_is_not_numerical_identity(tmp_path, monkeypatch):
    from nhis_fairbias.benchmark import mds_completed_cache as module
    paths = [tmp_path / 'numpy-blas.so', tmp_path / 'scipy-blas.so']
    for i, path in enumerate(paths):
        path.write_bytes(('distinct-library-' + str(i)).encode())
    pools = [{'user_api': 'blas', 'num_threads': 1, 'filepath': str(p),
              'architecture': 'same-cpu', 'version': str(i)} for i, p in enumerate(paths)]
    monkeypatch.setattr(module, 'threadpool_info', lambda: pools)
    first = module._environment_identity({})
    pools.reverse()
    assert module._environment_identity({}) == first
    pools[0]['architecture'] = 'different-cpu-dispatch'
    assert module._environment_identity({}) != first
    pools[0]['num_threads'] = 2
    assert module._environment_identity({}) is None


def test_ram_capacity_is_bounded_and_evicted_entries_still_replay(tmp_path, monkeypatch):
    from nhis_fairbias.benchmark import mds_completed_cache as module
    monkeypatch.setattr(module, "_MAX_RAM_ENTRIES", 2)
    X = distance()
    with completed_mds_cache(tmp_path, "bounded") as stats:
        for seed in (1, 2, 3, 1):
            fit(X, random_state=seed)
    assert stats["actual_fits"] == 3 and stats["disk_hits"] == 1
    assert stats["ram_entries"] == stats["max_ram_entries"] == 2
    assert stats["ram_evictions"] == 2


def test_orphan_atomic_write_is_ignored_and_unknown_version_rejected(tmp_path, monkeypatch):
    import hashlib
    import sklearn
    namespace = "interrupted-write"
    incomplete = tmp_path / hashlib.sha256(namespace.encode()).hexdigest() / ".partial-abandoned"
    incomplete.mkdir(parents=True)
    (incomplete / "payload.npz").write_bytes(b"interrupted")
    with completed_mds_cache(tmp_path, namespace) as stats:
        fit(distance())
    assert stats["writes"] == 1 and incomplete.exists()
    monkeypatch.setattr(sklearn, "__version__", "unreviewed")
    with pytest.raises(RuntimeError, match="original sklearn 1.9.1"):
        with completed_mds_cache(tmp_path, namespace):
            pass


def test_disk_hit_in_another_python_process(tmp_path):
    code = r'''
import hashlib,json,sys,warnings,numpy as np
from fairbias import bias_metric
from sklearn.metrics import pairwise
from threadpoolctl import threadpool_limits
from nhis_fairbias.benchmark.mds_completed_cache import completed_mds_cache
X=pairwise.euclidean_distances(np.random.default_rng(5).normal(size=(9,4)))
X=(X+X.T)/2
with warnings.catch_warnings(),threadpool_limits(1):
 warnings.simplefilter('ignore',FutureWarning)
 with completed_mds_cache(sys.argv[1],'subprocess') as stats:
  m=bias_metric.MDS(random_state=7,dissimilarity='precomputed',n_init=4,max_iter=10000,eps=1e-8).fit(X)
print(json.dumps({'bytes':hashlib.sha256(m.embedding_.tobytes()).hexdigest(),'stress':float(m.stress_),'iterations':int(m.n_iter_),'actual':stats['actual_fits'],'disk':stats['disk_hits']}))
'''
    def run():
        result = subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path)],
                                capture_output=True, text=True, check=True, timeout=30)
        return json.loads(result.stdout.strip().splitlines()[-1])
    cold, warm = run(), run()
    assert {k: cold[k] for k in ("bytes", "stress", "iterations")} == {
        k: warm[k] for k in ("bytes", "stress", "iterations")}
    assert cold["actual"] == 1 and warm["actual"] == 0 and warm["disk"] == 1


@pytest.mark.parametrize("mode", ["BM_AE", "JOINT"])
@pytest.mark.parametrize("backbone", ["LR", "GBDT"])
def test_actual_adapter_candidates_geometry_audits_and_predictions_exact(tmp_path, mode, backbone):
    from nhis_fairbias.benchmark.adapters.adapter_fairbias_ae import FairBiasAEAdapter
    training = pd.DataFrame({
        "cat": ["a"] * 12 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4
               + ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 12,
        "z": np.tile([0.0, 1.0], 24),
    })
    X = pd.concat([training, training], ignore_index=True)
    y = np.tile([0, 1], 48)
    A = pd.DataFrame({"A": ([0] * 24 + [1] * 24) * 2})
    def one():
        adapter = FairBiasAEAdapter(mode=mode, backbone=backbone, random_state=0,
                                   epsilon_ratio=0.75, max_outer_iterations=2,
                                   max_bm_steps=5, max_geometry_evaluations=1000)
        adapter.fit_development(X.iloc[:48], y[:48], A.iloc[:48],
                                X.iloc[48:], y[48:], A.iloc[48:])
        return adapter, adapter.predict_event_probability(X).tobytes()
    with numpy_mds_acceleration():
        expected, wanted = one()
    with completed_mds_cache(tmp_path, mode + backbone) as cold, numpy_mds_acceleration():
        actual, actual_predictions = one()
    with completed_mds_cache(tmp_path, mode + backbone) as warm, numpy_mds_acceleration():
        replay, replay_predictions = one()
    assert expected.changed_dict_ == {"cat": {"d": "a"}}
    for adapter in (actual, replay):
        assert adapter.provenance_ == expected.provenance_
        assert adapter.candidate_traces_ == expected.candidate_traces_
        assert adapter._geometry_evaluations_ == expected._geometry_evaluations_
        assert adapter._utility_evaluations_ == expected._utility_evaluations_
    assert wanted == actual_predictions == replay_predictions
    assert cold["actual_fits"] > 0 and warm["actual_fits"] == 0 and warm["disk_hits"] > 0


@pytest.mark.parametrize("mode", ["BM_AE", "JOINT"])
def test_real_positive_gain_ae_commit_and_exhaustion_are_preserved(tmp_path, mode):
    from nhis_fairbias.benchmark.adapters.adapter_fairbias_ae import FairBiasAEAdapter
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"agep_a": rng.uniform(-1, 1, 400),
                      "pcnt18uptc": rng.uniform(-1, 1, 400)})
    y = (X.agep_a ** 3 + 0.8 * X.pcnt18uptc > 0).to_numpy(dtype=int)
    A = pd.DataFrame({"A": np.tile([0, 1], 200)})
    def one():
        adapter = FairBiasAEAdapter(mode=mode, backbone="LR", random_state=0,
                                   epsilon_ratio=100, max_outer_iterations=10,
                                   max_geometry_evaluations=1000)
        adapter.fit_development(X.iloc[:240], y[:240], A.iloc[:240],
                                X.iloc[240:], y[240:], A.iloc[240:])
        return adapter, adapter.predict_event_probability(X).tobytes()
    with numpy_mds_acceleration():
        expected, wanted = one()
    with completed_mds_cache(tmp_path, "positive-ae-" + mode) as cold, numpy_mds_acceleration():
        actual, predicted = one()
    with completed_mds_cache(tmp_path, "positive-ae-" + mode) as warm, numpy_mds_acceleration():
        replay, replayed_predictions = one()
    assert expected.provenance_["ae_commits"] >= 1
    assert expected.changed_dict_ == {"pcnt18uptc": {"power": 1 / 3}}
    assert expected.termination_reason_ == "STRICT_FEASIBLE_SEARCH_EXHAUSTED"
    assert expected.provenance_ == actual.provenance_ == replay.provenance_
    assert expected.candidate_traces_ == actual.candidate_traces_ == replay.candidate_traces_
    assert wanted == predicted == replayed_predictions
    assert warm["actual_fits"] == 0 and warm["disk_hits"] == cold["writes"]


def test_cache_does_not_bypass_original_logical_geometry_budget(tmp_path):
    from nhis_fairbias.benchmark.adapters.adapter_fairbias_ae import FairBiasAEAdapter
    X = pd.DataFrame({"agep_a": np.linspace(20.0, 80.0, 48),
                      "pcnt18uptc": np.linspace(1.0, 12.0, 48),
                      "status": ["a"] * 24 + ["b"] * 24,
                      "region": np.tile(["x", "y", "z"], 16)})
    y, A = np.tile([0, 1], 24), pd.DataFrame({"A": np.tile([1, 2], 24)})
    def one():
        adapter = FairBiasAEAdapter(mode="JOINT", random_state=0, epsilon_ratio=0.25,
                                   max_geometry_evaluations=1)
        with pytest.raises(RuntimeError, match="BUDGET_EXHAUSTED") as exc:
            adapter.fit_development(X.iloc[:24], y[:24], A.iloc[:24],
                                    X.iloc[24:], y[24:], A.iloc[24:])
        assert not adapter.is_fitted_
        return (str(exc.value), adapter.termination_reason_, adapter._geometry_evaluations_,
                adapter._utility_evaluations_, adapter.mds_diagnostics_)
    with numpy_mds_acceleration():
        expected = one()
    with completed_mds_cache(tmp_path, "geometry-limit") as cold, numpy_mds_acceleration():
        actual = one()
    with completed_mds_cache(tmp_path, "geometry-limit") as warm, numpy_mds_acceleration():
        replay = one()
    assert expected == actual == replay
    assert cold["writes"] > 0 and warm["actual_fits"] == 0 and warm["disk_hits"] > 0
