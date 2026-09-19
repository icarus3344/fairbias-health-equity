"""Synthetic worker evidence for registered extension configurations."""

import copy
import json
import dataclasses
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from nhis_fairbias.benchmark.data_contracts import (
    ARM_SPECS,
    generate_synthetic_nhis_cohort,
    load_arm_partitions,
)
from nhis_fairbias.benchmark.experiment_extensions import enumerate_extensions
from nhis_fairbias.benchmark.experiment_registry import identity
from nhis_fairbias.benchmark.experiment_worker import execute_job, file_sha, write_json
from nhis_fairbias.benchmark.preprocessing import BenchmarkPreprocessor
from nhis_fairbias.benchmark.adapters.adapter_fairbias import FairBiasAdapter


@pytest.fixture(scope="module")
def worker_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("extension_worker")
    cohort = generate_synthetic_nhis_cohort(n_records_per_year=120, n_strata=8, seed=5)
    partitions = load_arm_partitions(cohort, "arm_001")
    partitions.pop("evaluation_T")
    # The deterministic tiny design can place all six C labels in one class;
    # retain the synthetic features/design while making C estimable for worker
    # contract coverage.
    c = partitions["calibration_C"]
    partitions["calibration_C"] = dataclasses.replace(c, y=np.arange(len(c)) % 2)
    fitting = partitions["fitting_F"]
    prep = BenchmarkPreprocessor(ARM_SPECS["arm_001"]["features"]).fit(fitting.X_semantic)
    data = {
        "partitions": partitions,
        "preprocessor": prep,
        "data_identity": "synthetic_extension_worker_v1",
        "X_F": prep.transform(fitting.X_semantic),
        "X_C": prep.transform(partitions["calibration_C"].X_semantic),
        "X_S": prep.transform(partitions["selection_S"].X_semantic),
    }
    data_path = root / "data.joblib"
    joblib.dump(data, data_path)
    (root / "cache").mkdir()
    return root, data_path


def _record(method, backbone):
    for row in enumerate_extensions():
        if row["arm_id"] == "arm_001" and row["method"] == method and row["backbone"] == backbone:
            return copy.deepcopy(row)
    raise AssertionError((method, backbone))


def _run(root, data_path, config):
    config["candidate_id"] = identity(config)
    out = root / (config["method"] + "_" + config["backbone"])
    out.mkdir()
    job = {
        "config": config,
        "seed": 0,
        "data_path": str(data_path),
        "data_identity": "synthetic_extension_worker_v1",
        "data_sha256": file_sha(data_path),
        "cache_path": str(root / "cache"),
        "source_identity": "synthetic_extension_test",
    }
    write_json(out / "job.json", job)
    return execute_job(out / "job.json")


def test_registered_extension_records_and_candidate_ids(worker_fixture):
    root, data_path = worker_fixture
    registered = enumerate_extensions()
    assert any(r["method"] == "FAIRBIAS_BM_AE" for r in registered)
    assert any(r["method"] == "FAIRBIAS_JOINT" for r in registered)
    configs = []
    for method, backbone in (
        ("FAIRGBM_EO", "FAIRGBM_BASE"),
        ("FAIRBIAS_BM", "FAIRGBM_BASE"),
        ("UNMITIGATED", "TABM"),
    ):
        config = _record(method, backbone)
        params = config["params"]
        if method == "FAIRBIAS_BM":
            params["epsilon_ratio"] = 100.0
            params["max_geometry_evaluations"] = 100
        if method == "FAIRGBM_EO":
            params.update(n_estimators=2, num_leaves=4, n_jobs=1)
        if backbone == "TABM":
            params["estimator_params"].update(epochs=2, batch_size=64)
        configs.append(config)

    results = [_run(root, data_path, config) for config in configs]
    assert all(result["status"] == "VALID" for result in results), results
    assert all(result["reload_verified"] for result in results)
    assert all(result["candidate_id"] == config["candidate_id"] for result, config in zip(results, configs))
    assert all(result["output_type"] == "event_probability_p" for result in results)


def test_geometry_profiles_are_registered_with_distinct_profile_names(worker_fixture):
    root, data_path = worker_fixture
    rows = [r for r in enumerate_extensions() if r["family"] == "fixed_geometry_ablation" and r["arm_id"] == "arm_004"]
    profiles = {r["params"]["geometry_profile"] for r in rows}
    assert profiles == {"stress_elbow", "fixed2_same_epsilon", "fixed2_own_epsilon"}
    assert len({r["params"]["epsilon_ratio"] for r in rows}) == 1


def test_geometry_profiles_run_on_small_synthetic_fixture():
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 24), "z": np.tile([0.0, 1.0, 2.0], 8)})
    y = np.array([0, 1] * 12)
    A = np.array([1, 2] * 12)
    manifests = {}
    for profile in ("stress_elbow", "fixed2_same_epsilon", "fixed2_own_epsilon"):
        adapter = FairBiasAdapter(
            arm_id="arm_001", epsilon_ratio=1.0, max_iterations=1,
            max_geometry_evaluations=100, geometry_profile=profile,
        )
        adapter.fit(np.zeros((len(X), 2)), y, A, X_semantic=X)
        manifests[profile] = adapter.fit_manifest_
        assert manifests[profile]["geometry_profile"] == profile
    assert manifests["fixed2_same_epsilon"]["epsilon_threshold"] == manifests["stress_elbow"]["epsilon_threshold"]
