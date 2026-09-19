"""Synthetic orchestration only: never load a prepared benchmark payload."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/pilot_joint_mds_retry.py"
SPEC = importlib.util.spec_from_file_location("joint_retry_pilot_tested", SCRIPT)
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "source"
    monkeypatch.setattr(wrapper, "ROOT", root)
    for name in wrapper.SOURCE_NAMES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# synthetic source: " + name + "\n")
    args = argparse.Namespace(run=tmp_path / "registered", output=tmp_path / "pilot",
        candidate="synthetic", variant="scheduled_joint_v1", seed=0,
        hard_seconds=420, search_seconds=360, ae_interval=3, geometry_prefix=0)
    return args, root


def install_worker(monkeypatch, action, calls):
    @contextmanager
    def retry():
        calls.append("retry_enter")
        stats = {"version": "mds_budget_retry_v1", "attempts": [], "fits": [], "recoveries": 0}
        try:
            yield stats
        finally:
            stats["attempts"].append({"logical_fit": 1, "attempt": 1, "status": "SYNTHETIC"})
            calls.append("retry_exit")
    monkeypatch.setattr(wrapper, "_retry_context", retry)
    monkeypatch.setattr(wrapper, "_load_sibling", lambda name: SimpleNamespace(run=action))


def test_outer_context_and_combined_receipt_preserve_base_result_bytes(setup, monkeypatch):
    args, root = setup
    calls = []
    original = b'{"status":"FIT_EXCEPTION","variant":"scheduled_joint_v1"}\n'
    def base(options):
        assert calls == ["retry_enter"]
        calls.append("base_run")
        options.output.mkdir()
        (options.output / "result.json").write_bytes(original)
    install_worker(monkeypatch, base, calls)
    wrapper.run_worker(args)
    assert calls == ["retry_enter", "base_run", "retry_exit"]
    assert (args.output / "result.json").read_bytes() == original
    receipt = json.loads((args.output / "mds_retry_receipt.json").read_text())
    assert receipt["effective_variant"] == "scheduled_joint_v1_with_mds_budget_retry_v1"
    assert receipt["base_result_status"] == "FIT_EXCEPTION"
    assert receipt["base_result_sha256"] == hashlib.sha256(original).hexdigest()
    assert receipt["base_result_rewritten"] is False and receipt["fit_success_established"] is False
    assert receipt["consumers_must_read_both_receipts"] is True
    assert receipt["retry_stats"]["attempts"] == [{"logical_fit": 1, "attempt": 1, "status": "SYNTHETIC"}]
    assert receipt["all_sources_unchanged_after"] is True
    for name in (wrapper.SOURCE_NAMES[0], wrapper.SOURCE_NAMES[3]):
        assert (args.output / Path(name).name).read_bytes() == (root / name).read_bytes()


@pytest.mark.parametrize("exception", [ValueError, KeyboardInterrupt])
def test_exception_preserves_attempts_and_does_not_fabricate_base_result(setup, monkeypatch, exception):
    args, _ = setup
    calls = []
    def base(options):
        options.output.mkdir()
        raise exception("synthetic-only failure")
    install_worker(monkeypatch, base, calls)
    with pytest.raises(exception):
        wrapper.run_worker(args)
    assert calls == ["retry_enter", "retry_exit"]
    receipt = json.loads((args.output / "mds_retry_receipt.json").read_text())
    assert receipt["wrapper_status"] == "WRAPPER_EXCEPTION"
    assert receipt["wrapper_error_type"] == exception.__name__
    assert receipt["base_result_sha256"] is None
    assert len(receipt["retry_stats"]["attempts"]) == 1
    assert not (args.output / "result.json").exists()


def test_preflight_failure_before_output_does_not_create_child_directory(setup, monkeypatch):
    args, _ = setup
    def fail(options):
        raise ValueError("synthetic preflight")
    install_worker(monkeypatch, fail, [])
    with pytest.raises(ValueError):
        wrapper.run_worker(args)
    assert not args.output.exists()


def test_source_change_marks_retry_receipt_without_rewriting_base_result(setup, monkeypatch):
    args, root = setup
    def base(options):
        options.output.mkdir()
        (options.output / "result.json").write_text('{"status":"synthetic"}')
        (root / wrapper.SOURCE_NAMES[3]).write_text("# changed synthetic source")
    install_worker(monkeypatch, base, [])
    wrapper.run_worker(args)
    receipt = json.loads((args.output / "mds_retry_receipt.json").read_text())
    assert receipt["wrapper_status"] == "INTEGRITY_FAILURE"
    assert receipt["all_sources_unchanged_after"] is False
    assert (args.output / "result.json").read_text() == '{"status":"synthetic"}'
    assert (args.output / "mds_budget_retry.py").read_text().startswith("# synthetic source:")


@pytest.mark.parametrize("kind", ["existing", "inside_registered"])
def test_worker_rejects_nonfresh_or_registered_outputs_before_context(setup, monkeypatch, kind):
    args, _ = setup
    if kind == "existing":
        args.output.mkdir()
        (args.output / "keep").write_text("original")
    else:
        args.output = args.run / "nested"
    monkeypatch.setattr(wrapper, "_retry_context", lambda: pytest.fail("must not enter context"))
    with pytest.raises(ValueError):
        wrapper.run_worker(args)
    if kind == "existing":
        assert (args.output / "keep").read_text() == "original"


def test_default_launcher_reuses_supervisor_and_forwards_internal_worker(setup, monkeypatch):
    args, _ = setup
    monkeypatch.setattr(wrapper.sys, "platform", "linux")
    captured = {}
    def supervise(command, **kwargs):
        captured.update(command=command, **kwargs)
        return {"synthetic": True}
    monkeypatch.setattr(wrapper, "_load_sibling", lambda name: SimpleNamespace(supervise_command=supervise))
    assert wrapper.run_supervised(args) == {"synthetic": True}
    assert not args.output.exists()
    assert "--worker" in captured["command"]
    for flag, value in (("--search-seconds", "360"), ("--hard-seconds", "420"), ("--variant", "scheduled_joint_v1")):
        assert captured["command"][captured["command"].index(flag) + 1] == value
    assert captured["elapsed_limit_seconds"] == 435
    assert captured["memory_limit_bytes"] == 4 * 1024 ** 3
    assert captured["sample_seconds"] == .5 and captured["grace_seconds"] == 5
    assert captured["receipt_path"].parent == args.output.parent
    assert captured["stdout_path"].parent == args.output.parent
    assert captured["receipt_metadata"]["effective_variant"] == wrapper.EFFECTIVE_VARIANT


def test_launcher_refuses_existing_sidecar(setup, monkeypatch):
    args, _ = setup
    monkeypatch.setattr(wrapper.sys, "platform", "linux")
    log = args.output.with_name(args.output.name + ".stdout.log")
    log.write_text("preserve")
    with pytest.raises(FileExistsError):
        wrapper.run_supervised(args)
    assert log.read_text() == "preserve" and not args.output.exists()


def test_production_launcher_is_linux_only(setup, monkeypatch):
    args, _ = setup
    monkeypatch.setattr(wrapper.sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="Linux-only"):
        wrapper.run_supervised(args)
    assert not args.output.exists()


def test_actual_retry_numpy_and_audit_context_nesting_records_both_attempts(tmp_path, monkeypatch):
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances
    from fairbias import bias_metric
    from nhis_fairbias.benchmark.adapters.geometry_audit import audited_mds
    from nhis_fairbias.benchmark.joint_mds_numpy import numpy_mds_acceleration

    args = argparse.Namespace(run=tmp_path / "registered", output=tmp_path / "pilot",
        candidate="synthetic", variant="scheduled_joint_v1", seed=0,
        hard_seconds=420, search_seconds=360, ae_interval=3, geometry_prefix=0)
    original_mds = bias_metric.MDS
    records = []
    result_bytes = b'{"status":"FIT_EXCEPTION","variant":"scheduled_joint_v1"}\n'
    def base(options):
        options.output.mkdir()
        distance = euclidean_distances(np.random.default_rng(41).normal(size=(6, 3)))
        with warnings.catch_warnings(), numpy_mds_acceleration(), audited_mds(records, lambda: 1):
            warnings.simplefilter("ignore", FutureWarning)
            model = bias_metric.MDS(n_components=2, random_state=0,
                dissimilarity="precomputed", n_init=1, max_iter=1, eps=1e-10)
            with pytest.raises(RuntimeError, match="MDS_ITERATION_CAP"):
                model.fit_transform(distance)
        (options.output / "result.json").write_bytes(result_bytes)
    monkeypatch.setattr(wrapper, "_load_sibling", lambda name: SimpleNamespace(run=base))
    wrapper.run_worker(args)
    assert bias_metric.MDS is original_mds
    receipt = json.loads((args.output / "mds_retry_receipt.json").read_text())
    assert receipt["retry_stats"]["version"] == "mds_budget_retry_v1"
    assert [row["max_iter"] for row in receipt["retry_stats"]["attempts"]] == [1, 2]
    assert receipt["retry_stats"]["fits"][0]["complete"] is False
    assert records[0]["max_iter"] == 2 and records[0]["status"] == "MDS_ITERATION_CAP"
    assert (args.output / "result.json").read_bytes() == result_bytes
