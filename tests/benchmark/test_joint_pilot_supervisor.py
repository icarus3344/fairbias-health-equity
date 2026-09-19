"""Actual short synthetic subprocess probes; no benchmark input is loaded."""
import argparse
import importlib.util
import json
from pathlib import Path
import signal
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/supervise_joint_pilot.py"
SPEC = importlib.util.spec_from_file_location("joint_pilot_supervisor_tested", SCRIPT)
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)


def launch(tmp_path, code, **kwargs):
    values = dict(
        receipt_path=tmp_path / "receipt.json", stdout_path=tmp_path / "stdout.log",
        elapsed_limit_seconds=5, sample_seconds=0.02, grace_seconds=0.15,
        rss_reader=lambda pid: 1024,
    )
    values.update(kwargs)
    return supervisor.supervise_command([sys.executable, "-B", "-c", code], **values)


def test_actual_success_is_process_exit_and_not_fit_success(tmp_path):
    result = launch(tmp_path, "import os,json; print(json.dumps({'threads':os.environ['OMP_NUM_THREADS'],'nice':os.getpriority(os.PRIO_PROCESS,0)}))")
    assert result["status"] == "CHILD_PROCESS_EXITED"
    assert result["exit_code"] == 0 and result["fit_success_established"] is False
    assert result["observed_nice"] == 19
    assert result["peak_observed_rss_bytes"] == 1024
    assert json.loads((tmp_path / "stdout.log").read_text()) == {"threads": "1", "nice": 19}
    assert json.loads((tmp_path / "receipt.json").read_text()) == result


def test_actual_nonzero_exit_does_not_become_success(tmp_path):
    result = launch(tmp_path, "raise SystemExit(7)")
    assert result["status"] == "CHILD_PROCESS_EXITED"
    assert result["exit_code"] == 7 and result["fit_success_established"] is False


def test_actual_timeout_is_external_and_receipt_survives_missing_child_result(tmp_path):
    result = launch(tmp_path, "import time; time.sleep(30)", elapsed_limit_seconds=0.15)
    assert result["termination_reason"] == "EXTERNAL_WALL_LIMIT"
    assert result["status"] == "CHILD_TERMINATED_BY_SUPERVISOR"
    assert result["sigterm_sent"] is True
    assert result["exit_code"] == -signal.SIGTERM
    assert result["elapsed_seconds"] < 2
    assert (tmp_path / "receipt.json").exists()
    assert not (tmp_path / "result.json").exists()


def test_actual_sigterm_resistance_escalates_to_sigkill(tmp_path):
    result = launch(
        tmp_path, "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(30)",
        elapsed_limit_seconds=0.4,
    )
    assert "ready" in (tmp_path / "stdout.log").read_text()
    assert result["sigterm_sent"] and result["sigkill_sent"]
    assert result["exit_code"] == -signal.SIGKILL


def test_sampled_rss_threshold_terminates_actual_child(tmp_path):
    result = launch(tmp_path, "import time; time.sleep(30)", memory_limit_bytes=4096, rss_reader=lambda pid: 8192)
    assert result["termination_reason"] == "EXTERNAL_RSS_LIMIT"
    assert result["peak_observed_rss_bytes"] == 8192
    assert result["rss_limit_kind"] == "sampled_threshold_not_kernel_cap"
    assert result["rss_scope"] == "main_child_only"
    assert result["exit_code"] == -signal.SIGTERM


def test_monitor_failure_is_recorded_and_child_stopped(tmp_path):
    def denied(pid):
        raise PermissionError("synthetic")

    result = launch(tmp_path, "import time; time.sleep(30)", rss_reader=denied)
    assert result["termination_reason"] == "RSS_MONITOR_ERROR"
    assert result["monitor_error_type"] == "PermissionError"
    assert result["sigterm_sent"]


def test_existing_log_or_receipt_is_never_overwritten(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("original")
    with pytest.raises(FileExistsError):
        launch(tmp_path, "raise AssertionError('must not launch')")
    assert log.read_text() == "original"
    assert not (tmp_path / "receipt.json").exists()


def test_linux_proc_parser_and_exited_process(monkeypatch):
    import io

    monkeypatch.setattr("builtins.open", lambda *a, **k: io.StringIO("Name:\tpython\nVmRSS:\t1234 kB\n"))
    assert supervisor.read_proc_rss_bytes(9) == 1234 * 1024
    def missing(*args, **kwargs):
        raise FileNotFoundError()
    monkeypatch.setattr("builtins.open", missing)
    assert supervisor.read_proc_rss_bytes(9) is None


def test_linux_launcher_forwards_command_and_does_not_create_child_output(tmp_path, monkeypatch):
    args = argparse.Namespace(run=tmp_path / "registered", output=tmp_path / "pilot",
        candidate="synthetic-candidate", variant="scheduled_joint_v1", seed=7,
        hard_seconds=60, search_seconds=20, ae_interval=3, geometry_prefix=16)
    monkeypatch.setattr(supervisor.sys, "platform", "linux")
    received = {}
    def capture(command, **kwargs):
        received.update(command=command, **kwargs)
        return {"captured": True}
    monkeypatch.setattr(supervisor, "supervise_command", capture)
    assert supervisor.run_pilot(args) == {"captured": True}
    assert not args.output.exists()
    command = received["command"]
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--geometry-prefix") + 1] == "16"
    assert command[command.index("--variant") + 1] == "scheduled_joint_v1"
    assert received["elapsed_limit_seconds"] == 75
    assert received["memory_limit_bytes"] == 4 * 1024 ** 3
    assert received["sample_seconds"] == 0.5 and received["grace_seconds"] == 5.0
    assert received["receipt_path"].parent == args.output.parent
    assert received["stdout_path"].parent == args.output.parent
    assert "scripts/pilot_scheduled_joint.py" in received["receipt_metadata"]["runtime_source_hashes"]
    assert received["environment"]["PYTHONPATH"].endswith("/src")


def test_production_launcher_is_linux_only_before_any_output(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor.sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="Linux-only"):
        supervisor.run_pilot(argparse.Namespace())
    assert list(tmp_path.iterdir()) == []
