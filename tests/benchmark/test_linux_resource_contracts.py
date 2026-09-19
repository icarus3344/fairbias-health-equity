"""Portable worker resource accounting contracts.

These tests exercise only the telemetry boundary.  They do not allocate large
objects, start workers, access NHIS files, or change adapter budgets.
"""

from types import SimpleNamespace

import pytest

from nhis_fairbias.benchmark import experiment_worker


def test_ru_maxrss_is_normalized_from_linux_kib_to_bytes():
    usage = SimpleNamespace(ru_maxrss=4096)

    assert experiment_worker._ru_maxrss_bytes(usage, platform="linux") == 4096 * 1024


def test_ru_maxrss_preserves_macos_bytes():
    usage = SimpleNamespace(ru_maxrss=4096)

    assert experiment_worker._ru_maxrss_bytes(usage, platform="darwin") == 4096


def test_negative_ru_maxrss_is_rejected():
    usage = SimpleNamespace(ru_maxrss=-1)

    with pytest.raises(ValueError, match="non-negative"):
        experiment_worker._ru_maxrss_bytes(usage, platform="linux")


def test_resource_telemetry_records_normalized_memory_cpu_and_thread_controls(monkeypatch):
    usage = SimpleNamespace(ru_maxrss=2048, ru_utime=1.25, ru_stime=0.75)
    monkeypatch.setattr(experiment_worker.resource, "getrusage", lambda _: usage)
    monkeypatch.setattr(experiment_worker.sys, "platform", "linux")
    for name in experiment_worker._THREAD_CONTROL_ENV_VARS:
        monkeypatch.setenv(name, "1")

    telemetry = experiment_worker._resource_telemetry()

    assert telemetry["max_rss_bytes"] == 2048 * 1024
    assert telemetry["ru_maxrss_native"] == 2048.0
    assert telemetry["ru_maxrss_unit"] == "KiB"
    assert telemetry["cpu_user_seconds"] == 1.25
    assert telemetry["cpu_system_seconds"] == 0.75
    assert telemetry["thread_controls"] == {
        name: "1" for name in experiment_worker._THREAD_CONTROL_ENV_VARS
    }
    assert telemetry["thread_controls_all_one"] is True


def test_resource_telemetry_reports_uncontrolled_thread_environment(monkeypatch):
    usage = SimpleNamespace(ru_maxrss=1, ru_utime=0.0, ru_stime=0.0)
    monkeypatch.setattr(experiment_worker.resource, "getrusage", lambda _: usage)
    for name in experiment_worker._THREAD_CONTROL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    telemetry = experiment_worker._resource_telemetry()

    assert telemetry["thread_controls_all_one"] is False
    assert all(value is None for value in telemetry["thread_controls"].values())
