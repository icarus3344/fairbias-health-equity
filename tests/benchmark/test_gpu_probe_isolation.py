"""Isolation and no-CUDA contracts for the synthetic GPU probe."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("gpu_probe_isolated", ROOT / "scripts/probe_nhis_gpu_isolated.py")
probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(probe)


def test_output_directory_rejects_existing_and_benchmark_run_paths(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="new directory"):
        probe._validate_output_dir(existing)
    benchmark = tmp_path / "benchmark_autodl_20260916_abc"
    benchmark.mkdir()
    with pytest.raises(ValueError, match="benchmark_autodl"):
        probe._validate_output_dir(benchmark / "diagnostic")


def test_no_cuda_is_explicit_and_writes_only_isolated_diagnostics(tmp_path, monkeypatch):
    output = tmp_path / "gpu_probe_diagnostic"
    probe._validate_output_dir(output)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    status = probe.run_probe(output, rows=16, epochs=1, seed=7)
    assert status == 0
    result = json.loads((output / "result.json").read_text())
    manifest = json.loads((output / "probe_manifest.json").read_text())
    assert result["status"] == "NO_CUDA"
    assert manifest["status"] == "NO_CUDA"
    assert manifest["diagnostic_only"] is True
    assert manifest["real_data_loaded"] is False
    assert not list(output.glob("model_*"))
    assert not list(output.glob("synthetic_input*"))
    assert all(not part.startswith("benchmark_autodl_") for part in output.resolve().parts)


def test_source_manifest_records_cpu_reference_and_pinned_tabm_hashes():
    manifest = probe._source_manifest()
    assert {"probe_script", "tabm_adapter", "mlp_adapter", "official_tabm_source"} <= set(manifest)
    assert all(len(item["sha256"]) == 64 for item in manifest.values())
