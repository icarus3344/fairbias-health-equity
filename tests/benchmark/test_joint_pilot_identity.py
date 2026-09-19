"""Source-origin admission must fail before a pilot consumes prepared data."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/pilot_scheduled_joint.py"
SPEC = importlib.util.spec_from_file_location("joint_pilot_identity_tested", SCRIPT)
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def module_fixture(tmp_path, monkeypatch):
    path = tmp_path / "src/fairbias/fixture.py"
    path.parent.mkdir(parents=True)
    path.write_text("CONSTANT = 1\n")
    module = types.ModuleType("fairbias.fixture")
    module.__file__ = str(path)
    modules = {k: v for k, v in sys.modules.items()
               if k.split(".")[0] not in {"fairbias", "nhis_fairbias"}}
    modules[module.__name__] = module
    monkeypatch.setattr(sys, "modules", modules)
    return path, module, {str(path.relative_to(tmp_path)): pilot.sha(path)}


def test_snapshot_binds_loaded_origin_and_detects_source_changed_after_snapshot(tmp_path, monkeypatch):
    path, module, expected = module_fixture(tmp_path, monkeypatch)
    result = pilot.runtime_modules(tmp_path, expected)
    assert result[module.__name__]["sha256"] == expected["src/fairbias/fixture.py"]
    path.write_text("CONSTANT = 2\n")
    with pytest.raises(ValueError, match="source mismatch"):
        pilot.runtime_modules(tmp_path, expected)


def test_shadow_module_is_rejected_even_when_project_disk_hash_is_valid(tmp_path, monkeypatch):
    path, module, expected = module_fixture(tmp_path, monkeypatch)
    shadow = tmp_path / "elsewhere.py"
    shadow.write_text(path.read_text())
    module.__file__ = str(shadow)
    assert pilot.sha(path) == expected["src/fairbias/fixture.py"]
    with pytest.raises(ValueError, match="outside source tree"):
        pilot.runtime_modules(tmp_path, expected)


def test_unknown_source_or_unverifiable_namespace_fails_closed(tmp_path, monkeypatch):
    _, module, expected = module_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="source mismatch"):
        pilot.runtime_modules(tmp_path, {})
    del module.__file__
    with pytest.raises(ValueError, match="no verifiable source"):
        pilot.runtime_modules(tmp_path, expected)
