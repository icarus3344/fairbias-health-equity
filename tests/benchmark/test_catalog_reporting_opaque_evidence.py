"""Regression coverage for opaque supplemental evidence in paper exports."""

import json

import pytest

from nhis_fairbias.benchmark import catalog_reporting as reporting
from test_catalog_reporting import _bound, _export, _reseal, case, selection_case, sha, write


def _supplemental(case, evidence_files):
    case["study"]["supplemental_evidence"] = [{
        "study_branch": "bmae_cap40_sensitivity",
        "status": "AWAITING_SEPARATE_FROZEN_EVALUATION",
        "evidence_files": evidence_files,
    }]
    _reseal(case)


def test_opaque_supplemental_markdown_and_bytes_are_hash_bound_but_not_decoded(case, monkeypatch):
    markdown = case["root"] / "sensitivity_report.md"
    markdown.write_text("# separate sensitivity evidence\n", encoding="utf-8")
    opaque = case["root"] / "sensitivity_receipt.bin"
    opaque.write_bytes(bytes([0, 255]) + b"opaque bytes" + bytes([0]))
    _supplemental(case, [_bound(markdown), _bound(opaque)])

    import joblib
    import numpy as np
    monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("No models may be deserialized"))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("No prediction arrays may be loaded"))

    outputs = _export(case)
    manifest = json.loads(outputs["manifest"].read_text())
    assert manifest["supplemental_evidence"] == case["study"]["supplemental_evidence"]
    assert {
        "registered_jobs": 12,
        "catalog_attempts": 13,
        "resolved_failed_jobs": 1,
        "not_supported_configurations": 1,
    }.items() <= manifest["denominators"].items()


@pytest.mark.parametrize("fault", ["wrong_hash", "missing_file"])
def test_opaque_supplemental_wrong_hash_or_missing_file_rejected(case, fault):
    evidence = case["root"] / "opaque_supplemental.bin"
    evidence.write_bytes(b"opaque supplemental evidence")
    bound = _bound(evidence)
    if fault == "wrong_hash":
        bound["sha256"] = "0" * 64
    else:
        evidence.unlink()
    _supplemental(case, [bound])

    expected_code = "FILE_HASH_MISMATCH" if fault == "wrong_hash" else "MISSING_FILE"
    with pytest.raises(reporting.ReportingError, match=expected_code):
        _export(case)
    assert not (case["root"] / "report").exists()


def test_required_release_binding_rejects_hash_matching_markdown(case):
    release = case["release_path"]
    release.write_text("# this is not a release JSON document\n", encoding="utf-8")
    case["evaluation"]["release"] = _bound(release)
    write(case["evaluation_path"], case["evaluation"])
    case["summary"]["evaluation_manifest_sha256"] = sha(case["evaluation_path"])
    write(case["summary_path"], case["summary"])

    with pytest.raises(reporting.ReportingError, match="JSON_OBJECT_REQUIRED"):
        _export(case)
    assert not (case["root"] / "report").exists()
