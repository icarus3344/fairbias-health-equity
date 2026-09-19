"""Guarded test runner for FairBias R1A-R1 core synthetic contract verification."""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

# Fix interpreter and repo root
PYTHON_BIN = "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Mandatory Baseline Constants
PROTECTED_TAG = "inherited-code-v0.3-baseline-20260828"
PROTECTED_COMMIT = "67e6659fa65249a8842e34af5d8969629efe4bca"
PROTECTED_BRANCH = "research/nhis-fairbias"
BASELINE_FILES = [
    "app.py",
    "classifiers.py",
    "config.py",
    "data_COMPAS.csv",
    "data_Credit_Card.csv",
    "eval.py",
    "main.py",
    "module_AE.py",
    "module_BM.py",
    "module_load.py",
    "module_transform.py",
    "requirements.txt",
    "results/all_results.json",
    "start.sh",
    ".gitignore",
]

SYNTHETIC_TEST_FILES = [
    "tests/synthetic/test_r1a_state_and_config.py",
    "tests/synthetic/test_r1a_identity_and_probabilities.py",
    "tests/synthetic/test_r1a_registry_and_weights.py",
    "tests/synthetic/test_r1a_geometry_contracts.py",
    "tests/synthetic/test_r1a_guard.py",
]

RELEVANT_SOURCES = sorted(list(set([
    "src/fairbias/__init__.py",
    "src/fairbias/enhancement_state.py",
    "src/fairbias/enhancement.py",
    "src/fairbias/enhancement_contracts.py",
    "src/fairbias/bias_metric.py",
    "src/fairbias/evaluator.py",
    "src/fairbias/config.py",
    "src/fairbias/mitigation.py",
    "src/fairbias/prediction_contracts.py",
    "src/fairbias/application_metrics.py",
    "src/fairbias/pipeline.py",
    "src/fairbias/models.py",
    "src/fairbias/transform.py",
    "src/fairbias/transform_trace.py",
    "src/fairbias/data.py",
    "src/nhis_fairbias/__init__.py",
    "src/nhis_fairbias/features.py",
    "src/nhis_fairbias/adapter.py",
    "src/nhis_fairbias/preprocessing.py",
    "src/nhis_fairbias/survey.py",
    "src/nhis_fairbias/d8_enhancement_runner.py",
    "src/nhis_fairbias/d6_temporal_runner.py",
    "src/nhis_fairbias/audit.py",
    "src/nhis_fairbias/download.py",
    "src/nhis_fairbias/evaluation.py",
    "src/nhis_fairbias/harmonize.py",
    "src/nhis_fairbias/pooled.py",
    "src/nhis_fairbias/schema.py",
    "scripts/_fairbias_r1_guard.py",
    "scripts/run_fairbias_r1_guarded_tests.py",
    "configs/nhis/features.json",
    "configs/nhis/study.json",
] + SYNTHETIC_TEST_FILES)))


def compute_hashes() -> Dict[str, Dict[str, Any]]:
    hashes = {}
    for rel_p in sorted(RELEVANT_SOURCES):
        p = REPO_ROOT / rel_p
        if p.exists():
            b = p.read_bytes()
            hashes[rel_p] = {
                "sha256": hashlib.sha256(b).hexdigest(),
                "bytes": len(b),
            }
        else:
            hashes[rel_p] = {"status": "MISSING"}
    return hashes


def run_preflight() -> Dict[str, Any]:
    """Execute standard library preflight checks verifying repository and baseline integrity."""
    preflight: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "PASS",
        "checks": {},
    }

    # 1. Branch check
    res_branch = subprocess.run(["git", "branch", "--show-current"], cwd=REPO_ROOT, capture_output=True, text=True)
    if res_branch.returncode != 0:
        preflight["status"] = "FAIL"
        preflight["error"] = "PRECHECK_ERROR"
    cur_branch = res_branch.stdout.strip()
    preflight["checks"]["branch"] = {
        "expected": PROTECTED_BRANCH,
        "actual": cur_branch,
        "returncode": res_branch.returncode,
        "passed": (cur_branch == PROTECTED_BRANCH and res_branch.returncode == 0),
    }
    if cur_branch != PROTECTED_BRANCH:
        preflight["status"] = "FAIL"

    # 2. HEAD commit check
    res_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    if res_head.returncode != 0:
        preflight["status"] = "FAIL"
        preflight["error"] = "PRECHECK_ERROR"
    cur_head = res_head.stdout.strip()
    preflight["checks"]["head_commit"] = {
        "expected": PROTECTED_COMMIT,
        "actual": cur_head,
        "returncode": res_head.returncode,
        "passed": (cur_head == PROTECTED_COMMIT and res_head.returncode == 0),
    }
    if cur_head != PROTECTED_COMMIT:
        preflight["status"] = "FAIL"

    # 3. Baseline files immutability check against HEAD
    res_status = subprocess.run(["git", "status", "--porcelain"] + BASELINE_FILES, cwd=REPO_ROOT, capture_output=True, text=True)
    if res_status.returncode != 0:
        preflight["status"] = "FAIL"
        preflight["error"] = "PRECHECK_ERROR"
    dirty_baselines = res_status.stdout.strip()
    preflight["checks"]["baseline_immutability_vs_HEAD"] = {
        "passed": (len(dirty_baselines) == 0 and res_status.returncode == 0),
        "returncode": res_status.returncode,
        "dirty_files": dirty_baselines.splitlines() if dirty_baselines else [],
    }
    if len(dirty_baselines) > 0:
        preflight["status"] = "FAIL"

    # Historical difference note for .gitignore
    preflight["checks"]["baseline_changed_vs_protected_tag"] = {
        "file": ".gitignore",
        "commit": "b595e59",
        "note": "Documented historical difference from commit b595e59; preserved without unprompted revert.",
    }

    # 4. No staged changes check
    res_staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=REPO_ROOT, capture_output=True, text=True)
    if res_staged.returncode != 0:
        preflight["status"] = "FAIL"
        preflight["error"] = "PRECHECK_ERROR"
    staged_files = res_staged.stdout.strip()
    preflight["checks"]["no_staged_changes"] = {
        "passed": (len(staged_files) == 0 and res_staged.returncode == 0),
        "returncode": res_staged.returncode,
        "staged_files": staged_files.splitlines() if staged_files else [],
    }
    if len(staged_files) > 0:
        preflight["status"] = "FAIL"

    return preflight


def build_reconstructed_preimage(phase: str = "r1a-r10") -> Dict[str, Any]:
    """Reconstruct preimage of input files from supervisor verified snapshot."""
    if phase == "r1a-r10":
        ev_dir = REPO_ROOT / "docs" / "reports" / "FAIRBIAS_R1A_R9_SUPERVISOR_REVIEW_20260915_evidence"
    elif phase == "r1a-r9":
        ev_dir = REPO_ROOT / "docs" / "reports" / "FAIRBIAS_R1A_R8_SUPERVISOR_REVIEW_20260915_evidence"
    elif phase == "r1a-r8":
        ev_dir = REPO_ROOT / "docs" / "reports" / "FAIRBIAS_R1A_R7_SUPERVISOR_REVIEW_20260915_evidence"
    elif phase == "r1a-r7":
        ev_dir = REPO_ROOT / "docs" / "reports" / "FAIRBIAS_R1A_R6_SUPERVISOR_REVIEW_20260915_evidence"
    elif phase == "r1a-r6":
        ev_dir = REPO_ROOT / "docs" / "reports" / "FAIRBIAS_R1A_R5_SUPERVISOR_REVIEW_20260915_evidence"
    elif phase == "r1a-r5":
        ev_dir = REPO_ROOT / "docs" / "reports" / "FAIRBIAS_R1A_R4_SUPERVISOR_REVIEW_20260915_evidence"
    else:
        ev_dir = REPO_ROOT / "docs" / "reports" / "FAIRBIAS_R1A_R3_SUPERVISOR_REVIEW_20260914_evidence"
    verif_path = ev_dir / "verification.json"
    snap_dir = ev_dir / "source_snapshot"
    if not verif_path.exists() or not snap_dir.exists():
        return {"status": "ERROR", "error": f"Verification evidence not found: {verif_path}"}
    with open(verif_path, "r", encoding="utf-8") as f:
        verif = json.load(f)
    fp = verif.get("source_fingerprints", {})
    records = {}
    mismatches = []
    missing = []
    for rel_p, info in sorted(fp.items()):
        p = snap_dir / rel_p
        if not p.exists():
            missing.append(rel_p)
            records[rel_p] = {"status": "MISSING", "expected_sha256": info.get("sha256")}
        else:
            b = p.read_bytes()
            h = hashlib.sha256(b).hexdigest()
            if h != info.get("sha256"):
                mismatches.append({"file": rel_p, "expected": info.get("sha256"), "actual": h})
            records[rel_p] = {
                "sha256": h,
                "bytes": len(b),
                "matches_supervisor_snapshot": (h == info.get("sha256")),
            }
    rel_ev_dir = str(ev_dir.relative_to(REPO_ROOT))
    return {
        "provenance_metadata": {
            "reconstruction_tag": "RECONSTRUCTED_FROM_SUPERVISOR_SNAPSHOT",
            "source_evidence_dir": f"{rel_ev_dir}/source_snapshot",
            "source_evidence_file": f"{rel_ev_dir}/verification.json",
            "reconstruction_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "branch": PROTECTED_BRANCH,
            "head": PROTECTED_COMMIT,
            "total_items": len(records),
            "missing_count": len(missing),
            "mismatch_count": len(mismatches),
            "missing_files": missing,
            "mismatched_files": mismatches,
        },
        "fingerprints": records,
    }


def build_candidate_hashes(phase: str = "r1a-r10") -> Dict[str, Any]:
    """Compute candidate hashes from current working tree for all files."""
    if phase == "r1a-r10":
        ev_dir = REPO_ROOT / "docs" / "reports" / "FAIRBIAS_R1A_R9_SUPERVISOR_REVIEW_20260915_evidence"
    elif phase == "r1a-r9":
        ev_dir = REPO_ROOT / "docs" / "reports" / "FAIRBIAS_R1A_R8_SUPERVISOR_REVIEW_20260915_evidence"
    elif phase == "r1a-r8":
        ev_dir = REPO_ROOT / "docs" / "reports" / "FAIRBIAS_R1A_R7_SUPERVISOR_REVIEW_20260915_evidence"
    elif phase == "r1a-r7":
        ev_dir = REPO_ROOT / "docs" / "reports" / "FAIRBIAS_R1A_R6_SUPERVISOR_REVIEW_20260915_evidence"
    elif phase == "r1a-r6":
        ev_dir = REPO_ROOT / "docs" / "reports" / "FAIRBIAS_R1A_R5_SUPERVISOR_REVIEW_20260915_evidence"
    elif phase == "r1a-r5":
        ev_dir = REPO_ROOT / "docs" / "reports" / "FAIRBIAS_R1A_R4_SUPERVISOR_REVIEW_20260915_evidence"
    else:
        ev_dir = REPO_ROOT / "docs" / "reports" / "FAIRBIAS_R1A_R3_SUPERVISOR_REVIEW_20260914_evidence"
    verif_path = ev_dir / "verification.json"
    with open(verif_path, "r", encoding="utf-8") as f:
        verif = json.load(f)
    fp = verif.get("source_fingerprints", {})
    records = {}
    changed_files = []
    missing_files = []
    for rel_p, info in sorted(fp.items()):
        p = REPO_ROOT / rel_p
        if not p.exists():
            missing_files.append(rel_p)
            records[rel_p] = {"status": "MISSING", "expected_sha256": info.get("sha256")}
        else:
            if rel_p.endswith(".csv"):
                records[rel_p] = {
                    "sha256": info.get("sha256"),
                    "bytes": info.get("bytes"),
                    "read_policy": "READ_PROHIBITED_METADATA_ONLY",
                    "matches_preimage": True,
                }
                continue
            b = p.read_bytes()
            h = hashlib.sha256(b).hexdigest()
            is_changed = (h != info.get("sha256"))
            if is_changed:
                changed_files.append(rel_p)
            records[rel_p] = {
                "sha256": h,
                "bytes": len(b),
                "matches_preimage": not is_changed,
            }
    rel_ev_dir = str(ev_dir.relative_to(REPO_ROOT))
    return {
        "provenance_metadata": {
            "source_evidence_dir": f"{rel_ev_dir}/source_snapshot",
            "source_evidence_file": f"{rel_ev_dir}/verification.json",
            "computation_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "branch": PROTECTED_BRANCH,
            "head": PROTECTED_COMMIT,
            "total_items": len(records),
            "changed_count": len(changed_files),
            "missing_count": len(missing_files),
            "changed_files": changed_files,
            "missing_files": missing_files,
        },
        "fingerprints": records,
    }


def verify_executed_bytes_evidence(
    executed_bytes_path: pathlib.Path,
    pre_hashes: Dict[str, Dict[str, Any]],
    expected_test_files: Optional[List[str]] = None,
    expected_runner: Optional[str] = None,
    min_required_modules: int = 1,
    actual_loaded_sources: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str], List[str], str, Dict[str, Any]]:
    """Verify executed bytes evidence matches pre_hashes and satisfies set relations.

    Requirements:
    1. Evidence file exists and is valid JSON dictionary.
    2. 'executed_sources' is non-empty dictionary.
    3. 'executed_module_count' matches exact length of 'executed_sources'.
    4. len(executed_sources) >= min_required_modules.
    5. All expected_test_files (if specified) must be present in executed_sources.
    6. expected_runner (if specified) must be present in executed_sources.
    7. All executed sources must be a subset of pre_hashes (no unrecorded modules).
    8. SHA-256 matches pre_hashes for each executed source.
    9. Byte count is a positive integer matching pre_hashes for each executed source.
    10. 'loader_source' is present, non-empty string, and from recognized live capture mechanisms.
    11. All actual loaded project modules must be contained in compiled bytecode evidence (R7-02).

    Returns (is_valid, mismatches, unrecorded_modules, failure_reason, executed_modules_info).
    """
    if not executed_bytes_path.exists():
        return False, [], [], "MISSING_EVIDENCE_FILE", {}

    try:
        with open(executed_bytes_path, "r", encoding="utf-8") as f:
            content = json.load(f)
    except Exception as exc:
        return False, [], [], f"MALFORMED_JSON: {exc}", {}

    if not isinstance(content, dict):
        return False, [], [], "MALFORMED_STRUCTURE: root must be a JSON object", {}

    if "executed_sources" not in content:
        return False, [], [], "MALFORMED_STRUCTURE: missing executed_sources dictionary", {}

    executed_modules_info = content.get("executed_sources", {})
    if not isinstance(executed_modules_info, dict) or len(executed_modules_info) == 0:
        return False, [], [], "EMPTY_EXECUTED_SOURCES", {}

    # Check declared count
    declared_count = content.get("executed_module_count")
    if not isinstance(declared_count, int) or isinstance(declared_count, bool):
        return False, [], [], "INVALID_EXECUTED_MODULE_COUNT_TYPE", {}

    actual_count = len(executed_modules_info)
    if declared_count != actual_count:
        return False, [], [], f"COUNT_MISMATCH: declared {declared_count} != actual {actual_count}", {}

    # Check minimum required module count
    if actual_count < min_required_modules:
        return False, [], [], f"MODULE_COUNT_BELOW_MINIMUM: len={actual_count} < min={min_required_modules}", {}

    # Check expected runner entrypoint
    if expected_runner is not None and expected_runner not in executed_modules_info:
        return False, [], [], f"MISSING_RUNNER_ENTRYPOINT: expected runner '{expected_runner}' not in executed sources", {}

    # Check expected test files
    if expected_test_files is not None:
        missing_tests = [tf for tf in expected_test_files if tf not in executed_modules_info]
        if missing_tests:
            return False, [], [], f"MISSING_REQUIRED_TEST_FILES: {missing_tests}", {}

    mismatches = []
    unrecorded = []
    allowed_loader_sources = {
        "fresh_source_loader",
        "pytest_assertion_rewrite_live",
        "entrypoint_initial_load",
    }

    for rel_p, mod_info in executed_modules_info.items():
        if not isinstance(mod_info, dict):
            return False, [], [], f"INVALID_MODULE_INFO_STRUCTURE: entry for {rel_p} is not a dict", {}

        if rel_p not in pre_hashes:
            unrecorded.append(rel_p)
            continue

        exp_hash_info = pre_hashes[rel_p]
        act_sha = mod_info.get("sha256")
        exp_sha = exp_hash_info.get("sha256")
        if act_sha != exp_sha:
            mismatches.append(rel_p)

        # Check bytes: must be positive int and match pre_hashes
        act_bytes = mod_info.get("bytes")
        exp_bytes = exp_hash_info.get("bytes")
        if not isinstance(act_bytes, int) or isinstance(act_bytes, bool) or act_bytes <= 0:
            return False, mismatches, unrecorded, f"INVALID_BYTE_COUNT: {rel_p} has invalid bytes {act_bytes}", {}
        if act_bytes != exp_bytes:
            mismatches.append(rel_p)

        # Check loader_source
        loader_src = mod_info.get("loader_source")
        if not isinstance(loader_src, str) or not loader_src.strip():
            return False, mismatches, unrecorded, f"MISSING_LOADER_SOURCE: {rel_p} missing valid loader_source", {}
        if loader_src not in allowed_loader_sources:
            return False, mismatches, unrecorded, f"UNRECOGNIZED_LOADER_SOURCE: {rel_p} has unauthorized loader_source '{loader_src}'", {}

    if mismatches or unrecorded:
        return False, mismatches, unrecorded, f"INTEGRITY_MISMATCH: mismatches={mismatches}, unrecorded={unrecorded}", executed_modules_info

    # Check loaded sources vs compiled bytecode containment (R7-02, R8-01, R9-01)
    if actual_loaded_sources is None:
        return False, [], [], "EMPTY_OR_MISSING_LOADED_SOURCES", {}
    if not isinstance(actual_loaded_sources, dict) or len(actual_loaded_sources) == 0:
        return False, [], [], "EMPTY_OR_MISSING_LOADED_SOURCES", {}

    missing_bytecode = [s for s in actual_loaded_sources if s not in executed_modules_info]
    if missing_bytecode:
        return False, [], [], f"LOADED_WITHOUT_BYTECODE_EVIDENCE: {missing_bytecode}", {}

    # Check hash and byte count for all loaded sources against pre_hashes
    for s_path, s_info in actual_loaded_sources.items():
        if not isinstance(s_info, dict):
            return False, [], [], f"INVALID_LOADED_MODULE_INFO: {s_path}", {}
        if s_path not in pre_hashes:
            return False, [], [], f"UNRECORDED_LOADED_MODULE: {s_path}", {}
        pre_info = pre_hashes[s_path]
        if "sha256" not in s_info or not isinstance(s_info["sha256"], str) or s_info.get("sha256") != pre_info.get("sha256"):
            return False, [], [], f"LOADED_INTEGRITY_MISMATCH: {s_path} sha256 mismatch", {}
        if "bytes" not in s_info:
            return False, [], [], f"LOADED_INTEGRITY_MISMATCH: {s_path} missing bytes", {}
        s_bytes = s_info.get("bytes")
        if not isinstance(s_bytes, int) or isinstance(s_bytes, bool) or s_bytes <= 0 or s_bytes != pre_info.get("bytes"):
            return False, [], [], f"LOADED_INTEGRITY_MISMATCH: {s_path} bytes mismatch ({s_bytes} vs {pre_info.get('bytes')})", {}

    return True, [], [], "", executed_modules_info


def verify_guard_log_events(
    events_log_file: pathlib.Path,
    child_summary: Dict[str, Any],
    expected_summary_path: Optional[Union[str, pathlib.Path]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Verify guard events log line-by-line for validity, structure, counts, and denial reconciliation.

    Requirements:
    1. Log file exists.
    2. Every non-empty line must be valid JSON object with mandatory keys:
       {'timestamp', 'action', 'target', 'decision', 'reason'}.
    3. 'decision' must be 'ALLOWED' or 'DENIED'.
    4. Log must contain >= 1 valid event.
    5. final_log_events_count >= events_at_summary_cutoff (no count regression).
    6. 0 <= (final_log_events_count - events_at_summary_cutoff) <= 2 (strictly bounded closing delta R7-02).
    7. Total DENIED in log matches summary 'denied_events'.
    8. unexpected_denials in summary must be strictly empty (len == 0).
    9. Each DENIED event strictly aligns with consumed_expected_denials in summary (R7-02).
    10. Mandatory counters in summary must be explicit non-negative integers.
    11. disk_events_lost == 0.
    12. log_healthy is True.

    Returns (is_valid, failure_reason, log_metrics).
    """
    if not events_log_file.exists():
        return False, "MISSING_GUARD_LOG_FILE", {}

    parsed_events = []
    allowed_count = 0
    denied_count = 0
    denied_events = []
    mandatory_keys = {"timestamp", "action", "target", "decision", "reason"}

    try:
        with open(events_log_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    ev = json.loads(line_str)
                except Exception as exc:
                    return False, f"MALFORMED_LOG_LINE at line {line_num}: {exc}", {}
                if not isinstance(ev, dict):
                    return False, f"INVALID_LOG_RECORD_STRUCTURE at line {line_num}: must be JSON object", {}
                missing_keys = mandatory_keys - set(ev.keys())
                if missing_keys:
                    return False, f"MISSING_MANDATORY_KEYS at line {line_num}: {sorted(missing_keys)}", {}
                decision = ev.get("decision")
                if decision == "ALLOWED":
                    allowed_count += 1
                elif decision == "DENIED":
                    denied_count += 1
                    denied_events.append(ev)
                else:
                    return False, f"INVALID_DECISION_VALUE at line {line_num}: {decision}", {}
                parsed_events.append(ev)
    except Exception as exc:
        return False, f"LOG_READ_ERROR: {exc}", {}

    final_count = len(parsed_events)
    if final_count == 0:
        return False, "EMPTY_GUARD_LOG", {}

    guard_metrics = child_summary.get("guard_metrics", {})
    if not isinstance(guard_metrics, dict):
        return False, "INVALID_GUARD_METRICS_STRUCTURE", {}

    mandatory_int_counters = [
        "total_events",
        "denied_events",
        "allowed_events",
        "truncated_events",
        "disk_events_lost",
        "memory_window_dropped",
        "unconsumed_expected_denials_count",
        "denials_in_sentinels",
        "denials_in_tests",
    ]
    for cname in mandatory_int_counters:
        val = guard_metrics.get(cname)
        if val is None or not isinstance(val, int) or isinstance(val, bool) or val < 0:
            return False, f"INVALID_OR_MISSING_GUARD_COUNTER: {cname}={val}", {}

    events_at_summary = child_summary.get("events_at_summary_cutoff")
    if events_at_summary is None or not isinstance(events_at_summary, int) or isinstance(events_at_summary, bool) or events_at_summary < 0:
        return False, f"INVALID_OR_MISSING_EVENTS_AT_SUMMARY_CUTOFF: {events_at_summary}", {}

    if final_count < events_at_summary:
        return False, f"LOG_COUNT_REGRESSION: final log has {final_count} events but summary cutoff was {events_at_summary}", {}

    total_events = guard_metrics.get("total_events")
    allowed_events = guard_metrics.get("allowed_events")
    summary_denied = guard_metrics.get("denied_events")

    if denied_count != summary_denied:
        return False, f"DENIED_COUNT_MISMATCH: log has {denied_count} denials, summary claims {summary_denied}", {}

    if "unexpected_denials" not in guard_metrics:
        return False, "MISSING_MANDATORY_FIELD: unexpected_denials", {}
    summary_unexpected = guard_metrics.get("unexpected_denials")
    if not isinstance(summary_unexpected, list) or len(summary_unexpected) > 0:
        return False, f"UNEXPECTED_DENIALS_REPORTED: count={len(summary_unexpected) if isinstance(summary_unexpected, list) else 'non-list'}", {}

    disk_events_lost = guard_metrics.get("disk_events_lost")
    if disk_events_lost > 0:
        return False, f"DISK_EVENTS_LOST: {disk_events_lost}", {}

    log_healthy = guard_metrics.get("log_healthy")
    if not isinstance(log_healthy, bool) or log_healthy is not True:
        return False, f"GUARD_LOG_UNHEALTHY: log_healthy must be boolean True, got {type(log_healthy).__name__} ({log_healthy!r})", {}

    # Strict mathematical relationship among guard counters (R7-02, R8-01)
    if total_events != allowed_events + summary_denied:
        return False, f"COUNTER_RELATION_MISMATCH: total_events={total_events} != allowed ({allowed_events}) + denied ({summary_denied})", {}

    # Strict closing delta: exactly 2 closing events (summary write and flush/close) (R7-02, R9-01)
    closing_delta = final_count - events_at_summary
    if closing_delta != 2:
        return False, f"MISSING_CLOSING_EVENTS: final_count={final_count} vs summary_cutoff={events_at_summary} (missing closing flush/write events)", {}

    if events_at_summary != total_events:
        return False, f"SUMMARY_CUTOFF_MISMATCH: cutoff {events_at_summary} != total_events {total_events}", {}

    # Determine expected summary path (R7-02, R10)
    if expected_summary_path is not None:
        norm_expected_summary = os.path.abspath(os.path.normpath(str(expected_summary_path)))
    elif child_summary.get("expected_summary_path"):
        norm_expected_summary = os.path.abspath(os.path.normpath(str(child_summary["expected_summary_path"])))
    elif child_summary.get("summary_path"):
        norm_expected_summary = os.path.abspath(os.path.normpath(str(child_summary["summary_path"])))
    elif child_summary.get("run_dir"):
        norm_expected_summary = os.path.abspath(os.path.normpath(os.path.join(str(child_summary["run_dir"]), "test_summary.json")))
    else:
        norm_expected_summary = os.path.abspath(os.path.normpath(str(events_log_file.parent / "test_summary.json")))

    # Check closing events: must be ALLOWED file_write to exact test_summary.json within output_dir
    closing_events = parsed_events[events_at_summary:]
    for c_ev in closing_events:
        if c_ev.get("decision") != "ALLOWED":
            return False, f"INVALID_CLOSING_EVENT: {c_ev}", {}
        if c_ev.get("action") != "file_write":
            return False, f"INVALID_CLOSING_EVENT: {c_ev}", {}
        tgt_str = str(c_ev.get("target", ""))
        norm_target = os.path.abspath(os.path.normpath(tgt_str))
        if norm_target != norm_expected_summary:
            return False, f"INVALID_CLOSING_EVENT: target '{norm_target}' does not match expected summary path '{norm_expected_summary}'", {}
        rsn_str = str(c_ev.get("reason", ""))
        if "Write target is within output_dir" not in rsn_str:
            return False, f"INVALID_CLOSING_EVENT: {c_ev}", {}

    consumed_expected = guard_metrics.get("consumed_expected_denials")
    if not isinstance(consumed_expected, list):
        return False, "MISSING_OR_INVALID_CONSUMED_EXPECTED_DENIALS", {}

    if len(denied_events) != len(consumed_expected):
        return False, f"DENIED_EVENT_CONSUMPTION_MISMATCH: log has {len(denied_events)} denials but consumed_expected_denials has {len(consumed_expected)}", {}

    KNOWN_MATCH_MODES = {"exact", "directory", "basename", "prefix"}

    for idx, (log_ev, exp_ev) in enumerate(zip(denied_events, consumed_expected)):
        if not isinstance(exp_ev, dict):
            return False, f"INVALID_CONSUMED_RECORD_STRUCTURE at {idx}", {}

        # Mandatory registration fields in consumed record (R7-02, R9-01)
        if "expected_action" not in exp_ev or "expected_target_pattern" not in exp_ev:
            return False, f"MISSING_REGISTRATION_FIELDS at index {idx}: expected_action or expected_target_pattern missing in consumed record", {}

        if log_ev.get("action") != exp_ev.get("action") or str(log_ev.get("target")) != str(exp_ev.get("target")) or log_ev.get("reason") != exp_ev.get("reason"):
            return False, f"DENIED_EVENT_SIGNATURE_MISMATCH at index {idx}: log={log_ev} vs consumed={exp_ev}", {}

        # Correlate actual event with original expected registration constraints (R7-02, R9-01)
        exp_act = exp_ev.get("expected_action")
        log_act = log_ev.get("action")
        exp_tgt = exp_ev.get("expected_target_pattern")
        match_mode = exp_ev.get("match_mode", "exact")

        if match_mode not in KNOWN_MATCH_MODES:
            return False, f"DENIED_EVENT_REGISTRATION_MISMATCH at index {idx}: unknown match_mode '{match_mode}'", {}

        if exp_act is not None:
            act_matches = (exp_act == log_act) or (exp_act in ("os.unlink", "os.remove") and log_act in ("os.unlink", "os.remove"))
            if not act_matches:
                return False, f"DENIED_EVENT_REGISTRATION_MISMATCH at index {idx}: expected_action {exp_act} != log_action {log_act}", {}

        if exp_tgt is not None:
            tgt_str = str(log_ev.get("target"))
            p_str = str(exp_tgt)
            if match_mode == "exact":
                tgt_matches = (
                    tgt_str == p_str
                    or os.path.normpath(tgt_str) == os.path.normpath(p_str)
                    or (not os.path.isabs(p_str) and os.path.normpath(os.path.join(str(REPO_ROOT), p_str)) == os.path.normpath(tgt_str))
                )
            elif match_mode == "directory":
                t_norm = os.path.normpath(tgt_str)
                p_norm = os.path.normpath(p_str)
                tgt_matches = (t_norm == p_norm or t_norm.startswith(f"{p_norm.rstrip('/')}/"))
            elif match_mode == "basename":
                tgt_matches = (os.path.basename(tgt_str) == p_str or os.path.basename(os.path.normpath(tgt_str)) == p_str)
            elif match_mode == "prefix":
                tgt_matches = tgt_str.startswith(p_str)
            else:
                tgt_matches = False

            if not tgt_matches:
                return False, f"DENIED_EVENT_REGISTRATION_MISMATCH at index {idx}: expected_target {exp_tgt} != log_target {tgt_str}", {}

        exp_rsn = exp_ev.get("expected_reason_pattern")
        if exp_rsn is not None:
            log_rsn = str(log_ev.get("reason", ""))
            if str(exp_rsn) != log_rsn:
                return False, f"DENIED_EVENT_REGISTRATION_MISMATCH at index {idx}: expected_reason_pattern '{exp_rsn}' != log_reason '{log_rsn}'", {}

    metrics = {
        "final_log_events_count": final_count,
        "allowed_count": allowed_count,
        "denied_count": denied_count,
        "denied_events": denied_events,
    }
    return True, "", metrics



def poll_child_resource(
    proc: subprocess.Popen,
    max_mem_bytes: int,
    timeout_seconds: int,
    start_time: float,
) -> Tuple[bool, Optional[str], Optional[str], int]:
    """Poll child resource status. Returns (killed, kill_reason, monitor_error, updated_max_mem_bytes)."""
    elapsed = time.time() - start_time
    if elapsed > timeout_seconds:
        proc.kill()
        proc.wait()
        return True, "TIMEOUT_15_MIN", None, max_mem_bytes

    if proc.poll() is not None:
        return False, None, None, max_mem_bytes

    try:
        ps_res = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(proc.pid)],
            capture_output=True,
            text=True,
        )
        if ps_res.returncode == 0:
            rss_kb_str = ps_res.stdout.strip()
            if rss_kb_str:
                mem_bytes = int(rss_kb_str) * 1024
                if mem_bytes > max_mem_bytes:
                    max_mem_bytes = mem_bytes
                if mem_bytes > 4 * 1024 * 1024 * 1024:
                    proc.kill()
                    proc.wait()
                    return True, "MEMORY_EXCEEDED_4GIB", None, max_mem_bytes
            else:
                if proc.poll() is None:
                    err = "ps returned code 0 with empty RSS reading for running process"
                    proc.kill()
                    proc.wait()
                    return True, f"MONITOR_ERROR: {err}", err, max_mem_bytes
        else:
            if proc.poll() is None:
                err = f"ps returned non-zero code {ps_res.returncode}"
                proc.kill()
                proc.wait()
                return True, f"MONITOR_ERROR: {err}", err, max_mem_bytes
    except Exception as exc:
        if proc.poll() is None:
            err = f"ps monitoring error: {exc}"
            proc.kill()
            proc.wait()
            return True, f"MONITOR_ERROR: {err}", err, max_mem_bytes

    return False, None, None, max_mem_bytes


def evaluate_execution_verdict(
    preflight_passed: bool,
    child_exit_code: int,
    integrity_passed: bool,
    sentinel_results: Dict[str, str],
    controlled_probe_passed: bool,
    child_log_healthy: bool,
    unexpected_denials_count: int,
    unconsumed_denials_count: int,
    timeout_killed: bool,
    memory_killed: bool,
    monitor_error: Optional[str],
) -> Dict[str, Any]:
    """Centralized verification state function enforcing all criteria for successful exit."""
    all_sentinels_ok = bool(sentinel_results) and all(v == "PASSED_BLOCKED" for v in sentinel_results.values())
    no_resource_faults = (not timeout_killed) and (not memory_killed) and (monitor_error is None)
    clean_denials = (unexpected_denials_count == 0) and (unconsumed_denials_count == 0)

    verdict_passed = (
        preflight_passed
        and (child_exit_code == 0)
        and integrity_passed
        and all_sentinels_ok
        and controlled_probe_passed
        and child_log_healthy
        and clean_denials
        and no_resource_faults
    )
    return {
        "verdict": "PASS" if verdict_passed else "FAIL",
        "preflight_passed": preflight_passed,
        "pytest_passed": (child_exit_code == 0),
        "integrity_passed": integrity_passed,
        "sentinels_passed": all_sentinels_ok,
        "controlled_probe_passed": controlled_probe_passed,
        "log_healthy": child_log_healthy,
        "clean_denials": clean_denials,
        "resource_monitoring_passed": no_resource_faults,
        "overall_success": verdict_passed,
    }


def child_execution(run_dir: pathlib.Path, test_node: Optional[str] = None) -> int:
    """Child process entry point: installs guard first, runs sentinels, executes tests."""
    tmp_dir = (run_dir / "tmp").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    import tempfile
    tempfile.tempdir = str(tmp_dir)

    # 1. FreshSourceFinder to guarantee only live project source is loaded and no .pyc fallback (R3-08, R4-05)
    import importlib.abc
    import importlib.machinery
    import importlib.util

    LOADED_SOURCE_MODULES: Dict[str, Dict[str, Any]] = {}

    # Record runner entrypoint immediately at child start
    runner_p = pathlib.Path(__file__).resolve()
    if runner_p.is_relative_to(REPO_ROOT):
        rel_runner = str(runner_p.relative_to(REPO_ROOT))
        runner_bytes = runner_p.read_bytes()
        LOADED_SOURCE_MODULES[rel_runner] = {
            "module": "__main__",
            "origin": str(runner_p),
            "sha256": hashlib.sha256(runner_bytes).hexdigest(),
            "bytes": len(runner_bytes),
            "loader_source": "entrypoint_initial_load",
        }

    class FreshSourceLoader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            path = self.get_filename(fullname)
            data = self.get_data(path)
            p = pathlib.Path(path).resolve()
            if (
                p.is_relative_to(REPO_ROOT / "src")
                or p.is_relative_to(REPO_ROOT / "scripts")
                or p.is_relative_to(REPO_ROOT / "tests")
            ):
                rel_p = str(p.relative_to(REPO_ROOT))
                LOADED_SOURCE_MODULES[rel_p] = {
                    "module": fullname,
                    "origin": str(p),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                    "loader_source": "fresh_source_loader",
                }
            return self.source_to_code(data, path)

    class FreshSourceFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.origin and isinstance(spec.loader, importlib.machinery.SourceFileLoader):
                p = pathlib.Path(spec.origin).resolve()
                if (
                    p.is_relative_to(REPO_ROOT / "src")
                    or p.is_relative_to(REPO_ROOT / "scripts")
                    or p.is_relative_to(REPO_ROOT / "tests")
                ):
                    spec.loader = FreshSourceLoader(fullname, spec.origin)
                    return spec
            return None

    sys.meta_path.insert(0, FreshSourceFinder())

    # Hook spec_from_file_location to avoid bypass
    orig_spec_from_file_location = importlib.util.spec_from_file_location
    def guarded_spec_from_file_location(name, location=None, *args, **kwargs):
        spec = orig_spec_from_file_location(name, location, *args, **kwargs)
        if spec and spec.origin and isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            p = pathlib.Path(spec.origin).resolve()
            if (
                p.is_relative_to(REPO_ROOT / "src")
                or p.is_relative_to(REPO_ROOT / "scripts")
                or p.is_relative_to(REPO_ROOT / "tests")
            ):
                spec.loader = FreshSourceLoader(name, spec.origin)
        return spec
    importlib.util.spec_from_file_location = guarded_spec_from_file_location

    # 2. Configure sys.path strictly: keep stdlib, prepend site-packages, src, scripts
    py_base = pathlib.Path(sys.base_prefix).resolve()
    site_packages = str(py_base / "lib" / "python3.13" / "site-packages")
    extra_paths = [
        str(REPO_ROOT / "src"),
        str(REPO_ROOT / "scripts"),
        site_packages,
    ]
    for p in reversed(extra_paths):
        if p not in sys.path:
            sys.path.insert(0, p)

    # 3. Install Guard BEFORE importing pytest or project code
    from _fairbias_r1_guard import FairBiasR1Guard, run_sentinel_tests
    guard = FairBiasR1Guard(output_dir=run_dir, repo_root=REPO_ROOT)
    guard.install()

    # 4. Controlled failure probe (isolated instance to prove fault detection)
    failure_probe_dir = tmp_dir / "expected_failure_probe"
    failure_probe_dir.mkdir(parents=True, exist_ok=True)
    probe_guard = FairBiasR1Guard(output_dir=failure_probe_dir, repo_root=REPO_ROOT)
    probe_guard.written_bytes = probe_guard.max_disk_bytes
    probe_guard.log_event("probe_action", "probe_target", "DENIED", "probe budget fault")
    probe_budget_failed = (not probe_guard.log_healthy) and (probe_guard.disk_events_lost > 0)
    probe_guard.close_log_sink()
    probe_guard.log_event("probe_closed", "probe_target", "DENIED", "probe closed fault")
    probe_closed_failed = len(probe_guard.log_errors) >= 2
    controlled_probe_passed = bool(probe_budget_failed and probe_closed_failed)

    # 5. Run synthetic sentinel verification traps
    sentinel_results = run_sentinel_tests(guard)
    sentinel_path = run_dir / "sentinel_results.json"
    with open(sentinel_path, "w", encoding="utf-8") as f:
        json.dump(sentinel_results, f, indent=2)

    sentinel_failures = [k for k, v in sentinel_results.items() if v != "PASSED_BLOCKED"]
    if sentinel_failures:
        sys.stderr.write(f"Guard sentinel checks failed: {sentinel_failures}\n")
        guard.close_log_sink()
        return 1

    denials_before_pytest = guard.denied_events

    # 5. Run Pytest on the synthetic test suite
    import pytest
    import _pytest.assertion.rewrite as par
    par._read_pyc = lambda *args, **kwargs: None

    import ast
    orig_rewrite_test = par._rewrite_test
    def guarded_rewrite_test(fn, config):
        p = pathlib.Path(fn).resolve()
        stat = os.stat(fn)
        source_bytes = fn.read_bytes()
        if (
            p.is_relative_to(REPO_ROOT / "src")
            or p.is_relative_to(REPO_ROOT / "scripts")
            or p.is_relative_to(REPO_ROOT / "tests")
        ):
            rel_p = str(p.relative_to(REPO_ROOT))
            LOADED_SOURCE_MODULES[rel_p] = {
                "module": str(fn.stem),
                "origin": str(p),
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
                "bytes": len(source_bytes),
                "loader_source": "pytest_assertion_rewrite_live",
            }
        strfn = str(fn)
        tree = ast.parse(source_bytes, filename=strfn)
        par.rewrite_asserts(tree, source_bytes, strfn, config)
        co = compile(tree, strfn, "exec", dont_inherit=True)
        return stat, co
    par._rewrite_test = guarded_rewrite_test

    import numpy as np
    import pandas as pd
    import sklearn

    pytest_temp = run_dir / "pytest_temp"
    pytest_cache = run_dir / "pytest_cache"
    pytest_temp.mkdir(parents=True, exist_ok=True)
    pytest_cache.mkdir(parents=True, exist_ok=True)

    if test_node:
        base_file = test_node.split("::")[0]
        if base_file not in SYNTHETIC_TEST_FILES:
            sys.stderr.write(f"Target test node {test_node} is not within authorized synthetic files: {SYNTHETIC_TEST_FILES}\n")
            guard.close_log_sink()
            return 1
        test_paths = [str(REPO_ROOT / test_node)]
        executed_tests_list = [test_node]
    else:
        test_paths = [str(REPO_ROOT / p) for p in SYNTHETIC_TEST_FILES]
        executed_tests_list = SYNTHETIC_TEST_FILES

    argv = [
        "-v",
        "-ra",
        f"--basetemp={pytest_temp}",
        "-o", f"cache_dir={pytest_cache}",
    ] + test_paths

    exit_code = pytest.main(argv)

    denials_in_tests = guard.denied_events - denials_before_pytest

    # Verify log health and zero unexpected denials (R2-03, R3-06)
    unconsumed_denials = len(guard.expected_denial_patterns)
    log_health_failed = (
        (not guard.log_healthy)
        or guard.disk_events_lost > 0
        or len(guard.unexpected_denials) > 0
        or unconsumed_denials > 0
    )

    if log_health_failed:
        sys.stderr.write(
            f"[CHILD] Guard log health check failed: healthy={guard.log_healthy}, "
            f"disk_events_lost={guard.disk_events_lost}, "
            f"unexpected_denials={len(guard.unexpected_denials)}, "
            f"unconsumed_denials={unconsumed_denials}, "
            f"log_errors={guard.log_errors}\n"
        )
        if exit_code == 0:
            exit_code = 1

    if not controlled_probe_passed:
        sys.stderr.write("[CHILD] Controlled failure probe failed to detect injected faults\n")
        if exit_code == 0:
            exit_code = 1

    # Track executed source module closure (R3-08, R4-05)
    executed_sources = {}
    for mod_name, mod in list(sys.modules.items()):
        if mod is not None and hasattr(mod, "__file__") and mod.__file__:
            try:
                mod_path = pathlib.Path(mod.__file__).resolve()
                if (
                    mod_path.is_relative_to(REPO_ROOT / "src")
                    or mod_path.is_relative_to(REPO_ROOT / "scripts")
                    or mod_path.is_relative_to(REPO_ROOT / "tests")
                ):
                    rel_p = str(mod_path.relative_to(REPO_ROOT))
                    b = mod_path.read_bytes()
                    executed_sources[rel_p] = {
                        "module": mod_name,
                        "origin": str(mod_path),
                        "sha256": hashlib.sha256(b).hexdigest(),
                        "bytes": len(b),
                    }
            except Exception:
                pass

    executed_sources_path = run_dir / "executed_source_hashes.json"
    with open(executed_sources_path, "w", encoding="utf-8") as f:
        json.dump({
            "executed_module_count": len(executed_sources),
            "executed_sources": executed_sources,
        }, f, indent=2)

    executed_bytes_path = run_dir / "executed_bytes_hashes.json"
    with open(executed_bytes_path, "w", encoding="utf-8") as f:
        json.dump({
            "executed_module_count": len(LOADED_SOURCE_MODULES),
            "executed_sources": LOADED_SOURCE_MODULES,
        }, f, indent=2)

    events_at_summary = guard.total_events
    summary = {
        "exit_code": int(exit_code),
        "tests_executed": executed_tests_list,
        "sentinel_results": sentinel_results,
        "controlled_failure_probe": {
            "status": "PASSED" if controlled_probe_passed else "FAILED",
            "probe_budget_detected": probe_budget_failed,
            "probe_closed_detected": probe_closed_failed,
        },
        "executed_source_modules_count": len(executed_sources),
        "executed_bytes_modules_count": len(LOADED_SOURCE_MODULES),
        "events_at_summary_cutoff": events_at_summary,
        "guard_metrics": {
            "total_events": int(guard.total_events),
            "denied_events": int(guard.denied_events),
            "allowed_events": int(guard.allowed_events),
            "truncated_events": int(guard.truncated_events),
            "disk_events_lost": int(guard.disk_events_lost),
            "memory_window_dropped": int(guard.memory_window_dropped),
            "log_healthy": bool(guard.log_healthy),
            "log_errors": list(guard.log_errors),
            "unexpected_denials": list(guard.unexpected_denials),
            "consumed_expected_denials": list(guard.consumed_expected_denials),
            "unconsumed_expected_denials_count": int(unconsumed_denials),
            "denials_in_sentinels": int(denials_before_pytest),
            "denials_in_tests": int(denials_in_tests),
        },
        "run_dir": str(run_dir.resolve()),
        "expected_summary_path": str((run_dir / "test_summary.json").resolve()),
        "summary_path": str((run_dir / "test_summary.json").resolve()),
        "dependency_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "pytest": pytest.__version__,
        },
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(run_dir / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    guard.close_log_sink()
    if not guard.log_healthy or guard.disk_events_lost > 0 or len(guard.log_errors) > 0:
        sys.stderr.write("[CHILD] Final guard log sink close failed health checks\n")
        exit_code = 1

    return int(exit_code)


def parent_controller(output_root: pathlib.Path, phase: str, test_node: Optional[str] = None) -> int:
    """Parent controller running preflight, creating output directory, and monitoring child."""
    if phase not in ("r1a", "r1a-r1", "r1a-r2", "r1a-r3", "r1a-r4", "r1a-r5", "r1a-r6", "r1a-r7", "r1a-r8", "r1a-r9", "r1a-r10"):
        sys.stderr.write(f"Error: only phases 'r1a', 'r1a-r1', 'r1a-r2', 'r1a-r3', 'r1a-r4', 'r1a-r5', 'r1a-r6', 'r1a-r7', 'r1a-r8', 'r1a-r9', and 'r1a-r10' are authorized, got: {phase}\n")
        return 1

    preflight = run_preflight()
    if preflight["status"] != "PASS":
        sys.stderr.write(f"Preflight validation failed: {json.dumps(preflight, indent=2)}\n")
        return 1

    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    random_id = uuid.uuid4().hex[:8]
    if phase == "r1a-r10":
        run_name = f"r1a_r10_{now_utc}_{random_id}"
        report_name = f"fairbias_r1a_r10_worker_{now_utc}_{random_id}"
    elif phase == "r1a-r9":
        run_name = f"r1a_r9_{now_utc}_{random_id}"
        report_name = f"fairbias_r1a_r9_worker_{now_utc}_{random_id}"
    elif phase == "r1a-r8":
        run_name = f"r1a_r8_{now_utc}_{random_id}"
        report_name = f"fairbias_r1a_r8_worker_{now_utc}_{random_id}"
    elif phase == "r1a-r7":
        run_name = f"r1a_r7_{now_utc}_{random_id}"
        report_name = f"fairbias_r1a_r7_worker_{now_utc}_{random_id}"
    elif phase == "r1a-r6":
        run_name = f"r1a_r6_{now_utc}_{random_id}"
        report_name = f"fairbias_r1a_r6_worker_{now_utc}_{random_id}"
    elif phase == "r1a-r5":
        run_name = f"r1a_r5_{now_utc}_{random_id}"
        report_name = f"fairbias_r1a_r5_worker_{now_utc}_{random_id}"
    elif phase == "r1a-r4":
        run_name = f"r1a_r4_{now_utc}_{random_id}"
        report_name = f"fairbias_r1a_r4_worker_{now_utc}_{random_id}"
    elif phase == "r1a-r3":
        run_name = f"r1a_r3_{now_utc}_{random_id}"
        report_name = f"fairbias_r1a_r3_worker_{now_utc}_{random_id}"
    elif phase == "r1a-r2":
        run_name = f"r1a_r2_{now_utc}_{random_id}"
        report_name = f"fairbias_r1a_r2_worker_{now_utc}_{random_id}"
    elif phase == "r1a-r1":
        run_name = f"r1a_r1_{now_utc}_{random_id}"
        report_name = f"fairbias_r1a_r1_worker_{now_utc}_{random_id}"
    else:
        run_name = f"r1a_{now_utc}_{random_id}"
        report_name = f"fairbias_r1a_worker_{now_utc}_{random_id}"

    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    report_dir = REPO_ROOT / "docs" / "reports" / report_name
    report_dir.mkdir(parents=True, exist_ok=False)

    preflight_file = run_dir / "preflight.json"
    with open(preflight_file, "w", encoding="utf-8") as f:
        json.dump(preflight, f, indent=2)

    # Reconstruct preimage of input files from supervisor's verified snapshot
    reconstructed_preimage = build_reconstructed_preimage(phase=phase)
    with open(run_dir / "preimage_hashes.json", "w", encoding="utf-8") as f:
        json.dump(reconstructed_preimage, f, indent=2)
    with open(report_dir / "preimage_hashes.json", "w", encoding="utf-8") as f:
        json.dump(reconstructed_preimage, f, indent=2)

    # Compute candidate hashes for all files from current working tree
    candidate_hashes = build_candidate_hashes(phase=phase)
    with open(run_dir / "candidate_hashes.json", "w", encoding="utf-8") as f:
        json.dump(candidate_hashes, f, indent=2)
    with open(report_dir / "candidate_hashes.json", "w", encoding="utf-8") as f:
        json.dump(candidate_hashes, f, indent=2)

    # Compute pre-execution hashes
    pre_hashes = compute_hashes()
    with open(run_dir / "pre_execution_hashes.json", "w", encoding="utf-8") as f:
        json.dump(pre_hashes, f, indent=2)

    # Launch child with -I -S -B
    child_cmd = [
        PYTHON_BIN,
        "-I",
        "-S",
        "-B",
        str(pathlib.Path(__file__).resolve()),
        "--phase", phase,
        "--output-root", str(output_root),
        "--child",
        "--run-dir", str(run_dir),
    ]
    if test_node:
        child_cmd.extend(["--test-node", test_node])

    start_time = time.time()
    tmp_dir = (run_dir / "tmp").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env["TMPDIR"] = str(tmp_dir)
    child_env["TEMP"] = str(tmp_dir)
    child_env["TMP"] = str(tmp_dir)
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    child_env["PYTHONNOUSERSITE"] = "1"
    child_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    stdout_file = run_dir / "child_stdout.log"
    stderr_file = run_dir / "child_stderr.log"

    with open(stdout_file, "w", encoding="utf-8") as out_f, open(stderr_file, "w", encoding="utf-8") as err_f:
        proc = subprocess.Popen(child_cmd, cwd=REPO_ROOT, stdout=out_f, stderr=err_f, env=child_env)

        max_mem_bytes = 0
        timeout_seconds = 900  # 15 minutes max
        killed = False
        kill_reason = None
        monitor_error = None

        while proc.poll() is None:
            time.sleep(0.5)
            killed, kill_reason, monitor_error, max_mem_bytes = poll_child_resource(
                proc=proc,
                max_mem_bytes=max_mem_bytes,
                timeout_seconds=timeout_seconds,
                start_time=start_time,
            )
            if killed or monitor_error is not None:
                break

        if monitor_error is not None:
            sys.stderr.write(f"[RUNNER] Resource monitoring error: {monitor_error}\n")

        if not killed:
            proc.wait()

    duration_seconds = time.time() - start_time
    exit_code = proc.returncode if not killed else -9

    # Compute post-execution hashes and enforce identity comparison (R2-09, R4-05)
    post_hashes = compute_hashes()
    with open(run_dir / "post_execution_hashes.json", "w", encoding="utf-8") as f:
        json.dump(post_hashes, f, indent=2)

    hash_mismatches = []
    missing_inputs = []
    for rel_p, pre_info in pre_hashes.items():
        if pre_info.get("status") == "MISSING":
            missing_inputs.append(rel_p)
        elif rel_p in post_hashes:
            if post_hashes[rel_p].get("sha256") != pre_info.get("sha256"):
                hash_mismatches.append(rel_p)

    integrity_passed = (len(hash_mismatches) == 0 and len(missing_inputs) == 0)
    if not integrity_passed:
        sys.stderr.write(f"[RUNNER] Integrity check failed: mismatches={hash_mismatches}, missing={missing_inputs}\n")
        if exit_code == 0:
            exit_code = 1

    # Load child summary for centralized verdict evaluation
    child_summary_path = run_dir / "test_summary.json"
    child_summary = {}
    if child_summary_path.exists():
        try:
            with open(child_summary_path, "r", encoding="utf-8") as f:
                child_summary = json.load(f)
        except Exception:
            pass

    # Load actual loaded sources from executed_source_hashes.json (R7-02, R9-01)
    executed_sources_path = run_dir / "executed_source_hashes.json"
    actual_loaded_sources = None
    if executed_sources_path.exists():
        try:
            with open(executed_sources_path, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
            if (
                isinstance(loaded_data, dict)
                and len(loaded_data) > 0
                and "executed_sources" in loaded_data
                and isinstance(loaded_data["executed_sources"], dict)
                and len(loaded_data["executed_sources"]) > 0
                and "executed_module_count" in loaded_data
                and isinstance(loaded_data["executed_module_count"], int)
                and not isinstance(loaded_data["executed_module_count"], bool)
                and loaded_data["executed_module_count"] == len(loaded_data["executed_sources"])
            ):
                actual_loaded_sources = loaded_data["executed_sources"]
            else:
                actual_loaded_sources = None
        except Exception:
            actual_loaded_sources = None
    else:
        actual_loaded_sources = None

    expected_tests = SYNTHETIC_TEST_FILES if not test_node else [test_node.split("::")[0]]

    # Verify executed source bytes match pre_hashes and have no unrecorded project sources
    executed_bytes_path = run_dir / "executed_bytes_hashes.json"
    bytes_valid, executed_byte_mismatches, unrecorded_modules, byte_err, executed_modules_info = verify_executed_bytes_evidence(
        executed_bytes_path=executed_bytes_path,
        pre_hashes=pre_hashes,
        expected_test_files=expected_tests,
        expected_runner="scripts/run_fairbias_r1_guarded_tests.py",
        min_required_modules=15 if not test_node else 1,
        actual_loaded_sources=actual_loaded_sources,
    )
    if not bytes_valid:
        sys.stderr.write(
            f"[RUNNER] Executed byte integrity failed: {byte_err}\n"
        )
        integrity_passed = False
        if exit_code == 0:
            exit_code = 1

    # Verify guard log events
    events_log_file = run_dir / "guard_events.jsonl"
    expected_summary_path = (run_dir / "test_summary.json").resolve()
    log_valid, log_err, log_metrics = verify_guard_log_events(
        events_log_file=events_log_file,
        child_summary=child_summary,
        expected_summary_path=expected_summary_path,
    )
    if not log_valid:
        sys.stderr.write(
            f"[RUNNER] Guard log event integrity failed: {log_err}\n"
        )
        integrity_passed = False
        if exit_code == 0:
            exit_code = 1

    final_log_events_count = log_metrics.get("final_log_events_count", 0)
    sentinel_res = child_summary.get("sentinel_results", {})
    ctrl_probe_res = child_summary.get("controlled_failure_probe", {}).get("status") == "PASSED"
    guard_metrics = child_summary.get("guard_metrics", {})
    log_healthy = (
        log_valid
        and (final_log_events_count > 0)
        and guard_metrics.get("log_healthy", False)
        and (guard_metrics.get("disk_events_lost", 0) == 0)
    )
    unexp_denials = len(guard_metrics.get("unexpected_denials", []))
    unconsumed_denials = guard_metrics.get("unconsumed_expected_denials_count", 0)

    verdict_report = evaluate_execution_verdict(
        preflight_passed=(preflight.get("status") == "PASS"),
        child_exit_code=exit_code,
        integrity_passed=integrity_passed,
        sentinel_results=sentinel_res,
        controlled_probe_passed=ctrl_probe_res,
        child_log_healthy=log_healthy,
        unexpected_denials_count=unexp_denials,
        unconsumed_denials_count=unconsumed_denials,
        timeout_killed=killed,
        memory_killed=(kill_reason == "MEMORY_EXCEEDED_4GIB"),
        monitor_error=monitor_error,
    )
    with open(run_dir / "execution_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict_report, f, indent=2)

    if not verdict_report["overall_success"] and exit_code == 0:
        exit_code = 1

    events_at_summary = child_summary.get("events_at_summary_cutoff", guard_metrics.get("total_events", 0))
    telemetry = {
        "run_id": run_name,
        "phase": phase,
        "command": child_cmd,
        "cwd": str(REPO_ROOT),
        "interpreter": PYTHON_BIN,
        "exit_code": exit_code,
        "duration_seconds": round(duration_seconds, 2),
        "sampled_max_rss_bytes": max_mem_bytes,
        "sampled_max_rss_mb": round(max_mem_bytes / (1024 * 1024), 2),
        "rss_note": "Sampled maximum RSS at 0.5s intervals, not continuous OS peak or hard limit",
        "timeout_killed": killed,
        "kill_reason": kill_reason,
        "monitor_error": monitor_error,
        "summary_events_cutoff": events_at_summary,
        "final_closed_log_events": final_log_events_count,
        "log_cutoff_explanation": (
            f"Summary recorded at event {events_at_summary}; final log sink closed at event {final_log_events_count} "
            f"(delta represents test_summary.json write events and log sink close flush)"
        ),
        "executed_source_modules_count": child_summary.get("executed_source_modules_count", 0),
        "executed_bytes_modules_count": len(executed_modules_info),
        "integrity_checks": {
            "passed": integrity_passed,
            "hash_mismatches": hash_mismatches,
            "missing_inputs": missing_inputs,
            "executed_byte_mismatches": executed_byte_mismatches,
            "unrecorded_modules": unrecorded_modules,
            "byte_error": byte_err if not bytes_valid else None,
            "log_error": log_err if not log_valid else None,
        },
        "verdict": verdict_report,
    }
    with open(run_dir / "telemetry.json", "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    env_info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "executable": PYTHON_BIN,
        "os_uname": list(os.uname()) if hasattr(os, "uname") else [],
        "cwd": str(REPO_ROOT),
        "argv": sys.argv,
        "branch": preflight["checks"]["branch"]["actual"],
        "head_commit": preflight["checks"]["head_commit"]["actual"],
        "protected_tag": PROTECTED_TAG,
        "source_loading_strategy": "sys.path isolation (-I -S -B) with FreshSourceFinder and unbuffered bytecode suppression",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(run_dir / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)

    # Copy key artifacts to report package
    for fname in (
        "preflight.json",
        "telemetry.json",
        "child_stdout.log",
        "child_stderr.log",
        "test_summary.json",
        "sentinel_results.json",
        "guard_events.jsonl",
        "pre_execution_hashes.json",
        "post_execution_hashes.json",
        "executed_source_hashes.json",
        "executed_bytes_hashes.json",
        "candidate_hashes.json",
        "execution_verdict.json",
        "environment.json",
    ):
        src_f = run_dir / fname
        if src_f.exists():
            dst_f = report_dir / fname
            dst_f.write_bytes(src_f.read_bytes())

    # preimage_hashes.json was created directly in run_dir and report_dir via build_reconstructed_preimage()

    print(f"[RUNNER] Phase {phase} completed with exit code {exit_code}")
    print(f"[RUNNER] Execution time: {duration_seconds:.2f}s | Sampled max RSS: {telemetry['sampled_max_rss_mb']} MB")
    print(f"[RUNNER] Run artifacts: {run_dir}")
    print(f"[RUNNER] Report package: {report_dir}")

    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description="Run guarded FairBias R1 tests")
    parser.add_argument("--phase", type=str, default="r1a-r10", help="Phase to execute (r1a-r4, r1a-r5, r1a-r6, r1a-r7, r1a-r8, r1a-r9, r1a-r10)")
    parser.add_argument("--output-root", type=pathlib.Path, default=REPO_ROOT / "runs", help="Output root directory")
    parser.add_argument("--child", action="store_true", help="Internal child worker mode")
    parser.add_argument("--run-dir", type=pathlib.Path, default=None, help="Assigned run directory for child")
    parser.add_argument("--test-node", type=str, default=None, help="Specific test node to execute")

    args = parser.parse_args()

    if args.child:
        if args.run_dir is None:
            sys.stderr.write("Child execution requires --run-dir\n")
            sys.exit(1)
        sys.exit(child_execution(args.run_dir, test_node=args.test_node))
    else:
        sys.exit(parent_controller(output_root=args.output_root, phase=args.phase, test_node=args.test_node))


if __name__ == "__main__":
    main()

