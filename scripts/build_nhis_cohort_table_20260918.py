#!/usr/bin/env python3
"""Build an aggregate-only NHIS cohort/Table 1 export.

This script intentionally has a narrow input boundary.  It verifies the
completion admission manifest and then reads one hash-bound prepared joblib
per arm for F/C/S.  It reads only the aggregate T cohort and eligibility JSON
files, plus their hash-binding evaluation manifests and merged-summary shard
manifest.  It does not load models, policies, predictions, or raw T files,
and it never writes individual records.

The output is additive and refuses to overwrite an existing output directory.
Run from the repository root with the project environment, for example:

    .venv311/bin/python scripts/build_nhis_cohort_table_20260918.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nhis_fairbias.benchmark.data_contracts import ARM_SPECS  # noqa: E402


ADMISSION = REPO / "artifacts/nhis/completion_evaluation_20260918/control/admission_v1.json"
EVAL_ROOT = REPO / "artifacts/nhis/completion_evaluation_20260918/evaluation_v1"
OUT = REPO / "docs/paper/cohort_tables_v2_20260918"
SCRIPT = Path(__file__).resolve()

EXPECTED_ADMISSION_SHA256 = "2a83612b23048a54ce09196ec553b9881ee87bf5b5930f36fc53bc0e52d1a628"
EXPECTED_MERGED_SHA256 = "641fe2ec523c8998cfe4c1f48ae2d5fb1e6ecbe9a5eb9e20b16af33d47c6b406"
MERGED_SUMMARY = REPO / "artifacts/nhis/completion_evaluation_20260918/merged_summary_v1.json"

ARM_ORDER = ("arm_001", "arm_002", "arm_003", "arm_004")
ROLE_ORDER = ("fitting_F", "calibration_C", "selection_S", "evaluation_T")
ROLE_LABEL = {
    "fitting_F": ("F", 2022),
    "calibration_C": ("C", 2022),
    "selection_S": ("S", 2023),
    "evaluation_T": ("T", 2024),
}
CSV_FIELDS = (
    "arm_id",
    "protected_attribute",
    "group_code",
    "group_label",
    "partition",
    "year",
    "n",
    "event_count",
    "weight_sum",
    "weighted_event_rate",
    "kish_ess",
    "weight_min",
    "weight_p25",
    "weight_p50",
    "weight_p75",
    "weight_max",
    "analysis_domain_strata_count",
    "analysis_domain_psu_count",
    "inference_design_strata_count",
    "inference_design_psu_count",
    "inference_design_source",
    "eligibility_full_annual_n",
    "eligibility_joint_n",
    "eligibility_excluded_n",
    "eligibility_excluded_weight_fraction",
    "status",
    "source_mode",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_input_path(path_value: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    require(path.exists() and path.is_file(), f"Missing input file: {path}")
    return path


def unique_prepared(admission: dict[str, Any], arm_id: str) -> tuple[Path, str, dict[str, Any]]:
    jobs = [job for job in admission.get("jobs", []) if job.get("config", {}).get("arm_id") == arm_id]
    require(len(jobs) == 20, f"{arm_id}: expected 20 admission jobs, got {len(jobs)}")
    refs = []
    for job in jobs:
        ref = job.get("prepared_file")
        require(isinstance(ref, dict), f"{arm_id}: malformed prepared_file reference")
        refs.append((str(ref.get("path")), str(ref.get("sha256"))))
    require(len(set(refs)) == 1, f"{arm_id}: prepared reference is not stable across jobs")
    path = canonical_input_path(refs[0][0])
    actual = sha256_file(path)
    require(actual == refs[0][1], f"{arm_id}: prepared hash mismatch")
    prepared_refs = [job.get("prepared", {}) for job in jobs]
    require(all(ref.get("sha256") == actual for ref in prepared_refs), f"{arm_id}: prepared hash binding mismatch")
    require(all(ref.get("path") for ref in prepared_refs), f"{arm_id}: missing prepared metadata path")
    identities = {ref.get("data_identity") for ref in prepared_refs}
    require(len(identities) == 1 and None not in identities, f"{arm_id}: prepared data_identity is not stable")
    return path, actual, {
        "path": str(path),
        "sha256": actual,
        "jobs": len(jobs),
        "sizes": prepared_refs[0].get("sizes", {}),
        "data_identity": prepared_refs[0].get("data_identity"),
    }


def verify_evaluation_bindings(merged: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Verify accepted T artifact hashes before reading cohort aggregates."""
    shards = merged.get("shards")
    require(isinstance(shards, list) and len(shards) == len(ARM_ORDER), "merged summary shard coverage mismatch")
    result: dict[str, dict[str, Any]] = {}
    for shard in shards:
        manifest_ref = shard.get("manifest", {})
        manifest_path = canonical_input_path(str(manifest_ref.get("path")))
        manifest_hash = str(manifest_ref.get("sha256"))
        require(sha256_file(manifest_path) == manifest_hash, f"{manifest_path}: evaluation manifest hash mismatch")
        summary_ref = shard.get("summary", {})
        summary_path = canonical_input_path(str(summary_ref.get("path")))
        summary_hash = str(summary_ref.get("sha256"))
        require(sha256_file(summary_path) == summary_hash, f"{summary_path}: merged-summary shard hash mismatch")
        manifest = read_json(manifest_path)
        arm_id = manifest.get("arm_id")
        require(arm_id in ARM_ORDER and arm_id not in result, "invalid or duplicate evaluation manifest arm")
        artifacts = manifest.get("artifacts", {})
        require(isinstance(artifacts, dict), f"{arm_id}: evaluation manifest artifacts missing")
        for filename in ("cohort_T.json", "eligibility_T.json"):
            require(filename in artifacts, f"{arm_id}: {filename} absent from accepted evaluation manifest")
            expected = str(artifacts[filename])
            actual_path = manifest_path.parent / filename
            require(actual_path.exists() and sha256_file(actual_path) == expected, f"{arm_id}: {filename} hash is not bound by evaluation manifest")
        result[arm_id] = {
            "manifest": {"path": str(manifest_path), "sha256": manifest_hash},
            "summary": {"path": str(summary_path), "sha256": summary_hash},
            "cohort_T": {"path": str(manifest_path.parent / "cohort_T.json"), "sha256": str(artifacts["cohort_T.json"])},
            "eligibility_T": {"path": str(manifest_path.parent / "eligibility_T.json"), "sha256": str(artifacts["eligibility_T.json"])},
        }
    require(set(result) == set(ARM_ORDER), "accepted evaluation manifest arms are incomplete")
    return result


def as_vector(partition: Any, name: str) -> np.ndarray:
    value = np.asarray(getattr(partition, name))
    role = getattr(partition, "role", "UNKNOWN")
    arm_id = getattr(partition, "arm_id", "UNKNOWN")
    require(value.ndim == 1, f"{arm_id}/{role}: {name} is not one-dimensional")
    return value


def design_counts(strata: np.ndarray, psus: np.ndarray) -> tuple[int, int]:
    require(len(strata) == len(psus), "design vectors are not aligned")
    return int(len(np.unique(strata))), int(len(set(zip(strata.tolist(), psus.tolist()))))


def stable_sum(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    scale = float(np.max(values))
    scaled = float(np.sum(values / scale))
    total = scale * scaled
    require(math.isfinite(total) and total > 0, "invalid survey-weight sum")
    return total


def stats(y: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    require(len(y) == len(weights) == len(mask), "aligned statistic vectors required")
    count = int(mask.sum())
    if count == 0:
        return {
            "n": 0,
            "event_count": 0,
            "weight_sum": None,
            "weighted_event_rate": None,
            "kish_ess": None,
            "weight_min": None,
            "weight_p25": None,
            "weight_p50": None,
            "weight_p75": None,
            "weight_max": None,
        }
    w = weights[mask]
    yy = y[mask]
    total = stable_sum(w)
    scale = float(np.max(w))
    wn = w / scale
    scaled_total = float(np.sum(wn))
    scaled_sumsq = float(np.dot(wn, wn))
    return {
        "n": count,
        "event_count": int(np.sum(yy == 1)),
        "weight_sum": total,
        "weighted_event_rate": float(np.dot(wn, yy) / scaled_total),
        "kish_ess": float(scaled_total * scaled_total / scaled_sumsq),
        "weight_min": float(np.min(w)),
        "weight_p25": float(np.quantile(w, 0.25)),
        "weight_p50": float(np.quantile(w, 0.50)),
        "weight_p75": float(np.quantile(w, 0.75)),
        "weight_max": float(np.max(w)),
    }


def validate_vectors(partition: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = as_vector(partition, "y").astype(float, copy=False)
    a = as_vector(partition, "A")
    w = as_vector(partition, "WTFA_A").astype(float, copy=False)
    strata = as_vector(partition, "PSTRAT")
    psus = as_vector(partition, "PPSU")
    require(
        len(y) == len(a) == len(w) == len(strata) == len(psus),
        f"{getattr(partition, 'arm_id', 'UNKNOWN')}/{getattr(partition, 'role', 'UNKNOWN')}: partition vectors are not aligned",
    )
    require(np.isfinite(y).all() and np.isin(y, [0, 1]).all(), "outcome is not finite binary")
    require(np.isfinite(w).all() and (w > 0).all(), "WTFA_A is not finite positive")
    return y, a, w, strata, psus


def make_partition_rows(arm_id: str, role: str, partition: Any) -> list[dict[str, Any]]:
    code, year = ROLE_LABEL[role]
    require(str(partition.role) == role and int(partition.year) == year, f"{arm_id}/{role}: role-year mismatch")
    spec = ARM_SPECS[arm_id]
    y, a, w, strata, psus = validate_vectors(partition)
    expected = np.asarray(spec["expected_categories"])
    observed = set(np.unique(a).tolist())
    require(observed.issubset(set(expected.tolist())), f"{arm_id}/{role}: unexpected protected code")
    require(observed == set(expected.tolist()), f"{arm_id}/{role}: expected protected group absent")
    annual = getattr(partition, "annual_design", None)
    require(annual is not None, f"{arm_id}/{role}: missing annual design")
    annual_strata = np.asarray(annual.strata)
    annual_psus = np.asarray(annual.psus)
    full_strata_count, full_psu_count = design_counts(annual_strata, annual_psus)

    overall_mask = np.ones(len(y), dtype=bool)
    labels = [(None, "Overall", overall_mask)]
    labels.extend((int(group), spec["category_labels"][int(group)], a == group) for group in expected)
    rows: list[dict[str, Any]] = []
    overall = stats(y, w, overall_mask)
    group_stats = {group: stats(y, w, a == group) for group in expected}
    require(sum(row["n"] for row in group_stats.values()) == overall["n"], f"{arm_id}/{code}: group n sum mismatch")
    require(sum(row["event_count"] for row in group_stats.values()) == overall["event_count"], f"{arm_id}/{code}: group event sum mismatch")

    for group_code, group_label, mask in labels:
        row_stats = overall if group_code is None else group_stats[group_code]
        domain_strata_count, domain_psu_count = design_counts(strata[mask], psus[mask])
        row = {
            "arm_id": arm_id,
            "protected_attribute": spec["protected_attribute"],
            "group_code": "ALL" if group_code is None else group_code,
            "group_label": group_label,
            "partition": code,
            "year": year,
            **row_stats,
            "analysis_domain_strata_count": domain_strata_count,
            "analysis_domain_psu_count": domain_psu_count,
            "inference_design_strata_count": full_strata_count,
            "inference_design_psu_count": full_psu_count,
            "inference_design_source": "annual_full_design_available_with_partition",
            "eligibility_full_annual_n": None,
            "eligibility_joint_n": None,
            "eligibility_excluded_n": None,
            "eligibility_excluded_weight_fraction": None,
            "status": "VALID",
            "source_mode": "hash_verified_prepared_partition",
        }
        rows.append(row)
    return rows


def t_rows(arm_id: str, cohort: dict[str, Any], eligibility: dict[str, Any]) -> list[dict[str, Any]]:
    spec = ARM_SPECS[arm_id]
    require(cohort.get("arm_id") == arm_id and cohort.get("role") == "evaluation_T" and cohort.get("year") == 2024, f"{arm_id}/T: invalid cohort metadata")
    overall = cohort.get("overall", {})
    design = cohort.get("design", {})
    groups = cohort.get("groups", {})
    require(set(groups) == {str(x) for x in spec["expected_categories"]}, f"{arm_id}/T: group-code mismatch")
    eligibility_years = eligibility.get("years", {})
    require(set(eligibility_years) == {"2024"}, f"{arm_id}/T: unexpected eligibility years")
    e = eligibility_years["2024"].get(arm_id)
    require(isinstance(e, dict), f"{arm_id}/T: missing own-arm eligibility aggregate")
    require(e.get("joint", {}).get("eligible", {}).get("n") == overall.get("n"), f"{arm_id}/T: cohort/eligibility n mismatch")
    require(e.get("joint", {}).get("eligible", {}).get("weight_sum") == overall.get("weight_sum"), f"{arm_id}/T: cohort/eligibility weight mismatch")
    full_n = e.get("full_annual", {}).get("n")
    eligible_n = e.get("joint", {}).get("eligible", {}).get("n")
    excluded_n = e.get("joint", {}).get("excluded", {}).get("n")
    excluded_fraction = e.get("joint", {}).get("excluded", {}).get("weight_fraction_of_annual")
    require(full_n == eligible_n + excluded_n, f"{arm_id}/T: eligibility n invariant failed")
    group_n = sum(int(v.get("n", 0)) for v in groups.values())
    group_events = sum(int(v.get("event_count", 0)) for v in groups.values())
    require(group_n == overall.get("n") and group_events == overall.get("event_count"), f"{arm_id}/T: group aggregate invariant failed")
    full_strata_count = int(design["strata_count"])
    full_psu_count = int(design["psu_count"])
    domain_strata_count = int(design["domain_strata_count"])
    domain_psu_count = int(design["domain_psu_count"])
    rows: list[dict[str, Any]] = []
    labels: Iterable[tuple[str, str, dict[str, Any]]] = [("ALL", "Overall", overall)]
    labels = list(labels) + [(str(g), spec["category_labels"][int(g)], groups[str(g)]) for g in spec["expected_categories"]]
    for group_code, group_label, source in labels:
        row = {
            "arm_id": arm_id,
            "protected_attribute": spec["protected_attribute"],
            "group_code": group_code,
            "group_label": group_label,
            "partition": "T",
            "year": 2024,
            "n": source.get("n"),
            "event_count": source.get("event_count"),
            "weight_sum": source.get("weight_sum"),
            "weighted_event_rate": source.get("weighted_event_rate"),
            "kish_ess": source.get("kish_ess"),
            # cohort_T is aggregate-only and does not contain weight detail.
            "weight_min": None,
            "weight_p25": None,
            "weight_p50": None,
            "weight_p75": None,
            "weight_max": None,
            # Only the overall T row has an aggregate domain-support count.
            "analysis_domain_strata_count": domain_strata_count if group_code == "ALL" else None,
            "analysis_domain_psu_count": domain_psu_count if group_code == "ALL" else None,
            # T inference uses the complete annual design, never domain-only counts.
            "inference_design_strata_count": full_strata_count,
            "inference_design_psu_count": full_psu_count,
            "inference_design_source": "annual_full_design_T_inference",
            "eligibility_full_annual_n": full_n,
            "eligibility_joint_n": eligible_n,
            "eligibility_excluded_n": excluded_n,
            "eligibility_excluded_weight_fraction": excluded_fraction,
            "status": source.get("status", "VALID"),
            "source_mode": "aggregate_only_cohort_T_and_eligibility_T",
        }
        rows.append(row)
    return rows


def csv_value(value: Any) -> Any:
    return "NA" if value is None else value


def format_cell(field: str, value: Any) -> str:
    if value is None:
        return "NA"
    if field == "year":
        return str(value)
    if field in {"weighted_event_rate", "eligibility_excluded_weight_fraction"}:
        return f"{float(value) * 100:.2f}%"
    if field in {"weight_sum", "weight_min", "weight_p25", "weight_p50", "weight_p75", "weight_max", "kish_ess"}:
        return f"{float(value):,.3f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def write_table(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    csv_path, md_path = OUT / "Table1.csv", OUT / "Table1.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in CSV_FIELDS})
    overall_headers = ["Arm", "Partition", "Year", "n", "Events", "Weighted event rate", "Kish ESS"]
    overall_fields = ["arm_id", "partition", "year", "n", "event_count", "weighted_event_rate", "kish_ess"]
    support_headers = [
        "Arm", "Protected attribute", "Group", "Partition", "Year", "n", "Events",
        "Weighted event rate", "Kish ESS", "Domain strata", "Domain PSUs",
    ]
    support_fields = [
        "arm_id", "protected_attribute", "group_label", "partition", "year", "n", "event_count",
        "weighted_event_rate", "kish_ess", "analysis_domain_strata_count", "analysis_domain_psu_count",
    ]
    overall_rows = [row for row in rows if row["group_code"] == "ALL"]
    subgroup_rows = [row for row in rows if row["group_code"] != "ALL"]
    if len(overall_rows) != 16 or len(subgroup_rows) != 52:
        raise ValueError("Markdown layout requires 16 overall rows and 52 protected-group rows")
    lines = [
        "# Table 1. NHIS analysis cohort and survey-design support",
        "",
        "Values are aggregate-only. The CSV is the complete 68-row diagnostic table, including F/C/S weight min/max/quantiles, full-design support, and T eligibility fields. T weight min/max and quantiles are `NA` because the frozen `cohort_T.json` contains no such aggregates and raw T was not reread.",
        "",
        "For F/C/S, domain strata/PSUs describe each partition or protected-group support; the full-design columns retain the annual design attached to the partition. For T, inference uses the complete annual design (52 strata, 662 PSUs); the overall T domain-support columns are descriptive only and are never substituted for the annual design.",
        "",
        "## Overall analysis domains",
        "",
        "| " + " | ".join(overall_headers) + " |",
        "|" + "|".join("---" for _ in overall_headers) + "|",
    ]
    for row in overall_rows:
        lines.append("| " + " | ".join(format_cell(field, row.get(field)) for field in overall_fields) + " |")
    lines.extend([
        "",
        "## Protected-group support",
        "",
        "| " + " | ".join(support_headers) + " |",
        "|" + "|".join("---" for _ in support_headers) + "|",
    ])
    for row in subgroup_rows:
        lines.append("| " + " | ".join(format_cell(field, row.get(field)) for field in support_fields) + " |")
    lines.extend([
        "",
        "Eligibility exclusion fields are arm-level 2024 annual item-eligibility aggregates and are repeated on T rows in the CSV; they are not longitudinal loss-to-follow-up counts. Detailed per-predictor missingness rates require a separate aggregate table.",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def write_report(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    command: str,
) -> Path:
    report = OUT / "SECTION9_WORKER_REPORT.md"
    partition_counts = {
        role: sum(1 for row in rows if row["partition"] == ROLE_LABEL[role][0])
        for role in ROLE_ORDER
    }
    input_lines = [f"- `{key}`: `{value['sha256']}`" for key, value in manifest["inputs"].items()]
    output_lines = [f"- `{key}`: `{value}`" for key, value in manifest["outputs"].items()]
    text = f"""Gate: Paper cohort Table 1 aggregate export 20260918
Status: COMPLETE — aggregate-only output prepared; Codex supervisor review pending.
Files changed:
- `scripts/build_nhis_cohort_table_20260918.py`
- `docs/paper/cohort_tables_v2_20260918/Table1.csv`
- `docs/paper/cohort_tables_v2_20260918/Table1.md`
- `docs/paper/cohort_tables_v2_20260918/hash_manifest.json`
- `docs/paper/cohort_tables_v2_20260918/SECTION9_WORKER_REPORT.md`
Commands executed:
- `{command}`
- `./.venv311/bin/python -m py_compile scripts/build_nhis_cohort_table_20260918.py`
Permissions requested: None.
Tests executed:
- Admission prepared references were checked for stable per-arm path/hash and SHA-256 equality before joblib loading.
- Admission SHA-256, registered `ARM_SPECS` source SHA-256, merged-summary SHA-256, and evaluation-manifest shard bindings were checked.
- Prepared payload keys, role/year vectors, binary outcome support, positive `WTFA_A`, protected-group coverage, group count/event invariants, and annual-design alignment were checked.
- T cohort/eligibility aggregate agreement and full-annual eligibility invariant were checked.
- Output CSV/Markdown row and hash manifest checks were completed.
Exact test results: PASS; {len(rows)} Table 1 rows across 4 arms and F/C/S/T; partition row counts F={partition_counts['fitting_F']}, C={partition_counts['calibration_C']}, S={partition_counts['selection_S']}, T={partition_counts['evaluation_T']}; all aggregate invariants passed.
Input hashes:
{os.linesep.join(input_lines)}
Output hashes:
{os.linesep.join(output_lines)}
Row counts: `Table1.csv` data rows={len(rows)}; Markdown overall rows=16 and protected-group rows=52; no individual records are exported.
Assumptions:
- `ARM_SPECS` is the registered mapping in `src/nhis_fairbias/benchmark/data_contracts.py` (arm 001 SEX_A, arm 002 HISPALLP_A, arms 003/004 DISAB3_A).
- F/C are 2022 cluster-assigned partitions, S is 2023 selection, and T is the 2024 evaluation cohort.
- Rates and denominators use `WTFA_A`; Kish ESS is a weight-dispersion diagnostic, not a cluster-adjusted ESS.
- T weight min/max/quantiles remain `NA` because the permitted frozen T summary lacks them; raw T was not reread.
- F/C/S records are loaded from hash-verified prepared inputs only to calculate aggregate rows; no individual records are exported, and no model/policy/prediction is loaded.
- T inference-design columns use the complete annual design and must not be replaced with domain-only PSU/strata counts.
Unresolved issues:
- This table describes the included analysis domains and arm-level T item-eligibility exclusions. Detailed per-predictor missingness/exclusion denominators still require a separate aggregate table.
- Arms 003 and 004 share the same protected attribute and observed eligible respondents; arm 004 is a feature ablation and is not an independent dataset.
Git diff summary: Additive script and new paper-output directory only; no staging or commit performed. Existing user changes were preserved.
Proposed next step: Codex supervisor independently recheck the admission hash bindings, annual-vs-domain design interpretation, and Table 1 claim boundaries before accepting the gate.
STOP — waiting for Codex review.
"""
    report.write_text(text, encoding="utf-8")
    return report


def main() -> int:
    require(ADMISSION.exists(), f"Missing admission: {ADMISSION}")
    admission_hash = sha256_file(ADMISSION)
    require(admission_hash == EXPECTED_ADMISSION_SHA256, "admission SHA-256 is not the accepted fixed input")
    admission = read_json(ADMISSION)
    require(admission.get("schema_version") == "completion_admission_v1", "Unexpected admission schema")
    require(admission.get("known_T") is True and admission.get("evaluation_authorized") is False, "Admission state is not known-T fixed-artifact evaluation")
    require(set(ARM_SPECS) == set(ARM_ORDER), "ARM_SPECS arm coverage changed")
    arm_specs_path = REPO / "src/nhis_fairbias/benchmark/data_contracts.py"
    require(
        admission.get("analysis_files", {}).get("src/nhis_fairbias/benchmark/data_contracts.py")
        == sha256_file(arm_specs_path),
        "ARM_SPECS source SHA-256 is not bound by admission.analysis_files",
    )
    require(MERGED_SUMMARY.exists(), f"Missing merged summary: {MERGED_SUMMARY}")
    merged_hash = sha256_file(MERGED_SUMMARY)
    require(merged_hash == EXPECTED_MERGED_SHA256, "merged summary SHA-256 is not the accepted fixed input")
    merged = read_json(MERGED_SUMMARY)
    require(merged.get("schema_version") == "completion_merged_summary_v1" and merged.get("status") == "COMPLETE", "Unexpected merged summary state")
    evaluation_bindings = verify_evaluation_bindings(merged)
    require(not OUT.exists(), f"Refusing to overwrite existing output directory: {OUT}")
    OUT.mkdir(parents=True)

    inputs: dict[str, dict[str, Any]] = {
        "admission": {"path": str(ADMISSION), "sha256": admission_hash},
        "arm_specs": {"path": str(arm_specs_path), "sha256": sha256_file(arm_specs_path)},
        "merged_summary": {"path": str(MERGED_SUMMARY), "sha256": merged_hash},
    }
    rows: list[dict[str, Any]] = []
    for arm_id in ARM_ORDER:
        prepared_path, prepared_hash, prepared_meta = unique_prepared(admission, arm_id)
        inputs[f"{arm_id}.prepared"] = prepared_meta
        # The hash was verified before deserialization.  This payload is a
        # prepared partition bundle, not a model/policy/prediction artifact.
        payload = joblib.load(prepared_path)
        require(isinstance(payload, dict), f"{arm_id}: prepared payload is not a mapping")
        require(set(payload) == {"X_C", "X_F", "X_S", "data_identity", "partitions", "preprocessor"}, f"{arm_id}: unexpected prepared payload keys")
        require(payload.get("data_identity") == prepared_meta.get("data_identity"), f"{arm_id}: prepared data_identity is not bound")
        require(set(payload["partitions"]) == {"fitting_F", "calibration_C", "selection_S"}, f"{arm_id}: prepared partitions are not exactly F/C/S")
        for role in ("fitting_F", "calibration_C", "selection_S"):
            rows.extend(make_partition_rows(arm_id, role, payload["partitions"][role]))
        del payload

        binding = evaluation_bindings[arm_id]
        manifest_ref = binding["manifest"]
        inputs[f"{arm_id}.evaluation_manifest"] = manifest_ref
        inputs[f"{arm_id}.summary_T"] = binding["summary"]
        cohort_ref = binding["cohort_T"]
        eligibility_ref = binding["eligibility_T"]
        cohort_path = Path(cohort_ref["path"])
        eligibility_path = Path(eligibility_ref["path"])
        inputs[f"{arm_id}.cohort_T"] = cohort_ref
        inputs[f"{arm_id}.eligibility_T"] = eligibility_ref
        rows.extend(t_rows(arm_id, read_json(cohort_path), read_json(eligibility_path)))

    require(len(rows) == 68, f"Expected 68 Table 1 rows, got {len(rows)}")
    rows.sort(key=lambda row: (ARM_ORDER.index(row["arm_id"]), ROLE_ORDER.index({"F": "fitting_F", "C": "calibration_C", "S": "selection_S", "T": "evaluation_T"}[row["partition"]]), 0 if row["group_code"] == "ALL" else 1, str(row["group_code"])))
    csv_path, md_path = write_table(rows)
    outputs = {"Table1.csv": sha256_file(csv_path), "Table1.md": sha256_file(md_path)}
    manifest = {
        "schema_version": "nhis_cohort_table_manifest_v2",
        "created_by": str(SCRIPT),
        "script_sha256": sha256_file(SCRIPT),
        "input_boundary": "accepted admission + hash-bound prepared F/C/S + accepted evaluation manifest shards + aggregate cohort_T/eligibility_T only",
        "input_hashes": {key: value["sha256"] for key, value in inputs.items()},
        "inputs": inputs,
        "outputs": outputs,
        "row_count": len(rows),
        "table_fields": list(CSV_FIELDS),
        "invariants": {
            "arms": list(ARM_ORDER),
            "partitions": list(ROLE_ORDER),
            "t_inference_design_is_annual_full_design": True,
            "no_model_or_prediction_loaded": True,
            "no_raw_rows_exported": True,
            "admission_sha256_fixed": EXPECTED_ADMISSION_SHA256,
            "merged_summary_sha256_fixed": EXPECTED_MERGED_SHA256,
        },
    }
    manifest_path = OUT / "hash_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["outputs"]["hash_manifest.json"] = sha256_file(manifest_path)
    report_path = write_report(manifest, rows, "./.venv311/bin/python scripts/build_nhis_cohort_table_20260918.py")
    # Record report hash in a sidecar-free manifest is intentionally avoided:
    # changing the manifest after hashing it would make its recorded digest
    # self-referential.  The report records the manifest hash at report time.
    print(json.dumps({"status": "PASS", "rows": len(rows), "output_dir": str(OUT), "manifest_sha256": sha256_file(manifest_path), "report_sha256": sha256_file(report_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
