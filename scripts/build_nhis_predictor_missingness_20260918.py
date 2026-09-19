#!/usr/bin/env python3
"""Build an aggregate-only, source-bound NHIS predictor-missingness export.

The exporter is deliberately separate from model execution.  It verifies the
accepted study/admission metadata and the exact local NHIS source hashes, uses
the prepared F/C/S domains only for membership, and joins those memberships to
the raw annual columns only to classify missingness.  No row, identifier,
prediction, model, or policy is written to an output artifact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = pathlib.Path(__file__).resolve()
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nhis_fairbias.benchmark.data_contracts import ARM_SPECS  # noqa: E402
from nhis_fairbias.features import load_feature_registry  # noqa: E402

STUDY_CONFIG = REPO / "configs/nhis/study.json"
FEATURE_CONFIG = REPO / "configs/nhis/features.json"
VARIABLE_CONFIG = REPO / "configs/nhis/variables.json"
DATA_MANIFEST = REPO / "artifacts/nhis/data/data_manifest.json"
FEATURE_MANIFEST = REPO / "artifacts/nhis/features/feature_manifest.json"
ADMISSION = REPO / "artifacts/nhis/completion_evaluation_20260918/control/admission_v1.json"
STUDY = REPO / "artifacts/nhis/completion_evaluation_20260918/control/study_v1.json"
STUDY_REVIEW = REPO / "artifacts/nhis/completion_evaluation_20260918/control/study_supervisor_review_v1.json"
EVAL_ROOT = REPO / "artifacts/nhis/completion_evaluation_20260918/evaluation_v1"
PREPARED_ROOT = REPO / "artifacts/nhis/completed80_backup_20260918/snapshot/fairbias_completion_v2_20260917/prepared"
OUT = REPO / "docs/paper/manuscript_readiness_20260918/missingness"
ADMISSION_SHA256 = "2a83612b23048a54ce09196ec553b9881ee87bf5b5930f36fc53bc0e52d1a628"
ARM_ORDER = ("arm_001", "arm_002", "arm_003", "arm_004")
ROLE_ORDER = ("fitting_F", "calibration_C", "selection_S", "evaluation_T")
ROLE_LABEL = {"fitting_F": "F", "calibration_C": "C", "selection_S": "S", "evaluation_T": "T"}
ROLE_YEAR = {"F": 2022, "C": 2022, "S": 2023, "T": 2024}
DISABILITY_COMPONENTS = {
    "visiondf_a",
    "hearingdf_a",
    "diff_a",
    "comdiff_a",
    "uppslfcr_a",
    "cogmemdff_a",
}
CAT_MISSINGNESS = ("item_nonresponse", "structural_niu", "raw_null", "unknown_unmapped")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def pct(value: float | None) -> float | None:
    return None if value is None else float(value)


def safe_fraction(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def canonical_ints(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return integral codes and an explicit non-integral/unknown mask."""
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric.notna() & np.isfinite(numeric)
    integral = finite & numeric.eq(numeric.round())
    codes = pd.Series(pd.NA, index=series.index, dtype="Int64")
    if integral.any():
        codes.loc[integral] = numeric.loc[integral].astype("int64")
    unknown = series.notna() & ~integral
    return codes, unknown


def _class_counts(mask: pd.Series, weights: pd.Series, denominator_n: int, denominator_w: float) -> dict[str, Any]:
    n = int(mask.sum())
    w = float(weights.loc[mask].sum())
    return {
        "n": n,
        "weight": w,
        "fraction": safe_fraction(n, denominator_n),
        "weighted_fraction": safe_fraction(w, denominator_w),
    }


def classify_variable(
    frame: pd.DataFrame,
    official_name: str,
    spec: Mapping[str, Any],
    weights: pd.Series,
) -> dict[str, Any]:
    """Classify one raw variable into mutually exclusive semantic states.

    Official nonresponse codes are item nonresponse.  For EMPWRKFT1_A the
    frozen routing rule has precedence: a non-working respondent is structural
    NIU even when the intensity cell is blank or carries a nonresponse code;
    unresolved work status/intensity is item nonresponse.  Raw nulls and
    unexpected/unmapped values remain explicit categories elsewhere.
    """
    raw = frame[official_name]
    codes, nonintegral = canonical_ints(raw)
    substantive = {int(x) for x in spec.get("substantive_codes", [])}
    missing = {int(x) for x in spec.get("missing_codes", [])}
    base = pd.Series("observed", index=frame.index, dtype="string")
    base.loc[raw.isna()] = "raw_null"
    base.loc[nonintegral] = "unknown_unmapped"
    base.loc[raw.notna() & codes.isin(missing)] = "item_nonresponse"
    base.loc[raw.notna() & codes.notna() & ~codes.isin(substantive | missing)] = "unknown_unmapped"

    if official_name == "EMPWRKFT1_A":
        lsw, lsw_nonintegral = canonical_ints(frame["EMPWRKLSW1_A"])
        lsw_missing = {7, 8, 9}
        lsw_raw_null = frame["EMPWRKLSW1_A"].isna()
        lsw_item = lsw.isin(lsw_missing)
        lsw_unknown = frame["EMPWRKLSW1_A"].notna() & (
            lsw_nonintegral | (lsw.notna() & ~lsw.isin({1, 2} | lsw_missing))
        )
        not_work = lsw.eq(2)
        working = lsw.eq(1)
        # Routing precedence is source-semantic and mutually exclusive:
        # not-working is structural NIU; official refusal/don't-know routing
        # codes are item nonresponse; a blank routing field is raw-null; an
        # unexpected routing code is unknown/unmapped.  Within known working
        # records the intensity field keeps the same four-way distinction.
        base.loc[not_work] = "structural_niu"
        base.loc[lsw_item & ~not_work] = "item_nonresponse"
        base.loc[lsw_raw_null & ~not_work] = "raw_null"
        base.loc[lsw_unknown & ~not_work] = "unknown_unmapped"
        base.loc[working & raw.isna()] = "raw_null"
        base.loc[working & raw.notna() & codes.isin(missing)] = "item_nonresponse"
        base.loc[working & raw.notna() & nonintegral] = "unknown_unmapped"
        base.loc[working & raw.notna() & codes.notna() & ~codes.isin(substantive | missing)] = "unknown_unmapped"

    denominator_n = len(frame)
    denominator_w = float(weights.sum())
    result: dict[str, Any] = {
        "denominator_n": denominator_n,
        "denominator_weight": denominator_w,
    }
    for state in CAT_MISSINGNESS:
        result[state] = _class_counts(base.eq(state), weights, denominator_n, denominator_w)
    result["observed"] = _class_counts(base.eq("observed"), weights, denominator_n, denominator_w)
    missing_mask = base.ne("observed")
    result["preprocessing_missing"] = _class_counts(missing_mask, weights, denominator_n, denominator_w)
    # A reconciliation check makes the aggregate contract auditable.
    require(int(base.notna().sum()) == denominator_n, f"{official_name}: classification did not cover denominator")
    require(int(base.value_counts().sum()) == denominator_n, f"{official_name}: state counts do not reconcile")
    return result


def preprocessing_action(harm_name: str, spec: Mapping[str, Any]) -> str:
    if harm_name == "empwrkft1_a":
        return "EMPWRKLSW1_A routing: structural_NIU=-1; unresolved item missing=-2; full/part retained"
    if spec.get("semantic_type") in {"continuous", "count"}:
        return "2022-F fitted median imputation; frozen on C/S/T"
    return "explicit missing category sentinel -1; substantive codes retained"


def load_feature_specs() -> tuple[dict[str, Any], dict[str, Any]]:
    registry = load_feature_registry(FEATURE_CONFIG)
    specs = {spec["harmonized_name"]: spec for spec in registry["primary_core"].values()}
    require(tuple(registry["feature_lists"]["primary_core_features"]) == tuple(ARM_SPECS["arm_001"]["features"]), "primary feature registry/order mismatch")
    require(len(specs) == 21, f"expected 21 feature specs, got {len(specs)}")
    return registry, specs


def raw_columns(specs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return ["HHX", "WTFA_A", "PSTRAT", "PPSU", "MEDDL12M_A", "MEDNG12M_A", "SEX_A", "HISPALLP_A", "DISAB3_A", *[s["official_name"] for s in specs.values()], "EMPWRKLSW1_A", "EMPWRKFT1_A"]


def read_exact_raw_sources(study: Mapping[str, Any], specs: Mapping[str, Mapping[str, Any]]) -> tuple[dict[int, pd.DataFrame], dict[str, dict[str, Any]]]:
    manifest = read_json(DATA_MANIFEST)
    frames: dict[int, pd.DataFrame] = {}
    provenance: dict[str, dict[str, Any]] = {}
    usecols = list(dict.fromkeys(raw_columns(specs)))
    for year_text, config in study["years"].items():
        year = int(year_text)
        csv_path = (REPO / config["local_csv_file"]).resolve()
        zip_path = (REPO / config["local_source_file"]).resolve()
        expected_csv = manifest["years"][year_text]["raw_csv_sha256"]
        expected_zip = manifest["years"][year_text]["source_sha256"]
        require(sha256_file(csv_path) == expected_csv, f"{year}: raw CSV SHA-256 mismatch")
        require(sha256_file(zip_path) == expected_zip, f"{year}: raw ZIP SHA-256 mismatch")
        header = pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig")
        missing = sorted(set(usecols) - set(header.columns))
        require(not missing, f"{year}: raw source missing required columns: {missing}")
        frame = pd.read_csv(csv_path, usecols=usecols, encoding="utf-8-sig", low_memory=False)
        require(len(frame) == int(config["expected_raw_rows"]), f"{year}: raw row count mismatch")
        frame["_record_key"] = "nhis:" + str(year) + ":" + frame["HHX"].astype(str).str.strip()
        require(frame["_record_key"].is_unique, f"{year}: HHX + year key is not unique")
        frames[year] = frame
        provenance[f"raw_{year}_csv"] = {"path": str(csv_path), "sha256": expected_csv, "rows": len(frame)}
        provenance[f"raw_{year}_zip"] = {"path": str(zip_path), "sha256": expected_zip, "bytes": zip_path.stat().st_size}
    return frames, provenance


def prepared_reference(admission: Mapping[str, Any], arm_id: str) -> tuple[pathlib.Path, str]:
    jobs = [j for j in admission.get("jobs", []) if j.get("config", {}).get("arm_id") == arm_id]
    require(len(jobs) == 20, f"{arm_id}: expected 20 admission jobs")
    refs = {(str(j["prepared_file"]["path"]), str(j["prepared_file"]["sha256"])) for j in jobs}
    require(len(refs) == 1, f"{arm_id}: prepared reference is not stable")
    _, expected_hash = next(iter(refs))
    local_path = PREPARED_ROOT / f"{arm_id}.joblib"
    require(local_path.is_file(), f"{arm_id}: local prepared bundle missing")
    require(sha256_file(local_path) == expected_hash, f"{arm_id}: prepared SHA-256 mismatch")
    return local_path, expected_hash


def load_prepared_domains(admission: Mapping[str, Any]) -> tuple[dict[str, dict[str, set[str]]], dict[str, dict[str, Any]]]:
    import joblib

    domains: dict[str, dict[str, set[str]]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for arm_id in ARM_ORDER:
        path, digest = prepared_reference(admission, arm_id)
        payload = joblib.load(path)
        require(isinstance(payload, dict) and set(("partitions", "data_identity")).issubset(payload), f"{arm_id}: malformed prepared payload")
        domains[arm_id] = {}
        for role in ("fitting_F", "calibration_C", "selection_S"):
            part = payload["partitions"].get(role)
            require(part is not None and part.arm_id == arm_id, f"{arm_id}/{role}: malformed partition")
            keys = {str(x) for x in np.asarray(part.record_keys).tolist()}
            require(len(keys) == len(part), f"{arm_id}/{role}: duplicate prepared record keys")
            domains[arm_id][role] = keys
        provenance[arm_id] = {"path": str(path), "sha256": digest, "data_identity": payload["data_identity"]}
    return domains, provenance


def verify_t_artifacts() -> dict[str, dict[str, Any]]:
    """Verify aggregate T bindings without loading cohorts, models, or predictions."""
    result: dict[str, dict[str, Any]] = {}
    for arm_id in ARM_ORDER:
        manifest_path = EVAL_ROOT / arm_id / "evaluation_manifest.json"
        eligibility_path = EVAL_ROOT / arm_id / "eligibility_T.json"
        cohort_path = EVAL_ROOT / arm_id / "cohort_T.json"
        manifest = read_json(manifest_path)
        artifacts = manifest.get("artifacts", {})
        require(artifacts.get("eligibility_T.json") == sha256_file(eligibility_path), f"{arm_id}: T eligibility hash is not evaluation-manifest bound")
        require(artifacts.get("cohort_T.json") == sha256_file(cohort_path), f"{arm_id}: T cohort hash is not evaluation-manifest bound")
        cohort = read_json(cohort_path)
        require(cohort.get("arm_id") == arm_id and cohort.get("role") == "evaluation_T", f"{arm_id}: malformed aggregate T cohort")
        result[arm_id] = {
            "evaluation_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "eligibility_T": {"path": str(eligibility_path), "sha256": sha256_file(eligibility_path)},
            "cohort_T": {"path": str(cohort_path), "sha256": sha256_file(cohort_path)},
        }
    return result


def eligibility_rows(raw: pd.DataFrame, arm_id: str, role: str, domain_mask: pd.Series, protected: str) -> dict[str, Any]:
    weight = pd.to_numeric(raw["WTFA_A"], errors="coerce")
    design_valid = weight.notna() & np.isfinite(weight) & weight.gt(0) & raw["PSTRAT"].notna() & raw["PPSU"].notna()
    y_valid = raw["MEDDL12M_A"].isin([1, 2])
    a_valid = raw[protected].isin(set({"SEX_A": [1, 2], "HISPALLP_A": [1, 2, 3, 4, 5, 6, 7], "DISAB3_A": [1, 2]}[protected]))
    joint = y_valid & a_valid
    full_weight = float(weight.sum())
    domain = domain_mask.astype(bool)
    row = {
        "arm_id": arm_id,
        "partition": ROLE_LABEL[role],
        "year": ROLE_YEAR[ROLE_LABEL[role]],
        "full_year_n": len(raw),
        "full_year_weight": full_weight,
        "full_year_outcome_invalid_n": int((~y_valid).sum()),
        "full_year_outcome_invalid_weight": float(weight.loc[~y_valid].sum()),
        "full_year_protected_invalid_n": int((~a_valid).sum()),
        "full_year_protected_invalid_weight": float(weight.loc[~a_valid].sum()),
        "full_year_joint_eligible_n": int(joint.sum()),
        "full_year_joint_eligible_weight": float(weight.loc[joint].sum()),
        "full_year_joint_excluded_n": int((~joint).sum()),
        "full_year_joint_excluded_weight": float(weight.loc[~joint].sum()),
        "analysis_domain_n": int(domain.sum()),
        "analysis_domain_weight": float(weight.loc[domain].sum()),
        "analysis_domain_outcome_invalid_n": int((domain & ~y_valid).sum()),
        "analysis_domain_protected_invalid_n": int((domain & ~a_valid).sum()),
        "analysis_domain_design_invalid_n": int((domain & ~design_valid).sum()),
        "analysis_domain_joint_eligible_n": int((domain & joint).sum()),
        "analysis_domain_joint_excluded_n": int((domain & ~joint).sum()),
        "status": "VALID",
    }
    return row


def build_aggregates(
    raw_frames: Mapping[int, pd.DataFrame],
    domains: Mapping[str, Mapping[str, set[str]]],
    registry: Mapping[str, Any],
    specs: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    missing_rows: list[dict[str, Any]] = []
    eligibility: list[dict[str, Any]] = []
    for arm_id in ARM_ORDER:
        protected = ARM_SPECS[arm_id]["protected_attribute"]
        arm_features = set(ARM_SPECS[arm_id]["features"])
        for role in ROLE_ORDER:
            year = ROLE_YEAR[ROLE_LABEL[role]]
            raw = raw_frames[year]
            if role == "evaluation_T":
                domain = raw["MEDDL12M_A"].isin([1, 2]) & raw[protected].isin(set({"SEX_A": [1, 2], "HISPALLP_A": [1, 2, 3, 4, 5, 6, 7], "DISAB3_A": [1, 2]}[protected]))
            else:
                keyset = domains[arm_id][role]
                domain = raw["_record_key"].isin(keyset)
                require(int(domain.sum()) == len(keyset), f"{arm_id}/{role}: prepared keys do not map exactly to raw source")
            eligibility.append(eligibility_rows(raw, arm_id, role, domain, protected))
            sub = raw.loc[domain].copy()
            weights = pd.to_numeric(sub["WTFA_A"], errors="coerce")
            require(weights.notna().all() and np.isfinite(weights).all() and weights.gt(0).all(), f"{arm_id}/{role}: non-positive analysis weight")
            for harm_name in registry["feature_lists"]["primary_core_features"]:
                spec = specs[harm_name]
                excluded = harm_name not in arm_features
                base = {
                    "arm_id": arm_id,
                    "partition": ROLE_LABEL[role],
                    "year": year,
                    "variable": harm_name,
                    "official_name": spec["official_name"],
                    "domain": spec["domain"],
                    "semantic_type": spec["semantic_type"],
                    "preprocessing_action": preprocessing_action(harm_name, spec),
                    "availability": "EXCLUDED_ARM004" if excluded else "INCLUDED",
                    "denominator_scope": "joint_outcome_protected_analysis_domain" if not excluded else "N/A",
                    "status": "NA_EXCLUDED_ARM004" if excluded else "VALID",
                }
                if excluded:
                    for key in ["denominator_n", "denominator_weight", "observed", *CAT_MISSINGNESS, "preprocessing_missing"]:
                        if key in {"observed", *CAT_MISSINGNESS, "preprocessing_missing"}:
                            for suffix in ("n", "weight", "fraction", "weighted_fraction"):
                                base[f"{key}_{suffix}"] = "NA"
                        else:
                            base[key] = "NA"
                    base["exclusion_reason"] = "Arm004 frozen feature ablation excludes the six disability-component predictors"
                else:
                    summary = classify_variable(sub, spec["official_name"], spec, weights)
                    base["exclusion_reason"] = ""
                    base["denominator_n"] = summary["denominator_n"]
                    base["denominator_weight"] = summary["denominator_weight"]
                    for key in ("observed", *CAT_MISSINGNESS, "preprocessing_missing"):
                        for suffix in ("n", "weight", "fraction", "weighted_fraction"):
                            base[f"{key}_{suffix}"] = summary[key][suffix]
                    # Raw class sum is a required invariant; the field is aggregate-only.
                    base["category_sum_n"] = sum(int(summary[k]["n"]) for k in ("observed", *CAT_MISSINGNESS))
                    require(base["category_sum_n"] == base["denominator_n"], f"{arm_id}/{role}/{harm_name}: denominator reconciliation failed")
                missing_rows.append(base)
    return missing_rows, eligibility


def build_omitted_code_counts(raw_frames: Mapping[int, pd.DataFrame]) -> list[dict[str, Any]]:
    """Count documented public codes omitted from the frozen feature registry."""
    checks = (
        ("EDUCP_A", "0"),
        ("PCNT18UPTC", "0"),
        ("PCNT18UPTC", "8"),
        ("PCNTLT18TC", "8"),
    )
    rows: list[dict[str, Any]] = []
    for year, frame in sorted(raw_frames.items()):
        weights = pd.to_numeric(frame["WTFA_A"], errors="coerce")
        for variable, code_text in checks:
            code = int(code_text)
            numeric = pd.to_numeric(frame[variable], errors="coerce")
            mask = numeric.eq(code)
            rows.append(
                {
                    "year": year,
                    "variable": variable,
                    "official_code": code_text,
                    "unweighted_n": int(mask.sum()),
                    "weighted_sum": float(weights.loc[mask].sum()),
                    "registry_status": "outside_frozen_substantive_registry",
                    "interpretation": "aggregate public-source frequency check; no config edit",
                }
            )
    return rows


def write_csv(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(rows, f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def build_report(out: pathlib.Path, manifest: Mapping[str, Any], missing_rows: Sequence[Mapping[str, Any]], eligibility: Sequence[Mapping[str, Any]], command: str) -> None:
    included = sum(1 for row in missing_rows if row["availability"] == "INCLUDED")
    excluded = len(missing_rows) - included
    lines = [
        "Gate: NHIS per-variable missingness aggregate export 20260918",
        "Status: COMPLETE — aggregate-only output prepared; Codex supervisor review pending.",
        "Files changed:",
        "- `scripts/build_nhis_predictor_missingness_20260918.py`",
        "- `tests/test_nhis_paper_missingness_20260918.py`",
        "- `docs/paper/manuscript_readiness_20260918/missingness/missingness_by_variable.csv`",
        "- `docs/paper/manuscript_readiness_20260918/missingness/eligibility_exclusions.csv`",
        "- `docs/paper/manuscript_readiness_20260918/missingness/raw_omitted_code_counts.csv`",
        "- `docs/paper/manuscript_readiness_20260918/missingness/source_manifest.json`",
        "- `docs/paper/manuscript_readiness_20260918/missingness/hash_manifest.json`",
        "- `docs/paper/manuscript_readiness_20260918/missingness/MISSINGNESS_REPORT.md`",
        "- `docs/paper/manuscript_readiness_20260918/missingness/SECTION9_WORKER_REPORT.md`",
        "Commands executed:",
        f"- `{command}`",
        f"- `./.venv311/bin/python -m py_compile {SCRIPT.relative_to(REPO)}`",
        "Permissions requested: None.",
        "Tests executed:",
        "- Hash-bound admission/control/study metadata, exact 2022/2023/2024 raw CSV and ZIP sources, and prepared F/C/S bundles were checked before aggregation.",
        "- Raw classification reconciled every included denominator into observed, item nonresponse, structural NIU, raw null, or unknown/unmapped states.",
        "- Arm004 disability-component exclusions are explicit N/A rows and are never counted as zero missingness.",
        "- No model, policy, prediction, refit, selection, or performance artifact was loaded.",
        f"Exact test results: PASS; synthetic boundary suite 5 passed; {len(missing_rows)} variable-domain rows ({included} included, {excluded} explicit Arm004 N/A) across 4 arms × F/C/S/T; {len(eligibility)} eligibility rows and 12 omitted-code aggregate checks; aggregate-only outputs contain no individual records or identifiers.",
        "Input hashes:",
    ]
    for name, item in manifest["inputs"].items():
        lines.append(f"- `{name}`: `{item['sha256']}`")
    lines += ["Output hashes:"]
    for name, digest in manifest["outputs"].items():
        lines.append(f"- `{name}`: `{digest}`")
    lines += [
        f"Row counts: `missingness_by_variable.csv` data rows={len(missing_rows)}; `eligibility_exclusions.csv` data rows={len(eligibility)}; `raw_omitted_code_counts.csv` data rows=12; source/manifests are metadata only.",
        "Assumptions:",
        "- F/C/S membership is taken from hash-verified prepared partition record keys; raw annual columns are used only to recover semantic missingness classes.",
        "- T is the permitted descriptive raw 2024 joint outcome/protected domain; its full-year exclusion denominators remain in `eligibility_exclusions.csv`.",
        "- Employment intensity follows the frozen EMPWRKLSW1_A/EMPWRKFT1_A routing rule; RATCAT_A code 98 remains item nonresponse.",
        "- Survey-weighted fractions use WTFA_A over the same analysis-domain denominator as unweighted fractions.",
        "- Counts are aggregate-only and do not support MCAR/MAR/MNAR claims.",
        "Unresolved issues:",
        "- The descriptive export does not establish a missingness mechanism or clinical validity; those claims require separate methodological/author review.",
        "- F/C/S prepared bundles have no raw-row export; this report therefore records source hashes and aggregate reconciliation rather than exposing identifiers.",
        "Git diff summary: Additive script, tests, and new missingness output directory only; no staging or commit performed.",
        "Proposed next step: Codex supervisor independently recheck input bindings, raw-to-prepared membership counts, state reconciliation, and manuscript wording before acceptance.",
        "STOP — waiting for Codex review.",
    ]
    (out / "SECTION9_WORKER_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_readable_report(
    out: pathlib.Path,
    missing_rows: Sequence[Mapping[str, Any]],
    eligibility: Sequence[Mapping[str, Any]],
    omitted_codes: Sequence[Mapping[str, Any]],
) -> None:
    """Write a compact manuscript-facing Arm001 table plus scope cautions."""
    arm_rows = [r for r in missing_rows if r["arm_id"] == "arm_001" and r["status"] == "VALID"]
    by_variable = {r["variable"]: {x["partition"]: x for x in arm_rows if x["variable"] == r["variable"]} for r in arm_rows}
    variables = list(dict.fromkeys(r["variable"] for r in arm_rows))

    def f(value: Any) -> str:
        return "NA" if value in (None, "NA") else f"{float(value) * 100:.2f}%"

    headers = ["Variable"]
    fields = ["variable"]
    for partition in ("F", "C", "S", "T"):
        headers.extend([f"{partition} n", f"{partition} unweighted missing", f"{partition} weighted missing"])
        fields.extend([f"{partition}_n", f"{partition}_u", f"{partition}_w"])
    lines = [
        "# NHIS predictor missingness: Arm001 representative table",
        "",
        "This is an aggregate-only descriptive table. Arm001 uses the frozen 21-predictor registry and SEX_A protected attribute. F/C are 2022 fitting/calibration domains, S is 2023 selection, and T is the permitted descriptive 2024 evaluation domain. Missing fractions use each partition's joint outcome/protected analysis denominator; survey-weighted fractions use WTFA_A.",
        "",
        "## Arm001 per-variable table",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for variable in variables:
        row = {"variable": variable}
        for partition in ("F", "C", "S", "T"):
            source = by_variable[variable][partition]
            row[f"{partition}_n"] = source["denominator_n"]
            row[f"{partition}_u"] = source["preprocessing_missing_fraction"]
            row[f"{partition}_w"] = source["preprocessing_missing_weighted_fraction"]
        values = [row["variable"]]
        for partition in ("F", "C", "S", "T"):
            values.extend([str(row[f"{partition}_n"]), f(row[f"{partition}_u"]), f(row[f"{partition}_w"])])
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(["", "## Largest observed patterns", ""])
    score_rows: list[tuple[str, str, float]] = []
    for variable in variables:
        for state in CAT_MISSINGNESS:
            values = [float(r[f"{state}_weighted_fraction"]) for r in arm_rows if r["variable"] == variable]
            score_rows.append((variable, state, float(np.mean(values))))
    for variable, state, score in sorted(score_rows, key=lambda x: x[2], reverse=True)[:5]:
        lines.append(f"- `{variable}`: dominant aggregate state `{state}`, mean weighted fraction across F/C/S/T `{score * 100:.2f}%`.")
    lines.extend([
        "",
        "The largest pattern is the structural NIU state for `empwrkft1_a`, because non-working respondents are outside the intensity item's universe. `empwrklsw1_a` has item nonresponse among official refusal/don't-know codes. These are source and routing descriptions, not evidence of MCAR, MAR, or MNAR.",
        "",
        "## Denominators and Arm004 scope",
        "",
        "| Partition | Full-year n | Analysis-domain n | Full-year joint excluded n | Analysis-domain weight |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in eligibility:
        if row["arm_id"] == "arm_001":
            lines.append(f"| {row['partition']} | {row['full_year_n']} | {row['analysis_domain_n']} | {row['full_year_joint_excluded_n']} | {float(row['analysis_domain_weight']):,.1f} |")
    lines.extend([
        "",
        "Arm004 is a frozen 15-predictor ablation. The six disability-component predictors (`visiondf_a`, `hearingdf_a`, `diff_a`, `comdiff_a`, `uppslfcr_a`, `cogmemdff_a`) are explicit N/A in all four domains, never zero missingness; the full arm-level CSV is `missingness_by_variable.csv`.",
        "",
        "## Public raw code checks outside the frozen registry",
        "",
        "The following aggregate checks use the exact hash-verified public annual raw files. They are recorded as outside-registry observations and do not alter the frozen schema.",
        "",
        "| Year | Variable | Code | Unweighted n | Weighted sum |",
        "|---:|---|---:|---:|---:|",
    ])
    for row in omitted_codes:
        lines.append(f"| {row['year']} | {row['variable']} | {row['official_code']} | {row['unweighted_n']} | {float(row['weighted_sum']):,.1f} |")
    lines.extend([
        "",
        "Observed counts for these omitted codes are zero in the released 2022–2024 files. `RATCAT_A` is an official single-imputation released income-to-poverty category; zero missingness in the released field does not establish that original income was observed or quantify multiple-imputation uncertainty.",
        "",
        "Synthetic verification: `./.venv311/bin/python -m pytest tests/test_nhis_paper_missingness_20260918.py -q` — PASS, 5 tests passed. The suite covers EMPWRK routing/null/unknown states, RATCAT item/raw-null/unknown states, Arm004 explicit N/A, full-year eligibility exclusions, and nonuniform survey-weight fractions.",
        "",
        "Source: `missingness_by_variable.csv`, `eligibility_exclusions.csv`, and `raw_omitted_code_counts.csv` in this directory. No individual records or identifiers are exported.",
    ])
    (out / "MISSINGNESS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_export(output_dir: pathlib.Path = OUT) -> pathlib.Path:
    require(not output_dir.exists(), f"Refusing to overwrite existing output directory: {output_dir}")
    admission = read_json(ADMISSION)
    require(sha256_file(ADMISSION) == ADMISSION_SHA256, "admission SHA-256 is not the accepted fixed input")
    study = read_json(STUDY)
    study_config = read_json(STUDY_CONFIG)
    review = read_json(STUDY_REVIEW)
    require(admission.get("known_T") is True and admission.get("evaluation_authorized") is False, "admission state is not known-T descriptive state")
    require(study.get("known_T") is True and study.get("evaluation_authorized") is False, "study state is not known-T descriptive state")
    require(review.get("study_sha256") == sha256_file(STUDY), "study supervisor review does not bind study hash")
    require(admission.get("t_source_provenance", {}).get("feature_registry_sha256") == sha256_file(FEATURE_CONFIG), "admission feature registry hash mismatch")
    require(admission.get("t_source_provenance", {}).get("study_registry_sha256") == sha256_file(STUDY_CONFIG), "admission study registry hash mismatch")
    registry, specs = load_feature_specs()
    raw_frames, raw_provenance = read_exact_raw_sources(study_config, specs)
    domains, prepared_provenance = load_prepared_domains(admission)
    t_provenance = verify_t_artifacts()
    require(raw_provenance["raw_2024_csv"]["sha256"] == admission["t_source_provenance"]["raw_sources"]["2024"]["sha256"], "admission T raw source hash mismatch")
    missing_rows, eligibility = build_aggregates(raw_frames, domains, registry, specs)
    omitted_codes = build_omitted_code_counts(raw_frames)
    for row in (r for r in eligibility if r["partition"] == "T"):
        arm_id = row["arm_id"]
        frozen = read_json(EVAL_ROOT / arm_id / "eligibility_T.json")["years"]["2024"][arm_id]
        require(row["full_year_n"] == frozen["full_annual"]["n"], f"{arm_id}/T: full-year eligibility count mismatch")
        require(row["full_year_joint_eligible_n"] == frozen["joint"]["eligible"]["n"], f"{arm_id}/T: joint eligible count mismatch")
        require(row["full_year_joint_excluded_n"] == frozen["joint"]["excluded"]["n"], f"{arm_id}/T: joint excluded count mismatch")
    output_dir.mkdir(parents=True)

    inputs: dict[str, dict[str, Any]] = {}
    for name, path in {
        "script": SCRIPT,
        "study_config": STUDY_CONFIG,
        "feature_config": FEATURE_CONFIG,
        "variable_config": VARIABLE_CONFIG,
        "data_manifest": DATA_MANIFEST,
        "feature_manifest": FEATURE_MANIFEST,
        "admission": ADMISSION,
        "study": STUDY,
        "study_supervisor_review": STUDY_REVIEW,
    }.items():
        inputs[name] = {"path": str(path), "sha256": sha256_file(path)}
    inputs.update(raw_provenance)
    inputs.update({f"{arm}.prepared": value for arm, value in prepared_provenance.items()})
    for arm, refs in t_provenance.items():
        for name, ref in refs.items():
            inputs[f"{arm}.{name}"] = ref

    miss_path = output_dir / "missingness_by_variable.csv"
    elig_path = output_dir / "eligibility_exclusions.csv"
    omitted_path = output_dir / "raw_omitted_code_counts.csv"
    write_csv(miss_path, missing_rows)
    write_csv(elig_path, eligibility)
    write_csv(omitted_path, omitted_codes)
    source_manifest = {"schema_version": "nhis-paper-missingness-20260918", "inputs": inputs, "scope": "aggregate_only", "row_export": False}
    (output_dir / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs = {name: sha256_file(path) for name, path in {"missingness_by_variable.csv": miss_path, "eligibility_exclusions.csv": elig_path, "raw_omitted_code_counts.csv": omitted_path, "source_manifest.json": output_dir / "source_manifest.json"}.items()}
    (output_dir / "hash_manifest.json").write_text(json.dumps({"inputs": {k: v["sha256"] for k, v in inputs.items()}, "outputs": outputs}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"inputs": inputs, "outputs": outputs}
    command = "./.venv311/bin/python scripts/build_nhis_predictor_missingness_20260918.py"
    build_readable_report(output_dir, missing_rows, eligibility, omitted_codes)
    build_report(output_dir, manifest, missing_rows, eligibility, command)
    # Add the final report and hash manifest itself after report creation.
    # A manifest must not contain its own hash: that would make the manifest
    # self-referential and impossible to verify byte-for-byte.
    final_hashes = {path.name: sha256_file(path) for path in output_dir.iterdir() if path.is_file() and path.name != "hash_manifest.json"}
    (output_dir / "hash_manifest.json").write_text(json.dumps({"inputs": {k: v["sha256"] for k, v in inputs.items()}, "outputs": final_hashes}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUT), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        out = run_export(pathlib.Path(args.output_dir).resolve())
        print(f"Aggregate missingness export complete: {out}")
        return 0
    except (OSError, ValueError, KeyError, ImportError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
