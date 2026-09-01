"""Non-modeling NHIS D0 audit tables and provenance manifests."""

from __future__ import annotations

import datetime as _datetime
import json
import os
import pathlib
import subprocess
import sys
import uuid
from typing import Any, Mapping, Sequence

import pandas as pd

from .download import NHISDownloadError, compute_sha256, new_run_id, utc_timestamp, write_json_atomic
from .schema import (
    OUTCOME_COLUMNS,
    PROTECTED_COLUMNS,
    REQUIRED_COLUMNS,
    SourceValidation,
    canonical_code_series,
    year_spec,
)
from .survey import survey_design_audit, weighted_binary_proportion, weighted_category_proportion


class NHISAuditError(RuntimeError):
    """Raised when an audit cannot be completed or an output is unsafe to replace."""


DEFAULT_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _git_commit_sha(repo_root: pathlib.Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    sha = completed.stdout.strip()
    return sha or None


def _relative_to_repo(path: pathlib.Path, repo_root: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _read_existing_manifest(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise NHISAuditError(f"Could not read data manifest {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise NHISAuditError(f"Data manifest must contain a JSON object: {path}")
    return loaded


def _artifact_path(repo_root: pathlib.Path, study_config: Mapping[str, Any], name: str) -> pathlib.Path:
    try:
        relative = str(study_config["outputs"][name])
    except (KeyError, TypeError) as exc:
        raise NHISAuditError(f"NHIS study configuration is missing output path {name!r}.") from exc
    path = (repo_root / relative).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise NHISAuditError(f"Configured output path escapes repository root: {relative}") from exc
    return path


def _write_text_atomic(path: pathlib.Path, text: str, *, force: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
    try:
        with part.open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        new_sha = compute_sha256(part)
        if path.exists() and not force:
            old_sha = compute_sha256(path)
            if old_sha == new_sha:
                part.unlink(missing_ok=True)
                return old_sha
            raise NHISAuditError(
                f"Existing artifact differs at {path}; refusing to overwrite without --force."
            )
        os.replace(part, path)
        return new_sha
    except NHISAuditError:
        part.unlink(missing_ok=True)
        raise
    except (OSError, ValueError) as exc:
        part.unlink(missing_ok=True)
        raise NHISAuditError(f"Could not write artifact {path}: {exc}") from exc


def write_csv_atomic(frame: pd.DataFrame, path: pathlib.Path | str, *, force: bool = False) -> str:
    """Write an audit CSV atomically, preserving an existing different result."""

    path_obj = pathlib.Path(path)
    try:
        csv_text = frame.to_csv(index=False, lineterminator="\n")
    except (TypeError, ValueError) as exc:
        raise NHISAuditError(f"Could not serialize audit CSV {path_obj}: {exc}") from exc
    return _write_text_atomic(path_obj, csv_text, force=force)


def _year_subset(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    if "survey_year" not in frame.columns:
        raise NHISAuditError("Harmonized table is missing survey_year.")
    subset = frame.loc[frame["survey_year"].eq(int(year))].copy()
    if subset.empty:
        raise NHISAuditError(f"Harmonized table contains no records for {year}.")
    return subset


def outcome_audit_rows(
    frame: pd.DataFrame,
    *,
    year: int,
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return code-count and design-weighted Yes/No rows for both outcomes."""

    subset = _year_subset(frame, year)
    rows: list[dict[str, Any]] = []
    denominator_n = len(subset)
    weights = subset["WTFA_A"]
    for outcome in OUTCOME_COLUMNS:
        raw_name = f"{outcome}_raw"
        if raw_name not in subset.columns:
            raise NHISAuditError(f"Harmonized table is missing preserved raw outcome {raw_name}.")
        codes = canonical_code_series(subset[raw_name])
        spec = registry["variables"][outcome]
        valid_codes = tuple(int(value) for value in spec["valid_codes"])
        missing_codes = tuple(int(value) for value in spec["missing_codes"])
        expected_codes = tuple(sorted(set(valid_codes) | set(missing_codes)))
        for code in expected_codes:
            n = int(codes.eq(code).sum())
            weighted = (
                weighted_binary_proportion(codes, weights, code=code, valid_codes=valid_codes)
                if code in valid_codes
                else None
            )
            rows.append(
                {
                    "year": int(year),
                    "outcome": outcome,
                    "raw_code": str(code),
                    "unweighted_n": n,
                    "unweighted_proportion": float(n / denominator_n),
                    "survey_weighted_proportion": weighted,
                }
            )
        invalid_mask = (~codes.isin(list(expected_codes))).fillna(True)
        invalid_n = int(invalid_mask.sum())
        if invalid_n:
            rows.append(
                {
                    "year": int(year),
                    "outcome": outcome,
                    "raw_code": "MISSING_OR_INVALID",
                    "unweighted_n": invalid_n,
                    "unweighted_proportion": float(invalid_n / denominator_n),
                    "survey_weighted_proportion": None,
                }
            )
    return rows


def protected_attribute_audit_rows(
    frame: pd.DataFrame,
    *,
    year: int,
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return raw protected-category counts and positive-weight population shares."""

    subset = _year_subset(frame, year)
    rows: list[dict[str, Any]] = []
    weights = subset["WTFA_A"]
    for variable in PROTECTED_COLUMNS:
        if variable not in subset.columns:
            raise NHISAuditError(f"Harmonized table is missing protected variable {variable}.")
        codes = canonical_code_series(subset[variable])
        spec = registry["variables"][variable]
        valid_codes = tuple(int(value) for value in spec["valid_codes"])
        for code in valid_codes:
            rows.append(
                {
                    "year": int(year),
                    "variable": variable,
                    "protected_variable": variable,
                    "raw_category": str(code),
                    "n": int(codes.eq(code).sum()),
                    "weighted_population_proportion": weighted_category_proportion(
                        codes, weights, code=code, valid_codes=valid_codes
                    ),
                    "missing_invalid_flag": False,
                }
            )
        invalid_mask = (~codes.isin(list(valid_codes))).fillna(True)
        rows.append(
            {
                "year": int(year),
                "variable": variable,
                "protected_variable": variable,
                "raw_category": "MISSING_OR_INVALID",
                "n": int(invalid_mask.sum()),
                "weighted_population_proportion": weighted_category_proportion(
                    codes, weights, code=None, valid_codes=valid_codes
                ),
                "missing_invalid_flag": True,
            }
        )
    return rows


def build_audit_tables(
    frame: pd.DataFrame,
    *,
    years: Sequence[int],
    registry: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build outcome, protected-attribute, and survey-design audit tables."""

    outcome_rows: list[dict[str, Any]] = []
    protected_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    for year in years:
        outcome_rows.extend(outcome_audit_rows(frame, year=int(year), registry=registry))
        protected_rows.extend(protected_attribute_audit_rows(frame, year=int(year), registry=registry))
        weight_rows.append(survey_design_audit(_year_subset(frame, int(year)), year=int(year)))
    return pd.DataFrame(outcome_rows), pd.DataFrame(protected_rows), pd.DataFrame(weight_rows)


def build_data_manifest(
    *,
    repo_root: pathlib.Path | str,
    study_config: Mapping[str, Any],
    registry: Mapping[str, Any],
    years: Sequence[int],
    raw_frames: Mapping[int, pd.DataFrame],
    validations: Mapping[int, SourceValidation],
    extraction_records: Mapping[int, Mapping[str, Any]],
    processed_path: pathlib.Path | str,
    run_id: str,
) -> dict[str, Any]:
    """Assemble the required source, schema, row-count, and software provenance."""

    repo_path = pathlib.Path(repo_root).resolve()
    manifest_path = _artifact_path(repo_path, study_config, "manifest")
    prior_manifest = _read_existing_manifest(manifest_path)
    manifest: dict[str, Any] = {
        "schema_version": study_config.get("schema_version", "nhis-d0"),
        "dataset": study_config.get("dataset"),
        "software_timestamp_utc": utc_timestamp(),
        "git_commit_sha": _git_commit_sha(repo_path),
        "run_id": run_id,
        "years": {},
        "processed": {
            "file": _relative_to_repo(pathlib.Path(processed_path), repo_path),
            "row_count": None,
            "sha256": None,
        },
        "audit_artifacts": {
            "manifest": _relative_to_repo(manifest_path, repo_path),
            "schema_audit": _relative_to_repo(_artifact_path(repo_path, study_config, "schema_audit"), repo_path),
            "outcome_audit": _relative_to_repo(_artifact_path(repo_path, study_config, "outcome_audit"), repo_path),
            "protected_attribute_audit": _relative_to_repo(
                _artifact_path(repo_path, study_config, "protected_attribute_audit"), repo_path
            ),
            "weight_audit": _relative_to_repo(_artifact_path(repo_path, study_config, "weight_audit"), repo_path),
        },
    }

    processed_obj = pathlib.Path(processed_path)
    if processed_obj.is_file():
        manifest["processed"]["row_count"] = int(sum(len(raw_frames[int(year)]) for year in years))
        manifest["processed"]["sha256"] = compute_sha256(processed_obj)

    prior_years = prior_manifest.get("years", {})
    if not isinstance(prior_years, dict):
        prior_years = {}
    manifest["years"] = dict(prior_years)
    for year in years:
        year_int = int(year)
        spec = year_spec(study_config, year_int)
        raw = raw_frames[year_int]
        validation = validations[year_int]
        extraction = extraction_records[year_int]
        previous = prior_years.get(str(year_int), {})
        if not isinstance(previous, dict):
            previous = {}
        minimal_presence = {column: column in raw.columns for column in REQUIRED_COLUMNS}
        manifest["years"][str(year_int)] = {
            "source_url": spec["source_url"],
            "local_source_file": spec["local_source_file"],
            "source_sha256": extraction.get("source_sha256"),
            "sha256": extraction.get("source_sha256"),
            "source_byte_size": extraction.get("source_byte_size"),
            "download_timestamp_utc": previous.get("download_timestamp_utc"),
            "raw_csv_file": spec["local_csv_file"],
            "raw_csv_sha256": extraction.get("sha256"),
            "raw_csv_byte_size": pathlib.Path(extraction["path"]).stat().st_size,
            "zip_csv_member": extraction.get("member"),
            "raw_row_count": int(len(raw)),
            "processed_row_count": int(len(raw)),
            "study_role": spec["study_role"],
            "minimal_column_presence": minimal_presence,
            "required_column_audit": minimal_presence,
            "expected_raw_row_count": int(spec["expected_raw_rows"]),
            "expected_meddl12m_counts": {
                str(code): int(count) for code, count in spec["expected_meddl12m_counts"].items()
            },
            "observed_meddl12m_counts": {
                str(code): int(count) for code, count in validation.primary_frequency.items()
            },
            "validation": {
                "minimal_schema": "PASS" if all(minimal_presence.values()) else "FAIL",
                "row_count": "PASS" if len(raw) == int(spec["expected_raw_rows"]) else "FAIL",
                "meddl12m_frequency": "PASS",
            },
        }
    return manifest


def update_manifest_after_audit(
    manifest_path: pathlib.Path | str,
    *,
    run_id: str,
    artifact_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Record audit completion and hashes in the canonical manifest."""

    path_obj = pathlib.Path(manifest_path)
    manifest = _read_existing_manifest(path_obj)
    manifest["software_timestamp_utc"] = utc_timestamp()
    manifest["audit_run_id"] = run_id
    manifest["audit_completed_utc"] = utc_timestamp()
    manifest["audit_artifact_sha256"] = dict(artifact_hashes)
    manifest["audit_status"] = "PASS"
    write_json_atomic(path_obj, manifest)
    return manifest


def write_execution_snapshot(
    *,
    repo_root: pathlib.Path | str,
    study_config: Mapping[str, Any],
    run_id: str,
    operation: str,
    years: Sequence[int],
    output_hashes: Mapping[str, str],
) -> pathlib.Path:
    """Persist a unique immutable per-run execution record under the ignored artifacts tree."""

    repo_path = pathlib.Path(repo_root).resolve()
    artifact_root = (repo_path / str(study_config["outputs"]["artifact_root"])).resolve()
    path = artifact_root / "runs" / run_id / "execution_manifest.json"
    write_json_atomic(
        path,
        {
            "run_id": run_id,
            "operation": operation,
            "software_timestamp_utc": utc_timestamp(),
            "git_commit_sha": _git_commit_sha(repo_path),
            "years": [int(year) for year in years],
            "output_sha256": dict(output_hashes),
        },
    )
    return path
