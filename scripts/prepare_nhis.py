#!/usr/bin/env python3
"""Extract, validate, and harmonize the NHIS D0 record-level foundation."""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Sequence

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_DIR = str(_REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from nhis_fairbias.audit import (
    NHISAuditError,
    build_data_manifest,
    write_csv_atomic,
    write_execution_snapshot,
)
from nhis_fairbias.download import (
    NHISDownloadError,
    extract_year_csv,
    new_run_id,
    parse_years,
    compute_sha256,
)
from nhis_fairbias.harmonize import (
    NHISHarmonizationError,
    combine_harmonized_years,
    harmonize_year,
    validate_harmonized_roles,
    write_harmonized_parquet,
)
from nhis_fairbias.schema import (
    DEFAULT_STUDY_CONFIG,
    DEFAULT_VARIABLE_CONFIG,
    NHISSchemaError,
    build_schema_audit_rows,
    configured_years,
    load_study_config,
    load_variable_registry,
    read_nhis_csv,
    validate_source_frame,
    year_spec,
)
from nhis_fairbias.download import write_json_atomic


def _resolve_repo_path(repo_root: pathlib.Path, relative: str) -> pathlib.Path:
    path = (repo_root / relative).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise NHISHarmonizationError(f"Configured path escapes repository root: {relative}") from exc
    return path


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the audited NHIS 2022-2024 D0 data foundation without modeling."
    )
    parser.add_argument("--years", nargs="+", type=int, default=None, help="Exact configured years to prepare.")
    parser.add_argument("--force", action="store_true", help="Explicitly permit replacing differing derived artifacts.")
    parser.add_argument("--output-parquet", default=None, help="Explicit destination path for harmonized Parquet.")
    parser.add_argument("--output-schema", default=None, help="Explicit destination path for schema audit CSV.")
    parser.add_argument("--repo-root", default=str(_REPO_ROOT), help=argparse.SUPPRESS)
    parser.add_argument("--study-config", default=str(DEFAULT_STUDY_CONFIG), help=argparse.SUPPRESS)
    parser.add_argument("--variables", default=str(DEFAULT_VARIABLE_CONFIG), help=argparse.SUPPRESS)
    parser.add_argument("--manifest", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> int:
    parsed = parse_args(args)
    repo_root = pathlib.Path(parsed.repo_root).resolve()
    try:
        study_config = load_study_config(parsed.study_config)
        registry = load_variable_registry(parsed.variables)
        years = parse_years(parsed.years, study_config)
        raw_frames = {}
        harmonized_frames = {}
        validations = {}
        extraction_records = {}
        schema_rows = []

        for year in years:
            extraction = extract_year_csv(
                year,
                repo_root=repo_root,
                study_config=study_config,
                force=parsed.force,
                manifest_path=parsed.manifest,
            )
            raw_path = pathlib.Path(extraction["path"])
            raw_frame = read_nhis_csv(raw_path, registry=registry)
            validation = validate_source_frame(
                raw_frame,
                year=year,
                study_config=study_config,
                registry=registry,
            )
            spec = year_spec(study_config, year)
            harmonized = harmonize_year(
                raw_frame,
                year=year,
                study_role=str(spec["study_role"]),
                registry=registry,
            )
            raw_frames[year] = raw_frame
            harmonized_frames[year] = harmonized
            validations[year] = validation
            extraction_records[year] = extraction
            schema_rows.extend(validation.schema_rows)
            observed = ", ".join(
                f"{code}={count}" for code, count in sorted(validation.primary_frequency.items())
            )
            print(
                f"Validated NHIS {year}: rows={validation.raw_row_count:,}; "
                f"MEDDL12M_A [{observed}]; role={spec['study_role']}"
            )

        combined = combine_harmonized_years(harmonized_frames, years)
        validate_harmonized_roles(combined, years)
        is_full_study = set(years) == set(configured_years(study_config))
        if parsed.output_parquet:
            processed_path = _resolve_repo_path(repo_root, parsed.output_parquet)
        elif is_full_study:
            processed_path = _resolve_repo_path(
                repo_root, str(study_config["outputs"]["harmonized_parquet"])
            )
        else:
            subset_name = "_".join(str(y) for y in sorted(years))
            processed_path = _resolve_repo_path(
                repo_root, f"data/processed/nhis/nhis_{subset_name}_core.parquet"
            )
        processed_sha = write_harmonized_parquet(combined, processed_path, force=parsed.force)

        if parsed.output_schema:
            schema_path = _resolve_repo_path(repo_root, parsed.output_schema)
        elif is_full_study:
            schema_path = _resolve_repo_path(repo_root, str(study_config["outputs"]["schema_audit"]))
        else:
            subset_name = "_".join(str(y) for y in sorted(years))
            schema_path = _resolve_repo_path(
                repo_root, f"artifacts/nhis/data/schema_audit_{subset_name}.csv"
            )
        schema_sha = write_csv_atomic(
            __import__("pandas").DataFrame(schema_rows), schema_path, force=parsed.force
        )

        run_id = new_run_id()
        manifest = build_data_manifest(
            repo_root=repo_root,
            study_config=study_config,
            registry=registry,
            years=years,
            raw_frames=raw_frames,
            validations=validations,
            extraction_records=extraction_records,
            processed_path=processed_path,
            run_id=run_id,
        )
        if parsed.manifest:
            manifest_path = _resolve_repo_path(repo_root, parsed.manifest)
        elif is_full_study:
            manifest_path = _resolve_repo_path(repo_root, str(study_config["outputs"]["manifest"]))
        else:
            subset_name = "_".join(str(y) for y in sorted(years))
            manifest_path = _resolve_repo_path(
                repo_root, f"artifacts/nhis/data/data_manifest_{subset_name}.json"
            )
        write_json_atomic(manifest_path, manifest)
        manifest_sha = compute_sha256(manifest_path)
        snapshot = write_execution_snapshot(
            repo_root=repo_root,
            study_config=study_config,
            run_id=run_id,
            operation="prepare_nhis",
            years=years,
            output_hashes={
                "harmonized_parquet": processed_sha,
                "schema_audit": schema_sha,
                "data_manifest": manifest_sha,
            },
        )
        print(f"Wrote harmonized Parquet: {processed_path} ({len(combined):,} rows)")
        print(f"Wrote schema audit: {schema_path}")
        print(f"Wrote execution manifest: {snapshot}")
        return 0
    except (
        NHISDownloadError,
        NHISSchemaError,
        NHISHarmonizationError,
        NHISAuditError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
