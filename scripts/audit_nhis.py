#!/usr/bin/env python3
"""CLI entrypoint for the NHIS D0 outcome, protected-attribute, and survey audits."""

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
    build_audit_tables,
    update_manifest_after_audit,
    write_csv_atomic,
    write_execution_snapshot,
)
from nhis_fairbias.download import compute_sha256, new_run_id
from nhis_fairbias.harmonize import NHISHarmonizationError, read_harmonized_parquet, validate_harmonized_roles
from nhis_fairbias.schema import (
    DEFAULT_STUDY_CONFIG,
    DEFAULT_VARIABLE_CONFIG,
    NHISSchemaError,
    REQUIRED_COLUMNS,
    load_study_config,
    load_variable_registry,
    validate_source_frame,
)
from nhis_fairbias.download import parse_years, write_json_atomic


def _resolve_repo_path(repo_root: pathlib.Path, relative: str) -> pathlib.Path:
    path = (repo_root / relative).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise NHISAuditError(f"Configured path escapes repository root: {relative}") from exc
    return path


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit prepared NHIS D0 data without modeling.")
    parser.add_argument("--years", nargs="+", type=int, default=None, help="Exact configured years to audit.")
    parser.add_argument("--force", action="store_true", help="Explicitly permit replacing differing audit CSVs.")
    parser.add_argument("--repo-root", default=str(_REPO_ROOT), help=argparse.SUPPRESS)
    parser.add_argument("--study-config", default=str(DEFAULT_STUDY_CONFIG), help=argparse.SUPPRESS)
    parser.add_argument("--variables", default=str(DEFAULT_VARIABLE_CONFIG), help=argparse.SUPPRESS)
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> int:
    parsed = parse_args(args)
    repo_root = pathlib.Path(parsed.repo_root).resolve()
    try:
        study_config = load_study_config(parsed.study_config)
        registry = load_variable_registry(parsed.variables)
        years = parse_years(parsed.years, study_config)
        processed_path = _resolve_repo_path(repo_root, str(study_config["outputs"]["harmonized_parquet"]))
        frame = read_harmonized_parquet(processed_path)
        validate_harmonized_roles(frame, years)

        # Re-run the source contracts from the preserved official columns. This
        # prevents an audit CSV from being treated as evidence of a valid input.
        for year in years:
            subset = frame.loc[frame["survey_year"].eq(int(year)), list(REQUIRED_COLUMNS)].copy()
            validation = validate_source_frame(
                subset,
                year=year,
                study_config=study_config,
                registry=registry,
            )
            for outcome in ("MEDDL12M_A", "MEDNG12M_A"):
                raw_alias = f"{outcome}_raw"
                if raw_alias not in frame.columns:
                    raise NHISAuditError(f"Preserved raw outcome is missing from harmonized data: {raw_alias}")
                if not frame.loc[frame["survey_year"].eq(int(year)), outcome].reset_index(drop=True).equals(
                    frame.loc[frame["survey_year"].eq(int(year)), raw_alias].reset_index(drop=True)
                ):
                    raise NHISAuditError(f"Source variable {outcome} was not preserved unchanged for {year}.")
            print(f"Revalidated NHIS {year}: rows={validation.raw_row_count:,}; source frequency contract=PASS")

        outcome_table, protected_table, weight_table = build_audit_tables(
            frame, years=years, registry=registry
        )
        if (weight_table["status"] == "FAIL").any():
            raise NHISAuditError("WTFA_A contains non-numeric values; survey audit cannot pass.")

        output_paths = {
            "outcome_audit": _resolve_repo_path(repo_root, str(study_config["outputs"]["outcome_audit"])),
            "protected_attribute_audit": _resolve_repo_path(
                repo_root, str(study_config["outputs"]["protected_attribute_audit"])
            ),
            "weight_audit": _resolve_repo_path(repo_root, str(study_config["outputs"]["weight_audit"])),
        }
        hashes = {
            key: write_csv_atomic(table, output_paths[key], force=parsed.force)
            for key, table in (
                ("outcome_audit", outcome_table),
                ("protected_attribute_audit", protected_table),
                ("weight_audit", weight_table),
            )
        }
        manifest_path = _resolve_repo_path(repo_root, str(study_config["outputs"]["manifest"]))
        run_id = new_run_id()
        update_manifest_after_audit(manifest_path, run_id=run_id, artifact_hashes=hashes)
        hashes["data_manifest"] = compute_sha256(manifest_path)
        snapshot = write_execution_snapshot(
            repo_root=repo_root,
            study_config=study_config,
            run_id=run_id,
            operation="audit_nhis",
            years=years,
            output_hashes=hashes,
        )
        for year in years:
            weight_row = weight_table.loc[weight_table["year"].eq(int(year))].iloc[0]
            print(
                f"Survey audit {year}: WTFA_A missing={int(weight_row['wtfa_a_missing_n'])}; "
                f"nonpositive={int(weight_row['wtfa_a_nonpositive_n'])}; "
                f"PSTRAT unique={int(weight_row['pstrat_nunique'])}; "
                f"PPSU unique={int(weight_row['ppsu_nunique'])}"
            )
        print("Wrote outcome, protected-attribute, and survey-weight audit tables.")
        print(f"Wrote execution manifest: {snapshot}")
        return 0
    except (
        NHISAuditError,
        NHISSchemaError,
        NHISHarmonizationError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
