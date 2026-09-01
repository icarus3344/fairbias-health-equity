#!/usr/bin/env python3
"""CLI script to build the audited Gate D1 feature space for NHIS 2022-2024."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Sequence

import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_DIR = str(_REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from nhis_fairbias.audit import NHISAuditError, write_csv_atomic, write_execution_snapshot
from nhis_fairbias.download import compute_sha256, new_run_id, parse_years, write_json_atomic
from nhis_fairbias.features import (
    DEFAULT_FEATURE_CONFIG,
    NHISFeatureError,
    build_feature_availability_audit_table,
    build_feature_code_domain_audit_table,
    build_feature_distribution_audit_table,
    build_feature_manifest,
    build_feature_missingness_audit_table,
    build_feature_registry_audit_table,
    build_leakage_guard_audit_table,
    combine_feature_years,
    harmonize_features_year,
    load_feature_registry,
    write_features_parquet,
)
from nhis_fairbias.schema import (
    DEFAULT_STUDY_CONFIG,
    NHISSchemaError,
    exact_primary_frequency,
    load_study_config,
    year_spec,
)


def _resolve_repo_path(repo_root: pathlib.Path, relative: str) -> pathlib.Path:
    path = (repo_root / relative).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise NHISFeatureError(f"Configured path escapes repository root: {relative}") from exc
    return path


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the harmonized NHIS 2022-2024 Gate D1 feature space and audit artifacts."
    )
    parser.add_argument("--years", nargs="+", type=int, default=None, help="Exact configured years to process.")
    parser.add_argument("--force", action="store_true", help="Explicitly permit replacing differing derived artifacts.")
    parser.add_argument("--repo-root", default=str(_REPO_ROOT), help=argparse.SUPPRESS)
    parser.add_argument("--study-config", default=str(DEFAULT_STUDY_CONFIG), help=argparse.SUPPRESS)
    parser.add_argument("--features-config", default=str(DEFAULT_FEATURE_CONFIG), help=argparse.SUPPRESS)
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> int:
    parsed = parse_args(args)
    repo_root = pathlib.Path(parsed.repo_root).resolve()
    run_id = new_run_id()

    try:
        study_config = load_study_config(parsed.study_config)
        feature_registry = load_feature_registry(parsed.features_config)
        years = parse_years(parsed.years, study_config)

        # 1. Read raw CSVs and extract headers
        raw_frames: dict[int, pd.DataFrame] = {}
        raw_headers: dict[int, list[str]] = {}
        harmonized_year_frames: dict[int, pd.DataFrame] = {}

        # Columns needed for D1 feature extraction:
        needed_cols = set()
        # Design / ID
        needed_cols.update(["SRVY_YR", "WTFA_A", "PSTRAT", "PPSU"])
        # Outcomes
        needed_cols.update(["MEDDL12M_A", "MEDNG12M_A"])
        # Protected
        needed_cols.update(["SEX_A", "HISPALLP_A", "DISAB3_A"])
        # Primary core
        needed_cols.update(feature_registry["primary_core"].keys())
        # Expanded utilization
        needed_cols.update(feature_registry["expanded_utilization"].keys())

        for year in years:
            year_int = int(year)
            spec = year_spec(study_config, year_int)
            csv_path = _resolve_repo_path(repo_root, spec["local_csv_file"])
            if not csv_path.is_file():
                raise NHISFeatureError(f"Raw CSV file missing for {year_int}: {csv_path}")

            # Read full header
            header_cols = pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig").columns.tolist()
            raw_headers[year_int] = header_cols

            # Read columns present in this year
            cols_to_load = [c for c in needed_cols if c in header_cols]
            df = pd.read_csv(
                csv_path,
                usecols=cols_to_load,
                encoding="utf-8-sig",
                low_memory=False,
            )

            # Contract checks: exact row count and MEDDL12M_A frequency
            expected_rows = int(spec["expected_raw_rows"])
            if len(df) != expected_rows:
                raise NHISFeatureError(
                    f"Row count mismatch for {year_int}: expected {expected_rows}, observed {len(df)}."
                )
            exact_primary_frequency(
                df,
                year=year_int,
                expected_counts=spec["expected_meddl12m_counts"],
            )

            raw_frames[year_int] = df

            # Harmonize this year
            study_role = spec["study_role"]
            harm_df = harmonize_features_year(
                df,
                year=year_int,
                study_role=study_role,
                feature_registry=feature_registry,
            )
            harmonized_year_frames[year_int] = harm_df
            print(f"Harmonized NHIS {year_int} ({study_role}): {len(harm_df):,} rows, {len(harm_df.columns)} columns.")

        # Combine across years
        combined_features = combine_feature_years(harmonized_year_frames, years)
        expected_total = sum(int(year_spec(study_config, y)["expected_raw_rows"]) for y in years)
        if len(combined_features) != expected_total:
            raise NHISFeatureError(
                f"Combined row count mismatch: expected {expected_total}, observed {len(combined_features)}."
            )

        # Output paths
        outputs = study_config["outputs"]
        features_parquet_path = _resolve_repo_path(repo_root, outputs["features_parquet"])
        artifact_root = _resolve_repo_path(repo_root, outputs["features_artifact_root"])
        artifact_root.mkdir(parents=True, exist_ok=True)

        audit_file_map = {
            "feature_manifest": _resolve_repo_path(repo_root, outputs["feature_manifest"]),
            "feature_registry_audit": _resolve_repo_path(repo_root, outputs["feature_registry_audit"]),
            "feature_availability_audit": _resolve_repo_path(repo_root, outputs["feature_availability_audit"]),
            "feature_code_domain_audit": _resolve_repo_path(repo_root, outputs["feature_code_domain_audit"]),
            "feature_missingness_audit": _resolve_repo_path(repo_root, outputs["feature_missingness_audit"]),
            "feature_distribution_audit": _resolve_repo_path(repo_root, outputs["feature_distribution_audit"]),
            "leakage_guard_audit": _resolve_repo_path(repo_root, outputs["leakage_guard_audit"]),
        }

        # Build audits
        print("Generating feature registry audit...")
        reg_audit = build_feature_registry_audit_table(feature_registry)
        write_csv_atomic(reg_audit, audit_file_map["feature_registry_audit"], force=parsed.force)

        print("Generating feature availability audit...")
        avail_audit = build_feature_availability_audit_table(raw_headers, years, feature_registry)
        write_csv_atomic(avail_audit, audit_file_map["feature_availability_audit"], force=parsed.force)

        print("Generating feature code domain audit...")
        domain_audit = build_feature_code_domain_audit_table(raw_frames, years, feature_registry)
        write_csv_atomic(domain_audit, audit_file_map["feature_code_domain_audit"], force=parsed.force)

        print("Generating feature missingness audit...")
        miss_audit = build_feature_missingness_audit_table(combined_features, years, feature_registry)
        write_csv_atomic(miss_audit, audit_file_map["feature_missingness_audit"], force=parsed.force)

        print("Generating feature distribution audit...")
        dist_audit = build_feature_distribution_audit_table(combined_features, years, feature_registry)
        write_csv_atomic(dist_audit, audit_file_map["feature_distribution_audit"], force=parsed.force)

        print("Running leakage guard audit...")
        leak_audit = build_leakage_guard_audit_table(feature_registry, combined_features)
        write_csv_atomic(leak_audit, audit_file_map["leakage_guard_audit"], force=parsed.force)

        # Write Parquet
        print(f"Writing harmonized Parquet to {features_parquet_path}...")
        parquet_sha = write_features_parquet(combined_features, features_parquet_path, force=parsed.force)
        print(f"Parquet written: SHA256={parquet_sha}")

        # Build and write manifest
        print("Generating feature manifest...")
        manifest = build_feature_manifest(
            repo_root=repo_root,
            study_config=study_config,
            feature_registry=feature_registry,
            years=years,
            processed_parquet_path=features_parquet_path,
            audit_paths=audit_file_map,
            run_id=run_id,
            feature_frame=combined_features,
        )
        write_json_atomic(audit_file_map["feature_manifest"], manifest)

        # Execution snapshot in runs/
        run_dir = artifact_root / "runs" / run_id
        run_manifest_path = run_dir / "execution_manifest.json"
        write_json_atomic(run_manifest_path, manifest)

        print("Gate D1 feature space and audits successfully established!")
        return 0

    except (NHISFeatureError, NHISSchemaError, NHISAuditError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
