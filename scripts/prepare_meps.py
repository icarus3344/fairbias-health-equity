#!/usr/bin/env python3
"""CLI entrypoint for secure MEPS data extraction and bounded structural schema verification.

Usage:
  python3 scripts/prepare_meps.py [options]

Modes:
  Default: Extracts raw archives (with pure-skip idempotency) and executes structural schema scan.
  --extract-only: Performs secure extraction and preparation provenance recording only.
  --schema-only: Performs bounded structural schema scan on interim files only.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

# Ensure src/ is on sys.path regardless of execution working directory
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

try:
    from meps_fairness.data.prepare import (
        BOUNDED_SCAN_CHUNK_SIZE,
        DEFAULT_EXPECTATIONS_PATH,
        DEFAULT_INTERIM_ROOT,
        DEFAULT_PREPARATION_PROVENANCE_PATH,
        DEFAULT_PROVENANCE_PATH,
        DEFAULT_RAW_ROOT,
        DEFAULT_SNAPSHOT_PATH,
        MAX_RSS_GB_LIMIT,
        MAX_SCAN_CHUNK_SIZE,
        DownloadError,
        PreparationProvenance,
        SchemaExpectations,
        StructuralCheckResult,
        compare_cross_panel_schemas,
        extract_meps_archives,
        generate_and_save_schema_snapshot,
        get_peak_rss_gb,
        scan_single_panel_schema,
    )
except ImportError as imp_err:
    sys.stderr.write(
        f"[ERROR] Failed importing meps_fairness.data.prepare: {imp_err}. "
        "Please run with Python environment containing pandas/numpy (e.g. .venv311).\\n"
    )
    sys.exit(1)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and verify MEPS data and structural schema in bounded chunks."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--extract-only",
        action="store_true",
        help="Extract archives to interim .dta files and update preparation provenance without scanning schema.",
    )
    group.add_argument(
        "--schema-only",
        action="store_true",
        help="Run structural schema scan and snapshot on existing interim files without extracting archives.",
    )

    parser.add_argument(
        "--expectations",
        type=str,
        default=DEFAULT_EXPECTATIONS_PATH,
        help=f"Path to schema expectations configuration (default: {DEFAULT_EXPECTATIONS_PATH}).",
    )
    parser.add_argument(
        "--provenance",
        type=str,
        default=DEFAULT_PROVENANCE_PATH,
        help=f"Path to local raw provenance snapshot (default: {DEFAULT_PROVENANCE_PATH}).",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default=DEFAULT_RAW_ROOT,
        help=f"Root directory for raw archives (default: {DEFAULT_RAW_ROOT}).",
    )
    parser.add_argument(
        "--interim-root",
        type=str,
        default=DEFAULT_INTERIM_ROOT,
        help=f"Root directory for interim .dta files (default: {DEFAULT_INTERIM_ROOT}).",
    )
    parser.add_argument(
        "--preparation-provenance",
        type=str,
        default=DEFAULT_PREPARATION_PROVENANCE_PATH,
        help=f"Path to preparation provenance manifest (default: {DEFAULT_PREPARATION_PROVENANCE_PATH}).",
    )
    parser.add_argument(
        "--snapshot-output",
        type=str,
        default=DEFAULT_SNAPSHOT_PATH,
        help=f"Path to output schema snapshot JSON (default: {DEFAULT_SNAPSHOT_PATH}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=BOUNDED_SCAN_CHUNK_SIZE,
        help=f"Bounded chunk size for Stata streaming (default: {BOUNDED_SCAN_CHUNK_SIZE}, max: {MAX_SCAN_CHUNK_SIZE}).",
    )

    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    t_start = time.monotonic()
    parsed = parse_args(args)

    repo_root = _REPO_ROOT
    expectations_file = repo_root / parsed.expectations if not pathlib.Path(parsed.expectations).is_absolute() else pathlib.Path(parsed.expectations)
    prep_prov_file = repo_root / parsed.preparation_provenance if not pathlib.Path(parsed.preparation_provenance).is_absolute() else pathlib.Path(parsed.preparation_provenance)

    try:
        exp = SchemaExpectations.load(expectations_file)
    except Exception as err:
        sys.stderr.write(f"[ERROR] Failed to load schema expectations from {expectations_file}: {err}\n")
        return 1

    # Stage 1: Secure Extraction (unless --schema-only)
    prep_prov: PreparationProvenance
    if not parsed.schema_only:
        try:
            prep_prov, extract_statuses = extract_meps_archives(
                expectations_path=parsed.expectations,
                provenance_path=parsed.provenance,
                raw_root=parsed.raw_root,
                interim_root=parsed.interim_root,
                preparation_provenance_path=parsed.preparation_provenance,
                repo_root=repo_root,
            )
            for status_msg in extract_statuses.values():
                sys.stdout.write(f"{status_msg}\n")
        except DownloadError as err:
            err_msg = str(err)
            if "PENDING_OFFICIAL_STATA_DOWNLOAD" in err_msg:
                sys.stderr.write(f"[PENDING_OFFICIAL_STATA_DOWNLOAD] {err_msg}\n")
            else:
                sys.stderr.write(f"[ERROR] Extraction failure: {err_msg}\n")
            return 1
        except Exception as err:
            sys.stderr.write(f"[ERROR] Unexpected failure during extraction: {err}\n")
            return 1
    else:
        if not prep_prov_file.is_file():
            sys.stderr.write(
                f"[ERROR] --schema-only requested but preparation provenance not found at {prep_prov_file}. "
                "Run extraction first.\n"
            )
            return 1
        prep_prov = PreparationProvenance.load(prep_prov_file)

    if parsed.extract_only:
        peak_rss = get_peak_rss_gb()
        elapsed = time.monotonic() - t_start
        sys.stdout.write(
            f"Extraction completed successfully. Peak RSS: {peak_rss:.2f} GB | Elapsed: {elapsed:.2f}s\n"
        )
        return 0

    # Stage 2: Structural Schema Scan
    scan_results: dict[str, StructuralCheckResult] = {}
    all_checks_passed = True

    for panel_key, panel_exp in sorted(exp.panels.items()):
        interim_file = repo_root / panel_exp.interim_relative_path
        if not interim_file.is_file():
            sys.stderr.write(
                f"[PENDING_OFFICIAL_STATA_DOWNLOAD] Interim Stata file not found: {interim_file}. "
                "Official Stata download and extraction pending.\n"
            )
            return 1

        try:
            res = scan_single_panel_schema(
                interim_path=interim_file,
                panel_exp=panel_exp,
                chunk_size=parsed.chunk_size,
                repo_root=repo_root,
                preparation_provenance_path=parsed.preparation_provenance,
                download_provenance_path=parsed.provenance,
            )
            scan_results[panel_key] = res

            sys.stdout.write(
                f"Structural Scan: {panel_exp.puf_id} (Panel {panel_exp.panel_number}) -> "
                f"{res.total_rows:,} rows x {res.total_columns:,} columns | "
                f"ALL5RDS==1: {res.all5rds_1_count:,} | "
                f"Schema Hash: {res.schema_hash[:16]}... | "
                f"Status: {'PASS' if res.checks_passed else 'FAIL'}\n"
            )

            if not res.checks_passed:
                all_checks_passed = False
                for msg in res.validation_messages:
                    sys.stderr.write(f"  [CHECK FAILURE] {panel_exp.puf_id}: {msg}\n")

        except Exception as err:
            sys.stderr.write(f"[ERROR] Failed scanning {panel_exp.puf_id}: {err}\n")
            return 1

    if not all_checks_passed:
        sys.stderr.write("[ERROR] One or more structural checks failed. Aborting snapshot generation.\n")
        return 1

    # Stage 3: Cross-panel Schema Comparison & Snapshot
    if "hc244" in scan_results and "hc252" in scan_results:
        cross_comp = compare_cross_panel_schemas(scan_results["hc244"], scan_results["hc252"])
        sys.stdout.write(
            f"Cross-Panel Schema: {cross_comp.shared_columns_count:,} shared columns | "
            f"HC-244 only: {cross_comp.hc244_only_columns_count:,} | "
            f"HC-252 only: {cross_comp.hc252_only_columns_count:,}\n"
        )
    else:
        sys.stderr.write("[ERROR] Both hc244 and hc252 scan results are required for cross-panel comparison.\n")
        return 1

    try:
        generate_and_save_schema_snapshot(
            prep_prov=prep_prov,
            scan_results=scan_results,
            cross_comparison=cross_comp,
            expectations=exp,
            output_path=parsed.snapshot_output,
            repo_root=repo_root,
        )
        sys.stdout.write(f"Tracked Snapshot: Saved to {parsed.snapshot_output}\n")
    except Exception as err:
        sys.stderr.write(f"[ERROR] Failed generating schema snapshot: {err}\n")
        return 1

    # Stage 4: Holdout Lock Status & Resource Audit
    peak_rss = get_peak_rss_gb()
    elapsed = time.monotonic() - t_start

    sys.stdout.write(
        "Temporal Holdout Lock: Panel 27 (HC-252) LOCKED "
        "(Prohibitions enforced: outcome, protected, and feature distributions protected)\n"
    )
    sys.stdout.write(
        f"Resource Audit: Peak RSS = {peak_rss:.2f} GB (limit: < {MAX_RSS_GB_LIMIT:.1f} GB) | "
        f"Elapsed Time = {elapsed:.2f}s\n"
    )

    if peak_rss >= MAX_RSS_GB_LIMIT:
        sys.stderr.write(f"[ERROR] Peak RSS {peak_rss:.2f} GB exceeded limit {MAX_RSS_GB_LIMIT} GB\n")
        return 1

    sys.stdout.write("Status: SUCCESS\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
