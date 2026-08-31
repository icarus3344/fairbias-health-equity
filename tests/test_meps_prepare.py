"""Offline unit and integration tests for MEPS data preparation and structural schema verification.

All tests operate strictly on synthetic mock data fixtures created within isolated temporary
directories. No external network access or real patient/cohort data is used.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

# Deterministically add repository src directory derived from resolved test file path
_SRC_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from meps_fairness.data.download import (
    CHUNK_SIZE,
    IntegrityError,
    ProvenanceError,
    SecurityError,
)
from meps_fairness.data.prepare import (
    BOUNDED_SCAN_CHUNK_SIZE,
    MAX_RSS_GB_LIMIT,
    MAX_SCAN_CHUNK_SIZE,
    REQUIRED_STRUCTURAL_COLUMNS,
    ArchiveMemberMetadata,
    CrossPanelComparison,
    ExtractionRecord,
    PanelExpectation,
    PreparationProvenance,
    SchemaExpectations,
    StructuralCheckResult,
    _compute_sha256_file,
    _promote_interim_artifact_no_clobber,
    compare_cross_panel_schemas,
    compute_list_hash,
    compute_schema_hash,
    extract_meps_archives,
    extract_single_archive,
    generate_and_save_schema_snapshot,
    get_peak_rss_gb,
    guard_against_prohibited_inspections,
    scan_single_panel_schema,
)


def _create_mock_zip(
    zip_path: pathlib.Path,
    member_name: str,
    content: bytes,
    compress_type: int = zipfile.ZIP_DEFLATED,
) -> tuple[int, str, int, int, int]:
    """Helper to create a zip file and return (archive_size, archive_sha256, uncomp_size, comp_size, crc32)."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=compress_type) as zf:
        zf.writestr(member_name, content)

    archive_size = zip_path.stat().st_size
    archive_sha = _compute_sha256_file(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        info = zf.getinfo(member_name)
        uncomp_size = info.file_size
        comp_size = info.compress_size
        crc32 = info.CRC

    return archive_size, archive_sha, uncomp_size, comp_size, crc32


def _create_synthetic_stata_df(
    n_rows: int = 100,
    panel_val: int = 26,
    all5rds_1_n: int = 90,
) -> pd.DataFrame:
    """Create a synthetic DataFrame matching MEPS structural requirements."""
    data = {
        "DUPERSID": [f"ID{i:05d}" for i in range(n_rows)],
        "DUID": [int(i // 5 + 1000) for i in range(n_rows)],
        "PID": [int(i % 5 + 1) for i in range(n_rows)],
        "PANEL": [int(panel_val)] * n_rows,
        "YEARIND": [1] * n_rows,
        "ALL5RDS": [1 if i < all5rds_1_n else 2 for i in range(n_rows)],
        "LONGWT": [float(1000.0 + i) for i in range(n_rows)],
        "VARSTR": [int(100 + (i % 10)) for i in range(n_rows)],
        "VARPSU": [int((i % 3) + 1) for i in range(n_rows)],
    }
    return pd.DataFrame(data)


def _write_synthetic_stata_file(
    dta_path: pathlib.Path,
    n_rows: int = 100,
    panel_val: int = 26,
    all5rds_1_n: int = 90,
) -> pd.DataFrame:
    """Write a real synthetic Stata (.dta) file using pandas."""
    dta_path.parent.mkdir(parents=True, exist_ok=True)
    df = _create_synthetic_stata_df(n_rows=n_rows, panel_val=panel_val, all5rds_1_n=all5rds_1_n)
    df.to_stata(str(dta_path), write_index=False)
    return df


class TestArchiveExtractionSecurity(unittest.TestCase):
    """Tests for secure archive member validation and extraction attacks."""

    def setUp(self) -> None:
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = pathlib.Path(self.tmp_dir_obj.name)

    def tearDown(self) -> None:
        self.tmp_dir_obj.cleanup()

    def test_missing_source_archive(self) -> None:
        missing_zip = self.tmp_dir / "missing.zip"
        dest_dta = self.tmp_dir / "out.dta"
        with self.assertRaises(IntegrityError) as ctx:
            extract_single_archive(
                archive_path=missing_zip,
                dest_path=dest_dta,
                expected_member_name="h244.dta",
                expected_uncompressed_bytes=100,
                expected_archive_sha256="abc",
                expected_archive_bytes=100,
                expected_compressed_bytes=50,
                expected_crc32=123,
            )
        self.assertIn("Source archive not found", str(ctx.exception))

    def test_archive_size_mismatch(self) -> None:
        zip_path = self.tmp_dir / "test.zip"
        _create_mock_zip(zip_path, "h244.dta", b"data")
        dest_dta = self.tmp_dir / "out.dta"
        with self.assertRaises(IntegrityError) as ctx:
            extract_single_archive(
                archive_path=zip_path,
                dest_path=dest_dta,
                expected_member_name="h244.dta",
                expected_uncompressed_bytes=4,
                expected_archive_sha256="abc",
                expected_archive_bytes=999999,  # Mismatched size
                expected_compressed_bytes=4,
                expected_crc32=123,
            )
        self.assertIn("Source archive size mismatch", str(ctx.exception))

    def test_archive_sha256_mismatch(self) -> None:
        zip_path = self.tmp_dir / "test.zip"
        size, sha, uncomp, comp, crc = _create_mock_zip(zip_path, "h244.dta", b"data")
        dest_dta = self.tmp_dir / "out.dta"
        with self.assertRaises(IntegrityError) as ctx:
            extract_single_archive(
                archive_path=zip_path,
                dest_path=dest_dta,
                expected_member_name="h244.dta",
                expected_uncompressed_bytes=uncomp,
                expected_archive_sha256="0000000000000000000000000000000000000000000000000000000000000000",
                expected_archive_bytes=size,
                expected_compressed_bytes=comp,
                expected_crc32=crc,
            )
        self.assertIn("Source archive SHA-256 mismatch", str(ctx.exception))

    def test_multiple_members_rejected(self) -> None:
        zip_path = self.tmp_dir / "multi.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("h244.dta", b"data1")
            zf.writestr("extra.txt", b"data2")
        size = zip_path.stat().st_size
        sha = _compute_sha256_file(zip_path)
        dest_dta = self.tmp_dir / "out.dta"

        with self.assertRaises(SecurityError) as ctx:
            extract_single_archive(
                archive_path=zip_path,
                dest_path=dest_dta,
                expected_member_name="h244.dta",
                expected_uncompressed_bytes=5,
                expected_archive_sha256=sha,
                expected_archive_bytes=size,
                expected_compressed_bytes=5,
                expected_crc32=123,
            )
        self.assertIn("Archive must contain exactly one member", str(ctx.exception))

    def test_path_traversal_rejected(self) -> None:
        zip_path = self.tmp_dir / "traversal.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../evil.dta", b"malicious")
        size = zip_path.stat().st_size
        sha = _compute_sha256_file(zip_path)
        dest_dta = self.tmp_dir / "out.dta"

        with self.assertRaises(SecurityError) as ctx:
            extract_single_archive(
                archive_path=zip_path,
                dest_path=dest_dta,
                expected_member_name="../evil.dta",
                expected_uncompressed_bytes=9,
                expected_archive_sha256=sha,
                expected_archive_bytes=size,
                expected_compressed_bytes=9,
                expected_crc32=123,
            )
        self.assertIn("forbidden path traversal sequence", str(ctx.exception))

    def test_wrong_extension_rejected(self) -> None:
        zip_path = self.tmp_dir / "wrong_ext.zip"
        size, sha, uncomp, comp, crc = _create_mock_zip(zip_path, "h244.csv", b"csv data")
        dest_dta = self.tmp_dir / "out.dta"

        with self.assertRaises(SecurityError) as ctx:
            extract_single_archive(
                archive_path=zip_path,
                dest_path=dest_dta,
                expected_member_name="h244.csv",
                expected_uncompressed_bytes=uncomp,
                expected_archive_sha256=sha,
                expected_archive_bytes=size,
                expected_compressed_bytes=comp,
                expected_crc32=crc,
                allowed_member_extensions=[".dta"],
            )
        self.assertIn("not in permitted extensions", str(ctx.exception))

    def test_member_name_mismatch_rejected(self) -> None:
        zip_path = self.tmp_dir / "mismatch.zip"
        size, sha, uncomp, comp, crc = _create_mock_zip(zip_path, "other.dta", b"data")
        dest_dta = self.tmp_dir / "out.dta"

        with self.assertRaises(SecurityError) as ctx:
            extract_single_archive(
                archive_path=zip_path,
                dest_path=dest_dta,
                expected_member_name="h244.dta",
                expected_uncompressed_bytes=uncomp,
                expected_archive_sha256=sha,
                expected_archive_bytes=size,
                expected_compressed_bytes=comp,
                expected_crc32=crc,
            )
        self.assertIn("Archive member name mismatch", str(ctx.exception))

    def test_member_uncompressed_size_mismatch(self) -> None:
        zip_path = self.tmp_dir / "uncomp_mismatch.zip"
        size, sha, uncomp, comp, crc = _create_mock_zip(zip_path, "h244.dta", b"hello world")
        dest_dta = self.tmp_dir / "out.dta"

        with self.assertRaises(IntegrityError) as ctx:
            extract_single_archive(
                archive_path=zip_path,
                dest_path=dest_dta,
                expected_member_name="h244.dta",
                expected_uncompressed_bytes=9999,
                expected_archive_sha256=sha,
                expected_archive_bytes=size,
                expected_compressed_bytes=comp,
                expected_crc32=crc,
            )
        self.assertIn("Member uncompressed size mismatch", str(ctx.exception))

    def test_member_crc32_mismatch(self) -> None:
        zip_path = self.tmp_dir / "crc_mismatch.zip"
        content = b"".join(f"REC_{i:04d}_{i * 13}\n".encode("utf-8") for i in range(100))
        size, sha, uncomp, comp, crc = _create_mock_zip(zip_path, "h244.dta", content)
        dest_dta = self.tmp_dir / "out.dta"

        with self.assertRaises(IntegrityError) as ctx:
            extract_single_archive(
                archive_path=zip_path,
                dest_path=dest_dta,
                expected_member_name="h244.dta",
                expected_uncompressed_bytes=uncomp,
                expected_archive_sha256=sha,
                expected_archive_bytes=size,
                expected_compressed_bytes=comp,
                expected_crc32=99999999,
            )
        self.assertIn("Member CRC32 mismatch", str(ctx.exception))

    def test_symlink_member_rejected(self) -> None:
        zip_path = self.tmp_dir / "symlink.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zinfo = zipfile.ZipInfo("h244.dta")
            zinfo.external_attr = 0o120777 << 16
            zf.writestr(zinfo, b"/etc/passwd")

        size = zip_path.stat().st_size
        sha = _compute_sha256_file(zip_path)
        dest_dta = self.tmp_dir / "out.dta"

        with self.assertRaises(SecurityError) as ctx:
            extract_single_archive(
                archive_path=zip_path,
                dest_path=dest_dta,
                expected_member_name="h244.dta",
                expected_uncompressed_bytes=11,
                expected_archive_sha256=sha,
                expected_archive_bytes=size,
                expected_compressed_bytes=11,
                expected_crc32=123,
            )
        self.assertIn("Archive member is a symlink", str(ctx.exception))

    def test_encrypted_member_rejected(self) -> None:
        zip_path = self.tmp_dir / "encrypted.zip"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("h244.dta", b"encrypted_payload")
        data = bytearray(buf.getvalue())
        data[6] |= 0x1
        cd_offset = data.find(b"PK\x01\x02")
        data[cd_offset + 8] |= 0x1
        zip_path.write_bytes(bytes(data))

        with zipfile.ZipFile(zip_path, "r") as zf:
            info = zf.getinfo("h244.dta")
            crc = info.CRC
            uncomp = info.file_size
            comp = info.compress_size

        size = zip_path.stat().st_size
        sha = _compute_sha256_file(zip_path)
        dest_dta = self.tmp_dir / "out.dta"

        with self.assertRaises(SecurityError) as ctx:
            extract_single_archive(
                archive_path=zip_path,
                dest_path=dest_dta,
                expected_member_name="h244.dta",
                expected_uncompressed_bytes=uncomp,
                expected_archive_sha256=sha,
                expected_archive_bytes=size,
                expected_compressed_bytes=comp,
                expected_crc32=crc,
            )
        self.assertIn("Archive member is encrypted", str(ctx.exception))

    def test_decompression_bomb_ratio_rejected(self) -> None:
        zip_path = self.tmp_dir / "bomb.zip"
        content = b"0" * 100000
        size, sha, uncomp, comp, crc = _create_mock_zip(zip_path, "h244.dta", content)
        dest_dta = self.tmp_dir / "out.dta"

        with self.assertRaises(SecurityError) as ctx:
            extract_single_archive(
                archive_path=zip_path,
                dest_path=dest_dta,
                expected_member_name="h244.dta",
                expected_uncompressed_bytes=uncomp,
                expected_archive_sha256=sha,
                expected_archive_bytes=size,
                expected_compressed_bytes=comp,
                expected_crc32=crc,
            )
        self.assertIn("Compression ratio", str(ctx.exception))


class TestExtractMechanicsAndAtomicity(unittest.TestCase):
    """Tests for atomic extraction mechanics, promotion, and idempotency."""

    def setUp(self) -> None:
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = pathlib.Path(self.tmp_dir_obj.name)

    def tearDown(self) -> None:
        self.tmp_dir_obj.cleanup()

    def test_successful_extraction_and_promotion(self) -> None:
        zip_path = self.tmp_dir / "h244dta.zip"
        content = b"".join(f"STATA_RECORD_FIELD_{i:04d}_VALUE_{i * 37}\n".encode("utf-8") for i in range(100))
        size, sha, uncomp, comp, crc = _create_mock_zip(zip_path, "h244.dta", content)
        dest_dta = self.tmp_dir / "interim" / "h244" / "h244.dta"

        extracted_size, extracted_sha, member_meta = extract_single_archive(
            archive_path=zip_path,
            dest_path=dest_dta,
            expected_member_name="h244.dta",
            expected_uncompressed_bytes=uncomp,
            expected_archive_sha256=sha,
            expected_archive_bytes=size,
            expected_compressed_bytes=comp,
            expected_crc32=crc,
        )

        self.assertEqual(extracted_size, len(content))
        self.assertTrue(dest_dta.is_file())
        self.assertEqual(dest_dta.read_bytes(), content)
        self.assertFalse(dest_dta.with_name("h244.dta.part").exists())
        self.assertEqual(member_meta.member_name, "h244.dta")

    def test_no_clobber_on_existing_destination(self) -> None:
        dest_dta = self.tmp_dir / "target.dta"
        dest_dta.write_bytes(b"pre-existing data")
        part_dta = self.tmp_dir / "target.dta.part"
        part_dta.write_bytes(b"new data")

        with self.assertRaises(IntegrityError) as ctx:
            _promote_interim_artifact_no_clobber(part_dta, dest_dta)
        self.assertIn("Destination file already exists", str(ctx.exception))
        self.assertEqual(dest_dta.read_bytes(), b"pre-existing data")
        self.assertFalse(part_dta.exists())

    def test_full_extract_meps_archives_workflow_and_idempotency(self) -> None:
        raw_root = self.tmp_dir / "data" / "raw" / "meps"
        interim_root = self.tmp_dir / "data" / "interim" / "meps"
        prep_prov_path = interim_root / "preparation_provenance.json"

        # HC-244 Stata
        h244_content = b"".join(f"HC244_STATA_{i:04d}_{i * 17}\n".encode("utf-8") for i in range(100))
        h244_zip = raw_root / "h244" / "h244dta.zip"
        s244, sha244, u244, c244, crc244 = _create_mock_zip(h244_zip, "h244.dta", h244_content)

        # HC-252 Stata
        h252_content = b"".join(f"HC252_STATA_{i:04d}_{i * 31}\n".encode("utf-8") for i in range(100))
        h252_zip = raw_root / "h252" / "h252dta.zip"
        s252, sha252, u252, c252, crc252 = _create_mock_zip(h252_zip, "h252.dta", h252_content)

        local_prov_dict = {
            "schema_version": "1.0.0",
            "artifacts": {
                "hc244_stata_archive": {
                    "puf_id": "HC-244",
                    "artifact_type": "data_archives",
                    "local_relative_path": "h244/h244dta.zip",
                    "byte_size": s244,
                    "sha256": sha244,
                    "archive_member_metadata": {
                        "member_name": "h244.dta",
                        "uncompressed_size": u244,
                        "compressed_size": c244,
                        "crc32": crc244,
                    },
                },
                "hc252_stata_archive": {
                    "puf_id": "HC-252",
                    "artifact_type": "data_archives",
                    "local_relative_path": "h252/h252dta.zip",
                    "byte_size": s252,
                    "sha256": sha252,
                    "archive_member_metadata": {
                        "member_name": "h252.dta",
                        "uncompressed_size": u252,
                        "compressed_size": c252,
                        "crc32": crc252,
                    },
                },
            },
        }
        prov_file = self.tmp_dir / "docs" / "data" / "MEPS_LOCAL_PROVENANCE.json"
        prov_file.parent.mkdir(parents=True, exist_ok=True)
        prov_file.write_text(json.dumps(local_prov_dict, indent=2))

        expectations_dict = {
            "schema_version": "1.0.0",
            "computational_constraints": {"bounded_chunk_size": 512, "max_rss_gb": 16.0},
            "required_columns": list(REQUIRED_STRUCTURAL_COLUMNS),
            "panels": {
                "hc244": {
                    "puf_id": "HC-244",
                    "panel_number": 26,
                    "role": "development_cohort",
                    "expected_rows": 100,
                    "expected_columns": 9,
                    "expected_all5rds_1_count": 90,
                    "expected_panel_value": 26,
                    "source_archive_relative_path": "data/raw/meps/h244/h244dta.zip",
                    "expected_member_name": "h244.dta",
                    "interim_relative_path": "data/interim/meps/h244/h244.dta",
                    "expected_uncompressed_bytes": None,
                },
                "hc252": {
                    "puf_id": "HC-252",
                    "panel_number": 27,
                    "role": "temporal_holdout_cohort",
                    "expected_rows": 100,
                    "expected_columns": 9,
                    "expected_all5rds_1_count": 90,
                    "expected_panel_value": 27,
                    "source_archive_relative_path": "data/raw/meps/h252/h252dta.zip",
                    "expected_member_name": "h252.dta",
                    "interim_relative_path": "data/interim/meps/h252/h252.dta",
                    "expected_uncompressed_bytes": None,
                },
            },
        }
        exp_file = self.tmp_dir / "configs" / "meps_schema_expectations.json"
        exp_file.parent.mkdir(parents=True, exist_ok=True)
        exp_file.write_text(json.dumps(expectations_dict, indent=2))

        # First extraction run
        prep_prov, statuses = extract_meps_archives(
            expectations_path=exp_file,
            provenance_path=prov_file,
            raw_root=raw_root,
            interim_root=interim_root,
            preparation_provenance_path=prep_prov_path,
            repo_root=self.tmp_dir,
        )

        self.assertEqual(prep_prov.total_extracted_files, 2)
        self.assertTrue((interim_root / "h244" / "h244.dta").is_file())
        self.assertTrue((interim_root / "h252" / "h252.dta").is_file())
        self.assertIn("EXTRACTED", statuses["hc244"])
        self.assertIn("EXTRACTED", statuses["hc252"])

        # Record provenance mtime before rerun
        prov_mtime_before = prep_prov_path.stat().st_mtime_ns

        # Second extraction run (idempotent skip)
        prep_prov2, statuses2 = extract_meps_archives(
            expectations_path=exp_file,
            provenance_path=prov_file,
            raw_root=raw_root,
            interim_root=interim_root,
            preparation_provenance_path=prep_prov_path,
            repo_root=self.tmp_dir,
        )

        self.assertIn("SKIPPED", statuses2["hc244"])
        self.assertIn("SKIPPED", statuses2["hc252"])
        self.assertEqual(prep_prov_path.stat().st_mtime_ns, prov_mtime_before)

    def test_pending_download_error_when_raw_archives_missing(self) -> None:
        exp_file = self.tmp_dir / "configs" / "meps_schema_expectations.json"
        exp_file.parent.mkdir(parents=True, exist_ok=True)
        exp_dict = {
            "schema_version": "1.0.0",
            "panels": {
                "hc244": {
                    "puf_id": "HC-244",
                    "panel_number": 26,
                    "expected_rows": 10,
                    "expected_columns": 9,
                    "expected_all5rds_1_count": 8,
                    "expected_panel_value": 26,
                    "source_archive_relative_path": "data/raw/meps/h244/h244dta.zip",
                    "expected_member_name": "h244.dta",
                    "interim_relative_path": "data/interim/meps/h244/h244.dta",
                }
            },
        }
        exp_file.write_text(json.dumps(exp_dict, indent=2))

        with self.assertRaises(ProvenanceError) as ctx:
            extract_meps_archives(
                expectations_path=exp_file,
                provenance_path="nonexistent_prov.json",
                repo_root=self.tmp_dir,
            )
        self.assertIn("PENDING_OFFICIAL_STATA_DOWNLOAD", str(ctx.exception))


class TestStructuralSchemaScan(unittest.TestCase):
    """Tests for chunked reading via pandas.read_stata, structural counters, and validation rules."""

    def setUp(self) -> None:
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = pathlib.Path(self.tmp_dir_obj.name)

    def tearDown(self) -> None:
        self.tmp_dir_obj.cleanup()

    def test_guard_against_prohibited_inspections(self) -> None:
        for prohibited in ["INSCOV", "unins", "RACETHX", "Sex", "POVCAT", "AGE1X"]:
            with self.assertRaises(SecurityError):
                guard_against_prohibited_inspections(prohibited)

        for allowed in ["DUPERSID", "DUID", "PID", "PANEL", "YEARIND", "ALL5RDS", "LONGWT", "VARSTR", "VARPSU"]:
            guard_against_prohibited_inspections(allowed)

    def test_chunk_size_limits(self) -> None:
        panel_exp = PanelExpectation(
            puf_id="HC-244",
            panel_number=26,
            role="development_cohort",
            expected_rows=100,
            expected_columns=9,
            expected_all5rds_1_count=90,
            expected_panel_value=26,
            source_archive_relative_path="",
            expected_member_name="h244.dta",
            interim_relative_path="",
        )
        fake_file = self.tmp_dir / "dummy.dta"
        _write_synthetic_stata_file(fake_file, n_rows=10)

        with self.assertRaises(SecurityError):
            scan_single_panel_schema(fake_file, panel_exp, chunk_size=1024)

        with self.assertRaises(SecurityError):
            scan_single_panel_schema(fake_file, panel_exp, chunk_size=0)

    def test_scan_aggregations_and_counters_with_real_stata_file(self) -> None:
        dta_path = self.tmp_dir / "h244.dta"
        _write_synthetic_stata_file(dta_path, n_rows=200, panel_val=26, all5rds_1_n=180)

        panel_exp = PanelExpectation(
            puf_id="HC-244",
            panel_number=26,
            role="development_cohort",
            expected_rows=200,
            expected_columns=9,
            expected_all5rds_1_count=180,
            expected_panel_value=26,
            source_archive_relative_path="",
            expected_member_name="h244.dta",
            interim_relative_path="",
        )

        res = scan_single_panel_schema(dta_path, panel_exp, chunk_size=50)

        self.assertEqual(res.total_rows, 200)
        self.assertEqual(res.total_columns, 9)
        self.assertEqual(res.all5rds_1_count, 180)
        self.assertEqual(res.all5rds_other_count, 20)
        self.assertEqual(res.invalid_longwt_count, 0)
        self.assertEqual(res.missing_varstr_count, 0)
        self.assertEqual(res.missing_varpsu_count, 0)
        self.assertEqual(res.duplicate_dupersid_count, 0)
        self.assertEqual(res.missing_dupersid_count, 0)
        self.assertEqual(res.unexpected_panel_count, 0)
        self.assertTrue(res.required_columns_present)
        self.assertTrue(res.checks_passed)
        self.assertIn("VALIDATED_ALL_RECORDS_YEARIND_1", res.yearind_check_status)

    def test_scan_detects_structural_failures_in_stata_file(self) -> None:
        dta_path = self.tmp_dir / "h244_bad.dta"
        df = _create_synthetic_stata_df(n_rows=50, panel_val=26, all5rds_1_n=40)
        df.loc[0, "LONGWT"] = -5.0
        df.loc[1, "VARSTR"] = np.nan
        df.loc[2, "DUPERSID"] = df.loc[3, "DUPERSID"]
        df.loc[4, "PANEL"] = 99
        df.to_stata(str(dta_path), write_index=False)

        panel_exp = PanelExpectation(
            puf_id="HC-244",
            panel_number=26,
            role="development_cohort",
            expected_rows=50,
            expected_columns=9,
            expected_all5rds_1_count=40,
            expected_panel_value=26,
            source_archive_relative_path="",
            expected_member_name="h244.dta",
            interim_relative_path="",
        )

        res = scan_single_panel_schema(dta_path, panel_exp, chunk_size=20)

        self.assertFalse(res.checks_passed)
        self.assertEqual(res.invalid_longwt_count, 1)
        self.assertEqual(res.missing_varstr_count, 1)
        self.assertEqual(res.duplicate_dupersid_count, 1)
        self.assertEqual(res.unexpected_panel_count, 1)

    def test_missing_each_required_column(self) -> None:
        for missing_col in REQUIRED_STRUCTURAL_COLUMNS:
            dta_path = self.tmp_dir / f"h244_missing_{missing_col}.dta"
            df = _create_synthetic_stata_df(n_rows=20, panel_val=26, all5rds_1_n=20)
            df = df.drop(columns=[missing_col])
            df.to_stata(str(dta_path), write_index=False)

            panel_exp = PanelExpectation(
                puf_id="HC-244",
                panel_number=26,
                role="development_cohort",
                expected_rows=20,
                expected_columns=len(df.columns),
                expected_all5rds_1_count=20,
                expected_panel_value=26,
                source_archive_relative_path="",
                expected_member_name="h244.dta",
                interim_relative_path="",
            )

            res = scan_single_panel_schema(dta_path, panel_exp, chunk_size=10)
            self.assertFalse(res.required_columns_present)
            self.assertIn(missing_col, res.required_columns_missing)
            self.assertFalse(res.checks_passed)

    def test_active_scanner_uses_read_stata_and_no_read_sas(self) -> None:
        """Assert that scan_single_panel_schema uses pd.read_stata and not pd.read_sas."""
        dta_path = self.tmp_dir / "h244_check_calls.dta"
        _write_synthetic_stata_file(dta_path, n_rows=10, panel_val=26, all5rds_1_n=10)

        panel_exp = PanelExpectation(
            puf_id="HC-244",
            panel_number=26,
            role="development_cohort",
            expected_rows=10,
            expected_columns=9,
            expected_all5rds_1_count=10,
            expected_panel_value=26,
            source_archive_relative_path="",
            expected_member_name="h244.dta",
            interim_relative_path="",
        )

        read_stata_called = False
        orig_read_stata = pd.read_stata

        def mock_read_stata(*args, **kwargs):
            nonlocal read_stata_called
            read_stata_called = True
            # Verify convert_categoricals=False and chunksize <= 512
            self.assertFalse(kwargs.get("convert_categoricals", True))
            self.assertLessEqual(kwargs.get("chunksize", 999), 512)
            return orig_read_stata(*args, **kwargs)

        def mock_read_sas(*args, **kwargs):
            raise AssertionError("Active schema scanning must NOT call pd.read_sas!")

        orig_read_sas = getattr(pd, "read_sas", None)
        try:
            pd.read_stata = mock_read_stata
            pd.read_sas = mock_read_sas

            res = scan_single_panel_schema(dta_path, panel_exp, chunk_size=5)
            self.assertTrue(read_stata_called)
            self.assertTrue(res.checks_passed)
        finally:
            pd.read_stata = orig_read_stata
            if orig_read_sas is not None:
                pd.read_sas = orig_read_sas


class TestCrossPanelAndSnapshot(unittest.TestCase):
    """Tests for cross-panel schema comparison and deterministic snapshot persistence."""

    def setUp(self) -> None:
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = pathlib.Path(self.tmp_dir_obj.name)

    def tearDown(self) -> None:
        self.tmp_dir_obj.cleanup()

    def test_cross_panel_comparison_and_hashes(self) -> None:
        cols_244 = ("DUPERSID", "DUID", "PID", "PANEL", "YEARIND", "ALL5RDS", "LONGWT", "VARSTR", "VARPSU", "COL_A", "COL_B")
        cols_252 = ("DUPERSID", "DUID", "PID", "PANEL", "YEARIND", "ALL5RDS", "LONGWT", "VARSTR", "VARPSU", "COL_B", "COL_C")

        dtypes_244 = {c: "float64" for c in cols_244}
        dtypes_252 = {c: "float64" for c in cols_252}

        res_244 = StructuralCheckResult(
            puf_id="HC-244",
            panel_number=26,
            total_rows=10,
            total_columns=len(cols_244),
            ordered_columns=cols_244,
            column_dtypes=dtypes_244,
            schema_hash=compute_schema_hash(cols_244, dtypes_244),
            duplicate_columns_count=0,
            required_columns_present=True,
            required_columns_missing=(),
            all5rds_1_count=10,
            all5rds_other_count=0,
            invalid_longwt_count=0,
            missing_varstr_count=0,
            missing_varpsu_count=0,
            duplicate_dupersid_count=0,
            missing_dupersid_count=0,
            unexpected_panel_count=0,
            yearind_check_status="VALIDATED_ALL_RECORDS_YEARIND_1",
            yearind_1_count=10,
            yearind_other_count=0,
            checks_passed=True,
            validation_messages=("PASS",),
        )

        res_252 = StructuralCheckResult(
            puf_id="HC-252",
            panel_number=27,
            total_rows=12,
            total_columns=len(cols_252),
            ordered_columns=cols_252,
            column_dtypes=dtypes_252,
            schema_hash=compute_schema_hash(cols_252, dtypes_252),
            duplicate_columns_count=0,
            required_columns_present=True,
            required_columns_missing=(),
            all5rds_1_count=12,
            all5rds_other_count=0,
            invalid_longwt_count=0,
            missing_varstr_count=0,
            missing_varpsu_count=0,
            duplicate_dupersid_count=0,
            missing_dupersid_count=0,
            unexpected_panel_count=0,
            yearind_check_status="VALIDATED_ALL_RECORDS_YEARIND_1",
            yearind_1_count=12,
            yearind_other_count=0,
            checks_passed=True,
            validation_messages=("PASS",),
        )

        comp = compare_cross_panel_schemas(res_244, res_252)

        self.assertEqual(comp.shared_columns_count, 10)
        self.assertIn("COL_B", comp.shared_columns)
        self.assertEqual(comp.hc244_only_columns, ("COL_A",))
        self.assertEqual(comp.hc252_only_columns, ("COL_C",))
        self.assertEqual(comp.hc244_only_columns_count, 1)
        self.assertEqual(comp.hc252_only_columns_count, 1)

        hash1 = compute_list_hash(comp.shared_columns)
        hash2 = compute_list_hash(comp.shared_columns)
        self.assertEqual(hash1, hash2)

    def test_schema_snapshot_generation_and_roundtrip(self) -> None:
        prep_prov = PreparationProvenance.create_empty()
        rec_meta = ArchiveMemberMetadata("h244.dta", 100, 50, 123)
        rec = ExtractionRecord(
            puf_id="HC-244",
            panel_number=26,
            source_archive_relative_path="data/raw/meps/h244/h244dta.zip",
            source_archive_sha256="abc",
            member_name="h244.dta",
            extracted_relative_path="data/interim/meps/h244/h244.dta",
            extracted_byte_size=100,
            extracted_sha256="def",
            extraction_timestamp_utc="2026-08-28T00:00:00+00:00",
            archive_member_metadata=rec_meta,
        )
        prep_prov.add_record("hc244", rec)

        cols = ("DUPERSID", "DUID", "PID", "PANEL", "YEARIND", "ALL5RDS", "LONGWT", "VARSTR", "VARPSU")
        dtypes = {c: "float64" for c in cols}
        res_244 = StructuralCheckResult(
            puf_id="HC-244",
            panel_number=26,
            total_rows=10,
            total_columns=len(cols),
            ordered_columns=cols,
            column_dtypes=dtypes,
            schema_hash=compute_schema_hash(cols, dtypes),
            duplicate_columns_count=0,
            required_columns_present=True,
            required_columns_missing=(),
            all5rds_1_count=10,
            all5rds_other_count=0,
            invalid_longwt_count=0,
            missing_varstr_count=0,
            missing_varpsu_count=0,
            duplicate_dupersid_count=0,
            missing_dupersid_count=0,
            unexpected_panel_count=0,
            yearind_check_status="VALIDATED_ALL_RECORDS_YEARIND_1",
            yearind_1_count=10,
            yearind_other_count=0,
            checks_passed=True,
            validation_messages=("PASS",),
        )

        comp = CrossPanelComparison(
            shared_columns_count=len(cols),
            shared_columns_hash=compute_list_hash(cols),
            shared_columns=cols,
            hc244_only_columns_count=0,
            hc244_only_columns_hash=compute_list_hash(()),
            hc244_only_columns=(),
            hc252_only_columns_count=0,
            hc252_only_columns_hash=compute_list_hash(()),
            hc252_only_columns=(),
        )

        exp = SchemaExpectations(
            schema_version="1.0.0",
            bounded_chunk_size=512,
            max_rss_gb=16.0,
            required_columns=cols,
            panels={},
            holdout_lock={"status": "locked"},
        )

        snapshot_path = self.tmp_dir / "MEPS_SCHEMA_SNAPSHOT.json"
        snap_dict = generate_and_save_schema_snapshot(
            prep_prov=prep_prov,
            scan_results={"hc244": res_244},
            cross_comparison=comp,
            expectations=exp,
            output_path=snapshot_path,
            repo_root=self.tmp_dir,
        )

        self.assertTrue(snapshot_path.is_file())
        with open(snapshot_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded["schema_version"], "1.0.0")
        self.assertIn("hc244", loaded["panels"])
        self.assertEqual(loaded["temporal_holdout_lock"]["status"], "LOCKED")


class TestExpectationsAndProvenanceClasses(unittest.TestCase):
    """Tests for SchemaExpectations and PreparationProvenance loading and edge cases."""

    def setUp(self) -> None:
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = pathlib.Path(self.tmp_dir_obj.name)

    def tearDown(self) -> None:
        self.tmp_dir_obj.cleanup()

    def test_missing_expectations_file(self) -> None:
        with self.assertRaises(IntegrityError):
            SchemaExpectations.load(self.tmp_dir / "nonexistent.json")

    def test_missing_provenance_file(self) -> None:
        with self.assertRaises(ProvenanceError):
            PreparationProvenance.load(self.tmp_dir / "nonexistent.json")

    def test_corrupt_provenance_file(self) -> None:
        corrupt = self.tmp_dir / "corrupt.json"
        corrupt.write_text("{not-valid-json")
        with self.assertRaises(ProvenanceError):
            PreparationProvenance.load(corrupt)

    def test_peak_rss_sanity(self) -> None:
        rss = get_peak_rss_gb()
        self.assertGreater(rss, 0.0)
        self.assertLess(rss, MAX_RSS_GB_LIMIT)


class TestCLIScriptSubprocess(unittest.TestCase):
    """Subprocess integration tests for CLI script execution from different working directories."""

    def setUp(self) -> None:
        self.repo_root = pathlib.Path(__file__).resolve().parents[1]
        self.py311_bin = sys.executable

    def test_cli_help_from_repo_root(self) -> None:
        proc = subprocess.run(
            [self.py311_bin, str(self.repo_root / "scripts" / "prepare_meps.py"), "--help"],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("prepare_meps.py", proc.stdout)

    def test_cli_pending_download_status_from_repo_root(self) -> None:
        """Verify CLI fails cleanly with PENDING_OFFICIAL_STATA_DOWNLOAD when raw Stata zip is absent."""
        with tempfile.NamedTemporaryFile("w", suffix=".json") as exp_tmp:
            fake_exp = {
                "schema_version": "1.0.0",
                "panels": {
                    "hc244": {
                        "puf_id": "HC-244",
                        "panel_number": 26,
                        "expected_rows": 10,
                        "expected_columns": 9,
                        "expected_all5rds_1_count": 8,
                        "expected_panel_value": 26,
                        "source_archive_relative_path": "nonexistent/h244dta.zip",
                        "expected_member_name": "h244.dta",
                        "interim_relative_path": "nonexistent/h244.dta",
                    }
                },
            }
            json.dump(fake_exp, exp_tmp)
            exp_tmp.flush()
            proc = subprocess.run(
                [self.py311_bin, str(self.repo_root / "scripts" / "prepare_meps.py"), "--expectations", exp_tmp.name],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 1)
            self.assertIn("PENDING_OFFICIAL_STATA_DOWNLOAD", proc.stderr)


if __name__ == "__main__":
    unittest.main()
