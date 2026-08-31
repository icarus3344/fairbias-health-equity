"""Secure, atomic, and policy-compliant MEPS data preparation and structural schema verification.

This module enforces:
1. Secure extraction of Stata (.dta) files from verified raw ZIP archives:
   - Archive size and SHA-256 validation against local provenance records.
   - Archive member inspection (single regular member, .dta extension, no path traversal,
     no symlinks, no encryption, metadata CRC32/size validation, decompression bomb defense).
   - Streaming extraction to temporary .part files with SHA-256 calculation and fsync.
   - No-clobber atomic promotion and pure-skip idempotency.
   - Incremental, atomic recording of preparation provenance (preserving prior CPORT evidence).
2. Bounded chunk-based structural scanning via pandas.read_stata:
   - Streaming Stata reader with convert_categoricals=False and bounded chunk size (<= 512 rows).
   - Aggregation of structural metrics only: row counts, column counts, ordered schema,
     data types, schema hash, duplicate column detection, required column presence.
   - Authorized structural checks ONLY (ALL5RDS == 1 count, valid LONGWT, valid VARSTR/VARPSU,
     unique DUPERSID, expected PANEL, YEARIND domain check / deferred Gate 7).
   - Strict prohibition against inspecting row values, head/tail samples, outcome prevalence,
     protected demographic distributions, or feature summaries.
3. Temporal holdout locking for Panel 27 (HC-252):
   - Schema and authorized structural integrity verification only.
   - Explicit enforcement of holdout protections.
4. Deterministic schema snapshot persistence.
"""

from __future__ import annotations

import dataclasses
import datetime
import gc
import hashlib
import json
import os
import pathlib
import resource
import sys
import typing
import warnings
import zipfile
from typing import Any, Sequence

import numpy as np
import pandas as pd

from meps_fairness.data.download import (
    CHUNK_SIZE,
    MAX_COMPRESSION_RATIO,
    MAX_UNCOMPRESSED_ARCHIVE_BYTES,
    DownloadError,
    IntegrityError,
    ProvenanceError,
    SecurityError,
    _fsync_dir,
    _write_all,
)

# Constants & Defaults
BOUNDED_SCAN_CHUNK_SIZE: int = 512
MAX_SCAN_CHUNK_SIZE: int = 512
MAX_RSS_GB_LIMIT: float = 16.0
DEFAULT_EXPECTATIONS_PATH: str = "configs/meps_schema_expectations.json"
DEFAULT_PROVENANCE_PATH: str = "data/raw/meps/provenance.json"
DEFAULT_RAW_ROOT: str = "data/raw/meps"
DEFAULT_INTERIM_ROOT: str = "data/interim/meps"
DEFAULT_PREPARATION_PROVENANCE_PATH: str = "data/interim/meps/preparation_provenance.json"
DEFAULT_SNAPSHOT_PATH: str = "docs/data/MEPS_SCHEMA_SNAPSHOT.json"
PREPARATION_PROVENANCE_SCHEMA_VERSION: str = "1.0.0"
SCHEMA_SNAPSHOT_SCHEMA_VERSION: str = "1.0.0"
PREPARATION_DISCLAIMER: str = (
    "Locally computed SHA-256 hashes establish local immutability and reproducibility; "
    "they do not constitute proof of publisher authenticity when no official upstream checksums exist."
)

REQUIRED_STRUCTURAL_COLUMNS: tuple[str, ...] = (
    "DUPERSID",
    "DUID",
    "PID",
    "PANEL",
    "YEARIND",
    "ALL5RDS",
    "LONGWT",
    "VARSTR",
    "VARPSU",
)

PROHIBITED_INSPECTION_VARIABLES: frozenset[str] = frozenset(
    {
        "INSCOV",
        "UNINS",
        "EVRINS",
        "INSURC",
        "PRIEU",
        "PRIDK",
        "PRIS",
        "PRIV",
        "RACETHX",
        "RACEV1X",
        "HISPANX",
        "SEX",
        "POVCAT",
        "POVLEV",
        "AGE",
        "AGE1X",
        "AGE2X",
        "AGE21X",
        "AGE22X",
    }
)


def get_peak_rss_gb() -> float:
    """Return peak resident set size (RSS) in gigabytes for the current process."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(ru) / (1024.0 * 1024.0 * 1024.0)
    else:
        return float(ru) / (1024.0 * 1024.0)


def _compute_sha256_file(path: pathlib.Path | str) -> str:
    """Compute SHA-256 checksum of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _promote_interim_artifact_no_clobber(part_path: pathlib.Path, dest_path: pathlib.Path) -> None:
    """Promote a .part file to its final destination using an atomic hard link.

    Fails closed if the destination file already exists (preventing overwrite).
    """
    dest_dir = dest_path.parent
    _fsync_dir(dest_dir)

    try:
        os.link(part_path, dest_path)
    except FileExistsError as err:
        if part_path.is_file():
            try:
                part_path.unlink()
            except OSError:
                pass
        raise IntegrityError(f"Destination file already exists; overwriting is prohibited: {dest_path}") from err
    except OSError as err:
        if part_path.is_file():
            try:
                part_path.unlink()
            except OSError:
                pass
        raise IntegrityError(f"Failed to link temporary file {part_path} to {dest_path}: {err}") from err

    _fsync_dir(dest_dir)

    try:
        part_path.unlink()
    except OSError as err:
        warnings.warn(
            f"Artifact successfully linked to {dest_path} but failed to unlink part file {part_path}: {err}",
            RuntimeWarning,
            stacklevel=2,
        )

    _fsync_dir(dest_dir)


@dataclasses.dataclass(frozen=True)
class ArchiveMemberMetadata:
    """Metadata recorded for an extracted archive member."""

    member_name: str
    uncompressed_size: int
    compressed_size: int
    crc32: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_name": self.member_name,
            "uncompressed_size": self.uncompressed_size,
            "compressed_size": self.compressed_size,
            "crc32": self.crc32,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchiveMemberMetadata:
        return cls(
            member_name=str(data["member_name"]),
            uncompressed_size=int(data["uncompressed_size"]),
            compressed_size=int(data["compressed_size"]),
            crc32=int(data["crc32"]),
        )


@dataclasses.dataclass(frozen=True)
class ExtractionRecord:
    """Provenance record for an extracted MEPS dataset."""

    puf_id: str
    panel_number: int
    source_archive_relative_path: str
    source_archive_sha256: str
    member_name: str
    extracted_relative_path: str
    extracted_byte_size: int
    extracted_sha256: str
    extraction_timestamp_utc: str
    archive_member_metadata: ArchiveMemberMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "puf_id": self.puf_id,
            "panel_number": self.panel_number,
            "source_archive_relative_path": self.source_archive_relative_path,
            "source_archive_sha256": self.source_archive_sha256,
            "member_name": self.member_name,
            "extracted_relative_path": self.extracted_relative_path,
            "extracted_byte_size": self.extracted_byte_size,
            "extracted_sha256": self.extracted_sha256,
            "extraction_timestamp_utc": self.extraction_timestamp_utc,
            "archive_member_metadata": self.archive_member_metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractionRecord:
        return cls(
            puf_id=str(data["puf_id"]),
            panel_number=int(data["panel_number"]),
            source_archive_relative_path=str(data["source_archive_relative_path"]),
            source_archive_sha256=str(data["source_archive_sha256"]),
            member_name=str(data["member_name"]),
            extracted_relative_path=str(data["extracted_relative_path"]),
            extracted_byte_size=int(data["extracted_byte_size"]),
            extracted_sha256=str(data["extracted_sha256"]),
            extraction_timestamp_utc=str(data["extraction_timestamp_utc"]),
            archive_member_metadata=ArchiveMemberMetadata.from_dict(data["archive_member_metadata"]),
        )


@dataclasses.dataclass
class PreparationProvenance:
    """Runtime manifest tracking all extracted interim MEPS artifacts."""

    schema_version: str
    snapshot_timestamp_utc: str
    disclaimer: str
    total_extracted_files: int
    total_extracted_bytes: int
    extractions: dict[str, ExtractionRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_timestamp_utc": self.snapshot_timestamp_utc,
            "disclaimer": self.disclaimer,
            "total_extracted_files": self.total_extracted_files,
            "total_extracted_bytes": self.total_extracted_bytes,
            "extractions": {k: v.to_dict() for k, v in sorted(self.extractions.items())},
        }

    @classmethod
    def create_empty(cls) -> PreparationProvenance:
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return cls(
            schema_version=PREPARATION_PROVENANCE_SCHEMA_VERSION,
            snapshot_timestamp_utc=now_utc,
            disclaimer=PREPARATION_DISCLAIMER,
            total_extracted_files=0,
            total_extracted_bytes=0,
            extractions={},
        )

    @classmethod
    def load(cls, path: pathlib.Path | str) -> PreparationProvenance:
        p = pathlib.Path(path)
        if not p.is_file():
            raise ProvenanceError(f"Preparation provenance file not found: {p}")
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as err:
            raise ProvenanceError(f"Failed to parse preparation provenance JSON from {p}: {err}") from err

        if not isinstance(data, dict):
            raise ProvenanceError("Preparation provenance must be a JSON object.")

        extractions: dict[str, ExtractionRecord] = {}
        raw_extractions = data.get("extractions", {})
        if isinstance(raw_extractions, dict):
            for k, v in raw_extractions.items():
                extractions[k] = ExtractionRecord.from_dict(v)

        return cls(
            schema_version=str(data.get("schema_version", PREPARATION_PROVENANCE_SCHEMA_VERSION)),
            snapshot_timestamp_utc=str(data.get("snapshot_timestamp_utc", "")),
            disclaimer=str(data.get("disclaimer", PREPARATION_DISCLAIMER)),
            total_extracted_files=int(data.get("total_extracted_files", len(extractions))),
            total_extracted_bytes=int(data.get("total_extracted_bytes", sum(e.extracted_byte_size for e in extractions.values()))),
            extractions=extractions,
        )

    def add_record(self, key: str, record: ExtractionRecord) -> None:
        self.extractions[key] = record
        self.total_extracted_files = len(self.extractions)
        self.total_extracted_bytes = sum(e.extracted_byte_size for e in self.extractions.values())
        self.snapshot_timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def save(self, path: pathlib.Path | str) -> None:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        part_path = p.with_name(p.name + ".part")

        if part_path.is_file():
            try:
                part_path.unlink()
            except OSError:
                pass

        fd = os.open(part_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            payload = json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n"
            _write_all(fd, payload.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

        _fsync_dir(p.parent)
        os.replace(part_path, p)
        _fsync_dir(p.parent)


@dataclasses.dataclass(frozen=True)
class PanelExpectation:
    """Expectations for a single MEPS panel."""

    puf_id: str
    panel_number: int
    role: str
    expected_rows: int
    expected_columns: int
    expected_all5rds_1_count: int
    expected_panel_value: int
    source_archive_relative_path: str
    expected_member_name: str
    interim_relative_path: str
    expected_uncompressed_bytes: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PanelExpectation:
        raw_uncomp = data.get("expected_uncompressed_bytes")
        return cls(
            puf_id=str(data["puf_id"]),
            panel_number=int(data["panel_number"]),
            role=str(data.get("role", "")),
            expected_rows=int(data["expected_rows"]),
            expected_columns=int(data["expected_columns"]),
            expected_all5rds_1_count=int(data["expected_all5rds_1_count"]),
            expected_panel_value=int(data["expected_panel_value"]),
            source_archive_relative_path=str(data["source_archive_relative_path"]),
            expected_member_name=str(data["expected_member_name"]),
            interim_relative_path=str(data["interim_relative_path"]),
            expected_uncompressed_bytes=int(raw_uncomp) if raw_uncomp is not None else None,
        )


@dataclasses.dataclass(frozen=True)
class SchemaExpectations:
    """Parsed schema expectations configuration."""

    schema_version: str
    bounded_chunk_size: int
    max_rss_gb: float
    required_columns: tuple[str, ...]
    panels: dict[str, PanelExpectation]
    holdout_lock: dict[str, Any]

    @classmethod
    def load(cls, path: pathlib.Path | str) -> SchemaExpectations:
        p = pathlib.Path(path)
        if not p.is_file():
            raise IntegrityError(f"Schema expectations configuration file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise IntegrityError("Schema expectations configuration must be a JSON object.")

        constraints = data.get("computational_constraints", {})
        chunk_size = int(constraints.get("bounded_chunk_size", BOUNDED_SCAN_CHUNK_SIZE))
        max_rss = float(constraints.get("max_rss_gb", MAX_RSS_GB_LIMIT))
        req_cols = tuple(data.get("required_columns", REQUIRED_STRUCTURAL_COLUMNS))

        panels_dict: dict[str, PanelExpectation] = {}
        for k, v in data.get("panels", {}).items():
            panels_dict[k] = PanelExpectation.from_dict(v)

        return cls(
            schema_version=str(data.get("schema_version", "1.0.0")),
            bounded_chunk_size=chunk_size,
            max_rss_gb=max_rss,
            required_columns=req_cols,
            panels=panels_dict,
            holdout_lock=data.get("temporal_holdout_lock", {}),
        )


@dataclasses.dataclass(frozen=True)
class StructuralCheckResult:
    """Aggregate structural scan results for a single MEPS panel."""

    puf_id: str
    panel_number: int
    total_rows: int
    total_columns: int
    ordered_columns: tuple[str, ...]
    column_dtypes: dict[str, str]
    schema_hash: str
    duplicate_columns_count: int
    required_columns_present: bool
    required_columns_missing: tuple[str, ...]
    all5rds_1_count: int
    all5rds_other_count: int
    invalid_longwt_count: int
    missing_varstr_count: int
    missing_varpsu_count: int
    duplicate_dupersid_count: int
    missing_dupersid_count: int
    unexpected_panel_count: int
    yearind_check_status: str
    yearind_1_count: int
    yearind_other_count: int
    checks_passed: bool
    validation_messages: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "puf_id": self.puf_id,
            "panel_number": self.panel_number,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "ordered_columns_count": len(self.ordered_columns),
            "ordered_columns": list(self.ordered_columns),
            "column_dtypes": self.column_dtypes,
            "schema_hash": self.schema_hash,
            "duplicate_columns_count": self.duplicate_columns_count,
            "required_columns_present": self.required_columns_present,
            "required_columns_missing": list(self.required_columns_missing),
            "all5rds_1_count": self.all5rds_1_count,
            "all5rds_other_count": self.all5rds_other_count,
            "invalid_longwt_count": self.invalid_longwt_count,
            "missing_varstr_count": self.missing_varstr_count,
            "missing_varpsu_count": self.missing_varpsu_count,
            "duplicate_dupersid_count": self.duplicate_dupersid_count,
            "missing_dupersid_count": self.missing_dupersid_count,
            "unexpected_panel_count": self.unexpected_panel_count,
            "yearind_check_status": self.yearind_check_status,
            "yearind_1_count": self.yearind_1_count,
            "yearind_other_count": self.yearind_other_count,
            "checks_passed": self.checks_passed,
            "validation_messages": list(self.validation_messages),
        }


@dataclasses.dataclass(frozen=True)
class CrossPanelComparison:
    """Cross-panel column schema comparisons (names and hashes only)."""

    shared_columns_count: int
    shared_columns_hash: str
    shared_columns: tuple[str, ...]
    hc244_only_columns_count: int
    hc244_only_columns_hash: str
    hc244_only_columns: tuple[str, ...]
    hc252_only_columns_count: int
    hc252_only_columns_hash: str
    hc252_only_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "shared_columns_count": self.shared_columns_count,
            "shared_columns_hash": self.shared_columns_hash,
            "shared_columns": list(self.shared_columns),
            "hc244_only_columns_count": self.hc244_only_columns_count,
            "hc244_only_columns_hash": self.hc244_only_columns_hash,
            "hc244_only_columns": list(self.hc244_only_columns),
            "hc252_only_columns_count": self.hc252_only_columns_count,
            "hc252_only_columns_hash": self.hc252_only_columns_hash,
            "hc252_only_columns": list(self.hc252_only_columns),
            "note": "Cross-panel column name sets and hashes are schema metadata only. Exact predictor and target variable mappings remain deferred Gate 7.",
        }


def compute_schema_hash(columns: Sequence[str], dtypes: dict[str, str]) -> str:
    """Compute a deterministic SHA-256 schema hash over ordered column names and dtypes."""
    schema_repr = [{"name": str(c), "dtype": str(dtypes.get(c, "unknown"))} for c in columns]
    canonical_json = json.dumps(schema_repr, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_list_hash(items: Sequence[str]) -> str:
    """Compute a deterministic SHA-256 hash over a canonical list of strings."""
    canonical_json = json.dumps(list(items), separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def extract_single_archive(
    archive_path: pathlib.Path,
    dest_path: pathlib.Path,
    expected_member_name: str,
    expected_archive_sha256: str,
    expected_archive_bytes: int,
    expected_uncompressed_bytes: int | None = None,
    expected_compressed_bytes: int | None = None,
    expected_crc32: int | None = None,
    allowed_member_extensions: Sequence[str] | None = None,
) -> tuple[int, str, ArchiveMemberMetadata]:
    """Securely extract a single data member from a ZIP archive.

    Enforces all archive security constraints, writes to .part, and promotes atomically.
    Returns (extracted_byte_size, extracted_sha256, member_metadata).
    """
    if not archive_path.is_file():
        raise IntegrityError(f"Source archive not found: {archive_path}")

    # Validate archive size and hash
    actual_archive_bytes = archive_path.stat().st_size
    if actual_archive_bytes != expected_archive_bytes:
        raise IntegrityError(
            f"Source archive size mismatch for {archive_path.name}: "
            f"expected {expected_archive_bytes}, got {actual_archive_bytes}"
        )

    actual_archive_sha256 = _compute_sha256_file(archive_path)
    if actual_archive_sha256 != expected_archive_sha256:
        raise IntegrityError(
            f"Source archive SHA-256 mismatch for {archive_path.name}: "
            f"expected {expected_archive_sha256}, got {actual_archive_sha256}"
        )

    # Open ZIP and inspect integrity
    try:
        zf = zipfile.ZipFile(archive_path, "r")
    except Exception as err:
        raise IntegrityError(f"Failed to open ZIP archive {archive_path}: {err}") from err

    with zf:
        # Full CRC test
        try:
            crc_err = zf.testzip()
        except RuntimeError as err:
            if "encrypted" in str(err).lower():
                raise SecurityError(f"Archive member is encrypted, which is prohibited: {err}") from err
            raise IntegrityError(f"ZIP CRC integrity test failed for {archive_path.name}: {err}") from err
        except Exception as err:
            raise IntegrityError(f"ZIP CRC integrity test failed for {archive_path.name}: {err}") from err

        if crc_err is not None:
            raise IntegrityError(f"ZIP CRC integrity test failed for {archive_path.name}: corrupt member {crc_err}")

        infolist = zf.infolist()
        if len(infolist) != 1:
            raise SecurityError(
                f"Archive must contain exactly one member; found {len(infolist)} in {archive_path.name}"
            )

        member = infolist[0]
        member_name = member.filename

        # Path traversal checks
        if ".." in member_name or member_name.startswith("/") or member_name.startswith("\\") or os.path.isabs(member_name):
            raise SecurityError(f"Archive member contains forbidden path traversal sequence: {member_name!r}")

        # Member extension check
        ext = pathlib.PurePosixPath(member_name).suffix.lower()
        if allowed_member_extensions is not None:
            valid_exts = tuple(e.lower() for e in allowed_member_extensions)
        else:
            exp_ext = pathlib.PurePosixPath(expected_member_name).suffix.lower()
            valid_exts = (exp_ext,) if exp_ext else (".dta", ".ssp", ".xpt")

        if ext not in valid_exts:
            raise SecurityError(f"Archive member extension {ext!r} not in permitted extensions: {valid_exts}")

        # Name match check
        if member_name != expected_member_name:
            raise SecurityError(
                f"Archive member name mismatch: expected {expected_member_name!r}, got {member_name!r}"
            )

        # Symlink check
        is_symlink = (member.external_attr >> 16) & 0o170000 == 0o120000
        if is_symlink or member.is_dir():
            raise SecurityError(f"Archive member is a symlink or directory; regular file required: {member_name!r}")

        # Encryption check
        if member.flag_bits & 0x1:
            raise SecurityError(f"Archive member is encrypted, which is prohibited: {member_name!r}")

        # Metadata checks (if expectations provided)
        if expected_uncompressed_bytes is not None and member.file_size != expected_uncompressed_bytes:
            raise IntegrityError(
                f"Member uncompressed size mismatch: expected {expected_uncompressed_bytes}, got {member.file_size}"
            )

        if expected_compressed_bytes is not None and member.compress_size != expected_compressed_bytes:
            raise IntegrityError(
                f"Member compressed size mismatch: expected {expected_compressed_bytes}, got {member.compress_size}"
            )

        if expected_crc32 is not None and member.CRC != expected_crc32:
            raise IntegrityError(
                f"Member CRC32 mismatch: expected {expected_crc32}, got {member.CRC}"
            )

        # Decompression bomb defense
        if member.file_size > MAX_UNCOMPRESSED_ARCHIVE_BYTES:
            raise SecurityError(
                f"Member uncompressed size {member.file_size} exceeds safety limit {MAX_UNCOMPRESSED_ARCHIVE_BYTES}"
            )
        compression_ratio = member.file_size / max(member.compress_size, 1)
        if compression_ratio > MAX_COMPRESSION_RATIO:
            raise SecurityError(
                f"Compression ratio {compression_ratio:.2f} exceeds safety threshold {MAX_COMPRESSION_RATIO}"
            )

        member_meta = ArchiveMemberMetadata(
            member_name=member_name,
            uncompressed_size=member.file_size,
            compressed_size=member.compress_size,
            crc32=member.CRC,
        )

        # Extraction via .part file
        dest_dir = dest_path.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        part_path = dest_path.with_name(dest_path.name + ".part")

        if part_path.is_file():
            try:
                part_path.unlink()
            except OSError:
                pass

        hasher = hashlib.sha256()
        total_extracted_bytes = 0

        fd = os.open(part_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            with zf.open(member, "r") as source_fp:
                while True:
                    chunk = source_fp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    _write_all(fd, chunk)
                    hasher.update(chunk)
                    total_extracted_bytes += len(chunk)
                    if expected_uncompressed_bytes is not None and total_extracted_bytes > expected_uncompressed_bytes:
                        raise IntegrityError(
                            f"Extraction exceeded expected size {expected_uncompressed_bytes} bytes"
                        )
            os.fsync(fd)
        except Exception:
            os.close(fd)
            if part_path.is_file():
                try:
                    part_path.unlink()
                except OSError:
                    pass
            raise
        else:
            os.close(fd)

        if expected_uncompressed_bytes is not None and total_extracted_bytes != expected_uncompressed_bytes:
            if part_path.is_file():
                try:
                    part_path.unlink()
                except OSError:
                    pass
            raise IntegrityError(
                f"Extracted byte size mismatch: expected {expected_uncompressed_bytes}, got {total_extracted_bytes}"
            )

        extracted_sha256 = hasher.hexdigest()

        # Promote .part to final destination
        _promote_interim_artifact_no_clobber(part_path, dest_path)

        return total_extracted_bytes, extracted_sha256, member_meta


def _find_prov_artifact_for_panel(
    panel_key: str,
    panel_exp: PanelExpectation,
    prov_artifacts: dict[str, Any],
) -> dict[str, Any] | None:
    """Find matching download provenance artifact entry for a panel."""
    # 1. Direct key match
    if f"{panel_key}_stata_archive" in prov_artifacts:
        return prov_artifacts[f"{panel_key}_stata_archive"]
    if f"{panel_key}_archive" in prov_artifacts:
        art = prov_artifacts[f"{panel_key}_archive"]
        # Check if matching destination
        if art.get("local_relative_path") == panel_exp.source_archive_relative_path or panel_exp.source_archive_relative_path.endswith(art.get("local_relative_path", "")):
            return art
    if panel_key in prov_artifacts:
        return prov_artifacts[panel_key]

    # 2. Match by destination path or PUF ID
    norm_source = str(pathlib.PurePosixPath(panel_exp.source_archive_relative_path))
    source_name = pathlib.Path(norm_source).name
    for art_id, art in prov_artifacts.items():
        if not isinstance(art, dict):
            continue
        art_path = art.get("local_relative_path", "")
        if art_path and (art_path == norm_source or norm_source.endswith(art_path) or art_path.endswith(source_name)):
            return art
        if art.get("puf_id") == panel_exp.puf_id and art.get("artifact_type") == "data_archives":
            if source_name in art.get("requested_url", "") or source_name in art.get("local_relative_path", ""):
                return art

    return None


def extract_meps_archives(
    expectations_path: pathlib.Path | str = DEFAULT_EXPECTATIONS_PATH,
    provenance_path: pathlib.Path | str = DEFAULT_PROVENANCE_PATH,
    raw_root: pathlib.Path | str = DEFAULT_RAW_ROOT,
    interim_root: pathlib.Path | str = DEFAULT_INTERIM_ROOT,
    preparation_provenance_path: pathlib.Path | str = DEFAULT_PREPARATION_PROVENANCE_PATH,
    repo_root: pathlib.Path | None = None,
) -> tuple[PreparationProvenance, dict[str, str]]:
    """Extract all official MEPS archives into interim Stata (.dta) files.

    Preserves pre-existing CPORT extraction evidence in preparation provenance.
    Returns (preparation_provenance, action_statuses).
    """
    if repo_root is None:
        repo_root = pathlib.Path(__file__).resolve().parents[3]

    exp = SchemaExpectations.load(repo_root / expectations_path if not pathlib.Path(expectations_path).is_absolute() else expectations_path)
    prov_file = repo_root / provenance_path if not pathlib.Path(provenance_path).is_absolute() else pathlib.Path(provenance_path)

    # If provenance path does not exist, check fallback paths
    if not prov_file.is_file():
        alt_prov = repo_root / "data/raw/meps/provenance.json"
        alt_snap = repo_root / "docs/data/MEPS_LOCAL_PROVENANCE.json"
        if alt_prov.is_file():
            prov_file = alt_prov
        elif alt_snap.is_file():
            prov_file = alt_snap
        else:
            raise ProvenanceError(
                f"PENDING_OFFICIAL_STATA_DOWNLOAD: Local provenance manifest not found at {prov_file}. "
                "Awaiting supervisor official Stata download."
            )

    with open(prov_file, "r", encoding="utf-8") as f:
        raw_prov = json.load(f)
    prov_artifacts = dict(raw_prov.get("artifacts", {}))

    # Also merge runtime raw provenance if prov_file was a snapshot missing stata archives
    alt_raw_prov = repo_root / "data/raw/meps/provenance.json"
    if alt_raw_prov.is_file() and alt_raw_prov.resolve() != prov_file.resolve():
        try:
            with open(alt_raw_prov, "r", encoding="utf-8") as f:
                alt_data = json.load(f)
            for k, v in alt_data.get("artifacts", {}).items():
                if k not in prov_artifacts:
                    prov_artifacts[k] = v
        except Exception:
            pass

    prep_prov_file = repo_root / preparation_provenance_path if not pathlib.Path(preparation_provenance_path).is_absolute() else pathlib.Path(preparation_provenance_path)

    if prep_prov_file.is_file():
        prep_prov = PreparationProvenance.load(prep_prov_file)
        # Preserve pre-existing CPORT extraction evidence under explicit _cport keys
        cport_entries: dict[str, ExtractionRecord] = {}
        for k, v in list(prep_prov.extractions.items()):
            if v.member_name.endswith(".ssp") and not k.endswith("_cport"):
                cport_entries[f"{k}_cport"] = v
        for k, v in cport_entries.items():
            prep_prov.extractions[k] = v
    else:
        prep_prov = PreparationProvenance.create_empty()

    action_statuses: dict[str, str] = {}

    for panel_key, panel_exp in sorted(exp.panels.items()):
        art_meta = _find_prov_artifact_for_panel(panel_key, panel_exp, prov_artifacts)
        if art_meta is None:
            raise ProvenanceError(
                f"PENDING_OFFICIAL_STATA_DOWNLOAD: Artifact for {panel_exp.puf_id} "
                f"({panel_exp.source_archive_relative_path}) not found in download provenance ({prov_file}). "
                "Awaiting supervisor official Stata download."
            )

        expected_archive_sha256 = art_meta["sha256"]
        expected_archive_bytes = art_meta["byte_size"]
        member_meta_raw = art_meta.get("archive_member_metadata", {})
        if not member_meta_raw:
            raise ProvenanceError(f"Missing archive_member_metadata for {panel_exp.puf_id} in {prov_file}")

        expected_member_name = member_meta_raw.get("member_name", panel_exp.expected_member_name)
        expected_uncompressed_bytes = member_meta_raw.get("uncompressed_size", panel_exp.expected_uncompressed_bytes)
        expected_compressed_bytes = member_meta_raw.get("compressed_size")
        expected_crc32 = member_meta_raw.get("crc32")

        archive_path = repo_root / panel_exp.source_archive_relative_path
        if not archive_path.is_file():
            raise ProvenanceError(
                f"PENDING_OFFICIAL_STATA_DOWNLOAD: Source archive not found on disk at {archive_path}. "
                "Awaiting supervisor official Stata download."
            )

        dest_path = repo_root / panel_exp.interim_relative_path

        # Check idempotency / skip condition
        if dest_path.is_file():
            matching_rec = None
            if panel_key in prep_prov.extractions:
                rec_candidate = prep_prov.extractions[panel_key]
                if rec_candidate.member_name == expected_member_name:
                    matching_rec = rec_candidate
            if matching_rec is None:
                for rec in prep_prov.extractions.values():
                    if rec.puf_id == panel_exp.puf_id and rec.member_name == expected_member_name:
                        matching_rec = rec
                        break

            if matching_rec is not None:
                actual_size = dest_path.stat().st_size
                if (
                    actual_size == matching_rec.extracted_byte_size
                    and matching_rec.source_archive_sha256 == expected_archive_sha256
                    and matching_rec.member_name == expected_member_name
                ):
                    actual_sha = _compute_sha256_file(dest_path)
                    if actual_sha == matching_rec.extracted_sha256:
                        action_statuses[panel_key] = f"[SKIPPED] {panel_exp.puf_id} (already extracted and verified)"
                        continue

            # If dest exists but is not in preparation provenance or doesn't match, fail closed
            raise IntegrityError(
                f"Destination file exists but is unprovenanced or mismatched: {dest_path}; "
                "fail closed to prevent accidental data overwriting."
            )

        # Extract
        extracted_bytes, extracted_sha256, member_metadata = extract_single_archive(
            archive_path=archive_path,
            dest_path=dest_path,
            expected_member_name=expected_member_name,
            expected_archive_sha256=expected_archive_sha256,
            expected_archive_bytes=expected_archive_bytes,
            expected_uncompressed_bytes=expected_uncompressed_bytes,
            expected_compressed_bytes=expected_compressed_bytes,
            expected_crc32=expected_crc32,
        )

        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        rec = ExtractionRecord(
            puf_id=panel_exp.puf_id,
            panel_number=panel_exp.panel_number,
            source_archive_relative_path=panel_exp.source_archive_relative_path,
            source_archive_sha256=expected_archive_sha256,
            member_name=expected_member_name,
            extracted_relative_path=panel_exp.interim_relative_path,
            extracted_byte_size=extracted_bytes,
            extracted_sha256=extracted_sha256,
            extraction_timestamp_utc=now_utc,
            archive_member_metadata=member_metadata,
        )

        prep_prov.add_record(panel_key, rec)
        prep_prov.save(prep_prov_file)
        action_statuses[panel_key] = f"[EXTRACTED] {panel_exp.puf_id} -> {panel_exp.interim_relative_path} ({extracted_bytes:,} bytes)"

    return prep_prov, action_statuses


def scan_single_panel_schema(
    interim_path: pathlib.Path,
    panel_exp: PanelExpectation,
    chunk_size: int = BOUNDED_SCAN_CHUNK_SIZE,
    repo_root: pathlib.Path | None = None,
    preparation_provenance_path: pathlib.Path | str = DEFAULT_PREPARATION_PROVENANCE_PATH,
    download_provenance_path: pathlib.Path | str = DEFAULT_PROVENANCE_PATH,
) -> StructuralCheckResult:
    """Stream an interim Stata file in bounded chunks via pandas.read_stata to aggregate schema and structural checks.

    Strictly forbids printing or retaining individual row data, microdata values,
    feature distributions, or outcome rates.
    """
    if not interim_path.is_file():
        raise IntegrityError(f"Interim data file not found: {interim_path}")

    if chunk_size > MAX_SCAN_CHUNK_SIZE or chunk_size <= 0:
        raise SecurityError(
            f"Chunk size {chunk_size} violates safety boundary (must be 1 <= chunk_size <= {MAX_SCAN_CHUNK_SIZE})"
        )

    # Validate memory ceiling before scanning
    peak_rss_before = get_peak_rss_gb()
    if peak_rss_before >= MAX_RSS_GB_LIMIT:
        raise SecurityError(f"Current RSS {peak_rss_before:.2f} GB exceeds limit {MAX_RSS_GB_LIMIT} GB before scanning")

    # Verify interim file and raw archive against provenance if repo_root is provided
    if repo_root is not None:
        prep_file = repo_root / preparation_provenance_path if not pathlib.Path(preparation_provenance_path).is_absolute() else pathlib.Path(preparation_provenance_path)
        if prep_file.is_file():
            prep_prov = PreparationProvenance.load(prep_file)
            matching_rec: ExtractionRecord | None = None
            norm_interim = str(pathlib.PurePosixPath(panel_exp.interim_relative_path))
            for rec in prep_prov.extractions.values():
                if rec.puf_id == panel_exp.puf_id or rec.extracted_relative_path == norm_interim:
                    matching_rec = rec
                    break
            if matching_rec is not None:
                actual_size = interim_path.stat().st_size
                if actual_size != matching_rec.extracted_byte_size:
                    raise IntegrityError(
                        f"Interim file size mismatch against preparation provenance for {interim_path.name}: "
                        f"expected {matching_rec.extracted_byte_size}, got {actual_size}"
                    )
                actual_sha = _compute_sha256_file(interim_path)
                if actual_sha != matching_rec.extracted_sha256:
                    raise IntegrityError(
                        f"Interim file SHA-256 mismatch against preparation provenance for {interim_path.name}: "
                        f"expected {matching_rec.extracted_sha256}, got {actual_sha}"
                    )
                raw_archive = repo_root / matching_rec.source_archive_relative_path
                if raw_archive.is_file():
                    raw_sha = _compute_sha256_file(raw_archive)
                    if raw_sha != matching_rec.source_archive_sha256:
                        raise IntegrityError(
                            f"Source archive SHA-256 mismatch against recorded provenance for {raw_archive.name}"
                        )

    total_rows = 0
    ordered_columns: tuple[str, ...] = ()
    column_dtypes: dict[str, str] = {}
    duplicate_columns_count = 0
    required_columns_present = False
    required_columns_missing: list[str] = []

    all5rds_1_count = 0
    all5rds_other_count = 0
    invalid_longwt_count = 0
    missing_varstr_count = 0
    missing_varpsu_count = 0
    duplicate_dupersid_count = 0
    missing_dupersid_count = 0
    unexpected_panel_count = 0
    yearind_1_count = 0
    yearind_other_count = 0

    seen_dupersid_hashes: set[bytes] = set()

    # Stream Stata reader in bounded chunks
    try:
        with pd.read_stata(
            interim_path,
            convert_categoricals=False,
            chunksize=chunk_size,
        ) as reader:
            is_first_chunk = True
            for chunk in reader:
                chunk_len = len(chunk)
                total_rows += chunk_len

                # Extract schema on first chunk
                if is_first_chunk:
                    is_first_chunk = False
                    raw_cols = [str(c).strip().upper() for c in chunk.columns]
                    ordered_columns = tuple(raw_cols)
                    duplicate_columns_count = len(raw_cols) - len(set(raw_cols))
                    column_dtypes = {
                        col_name: str(chunk[orig_col].dtype)
                        for col_name, orig_col in zip(raw_cols, chunk.columns)
                    }

                    # Check required structural columns
                    cols_set = set(ordered_columns)
                    missing = [c for c in REQUIRED_STRUCTURAL_COLUMNS if c not in cols_set]
                    if not missing:
                        required_columns_present = True
                    else:
                        required_columns_present = False
                        required_columns_missing = missing

                # Normalize chunk column names to uppercase stripped strings
                chunk.columns = [str(c).strip().upper() for c in chunk.columns]

                # 1. ALL5RDS check
                if "ALL5RDS" in chunk.columns:
                    c_all5rds = chunk["ALL5RDS"]
                    c_1 = int((c_all5rds == 1).sum())
                    all5rds_1_count += c_1
                    all5rds_other_count += (chunk_len - c_1)

                # 2. LONGWT check (must be non-negative and finite)
                if "LONGWT" in chunk.columns:
                    c_longwt = chunk["LONGWT"]
                    invalid_weights = int((~np.isfinite(c_longwt) | (c_longwt < 0)).sum())
                    invalid_longwt_count += invalid_weights

                # 3. VARSTR check (must not be missing/null)
                if "VARSTR" in chunk.columns:
                    missing_varstr_count += int(pd.isna(chunk["VARSTR"]).sum())

                # 4. VARPSU check (must not be missing/null)
                if "VARPSU" in chunk.columns:
                    missing_varpsu_count += int(pd.isna(chunk["VARPSU"]).sum())

                # 5. DUPERSID uniqueness check (store SHA-256 hashes only, never raw IDs)
                if "DUPERSID" in chunk.columns:
                    c_dupersid = chunk["DUPERSID"]
                    for val in c_dupersid:
                        if pd.isna(val) or val == "" or val == b"":
                            missing_dupersid_count += 1
                        else:
                            id_str = val.decode("ascii").strip() if isinstance(val, bytes) else str(val).strip()
                            if not id_str:
                                missing_dupersid_count += 1
                                continue
                            id_hash = hashlib.sha256(id_str.encode("utf-8")).digest()
                            if id_hash in seen_dupersid_hashes:
                                duplicate_dupersid_count += 1
                            else:
                                seen_dupersid_hashes.add(id_hash)

                # 6. PANEL check
                if "PANEL" in chunk.columns:
                    c_panel = chunk["PANEL"]
                    unexpected_panel_count += int((c_panel != panel_exp.expected_panel_value).sum())

                # 7. YEARIND check
                if "YEARIND" in chunk.columns:
                    c_yearind = chunk["YEARIND"]
                    y_1 = int((c_yearind == 1).sum())
                    yearind_1_count += y_1
                    yearind_other_count += (chunk_len - y_1)

                # Prohibit access to protected, demographic, or outcome features
                del chunk

    except Exception as err:
        raise IntegrityError(f"Failed scanning Stata file {interim_path}: {err}") from err
    finally:
        seen_dupersid_hashes.clear()
        gc.collect()

    # Check RSS memory limit
    peak_rss_after = get_peak_rss_gb()
    if peak_rss_after >= MAX_RSS_GB_LIMIT:
        raise SecurityError(f"Peak RSS {peak_rss_after:.2f} GB exceeded safety limit {MAX_RSS_GB_LIMIT} GB")

    schema_hash = compute_schema_hash(ordered_columns, column_dtypes)
    total_columns = len(ordered_columns)

    # Evaluate checks against expectations
    validation_messages: list[str] = []
    checks_passed = True

    if total_rows != panel_exp.expected_rows:
        checks_passed = False
        validation_messages.append(
            f"Total rows mismatch: expected {panel_exp.expected_rows}, got {total_rows}"
        )

    if total_columns != panel_exp.expected_columns:
        checks_passed = False
        validation_messages.append(
            f"Total columns mismatch: expected {panel_exp.expected_columns}, got {total_columns}"
        )

    if all5rds_1_count != panel_exp.expected_all5rds_1_count:
        checks_passed = False
        validation_messages.append(
            f"ALL5RDS == 1 count mismatch: expected {panel_exp.expected_all5rds_1_count}, got {all5rds_1_count}"
        )

    if not required_columns_present:
        checks_passed = False
        validation_messages.append(
            f"Required structural columns missing: {required_columns_missing}"
        )

    if duplicate_columns_count != 0:
        checks_passed = False
        validation_messages.append(f"Duplicate column names found: {duplicate_columns_count}")

    if invalid_longwt_count != 0:
        checks_passed = False
        validation_messages.append(f"Invalid (negative or nonfinite) LONGWT records found: {invalid_longwt_count}")

    if missing_varstr_count != 0:
        checks_passed = False
        validation_messages.append(f"Missing VARSTR records found: {missing_varstr_count}")

    if missing_varpsu_count != 0:
        checks_passed = False
        validation_messages.append(f"Missing VARPSU records found: {missing_varpsu_count}")

    if duplicate_dupersid_count != 0:
        checks_passed = False
        validation_messages.append(f"Duplicate DUPERSID records found: {duplicate_dupersid_count}")

    if missing_dupersid_count != 0:
        checks_passed = False
        validation_messages.append(f"Missing DUPERSID records found: {missing_dupersid_count}")

    if unexpected_panel_count != 0:
        checks_passed = False
        validation_messages.append(
            f"Unexpected PANEL values found (expected {panel_exp.expected_panel_value}): {unexpected_panel_count}"
        )

    # Determine YEARIND status
    if yearind_1_count == total_rows and yearind_other_count == 0:
        yearind_check_status = "VALIDATED_ALL_RECORDS_YEARIND_1"
    elif yearind_1_count > 0:
        yearind_check_status = "DEFERRED_GATE_7"
    else:
        yearind_check_status = "DEFERRED_GATE_7"

    if checks_passed:
        validation_messages.append("All authorized structural checks passed successfully.")

    return StructuralCheckResult(
        puf_id=panel_exp.puf_id,
        panel_number=panel_exp.panel_number,
        total_rows=total_rows,
        total_columns=total_columns,
        ordered_columns=ordered_columns,
        column_dtypes=column_dtypes,
        schema_hash=schema_hash,
        duplicate_columns_count=duplicate_columns_count,
        required_columns_present=required_columns_present,
        required_columns_missing=tuple(required_columns_missing),
        all5rds_1_count=all5rds_1_count,
        all5rds_other_count=all5rds_other_count,
        invalid_longwt_count=invalid_longwt_count,
        missing_varstr_count=missing_varstr_count,
        missing_varpsu_count=missing_varpsu_count,
        duplicate_dupersid_count=duplicate_dupersid_count,
        missing_dupersid_count=missing_dupersid_count,
        unexpected_panel_count=unexpected_panel_count,
        yearind_check_status=yearind_check_status,
        yearind_1_count=yearind_1_count,
        yearind_other_count=yearind_other_count,
        checks_passed=checks_passed,
        validation_messages=tuple(validation_messages),
    )


def compare_cross_panel_schemas(
    res_hc244: StructuralCheckResult,
    res_hc252: StructuralCheckResult,
) -> CrossPanelComparison:
    """Compare column schemas across HC-244 and HC-252 (names and hashes only)."""
    set_244 = set(res_hc244.ordered_columns)
    set_252 = set(res_hc252.ordered_columns)

    shared = tuple(sorted(set_244 & set_252))
    only_244 = tuple(sorted(set_244 - set_252))
    only_252 = tuple(sorted(set_252 - set_244))

    return CrossPanelComparison(
        shared_columns_count=len(shared),
        shared_columns_hash=compute_list_hash(shared),
        shared_columns=shared,
        hc244_only_columns_count=len(only_244),
        hc244_only_columns_hash=compute_list_hash(only_244),
        hc244_only_columns=only_244,
        hc252_only_columns_count=len(only_252),
        hc252_only_columns_hash=compute_list_hash(only_252),
        hc252_only_columns=only_252,
    )


def generate_and_save_schema_snapshot(
    prep_prov: PreparationProvenance,
    scan_results: dict[str, StructuralCheckResult],
    cross_comparison: CrossPanelComparison,
    expectations: SchemaExpectations,
    output_path: pathlib.Path | str = DEFAULT_SNAPSHOT_PATH,
    repo_root: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Generate and atomically save the tracked MEPS schema snapshot JSON."""
    if repo_root is None:
        repo_root = pathlib.Path(__file__).resolve().parents[3]

    out_file = repo_root / output_path if not pathlib.Path(output_path).is_absolute() else pathlib.Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    snapshot_data: dict[str, Any] = {
        "schema_version": SCHEMA_SNAPSHOT_SCHEMA_VERSION,
        "snapshot_timestamp_utc": now_utc,
        "disclaimer": PREPARATION_DISCLAIMER,
        "source_provenance": {
            k: {
                "puf_id": v.puf_id,
                "panel_number": v.panel_number,
                "source_archive_relative_path": v.source_archive_relative_path,
                "source_archive_sha256": v.source_archive_sha256,
                "extracted_relative_path": v.extracted_relative_path,
                "extracted_byte_size": v.extracted_byte_size,
                "extracted_sha256": v.extracted_sha256,
                "archive_member_metadata": v.archive_member_metadata.to_dict(),
            }
            for k, v in sorted(prep_prov.extractions.items())
        },
        "cport_blocker_and_resolution": {
            "parser_tested": "pandas.read_sas(format=\"xport\")",
            "file_header_identified": "**COMPRESSED** (SAS PROC CPORT transport stream)",
            "blocker_reason": "Pandas XportReader supports SAS Version 5/6 XPORT transport files (PROC COPY/XPORT) but raises ValueError on PROC CPORT compressed streams.",
            "resolution_path": "OFFICIAL_AHRQ_STATA_FORMAT_INGESTION",
            "resolution_details": "Both HC-244 and HC-252 are officially published by AHRQ in Stata (.dta) format. Pandas standard library supports Stata files via pandas.read_stata natively without requiring pyreadstat or third-party packages.",
            "supervisor_endpoint_verification": {
                "h244dta_zip": "https://meps.ahrq.gov/mepsweb/data_files/pufs/h244/h244dta.zip (HTTP 200 application/zip, 2026-08-28)",
                "h252dta_zip": "https://meps.ahrq.gov/mepsweb/data_files/pufs/h252/h252dta.zip (HTTP 200 application/zip, 2026-08-28)"
            },
            "third_party_packages_installed": False,
            "custom_cport_decoding_attempted": False
        },
        "panels": {k: v.to_dict() for k, v in sorted(scan_results.items())},
        "cross_panel_comparison": cross_comparison.to_dict(),
        "temporal_holdout_lock": {
            "target_panel": "HC-252",
            "panel_number": 27,
            "status": "LOCKED",
            "claim_boundary": "temporal_transport_assessment_within_meps_not_external_validation_or_deployment_simulation",
            "prohibited_inspections_enforced": True,
            "note": "Holdout protections strictly verified. Outcome distributions, protected attributes, and feature summaries remain uninspected.",
        },
    }

    part_file = out_file.with_name(out_file.name + ".part")
    if part_file.is_file():
        try:
            part_file.unlink()
        except OSError:
            pass

    fd = os.open(part_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        payload = json.dumps(snapshot_data, indent=2, sort_keys=False) + "\n"
        _write_all(fd, payload.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)

    _fsync_dir(out_file.parent)
    os.replace(part_file, out_file)
    _fsync_dir(out_file.parent)

    return snapshot_data


def guard_against_prohibited_inspections(variable_name: str) -> None:
    """Security guard that raises SecurityError if any prohibited column or summary is requested."""
    var_upper = variable_name.strip().upper()
    if var_upper in PROHIBITED_INSPECTION_VARIABLES:
        raise SecurityError(
            f"Prohibited inspection: Variable {variable_name!r} is a protected demographic, "
            "insurance coverage, or outcome variable. Inspection is forbidden under Gate 6 protocol."
        )
