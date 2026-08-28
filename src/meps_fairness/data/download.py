"""Secure, atomic, and policy-compliant MEPS data and documentation downloader.

This module enforces strict network policies, path validation, redirect controls,
file integrity checks (PDF magic headers, single-member SAS transport ZIP inspection),
exclusive-creation streaming via .part files, idempotent reruns, and atomic runtime
provenance recording.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import sys
import typing
import urllib.error
import urllib.parse
import urllib.request
import warnings
import zipfile
from typing import Any, Sequence

# Constants and limits
CHUNK_SIZE: int = 65536  # 64 KB streaming buffer
MAX_UNCOMPRESSED_ARCHIVE_BYTES: int = 1_073_741_824  # 1 GB max uncompressed member size
MAX_COMPRESSION_RATIO: float = 50.0  # Max compression ratio for archive bomb defense
MAX_REDIRECTS: int = 5  # Max redirect hops
DEFAULT_DATA_ACCESS_PATH: str = "configs/data_access.json"
DEFAULT_MANIFEST_PATH: str = "configs/meps_artifacts.json"
DEFAULT_OUTPUT_ROOT: str = "data/raw/meps"
DEFAULT_PROVENANCE_PATH: str = "data/raw/meps/provenance.json"
PROVENANCE_SCHEMA_VERSION: str = "1.0.0"
PROVENANCE_DISCLAIMER: str = (
    "Locally computed SHA-256 hashes establish local immutability and reproducibility; "
    "they do not constitute proof of publisher authenticity when no official upstream checksums exist."
)


class DownloadError(Exception):
    """Base exception for MEPS download and data access failures."""


class SecurityError(DownloadError):
    """Raised when a security policy, URL constraint, or path validation is violated."""


class IntegrityError(DownloadError):
    """Raised when checksum, magic bytes, content-length, or archive structure validation fails."""


class ProvenanceError(DownloadError):
    """Raised when runtime provenance records are missing, corrupt, or conflicting."""


@dataclasses.dataclass(frozen=True)
class DataAccessPolicy:
    """Parsed data access authorization and network security policy."""

    schema_version: str
    download_authorization: bool
    allowed_scheme: str
    allowed_hosts: frozenset[str]
    enforce_same_host_redirects: bool
    allowed_artifact_types: frozenset[str]
    authorized_puf_ids: frozenset[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataAccessPolicy:
        """Parse and validate data access configuration from dictionary."""
        if not isinstance(data, dict):
            raise SecurityError("Data access configuration must be a JSON object.")

        schema_ver = data.get("schema_version", "")
        net_policy = data.get("gate_5_network_policy", {})
        if not isinstance(net_policy, dict):
            raise SecurityError("Missing gate_5_network_policy section in data access config.")

        download_auth = net_policy.get("download_authorization", False)
        if download_auth is not True:
            raise SecurityError("Download authorization is not granted in data access configuration.")

        scheme = net_policy.get("allowed_scheme", "")
        if scheme != "https":
            raise SecurityError(f"Allowed scheme must be https, got {scheme!r}.")

        hosts = frozenset(h.lower() for h in net_policy.get("allowed_hosts", []))
        if not hosts:
            raise SecurityError("Allowed hosts list in gate_5_network_policy must not be empty.")

        types = frozenset(net_policy.get("allowed_artifact_types", []))
        if not types:
            raise SecurityError("Allowed artifact types list in gate_5_network_policy must not be empty.")

        pufs_list = data.get("scope", {}).get("authorized_pufs", [])
        puf_ids = frozenset(p.get("puf_id") for p in pufs_list if isinstance(p, dict) and "puf_id" in p)
        if not puf_ids:
            raise SecurityError("Authorized PUFs list in scope must not be empty.")

        return cls(
            schema_version=schema_ver,
            download_authorization=download_auth,
            allowed_scheme=scheme,
            allowed_hosts=hosts,
            enforce_same_host_redirects=bool(net_policy.get("enforce_same_host_redirects", True)),
            allowed_artifact_types=types,
            authorized_puf_ids=puf_ids,
        )

    @classmethod
    def load(cls, path: pathlib.Path | str) -> DataAccessPolicy:
        """Load and parse data access policy from a JSON file path."""
        p = pathlib.Path(path)
        if not p.is_file():
            raise SecurityError(f"Data access configuration file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclasses.dataclass(frozen=True)
class Artifact:
    """Specification for a single MEPS artifact to download."""

    artifact_id: str
    puf_id: str
    panel_number: int | None
    survey_years: str | None
    artifact_type: str
    description: str
    url: str
    relative_destination: str
    expected_content_type: str | None
    endpoint_preflight: dict[str, Any] | None
    publisher_checksum: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Artifact:
        """Parse artifact definition from dictionary."""
        if not isinstance(data, dict):
            raise IntegrityError("Artifact entry must be a JSON object.")

        art_id = data.get("artifact_id")
        puf_id = data.get("puf_id")
        art_type = data.get("artifact_type")
        url = data.get("url")
        dest = data.get("relative_destination")

        if not art_id or not isinstance(art_id, str):
            raise IntegrityError("Artifact must contain a non-empty string artifact_id.")
        if not puf_id or not isinstance(puf_id, str):
            raise IntegrityError("Artifact must contain a non-empty string puf_id.")
        if not art_type or not isinstance(art_type, str):
            raise IntegrityError("Artifact must contain a non-empty string artifact_type.")
        if not url or not isinstance(url, str):
            raise IntegrityError("Artifact must contain a non-empty string url.")
        if not dest or not isinstance(dest, str):
            raise IntegrityError("Artifact must contain a non-empty string relative_destination.")

        return cls(
            artifact_id=art_id,
            puf_id=puf_id,
            panel_number=data.get("panel_number"),
            survey_years=data.get("survey_years"),
            artifact_type=art_type,
            description=data.get("description", ""),
            url=url,
            relative_destination=dest,
            expected_content_type=data.get("expected_content_type"),
            endpoint_preflight=data.get("endpoint_preflight"),
            publisher_checksum=data.get("publisher_checksum"),
        )


@dataclasses.dataclass(frozen=True)
class ArtifactManifest:
    """Parsed manifest containing a collection of MEPS artifacts."""

    schema_version: str
    metadata: dict[str, Any]
    artifacts: tuple[Artifact, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactManifest:
        """Parse artifact manifest from dictionary and enforce uniqueness."""
        if not isinstance(data, dict):
            raise IntegrityError("Artifact manifest must be a JSON object.")

        schema_ver = data.get("schema_version", "")
        meta = data.get("metadata", {})
        raw_artifacts = data.get("artifacts", [])
        if not isinstance(raw_artifacts, list) or not raw_artifacts:
            raise IntegrityError("Artifact manifest must contain a non-empty artifacts list.")

        parsed_list: list[Artifact] = []
        seen_ids: set[str] = set()
        seen_dests: set[str] = set()

        for item in raw_artifacts:
            art = Artifact.from_dict(item)
            if art.artifact_id in seen_ids:
                raise IntegrityError(f"Duplicate artifact_id in manifest: {art.artifact_id}.")
            seen_ids.add(art.artifact_id)

            norm_dest = str(pathlib.PurePosixPath(art.relative_destination))
            if norm_dest in seen_dests:
                raise IntegrityError(
                    f"Duplicate destination path in manifest: {art.relative_destination}."
                )
            seen_dests.add(norm_dest)

            parsed_list.append(art)

        return cls(
            schema_version=schema_ver,
            metadata=meta,
            artifacts=tuple(parsed_list),
        )

    @classmethod
    def load(cls, path: pathlib.Path | str) -> ArtifactManifest:
        """Load and parse artifact manifest from a JSON file path."""
        p = pathlib.Path(path)
        if not p.is_file():
            raise IntegrityError(f"Artifact manifest file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclasses.dataclass(frozen=True)
class ProvenanceRecord:
    """Immutable runtime provenance record for a downloaded artifact."""

    artifact_id: str
    puf_id: str
    artifact_type: str
    requested_url: str
    final_url: str
    http_status: int
    content_type: str
    byte_size: int
    sha256: str
    local_relative_path: str
    archive_member_metadata: dict[str, Any] | None
    publisher_checksum: str | None
    retrieval_timestamp_utc: str

    def to_dict(self) -> dict[str, Any]:
        """Convert provenance record to dictionary representation."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvenanceRecord:
        """Construct provenance record from dictionary."""
        return cls(
            artifact_id=data["artifact_id"],
            puf_id=data["puf_id"],
            artifact_type=data["artifact_type"],
            requested_url=data["requested_url"],
            final_url=data["final_url"],
            http_status=int(data["http_status"]),
            content_type=data["content_type"],
            byte_size=int(data["byte_size"]),
            sha256=data["sha256"],
            local_relative_path=data["local_relative_path"],
            archive_member_metadata=data.get("archive_member_metadata"),
            publisher_checksum=data.get("publisher_checksum"),
            retrieval_timestamp_utc=data["retrieval_timestamp_utc"],
        )


@dataclasses.dataclass
class ProvenanceManifest:
    """Full runtime provenance document."""

    schema_version: str
    retrieval_timestamp_utc: str
    disclaimer: str
    artifacts: dict[str, ProvenanceRecord]

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "retrieval_timestamp_utc": self.retrieval_timestamp_utc,
            "disclaimer": self.disclaimer,
            "artifacts": {k: self.artifacts[k].to_dict() for k in sorted(self.artifacts)},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvenanceManifest:
        """Parse provenance manifest from dictionary."""
        if not isinstance(data, dict):
            raise ProvenanceError("Provenance file must contain a JSON object.")
        raw_artifacts = data.get("artifacts", {})
        parsed_records = {
            k: ProvenanceRecord.from_dict(v)
            for k, v in raw_artifacts.items()
            if isinstance(v, dict)
        }
        return cls(
            schema_version=data.get("schema_version", PROVENANCE_SCHEMA_VERSION),
            retrieval_timestamp_utc=data.get("retrieval_timestamp_utc", ""),
            disclaimer=data.get("disclaimer", PROVENANCE_DISCLAIMER),
            artifacts=parsed_records,
        )


def validate_url(url: str, policy: DataAccessPolicy) -> urllib.parse.SplitResult:
    """Validate URL syntax, scheme, host, port, and parameter constraints."""
    if not isinstance(url, str) or not url.strip():
        raise SecurityError("URL must be a non-empty string.")

    parsed = urllib.parse.urlsplit(url)

    if parsed.scheme.lower() != policy.allowed_scheme.lower():
        raise SecurityError(
            f"Disallowed URL scheme '{parsed.scheme}': must be '{policy.allowed_scheme}'."
        )

    hostname = parsed.hostname.lower() if parsed.hostname else ""
    if not hostname or hostname not in policy.allowed_hosts:
        raise SecurityError(
            f"Disallowed host '{parsed.hostname}': must be one of {sorted(policy.allowed_hosts)}."
        )

    if parsed.port is not None and parsed.port != 443:
        raise SecurityError(
            f"Disallowed non-default port '{parsed.port}': only standard HTTPS port (443) is permitted."
        )

    if parsed.username or parsed.password:
        raise SecurityError("URLs containing embedded user credentials are strictly prohibited.")

    if parsed.query:
        raise SecurityError(f"Disallowed query parameters in URL: '{parsed.query}'.")

    if parsed.fragment:
        raise SecurityError(f"Disallowed fragment in URL: '{parsed.fragment}'.")

    if not parsed.path or parsed.path == "/":
        raise SecurityError("URL must specify a concrete resource path.")

    return parsed


def resolve_destination_path(output_root: pathlib.Path, relative_dest: str) -> pathlib.Path:
    """Resolve and securely validate a relative destination path beneath output root.

    Rejects absolute paths, backslashes, path traversal, escapes, and symlinks.
    """
    if not isinstance(relative_dest, str) or not relative_dest.strip():
        raise SecurityError("Destination path must be a non-empty relative string.")

    if relative_dest.startswith("/") or relative_dest.startswith("\\"):
        raise SecurityError(f"Absolute destination path rejected: '{relative_dest}'.")

    if "\\" in relative_dest:
        raise SecurityError(f"Backslashes in destination path rejected: '{relative_dest}'.")

    dest_pure = pathlib.PurePosixPath(relative_dest)
    if ".." in dest_pure.parts:
        raise SecurityError(f"Path traversal ('..') rejected: '{relative_dest}'.")

    resolved_root = output_root.resolve()
    if os.path.islink(output_root):
        raise SecurityError(f"Symlinked output root rejected: '{output_root}'.")

    # Walk along each segment of the path and reject any symlinked directory or destination
    current = resolved_root
    for part in dest_pure.parts:
        current = current / part
        if os.path.islink(current):
            raise SecurityError(f"Symlink detected in destination path: '{current}'.")

    target_path = current.resolve()

    try:
        target_path.relative_to(resolved_root)
    except ValueError as err:
        raise SecurityError(
            f"Resolved path '{target_path}' escapes output root '{resolved_root}'."
        ) from err

    return current


def validate_pdf_magic(file_path: pathlib.Path) -> None:
    """Validate that the file begins with standard PDF magic bytes (%PDF-)."""
    if not file_path.is_file():
        raise IntegrityError(f"PDF file does not exist: {file_path}")
    size = file_path.stat().st_size
    if size < 5:
        raise IntegrityError(f"File too small to be a valid PDF ({size} bytes): {file_path}")
    with open(file_path, "rb") as f:
        header = f.read(5)
    if header != b"%PDF-":
        raise IntegrityError(
            f"Invalid PDF magic header in {file_path}: expected b'%PDF-', got {header!r}"
        )


def validate_zip_archive(
    file_path: pathlib.Path,
    max_uncompressed_bytes: int = MAX_UNCOMPRESSED_ARCHIVE_BYTES,
    max_ratio: float = MAX_COMPRESSION_RATIO,
) -> dict[str, Any]:
    """Validate ZIP archive structure, single-member SAS transport rule, and safety constraints.

    Does not extract archive contents.
    """
    if not file_path.is_file():
        raise IntegrityError(f"ZIP file does not exist: {file_path}")
    if file_path.stat().st_size == 0:
        raise IntegrityError(f"ZIP file is empty (0 bytes): {file_path}")

    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            infolist = zf.infolist()
            if len(infolist) == 0:
                raise IntegrityError(f"ZIP archive contains zero members: {file_path}")
            if len(infolist) > 1:
                member_names = [info.filename for info in infolist]
                raise IntegrityError(
                    f"ZIP archive must contain exactly 1 member, but found {len(infolist)}: {member_names}"
                )

            info = infolist[0]

            if info.is_dir() or info.filename.endswith("/"):
                raise IntegrityError(f"ZIP member cannot be a directory: '{info.filename}'")

            if info.flag_bits & 0x1:
                raise SecurityError(f"Encrypted ZIP member is prohibited: '{info.filename}'")

            if "\\" in info.filename:
                raise SecurityError(f"Backslash in ZIP member name rejected: '{info.filename}'")
            if info.filename.startswith("/"):
                raise SecurityError(f"Absolute path in ZIP member name rejected: '{info.filename}'")

            parts = pathlib.PurePosixPath(info.filename).parts
            if ".." in parts:
                raise SecurityError(f"Path traversal in ZIP member name rejected: '{info.filename}'")

            lower_name = info.filename.lower()
            if not (lower_name.endswith(".ssp") or lower_name.endswith(".xpt")):
                raise IntegrityError(
                    f"ZIP member '{info.filename}' does not have a permitted SAS transport extension (.ssp or .xpt)."
                )

            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise SecurityError(f"Symlink ZIP member rejected: '{info.filename}'")

            if info.file_size > max_uncompressed_bytes:
                raise SecurityError(
                    f"ZIP member uncompressed size ({info.file_size} bytes) exceeds safety limit ({max_uncompressed_bytes} bytes)."
                )

            if info.compress_size > 0:
                ratio = info.file_size / info.compress_size
                if ratio > max_ratio and info.file_size > 10_000_000:
                    raise SecurityError(
                        f"ZIP member compression ratio ({ratio:.1f}x) exceeds limit ({max_ratio}x)."
                    )

            try:
                bad_member = zf.testzip()
                if bad_member is not None:
                    raise IntegrityError(f"Corrupt ZIP member CRC checksum: '{bad_member}' in {file_path}")
            except RuntimeError as err:
                if "encrypted" in str(err).lower() or "password" in str(err).lower():
                    raise SecurityError(f"Encrypted ZIP member is prohibited: {err}") from err
                raise IntegrityError(f"Corrupt ZIP member: {err}") from err

            return {
                "member_name": info.filename,
                "uncompressed_size": info.file_size,
                "compressed_size": info.compress_size,
                "crc32": info.CRC,
            }
    except zipfile.BadZipFile as err:
        raise IntegrityError(f"Invalid ZIP archive file {file_path}: {err}") from err


class StrictRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Urllib redirect handler validating redirect targets against DataAccessPolicy."""

    def __init__(self, policy: DataAccessPolicy, max_redirects: int = MAX_REDIRECTS) -> None:
        super().__init__()
        self.policy = policy
        self.max_redirects = max_redirects

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        """Validate redirect before following."""
        redirect_count = getattr(req, "_strict_redirect_count", 0) + 1
        if redirect_count > self.max_redirects:
            raise SecurityError(
                f"Redirect limit exceeded ({self.max_redirects} hops) for request: {req.full_url}"
            )

        resolved_url = urllib.parse.urljoin(req.full_url, newurl)
        parsed_target = validate_url(resolved_url, self.policy)

        if self.policy.enforce_same_host_redirects:
            parsed_req = urllib.parse.urlsplit(req.full_url)
            prev_host = (parsed_req.hostname or "").lower()
            target_host = (parsed_target.hostname or "").lower()
            if prev_host != target_host:
                raise SecurityError(
                    f"Cross-host redirect rejected by policy (from '{prev_host}' to '{target_host}')."
                )

        new_headers = dict(req.headers)
        new_req = urllib.request.Request(
            resolved_url,
            headers=new_headers,
            origin_req_host=req.origin_req_host,
            unverifiable=True,
        )
        new_req._strict_redirect_count = redirect_count
        return new_req


def build_secure_opener(policy: DataAccessPolicy) -> urllib.request.OpenerDirector:
    """Build a urllib OpenerDirector equipped with the strict redirect handler."""
    redirect_handler = StrictRedirectHandler(policy=policy)
    opener = urllib.request.build_opener(redirect_handler)
    opener.addheaders = [
        ("User-Agent", "MEPSFairnessResearch/1.0 (AHRQ-MEPS-Longitudinal-Study; urllib)"),
    ]
    return opener


def load_provenance(path: pathlib.Path) -> ProvenanceManifest | None:
    """Load existing provenance manifest if present."""
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ProvenanceManifest.from_dict(data)


def _write_all(fd: int, data: bytes | bytearray | memoryview) -> int:
    """Write all bytes in data to file descriptor fd, handling low-level short writes.

    Uses a memoryview loop to avoid memory copying. Treats zero or negative return
    values from os.write as fatal errors.
    Returns the total number of bytes successfully written.
    """
    view = memoryview(data)
    total_written = 0
    total_bytes = len(view)
    while total_written < total_bytes:
        n = os.write(fd, view[total_written:])
        if n <= 0:
            raise OSError(
                f"Low-level write failed: os.write returned {n} (expected positive byte count)."
            )
        total_written += n
    return total_written


def _fsync_dir(dir_path: pathlib.Path) -> None:
    """Fsync a directory to ensure directory entries are persisted to disk, if supported."""
    try:
        dir_fd = os.open(str(dir_path), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        # Directory fsync is not supported or permitted on some platforms/filesystems
        pass


def _promote_data_artifact_no_clobber(
    part_path: pathlib.Path,
    dest_path: pathlib.Path,
) -> None:
    """Atomically promote a .part file to its final destination using os.link (no-clobber).

    Fails with FileExistsError if dest_path already exists (e.g. via concurrent race).
    Fsyncs the parent directory if supported, then unlinks part_path.
    Never deletes or clobbers dest_path.
    """
    os.link(str(part_path), str(dest_path))
    _fsync_dir(dest_path.parent)
    try:
        part_path.unlink()
        _fsync_dir(dest_path.parent)
    except OSError as unlink_err:
        warnings.warn(
            f"Incident Notice: Successfully hard-linked '{dest_path}', but failed to unlink "
            f"temporary .part file '{part_path}': {unlink_err}. Final data artifact is intact. "
            f"Manual cleanup of the .part file may be performed by an operator.",
            RuntimeWarning,
            stacklevel=2,
        )


def save_provenance_atomic(
    provenance_path: pathlib.Path,
    manifest: ProvenanceManifest,
) -> None:
    """Atomically write provenance manifest to disk via a .part file."""
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = provenance_path.with_name(provenance_path.name + ".part")

    if part_path.exists():
        raise ProvenanceError(
            f"Stale provenance .part file found at '{part_path}'. Refusing to overwrite."
        )

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    data_bytes = (json.dumps(manifest.to_dict(), indent=2) + "\n").encode("utf-8")

    created_part = False
    fd = -1
    save_successful = False
    try:
        fd = os.open(str(part_path), flags, 0o644)
        created_part = True
        _write_all(fd, data_bytes)
        os.fsync(fd)
        os.close(fd)
        fd = -1

        os.replace(part_path, provenance_path)
        _fsync_dir(provenance_path.parent)
        save_successful = True
    finally:
        if fd != -1:
            try:
                os.close(fd)
            except OSError:
                pass
        if created_part and not save_successful and part_path.exists():
            try:
                part_path.unlink()
            except OSError:
                pass


def compute_file_sha256_and_size(file_path: pathlib.Path) -> tuple[str, int]:
    """Compute SHA-256 hash and byte size of a file on disk."""
    hasher = hashlib.sha256()
    total_bytes = 0
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
            total_bytes += len(chunk)
    return hasher.hexdigest(), total_bytes


def download_single_artifact(
    artifact: Artifact,
    policy: DataAccessPolicy,
    output_root: pathlib.Path,
    opener: urllib.request.OpenerDirector,
    existing_records: dict[str, ProvenanceRecord],
) -> tuple[str, ProvenanceRecord]:
    """Download, validate, and record provenance for a single MEPS artifact.

    Returns (status, ProvenanceRecord) where status is 'downloaded' or 'skipped'.
    """
    # 1. Enforce data access authorizations
    validate_url(artifact.url, policy)
    if artifact.puf_id not in policy.authorized_puf_ids:
        raise SecurityError(
            f"PUF '{artifact.puf_id}' is not in authorized_pufs: {sorted(policy.authorized_puf_ids)}."
        )
    if artifact.artifact_type not in policy.allowed_artifact_types:
        raise SecurityError(
            f"Artifact type '{artifact.artifact_type}' is not in allowed_artifact_types: {sorted(policy.allowed_artifact_types)}."
        )

    # 2. Resolve target destination and .part file
    dest_path = resolve_destination_path(output_root, artifact.relative_destination)
    part_path = dest_path.with_name(dest_path.name + ".part")

    # 3. Check for existing file & verify idempotency against runtime provenance
    if dest_path.exists():
        if artifact.artifact_id in existing_records:
            prev_rec = existing_records[artifact.artifact_id]
            cur_sha256, cur_size = compute_file_sha256_and_size(dest_path)
            if cur_size == prev_rec.byte_size and cur_sha256 == prev_rec.sha256:
                # Re-validate file integrity
                if artifact.expected_content_type == "application/pdf" or dest_path.suffix.lower() == ".pdf":
                    validate_pdf_magic(dest_path)
                elif artifact.expected_content_type == "application/zip" or dest_path.suffix.lower() == ".zip":
                    validate_zip_archive(dest_path)
                return "skipped", prev_rec
            else:
                raise FileExistsError(
                    f"Final artifact already exists at '{dest_path}' but hash/size does not match "
                    f"recorded provenance ({cur_sha256} vs {prev_rec.sha256}). Never overwriting."
                )
        else:
            raise FileExistsError(
                f"Final artifact already exists at '{dest_path}' but is not recorded in runtime "
                f"provenance. Refusing to overwrite unprovenanced file."
            )

    # 4. Fail closed if stale .part file exists
    if part_path.exists():
        raise FileExistsError(
            f"Stale .part file detected at '{part_path}'. Refusing to proceed. Manual investigation required."
        )

    # 5. Execute streaming download to .part file
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(str(part_path), flags, 0o644)
    download_successful = False

    try:
        req = urllib.request.Request(artifact.url)
        with opener.open(req, timeout=60.0) as response:
            http_status = getattr(response, "status", 200)
            final_url = response.geturl()
            # Validate redirected final URL if redirect occurred
            if final_url != artifact.url:
                parsed_final = validate_url(final_url, policy)
                if policy.enforce_same_host_redirects:
                    parsed_orig = urllib.parse.urlsplit(artifact.url)
                    orig_host = (parsed_orig.hostname or "").lower()
                    final_host = (parsed_final.hostname or "").lower()
                    if orig_host != final_host:
                        raise SecurityError(
                            f"Cross-host redirect rejected by policy (from '{orig_host}' to '{final_host}')."
                        )

            raw_ct = response.headers.get("Content-Type", "")
            norm_ct = raw_ct.split(";")[0].strip().lower()

            # Content-type checking
            if artifact.expected_content_type == "application/pdf":
                valid_pdf_types = {"application/pdf", "application/x-pdf", "application/octet-stream", "binary/octet-stream"}
                if norm_ct and norm_ct not in valid_pdf_types:
                    raise IntegrityError(
                        f"Content-Type mismatch for {artifact.artifact_id}: expected application/pdf, got '{norm_ct}'."
                    )
            elif artifact.expected_content_type == "application/zip":
                valid_zip_types = {"application/zip", "application/x-zip-compressed", "application/x-zip", "application/octet-stream", "binary/octet-stream"}
                if norm_ct and norm_ct not in valid_zip_types:
                    raise IntegrityError(
                        f"Content-Type mismatch for {artifact.artifact_id}: expected application/zip, got '{norm_ct}'."
                    )

            hasher = hashlib.sha256()
            total_bytes = 0

            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                written = _write_all(fd, chunk)
                hasher.update(chunk[:written])
                total_bytes += written

            os.fsync(fd)

        os.close(fd)
        fd = -1  # Mark closed

        if total_bytes == 0:
            raise IntegrityError(f"Downloaded artifact '{artifact.artifact_id}' is empty (0 bytes).")

        cl_header = response.headers.get("Content-Length")
        if cl_header is not None:
            try:
                expected_len = int(cl_header)
                if total_bytes != expected_len:
                    raise IntegrityError(
                        f"Content-Length mismatch for {artifact.artifact_id}: expected {expected_len}, got {total_bytes}."
                    )
            except ValueError:
                pass

        # 6. File content validation
        archive_meta = None
        if artifact.expected_content_type == "application/pdf" or dest_path.suffix.lower() == ".pdf":
            validate_pdf_magic(part_path)
        elif artifact.expected_content_type == "application/zip" or dest_path.suffix.lower() == ".zip":
            archive_meta = validate_zip_archive(part_path)

        # 7. No-clobber atomic promotion to final path via os.link
        _promote_data_artifact_no_clobber(part_path, dest_path)
        download_successful = True

        retrieval_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
        computed_sha256 = hasher.hexdigest()

        rel_dest_norm = str(pathlib.PurePosixPath(artifact.relative_destination))
        record = ProvenanceRecord(
            artifact_id=artifact.artifact_id,
            puf_id=artifact.puf_id,
            artifact_type=artifact.artifact_type,
            requested_url=artifact.url,
            final_url=final_url,
            http_status=http_status,
            content_type=norm_ct or (artifact.expected_content_type or "application/octet-stream"),
            byte_size=total_bytes,
            sha256=computed_sha256,
            local_relative_path=rel_dest_norm,
            archive_member_metadata=archive_meta,
            publisher_checksum=None,
            retrieval_timestamp_utc=retrieval_time,
        )
        return "downloaded", record

    finally:
        if fd != -1:
            try:
                os.close(fd)
            except OSError:
                pass
        if not download_successful and part_path.exists():
            try:
                part_path.unlink()
            except OSError:
                pass


def download_artifacts(
    manifest: ArtifactManifest,
    policy: DataAccessPolicy,
    output_root: pathlib.Path,
    provenance_path: pathlib.Path,
    opener: urllib.request.OpenerDirector | None = None,
) -> dict[str, tuple[str, ProvenanceRecord]]:
    """Execute download and validation workflow for all artifacts in the manifest."""
    if opener is None:
        opener = build_secure_opener(policy)

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Load existing provenance if available
    existing_manifest = load_provenance(provenance_path)
    existing_records: dict[str, ProvenanceRecord] = (
        dict(existing_manifest.artifacts) if existing_manifest else {}
    )

    results: dict[str, tuple[str, ProvenanceRecord]] = {}
    updated_records: dict[str, ProvenanceRecord] = dict(existing_records)

    for artifact in manifest.artifacts:
        status, record = download_single_artifact(
            artifact=artifact,
            policy=policy,
            output_root=output_root,
            opener=opener,
            existing_records=existing_records,
        )
        results[artifact.artifact_id] = (status, record)

        if status == "downloaded":
            # Check compatibility before recording
            if artifact.artifact_id in updated_records:
                prev = updated_records[artifact.artifact_id]
                if prev.sha256 != record.sha256 or prev.byte_size != record.byte_size:
                    raise ProvenanceError(
                        f"Provenance conflict for {artifact.artifact_id}: previously recorded SHA-256 "
                        f"'{prev.sha256}' conflicts with new SHA-256 '{record.sha256}'."
                    )
            updated_records[artifact.artifact_id] = record
            existing_records[artifact.artifact_id] = record

            # Persist provenance atomically only after a newly downloaded record
            now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
            current_manifest = ProvenanceManifest(
                schema_version=PROVENANCE_SCHEMA_VERSION,
                retrieval_timestamp_utc=now_utc,
                disclaimer=PROVENANCE_DISCLAIMER,
                artifacts=dict(updated_records),
            )
            save_provenance_atomic(provenance_path, current_manifest)

    return results


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Securely download and verify MEPS official artifacts and record runtime provenance.",
    )
    parser.add_argument(
        "--data-access",
        default=DEFAULT_DATA_ACCESS_PATH,
        help=f"Path to data access configuration JSON (default: {DEFAULT_DATA_ACCESS_PATH})",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST_PATH,
        help=f"Path to artifact manifest JSON (default: {DEFAULT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root directory for downloaded files (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--provenance",
        default=DEFAULT_PROVENANCE_PATH,
        help=f"Path to runtime provenance JSON (default: {DEFAULT_PROVENANCE_PATH})",
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parsed_args = parse_args(args)

    data_access_path = pathlib.Path(parsed_args.data_access)
    manifest_path = pathlib.Path(parsed_args.manifest)
    output_root = pathlib.Path(parsed_args.output_root)
    provenance_path = pathlib.Path(parsed_args.provenance)

    try:
        policy = DataAccessPolicy.load(data_access_path)
        manifest = ArtifactManifest.load(manifest_path)

        results = download_artifacts(
            manifest=manifest,
            policy=policy,
            output_root=output_root,
            provenance_path=provenance_path,
        )

        downloaded_count = sum(1 for status, _ in results.values() if status == "downloaded")
        skipped_count = sum(1 for status, _ in results.values() if status == "skipped")

        for art_id, (status, rec) in results.items():
            print(f"[{status.upper()}] {art_id}: {rec.byte_size} bytes, SHA-256: {rec.sha256}")

        print(f"\nSummary: {downloaded_count} downloaded, {skipped_count} skipped, 0 failed.")
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
