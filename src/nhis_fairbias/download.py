"""Idempotent, allowlisted downloading and safe extraction of NHIS CSV ZIPs."""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from typing import Any, Mapping, Sequence

from .schema import (
    DEFAULT_STUDY_CONFIG,
    NHISSchemaError,
    configured_years,
    load_study_config,
    year_spec,
)


class NHISDownloadError(RuntimeError):
    """Raised for download, archive, provenance, or extraction failures."""


OFFICIAL_HOST = "ftp.cdc.gov"
CHUNK_SIZE = 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = DEFAULT_REPO_ROOT / "artifacts" / "nhis" / "data" / "data_manifest.json"


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp with an explicit offset."""

    return _datetime.datetime.now(_datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_run_id() -> str:
    """Create a unique timestamped execution identifier."""

    stamp = _datetime.datetime.now(_datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:10]}"


def compute_sha256(path: pathlib.Path | str) -> str:
    """Compute a file SHA-256 without loading the file into memory."""

    path_obj = pathlib.Path(path)
    digest = hashlib.sha256()
    try:
        with path_obj.open("rb") as handle:
            while True:
                block = handle.read(CHUNK_SIZE)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise NHISDownloadError(f"Could not hash file {path_obj}: {exc}") from exc
    return digest.hexdigest()


def _repo_relative_path(repo_root: pathlib.Path, relative_path: str) -> pathlib.Path:
    """Resolve a configured repository-relative path and reject traversal."""

    candidate = (repo_root / relative_path).resolve()
    root = repo_root.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise NHISDownloadError(f"Configured path escapes repository root: {relative_path}") from exc
    return candidate


def source_zip_path(
    year: int,
    *,
    repo_root: pathlib.Path | str = DEFAULT_REPO_ROOT,
    study_config: Mapping[str, Any] | None = None,
) -> pathlib.Path:
    config = study_config if study_config is not None else load_study_config()
    return _repo_relative_path(pathlib.Path(repo_root), str(year_spec(config, year)["local_source_file"]))


def source_csv_path(
    year: int,
    *,
    repo_root: pathlib.Path | str = DEFAULT_REPO_ROOT,
    study_config: Mapping[str, Any] | None = None,
) -> pathlib.Path:
    config = study_config if study_config is not None else load_study_config()
    return _repo_relative_path(pathlib.Path(repo_root), str(year_spec(config, year)["local_csv_file"]))


def _validate_official_url(year: int, url: str, study_config: Mapping[str, Any]) -> None:
    spec = year_spec(study_config, year)
    if url != spec["source_url"]:
        raise NHISDownloadError(f"URL does not match the configured official URL for {year}.")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != OFFICIAL_HOST:
        raise NHISDownloadError(
            f"Refusing non-allowlisted NHIS URL for {year}; expected https://{OFFICIAL_HOST}."
        )
    if parsed.username or parsed.password or parsed.port or parsed.query or parsed.fragment or not parsed.path:
        raise NHISDownloadError(f"Configured NHIS URL has unsafe URL components for {year}.")


def _validate_archive_member_name(name: str) -> None:
    normalized = name.replace("\\", "/")
    parts = pathlib.PurePosixPath(normalized).parts
    if normalized.startswith("/") or ".." in parts or "" in parts:
        raise NHISDownloadError(f"Unsafe ZIP member path rejected: {name!r}")


def find_csv_member(
    zip_path: pathlib.Path | str,
    *,
    expected_name: str | None = None,
) -> str:
    """Validate a ZIP and return its single CSV member name."""

    path_obj = pathlib.Path(zip_path)
    if not path_obj.is_file():
        raise NHISDownloadError(f"NHIS source ZIP not found: {path_obj}")
    try:
        with zipfile.ZipFile(path_obj, "r") as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise NHISDownloadError(f"ZIP CRC validation failed for member {bad_member!r}.")
            infos = archive.infolist()
            for info in infos:
                _validate_archive_member_name(info.filename)
            csv_infos = [
                info
                for info in infos
                if not info.is_dir() and info.filename.lower().endswith(".csv")
            ]
            if len(csv_infos) != 1:
                raise NHISDownloadError(
                    f"Expected exactly one CSV member in {path_obj.name}, found {len(csv_infos)}."
                )
            member_name = csv_infos[0].filename
            if expected_name is not None and pathlib.PurePosixPath(member_name).name.casefold() != expected_name.casefold():
                raise NHISDownloadError(
                    f"Unexpected CSV member in {path_obj.name}: expected {expected_name!r}, "
                    f"found {pathlib.PurePosixPath(member_name).name!r}."
                )
            return member_name
    except zipfile.BadZipFile as exc:
        raise NHISDownloadError(f"Invalid ZIP archive {path_obj}: {exc}") from exc
    except OSError as exc:
        raise NHISDownloadError(f"Could not inspect ZIP archive {path_obj}: {exc}") from exc


def validate_source_zip(
    year: int,
    *,
    repo_root: pathlib.Path | str = DEFAULT_REPO_ROOT,
    study_config: Mapping[str, Any] | None = None,
) -> tuple[pathlib.Path, str, str, int]:
    """Validate one local official archive and return path/member/hash/size."""

    config = study_config if study_config is not None else load_study_config()
    path_obj = source_zip_path(year, repo_root=repo_root, study_config=config)
    member = find_csv_member(path_obj, expected_name=str(year_spec(config, year)["expected_csv_member"]))
    return path_obj, member, compute_sha256(path_obj), path_obj.stat().st_size


def _load_json_object(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise NHISDownloadError(f"Could not read provenance manifest {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise NHISDownloadError(f"Provenance manifest must contain a JSON object: {path}")
    return loaded


def write_json_atomic(path: pathlib.Path | str, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically through a same-directory ``.part`` file."""

    path_obj = pathlib.Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    part = path_obj.with_name(f".{path_obj.name}.{uuid.uuid4().hex}.part")
    try:
        with part.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(part, path_obj)
    except (OSError, TypeError, ValueError) as exc:
        part.unlink(missing_ok=True)
        raise NHISDownloadError(f"Could not atomically write JSON {path_obj}: {exc}") from exc


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
    value = completed.stdout.strip()
    return value or None


def update_download_manifest(
    *,
    year: int,
    repo_root: pathlib.Path | str,
    study_config: Mapping[str, Any],
    source_sha256: str,
    source_size: int,
    member_name: str,
    downloaded: bool,
    manifest_path: pathlib.Path | str | None = None,
) -> dict[str, Any]:
    """Record source identity while preserving any prior preparation audit."""

    repo_path = pathlib.Path(repo_root).resolve()
    manifest = _load_json_object(pathlib.Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH)
    manifest.setdefault("schema_version", study_config.get("schema_version", "nhis-d0"))
    manifest.setdefault("dataset", study_config.get("dataset"))
    manifest.setdefault("years", {})
    prior = manifest["years"].get(str(year), {})
    if not isinstance(prior, dict):
        prior = {}
    spec = year_spec(study_config, year)
    prior.update(
        {
            "source_url": spec["source_url"],
            "local_source_file": spec["local_source_file"],
            "source_sha256": source_sha256,
            "source_byte_size": int(source_size),
            "zip_csv_member": member_name,
            "download_timestamp_utc": (
                utc_timestamp() if downloaded else prior.get("download_timestamp_utc")
            ),
        }
    )
    manifest["years"][str(year)] = prior
    manifest["software_timestamp_utc"] = utc_timestamp()
    manifest["git_commit_sha"] = _git_commit_sha(repo_path)
    destination = pathlib.Path(manifest_path) if manifest_path else (repo_path / DEFAULT_MANIFEST_PATH.relative_to(DEFAULT_REPO_ROOT))
    write_json_atomic(destination, manifest)
    return manifest


def _copy_response_to_part(response: Any, part_path: pathlib.Path) -> int:
    total = 0
    with part_path.open("xb") as handle:
        while True:
            block = response.read(CHUNK_SIZE)
            if not block:
                break
            handle.write(block)
            total += len(block)
        handle.flush()
        os.fsync(handle.fileno())
    return total


def download_year(
    year: int,
    *,
    repo_root: pathlib.Path | str = DEFAULT_REPO_ROOT,
    study_config: Mapping[str, Any] | None = None,
    force: bool = False,
    manifest_path: pathlib.Path | str | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Download one exact official ZIP with no-clobber and atomic promotion."""

    config = study_config if study_config is not None else load_study_config()
    repo_path = pathlib.Path(repo_root).resolve()
    spec = year_spec(config, year)
    url = str(spec["source_url"])
    _validate_official_url(year, url, config)
    destination = source_zip_path(year, repo_root=repo_path, study_config=config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_file = pathlib.Path(manifest_path) if manifest_path else (repo_path / DEFAULT_MANIFEST_PATH.relative_to(DEFAULT_REPO_ROOT))
    existing_manifest = _load_json_object(manifest_file)
    existing_record = existing_manifest.get("years", {}).get(str(year), {})
    if not isinstance(existing_record, dict):
        existing_record = {}

    if destination.exists() and not force:
        path_obj, member, digest, size = validate_source_zip(year, repo_root=repo_path, study_config=config)
        recorded_digest = existing_record.get("source_sha256")
        if recorded_digest and recorded_digest != digest:
            raise NHISDownloadError(
                f"Existing NHIS ZIP hash differs from the recorded source for {year}; "
                "refusing to silently accept or overwrite a different file."
            )
        update_download_manifest(
            year=year,
            repo_root=repo_path,
            study_config=config,
            source_sha256=digest,
            source_size=size,
            member_name=member,
            downloaded=False,
            manifest_path=manifest_file,
        )
        return {
            "year": int(year),
            "path": str(path_obj),
            "sha256": digest,
            "byte_size": int(size),
            "member": member,
            "downloaded": False,
        }

    part = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "nhis-fairbias-d0/0.1"})
        try:
            response_context = urllib.request.urlopen(request, timeout=timeout)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            raise NHISDownloadError(f"HTTP download failed for {year}: {exc}") from exc
        with response_context as response:
            status = getattr(response, "status", None)
            if status is not None and int(status) != 200:
                raise NHISDownloadError(f"HTTP download failed for {year}: status {status}.")
            final_url = response.geturl() if hasattr(response, "geturl") else url
            parsed_final = urllib.parse.urlparse(final_url)
            if parsed_final.scheme != "https" or parsed_final.hostname != OFFICIAL_HOST:
                raise NHISDownloadError(
                    f"Refusing redirect outside the allowlisted official host for {year}."
                )
            _copy_response_to_part(response, part)
        digest = compute_sha256(part)
        member = find_csv_member(part, expected_name=str(spec["expected_csv_member"]))
        size = part.stat().st_size
        if destination.exists() and not force:
            raise NHISDownloadError(
                f"Destination appeared during download for {year}; refusing to overwrite it without --force."
            )
        os.replace(part, destination)
    except NHISDownloadError:
        part.unlink(missing_ok=True)
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        part.unlink(missing_ok=True)
        raise NHISDownloadError(f"Download validation/promotion failed for {year}: {exc}") from exc

    update_download_manifest(
        year=year,
        repo_root=repo_path,
        study_config=config,
        source_sha256=digest,
        source_size=size,
        member_name=member,
        downloaded=True,
        manifest_path=manifest_file,
    )
    return {
        "year": int(year),
        "path": str(destination),
        "sha256": digest,
        "byte_size": int(size),
        "member": member,
        "downloaded": True,
    }


def extract_year_csv(
    year: int,
    *,
    repo_root: pathlib.Path | str = DEFAULT_REPO_ROOT,
    study_config: Mapping[str, Any] | None = None,
    force: bool = False,
    manifest_path: pathlib.Path | str | None = None,
) -> dict[str, Any]:
    """Extract the one contracted CSV member atomically and without clobbering."""

    config = study_config if study_config is not None else load_study_config()
    repo_path = pathlib.Path(repo_root).resolve()
    zip_path, member, zip_sha, zip_size = validate_source_zip(
        year, repo_root=repo_path, study_config=config
    )
    destination = source_csv_path(year, repo_root=repo_path, study_config=config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_file = pathlib.Path(manifest_path) if manifest_path else (repo_path / DEFAULT_MANIFEST_PATH.relative_to(DEFAULT_REPO_ROOT))
    manifest = _load_json_object(manifest_file)
    record = manifest.get("years", {}).get(str(year), {})
    if not isinstance(record, dict):
        record = {}

    if destination.exists() and not force:
        try:
            existing_sha = compute_sha256(destination)
        except NHISDownloadError:
            raise
        recorded_csv_sha = record.get("raw_csv_sha256")
        if recorded_csv_sha and recorded_csv_sha != existing_sha:
            raise NHISDownloadError(
                f"Existing extracted CSV hash differs from the recorded source for {year}; refusing to clobber it."
            )
        if pathlib.Path(destination).stat().st_size == 0:
            raise NHISDownloadError(f"Existing extracted NHIS CSV is empty: {destination}")
        member = find_csv_member(zip_path, expected_name=str(year_spec(config, year)["expected_csv_member"]))
        record.update(
            {
                "raw_csv_file": str(year_spec(config, year)["local_csv_file"]),
                "raw_csv_sha256": existing_sha,
                "raw_csv_byte_size": int(destination.stat().st_size),
                "source_sha256": zip_sha,
                "source_byte_size": int(zip_size),
                "zip_csv_member": member,
            }
        )
        manifest.setdefault("years", {})[str(year)] = record
        manifest["software_timestamp_utc"] = utc_timestamp()
        manifest["git_commit_sha"] = _git_commit_sha(repo_path)
        write_json_atomic(manifest_file, manifest)
        return {
            "year": int(year),
            "path": str(destination),
            "sha256": existing_sha,
            "source_sha256": zip_sha,
            "source_byte_size": int(zip_size),
            "member": member,
            "extracted": False,
        }

    part = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    try:
        with zipfile.ZipFile(zip_path, "r") as archive, archive.open(member, "r") as source, part.open("xb") as target:
            while True:
                block = source.read(CHUNK_SIZE)
                if not block:
                    break
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
        csv_sha = compute_sha256(part)
        if destination.exists() and not force:
            raise NHISDownloadError(
                f"Extracted CSV destination appeared during preparation for {year}; refusing to overwrite it."
            )
        os.replace(part, destination)
    except NHISDownloadError:
        part.unlink(missing_ok=True)
        raise
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        part.unlink(missing_ok=True)
        raise NHISDownloadError(f"CSV extraction failed for {year}: {exc}") from exc

    manifest = _load_json_object(manifest_file)
    manifest.setdefault("years", {})
    record = manifest["years"].get(str(year), {})
    if not isinstance(record, dict):
        record = {}
    record.update(
        {
            "raw_csv_file": str(year_spec(config, year)["local_csv_file"]),
            "raw_csv_sha256": csv_sha,
            "raw_csv_byte_size": int(destination.stat().st_size),
            "source_sha256": zip_sha,
            "source_byte_size": int(zip_size),
            "zip_csv_member": member,
        }
    )
    manifest["years"][str(year)] = record
    manifest["software_timestamp_utc"] = utc_timestamp()
    manifest["git_commit_sha"] = _git_commit_sha(repo_path)
    write_json_atomic(manifest_file, manifest)
    return {
        "year": int(year),
        "path": str(destination),
        "sha256": csv_sha,
        "source_sha256": zip_sha,
        "source_byte_size": int(zip_size),
        "member": member,
        "extracted": True,
    }


def parse_years(values: Sequence[int] | None, study_config: Mapping[str, Any]) -> tuple[int, ...]:
    """Validate a CLI year selection without silently substituting years."""

    allowed = set(configured_years(study_config))
    selected = configured_years(study_config) if not values else tuple(dict.fromkeys(int(v) for v in values))
    unknown = sorted(set(selected) - allowed)
    if unknown:
        raise NHISDownloadError(f"Requested year(s) are not configured: {unknown}")
    return tuple(selected)


def main(args: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download official CDC/NCHS NHIS Sample Adult CSV ZIP files.")
    parser.add_argument("--years", nargs="+", type=int, required=True, help="Exact configured years to download.")
    parser.add_argument("--force", action="store_true", help="Explicitly permit replacing existing ZIP files.")
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help=argparse.SUPPRESS)
    parser.add_argument("--study-config", default=str(DEFAULT_STUDY_CONFIG), help=argparse.SUPPRESS)
    parser.add_argument("--manifest", default=None, help=argparse.SUPPRESS)
    parsed = parser.parse_args(args)
    try:
        config = load_study_config(parsed.study_config)
        years = parse_years(parsed.years, config)
        for year in years:
            result = download_year(
                year,
                repo_root=parsed.repo_root,
                study_config=config,
                force=parsed.force,
                manifest_path=parsed.manifest,
            )
            action = "Downloaded" if result["downloaded"] else "Validated existing"
            print(f"{action} NHIS {year}: {result['byte_size']:,} bytes; SHA256 {result['sha256']}")
        run_id = new_run_id()
        repo_path = pathlib.Path(parsed.repo_root).resolve()
        execution_path = repo_path / str(config["outputs"]["artifact_root"]) / "runs" / run_id / "download_execution.json"
        write_json_atomic(
            execution_path,
            {
                "run_id": run_id,
                "operation": "download_nhis",
                "software_timestamp_utc": utc_timestamp(),
                "git_commit_sha": _git_commit_sha(repo_path),
                "years": list(years),
                "source_urls": {str(year): config["years"][str(year)]["source_url"] for year in years},
            },
        )
        return 0
    except (NHISDownloadError, NHISSchemaError, OSError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
