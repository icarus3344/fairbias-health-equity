"""Comprehensive offline test suite for MEPS data download, security, and provenance tracking.

All tests are completely offline and use temporary directories and mocked HTTP interactions.
"""

from __future__ import annotations

import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.request
import warnings
import zipfile
from unittest.mock import MagicMock

# Deterministically add repository src directory derived from resolved test file path
_SRC_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from meps_fairness import data as data_package
from meps_fairness.data.download import (
    CHUNK_SIZE,
    MAX_COMPRESSION_RATIO,
    MAX_UNCOMPRESSED_ARCHIVE_BYTES,
    PROVENANCE_DISCLAIMER,
    PROVENANCE_SCHEMA_VERSION,
    Artifact,
    ArtifactManifest,
    DataAccessPolicy,
    DownloadError,
    IntegrityError,
    ProvenanceError,
    ProvenanceManifest,
    ProvenanceRecord,
    SecurityError,
    StrictRedirectHandler,
    _fsync_dir,
    _promote_data_artifact_no_clobber,
    _write_all,
    build_secure_opener,
    compute_file_sha256_and_size,
    download_artifacts,
    download_single_artifact,
    load_provenance,
    main,
    resolve_destination_path,
    save_provenance_atomic,
    validate_pdf_magic,
    validate_url,
    validate_zip_archive,
)


class MockHTTPResponse:
    """Mock urllib HTTP response object."""

    def __init__(
        self,
        data: bytes,
        headers: dict[str, str] | None = None,
        status: int = 200,
        url: str = "",
    ) -> None:
        self._bio = io.BytesIO(data)
        self.status = status
        self._url = url
        self.headers = headers or {}

    def read(self, size: int = -1) -> bytes:
        return self._bio.read(size)

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> MockHTTPResponse:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        pass


def create_mock_zip(member_name: str = "h244.ssp", content: bytes = b"SAS_DATA_CONTENT") -> bytes:
    """Create valid in-memory ZIP bytes containing a single member."""
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member_name, content)
    return bio.getvalue()


class TestWriteAllHelper(unittest.TestCase):
    """Tests for private _write_all helper verifying memoryview looping and error semantics."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file = pathlib.Path(self.temp_dir.name) / "test_write.bin"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_write_all_full_write(self) -> None:
        fd = os.open(str(self.test_file), os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            data = b"Hello, world! Full write test."
            written = _write_all(fd, data)
            self.assertEqual(written, len(data))
        finally:
            os.close(fd)
        self.assertEqual(self.test_file.read_bytes(), data)

    def test_write_all_partial_writes_loops_to_completion(self) -> None:
        real_write = os.write
        write_call_count = 0

        def mock_short_write(fd: int, buf: Any) -> int:
            nonlocal write_call_count
            write_call_count += 1
            # Write at most 4 bytes per call
            slice_to_write = memoryview(buf)[:4]
            return real_write(fd, slice_to_write)

        fd = os.open(str(self.test_file), os.O_CREAT | os.O_WRONLY, 0o644)
        data = b"0123456789ABCDEF"  # 16 bytes -> 4 iterations of 4 bytes
        try:
            with unittest.mock.patch("os.write", side_effect=mock_short_write):
                written = _write_all(fd, data)
            self.assertEqual(written, 16)
            self.assertEqual(write_call_count, 4)
        finally:
            os.close(fd)
        self.assertEqual(self.test_file.read_bytes(), data)

    def test_write_all_zero_write_raises_oserror(self) -> None:
        def mock_zero_write(fd: int, buf: Any) -> int:
            return 0

        fd = os.open(str(self.test_file), os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            with unittest.mock.patch("os.write", side_effect=mock_zero_write):
                with self.assertRaises(OSError) as ctx:
                    _write_all(fd, b"some bytes")
            self.assertIn("os.write returned 0", str(ctx.exception))
        finally:
            os.close(fd)

    def test_write_all_negative_write_raises_oserror(self) -> None:
        def mock_neg_write(fd: int, buf: Any) -> int:
            return -1

        fd = os.open(str(self.test_file), os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            with unittest.mock.patch("os.write", side_effect=mock_neg_write):
                with self.assertRaises(OSError) as ctx:
                    _write_all(fd, b"some bytes")
            self.assertIn("os.write returned -1", str(ctx.exception))
        finally:
            os.close(fd)


class TestDataAccessPolicyAndURLValidation(unittest.TestCase):
    """Tests for data access policy enforcement and strict URL validation."""

    def setUp(self) -> None:
        self.valid_policy_dict = {
            "schema_version": "1.0.0",
            "scope": {
                "authorized_pufs": [
                    {"puf_id": "HC-244"},
                    {"puf_id": "HC-252"},
                ]
            },
            "gate_5_network_policy": {
                "download_authorization": True,
                "allowed_scheme": "https",
                "allowed_hosts": ["meps.ahrq.gov"],
                "enforce_same_host_redirects": True,
                "allowed_artifact_types": ["data_archives", "documentation", "codebooks"],
            },
        }
        self.policy = DataAccessPolicy.from_dict(self.valid_policy_dict)

    def test_valid_urls(self) -> None:
        urls = [
            "https://meps.ahrq.gov/data_files/pufs/h244/h244ssp.zip",
            "https://meps.ahrq.gov/data_stats/download_data/pufs/h244/h244doc.pdf",
            "https://meps.ahrq.gov/mepsweb/data_files/pufs/h252/h252ssp.zip",
        ]
        for u in urls:
            parsed = validate_url(u, self.policy)
            self.assertEqual(parsed.scheme, "https")
            self.assertEqual(parsed.hostname, "meps.ahrq.gov")

    def test_reject_http_scheme(self) -> None:
        with self.assertRaises(SecurityError):
            validate_url("http://meps.ahrq.gov/data_files/pufs/h244/h244ssp.zip", self.policy)

    def test_reject_disallowed_host(self) -> None:
        bad_hosts = [
            "https://evil.com/h244ssp.zip",
            "https://sub.meps.ahrq.gov/h244ssp.zip",
            "https://ahrq.gov/h244ssp.zip",
        ]
        for u in bad_hosts:
            with self.assertRaises(SecurityError):
                validate_url(u, self.policy)

    def test_reject_non_default_port(self) -> None:
        with self.assertRaises(SecurityError):
            validate_url("https://meps.ahrq.gov:8080/h244ssp.zip", self.policy)

    def test_reject_embedded_credentials(self) -> None:
        with self.assertRaises(SecurityError):
            validate_url("https://admin:secret@meps.ahrq.gov/h244ssp.zip", self.policy)

    def test_reject_query_string(self) -> None:
        with self.assertRaises(SecurityError):
            validate_url("https://meps.ahrq.gov/h244ssp.zip?token=123", self.policy)

    def test_reject_fragment(self) -> None:
        with self.assertRaises(SecurityError):
            validate_url("https://meps.ahrq.gov/h244ssp.zip#part", self.policy)

    def test_reject_empty_or_root_path(self) -> None:
        with self.assertRaises(SecurityError):
            validate_url("https://meps.ahrq.gov", self.policy)
        with self.assertRaises(SecurityError):
            validate_url("https://meps.ahrq.gov/", self.policy)

    def test_reject_download_authorization_false(self) -> None:
        bad_dict = dict(self.valid_policy_dict)
        bad_dict["gate_5_network_policy"] = dict(bad_dict["gate_5_network_policy"])
        bad_dict["gate_5_network_policy"]["download_authorization"] = False
        with self.assertRaises(SecurityError):
            DataAccessPolicy.from_dict(bad_dict)


class TestRedirectHandler(unittest.TestCase):
    """Tests for custom strict redirect handler."""

    def setUp(self) -> None:
        self.policy = DataAccessPolicy(
            schema_version="1.0.0",
            download_authorization=True,
            allowed_scheme="https",
            allowed_hosts=frozenset(["meps.ahrq.gov"]),
            enforce_same_host_redirects=True,
            allowed_artifact_types=frozenset(["data_archives", "documentation"]),
            authorized_puf_ids=frozenset(["HC-244", "HC-252"]),
        )
        self.handler = StrictRedirectHandler(self.policy, max_redirects=3)

    def test_valid_same_host_redirect(self) -> None:
        req = urllib.request.Request("https://meps.ahrq.gov/initial/path.zip")
        new_req = self.handler.redirect_request(
            req, None, 302, "Found", {}, "https://meps.ahrq.gov/final/path.zip"
        )
        self.assertIsNotNone(new_req)
        self.assertEqual(new_req.full_url, "https://meps.ahrq.gov/final/path.zip")

    def test_valid_relative_redirect(self) -> None:
        req = urllib.request.Request("https://meps.ahrq.gov/initial/path.zip")
        new_req = self.handler.redirect_request(
            req, None, 302, "Found", {}, "/target/path.zip"
        )
        self.assertIsNotNone(new_req)
        self.assertEqual(new_req.full_url, "https://meps.ahrq.gov/target/path.zip")

    def test_reject_cross_host_redirect(self) -> None:
        req = urllib.request.Request("https://meps.ahrq.gov/initial/path.zip")
        with self.assertRaises(SecurityError):
            self.handler.redirect_request(
                req, None, 302, "Found", {}, "https://evil.org/malware.zip"
            )

    def test_reject_http_downgrade_redirect(self) -> None:
        req = urllib.request.Request("https://meps.ahrq.gov/initial/path.zip")
        with self.assertRaises(SecurityError):
            self.handler.redirect_request(
                req, None, 302, "Found", {}, "http://meps.ahrq.gov/path.zip"
            )

    def test_reject_redirect_with_query(self) -> None:
        req = urllib.request.Request("https://meps.ahrq.gov/initial/path.zip")
        with self.assertRaises(SecurityError):
            self.handler.redirect_request(
                req, None, 302, "Found", {}, "https://meps.ahrq.gov/path.zip?session=xyz"
            )

    def test_reject_excessive_redirects(self) -> None:
        req = urllib.request.Request("https://meps.ahrq.gov/path1")
        req2 = self.handler.redirect_request(req, None, 302, "Found", {}, "/path2")
        req3 = self.handler.redirect_request(req2, None, 302, "Found", {}, "/path3")
        req4 = self.handler.redirect_request(req3, None, 302, "Found", {}, "/path4")
        with self.assertRaises(SecurityError):
            self.handler.redirect_request(req4, None, 302, "Found", {}, "/path5")

    def test_redirect_counter_is_per_request_chain(self) -> None:
        """Verify two separate requests do not share a redirect counter, while one chain exceeding max fails."""
        handler = StrictRedirectHandler(self.policy, max_redirects=2)

        # Request chain A: 2 redirects (allowed)
        req_a0 = urllib.request.Request("https://meps.ahrq.gov/req_a/start")
        req_a1 = handler.redirect_request(req_a0, None, 302, "Found", {}, "/req_a/hop1")
        req_a2 = handler.redirect_request(req_a1, None, 302, "Found", {}, "/req_a/final")
        self.assertIsNotNone(req_a2)
        self.assertEqual(req_a2.full_url, "https://meps.ahrq.gov/req_a/final")

        # Request chain B: 2 redirects (must succeed independently without sharing counter with chain A)
        req_b0 = urllib.request.Request("https://meps.ahrq.gov/req_b/start")
        req_b1 = handler.redirect_request(req_b0, None, 302, "Found", {}, "/req_b/hop1")
        req_b2 = handler.redirect_request(req_b1, None, 302, "Found", {}, "/req_b/final")
        self.assertIsNotNone(req_b2)
        self.assertEqual(req_b2.full_url, "https://meps.ahrq.gov/req_b/final")

        # Request chain C: 3 redirects (exceeds max_redirects=2 -> fails on hop 3)
        req_c0 = urllib.request.Request("https://meps.ahrq.gov/req_c/start")
        req_c1 = handler.redirect_request(req_c0, None, 302, "Found", {}, "/req_c/hop1")
        req_c2 = handler.redirect_request(req_c1, None, 302, "Found", {}, "/req_c/hop2")
        with self.assertRaises(SecurityError) as ctx:
            handler.redirect_request(req_c2, None, 302, "Found", {}, "/req_c/hop3")
        self.assertIn("Redirect limit exceeded", str(ctx.exception))

    def test_redirect_preserves_headers_safely(self) -> None:
        """Verify headers are safely preserved across redirects."""
        req = urllib.request.Request(
            "https://meps.ahrq.gov/initial",
            headers={"User-Agent": "CustomAgent/1.0", "X-Custom-Auth": "SecretToken"},
        )
        new_req = self.handler.redirect_request(req, None, 302, "Found", {}, "/redirected")
        self.assertIsNotNone(new_req)
        self.assertEqual(new_req.get_header("User-agent"), "CustomAgent/1.0")
        self.assertEqual(new_req.get_header("X-custom-auth"), "SecretToken")
        self.assertTrue(new_req.unverifiable)

    def test_enforce_same_host_redirects_true_rejects_allowed_secondary_host(self) -> None:
        """When enforce_same_host_redirects=True, reject host changes even if target host is in allowed_hosts."""
        policy = DataAccessPolicy(
            schema_version="1.0.0",
            download_authorization=True,
            allowed_scheme="https",
            allowed_hosts=frozenset(["meps.ahrq.gov", "mirror.ahrq.gov"]),
            enforce_same_host_redirects=True,
            allowed_artifact_types=frozenset(["data_archives"]),
            authorized_puf_ids=frozenset(["HC-244"]),
        )
        handler = StrictRedirectHandler(policy, max_redirects=3)
        req = urllib.request.Request("https://meps.ahrq.gov/file.zip")
        with self.assertRaises(SecurityError) as ctx:
            handler.redirect_request(
                req, None, 302, "Found", {}, "https://mirror.ahrq.gov/file.zip"
            )
        self.assertIn("Cross-host redirect rejected", str(ctx.exception))

    def test_enforce_same_host_redirects_false_allows_allowed_secondary_host(self) -> None:
        """When enforce_same_host_redirects=False, allow host changes between allowed_hosts."""
        policy = DataAccessPolicy(
            schema_version="1.0.0",
            download_authorization=True,
            allowed_scheme="https",
            allowed_hosts=frozenset(["meps.ahrq.gov", "mirror.ahrq.gov"]),
            enforce_same_host_redirects=False,
            allowed_artifact_types=frozenset(["data_archives"]),
            authorized_puf_ids=frozenset(["HC-244"]),
        )
        handler = StrictRedirectHandler(policy, max_redirects=3)
        req = urllib.request.Request("https://meps.ahrq.gov/file.zip")
        new_req = handler.redirect_request(
            req, None, 302, "Found", {}, "https://mirror.ahrq.gov/file.zip"
        )
        self.assertIsNotNone(new_req)
        self.assertEqual(new_req.full_url, "https://mirror.ahrq.gov/file.zip")

    def test_enforce_same_host_redirects_false_rejects_disallowed_host(self) -> None:
        """When enforce_same_host_redirects=False, still reject hosts not in allowed_hosts."""
        policy = DataAccessPolicy(
            schema_version="1.0.0",
            download_authorization=True,
            allowed_scheme="https",
            allowed_hosts=frozenset(["meps.ahrq.gov", "mirror.ahrq.gov"]),
            enforce_same_host_redirects=False,
            allowed_artifact_types=frozenset(["data_archives"]),
            authorized_puf_ids=frozenset(["HC-244"]),
        )
        handler = StrictRedirectHandler(policy, max_redirects=3)
        req = urllib.request.Request("https://meps.ahrq.gov/file.zip")
        with self.assertRaises(SecurityError) as ctx:
            handler.redirect_request(
                req, None, 302, "Found", {}, "https://evil.com/malware.zip"
            )
        self.assertIn("Disallowed host", str(ctx.exception))


class TestPathSecurity(unittest.TestCase):
    """Tests for destination path traversal and symlink validation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_valid_relative_path(self) -> None:
        target = resolve_destination_path(self.root, "h244/h244ssp.zip")
        self.assertEqual(target, (self.root.resolve() / "h244" / "h244ssp.zip"))

    def test_reject_absolute_path(self) -> None:
        with self.assertRaises(SecurityError):
            resolve_destination_path(self.root, "/etc/passwd")

    def test_reject_path_traversal(self) -> None:
        with self.assertRaises(SecurityError):
            resolve_destination_path(self.root, "../../outside.zip")
        with self.assertRaises(SecurityError):
            resolve_destination_path(self.root, "h244/../../outside.zip")

    def test_reject_backslash(self) -> None:
        with self.assertRaises(SecurityError):
            resolve_destination_path(self.root, "h244\\evil.zip")

    def test_reject_symlink_in_path(self) -> None:
        symlink_target = self.root / "real_dir"
        symlink_target.mkdir()
        symlink_dir = self.root / "sym_dir"
        try:
            os.symlink(symlink_target, symlink_dir)
        except OSError:
            self.skipTest("Symlinks not supported in test environment")

        with self.assertRaises(SecurityError):
            resolve_destination_path(self.root, "sym_dir/file.zip")


class TestContentValidation(unittest.TestCase):
    """Tests for PDF magic headers and SAS transport ZIP validation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_valid_pdf_magic(self) -> None:
        pdf_path = self.root / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.7\nSample PDF body content\n%%EOF")
        validate_pdf_magic(pdf_path)

    def test_invalid_pdf_magic(self) -> None:
        bad_pdf = self.root / "bad.pdf"
        bad_pdf.write_bytes(b"<html>404 Not Found</html>")
        with self.assertRaises(IntegrityError):
            validate_pdf_magic(bad_pdf)

    def test_valid_ssp_zip(self) -> None:
        zip_path = self.root / "h244ssp.zip"
        zip_path.write_bytes(create_mock_zip("h244.ssp", b"HEADER" * 100))
        meta = validate_zip_archive(zip_path)
        self.assertEqual(meta["member_name"], "h244.ssp")
        self.assertEqual(meta["uncompressed_size"], 600)

    def test_valid_xpt_zip(self) -> None:
        zip_path = self.root / "h252ssp.zip"
        zip_path.write_bytes(create_mock_zip("h252.xpt", b"SAS_XPT_DATA"))
        meta = validate_zip_archive(zip_path)
        self.assertEqual(meta["member_name"], "h252.xpt")

    def test_reject_empty_zip(self) -> None:
        zip_path = self.root / "empty.zip"
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w") as zf:
            pass
        zip_path.write_bytes(bio.getvalue())
        with self.assertRaises(IntegrityError):
            validate_zip_archive(zip_path)

    def test_reject_multi_member_zip(self) -> None:
        zip_path = self.root / "multi.zip"
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w") as zf:
            zf.writestr("h244.ssp", b"data1")
            zf.writestr("notes.txt", b"data2")
        zip_path.write_bytes(bio.getvalue())
        with self.assertRaises(IntegrityError):
            validate_zip_archive(zip_path)

    def test_reject_non_ssp_xpt_member(self) -> None:
        zip_path = self.root / "malicious.zip"
        zip_path.write_bytes(create_mock_zip("run.sh", b"#!/bin/sh\n"))
        with self.assertRaises(IntegrityError):
            validate_zip_archive(zip_path)

    def test_reject_directory_member(self) -> None:
        zip_path = self.root / "dir.zip"
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w") as zf:
            zf.writestr("h244/", b"")
        zip_path.write_bytes(bio.getvalue())
        with self.assertRaises(IntegrityError):
            validate_zip_archive(zip_path)

    def test_reject_path_traversal_in_zip(self) -> None:
        zip_path = self.root / "traversal.zip"
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w") as zf:
            zf.writestr("../../evil.ssp", b"data")
        zip_path.write_bytes(bio.getvalue())
        with self.assertRaises(SecurityError):
            validate_zip_archive(zip_path)

    def test_reject_backslash_in_zip(self) -> None:
        zip_path = self.root / "backslash.zip"
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w") as zf:
            zf.writestr("h244\\evil.ssp", b"data")
        zip_path.write_bytes(bio.getvalue())
        with self.assertRaises(SecurityError):
            validate_zip_archive(zip_path)

    def test_reject_encrypted_zip(self) -> None:
        zip_path = self.root / "encrypted.zip"
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w") as zf:
            zf.writestr("h244.ssp", b"data")
        raw = bytearray(bio.getvalue())
        raw[6] |= 1
        cd_offset = raw.rfind(b"PK\x01\x02")
        if cd_offset != -1:
            raw[cd_offset + 8] |= 1
        zip_path.write_bytes(bytes(raw))
        with self.assertRaises(SecurityError):
            validate_zip_archive(zip_path)

    def test_reject_symlink_zip_member(self) -> None:
        zip_path = self.root / "symlink.zip"
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w") as zf:
            zinfo = zipfile.ZipInfo("h244.ssp")
            zinfo.external_attr = 0o120777 << 16
            zf.writestr(zinfo, b"/etc/passwd")
        zip_path.write_bytes(bio.getvalue())
        with self.assertRaises(SecurityError):
            validate_zip_archive(zip_path)

    def test_reject_uncompressed_size_bomb(self) -> None:
        zip_path = self.root / "size_bomb.zip"
        zip_path.write_bytes(create_mock_zip("h244.ssp", b"data"))
        with self.assertRaises(SecurityError):
            validate_zip_archive(zip_path, max_uncompressed_bytes=2)

    def test_reject_corrupt_zip(self) -> None:
        zip_path = self.root / "corrupt.zip"
        zip_path.write_bytes(b"PK\x03\x04corrupted_data_not_a_valid_zip")
        with self.assertRaises(IntegrityError):
            validate_zip_archive(zip_path)


class TestManifestIntegrity(unittest.TestCase):
    """Tests for artifact manifest loading, parsing, and uniqueness constraints."""

    def test_load_official_manifest(self) -> None:
        manifest_path = pathlib.Path("configs/meps_artifacts.json")
        self.assertTrue(manifest_path.is_file(), "configs/meps_artifacts.json must exist")
        manifest = ArtifactManifest.load(manifest_path)
        self.assertEqual(len(manifest.artifacts), 6)
        self.assertEqual(manifest.schema_version, "1.0.0")

        expected_ids = {
            "hc244_archive",
            "hc244_doc",
            "hc244_codebook",
            "hc252_archive",
            "hc252_doc",
            "hc252_codebook",
        }
        self.assertEqual(set(a.artifact_id for a in manifest.artifacts), expected_ids)

        for art in manifest.artifacts:
            self.assertIn(art.puf_id, {"HC-244", "HC-252"})
            self.assertIn(art.artifact_type, {"data_archives", "documentation", "codebooks"})
            self.assertTrue(art.url.startswith("https://meps.ahrq.gov/"))
            self.assertIsNone(art.publisher_checksum)

    def test_reject_duplicate_artifact_id(self) -> None:
        bad_manifest = {
            "schema_version": "1.0.0",
            "artifacts": [
                {
                    "artifact_id": "duplicate_id",
                    "puf_id": "HC-244",
                    "artifact_type": "documentation",
                    "url": "https://meps.ahrq.gov/doc1.pdf",
                    "relative_destination": "h244/doc1.pdf",
                },
                {
                    "artifact_id": "duplicate_id",
                    "puf_id": "HC-244",
                    "artifact_type": "documentation",
                    "url": "https://meps.ahrq.gov/doc2.pdf",
                    "relative_destination": "h244/doc2.pdf",
                },
            ],
        }
        with self.assertRaises(IntegrityError):
            ArtifactManifest.from_dict(bad_manifest)

    def test_reject_duplicate_destination(self) -> None:
        bad_manifest = {
            "schema_version": "1.0.0",
            "artifacts": [
                {
                    "artifact_id": "art_1",
                    "puf_id": "HC-244",
                    "artifact_type": "documentation",
                    "url": "https://meps.ahrq.gov/doc1.pdf",
                    "relative_destination": "h244/same_dest.pdf",
                },
                {
                    "artifact_id": "art_2",
                    "puf_id": "HC-244",
                    "artifact_type": "documentation",
                    "url": "https://meps.ahrq.gov/doc2.pdf",
                    "relative_destination": "h244/same_dest.pdf",
                },
            ],
        }
        with self.assertRaises(IntegrityError):
            ArtifactManifest.from_dict(bad_manifest)


class TestDownloadWorkflowAndMocks(unittest.TestCase):
    """Tests for streaming download, .part file lifecycle, failure cleanup, and idempotency."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp_dir.name)
        self.output_root = self.root / "data" / "raw" / "meps"
        self.provenance_path = self.output_root / "provenance.json"

        self.policy = DataAccessPolicy(
            schema_version="1.0.0",
            download_authorization=True,
            allowed_scheme="https",
            allowed_hosts=frozenset(["meps.ahrq.gov"]),
            enforce_same_host_redirects=True,
            allowed_artifact_types=frozenset(["data_archives", "documentation", "codebooks"]),
            authorized_puf_ids=frozenset(["HC-244", "HC-252"]),
        )

        self.pdf_artifact = Artifact(
            artifact_id="hc244_doc",
            puf_id="HC-244",
            panel_number=26,
            survey_years="2021-2022",
            artifact_type="documentation",
            description="HC-244 Documentation",
            url="https://meps.ahrq.gov/data_stats/download_data/pufs/h244/h244doc.pdf",
            relative_destination="h244/h244doc.pdf",
            expected_content_type="application/pdf",
            endpoint_preflight={"date": "2026-08-28", "status": 200, "expected_type": "application/pdf"},
            publisher_checksum=None,
        )

        self.zip_artifact = Artifact(
            artifact_id="hc244_archive",
            puf_id="HC-244",
            panel_number=26,
            survey_years="2021-2022",
            artifact_type="data_archives",
            description="HC-244 SAS Archive",
            url="https://meps.ahrq.gov/data_files/pufs/h244/h244ssp.zip",
            relative_destination="h244/h244ssp.zip",
            expected_content_type="application/zip",
            endpoint_preflight={"date": "2026-08-28", "status": 200, "expected_type": "application/zip"},
            publisher_checksum=None,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_atomic_download_pdf_success(self) -> None:
        pdf_bytes = b"%PDF-1.7\nSample content\n%%EOF"
        mock_resp = MockHTTPResponse(
            data=pdf_bytes,
            headers={"Content-Type": "application/pdf", "Content-Length": str(len(pdf_bytes))},
            status=200,
            url=self.pdf_artifact.url,
        )
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp

        status, record = download_single_artifact(
            artifact=self.pdf_artifact,
            policy=self.policy,
            output_root=self.output_root,
            opener=mock_opener,
            existing_records={},
        )

        self.assertEqual(status, "downloaded")
        self.assertEqual(record.artifact_id, "hc244_doc")
        self.assertEqual(record.byte_size, len(pdf_bytes))
        self.assertIsNone(record.publisher_checksum)

        dest_file = self.output_root / "h244" / "h244doc.pdf"
        self.assertTrue(dest_file.is_file())
        self.assertFalse(dest_file.with_name("h244doc.pdf.part").exists())
        self.assertEqual(dest_file.read_bytes(), pdf_bytes)

    def test_atomic_download_zip_success(self) -> None:
        zip_bytes = create_mock_zip("h244.ssp", b"SAS_RECORD_MICRODATA_ROW")
        mock_resp = MockHTTPResponse(
            data=zip_bytes,
            headers={"Content-Type": "application/zip", "Content-Length": str(len(zip_bytes))},
            status=200,
            url=self.zip_artifact.url,
        )
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp

        status, record = download_single_artifact(
            artifact=self.zip_artifact,
            policy=self.policy,
            output_root=self.output_root,
            opener=mock_opener,
            existing_records={},
        )

        self.assertEqual(status, "downloaded")
        self.assertEqual(record.artifact_id, "hc244_archive")
        self.assertIsNotNone(record.archive_member_metadata)
        self.assertEqual(record.archive_member_metadata["member_name"], "h244.ssp")

        dest_file = self.output_root / "h244" / "h244ssp.zip"
        self.assertTrue(dest_file.is_file())
        self.assertFalse(dest_file.with_name("h244ssp.zip.part").exists())

    def test_stale_part_file_fails_closed(self) -> None:
        part_file = self.output_root / "h244" / "h244doc.pdf.part"
        part_file.parent.mkdir(parents=True, exist_ok=True)
        part_file.write_bytes(b"STALE_PART_DATA")

        mock_opener = MagicMock()
        with self.assertRaises(FileExistsError):
            download_single_artifact(
                artifact=self.pdf_artifact,
                policy=self.policy,
                output_root=self.output_root,
                opener=mock_opener,
                existing_records={},
            )

        self.assertTrue(part_file.is_file())

    def test_transfer_failure_cleans_up_new_part(self) -> None:
        class FailingResponse:
            status = 200
            headers = {"Content-Type": "application/pdf"}

            def geturl(self) -> str:
                return "https://meps.ahrq.gov/data_stats/download_data/pufs/h244/h244doc.pdf"

            def read(self, size: int = -1) -> bytes:
                raise OSError("Network stream disconnected mid-transfer")

            def __enter__(self) -> FailingResponse:
                return self

            def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
                pass

        mock_opener = MagicMock()
        mock_opener.open.return_value = FailingResponse()

        with self.assertRaises(OSError):
            download_single_artifact(
                artifact=self.pdf_artifact,
                policy=self.policy,
                output_root=self.output_root,
                opener=mock_opener,
                existing_records={},
            )

        dest_file = self.output_root / "h244" / "h244doc.pdf"
        part_file = dest_file.with_name("h244doc.pdf.part")
        self.assertFalse(dest_file.exists())
        self.assertFalse(part_file.exists())

    def test_empty_download_rejected(self) -> None:
        mock_resp = MockHTTPResponse(
            data=b"",
            headers={"Content-Type": "application/pdf", "Content-Length": "0"},
            status=200,
            url=self.pdf_artifact.url,
        )
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp

        with self.assertRaises(IntegrityError):
            download_single_artifact(
                artifact=self.pdf_artifact,
                policy=self.policy,
                output_root=self.output_root,
                opener=mock_opener,
                existing_records={},
            )

    def test_content_length_mismatch_rejected(self) -> None:
        mock_resp = MockHTTPResponse(
            data=b"%PDF-1.7\nTruncated",
            headers={"Content-Type": "application/pdf", "Content-Length": "99999"},
            status=200,
            url=self.pdf_artifact.url,
        )
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp

        with self.assertRaises(IntegrityError):
            download_single_artifact(
                artifact=self.pdf_artifact,
                policy=self.policy,
                output_root=self.output_root,
                opener=mock_opener,
                existing_records={},
            )

    def test_content_type_mismatch_rejected(self) -> None:
        mock_resp = MockHTTPResponse(
            data=b"<html>Access Denied</html>",
            headers={"Content-Type": "text/html", "Content-Length": "26"},
            status=200,
            url=self.pdf_artifact.url,
        )
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp

        with self.assertRaises(IntegrityError):
            download_single_artifact(
                artifact=self.pdf_artifact,
                policy=self.policy,
                output_root=self.output_root,
                opener=mock_opener,
                existing_records={},
            )

    def test_idempotent_skip_when_provenance_and_file_match(self) -> None:
        pdf_bytes = b"%PDF-1.7\nSample content\n%%EOF"
        dest_file = self.output_root / "h244" / "h244doc.pdf"
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        dest_file.write_bytes(pdf_bytes)

        cur_sha256, cur_size = compute_file_sha256_and_size(dest_file)
        existing_rec = ProvenanceRecord(
            artifact_id="hc244_doc",
            puf_id="HC-244",
            artifact_type="documentation",
            requested_url=self.pdf_artifact.url,
            final_url=self.pdf_artifact.url,
            http_status=200,
            content_type="application/pdf",
            byte_size=cur_size,
            sha256=cur_sha256,
            local_relative_path="h244/h244doc.pdf",
            archive_member_metadata=None,
            publisher_checksum=None,
            retrieval_timestamp_utc="2026-08-28T00:00:00Z",
        )

        mock_opener = MagicMock()
        status, record = download_single_artifact(
            artifact=self.pdf_artifact,
            policy=self.policy,
            output_root=self.output_root,
            opener=mock_opener,
            existing_records={"hc244_doc": existing_rec},
        )

        self.assertEqual(status, "skipped")
        self.assertEqual(record.sha256, cur_sha256)
        mock_opener.open.assert_not_called()

    def test_existing_file_mismatch_fails_closed(self) -> None:
        dest_file = self.output_root / "h244" / "h244doc.pdf"
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        dest_file.write_bytes(b"%PDF-1.7\nTAMPERED_CONTENT\n%%EOF")

        existing_rec = ProvenanceRecord(
            artifact_id="hc244_doc",
            puf_id="HC-244",
            artifact_type="documentation",
            requested_url=self.pdf_artifact.url,
            final_url=self.pdf_artifact.url,
            http_status=200,
            content_type="application/pdf",
            byte_size=100,
            sha256="original_sha256_hash",
            local_relative_path="h244/h244doc.pdf",
            archive_member_metadata=None,
            publisher_checksum=None,
            retrieval_timestamp_utc="2026-08-28T00:00:00Z",
        )

        mock_opener = MagicMock()
        with self.assertRaises(FileExistsError):
            download_single_artifact(
                artifact=self.pdf_artifact,
                policy=self.policy,
                output_root=self.output_root,
                opener=mock_opener,
                existing_records={"hc244_doc": existing_rec},
            )

    def test_unprovenanced_existing_file_fails_closed(self) -> None:
        dest_file = self.output_root / "h244" / "h244doc.pdf"
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        dest_file.write_bytes(b"%PDF-1.7\nUNTRACKED_FILE\n%%EOF")

        mock_opener = MagicMock()
        with self.assertRaises(FileExistsError):
            download_single_artifact(
                artifact=self.pdf_artifact,
                policy=self.policy,
                output_root=self.output_root,
                opener=mock_opener,
                existing_records={},
            )

    def test_incremental_provenance_resumes_after_batch_failure(self) -> None:
        """Verify provenance is saved after first artifact succeeds, allowing rerun to skip first and resume second."""
        manifest = ArtifactManifest(
            schema_version="1.0.0",
            metadata={"name": "test_batch"},
            artifacts=(self.pdf_artifact, self.zip_artifact),
        )

        pdf_bytes = b"%PDF-1.7\nSample content\n%%EOF"
        zip_bytes = create_mock_zip("h244.ssp", b"SAS_DATA_CONTENT")

        # Run 1: First artifact (pdf) succeeds, second artifact (zip) fails
        def mock_open_run1(req: urllib.request.Request, timeout: float = 60.0) -> MockHTTPResponse:
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if url == self.pdf_artifact.url:
                return MockHTTPResponse(
                    data=pdf_bytes,
                    headers={"Content-Type": "application/pdf", "Content-Length": str(len(pdf_bytes))},
                    status=200,
                    url=self.pdf_artifact.url,
                )
            elif url == self.zip_artifact.url:
                raise OSError("Network connection interrupted during zip download")
            raise ValueError(f"Unexpected URL: {url}")

        mock_opener1 = MagicMock()
        mock_opener1.open.side_effect = mock_open_run1

        with self.assertRaises(OSError):
            download_artifacts(
                manifest=manifest,
                policy=self.policy,
                output_root=self.output_root,
                provenance_path=self.provenance_path,
                opener=mock_opener1,
            )

        # Verify state after Run 1 failure:
        # 1. PDF data file exists
        pdf_dest = self.output_root / "h244" / "h244doc.pdf"
        self.assertTrue(pdf_dest.is_file())
        # 2. ZIP data file does NOT exist
        zip_dest = self.output_root / "h244" / "h244ssp.zip"
        self.assertFalse(zip_dest.exists())
        # 3. Provenance manifest exists and contains hc244_doc (and NOT hc244_archive)
        self.assertTrue(self.provenance_path.is_file())
        prov1 = load_provenance(self.provenance_path)
        self.assertIsNotNone(prov1)
        self.assertIn("hc244_doc", prov1.artifacts)
        self.assertNotIn("hc244_archive", prov1.artifacts)
        self.assertEqual(prov1.artifacts["hc244_doc"].byte_size, len(pdf_bytes))

        # Run 2: Resume batch. PDF should be skipped (opener.open should not be called for PDF), ZIP downloads.
        def mock_open_run2(req: urllib.request.Request, timeout: float = 60.0) -> MockHTTPResponse:
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if url == self.pdf_artifact.url:
                raise AssertionError("Opener was unexpectedly called for already-provenanced PDF artifact!")
            elif url == self.zip_artifact.url:
                return MockHTTPResponse(
                    data=zip_bytes,
                    headers={"Content-Type": "application/zip", "Content-Length": str(len(zip_bytes))},
                    status=200,
                    url=self.zip_artifact.url,
                )
            raise ValueError(f"Unexpected URL: {url}")

        mock_opener2 = MagicMock()
        mock_opener2.open.side_effect = mock_open_run2

        results = download_artifacts(
            manifest=manifest,
            policy=self.policy,
            output_root=self.output_root,
            provenance_path=self.provenance_path,
            opener=mock_opener2,
        )

        self.assertEqual(results["hc244_doc"][0], "skipped")
        self.assertEqual(results["hc244_archive"][0], "downloaded")
        self.assertTrue(pdf_dest.is_file())
        self.assertTrue(zip_dest.is_file())

        prov2 = load_provenance(self.provenance_path)
        self.assertIsNotNone(prov2)
        self.assertIn("hc244_doc", prov2.artifacts)
        self.assertIn("hc244_archive", prov2.artifacts)

    def test_artifact_streaming_partial_writes_exact_hash_and_size(self) -> None:
        """Verify artifact streaming with low-level short writes computes exact hash/size and writes full data."""
        pdf_bytes = b"%PDF-1.7\n" + (b"A" * 500) + b"\n%%EOF"
        real_write = os.write

        def mock_short_write(fd: int, buf: Any) -> int:
            slice_to_write = memoryview(buf)[:7]  # Write only 7 bytes at a time
            return real_write(fd, slice_to_write)

        mock_resp = MockHTTPResponse(
            data=pdf_bytes,
            headers={"Content-Type": "application/pdf", "Content-Length": str(len(pdf_bytes))},
            status=200,
            url=self.pdf_artifact.url,
        )
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp

        with unittest.mock.patch("os.write", side_effect=mock_short_write):
            status, record = download_single_artifact(
                artifact=self.pdf_artifact,
                policy=self.policy,
                output_root=self.output_root,
                opener=mock_opener,
                existing_records={},
            )

        self.assertEqual(status, "downloaded")
        self.assertEqual(record.byte_size, len(pdf_bytes))
        dest_file = self.output_root / "h244" / "h244doc.pdf"
        self.assertTrue(dest_file.is_file())
        self.assertEqual(dest_file.read_bytes(), pdf_bytes)
        actual_sha256, actual_size = compute_file_sha256_and_size(dest_file)
        self.assertEqual(record.sha256, actual_sha256)
        self.assertEqual(record.byte_size, actual_size)

    def test_artifact_streaming_zero_write_fatal_cleans_part(self) -> None:
        """Verify a zero write during artifact streaming raises OSError and cleans up worker .part file."""
        pdf_bytes = b"%PDF-1.7\nSample content\n%%EOF"
        mock_resp = MockHTTPResponse(
            data=pdf_bytes,
            headers={"Content-Type": "application/pdf", "Content-Length": str(len(pdf_bytes))},
            status=200,
            url=self.pdf_artifact.url,
        )
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp

        with unittest.mock.patch("os.write", return_value=0):
            with self.assertRaises(OSError) as ctx:
                download_single_artifact(
                    artifact=self.pdf_artifact,
                    policy=self.policy,
                    output_root=self.output_root,
                    opener=mock_opener,
                    existing_records={},
                )
            self.assertIn("os.write returned 0", str(ctx.exception))

        dest_file = self.output_root / "h244" / "h244doc.pdf"
        part_file = dest_file.with_name("h244doc.pdf.part")
        self.assertFalse(dest_file.exists())
        self.assertFalse(part_file.exists())

    def test_no_clobber_race_preserves_existing_final_and_cleans_part(self) -> None:
        """Verify no-clobber atomic promotion via os.link never overwrites a raced destination and cleans .part."""
        dest_file = self.output_root / "h244" / "h244doc.pdf"
        part_file = dest_file.with_name("h244doc.pdf.part")
        raced_final_content = b"%PDF-1.7\nRACED_WINNING_FILE_CONTENT\n%%EOF"
        new_download_content = b"%PDF-1.7\nLOSING_WORKER_DOWNLOAD_CONTENT\n%%EOF"

        class RacedMockResponse:
            status = 200
            headers = {"Content-Type": "application/pdf", "Content-Length": str(len(new_download_content))}

            def __init__(self, dest_path: pathlib.Path) -> None:
                self._dest_path = dest_path
                self._bio = io.BytesIO(new_download_content)

            def read(self, size: int = -1) -> bytes:
                # Simulate concurrent process creating dest_path while streaming is finishing
                if not self._dest_path.exists():
                    self._dest_path.parent.mkdir(parents=True, exist_ok=True)
                    self._dest_path.write_bytes(raced_final_content)
                return self._bio.read(size)

            def geturl(self) -> str:
                return "https://meps.ahrq.gov/data_stats/download_data/pufs/h244/h244doc.pdf"

            def __enter__(self) -> RacedMockResponse:
                return self

            def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
                pass

        mock_opener = MagicMock()
        mock_opener.open.return_value = RacedMockResponse(dest_file)

        with self.assertRaises(FileExistsError):
            download_single_artifact(
                artifact=self.pdf_artifact,
                policy=self.policy,
                output_root=self.output_root,
                opener=mock_opener,
                existing_records={},
            )

        # 1. Raced destination file was preserved intact and NEVER overwritten
        self.assertTrue(dest_file.is_file())
        self.assertEqual(dest_file.read_bytes(), raced_final_content)
        # 2. Worker's temporary .part file was cleaned up
        self.assertFalse(part_file.exists())

    def test_promotion_uses_os_link_not_replace(self) -> None:
        """Verify data artifact promotion uses os.link and not os.replace."""
        pdf_bytes = b"%PDF-1.7\nSample content\n%%EOF"
        mock_resp = MockHTTPResponse(
            data=pdf_bytes,
            headers={"Content-Type": "application/pdf", "Content-Length": str(len(pdf_bytes))},
            status=200,
            url=self.pdf_artifact.url,
        )
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp

        with unittest.mock.patch("os.replace") as mock_replace, unittest.mock.patch("os.link", wraps=os.link) as spy_link:
            status, _ = download_single_artifact(
                artifact=self.pdf_artifact,
                policy=self.policy,
                output_root=self.output_root,
                opener=mock_opener,
                existing_records={},
            )
            self.assertEqual(status, "downloaded")
            mock_replace.assert_not_called()
            spy_link.assert_called_once()

    def test_pure_skip_run_leaves_provenance_bytes_and_mtime_unchanged(self) -> None:
        """Verify a fully idempotent rerun with skipped artifacts does not rewrite provenance.json or alter mtime."""
        manifest = ArtifactManifest(
            schema_version="1.0.0",
            metadata={"name": "test_batch"},
            artifacts=(self.pdf_artifact, self.zip_artifact),
        )

        pdf_bytes = b"%PDF-1.7\nSample content\n%%EOF"
        zip_bytes = create_mock_zip("h244.ssp", b"SAS_DATA_CONTENT")

        def mock_open(req: urllib.request.Request, timeout: float = 60.0) -> MockHTTPResponse:
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if url == self.pdf_artifact.url:
                return MockHTTPResponse(
                    data=pdf_bytes,
                    headers={"Content-Type": "application/pdf", "Content-Length": str(len(pdf_bytes))},
                    status=200,
                    url=self.pdf_artifact.url,
                )
            elif url == self.zip_artifact.url:
                return MockHTTPResponse(
                    data=zip_bytes,
                    headers={"Content-Type": "application/zip", "Content-Length": str(len(zip_bytes))},
                    status=200,
                    url=self.zip_artifact.url,
                )
            raise ValueError(f"Unexpected URL: {url}")

        mock_opener = MagicMock()
        mock_opener.open.side_effect = mock_open

        # Run 1: Initial download populates data files and provenance.json
        res1 = download_artifacts(
            manifest=manifest,
            policy=self.policy,
            output_root=self.output_root,
            provenance_path=self.provenance_path,
            opener=mock_opener,
        )
        self.assertEqual(res1["hc244_doc"][0], "downloaded")
        self.assertEqual(res1["hc244_archive"][0], "downloaded")
        self.assertTrue(self.provenance_path.is_file())

        initial_bytes = self.provenance_path.read_bytes()
        initial_mtime_ns = self.provenance_path.stat().st_mtime_ns

        # Run 2: Pure-skip run
        mock_opener_skip = MagicMock()
        with unittest.mock.patch("meps_fairness.data.download.save_provenance_atomic") as mock_save:
            res2 = download_artifacts(
                manifest=manifest,
                policy=self.policy,
                output_root=self.output_root,
                provenance_path=self.provenance_path,
                opener=mock_opener_skip,
            )
            mock_save.assert_not_called()

        self.assertEqual(res2["hc244_doc"][0], "skipped")
        self.assertEqual(res2["hc244_archive"][0], "skipped")
        mock_opener_skip.open.assert_not_called()

        # Check provenance file bytes and mtime are strictly identical
        self.assertEqual(self.provenance_path.read_bytes(), initial_bytes)
        self.assertEqual(self.provenance_path.stat().st_mtime_ns, initial_mtime_ns)

    def test_promotion_link_success_part_unlink_failure_preserves_final(self) -> None:
        """Verify link success followed by part unlink failure preserves final data artifact and warns."""
        pdf_bytes = b"%PDF-1.7\nSample content\n%%EOF"
        mock_resp = MockHTTPResponse(
            data=pdf_bytes,
            headers={"Content-Type": "application/pdf", "Content-Length": str(len(pdf_bytes))},
            status=200,
            url=self.pdf_artifact.url,
        )
        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp

        dest_file = (self.output_root / "h244" / "h244doc.pdf").resolve()
        part_file = dest_file.with_name("h244doc.pdf.part")

        real_os_unlink = os.unlink

        def mock_os_unlink(path: Any, *args: Any, **kwargs: Any) -> None:
            if pathlib.Path(path).resolve() == part_file:
                raise OSError("Injected unlink failure: temporary file locked")
            return real_os_unlink(path, *args, **kwargs)

        with unittest.mock.patch("os.unlink", side_effect=mock_os_unlink):
            with warnings.catch_warnings(record=True) as recorded_warnings:
                warnings.simplefilter("always")
                status, record = download_single_artifact(
                    artifact=self.pdf_artifact,
                    policy=self.policy,
                    output_root=self.output_root,
                    opener=mock_opener,
                    existing_records={},
                )

        self.assertEqual(status, "downloaded")
        self.assertTrue(dest_file.is_file())
        self.assertEqual(dest_file.read_bytes(), pdf_bytes)
        # Part file remains on disk due to unlink failure
        self.assertTrue(part_file.is_file())
        # Destination was not deleted
        self.assertTrue(dest_file.exists())
        # Incident recovery warning was emitted
        self.assertTrue(any("Incident Notice" in str(w.message) for w in recorded_warnings))


class TestProvenanceManagement(unittest.TestCase):
    """Tests for atomic provenance reading, writing, and compatibility checks."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp_dir.name)
        self.provenance_path = self.root / "provenance.json"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_save_and_load_provenance(self) -> None:
        rec = ProvenanceRecord(
            artifact_id="art1",
            puf_id="HC-244",
            artifact_type="documentation",
            requested_url="https://meps.ahrq.gov/doc.pdf",
            final_url="https://meps.ahrq.gov/doc.pdf",
            http_status=200,
            content_type="application/pdf",
            byte_size=1234,
            sha256="abcdef123456",
            local_relative_path="h244/doc.pdf",
            archive_member_metadata=None,
            publisher_checksum=None,
            retrieval_timestamp_utc="2026-08-28T12:00:00Z",
        )
        manifest = ProvenanceManifest(
            schema_version=PROVENANCE_SCHEMA_VERSION,
            retrieval_timestamp_utc="2026-08-28T12:00:00Z",
            disclaimer=PROVENANCE_DISCLAIMER,
            artifacts={"art1": rec},
        )

        save_provenance_atomic(self.provenance_path, manifest)
        self.assertTrue(self.provenance_path.is_file())
        self.assertFalse(self.provenance_path.with_name("provenance.json.part").exists())

        loaded = load_provenance(self.provenance_path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.schema_version, "1.0.0")
        self.assertIn("art1", loaded.artifacts)
        self.assertEqual(loaded.artifacts["art1"].sha256, "abcdef123456")
        self.assertIsNone(loaded.artifacts["art1"].publisher_checksum)
        self.assertEqual(loaded.disclaimer, PROVENANCE_DISCLAIMER)

    def test_stale_provenance_part_fails_closed_and_preserves_stale_part(self) -> None:
        """Pre-existing stale .part file causes fail-closed error and is never deleted."""
        part_file = self.provenance_path.with_name("provenance.json.part")
        part_file.write_bytes(b"STALE_PROVENANCE_PART")
        manifest = ProvenanceManifest(
            schema_version="1.0.0",
            retrieval_timestamp_utc="2026-08-28T12:00:00Z",
            disclaimer=PROVENANCE_DISCLAIMER,
            artifacts={},
        )
        with self.assertRaises(ProvenanceError):
            save_provenance_atomic(self.provenance_path, manifest)

        self.assertTrue(part_file.is_file())
        self.assertEqual(part_file.read_bytes(), b"STALE_PROVENANCE_PART")

    def test_save_provenance_write_failure_cleans_new_part_and_preserves_final(self) -> None:
        """Injected write failure in save_provenance_atomic cleans up new .part without deleting existing final provenance."""
        initial_rec = ProvenanceRecord(
            artifact_id="init_art",
            puf_id="HC-244",
            artifact_type="documentation",
            requested_url="https://meps.ahrq.gov/init.pdf",
            final_url="https://meps.ahrq.gov/init.pdf",
            http_status=200,
            content_type="application/pdf",
            byte_size=500,
            sha256="init_sha256",
            local_relative_path="h244/init.pdf",
            archive_member_metadata=None,
            publisher_checksum=None,
            retrieval_timestamp_utc="2026-08-28T00:00:00Z",
        )
        initial_manifest = ProvenanceManifest(
            schema_version=PROVENANCE_SCHEMA_VERSION,
            retrieval_timestamp_utc="2026-08-28T00:00:00Z",
            disclaimer=PROVENANCE_DISCLAIMER,
            artifacts={"init_art": initial_rec},
        )
        save_provenance_atomic(self.provenance_path, initial_manifest)
        self.assertTrue(self.provenance_path.is_file())

        new_manifest = ProvenanceManifest(
            schema_version=PROVENANCE_SCHEMA_VERSION,
            retrieval_timestamp_utc="2026-08-28T01:00:00Z",
            disclaimer=PROVENANCE_DISCLAIMER,
            artifacts={"init_art": initial_rec, "new_art": initial_rec},
        )

        with unittest.mock.patch("os.write", side_effect=OSError("Injected write error")):
            with self.assertRaises(OSError):
                save_provenance_atomic(self.provenance_path, new_manifest)

        part_path = self.provenance_path.with_name(self.provenance_path.name + ".part")
        self.assertFalse(part_path.exists())
        self.assertTrue(self.provenance_path.is_file())
        loaded = load_provenance(self.provenance_path)
        self.assertIsNotNone(loaded)
        self.assertIn("init_art", loaded.artifacts)
        self.assertNotIn("new_art", loaded.artifacts)

    def test_save_provenance_replace_failure_cleans_new_part_and_preserves_final(self) -> None:
        """Injected replace failure in save_provenance_atomic cleans up new .part without deleting existing final provenance."""
        initial_rec = ProvenanceRecord(
            artifact_id="init_art",
            puf_id="HC-244",
            artifact_type="documentation",
            requested_url="https://meps.ahrq.gov/init.pdf",
            final_url="https://meps.ahrq.gov/init.pdf",
            http_status=200,
            content_type="application/pdf",
            byte_size=500,
            sha256="init_sha256",
            local_relative_path="h244/init.pdf",
            archive_member_metadata=None,
            publisher_checksum=None,
            retrieval_timestamp_utc="2026-08-28T00:00:00Z",
        )
        initial_manifest = ProvenanceManifest(
            schema_version=PROVENANCE_SCHEMA_VERSION,
            retrieval_timestamp_utc="2026-08-28T00:00:00Z",
            disclaimer=PROVENANCE_DISCLAIMER,
            artifacts={"init_art": initial_rec},
        )
        save_provenance_atomic(self.provenance_path, initial_manifest)
        self.assertTrue(self.provenance_path.is_file())

        new_manifest = ProvenanceManifest(
            schema_version=PROVENANCE_SCHEMA_VERSION,
            retrieval_timestamp_utc="2026-08-28T01:00:00Z",
            disclaimer=PROVENANCE_DISCLAIMER,
            artifacts={"init_art": initial_rec, "new_art": initial_rec},
        )

        with unittest.mock.patch("os.replace", side_effect=OSError("Injected replace error")):
            with self.assertRaises(OSError):
                save_provenance_atomic(self.provenance_path, new_manifest)

        part_path = self.provenance_path.with_name(self.provenance_path.name + ".part")
        self.assertFalse(part_path.exists())
        self.assertTrue(self.provenance_path.is_file())
        loaded = load_provenance(self.provenance_path)
        self.assertIsNotNone(loaded)
        self.assertIn("init_art", loaded.artifacts)
        self.assertNotIn("new_art", loaded.artifacts)

    def test_save_provenance_partial_writes(self) -> None:
        """Verify save_provenance_atomic handles partial writes and saves full valid JSON."""
        rec = ProvenanceRecord(
            artifact_id="art1",
            puf_id="HC-244",
            artifact_type="documentation",
            requested_url="https://meps.ahrq.gov/doc.pdf",
            final_url="https://meps.ahrq.gov/doc.pdf",
            http_status=200,
            content_type="application/pdf",
            byte_size=1234,
            sha256="abcdef123456",
            local_relative_path="h244/doc.pdf",
            archive_member_metadata=None,
            publisher_checksum=None,
            retrieval_timestamp_utc="2026-08-28T12:00:00Z",
        )
        manifest = ProvenanceManifest(
            schema_version=PROVENANCE_SCHEMA_VERSION,
            retrieval_timestamp_utc="2026-08-28T12:00:00Z",
            disclaimer=PROVENANCE_DISCLAIMER,
            artifacts={"art1": rec},
        )

        real_write = os.write
        write_call_count = 0

        def mock_short_write(fd: int, buf: Any) -> int:
            nonlocal write_call_count
            write_call_count += 1
            slice_to_write = memoryview(buf)[:9]  # Write 9 bytes at a time
            return real_write(fd, slice_to_write)

        with unittest.mock.patch("os.write", side_effect=mock_short_write):
            save_provenance_atomic(self.provenance_path, manifest)

        self.assertTrue(self.provenance_path.is_file())
        self.assertGreater(write_call_count, 1)
        loaded = load_provenance(self.provenance_path)
        self.assertIsNotNone(loaded)
        self.assertIn("art1", loaded.artifacts)
        self.assertEqual(loaded.artifacts["art1"].sha256, "abcdef123456")

    def test_save_provenance_zero_write_fatal_cleans_part_and_preserves_final(self) -> None:
        """Verify save_provenance_atomic with zero write raises OSError, cleans .part, and preserves final."""
        initial_rec = ProvenanceRecord(
            artifact_id="init_art",
            puf_id="HC-244",
            artifact_type="documentation",
            requested_url="https://meps.ahrq.gov/init.pdf",
            final_url="https://meps.ahrq.gov/init.pdf",
            http_status=200,
            content_type="application/pdf",
            byte_size=500,
            sha256="init_sha256",
            local_relative_path="h244/init.pdf",
            archive_member_metadata=None,
            publisher_checksum=None,
            retrieval_timestamp_utc="2026-08-28T00:00:00Z",
        )
        initial_manifest = ProvenanceManifest(
            schema_version=PROVENANCE_SCHEMA_VERSION,
            retrieval_timestamp_utc="2026-08-28T00:00:00Z",
            disclaimer=PROVENANCE_DISCLAIMER,
            artifacts={"init_art": initial_rec},
        )
        save_provenance_atomic(self.provenance_path, initial_manifest)
        self.assertTrue(self.provenance_path.is_file())

        new_manifest = ProvenanceManifest(
            schema_version=PROVENANCE_SCHEMA_VERSION,
            retrieval_timestamp_utc="2026-08-28T01:00:00Z",
            disclaimer=PROVENANCE_DISCLAIMER,
            artifacts={"init_art": initial_rec, "new_art": initial_rec},
        )

        with unittest.mock.patch("os.write", return_value=0):
            with self.assertRaises(OSError) as ctx:
                save_provenance_atomic(self.provenance_path, new_manifest)
            self.assertIn("os.write returned 0", str(ctx.exception))

        part_path = self.provenance_path.with_name(self.provenance_path.name + ".part")
        self.assertFalse(part_path.exists())
        self.assertTrue(self.provenance_path.is_file())
        loaded = load_provenance(self.provenance_path)
        self.assertIsNotNone(loaded)
        self.assertIn("init_art", loaded.artifacts)
        self.assertNotIn("new_art", loaded.artifacts)


class TestCLIMain(unittest.TestCase):
    """Tests for CLI argument parsing and main execution function."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp_dir.name)
        self.data_access_path = self.root / "data_access.json"
        self.manifest_path = self.root / "meps_artifacts.json"
        self.output_root = self.root / "data" / "raw" / "meps"
        self.provenance_path = self.output_root / "provenance.json"

        with open("configs/data_access.json", "r", encoding="utf-8") as f:
            self.data_access_path.write_text(f.read(), encoding="utf-8")

        with open("configs/meps_artifacts.json", "r", encoding="utf-8") as f:
            self.manifest_path.write_text(f.read(), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_cli_main_with_mocks(self) -> None:
        pdf_bytes = b"%PDF-1.7\nSample PDF\n%%EOF"
        zip_bytes = create_mock_zip("h244.ssp", b"SAS_DATA")

        def mock_open(req: urllib.request.Request, timeout: float = 60.0) -> MockHTTPResponse:
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if url.endswith(".pdf"):
                return MockHTTPResponse(pdf_bytes, {"Content-Type": "application/pdf"}, 200, url)
            elif url.endswith(".zip"):
                return MockHTTPResponse(zip_bytes, {"Content-Type": "application/zip"}, 200, url)
            raise ValueError(f"Unexpected URL: {url}")

        with unittest.mock.patch("urllib.request.OpenerDirector.open", side_effect=mock_open):
            exit_code = main(
                [
                    "--data-access",
                    str(self.data_access_path),
                    "--manifest",
                    str(self.manifest_path),
                    "--output-root",
                    str(self.output_root),
                    "--provenance",
                    str(self.provenance_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(self.provenance_path.is_file())

            prov = load_provenance(self.provenance_path)
            self.assertIsNotNone(prov)
            self.assertEqual(len(prov.artifacts), 6)


class TestCLIScriptSubprocess(unittest.TestCase):
    """Subprocess regression tests verifying scripts/download_meps.py CLI execution without PYTHONPATH."""

    def setUp(self) -> None:
        self.repo_root = pathlib.Path(__file__).resolve().parents[1]
        self.script_path = self.repo_root / "scripts" / "download_meps.py"
        self.py311_bin = sys.executable

        # Clean environment with PYTHONPATH removed
        self.clean_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    def test_cli_script_help_from_repo_root_without_pythonpath(self) -> None:
        """Verify scripts/download_meps.py --help executes successfully from repo root without PYTHONPATH."""
        proc = subprocess.run(
            [self.py311_bin, str(self.script_path), "--help"],
            cwd=str(self.repo_root),
            env=self.clean_env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(proc.returncode, 0, f"Expected exit code 0, got {proc.returncode}. Stderr: {proc.stderr}")
        self.assertIn("usage: download_meps.py", proc.stdout)
        self.assertIn("--data-access", proc.stdout)
        self.assertIn("--manifest", proc.stdout)
        self.assertIn("--output-root", proc.stdout)
        self.assertIn("--provenance", proc.stdout)
        self.assertEqual(proc.stderr.strip(), "")

    def test_cli_script_help_from_temporary_cwd_without_pythonpath(self) -> None:
        """Verify scripts/download_meps.py --help executes successfully from a temporary cwd without PYTHONPATH."""
        with tempfile.TemporaryDirectory() as temp_cwd:
            proc = subprocess.run(
                [self.py311_bin, str(self.script_path), "--help"],
                cwd=temp_cwd,
                env=self.clean_env,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(proc.returncode, 0, f"Expected exit code 0, got {proc.returncode}. Stderr: {proc.stderr}")
            self.assertIn("usage: download_meps.py", proc.stdout)
            self.assertIn("--data-access", proc.stdout)
            self.assertIn("--manifest", proc.stdout)
            self.assertIn("--output-root", proc.stdout)
            self.assertIn("--provenance", proc.stdout)
            self.assertEqual(proc.stderr.strip(), "")

    def test_cli_script_help_from_repo_root_relative_path_without_pythonpath(self) -> None:
        """Verify python scripts/download_meps.py --help executes from repo root with relative script path."""
        proc = subprocess.run(
            [self.py311_bin, "scripts/download_meps.py", "--help"],
            cwd=str(self.repo_root),
            env=self.clean_env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(proc.returncode, 0, f"Expected exit code 0, got {proc.returncode}. Stderr: {proc.stderr}")
        self.assertIn("usage: download_meps.py", proc.stdout)
        self.assertEqual(proc.stderr.strip(), "")


class TestManifestExtensionGeneralization(unittest.TestCase):
    """Tests verifying manifest-declared archive member extension generalization and safety boundaries."""

    def test_package_exports_preparation_security_helpers(self) -> None:
        """Package-level exports must bind every name advertised by ``data.__all__``."""
        for name in ("get_peak_rss_gb", "guard_against_prohibited_inspections"):
            self.assertIn(name, data_package.__all__)
            self.assertTrue(callable(getattr(data_package, name, None)))

    def test_default_extension_when_omitted(self) -> None:
        raw_art = {
            "artifact_id": "hc244_archive",
            "puf_id": "HC-244",
            "artifact_type": "data_archives",
            "url": "https://meps.ahrq.gov/data_files/pufs/h244/h244ssp.zip",
            "relative_destination": "h244/h244ssp.zip",
        }
        art = Artifact.from_dict(raw_art)
        self.assertEqual(art.archive_allowed_member_extensions, (".ssp", ".xpt"))

    def test_custom_dta_extension(self) -> None:
        raw_art = {
            "artifact_id": "hc244_stata_archive",
            "puf_id": "HC-244",
            "artifact_type": "data_archives",
            "url": "https://meps.ahrq.gov/mepsweb/data_files/pufs/h244/h244dta.zip",
            "relative_destination": "h244/h244dta.zip",
            "archive_allowed_member_extensions": [".dta"],
        }
        art = Artifact.from_dict(raw_art)
        self.assertEqual(art.archive_allowed_member_extensions, (".dta",))

    def test_reject_empty_extension_list(self) -> None:
        raw_art = {
            "artifact_id": "test",
            "puf_id": "HC-244",
            "artifact_type": "data_archives",
            "url": "https://meps.ahrq.gov/data.zip",
            "relative_destination": "test.zip",
            "archive_allowed_member_extensions": [],
        }
        with self.assertRaises(IntegrityError):
            Artifact.from_dict(raw_art)

    def test_reject_wildcard_extension(self) -> None:
        for bad in ["*", ".*", ".*.", "[a-z]"]:
            raw_art = {
                "artifact_id": "test",
                "puf_id": "HC-244",
                "artifact_type": "data_archives",
                "url": "https://meps.ahrq.gov/data.zip",
                "relative_destination": "test.zip",
                "archive_allowed_member_extensions": [bad],
            }
            with self.assertRaises(SecurityError):
                Artifact.from_dict(raw_art)

    def test_reject_missing_leading_dot(self) -> None:
        raw_art = {
            "artifact_id": "test",
            "puf_id": "HC-244",
            "artifact_type": "data_archives",
            "url": "https://meps.ahrq.gov/data.zip",
            "relative_destination": "test.zip",
            "archive_allowed_member_extensions": ["dta"],
        }
        with self.assertRaises(SecurityError):
            Artifact.from_dict(raw_art)

    def test_reject_path_traversal_extension(self) -> None:
        for bad in ["../.dta", ".dta/", "/.dta", ".."]:
            raw_art = {
                "artifact_id": "test",
                "puf_id": "HC-244",
                "artifact_type": "data_archives",
                "url": "https://meps.ahrq.gov/data.zip",
                "relative_destination": "test.zip",
                "archive_allowed_member_extensions": [bad],
            }
            with self.assertRaises(SecurityError):
                Artifact.from_dict(raw_art)

    def test_reject_non_alphanumeric_extension(self) -> None:
        for bad in [".dta$", ".dta#", ".d ta", ".dta\0"]:
            raw_art = {
                "artifact_id": "test",
                "puf_id": "HC-244",
                "artifact_type": "data_archives",
                "url": "https://meps.ahrq.gov/data.zip",
                "relative_destination": "test.zip",
                "archive_allowed_member_extensions": [bad],
            }
            with self.assertRaises(SecurityError):
                Artifact.from_dict(raw_art)

    def test_load_meps_stata_artifacts_manifest(self) -> None:
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        manifest_path = repo_root / "configs" / "meps_stata_artifacts.json"
        manifest = ArtifactManifest.load(manifest_path)
        self.assertEqual(len(manifest.artifacts), 2)
        for art in manifest.artifacts:
            self.assertEqual(art.archive_allowed_member_extensions, (".dta",))
            self.assertEqual(art.expected_content_type, "application/zip")
            self.assertIsNone(art.publisher_checksum)


class TestStataZipValidationAndWorkflow(unittest.TestCase):
    """Tests verifying ZIP validation for .dta members, attack defenses, and mock download workflow."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = pathlib.Path(self.temp_dir.name)
        self.output_root = self.temp_path / "raw"
        self.provenance_path = self.temp_path / "provenance.json"

        self.policy = DataAccessPolicy.from_dict(
            {
                "schema_version": "1.0.0",
                "scope": {
                    "authorized_pufs": [
                        {"puf_id": "HC-244"},
                        {"puf_id": "HC-252"},
                    ]
                },
                "gate_5_network_policy": {
                    "download_authorization": True,
                    "allowed_scheme": "https",
                    "allowed_hosts": ["meps.ahrq.gov"],
                    "enforce_same_host_redirects": True,
                    "allowed_artifact_types": ["data_archives", "documentation", "codebooks"],
                },
            }
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_validate_zip_archive_with_dta_valid(self) -> None:
        zip_path = self.temp_path / "h244dta.zip"
        zip_bytes = create_mock_zip(member_name="h244.dta", content=b"STATA_DATA_BYTES")
        zip_path.write_bytes(zip_bytes)

        meta = validate_zip_archive(zip_path, allowed_member_extensions=[".dta"])
        self.assertEqual(meta["member_name"], "h244.dta")
        self.assertEqual(meta["uncompressed_size"], len(b"STATA_DATA_BYTES"))

    def test_validate_zip_archive_dta_rejected_under_default_ssp(self) -> None:
        zip_path = self.temp_path / "h244dta.zip"
        zip_bytes = create_mock_zip(member_name="h244.dta", content=b"STATA_DATA_BYTES")
        zip_path.write_bytes(zip_bytes)

        with self.assertRaises(IntegrityError) as ctx:
            validate_zip_archive(zip_path)  # Default (".ssp", ".xpt")
        self.assertIn("permitted archive member extension", str(ctx.exception))

    def test_validate_zip_archive_ssp_rejected_under_dta_allowed(self) -> None:
        zip_path = self.temp_path / "h244ssp.zip"
        zip_bytes = create_mock_zip(member_name="h244.ssp", content=b"SAS_DATA_BYTES")
        zip_path.write_bytes(zip_bytes)

        with self.assertRaises(IntegrityError) as ctx:
            validate_zip_archive(zip_path, allowed_member_extensions=[".dta"])
        self.assertIn("permitted archive member extension", str(ctx.exception))

    def test_validate_zip_archive_dta_attacks_rejected(self) -> None:
        # Multiple members
        multi_zip = self.temp_path / "multi_dta.zip"
        with zipfile.ZipFile(multi_zip, "w") as zf:
            zf.writestr("h244.dta", b"data1")
            zf.writestr("extra.txt", b"data2")
        with self.assertRaises(IntegrityError) as ctx:
            validate_zip_archive(multi_zip, allowed_member_extensions=[".dta"])
        self.assertIn("must contain exactly 1 member", str(ctx.exception))

        # Traversal
        trav_zip = self.temp_path / "trav_dta.zip"
        with zipfile.ZipFile(trav_zip, "w") as zf:
            zf.writestr("../h244.dta", b"data1")
        with self.assertRaises(SecurityError) as ctx:
            validate_zip_archive(trav_zip, allowed_member_extensions=[".dta"])
        self.assertIn("Path traversal", str(ctx.exception))

        # Backslash
        bs_zip = self.temp_path / "bs_dta.zip"
        with zipfile.ZipFile(bs_zip, "w") as zf:
            zf.writestr("dir\\h244.dta", b"data1")
        with self.assertRaises(SecurityError) as ctx:
            validate_zip_archive(bs_zip, allowed_member_extensions=[".dta"])
        self.assertIn("Backslash", str(ctx.exception))

        # Compression bomb
        bomb_zip = self.temp_path / "bomb_dta.zip"
        with zipfile.ZipFile(bomb_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("h244.dta", b"0" * 15_000_000)
        with self.assertRaises(SecurityError) as ctx:
            validate_zip_archive(bomb_zip, allowed_member_extensions=[".dta"], max_ratio=2.0)
        self.assertIn("compression ratio", str(ctx.exception))

    def test_download_stata_artifact_mock_workflow(self) -> None:
        zip_bytes = create_mock_zip(member_name="h244.dta", content=b"MOCK_STATA_DATA_FOR_H244")
        mock_response = MockHTTPResponse(
            data=zip_bytes,
            headers={"Content-Type": "application/zip", "Content-Length": str(len(zip_bytes))},
            status=200,
            url="https://meps.ahrq.gov/mepsweb/data_files/pufs/h244/h244dta.zip",
        )

        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_response

        manifest = ArtifactManifest(
            schema_version="1.0.0",
            metadata={"title": "Test Manifest"},
            artifacts=(
                Artifact(
                    artifact_id="hc244_stata_archive",
                    puf_id="HC-244",
                    panel_number=26,
                    survey_years="2021-2022",
                    artifact_type="data_archives",
                    description="Test Stata ZIP",
                    url="https://meps.ahrq.gov/mepsweb/data_files/pufs/h244/h244dta.zip",
                    relative_destination="h244/h244dta.zip",
                    expected_content_type="application/zip",
                    archive_allowed_member_extensions=(".dta",),
                ),
            ),
        )

        # 1. First run -> downloads and records provenance
        results = download_artifacts(
            manifest=manifest,
            policy=self.policy,
            output_root=self.output_root,
            provenance_path=self.provenance_path,
            opener=mock_opener,
        )

        self.assertIn("hc244_stata_archive", results)
        status, rec = results["hc244_stata_archive"]
        self.assertEqual(status, "downloaded")
        self.assertEqual(rec.byte_size, len(zip_bytes))
        self.assertIsNotNone(rec.archive_member_metadata)
        self.assertEqual(rec.archive_member_metadata["member_name"], "h244.dta")

        dest_file = self.output_root / "h244/h244dta.zip"
        self.assertTrue(dest_file.is_file())

        # 2. Second run -> pure skip
        results_rerun = download_artifacts(
            manifest=manifest,
            policy=self.policy,
            output_root=self.output_root,
            provenance_path=self.provenance_path,
            opener=mock_opener,
        )
        status_rerun, rec_rerun = results_rerun["hc244_stata_archive"]
        self.assertEqual(status_rerun, "skipped")
        self.assertEqual(rec_rerun.sha256, rec.sha256)


if __name__ == "__main__":
    unittest.main()
