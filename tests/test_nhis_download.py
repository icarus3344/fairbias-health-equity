"""Offline tests for the allowlisted NHIS downloader and ZIP extraction contract."""

from __future__ import annotations

import io
import pathlib
import sys
import tempfile
import unittest
import urllib.request
import zipfile
from unittest import mock

_SRC_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from nhis_fairbias.download import (
    NHISDownloadError,
    download_year,
    find_csv_member,
    validate_source_zip,
)


def _zip_bytes(member_name: str = "adult22.csv", content: bytes = b"SRVY_YR\n2022\n") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member_name, content)
    return buffer.getvalue()


class _Response:
    def __init__(self, payload: bytes, url: str) -> None:
        self._buffer = io.BytesIO(payload)
        self.status = 200
        self._url = url

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None


class TestNHISDownload(unittest.TestCase):
    def test_single_expected_csv_member_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = pathlib.Path(tmp) / "adult22csv.zip"
            zip_path.write_bytes(_zip_bytes())
            self.assertEqual(find_csv_member(zip_path, expected_name="adult22.csv"), "adult22.csv")

    def test_path_traversal_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = pathlib.Path(tmp) / "bad.zip"
            zip_path.write_bytes(_zip_bytes("../adult22.csv"))
            with self.assertRaises(NHISDownloadError):
                find_csv_member(zip_path, expected_name="adult22.csv")

    def test_download_is_idempotent_and_does_not_reopen_valid_file(self) -> None:
        config = {
            "schema_version": "test",
            "dataset": "test",
            "years": {
                "2022": {
                    "source_url": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2022/adult22csv.zip",
                    "local_source_file": "data/raw/nhis/2022/adult22csv.zip",
                    "local_csv_file": "data/raw/nhis/2022/adult22.csv",
                    "expected_csv_member": "adult22.csv",
                    "expected_raw_rows": 1,
                    "expected_meddl12m_counts": {"1": 1},
                    "study_role": "development_train",
                }
            },
            "outputs": {},
        }
        payload = _zip_bytes()
        url = config["years"]["2022"]["source_url"]
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = pathlib.Path(tmp)
            manifest = repo_root / "manifest.json"
            with mock.patch.object(urllib.request, "urlopen", return_value=_Response(payload, url)) as opener:
                first = download_year(
                    2022,
                    repo_root=repo_root,
                    study_config=config,
                    manifest_path=manifest,
                )
            self.assertTrue(first["downloaded"])
            self.assertEqual(opener.call_count, 1)
            with mock.patch.object(urllib.request, "urlopen", side_effect=AssertionError("redownloaded")):
                second = download_year(
                    2022,
                    repo_root=repo_root,
                    study_config=config,
                    manifest_path=manifest,
                )
            self.assertFalse(second["downloaded"])
            self.assertEqual(first["sha256"], second["sha256"])

    def test_invalid_existing_zip_fails_loudly(self) -> None:
        config = {
            "years": {
                "2022": {
                    "source_url": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2022/adult22csv.zip",
                    "local_source_file": "data/raw/nhis/2022/adult22csv.zip",
                    "local_csv_file": "data/raw/nhis/2022/adult22.csv",
                    "expected_csv_member": "adult22.csv",
                    "expected_raw_rows": 1,
                    "expected_meddl12m_counts": {"1": 1},
                    "study_role": "development_train",
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = pathlib.Path(tmp)
            zip_path = repo_root / "data/raw/nhis/2022/adult22csv.zip"
            zip_path.parent.mkdir(parents=True)
            zip_path.write_bytes(b"not a zip")
            with self.assertRaises(NHISDownloadError):
                validate_source_zip(2022, repo_root=repo_root, study_config=config)
