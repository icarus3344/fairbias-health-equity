"""Offline tests for Gate 15's exact-artifact downloader boundary.

These tests use only synthetic Artifact objects and temporary permission JSON.
They do not open a network connection or a MEPS file.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import MagicMock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from meps_fairness.data.download import (
    Artifact,
    ArtifactManifest,
    DataAccessPolicy,
    SecurityError,
    download_artifacts,
    validate_artifact_download_scope,
)
from meps_fairness.gate15_source_review import (
    DEFAULT_PERMISSION_PATH,
    GATE14B_ACCEPTED_STATUS,
    PERMISSION_AUTHORIZED_STATUS,
    SCHEMA_CODEBOOK_ONLY_STAGE,
    load_gate15_artifact_permission,
)


class TestGate15ScopedDownloader(unittest.TestCase):
    def _authorized_permission_path(self) -> tuple[tempfile.TemporaryDirectory, pathlib.Path]:
        temp_dir = tempfile.TemporaryDirectory()
        path = pathlib.Path(temp_dir.name) / "permission.json"
        permission = json.loads(DEFAULT_PERMISSION_PATH.read_text(encoding="utf-8"))
        permission["permission_status"] = PERMISSION_AUTHORIZED_STATUS
        permission["active_stage"] = SCHEMA_CODEBOOK_ONLY_STAGE
        permission["gate14b_prerequisite_status"] = GATE14B_ACCEPTED_STATUS
        permission["schema_stage_authorized"] = True
        permission["stages"][SCHEMA_CODEBOOK_ONLY_STAGE]["enabled"] = True
        permission["stages"][SCHEMA_CODEBOOK_ONLY_STAGE]["local_artifact_read_allowed"] = True
        path.write_text(json.dumps(permission), encoding="utf-8")
        return temp_dir, path

    @staticmethod
    def _artifact_for(permission_item, **overrides) -> Artifact:
        values = {
            "artifact_id": permission_item.artifact_id,
            "puf_id": permission_item.puf_id,
            "panel_number": permission_item.panel_number,
            "survey_years": f"{permission_item.survey_years[0]}-{permission_item.survey_years[1]}",
            "artifact_type": permission_item.artifact_type,
            "description": "synthetic test artifact",
            "url": permission_item.url,
            "relative_destination": permission_item.relative_destination,
            "expected_content_type": permission_item.expected_content_type,
            "archive_allowed_member_extensions": permission_item.archive_allowed_member_extensions,
        }
        values.update(overrides)
        return Artifact(**values)

    def test_pending_scope_rejects_before_downloader_can_open_network(self) -> None:
        permission = load_gate15_artifact_permission()
        artifact = self._artifact_for(permission.artifacts[0])
        with self.assertRaises(SecurityError):
            validate_artifact_download_scope(artifact, permission.download_scope)

    def test_authorized_scope_matches_exact_stata_artifact_without_puf_list_entry(self) -> None:
        temp_dir, permission_path = self._authorized_permission_path()
        self.addCleanup(temp_dir.cleanup)
        permission = load_gate15_artifact_permission(permission_path)
        artifact = self._artifact_for(permission.artifacts[0])
        # The exact scope, not a generic authorized_pufs entry, is the grant
        # used by the scoped validator.  No network call is made here.
        validate_artifact_download_scope(artifact, permission.download_scope)

    def test_wrong_official_url_is_rejected_by_exact_scope(self) -> None:
        temp_dir, permission_path = self._authorized_permission_path()
        self.addCleanup(temp_dir.cleanup)
        permission = load_gate15_artifact_permission(permission_path)
        artifact = self._artifact_for(
            permission.artifacts[0],
            url="https://meps.ahrq.gov/mepsweb/data_files/pufs/h217/other.zip",
        )
        with self.assertRaises(SecurityError):
            validate_artifact_download_scope(artifact, permission.download_scope)

    def test_archive_scope_rejects_non_dta_member_allowlist(self) -> None:
        temp_dir, permission_path = self._authorized_permission_path()
        self.addCleanup(temp_dir.cleanup)
        permission = load_gate15_artifact_permission(permission_path)
        artifact = self._artifact_for(
            permission.artifacts[0],
            archive_allowed_member_extensions=(".ssp",),
        )
        with self.assertRaises(SecurityError):
            validate_artifact_download_scope(artifact, permission.download_scope)

    def test_scoped_download_path_is_checked_before_opener_use(self) -> None:
        permission = load_gate15_artifact_permission()
        artifact = self._artifact_for(permission.artifacts[0])
        opener = MagicMock()
        with tempfile.TemporaryDirectory() as temp_dir:
            from meps_fairness.data.download import download_single_artifact

            policy = DataAccessPolicy(
                schema_version="1.0.0",
                download_authorization=True,
                allowed_scheme="https",
                allowed_hosts=frozenset({"meps.ahrq.gov"}),
                enforce_same_host_redirects=True,
                allowed_artifact_types=frozenset({"data_archives", "documentation", "codebooks"}),
                authorized_puf_ids=frozenset(),
            )
            with self.assertRaises(SecurityError):
                download_single_artifact(
                    artifact=artifact,
                    policy=policy,
                    output_root=pathlib.Path(temp_dir),
                    opener=opener,
                    existing_records={},
                    artifact_scope=permission.download_scope,
                )
        opener.open.assert_not_called()

    def test_scoped_batch_is_prevalidated_before_any_output_or_opener_use(self) -> None:
        temp_dir, permission_path = self._authorized_permission_path()
        self.addCleanup(temp_dir.cleanup)
        permission = load_gate15_artifact_permission(permission_path)
        valid = self._artifact_for(permission.artifacts[0])
        invalid = self._artifact_for(
            permission.artifacts[1],
            url="https://meps.ahrq.gov/mepsweb/data_stats/download_data/pufs/h217/not-the-doc.pdf",
        )
        manifest = ArtifactManifest(
            schema_version="0.1.0",
            metadata={},
            artifacts=(valid, invalid),
        )
        policy = DataAccessPolicy(
            schema_version="1.0.0",
            download_authorization=True,
            allowed_scheme="https",
            allowed_hosts=frozenset({"meps.ahrq.gov"}),
            enforce_same_host_redirects=True,
            allowed_artifact_types=frozenset({"data_archives", "documentation", "codebooks"}),
            authorized_puf_ids=frozenset(),
        )
        opener = MagicMock()
        with tempfile.TemporaryDirectory() as output_dir:
            output_root = pathlib.Path(output_dir) / "out"
            with self.assertRaises(SecurityError):
                download_artifacts(
                    manifest=manifest,
                    policy=policy,
                    output_root=output_root,
                    provenance_path=output_root / "provenance.json",
                    opener=opener,
                    artifact_scope=permission.download_scope,
                )
            self.assertFalse(output_root.exists())
        opener.open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
