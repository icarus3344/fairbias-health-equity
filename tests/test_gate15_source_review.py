"""Pure metadata tests for the staged Gate 15 source review.

These tests load JSON and synthetic mutations only.  They do not open a MEPS
archive, read a PDF, inspect outcomes, run a pipeline, train a model, run
bootstrap inference, or access Panel 27.
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from meps_fairness.gate15_source_review import (
    DEFAULT_ACCESS_PATH,
    DEFAULT_PERMISSION_PATH,
    GATE14B_ACCEPTED_STATUS,
    GATE14B_PENDING_STATUS,
    HC217_DETAILS_URL,
    HC217_SOURCE_ARTIFACTS,
    HC217_SURVEY_YEARS,
    PERMISSION_AUTHORIZED_STATUS,
    SCHEMA_CODEBOOK_ONLY_STAGE,
    SEMANTIC_COMPLETE_STATUS,
    Gate15SourceReviewError,
    load_gate15_artifact_permission,
    load_gate15_source_preflight,
    require_schema_stage_authorized,
    validate_gate15_artifact_permission,
    validate_gate15_source_preflight,
)


REVIEW_PATH = ROOT / "configs" / "gate15_hc217_source_review.json"


def _load() -> dict:
    with REVIEW_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


class TestGate15SourceReview(unittest.TestCase):
    def assert_code(self, code: str, function, *args, **kwargs) -> None:
        with self.assertRaises(Gate15SourceReviewError) as context:
            function(*args, **kwargs)
        self.assertEqual(context.exception.code, code)

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

    def _authorized_schema_review(self, *, semantic_status: str) -> dict:
        review = copy.deepcopy(_load())
        review["review_status"] = "SCHEMA_REVIEW_COMPLETE"
        for artifact in review["source"]["artifacts"]:
            artifact["sha256"] = "a" * 64
            artifact["sha256_status"] = "VERIFIED_LOCAL_SHA256"
        review["semantic_review"]["status"] = semantic_status
        review["authorization"]["schema_read_allowed"] = True
        review["schema"] = {
            "status": "SCHEMA_VERIFIED",
            "column_names": ["PANEL", "DUID"],
            "column_types": {"PANEL": "int64", "DUID": "int64"},
            "schema_sha256": "b" * 64,
        }
        return review

    def test_hc217_metadata_preflight_passes_but_schema_is_not_ready(self) -> None:
        report = load_gate15_source_preflight()
        self.assertEqual(report.puf_id, "HC-217")
        self.assertEqual(report.panel_number, 23)
        self.assertFalse(report.schema_stage_authorized)
        self.assertFalse(report.ready_for_schema_review)
        self.assertFalse(report.structural_checks_complete)
        self.assertFalse(report.outcome_accessed)
        self.assertFalse(report.panel_27_accessed)
        self.assertFalse(report.source_hashes_complete)

    def test_schema_stage_requires_explicit_puf_scope(self) -> None:
        self.assert_code(
            "GATE15_DATA_SCOPE_NOT_AUTHORIZED",
            require_schema_stage_authorized,
            "HC-217",
            access_config_path=DEFAULT_ACCESS_PATH,
        )

    def test_panel27_is_not_a_gate15_candidate(self) -> None:
        self.assert_code(
            "GATE15_PANEL27_LOCKED",
            require_schema_stage_authorized,
            "HC-252",
            access_config_path=DEFAULT_ACCESS_PATH,
        )

    def test_source_must_stay_on_official_https_host(self) -> None:
        review = _load()
        review["source"]["details_url"] = "https://example.invalid/hc217"
        self.assert_code("GATE15_SOURCE_HOST_INVALID", validate_gate15_source_preflight, review)

    def test_hc217_details_url_is_exact_not_merely_official_host(self) -> None:
        review = _load()
        self.assertEqual(review["source"]["details_url"], HC217_DETAILS_URL)
        review["source"]["details_url"] = "https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_results.jsp"
        self.assert_code("GATE15_SOURCE_URL_INVALID", validate_gate15_source_preflight, review)

    def test_hc217_artifact_url_is_exact_not_merely_official_host(self) -> None:
        review = _load()
        review["source"]["artifacts"][0]["url"] = "https://meps.ahrq.gov/mepsweb/data_files/pufs/h217/other.zip"
        self.assert_code("GATE15_SOURCE_URL_INVALID", validate_gate15_source_preflight, review)

    def test_unauthorized_schema_claim_fails_closed(self) -> None:
        review = _load()
        review["schema"]["status"] = "SCHEMA_VERIFIED"
        review["schema"]["column_names"] = ["PANEL"]
        self.assert_code(
            "GATE15_SCHEMA_CLAIM_WITHOUT_AUTHORIZATION",
            validate_gate15_source_preflight,
            review,
        )

    def test_missing_hash_cannot_be_marked_verified(self) -> None:
        review = _load()
        review["source"]["artifacts"][0]["sha256_status"] = "VERIFIED_LOCAL_SHA256"
        self.assert_code("GATE15_SOURCE_HASH_INVALID", validate_gate15_source_preflight, review)

    def test_outcome_or_panel27_access_is_rejected(self) -> None:
        review = copy.deepcopy(_load())
        review["authorization"]["outcome_values_read"] = True
        self.assert_code("GATE15_BOUNDARY_VIOLATION", validate_gate15_source_preflight, review)

    def test_wrong_candidate_order_is_rejected(self) -> None:
        review = _load()
        review["panel"]["candidate_order"] = 2
        self.assert_code("GATE15_PANEL_ORDER_INVALID", validate_gate15_source_preflight, review)

    def test_documented_years_are_locked_to_panel_metadata(self) -> None:
        review = _load()
        review["documented_structure"]["two_year_period"] = [2017, 2018]
        self.assert_code(
            "GATE15_CALENDAR_STRUCTURE_INVALID",
            validate_gate15_source_preflight,
            review,
        )

    def test_hc217_years_are_locked_to_the_exact_panel_period(self) -> None:
        review = _load()
        review["panel"]["baseline_year"] = 1900
        review["panel"]["follow_up_year"] = 1901
        review["documented_structure"]["two_year_period"] = [1900, 1901]
        self.assert_code(
            "GATE15_CALENDAR_STRUCTURE_INVALID",
            validate_gate15_source_preflight,
            review,
        )

    def test_exact_permission_maps_gate15_types_and_requires_dta(self) -> None:
        permission = load_gate15_artifact_permission()
        self.assertEqual(permission.permission_status, "PENDING_SUPERVISOR_AUTHORIZATION")
        self.assertEqual(permission.gate14b_prerequisite_status, GATE14B_PENDING_STATUS)
        self.assertFalse(permission.schema_stage_authorized)
        self.assertEqual(permission.survey_years, HC217_SURVEY_YEARS)
        self.assertEqual(
            tuple(item.artifact_id for item in permission.artifacts),
            tuple(HC217_SOURCE_ARTIFACTS),
        )
        self.assertEqual(permission.artifacts[0].artifact_type, "data_archives")
        self.assertEqual(permission.artifacts[0].archive_allowed_member_extensions, (".dta",))
        self.assertEqual(permission.artifacts[2].artifact_type, "codebooks")

    def test_broad_puf_scope_does_not_unlock_pending_exact_permission(self) -> None:
        access = json.loads(DEFAULT_ACCESS_PATH.read_text(encoding="utf-8"))
        access["scope"]["authorized_pufs"].append(
            {
                "puf_id": "HC-217",
                "name": "temporary test only",
                "survey_years": "2018-2019",
                "role": "temporary test only",
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            access_path = pathlib.Path(temp_dir) / "data_access.json"
            access_path.write_text(json.dumps(access), encoding="utf-8")
            self.assert_code(
                "GATE15_DATA_SCOPE_NOT_AUTHORIZED",
                require_schema_stage_authorized,
                "HC-217",
                access_config_path=access_path,
            )

    def test_gate15_authorization_cannot_bypass_unaccepted_gate14b(self) -> None:
        permission = json.loads(DEFAULT_PERMISSION_PATH.read_text(encoding="utf-8"))
        permission["permission_status"] = PERMISSION_AUTHORIZED_STATUS
        permission["active_stage"] = SCHEMA_CODEBOOK_ONLY_STAGE
        permission["schema_stage_authorized"] = True
        permission["stages"][SCHEMA_CODEBOOK_ONLY_STAGE]["enabled"] = True
        permission["stages"][SCHEMA_CODEBOOK_ONLY_STAGE]["local_artifact_read_allowed"] = True
        with self.assertRaises(Gate15SourceReviewError) as context:
            validate_gate15_artifact_permission(permission)
        self.assertEqual(context.exception.code, "GATE15_PREREQUISITE_NOT_ACCEPTED")

    def test_schema_verified_but_pending_semantics_is_not_structural_closure(self) -> None:
        temp_dir, permission_path = self._authorized_permission_path()
        self.addCleanup(temp_dir.cleanup)
        report = validate_gate15_source_preflight(
            self._authorized_schema_review(semantic_status="PENDING_SCHEMA_AND_CODEBOOK_REVIEW"),
            permission_path=permission_path,
        )
        self.assertFalse(report.semantic_review_complete)
        self.assertFalse(report.structural_checks_complete)
        self.assertFalse(report.ready_for_schema_review)

    def test_completed_semantics_is_required_for_structural_closure(self) -> None:
        temp_dir, permission_path = self._authorized_permission_path()
        self.addCleanup(temp_dir.cleanup)
        report = validate_gate15_source_preflight(
            self._authorized_schema_review(semantic_status=SEMANTIC_COMPLETE_STATUS),
            permission_path=permission_path,
        )
        self.assertTrue(report.semantic_review_complete)
        self.assertTrue(report.structural_checks_complete)
        self.assertTrue(report.ready_for_schema_review)

    def test_semantic_checklist_order_is_locked(self) -> None:
        review = _load()
        review["semantic_review"]["required_checks"] = list(
            reversed(review["semantic_review"]["required_checks"])
        )
        self.assert_code(
            "GATE15_SEMANTIC_REVIEW_INVALID",
            validate_gate15_source_preflight,
            review,
        )

    def test_content_type_must_match_artifact_kind(self) -> None:
        review = _load()
        review["source"]["artifacts"][0]["content_type"] = "application/pdf"
        self.assert_code(
            "GATE15_SOURCE_METADATA_INVALID",
            validate_gate15_source_preflight,
            review,
        )

    def test_metadata_status_cannot_be_promoted_without_schema_access(self) -> None:
        review = _load()
        review["review_status"] = "SCHEMA_REVIEW_COMPLETE"
        self.assert_code(
            "GATE15_SCHEMA_CLAIM_WITHOUT_AUTHORIZATION",
            validate_gate15_source_preflight,
            review,
        )


if __name__ == "__main__":
    unittest.main()
