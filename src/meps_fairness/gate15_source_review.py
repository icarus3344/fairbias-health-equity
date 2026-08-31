"""Fail-closed Gate 15 source/schema preflight contracts.

Gate 15 is intentionally staged.  This module validates the official-source
metadata for one development panel at a time and refuses to treat a missing
data-scope authorization, an absent local SHA-256, or an unverified schema as
structural closure.  It never opens a MEPS data archive and it has no outcome,
model, bootstrap, or Panel 27 operations.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

from meps_fairness.data.download import (
    ArtifactDownloadPermission,
    ArtifactDownloadScope,
)


GATE15: Final = "15"
OFFICIAL_HOST: Final = "meps.ahrq.gov"
PANEL_ORDER: Final = ("HC-217", "HC-225", "HC-234", "HC-244")
PREFLIGHT_STATUS: Final = "SOURCE_METADATA_PREFLIGHT_ONLY"
SCHEMA_PENDING_STATUS: Final = "NOT_READ_PENDING_DATA_SCOPE"
SCHEMA_VERIFIED_STATUS: Final = "SCHEMA_VERIFIED"
SEMANTIC_PENDING_STATUS: Final = "PENDING_SCHEMA_AND_CODEBOOK_REVIEW"
SEMANTIC_COMPLETE_STATUS: Final = "SEMANTIC_REVIEW_COMPLETE"
SCHEMA_CODEBOOK_ONLY_STAGE: Final = "SCHEMA_CODEBOOK_ONLY"
PERMISSION_PENDING_STATUS: Final = "PENDING_SUPERVISOR_AUTHORIZATION"
PERMISSION_AUTHORIZED_STATUS: Final = "SUPERVISOR_AUTHORIZED"
GATE14B_PENDING_STATUS: Final = "PENDING_CODEX_ACCEPTANCE"
GATE14B_ACCEPTED_STATUS: Final = "ACCEPTED_AND_COMMITTED"

HC217_PUF_ID: Final = "HC-217"
HC217_PANEL_NUMBER: Final = 23
HC217_SURVEY_YEARS: Final = (2018, 2019)
HC217_DETAILS_URL: Final = (
    "https://meps.ahrq.gov/mepsweb/data_stats/"
    "download_data_files_detail.jsp?cboPufNumber=HC-217"
)
HC217_SOURCE_ARTIFACTS: Final = {
    "hc217_stata_archive": {
        "artifact_type": "archive",
        "url": "https://meps.ahrq.gov/mepsweb/data_files/pufs/h217/h217dta.zip",
    },
    "hc217_documentation": {
        "artifact_type": "documentation",
        "url": "https://meps.ahrq.gov/mepsweb/data_stats/download_data/pufs/h217/h217doc.pdf",
    },
    "hc217_codebook": {
        "artifact_type": "codebook",
        "url": "https://meps.ahrq.gov/mepsweb/data_stats/download_data/pufs/h217/h217cb.pdf",
    },
}
_SOURCE_TO_DOWNLOADER_ARTIFACT_TYPE: Final = {
    "archive": "data_archives",
    "documentation": "documentation",
    "codebook": "codebooks",
}
_HC217_DOWNLOAD_DETAILS: Final = {
    "hc217_stata_archive": {
        "downloader_artifact_type": "data_archives",
        "relative_destination": "h217/h217dta.zip",
        "expected_content_type": "application/zip",
        "archive_allowed_member_extensions": (".dta",),
    },
    "hc217_documentation": {
        "downloader_artifact_type": "documentation",
        "relative_destination": "h217/h217doc.pdf",
        "expected_content_type": "application/pdf",
        "archive_allowed_member_extensions": (),
    },
    "hc217_codebook": {
        "downloader_artifact_type": "codebooks",
        "relative_destination": "h217/h217cb.pdf",
        "expected_content_type": "application/pdf",
        "archive_allowed_member_extensions": (),
    },
}

DEFAULT_REVIEW_PATH: Final = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "gate15_hc217_source_review.json"
)
DEFAULT_ACCESS_PATH: Final = (
    Path(__file__).resolve().parents[2] / "configs" / "data_access.json"
)
DEFAULT_PERMISSION_PATH: Final = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "gate15_hc217_artifact_permissions.json"
)

_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_ARTIFACT_TYPES = frozenset({"archive", "documentation", "codebook"})
_ARTIFACT_CONTENT_TYPES = {
    "archive": "application/zip",
    "documentation": "application/pdf",
    "codebook": "application/pdf",
}
_REQUIRED_SEMANTIC_CHECKS = (
    "variable labels and substantive meaning",
    "calendar timing and round placement",
    "valid, missing, negative, and unknown codes",
    "universe and eligibility restrictions",
    "numeric versus categorical type and scale",
    "LONGWT target population and construction",
    "VARSTR and VARPSU sampling-design meaning",
    "identifier and calendar-year structure",
)


class Gate15SourceReviewError(ValueError):
    """Stable, machine-readable failure for a Gate 15 contract violation."""

    def __init__(self, code: str, message: str) -> None:
        if not code or any(character.isspace() for character in code):
            raise ValueError("Gate 15 error codes must be non-empty and whitespace-free")
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclasses.dataclass(frozen=True)
class Gate15SourcePreflightReport:
    """Validated status for one panel's Gate 15 source preflight."""

    gate: str
    puf_id: str
    panel_number: int
    baseline_year: int
    follow_up_year: int
    status: str
    schema_stage_authorized: bool
    schema_status: str
    source_hashes_complete: bool
    structural_checks_complete: bool
    semantic_review_complete: bool
    outcome_accessed: bool
    panel_27_accessed: bool

    @property
    def ready_for_schema_review(self) -> bool:
        """Whether the record has completed the authorized schema-only stage."""

        return self.structural_checks_complete


@dataclasses.dataclass(frozen=True)
class Gate15ArtifactPermission:
    """Validated, artifact-specific permission contract for one Gate 15 PUF."""

    gate: str
    puf_id: str
    panel_number: int
    survey_years: tuple[int, int]
    permission_status: str
    active_stage: str
    gate14b_prerequisite_status: str
    schema_stage_authorized: bool
    artifacts: tuple[ArtifactDownloadPermission, ...]
    download_scope: ArtifactDownloadScope


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Gate15SourceReviewError("GATE15_RECORD_INVALID", f"{label} must be an object")
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Gate15SourceReviewError("GATE15_RECORD_INVALID", f"{label} must be non-empty text")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise Gate15SourceReviewError("GATE15_RECORD_INVALID", f"{label} must be boolean")
    return value


def _require_int(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Gate15SourceReviewError("GATE15_RECORD_INVALID", f"{label} must be an integer")
    if positive and value <= 0:
        raise Gate15SourceReviewError("GATE15_RECORD_INVALID", f"{label} must be positive")
    return value


def _validate_official_url(value: Any, label: str) -> str:
    url = _require_text(value, label)
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "https" or parsed.hostname != OFFICIAL_HOST:
        raise Gate15SourceReviewError(
            "GATE15_SOURCE_HOST_INVALID",
            f"{label} must use https://{OFFICIAL_HOST}",
        )
    if parsed.username or parsed.password or parsed.port is not None:
        raise Gate15SourceReviewError(
            "GATE15_SOURCE_HOST_INVALID",
            f"{label} must not contain credentials or a non-default port",
        )
    return url


def _load_json_object(path: str | Path, *, error_code: str, label: str) -> Mapping[str, Any]:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise Gate15SourceReviewError(error_code, f"{label} cannot be read as JSON") from exc
    return _require_mapping(value, label)


def _require_year_pair(value: Any, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            f"{label} must contain exactly two integer years",
        )
    return int(value[0]), int(value[1])


def _require_exact_list(value: Any, expected: tuple[str, ...], label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or tuple(value) != expected:
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            f"{label} is missing, reordered, or contains unexpected values",
        )
    return expected


def _validate_permission_artifact(item: Mapping[str, Any]) -> ArtifactDownloadPermission:
    artifact_id = _require_text(item.get("artifact_id"), "permission artifact_id")
    source_spec = HC217_SOURCE_ARTIFACTS.get(artifact_id)
    download_spec = _HC217_DOWNLOAD_DETAILS.get(artifact_id)
    if source_spec is None or download_spec is None:
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            "permission artifact is not one of the three exact HC-217 artifacts",
        )

    if item.get("puf_id") != HC217_PUF_ID or item.get("panel_number") != HC217_PANEL_NUMBER:
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            f"{artifact_id} must remain bound to HC-217 Panel 23",
        )
    if _require_year_pair(item.get("survey_years"), f"{artifact_id}.survey_years") != HC217_SURVEY_YEARS:
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            f"{artifact_id} must remain bound to survey years 2018 and 2019",
        )
    if item.get("source_artifact_type") != source_spec["artifact_type"]:
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            f"{artifact_id} source artifact type disagrees with the exact source contract",
        )

    url = _validate_official_url(item.get("url"), f"{artifact_id}.url")
    if url != source_spec["url"]:
        raise Gate15SourceReviewError(
            "GATE15_SOURCE_URL_INVALID",
            f"{artifact_id}.url is not the exact HC-217 endpoint",
        )
    if item.get("downloader_artifact_type") != download_spec["downloader_artifact_type"]:
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            f"{artifact_id} has no valid mapping to the downloader artifact type",
        )
    if item.get("relative_destination") != download_spec["relative_destination"]:
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            f"{artifact_id} destination disagrees with the exact scoped destination",
        )
    if item.get("expected_content_type") != download_spec["expected_content_type"]:
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            f"{artifact_id} content type disagrees with the exact scoped type",
        )
    extensions = _require_exact_list(
        item.get("archive_allowed_member_extensions"),
        tuple(download_spec["archive_allowed_member_extensions"]),
        f"{artifact_id}.archive_allowed_member_extensions",
    )
    return ArtifactDownloadPermission(
        artifact_id=artifact_id,
        puf_id=HC217_PUF_ID,
        panel_number=HC217_PANEL_NUMBER,
        survey_years=HC217_SURVEY_YEARS,
        artifact_type=download_spec["downloader_artifact_type"],
        url=url,
        relative_destination=download_spec["relative_destination"],
        expected_content_type=download_spec["expected_content_type"],
        archive_allowed_member_extensions=extensions,
    )


def validate_gate15_artifact_permission(
    permission: Mapping[str, Any],
) -> Gate15ArtifactPermission:
    """Validate the separate exact-artifact/stage permission contract.

    The permission record is intentionally independent of
    ``configs/data_access.json``.  A PUF-level entry in that broad record is
    not enough to authorize this stage.
    """

    root = _require_mapping(permission, "permission")
    if root.get("gate") != GATE15:
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            "permission.gate must be '15'",
        )
    if root.get("puf_id") != HC217_PUF_ID or root.get("panel_number") != HC217_PANEL_NUMBER:
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            "permission must be bound to HC-217 Panel 23",
        )
    if _require_year_pair(root.get("survey_years"), "permission.survey_years") != HC217_SURVEY_YEARS:
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            "permission must be bound to survey years 2018 and 2019",
        )

    permission_status = _require_text(
        root.get("permission_status"), "permission.permission_status"
    )
    if permission_status not in {PERMISSION_PENDING_STATUS, PERMISSION_AUTHORIZED_STATUS}:
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            "unsupported permission_status",
        )
    active_stage = _require_text(root.get("active_stage"), "permission.active_stage")
    if active_stage not in {PREFLIGHT_STATUS, SCHEMA_CODEBOOK_ONLY_STAGE}:
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            "unsupported active Gate 15 stage",
        )
    gate14b_prerequisite_status = _require_text(
        root.get("gate14b_prerequisite_status"),
        "permission.gate14b_prerequisite_status",
    )
    if gate14b_prerequisite_status not in {
        GATE14B_PENDING_STATUS,
        GATE14B_ACCEPTED_STATUS,
    }:
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            "unsupported Gate 14B prerequisite status",
        )
    schema_stage_authorized = _require_bool(
        root.get("schema_stage_authorized"), "permission.schema_stage_authorized"
    )
    if schema_stage_authorized != (permission_status == PERMISSION_AUTHORIZED_STATUS):
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            "schema_stage_authorized disagrees with permission_status",
        )
    if permission_status == PERMISSION_PENDING_STATUS and active_stage != PREFLIGHT_STATUS:
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            "pending permission must remain in SOURCE_METADATA_PREFLIGHT_ONLY",
        )
    if permission_status == PERMISSION_AUTHORIZED_STATUS and active_stage != SCHEMA_CODEBOOK_ONLY_STAGE:
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            "authorized permission must identify the schema/codebook-only stage",
        )
    if permission_status == PERMISSION_AUTHORIZED_STATUS and gate14b_prerequisite_status != GATE14B_ACCEPTED_STATUS:
        raise Gate15SourceReviewError(
            "GATE15_PREREQUISITE_NOT_ACCEPTED",
            "Gate 14B must be independently accepted and committed before Gate 15 authorization",
        )

    for field in ("outcome_values_read", "microdata_rows_read", "panel_27_accessed"):
        if _require_bool(root.get(field), f"permission.{field}"):
            raise Gate15SourceReviewError(
                "GATE15_BOUNDARY_VIOLATION",
                f"permission.{field} must remain false",
            )

    stages = _require_mapping(root.get("stages"), "permission.stages")
    if set(stages) != {PREFLIGHT_STATUS, SCHEMA_CODEBOOK_ONLY_STAGE}:
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            "permission.stages must contain exactly the metadata and schema/codebook stages",
        )
    metadata_stage = _require_mapping(stages.get(PREFLIGHT_STATUS), "metadata stage")
    schema_stage = _require_mapping(stages.get(SCHEMA_CODEBOOK_ONLY_STAGE), "schema stage")
    if not _require_bool(metadata_stage.get("enabled"), "metadata stage.enabled"):
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            "source metadata preflight must remain enabled",
        )
    if _require_bool(metadata_stage.get("local_artifact_read_allowed"), "metadata stage.local_artifact_read_allowed"):
        raise Gate15SourceReviewError(
            "GATE15_BOUNDARY_VIOLATION",
            "metadata preflight cannot read local artifacts",
        )
    _require_exact_list(
        metadata_stage.get("allowed_operations"),
        ("HEAD_METADATA",),
        "metadata stage.allowed_operations",
    )
    if metadata_stage.get("allowed_artifact_ids") != []:
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            "metadata stage cannot grant local artifact IDs",
        )

    if _require_bool(schema_stage.get("enabled"), "schema stage.enabled") != schema_stage_authorized:
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            "schema stage.enabled disagrees with the root authorization state",
        )
    if _require_bool(
        schema_stage.get("local_artifact_read_allowed"),
        "schema stage.local_artifact_read_allowed",
    ) != schema_stage_authorized:
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            "schema stage local-read permission disagrees with the root authorization state",
        )
    expected_ids = tuple(HC217_SOURCE_ARTIFACTS)
    _require_exact_list(
        schema_stage.get("allowed_artifact_ids"),
        expected_ids,
        "schema stage.allowed_artifact_ids",
    )
    allowed_types = schema_stage.get("allowed_artifact_types")
    if (
        not isinstance(allowed_types, list)
        or any(not isinstance(item, str) for item in allowed_types)
        or frozenset(allowed_types) != frozenset(_SOURCE_TO_DOWNLOADER_ARTIFACT_TYPE.values())
    ):
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            "schema stage artifact types must map to data_archives/documentation/codebooks",
        )
    for field in ("outcome_values_read", "microdata_rows_read", "panel_27_accessed"):
        if _require_bool(schema_stage.get(field), f"schema stage.{field}"):
            raise Gate15SourceReviewError(
                "GATE15_BOUNDARY_VIOLATION",
                f"schema stage.{field} must remain false",
            )
    _require_exact_list(
        schema_stage.get("allowed_operations"),
        (
            "DOWNLOAD_EXACT_ARTIFACTS",
            "READ_SCHEMA_COLUMNS_AND_TYPES",
            "READ_DOCUMENTATION",
            "READ_CODEBOOK",
        ),
        "schema stage.allowed_operations",
    )

    raw_artifacts = root.get("artifacts")
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != len(expected_ids):
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            "permission.artifacts must contain exactly the three HC-217 artifacts",
        )
    artifacts = tuple(_validate_permission_artifact(_require_mapping(item, "permission artifact")) for item in raw_artifacts)
    if tuple(item.artifact_id for item in artifacts) != expected_ids:
        raise Gate15SourceReviewError(
            "GATE15_PERMISSION_INVALID",
            "permission artifacts must use the fixed three-artifact order",
        )

    download_scope = ArtifactDownloadScope(
        gate=GATE15,
        puf_id=HC217_PUF_ID,
        panel_number=HC217_PANEL_NUMBER,
        survey_years=HC217_SURVEY_YEARS,
        stage=active_stage,
        authorization_status=permission_status,
        allowed_artifact_types=frozenset(allowed_types),
        allowed_artifacts=artifacts,
        outcome_values_read=False,
        microdata_rows_read=False,
        panel_27_accessed=False,
        gate14b_prerequisite_status=gate14b_prerequisite_status,
    )
    return Gate15ArtifactPermission(
        gate=GATE15,
        puf_id=HC217_PUF_ID,
        panel_number=HC217_PANEL_NUMBER,
        survey_years=HC217_SURVEY_YEARS,
        permission_status=permission_status,
        active_stage=active_stage,
        gate14b_prerequisite_status=gate14b_prerequisite_status,
        schema_stage_authorized=schema_stage_authorized,
        artifacts=artifacts,
        download_scope=download_scope,
    )


def load_gate15_artifact_permission(
    path: str | Path = DEFAULT_PERMISSION_PATH,
) -> Gate15ArtifactPermission:
    """Load and validate the separate Gate 15 exact-artifact permission file."""

    permission = _load_json_object(
        path,
        error_code="GATE15_PERMISSION_UNREADABLE",
        label="Gate 15 artifact permission",
    )
    return validate_gate15_artifact_permission(permission)


def require_schema_stage_authorized(
    puf_id: str,
    *,
    access_config_path: str | Path = DEFAULT_ACCESS_PATH,
    permission_path: str | Path = DEFAULT_PERMISSION_PATH,
) -> None:
    """Refuse schema access unless the exact Gate 15 stage is authorized.

    ``access_config_path`` is retained for API compatibility only.  A broad
    PUF entry there is intentionally not treated as sufficient authority for
    HC-217; the separate exact-artifact permission must be supervisor-authorized.
    """

    del access_config_path
    normalized = _require_text(puf_id, "puf_id")
    if normalized == "HC-252":
        raise Gate15SourceReviewError(
            "GATE15_PANEL27_LOCKED",
            "Panel 27 is a locked temporal holdout outside Gate 15",
        )
    if normalized not in PANEL_ORDER:
        raise Gate15SourceReviewError(
            "GATE15_PANEL_NOT_CANDIDATE",
            "schema review accepts only the fixed Gate 15 candidate sequence",
        )
    permission = load_gate15_artifact_permission(permission_path)
    if (
        permission.puf_id != normalized
        or not permission.schema_stage_authorized
        or permission.active_stage != SCHEMA_CODEBOOK_ONLY_STAGE
    ):
        raise Gate15SourceReviewError(
            "GATE15_DATA_SCOPE_NOT_AUTHORIZED",
            f"{normalized} lacks an exact supervisor-authorized schema/codebook permission",
        )


def _validate_artifacts(source: Mapping[str, Any], *, puf_id: str) -> bool:
    details_url = _validate_official_url(source.get("details_url"), "source.details_url")
    expected_source_artifacts = (
        HC217_SOURCE_ARTIFACTS if puf_id == HC217_PUF_ID else None
    )
    if expected_source_artifacts is not None and details_url != HC217_DETAILS_URL:
        raise Gate15SourceReviewError(
            "GATE15_SOURCE_URL_INVALID",
            "source.details_url is not the exact HC-217 details endpoint",
        )
    raw_artifacts = source.get("artifacts")
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != 3:
        raise Gate15SourceReviewError(
            "GATE15_SOURCE_METADATA_INVALID",
            "source.artifacts must contain exactly archive, documentation, and codebook",
        )

    seen_types: set[str] = set()
    hashes_complete = True
    for artifact in raw_artifacts:
        item = _require_mapping(artifact, "source artifact")
        artifact_type = _require_text(item.get("artifact_type"), "artifact_type")
        if artifact_type not in _ARTIFACT_TYPES or artifact_type in seen_types:
            raise Gate15SourceReviewError(
                "GATE15_SOURCE_METADATA_INVALID",
                "source artifact types must be unique archive/documentation/codebook values",
        )
        seen_types.add(artifact_type)
        artifact_id = _require_text(item.get("artifact_id"), "artifact_id")
        expected_spec = (
            expected_source_artifacts.get(artifact_id)
            if expected_source_artifacts is not None
            else None
        )
        if expected_source_artifacts is not None and expected_spec is None:
            raise Gate15SourceReviewError(
                "GATE15_SOURCE_METADATA_INVALID",
                "source artifact ID is not one of the exact HC-217 artifacts",
            )
        if expected_spec is not None and artifact_type != expected_spec["artifact_type"]:
            raise Gate15SourceReviewError(
                "GATE15_SOURCE_METADATA_INVALID",
                f"{artifact_id} has an unexpected source artifact type",
            )
        url = _validate_official_url(item.get("url"), f"{artifact_type}.url")
        if expected_spec is not None and url != expected_spec["url"]:
            raise Gate15SourceReviewError(
                "GATE15_SOURCE_URL_INVALID",
                f"{artifact_id}.url is not the exact HC-217 endpoint",
            )
        status = _require_int(item.get("http_status"), f"{artifact_type}.http_status")
        if status != 200:
            raise Gate15SourceReviewError(
                "GATE15_SOURCE_HTTP_INVALID",
                f"{artifact_type} source did not return HTTP 200",
            )
        content_type = _require_text(item.get("content_type"), f"{artifact_type}.content_type")
        if content_type.lower() != _ARTIFACT_CONTENT_TYPES[artifact_type]:
            raise Gate15SourceReviewError(
                "GATE15_SOURCE_METADATA_INVALID",
                f"{artifact_type}.content_type disagrees with the expected artifact type",
            )
        _require_int(
            item.get("content_length_bytes"),
            f"{artifact_type}.content_length_bytes",
            positive=True,
        )
        sha256 = item.get("sha256")
        sha_status = _require_text(item.get("sha256_status"), f"{artifact_type}.sha256_status")
        if sha256 is None:
            hashes_complete = False
            if sha_status != "PENDING_AUTHORIZED_DOWNLOAD":
                raise Gate15SourceReviewError(
                    "GATE15_SOURCE_HASH_INVALID",
                    f"{artifact_type} without a SHA-256 must remain pending authorized download",
                )
        elif not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
            raise Gate15SourceReviewError(
                "GATE15_SOURCE_HASH_INVALID",
                f"{artifact_type}.sha256 must be a 64-character hexadecimal digest",
            )
        elif sha_status != "VERIFIED_LOCAL_SHA256":
            raise Gate15SourceReviewError(
                "GATE15_SOURCE_HASH_INVALID",
                f"{artifact_type} hash status must be VERIFIED_LOCAL_SHA256",
            )
    if seen_types != _ARTIFACT_TYPES or (
        expected_source_artifacts is not None
        and {item.get("artifact_id") for item in raw_artifacts} != set(expected_source_artifacts)
    ):
        raise Gate15SourceReviewError(
            "GATE15_SOURCE_METADATA_INVALID",
            "source artifacts are incomplete",
        )
    return hashes_complete


def validate_gate15_source_preflight(
    review: Mapping[str, Any],
    *,
    access_config_path: str | Path = DEFAULT_ACCESS_PATH,
    permission_path: str | Path = DEFAULT_PERMISSION_PATH,
    expected_puf_id: str = "HC-217",
) -> Gate15SourcePreflightReport:
    """Validate one Gate 15 metadata record without opening a data archive."""

    root = _require_mapping(review, "review")
    if root.get("gate") != GATE15:
        raise Gate15SourceReviewError("GATE15_RECORD_INVALID", "review.gate must be '15'")
    puf_id = _require_text(root.get("puf_id"), "puf_id")
    if puf_id != expected_puf_id:
        raise Gate15SourceReviewError(
            "GATE15_PANEL_ORDER_INVALID",
            f"this preflight expects {expected_puf_id}, not {puf_id}",
        )
    if puf_id not in PANEL_ORDER:
        raise Gate15SourceReviewError(
            "GATE15_PANEL_NOT_CANDIDATE",
            "puf_id is not in the fixed Gate 15 candidate sequence",
        )

    panel = _require_mapping(root.get("panel"), "panel")
    panel_number = _require_int(panel.get("panel_number"), "panel.panel_number")
    baseline_year = _require_int(panel.get("baseline_year"), "panel.baseline_year")
    follow_up_year = _require_int(panel.get("follow_up_year"), "panel.follow_up_year")
    candidate_order = _require_int(panel.get("candidate_order"), "panel.candidate_order", positive=True)
    expected_order = PANEL_ORDER.index(puf_id) + 1
    if candidate_order != expected_order or panel_number != 22 + candidate_order:
        raise Gate15SourceReviewError(
            "GATE15_PANEL_ORDER_INVALID",
            "panel order or panel number disagrees with the frozen candidate sequence",
        )
    if follow_up_year != baseline_year + 1:
        raise Gate15SourceReviewError(
            "GATE15_CALENDAR_STRUCTURE_INVALID",
            "Gate 15 candidate must span two consecutive calendar years",
        )
    if puf_id == HC217_PUF_ID and (baseline_year, follow_up_year) != HC217_SURVEY_YEARS:
        raise Gate15SourceReviewError(
            "GATE15_CALENDAR_STRUCTURE_INVALID",
            "HC-217 must remain bound to the exact 2018-2019 panel period",
        )

    status = _require_text(root.get("review_status"), "review_status")
    if status not in {PREFLIGHT_STATUS, "SCHEMA_REVIEW_COMPLETE"}:
        raise Gate15SourceReviewError("GATE15_RECORD_INVALID", "unsupported review_status")

    source = _require_mapping(root.get("source"), "source")
    source_hashes_complete = _validate_artifacts(source, puf_id=puf_id)
    documented = _require_mapping(root.get("documented_structure"), "documented_structure")
    if documented.get("status") != "DOCUMENTATION_METADATA_ONLY":
        raise Gate15SourceReviewError(
            "GATE15_DOCUMENTATION_STATUS_INVALID",
            "documented_structure must remain metadata-only before schema access",
        )
    if documented.get("rounds") != 5:
        raise Gate15SourceReviewError(
            "GATE15_CALENDAR_STRUCTURE_INVALID",
            "the documented longitudinal structure must retain five rounds",
        )
    if documented.get("two_year_period") != [baseline_year, follow_up_year]:
        raise Gate15SourceReviewError(
            "GATE15_CALENDAR_STRUCTURE_INVALID",
            "documented two-year period disagrees with the frozen panel years",
        )
    documented_variables = documented.get("documented_variables")
    if (
        not isinstance(documented_variables, list)
        or any(not isinstance(field, str) for field in documented_variables)
        or any(field in _REQUIRED_SEMANTIC_CHECKS for field in documented_variables)
    ):
        # The source record's documented_variables are field names, not the
        # semantic checklist.  This branch protects against accidentally
        # storing checklist prose in the schema-field list.
        raise Gate15SourceReviewError(
            "GATE15_DOCUMENTATION_STATUS_INVALID",
            "documented_variables must contain only source field names",
        )

    semantic = _require_mapping(root.get("semantic_review"), "semantic_review")
    semantic_status = _require_text(semantic.get("status"), "semantic_review.status")
    if semantic_status not in {SEMANTIC_PENDING_STATUS, SEMANTIC_COMPLETE_STATUS}:
        raise Gate15SourceReviewError(
            "GATE15_SEMANTIC_REVIEW_INVALID",
            "unsupported semantic review status",
        )
    checks = semantic.get("required_checks")
    if not isinstance(checks, list) or tuple(checks) != _REQUIRED_SEMANTIC_CHECKS:
        raise Gate15SourceReviewError(
            "GATE15_SEMANTIC_REVIEW_INVALID",
            "the Gate 15 semantic checklist is missing or reordered",
        )

    authorization = _require_mapping(root.get("authorization"), "authorization")
    source_metadata_allowed = _require_bool(
        authorization.get("source_metadata_read_allowed"),
        "authorization.source_metadata_read_allowed",
    )
    schema_allowed_in_record = _require_bool(
        authorization.get("schema_read_allowed"),
        "authorization.schema_read_allowed",
    )
    microdata_rows_read = _require_bool(
        authorization.get("microdata_rows_read"),
        "authorization.microdata_rows_read",
    )
    outcome_accessed = _require_bool(
        authorization.get("outcome_values_read"),
        "authorization.outcome_values_read",
    )
    panel_27_accessed = _require_bool(
        authorization.get("panel_27_accessed"),
        "authorization.panel_27_accessed",
    )
    if not source_metadata_allowed or microdata_rows_read or outcome_accessed or panel_27_accessed:
        raise Gate15SourceReviewError(
            "GATE15_BOUNDARY_VIOLATION",
            "metadata preflight must allow source metadata only and keep data/Panel 27 closed",
        )

    schema = _require_mapping(root.get("schema"), "schema")
    schema_status = _require_text(schema.get("status"), "schema.status")
    if schema_allowed_in_record and status != "SCHEMA_REVIEW_COMPLETE":
        raise Gate15SourceReviewError(
            "GATE15_RECORD_INVALID",
            "a completed schema review must use SCHEMA_REVIEW_COMPLETE status",
        )
    if not schema_allowed_in_record and status != PREFLIGHT_STATUS:
        raise Gate15SourceReviewError(
            "GATE15_SCHEMA_CLAIM_WITHOUT_AUTHORIZATION",
            "metadata-only status cannot be promoted before schema authorization",
        )
    if semantic_status == SEMANTIC_COMPLETE_STATUS and not schema_allowed_in_record:
        raise Gate15SourceReviewError(
            "GATE15_SEMANTIC_REVIEW_INVALID",
            "semantic review cannot be complete before authorized schema/codebook access",
        )
    if schema_allowed_in_record:
        require_schema_stage_authorized(
            puf_id,
            access_config_path=access_config_path,
            permission_path=permission_path,
        )
        if schema_status != SCHEMA_VERIFIED_STATUS:
            raise Gate15SourceReviewError(
                "GATE15_SCHEMA_INCOMPLETE",
                "authorized schema stage must provide a verified schema",
            )
        if not source_hashes_complete:
            raise Gate15SourceReviewError(
                "GATE15_SOURCE_HASH_INCOMPLETE",
                "schema stage requires local SHA-256 for every source artifact",
            )
        for field in ("column_names", "column_types", "schema_sha256"):
            if schema.get(field) in (None, [], {}):
                raise Gate15SourceReviewError(
                    "GATE15_SCHEMA_INCOMPLETE",
                    f"schema.{field} is required after schema access",
                )
        if not isinstance(schema.get("schema_sha256"), str) or not _SHA256.fullmatch(
            schema["schema_sha256"]
        ):
            raise Gate15SourceReviewError(
                "GATE15_SCHEMA_INCOMPLETE",
                "schema.schema_sha256 must be a 64-character hexadecimal digest",
            )
    else:
        if schema_status != SCHEMA_PENDING_STATUS:
            raise Gate15SourceReviewError(
                "GATE15_SCHEMA_CLAIM_WITHOUT_AUTHORIZATION",
                "an unauthorized record may not claim a read or verified schema",
            )
        if any(schema.get(field) not in (None, [], {}) for field in ("column_names", "column_types", "schema_sha256")):
            raise Gate15SourceReviewError(
                "GATE15_SCHEMA_CLAIM_WITHOUT_AUTHORIZATION",
                "schema evidence must be empty while schema access is unauthorized",
            )

    provenance = _require_mapping(root.get("provenance"), "provenance")
    _require_text(provenance.get("retrieved_at_utc"), "provenance.retrieved_at_utc")
    _require_text(provenance.get("source"), "provenance.source")

    return Gate15SourcePreflightReport(
        gate=GATE15,
        puf_id=puf_id,
        panel_number=panel_number,
        baseline_year=baseline_year,
        follow_up_year=follow_up_year,
        status=status,
        schema_stage_authorized=schema_allowed_in_record,
        schema_status=schema_status,
        source_hashes_complete=source_hashes_complete,
        structural_checks_complete=(
            schema_allowed_in_record
            and schema_status == SCHEMA_VERIFIED_STATUS
            and source_hashes_complete
            and semantic_status == SEMANTIC_COMPLETE_STATUS
        ),
        semantic_review_complete=semantic_status == SEMANTIC_COMPLETE_STATUS,
        outcome_accessed=outcome_accessed,
        panel_27_accessed=panel_27_accessed,
    )


def load_gate15_source_preflight(
    path: str | Path = DEFAULT_REVIEW_PATH,
    *,
    access_config_path: str | Path = DEFAULT_ACCESS_PATH,
    permission_path: str | Path = DEFAULT_PERMISSION_PATH,
) -> Gate15SourcePreflightReport:
    """Load and validate the checked-in Gate 15 preflight record."""

    review_path = Path(path)
    try:
        with review_path.open(encoding="utf-8") as handle:
            review = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise Gate15SourceReviewError(
            "GATE15_RECORD_UNREADABLE",
            "Gate 15 source preflight is missing or invalid JSON",
        ) from exc
    return validate_gate15_source_preflight(
        review,
        access_config_path=access_config_path,
        permission_path=permission_path,
    )


__all__ = [
    "DEFAULT_ACCESS_PATH",
    "DEFAULT_PERMISSION_PATH",
    "DEFAULT_REVIEW_PATH",
    "GATE15",
    "HC217_DETAILS_URL",
    "HC217_PANEL_NUMBER",
    "HC217_PUF_ID",
    "HC217_SOURCE_ARTIFACTS",
    "HC217_SURVEY_YEARS",
    "Gate15SourcePreflightReport",
    "Gate15ArtifactPermission",
    "Gate15SourceReviewError",
    "OFFICIAL_HOST",
    "PANEL_ORDER",
    "PERMISSION_AUTHORIZED_STATUS",
    "PERMISSION_PENDING_STATUS",
    "PREFLIGHT_STATUS",
    "SCHEMA_PENDING_STATUS",
    "SCHEMA_CODEBOOK_ONLY_STAGE",
    "SCHEMA_VERIFIED_STATUS",
    "SEMANTIC_COMPLETE_STATUS",
    "SEMANTIC_PENDING_STATUS",
    "load_gate15_artifact_permission",
    "load_gate15_source_preflight",
    "require_schema_stage_authorized",
    "validate_gate15_artifact_permission",
    "validate_gate15_source_preflight",
]
