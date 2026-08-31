"""Data ingestion, provenance tracking, and loading utilities for MEPS data."""

from __future__ import annotations

from meps_fairness.data.download import (
    CHUNK_SIZE,
    MAX_COMPRESSION_RATIO,
    MAX_UNCOMPRESSED_ARCHIVE_BYTES,
    Artifact,
    ArtifactManifest,
    DataAccessPolicy,
    DownloadError,
    IntegrityError,
    ProvenanceError,
    ProvenanceManifest,
    ProvenanceRecord,
    SecurityError,
    download_artifacts,
)

try:
    from meps_fairness.data.prepare import (
        BOUNDED_SCAN_CHUNK_SIZE,
        MAX_RSS_GB_LIMIT,
        REQUIRED_STRUCTURAL_COLUMNS,
        ArchiveMemberMetadata,
        CrossPanelComparison,
        ExtractionRecord,
        PanelExpectation,
        PreparationProvenance,
        SchemaExpectations,
        StructuralCheckResult,
        compare_cross_panel_schemas,
        compute_schema_hash,
        extract_meps_archives,
        extract_single_archive,
        generate_and_save_schema_snapshot,
        get_peak_rss_gb,
        guard_against_prohibited_inspections,
        scan_single_panel_schema,
    )
except ImportError:  # pragma: no cover
    pass

try:
    from meps_fairness.data.cohort import (
        ALL_BASELINE_PREDICTOR_COLUMNS,
        ALL_CATEGORICAL_PREDICTORS,
        ALL_CONTINUOUS_PREDICTORS,
        AUDIT_COLUMNS,
        BASELINE_INSURANCE_MONTHS,
        DESIGN_COLUMNS,
        FOLLOWUP_INSURANCE_MONTHS,
        CohortData,
        derive_age_band,
        derive_composite_disability,
        extract_meps_cohort,
    )
except ImportError:  # pragma: no cover
    pass

__all__ = [
    "CHUNK_SIZE",
    "MAX_COMPRESSION_RATIO",
    "MAX_UNCOMPRESSED_ARCHIVE_BYTES",
    "Artifact",
    "ArtifactManifest",
    "DataAccessPolicy",
    "DownloadError",
    "IntegrityError",
    "ProvenanceError",
    "ProvenanceManifest",
    "ProvenanceRecord",
    "SecurityError",
    "download_artifacts",
    "BOUNDED_SCAN_CHUNK_SIZE",
    "MAX_RSS_GB_LIMIT",
    "REQUIRED_STRUCTURAL_COLUMNS",
    "ArchiveMemberMetadata",
    "CrossPanelComparison",
    "ExtractionRecord",
    "PanelExpectation",
    "PreparationProvenance",
    "SchemaExpectations",
    "StructuralCheckResult",
    "compare_cross_panel_schemas",
    "compute_schema_hash",
    "extract_meps_archives",
    "extract_single_archive",
    "generate_and_save_schema_snapshot",
    "get_peak_rss_gb",
    "guard_against_prohibited_inspections",
    "scan_single_panel_schema",
    "ALL_BASELINE_PREDICTOR_COLUMNS",
    "ALL_CATEGORICAL_PREDICTORS",
    "ALL_CONTINUOUS_PREDICTORS",
    "AUDIT_COLUMNS",
    "BASELINE_INSURANCE_MONTHS",
    "DESIGN_COLUMNS",
    "FOLLOWUP_INSURANCE_MONTHS",
    "CohortData",
    "derive_age_band",
    "derive_composite_disability",
    "extract_meps_cohort",
]
