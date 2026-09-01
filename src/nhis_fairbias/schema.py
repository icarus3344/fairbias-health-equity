"""Explicit NHIS D0 schema and source-frequency validation.

The source CSV is read using the declared registry in
``configs/nhis/variables.json``.  In particular, category fields are not
treated as continuous merely because a CSV parser can represent their codes
as numbers.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any, Iterable, Mapping

import pandas as pd


class NHISSchemaError(ValueError):
    """Raised when a source file violates the explicit D0 contract."""


REQUIRED_COLUMNS: tuple[str, ...] = (
    "SRVY_YR",
    "WTFA_A",
    "PSTRAT",
    "PPSU",
    "MEDDL12M_A",
    "MEDNG12M_A",
    "SEX_A",
    "HISPALLP_A",
    "DISAB3_A",
)

SOURCE_COLUMNS: tuple[str, ...] = REQUIRED_COLUMNS
OUTCOME_COLUMNS: tuple[str, ...] = ("MEDDL12M_A", "MEDNG12M_A")
PROTECTED_COLUMNS: tuple[str, ...] = ("SEX_A", "HISPALLP_A", "DISAB3_A")
DESIGN_COLUMNS: tuple[str, ...] = ("WTFA_A", "PSTRAT", "PPSU")

_DEFAULT_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_STUDY_CONFIG = _DEFAULT_REPO_ROOT / "configs" / "nhis" / "study.json"
DEFAULT_VARIABLE_CONFIG = _DEFAULT_REPO_ROOT / "configs" / "nhis" / "variables.json"


@dataclasses.dataclass(frozen=True)
class SourceValidation:
    """Compact, non-row-level validation result for one source year."""

    year: int
    raw_row_count: int
    primary_frequency: dict[int, int]
    schema_rows: tuple[dict[str, Any], ...]


def _load_json(path: pathlib.Path | str) -> dict[str, Any]:
    path_obj = pathlib.Path(path)
    if not path_obj.is_file():
        raise NHISSchemaError(f"Configuration file not found: {path_obj}")
    try:
        with path_obj.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise NHISSchemaError(f"Could not read configuration {path_obj}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise NHISSchemaError(f"Configuration must be a JSON object: {path_obj}")
    return loaded


def load_study_config(path: pathlib.Path | str = DEFAULT_STUDY_CONFIG) -> dict[str, Any]:
    """Load and minimally validate the year/source study configuration."""

    config = _load_json(path)
    years = config.get("years")
    if not isinstance(years, dict) or not years:
        raise NHISSchemaError("NHIS study configuration must contain a non-empty years object.")
    for year_key, year_spec in years.items():
        if not isinstance(year_spec, dict):
            raise NHISSchemaError(f"Year configuration must be an object: {year_key!r}")
        for field in (
            "source_url",
            "local_source_file",
            "local_csv_file",
            "expected_csv_member",
            "expected_raw_rows",
            "expected_meddl12m_counts",
            "study_role",
        ):
            if field not in year_spec:
                raise NHISSchemaError(f"Year {year_key} is missing required configuration field {field!r}.")
    if not isinstance(config.get("outputs"), dict):
        raise NHISSchemaError("NHIS study configuration must contain an outputs object.")
    return config


def load_variable_registry(path: pathlib.Path | str = DEFAULT_VARIABLE_CONFIG) -> dict[str, Any]:
    """Load and validate the explicit variable registry."""

    registry = _load_json(path)
    required = registry.get("required_variables")
    variables = registry.get("variables")
    if not isinstance(required, list) or tuple(required) != REQUIRED_COLUMNS:
        raise NHISSchemaError(
            "Variable registry required_variables must list the exact D0 columns "
            f"in this order: {REQUIRED_COLUMNS}."
        )
    if not isinstance(variables, dict):
        raise NHISSchemaError("Variable registry must contain a variables object.")
    missing_specs = [name for name in REQUIRED_COLUMNS if name not in variables]
    if missing_specs:
        raise NHISSchemaError(f"Variable registry is missing specifications: {missing_specs}")
    for name in REQUIRED_COLUMNS:
        spec = variables[name]
        if not isinstance(spec, dict):
            raise NHISSchemaError(f"Variable specification must be an object: {name}")
        if not spec.get("required", False):
            raise NHISSchemaError(f"Required variable is not marked required: {name}")
        if not isinstance(spec.get("declared_type"), str):
            raise NHISSchemaError(f"Variable is missing a declared_type: {name}")
    return registry


def configured_years(study_config: Mapping[str, Any]) -> tuple[int, ...]:
    """Return configured years in numeric sorted order."""

    years_obj = study_config.get("years")
    if not isinstance(years_obj, Mapping):
        raise NHISSchemaError("Study configuration years must be a mapping.")
    try:
        years = tuple(sorted(int(year) for year in years_obj))
    except (TypeError, ValueError) as exc:
        raise NHISSchemaError("Study configuration contains a non-integer year key.") from exc
    if not years:
        raise NHISSchemaError("Study configuration contains no years.")
    return years


def year_spec(study_config: Mapping[str, Any], year: int) -> Mapping[str, Any]:
    """Return one year contract, rejecting years outside the configured set."""

    try:
        spec = study_config["years"][str(int(year))]
    except (KeyError, TypeError, ValueError) as exc:
        raise NHISSchemaError(f"Year is not configured for this NHIS study: {year}") from exc
    if not isinstance(spec, Mapping):
        raise NHISSchemaError(f"Year configuration is not an object: {year}")
    return spec


def declared_pandas_dtypes(registry: Mapping[str, Any]) -> dict[str, str]:
    """Translate declared registry types into explicit pandas dtypes."""

    type_map = {"integer": "Int64", "numeric": "float64", "string": "string"}
    variables = registry["variables"]
    dtypes: dict[str, str] = {}
    for name in REQUIRED_COLUMNS:
        declared = variables[name]["declared_type"]
        try:
            dtypes[name] = type_map[declared]
        except KeyError as exc:
            raise NHISSchemaError(
                f"Unsupported declared_type {declared!r} for {name}; no dtype inference is permitted."
            ) from exc
    return dtypes


def validate_required_columns(
    columns: Iterable[object],
    *,
    required: tuple[str, ...] = REQUIRED_COLUMNS,
) -> dict[str, bool]:
    """Validate presence and uniqueness of the required source columns."""

    column_names = [str(column) for column in columns]
    duplicated = sorted({name for name in column_names if column_names.count(name) > 1})
    if duplicated:
        raise NHISSchemaError(f"Source CSV contains duplicate column names: {duplicated}")
    presence = {name: name in column_names for name in required}
    missing = [name for name, present in presence.items() if not present]
    if missing:
        raise NHISSchemaError(f"Source CSV is missing required NHIS columns: {missing}")
    return presence


def read_nhis_csv(
    csv_path: pathlib.Path | str,
    *,
    registry: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Read only the D0 source columns with their explicitly declared types."""

    path_obj = pathlib.Path(csv_path)
    if not path_obj.is_file():
        raise NHISSchemaError(f"NHIS CSV file not found: {path_obj}")
    registry_obj = registry if registry is not None else load_variable_registry()
    try:
        header = pd.read_csv(path_obj, nrows=0, encoding="utf-8-sig")
    except Exception as exc:  # pandas exposes parser/encoding errors with different classes
        raise NHISSchemaError(f"Could not read NHIS CSV header {path_obj}: {exc}") from exc
    validate_required_columns(header.columns)
    try:
        frame = pd.read_csv(
            path_obj,
            usecols=list(REQUIRED_COLUMNS),
            dtype=declared_pandas_dtypes(registry_obj),
            encoding="utf-8-sig",
            low_memory=False,
            on_bad_lines="error",
        )
    except Exception as exc:  # keep a single evidence-bounded public error type
        raise NHISSchemaError(f"Could not parse NHIS CSV {path_obj}: {exc}") from exc
    validate_required_columns(frame.columns)
    return frame


def canonical_code_series(series: pd.Series) -> pd.Series:
    """Parse a coded field to nullable integral codes without treating it as continuous."""

    numeric = pd.to_numeric(series, errors="coerce")
    integral = numeric.notna() & numeric.eq(numeric.round())
    canonical = pd.Series(pd.NA, index=series.index, dtype="Int64", name=series.name)
    if integral.any():
        canonical.loc[integral] = numeric.loc[integral].astype("int64")
    return canonical


def _declared_code_set(registry: Mapping[str, Any], column: str, key: str) -> set[int]:
    values = registry["variables"][column].get(key, [])
    if not isinstance(values, list):
        raise NHISSchemaError(f"Registry field {key!r} for {column} must be a list.")
    try:
        return {int(value) for value in values}
    except (TypeError, ValueError) as exc:
        raise NHISSchemaError(f"Registry field {key!r} for {column} contains a non-integer code.") from exc


def exact_primary_frequency(
    frame: pd.DataFrame,
    *,
    year: int,
    expected_counts: Mapping[str, Any],
) -> dict[int, int]:
    """Check the complete raw MEDDL12M_A frequency table for one year."""

    if "MEDDL12M_A" not in frame.columns:
        raise NHISSchemaError("MEDDL12M_A is required for the primary frequency check.")
    try:
        expected = {int(code): int(count) for code, count in expected_counts.items()}
    except (AttributeError, TypeError, ValueError) as exc:
        raise NHISSchemaError(f"Invalid expected MEDDL12M_A frequency contract for {year}.") from exc
    observed_codes = canonical_code_series(frame["MEDDL12M_A"])
    observed = {code: int(observed_codes.eq(code).sum()) for code in sorted(expected)}
    unlisted = (~observed_codes.isin(list(expected))).fillna(True)
    unlisted_count = int(unlisted.sum())
    if observed != expected or unlisted_count != 0 or sum(expected.values()) != len(frame):
        raise NHISSchemaError(
            f"MEDDL12M_A frequency mismatch for {year}: expected {expected} across "
            f"{sum(expected.values())} rows, observed {observed} with {unlisted_count} "
            f"unlisted/missing-code rows across {len(frame)} rows. No discrepancy repair is allowed."
        )
    return observed


def validate_declared_code_contracts(
    frame: pd.DataFrame,
    *,
    registry: Mapping[str, Any],
    year: int,
) -> None:
    """Validate the explicit code domains for outcomes and protected fields."""

    for column in OUTCOME_COLUMNS:
        allowed = _declared_code_set(registry, column, "valid_codes") | _declared_code_set(
            registry, column, "missing_codes"
        )
        codes = canonical_code_series(frame[column])
        invalid = (~codes.isin(sorted(allowed))).fillna(False)
        if int(invalid.sum()) > 0:
            raise NHISSchemaError(
                f"{column} contains codes outside its declared D0 domain in {year}: "
                f"{int(invalid.sum())} invalid/non-integral records."
            )
    for column in PROTECTED_COLUMNS:
        allowed = _declared_code_set(registry, column, "valid_codes") | _declared_code_set(
            registry, column, "missing_codes"
        )
        codes = canonical_code_series(frame[column])
        # A missing CSV value is allowed to remain missing; non-missing values
        # must be either an explicit substantive or explicit missing code.
        invalid = frame[column].notna() & (~codes.isin(sorted(allowed))).fillna(True)
        if int(invalid.sum()) > 0:
            raise NHISSchemaError(
                f"{column} contains codes outside its declared D0 domain in {year}: "
                f"{int(invalid.sum())} invalid/non-integral records."
            )


def build_schema_audit_rows(
    frame: pd.DataFrame,
    *,
    year: int,
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build a complete minimal column-presence/type audit without row output."""

    presence = validate_required_columns(frame.columns)
    rows: list[dict[str, Any]] = []
    variables = registry["variables"]
    for column in REQUIRED_COLUMNS:
        spec = variables[column]
        parsed = pd.to_numeric(frame[column], errors="coerce")
        nonmissing = frame[column].notna()
        non_numeric_count = int((nonmissing & parsed.isna()).sum())
        rows.append(
            {
                "year": int(year),
                "column": column,
                "required": True,
                "present": bool(presence[column]),
                "role": spec["role"],
                "declared_type": spec["declared_type"],
                "observed_dtype": str(frame[column].dtype),
                "non_missing_n": int(nonmissing.sum()),
                "non_numeric_n": non_numeric_count,
                "status": "PASS" if presence[column] and non_numeric_count == 0 else "FAIL",
            }
        )
    return rows


def validate_source_frame(
    frame: pd.DataFrame,
    *,
    year: int,
    study_config: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> SourceValidation:
    """Run all D0 source checks for one already-read year frame."""

    validate_required_columns(frame.columns)
    if frame.empty:
        raise NHISSchemaError(f"NHIS source frame is empty for {year}.")
    survey_year = canonical_code_series(frame["SRVY_YR"])
    if survey_year.isna().any() or not survey_year.eq(int(year)).all():
        raise NHISSchemaError(f"SRVY_YR does not identify every record as {year}.")
    validate_declared_code_contracts(frame, registry=registry, year=year)
    spec = year_spec(study_config, year)
    raw_rows = int(spec["expected_raw_rows"])
    if len(frame) != raw_rows:
        raise NHISSchemaError(
            f"Raw row-count mismatch for {year}: expected {raw_rows}, observed {len(frame)}."
        )
    primary_frequency = exact_primary_frequency(
        frame,
        year=year,
        expected_counts=spec["expected_meddl12m_counts"],
    )
    schema_rows = tuple(build_schema_audit_rows(frame, year=year, registry=registry))
    return SourceValidation(
        year=int(year),
        raw_row_count=len(frame),
        primary_frequency=primary_frequency,
        schema_rows=schema_rows,
    )


def validate_harmonized_columns(frame: pd.DataFrame) -> None:
    """Ensure the prepared table contains the D0 contract columns."""

    required = (
        "survey_year",
        "study_role",
        "SRVY_YR",
        "WTFA_A",
        "WTFA_DEV",
        "PSTRAT",
        "PPSU",
        "MEDDL12M_A",
        "MEDDL12M_A_raw",
        "meddl12m",
        "meddly12m",
        "MEDNG12M_A",
        "MEDNG12M_A_raw",
        "medng12m",
        "SEX_A",
        "HISPALLP_A",
        "DISAB3_A",
    )
    validate_required_columns(frame.columns, required=required)
