"""Record-level NHIS D0 harmonization with raw-code preservation."""

from __future__ import annotations

import os
import pathlib
import uuid
from typing import Any, Mapping, Sequence

import pandas as pd

from .download import NHISDownloadError, compute_sha256
from .schema import (
    REQUIRED_COLUMNS,
    canonical_code_series,
    validate_harmonized_columns,
    validate_required_columns,
)


class NHISHarmonizationError(ValueError):
    """Raised when a source frame cannot be converted under the D0 contract."""


HARMONIZED_COLUMNS: tuple[str, ...] = (
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
    "sex_a",
    "HISPALLP_A",
    "hispallp_a",
    "DISAB3_A",
    "disab3_a",
)


def _codes_from_registry(registry: Mapping[str, Any], column: str, field: str) -> set[int]:
    values = registry["variables"][column].get(field, [])
    if not isinstance(values, list):
        raise NHISHarmonizationError(f"Registry field {field!r} for {column} must be a list.")
    try:
        return {int(value) for value in values}
    except (TypeError, ValueError) as exc:
        raise NHISHarmonizationError(f"Registry field {field!r} for {column} is not integral.") from exc


def recode_binary(
    series: pd.Series,
    *,
    valid_codes: Sequence[int] = (1, 2),
    missing_codes: Sequence[int] = (7, 8, 9),
) -> pd.Series:
    """Map an explicit 1/2 binary code to 1/0 and all non-substantive codes to NA."""

    codes = canonical_code_series(series)
    valid = set(int(code) for code in valid_codes)
    missing = set(int(code) for code in missing_codes)
    unsupported = (~codes.isin(sorted(valid | missing))).fillna(False)
    if int(unsupported.sum()) > 0:
        raise NHISHarmonizationError(
            f"Binary field {series.name!r} contains codes outside its declared domain: "
            f"{int(unsupported.sum())} records."
        )
    result = pd.Series(pd.NA, index=series.index, dtype="Int64", name=series.name)
    if 1 in valid:
        result.loc[codes.eq(1)] = 1
    if 2 in valid:
        result.loc[codes.eq(2)] = 0
    return result


def recode_categorical(
    series: pd.Series,
    *,
    valid_codes: Sequence[int],
    missing_codes: Sequence[int],
) -> pd.Series:
    """Retain official substantive category codes and map non-substantive codes to NA."""

    codes = canonical_code_series(series)
    valid = {int(code) for code in valid_codes}
    missing = {int(code) for code in missing_codes}
    unsupported = (~codes.isin(sorted(valid | missing))).fillna(False)
    if int(unsupported.sum()) > 0:
        raise NHISHarmonizationError(
            f"Categorical field {series.name!r} contains codes outside its declared domain: "
            f"{int(unsupported.sum())} records."
        )
    result = codes.copy()
    result.loc[~codes.isin(sorted(valid))] = pd.NA
    result.name = series.name
    return result


def _explicit_weight(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    non_numeric = series.notna() & numeric.isna()
    if int(non_numeric.sum()) > 0:
        raise NHISHarmonizationError(
            f"WTFA_A contains {int(non_numeric.sum())} non-numeric records; no coercion is permitted."
        )
    return numeric.astype("Float64")


def harmonize_year(
    frame: pd.DataFrame,
    *,
    year: int,
    study_role: str,
    registry: Mapping[str, Any],
) -> pd.DataFrame:
    """Create one year of the D0 table while leaving source fields untouched."""

    validate_required_columns(frame.columns)
    if not isinstance(study_role, str) or not study_role:
        raise NHISHarmonizationError("study_role must be an explicit non-empty string.")
    source = frame.loc[:, list(REQUIRED_COLUMNS)].copy(deep=True)
    output = source.copy(deep=True)

    # The explicit study year is metadata from the source-file contract, not
    # an inferred dtype or a random split label.
    output["survey_year"] = int(year)
    output["study_role"] = study_role

    # Preserve both official columns and exact raw aliases.  Derived columns
    # are added separately, so no raw source code is silently replaced.
    output["MEDDL12M_A_raw"] = source["MEDDL12M_A"].copy(deep=True)
    output["MEDNG12M_A_raw"] = source["MEDNG12M_A"].copy(deep=True)

    primary_spec = registry["variables"]["MEDDL12M_A"]
    sensitivity_spec = registry["variables"]["MEDNG12M_A"]
    output["meddl12m"] = recode_binary(
        source["MEDDL12M_A"],
        valid_codes=primary_spec["valid_codes"],
        missing_codes=primary_spec["missing_codes"],
    )
    output["meddly12m"] = output["meddl12m"].copy()
    output["medng12m"] = recode_binary(
        source["MEDNG12M_A"],
        valid_codes=sensitivity_spec["valid_codes"],
        missing_codes=sensitivity_spec["missing_codes"],
    )

    for column, derived_name in (
        ("SEX_A", "sex_a"),
        ("HISPALLP_A", "hispallp_a"),
        ("DISAB3_A", "disab3_a"),
    ):
        spec = registry["variables"][column]
        output[derived_name] = recode_categorical(
            source[column],
            valid_codes=spec["valid_codes"],
            missing_codes=spec["missing_codes"],
        )

    weight = _explicit_weight(source["WTFA_A"])
    development_weight = pd.Series(pd.NA, index=source.index, dtype="Float64", name="WTFA_DEV")
    if int(year) in (2022, 2023):
        development_weight.loc[weight.notna()] = weight.loc[weight.notna()] / 2.0
    output["WTFA_DEV"] = development_weight

    ordered = output.loc[:, list(HARMONIZED_COLUMNS)].copy()
    validate_harmonized_columns(ordered)
    return ordered.reset_index(drop=True)


def combine_harmonized_years(frames: Mapping[int, pd.DataFrame], years: Sequence[int]) -> pd.DataFrame:
    """Concatenate selected years in explicit temporal order."""

    missing = [int(year) for year in years if int(year) not in frames]
    if missing:
        raise NHISHarmonizationError(f"Missing harmonized year frame(s): {missing}")
    combined = pd.concat([frames[int(year)] for year in years], ignore_index=True)
    validate_harmonized_columns(combined)
    return combined


def validate_harmonized_roles(frame: pd.DataFrame, years: Sequence[int]) -> None:
    """Validate the fixed development/validation/frozen-test role mapping."""

    expected = {2022: "development_train", 2023: "development_validation", 2024: "frozen_test"}
    for year in years:
        year_int = int(year)
        if year_int not in expected:
            raise NHISHarmonizationError(f"No fixed D0 study role is configured for {year_int}.")
        subset = frame.loc[frame["survey_year"].eq(year_int)]
        if subset.empty:
            raise NHISHarmonizationError(f"No harmonized records found for configured year {year_int}.")
        roles = set(subset["study_role"].dropna().astype(str))
        if roles != {expected[year_int]}:
            raise NHISHarmonizationError(
                f"Study-role mismatch for {year_int}: expected {expected[year_int]!r}, observed {sorted(roles)}."
            )
        if year_int == 2024 and subset["WTFA_DEV"].notna().any():
            raise NHISHarmonizationError("2024 WTFA_DEV must be missing for every frozen_test record.")
        if year_int in (2022, 2023):
            weight = pd.to_numeric(subset["WTFA_A"], errors="coerce")
            development = pd.to_numeric(subset["WTFA_DEV"], errors="coerce")
            comparable = weight.notna()
            if not development.loc[comparable].sub(weight.loc[comparable] / 2.0).abs().lt(1e-9).all():
                raise NHISHarmonizationError(f"WTFA_DEV does not equal WTFA_A / 2 for {year_int}.")


def write_harmonized_parquet(
    frame: pd.DataFrame,
    path: pathlib.Path | str,
    *,
    force: bool = False,
) -> str:
    """Write a valid Parquet table atomically, refusing implicit replacement."""

    validate_harmonized_columns(frame)
    path_obj = pathlib.Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401  # explicit runtime capability check
    except ImportError as exc:
        raise NHISHarmonizationError(
            "A Parquet engine is required to create the D0 output; install pyarrow in the active environment."
        ) from exc

    part = path_obj.with_name(f".{path_obj.name}.{uuid.uuid4().hex}.part")
    try:
        frame.to_parquet(part, engine="pyarrow", index=False)
        new_sha = compute_sha256(part)
        if path_obj.exists() and not force:
            old_sha = compute_sha256(path_obj)
            if old_sha == new_sha:
                part.unlink(missing_ok=True)
                return old_sha
            raise NHISHarmonizationError(
                f"Existing harmonized Parquet differs at {path_obj}; refusing to overwrite without --force."
            )
        os.replace(part, path_obj)
        return new_sha
    except NHISHarmonizationError:
        part.unlink(missing_ok=True)
        raise
    except (OSError, ValueError, TypeError) as exc:
        part.unlink(missing_ok=True)
        raise NHISHarmonizationError(f"Could not write harmonized Parquet {path_obj}: {exc}") from exc


def read_harmonized_parquet(path: pathlib.Path | str) -> pd.DataFrame:
    """Read and contract-check the D0 Parquet output."""

    path_obj = pathlib.Path(path)
    if not path_obj.is_file():
        raise NHISHarmonizationError(f"Harmonized Parquet not found: {path_obj}")
    try:
        frame = pd.read_parquet(path_obj, engine="pyarrow")
    except ImportError as exc:
        raise NHISHarmonizationError(
            "A Parquet engine is required to read the D0 output; install pyarrow in the active environment."
        ) from exc
    except Exception as exc:
        raise NHISHarmonizationError(f"Could not read harmonized Parquet {path_obj}: {exc}") from exc
    validate_harmonized_columns(frame)
    return frame
