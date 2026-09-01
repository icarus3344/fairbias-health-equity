"""Feature registry, harmonization, and cross-year auditing for NHIS Gate D1.

This module defines the explicit candidate feature space for the NHIS
FairBias application study across survey years 2022, 2023, and 2024.
No feature typing is inferred from parser dtypes.
All forbidden outcome-proximal variables, protected attributes, and survey
design variables are strictly quarantined from ordinary feature matrices.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .download import compute_sha256, new_run_id, utc_timestamp, write_json_atomic
from .schema import (
    DEFAULT_STUDY_CONFIG,
    NHISSchemaError,
    canonical_code_series,
    load_study_config,
    year_spec,
)
from .survey import weighted_category_proportion


class NHISFeatureError(ValueError):
    """Raised when a feature contract or leakage guard is violated."""


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_CONFIG = _REPO_ROOT / "configs" / "nhis" / "features.json"

VALID_SEMANTIC_TYPES: tuple[str, ...] = (
    "categorical_nominal",
    "categorical_binary",
    "ordinal",
    "continuous",
    "count",
)

VALID_FEATURE_TIERS: tuple[str, ...] = (
    "primary_core",
    "expanded_utilization",
    "rejected_candidate",
    "forbidden_outcome_proximal",
    "protected_attribute",
    "outcome",
)


@dataclasses.dataclass(frozen=True)
class FeatureSpec:
    """Explicit metadata specification for one candidate variable."""

    official_name: str
    harmonized_name: str | None
    domain: str
    feature_tier: str
    role: str
    semantic_type: str
    official_description: str
    substantive_codes: tuple[int, ...]
    missing_codes: tuple[int, ...]
    year_available: tuple[int, ...]
    source_module: str
    source_section: str
    include_primary: bool
    include_expanded: bool
    exclusion_reason: str | None
    protected_proxy_note: str | None
    documentation_source: str
    code_labels: dict[str, str] = dataclasses.field(default_factory=dict)


def load_feature_registry(path: pathlib.Path | str = DEFAULT_FEATURE_CONFIG) -> dict[str, Any]:
    """Load and validate the explicit machine-readable feature registry."""

    path_obj = pathlib.Path(path)
    if not path_obj.is_file():
        raise NHISFeatureError(f"Feature configuration not found: {path_obj}")
    try:
        with path_obj.open("r", encoding="utf-8") as handle:
            registry = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise NHISFeatureError(f"Could not read feature configuration {path_obj}: {exc}") from exc
    if not isinstance(registry, dict):
        raise NHISFeatureError(f"Feature configuration must be a JSON object: {path_obj}")

    for key in (
        "schema_version",
        "feature_lists",
        "primary_core",
        "expanded_utilization",
        "forbidden_outcome_proximal",
        "protected_attributes",
        "outcomes",
    ):
        if key not in registry:
            raise NHISFeatureError(f"Feature registry is missing required key: {key!r}")

    # Validate semantic types and tiers in primary_core
    for name, spec in registry["primary_core"].items():
        _validate_spec_dict(name, spec)
        if spec["semantic_type"] not in VALID_SEMANTIC_TYPES:
            raise NHISFeatureError(
                f"Feature {name} has invalid semantic_type {spec['semantic_type']!r}. "
                f"Allowed: {VALID_SEMANTIC_TYPES}"
            )
        if not spec.get("include_primary", False):
            raise NHISFeatureError(f"PRIMARY_CORE feature {name} must have include_primary=True.")

    # Validate semantic types and tiers in expanded_utilization
    for name, spec in registry["expanded_utilization"].items():
        _validate_spec_dict(name, spec)
        if spec["semantic_type"] not in VALID_SEMANTIC_TYPES:
            raise NHISFeatureError(
                f"Feature {name} has invalid semantic_type {spec['semantic_type']!r}."
            )
        if not spec.get("include_expanded", False):
            raise NHISFeatureError(f"EXPANDED feature {name} must have include_expanded=True.")
        if spec.get("include_primary", False):
            raise NHISFeatureError(f"EXPANDED feature {name} must not have include_primary=True.")

    return registry


def _validate_spec_dict(name: str, spec: dict[str, Any]) -> None:
    required_keys = (
        "official_name",
        "harmonized_name",
        "domain",
        "feature_tier",
        "role",
        "semantic_type",
        "official_description",
        "substantive_codes",
        "missing_codes",
        "year_available",
        "source_module",
        "source_section",
        "include_primary",
        "include_expanded",
        "exclusion_reason",
        "protected_proxy_note",
        "documentation_source",
    )
    for rk in required_keys:
        if rk not in spec:
            raise NHISFeatureError(f"Variable specification for {name} missing required key: {rk!r}")


def get_feature_lists(registry: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Return immutable feature lists from the registry."""

    fl = registry.get("feature_lists", {})
    return {k: tuple(v) for k, v in fl.items()}


def recode_predictor_series(
    series: pd.Series,
    *,
    substantive_codes: Sequence[int],
    missing_codes: Sequence[int],
    semantic_type: str,
) -> pd.Series:
    """Recode a predictor series by mapping missing codes to NA and preserving substantive codes."""

    codes = canonical_code_series(series)
    substantive = {int(c) for c in substantive_codes}
    missing = {int(c) for c in missing_codes}
    allowed = substantive | missing

    # Identify any non-null codes outside declared domain
    invalid_mask = series.notna() & (~codes.isin(sorted(allowed))).fillna(True)
    if int(invalid_mask.sum()) > 0:
        raise NHISFeatureError(
            f"Field {series.name!r} contains {int(invalid_mask.sum())} unexpected codes "
            f"outside declared substantive + missing domain."
        )

    # Missing codes -> NA
    result = codes.copy()
    result.loc[~codes.isin(sorted(substantive))] = pd.NA
    result.name = series.name
    return result


def harmonize_features_year(
    raw_frame: pd.DataFrame,
    *,
    year: int,
    study_role: str,
    feature_registry: Mapping[str, Any],
) -> pd.DataFrame:
    """Harmonize all features for a single survey year, preserving raw columns."""

    year_int = int(year)
    out = pd.DataFrame(index=raw_frame.index)

    # 1. Identification / Design
    out["survey_year"] = year_int
    out["study_role"] = study_role
    out["SRVY_YR"] = canonical_code_series(raw_frame["SRVY_YR"])
    numeric_wtfa = pd.to_numeric(raw_frame["WTFA_A"], errors="coerce")
    if (raw_frame["WTFA_A"].notna() & numeric_wtfa.isna()).any():
        raise NHISFeatureError(f"WTFA_A contains non-numeric values in {year_int}.")
    out["WTFA_A"] = numeric_wtfa.astype("Float64")
    out["PSTRAT"] = canonical_code_series(raw_frame["PSTRAT"])
    out["PPSU"] = canonical_code_series(raw_frame["PPSU"])

    # Development weight (halved for 2-panel development 2022+2023, NA for frozen 2024)
    dev_weight = pd.Series(pd.NA, index=raw_frame.index, dtype="Float64", name="WTFA_DEV")
    if year_int in (2022, 2023):
        dev_weight = (out["WTFA_A"] / 2.0).astype("Float64")
    out["WTFA_DEV"] = dev_weight

    # 2. Outcomes
    out["MEDDL12M_A_raw"] = canonical_code_series(raw_frame["MEDDL12M_A"])
    out["MEDDL12M_A"] = out["MEDDL12M_A_raw"].copy()
    # Recode 1 -> 1, 2 -> 0, 7/8/9 -> NA
    meddl_recode = pd.Series(pd.NA, index=raw_frame.index, dtype="Int64", name="meddl12m")
    meddl_recode.loc[out["MEDDL12M_A_raw"].eq(1)] = 1
    meddl_recode.loc[out["MEDDL12M_A_raw"].eq(2)] = 0
    out["meddl12m"] = meddl_recode
    out["meddly12m"] = meddl_recode.copy()

    out["MEDNG12M_A_raw"] = canonical_code_series(raw_frame["MEDNG12M_A"])
    out["MEDNG12M_A"] = out["MEDNG12M_A_raw"].copy()
    medng_recode = pd.Series(pd.NA, index=raw_frame.index, dtype="Int64", name="medng12m")
    medng_recode.loc[out["MEDNG12M_A_raw"].eq(1)] = 1
    medng_recode.loc[out["MEDNG12M_A_raw"].eq(2)] = 0
    out["medng12m"] = medng_recode

    # 3. Protected Attributes
    for prot_col, prot_spec in feature_registry["protected_attributes"].items():
        out[prot_col] = canonical_code_series(raw_frame[prot_col])
        harm_name = prot_spec["harmonized_name"]
        recoded = canonical_code_series(raw_frame[prot_col])
        valid_set = set(prot_spec["substantive_codes"])
        recoded.loc[~recoded.isin(valid_set)] = pd.NA
        out[harm_name] = recoded

    # 4. Primary Core Features (21 variables)
    for raw_name, spec in feature_registry["primary_core"].items():
        harm_name = spec["harmonized_name"]
        if raw_name not in raw_frame.columns:
            raise NHISFeatureError(
                f"Required PRIMARY_CORE variable {raw_name!r} is missing from raw {year_int} frame."
            )
        # Preserve raw column
        out[raw_name] = canonical_code_series(raw_frame[raw_name])
        # Derive harmonized column
        out[harm_name] = recode_predictor_series(
            raw_frame[raw_name],
            substantive_codes=spec["substantive_codes"],
            missing_codes=spec["missing_codes"],
            semantic_type=spec["semantic_type"],
        )

    # 5. Expanded Utilization Features (3 variables)
    for raw_name, spec in feature_registry["expanded_utilization"].items():
        harm_name = spec["harmonized_name"]
        if raw_name not in raw_frame.columns:
            raise NHISFeatureError(
                f"EXPANDED_UTILIZATION variable {raw_name!r} is missing from raw {year_int} frame."
            )
        out[raw_name] = canonical_code_series(raw_frame[raw_name])
        out[harm_name] = recode_predictor_series(
            raw_frame[raw_name],
            substantive_codes=spec["substantive_codes"],
            missing_codes=spec["missing_codes"],
            semantic_type=spec["semantic_type"],
        )

    return out


def combine_feature_years(
    frames: Mapping[int, pd.DataFrame],
    years: Sequence[int],
) -> pd.DataFrame:
    """Concatenate yearly harmonized feature dataframes into a single panel."""

    ordered_frames = [frames[int(y)] for y in sorted(years)]
    combined = pd.concat(ordered_frames, ignore_index=True)
    return combined


def write_features_parquet(
    frame: pd.DataFrame,
    output_path: pathlib.Path | str,
    *,
    force: bool = False,
) -> str:
    """Write the harmonized feature frame atomically to Parquet."""

    path_obj = pathlib.Path(output_path).resolve()
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    part_path = path_obj.with_name(f".{path_obj.name}.part")

    try:
        frame.to_parquet(part_path, index=False, engine="pyarrow")
        new_sha = compute_sha256(part_path)
        if path_obj.exists() and not force:
            old_sha = compute_sha256(path_obj)
            if old_sha == new_sha:
                part_path.unlink(missing_ok=True)
                return old_sha
            raise NHISFeatureError(
                f"Existing Parquet differs at {path_obj}; refusing to overwrite without force."
            )
        os.replace(part_path, path_obj)
    finally:
        if part_path.exists():
            part_path.unlink(missing_ok=True)

    return compute_sha256(path_obj)


def build_feature_registry_audit_table(registry: Mapping[str, Any]) -> pd.DataFrame:
    """Generate the comprehensive feature registry audit table."""

    rows = []
    # Primary core
    for spec in registry["primary_core"].values():
        rows.append(_spec_to_audit_row(spec))
    # Expanded utilization
    for spec in registry["expanded_utilization"].values():
        rows.append(_spec_to_audit_row(spec))
    # Rejected candidates
    for spec in registry.get("rejected_candidates", {}).values():
        rows.append(_spec_to_audit_row(spec))
    # Protected attributes
    for spec in registry["protected_attributes"].values():
        rows.append(_spec_to_audit_row(spec))
    # Outcomes
    for spec in registry["outcomes"].values():
        rows.append(_spec_to_audit_row(spec))
    # Forbidden
    for name, fspec in registry["forbidden_outcome_proximal"].items():
        rows.append(
            {
                "official_name": fspec["official_name"],
                "harmonized_name": None,
                "domain": "Forbidden outcome proximal / cost barrier",
                "feature_tier": "forbidden_outcome_proximal",
                "role": "excluded",
                "semantic_type": "excluded",
                "official_description": fspec["official_description"],
                "substantive_codes": "[]",
                "missing_codes": "[]",
                "year_available": "[2022, 2023, 2024]",
                "source_module": "Sample Adult",
                "source_section": "Cost-related barriers / outcomes",
                "include_primary": False,
                "include_expanded": False,
                "exclusion_reason": fspec["reason"],
                "protected_proxy_note": None,
                "documentation_source": "CDC/NCHS NHIS 2022-2024 Adult Codebooks",
            }
        )

    return pd.DataFrame(rows)


def _spec_to_audit_row(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "official_name": spec["official_name"],
        "harmonized_name": spec.get("harmonized_name"),
        "domain": spec["domain"],
        "feature_tier": spec["feature_tier"],
        "role": spec["role"],
        "semantic_type": spec["semantic_type"],
        "official_description": spec["official_description"],
        "substantive_codes": str(spec.get("substantive_codes", [])),
        "missing_codes": str(spec.get("missing_codes", [])),
        "year_available": str(spec.get("year_available", [])),
        "source_module": spec.get("source_module"),
        "source_section": spec.get("source_section"),
        "include_primary": spec.get("include_primary", False),
        "include_expanded": spec.get("include_expanded", False),
        "exclusion_reason": spec.get("exclusion_reason"),
        "protected_proxy_note": spec.get("protected_proxy_note"),
        "documentation_source": spec.get("documentation_source"),
    }


def build_feature_availability_audit_table(
    raw_headers: Mapping[int, Sequence[str]],
    years: Sequence[int],
    registry: Mapping[str, Any],
) -> pd.DataFrame:
    """Audit cross-year availability and compatibility of all candidate features."""

    rows = []
    all_candidates = []
    for spec in registry["primary_core"].values():
        all_candidates.append((spec["official_name"], spec["feature_tier"]))
    for spec in registry["expanded_utilization"].values():
        all_candidates.append((spec["official_name"], spec["feature_tier"]))
    for spec in registry.get("rejected_candidates", {}).values():
        all_candidates.append((spec["official_name"], spec["feature_tier"]))

    for feature_name, tier in all_candidates:
        for year in years:
            year_int = int(year)
            present = feature_name in raw_headers[year_int]
            if tier == "primary_core":
                compat = "COMPATIBLE" if present else "INCOMPATIBLE_MISSING"
                status = "PASS" if present else "FAIL"
            elif tier == "expanded_utilization":
                compat = "COMPATIBLE" if present else "INCOMPATIBLE_MISSING"
                status = "PASS" if present else "FAIL"
            else:
                compat = "INCOMPATIBLE_REJECTED"
                status = "EXCLUDED"

            rows.append(
                {
                    "feature": feature_name,
                    "year": year_int,
                    "present": present,
                    "semantic_compatibility": compat,
                    "status": status,
                }
            )

    return pd.DataFrame(rows)


def build_feature_code_domain_audit_table(
    raw_frames: Mapping[int, pd.DataFrame],
    years: Sequence[int],
    registry: Mapping[str, Any],
) -> pd.DataFrame:
    """Audit observed raw code domains vs expected substantive and missing ranges."""

    rows = []
    candidates = {}
    candidates.update(registry["primary_core"])
    candidates.update(registry["expanded_utilization"])

    for feature_name, spec in candidates.items():
        substantive = set(spec["substantive_codes"])
        missing = set(spec["missing_codes"])
        allowed = substantive | missing

        for year in years:
            year_int = int(year)
            df = raw_frames[year_int]
            if feature_name not in df.columns:
                rows.append(
                    {
                        "feature": feature_name,
                        "year": year_int,
                        "observed_raw_codes_or_range": "NOT_PRESENT",
                        "expected_substantive_codes_or_range": f"{min(substantive)}-{max(substantive)}" if substantive else "NONE",
                        "unexpected_values": "FEATURE_ABSENT",
                        "status": "FAIL",
                    }
                )
                continue

            s = df[feature_name].dropna()
            numeric_s = pd.to_numeric(s, errors="coerce")
            obs_vals = sorted(numeric_s.dropna().astype(int).unique())
            unexpected = set(obs_vals) - allowed

            if substantive:
                min_s, max_s = min(substantive), max(substantive)
                exp_range = f"{min_s}-{max_s}" if max_s > min_s else f"{min_s}"
            else:
                exp_range = "NONE"

            if obs_vals:
                min_o, max_o = min(obs_vals), max(obs_vals)
                obs_range = f"{min_o}-{max_o}" if max_o > min_o else f"{min_o}"
            else:
                obs_range = "EMPTY"

            status = "PASS" if len(unexpected) == 0 else "FAIL"
            rows.append(
                {
                    "feature": feature_name,
                    "year": year_int,
                    "observed_raw_codes_or_range": obs_range,
                    "expected_substantive_codes_or_range": exp_range,
                    "unexpected_values": str(sorted(unexpected)) if unexpected else "NONE",
                    "status": status,
                }
            )

    return pd.DataFrame(rows)


def build_feature_missingness_audit_table(
    feature_frame: pd.DataFrame,
    years: Sequence[int],
    registry: Mapping[str, Any],
) -> pd.DataFrame:
    """Audit unweighted and survey-weighted missingness for all harmonized features."""

    rows = []
    candidates = {}
    candidates.update(registry["primary_core"])
    candidates.update(registry["expanded_utilization"])

    for raw_name, spec in candidates.items():
        harm_name = spec["harmonized_name"]
        for year in years:
            year_int = int(year)
            sub = feature_frame.loc[feature_frame["survey_year"].eq(year_int)]
            n = len(sub)
            missing_n = int(sub[harm_name].isna().sum())
            missing_frac = float(missing_n / n) if n > 0 else 0.0

            # Survey-weighted missingness fraction using WTFA_A
            wt = sub["WTFA_A"].astype(float)
            total_wt = float(wt.sum())
            missing_wt = float(wt.loc[sub[harm_name].isna()].sum())
            wt_missing_frac = float(missing_wt / total_wt) if total_wt > 0 else 0.0

            rows.append(
                {
                    "feature": harm_name,
                    "official_name": raw_name,
                    "year": year_int,
                    "n": n,
                    "missing_n": missing_n,
                    "missing_fraction": missing_frac,
                    "weighted_missing_fraction": wt_missing_frac,
                }
            )

    return pd.DataFrame(rows)


def build_feature_distribution_audit_table(
    feature_frame: pd.DataFrame,
    years: Sequence[int],
    registry: Mapping[str, Any],
) -> pd.DataFrame:
    """Produce cross-year category distributions and quantitative summary statistics."""

    rows = []
    candidates = {}
    candidates.update(registry["primary_core"])
    candidates.update(registry["expanded_utilization"])

    for raw_name, spec in candidates.items():
        harm_name = spec["harmonized_name"]
        sem_type = spec["semantic_type"]
        is_categorical = sem_type in ("categorical_nominal", "categorical_binary", "ordinal")

        for year in years:
            year_int = int(year)
            sub = feature_frame.loc[feature_frame["survey_year"].eq(year_int)]
            weights = sub["WTFA_A"].astype(float)
            series = sub[harm_name]

            if is_categorical:
                valid_codes = tuple(sorted(spec["substantive_codes"]))
                for code in valid_codes:
                    cat_mask = series.eq(code)
                    cat_n = int(cat_mask.sum())
                    wt_prop = weighted_category_proportion(
                        series, weights, code=code, valid_codes=valid_codes
                    )
                    rows.append(
                        {
                            "year": year_int,
                            "feature": harm_name,
                            "feature_type": "categorical",
                            "category": str(code),
                            "n": cat_n,
                            "weighted_proportion": wt_prop,
                            "metric": "category_proportion",
                            "value": wt_prop,
                        }
                    )
                # Also record missing/NA count and proportion
                missing_n = int(series.isna().sum())
                wt_missing = weighted_category_proportion(
                    series, weights, code=None, valid_codes=valid_codes
                )
                rows.append(
                    {
                        "year": year_int,
                        "feature": harm_name,
                        "feature_type": "categorical",
                        "category": "MISSING_OR_NA",
                        "n": missing_n,
                        "weighted_proportion": wt_missing,
                        "metric": "missing_proportion",
                        "value": wt_missing,
                    }
                )
            else:
                valid_mask = series.notna()
                valid_vals = series.loc[valid_mask].astype(float)
                valid_wts = weights.loc[valid_mask]
                n_valid = int(valid_mask.sum())

                if n_valid > 0:
                    mean_val = float(valid_vals.mean())
                    std_val = float(valid_vals.std()) if n_valid > 1 else 0.0
                    min_val = float(valid_vals.min())
                    p25_val = float(valid_vals.quantile(0.25))
                    med_val = float(valid_vals.median())
                    p75_val = float(valid_vals.quantile(0.75))
                    max_val = float(valid_vals.max())
                    wt_mean = float((valid_vals * valid_wts).sum() / valid_wts.sum()) if valid_wts.sum() > 0 else mean_val
                else:
                    mean_val = std_val = min_val = p25_val = med_val = p75_val = max_val = wt_mean = np.nan

                metrics = [
                    ("mean", mean_val),
                    ("std", std_val),
                    ("min", min_val),
                    ("p25", p25_val),
                    ("median", med_val),
                    ("p75", p75_val),
                    ("max", max_val),
                    ("weighted_mean", wt_mean),
                ]
                for m_name, m_val in metrics:
                    rows.append(
                        {
                            "year": year_int,
                            "feature": harm_name,
                            "feature_type": "quantitative",
                            "category": "ALL_SUBSTANTIVE",
                            "n": n_valid,
                            "weighted_proportion": np.nan,
                            "metric": m_name,
                            "value": m_val,
                        }
                    )

    return pd.DataFrame(rows)


def build_leakage_guard_audit_table(
    registry: Mapping[str, Any],
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Audit that every forbidden variable is strictly excluded from feature sets."""

    fl = registry.get("feature_lists", {})
    primary_list = set(fl.get("primary_core_features", []))
    expanded_list = set(fl.get("expanded_utilization_features", []))

    # All produced X feature columns
    primary_x_cols = set(primary_list)
    expanded_x_cols = set(primary_list) | set(expanded_list)

    rows = []

    # Check forbidden outcome-proximal variables
    for var_name, fspec in registry["forbidden_outcome_proximal"].items():
        in_p_list = var_name in primary_list or var_name.lower() in primary_list
        in_e_list = var_name in expanded_list or var_name.lower() in expanded_list
        in_p_x = var_name in primary_x_cols or var_name.lower() in primary_x_cols
        in_e_x = var_name in expanded_x_cols or var_name.lower() in expanded_x_cols

        status = "PASS" if not (in_p_list or in_e_list or in_p_x or in_e_x) else "FAIL_LEAKAGE"
        rows.append(
            {
                "variable": var_name,
                "category": "forbidden_outcome_proximal",
                "forbidden_from_primary_x": True,
                "forbidden_from_expanded_x": True,
                "in_primary_feature_list": in_p_list,
                "in_expanded_feature_list": in_e_list,
                "in_primary_x": in_p_x,
                "in_expanded_x": in_e_x,
                "status": status,
            }
        )

    # Check protected attributes
    for prot_name in registry["protected_attributes"]:
        in_p_list = prot_name in primary_list or prot_name.lower() in primary_list
        in_e_list = prot_name in expanded_list or prot_name.lower() in expanded_list
        in_p_x = prot_name in primary_x_cols or prot_name.lower() in primary_x_cols
        in_e_x = prot_name in expanded_x_cols or prot_name.lower() in expanded_x_cols

        status = "PASS" if not (in_p_list or in_e_list or in_p_x or in_e_x) else "FAIL_LEAKAGE"
        rows.append(
            {
                "variable": prot_name,
                "category": "protected_attribute",
                "forbidden_from_primary_x": True,
                "forbidden_from_expanded_x": True,
                "in_primary_feature_list": in_p_list,
                "in_expanded_feature_list": in_e_list,
                "in_primary_x": in_p_x,
                "in_expanded_x": in_e_x,
                "status": status,
            }
        )

    # Check outcomes
    for out_name in registry["outcomes"]:
        in_p_list = out_name in primary_list or out_name.lower() in primary_list or "meddl12m" in primary_list or "meddly12m" in primary_list
        in_e_list = out_name in expanded_list or out_name.lower() in expanded_list
        in_p_x = out_name in primary_x_cols or out_name.lower() in primary_x_cols or "meddl12m" in primary_x_cols or "meddly12m" in primary_x_cols
        in_e_x = out_name in expanded_x_cols or out_name.lower() in expanded_x_cols

        status = "PASS" if not (in_p_list or in_e_list or in_p_x or in_e_x) else "FAIL_LEAKAGE"
        rows.append(
            {
                "variable": out_name,
                "category": "outcome",
                "forbidden_from_primary_x": True,
                "forbidden_from_expanded_x": True,
                "in_primary_feature_list": in_p_list,
                "in_expanded_feature_list": in_e_list,
                "in_primary_x": in_p_x,
                "in_expanded_x": in_e_x,
                "status": status,
            }
        )

    df_result = pd.DataFrame(rows)
    if (df_result["status"] != "PASS").any():
        failed = df_result.loc[df_result["status"] != "PASS", "variable"].tolist()
        raise NHISFeatureError(f"Leakage guard FAILED for forbidden variables: {failed}")
    return df_result


def build_feature_manifest(
    *,
    repo_root: pathlib.Path | str,
    study_config: Mapping[str, Any],
    feature_registry: Mapping[str, Any],
    years: Sequence[int],
    processed_parquet_path: pathlib.Path | str,
    audit_paths: Mapping[str, pathlib.Path | str],
    run_id: str,
    feature_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Assemble the immutable Gate D1 feature manifest."""

    repo_path = pathlib.Path(repo_root).resolve()
    parquet_path = pathlib.Path(processed_parquet_path).resolve()

    manifest: dict[str, Any] = {
        "schema_version": "nhis-fairbias-d1-1.0",
        "gate": "D1",
        "dataset": study_config.get("dataset", "CDC/NCHS NHIS Sample Adult public-use CSV"),
        "software_timestamp_utc": utc_timestamp(),
        "git_commit_sha": _get_git_sha(repo_path),
        "run_id": run_id,
        "years": {str(int(y)): {"study_role": study_config["years"][str(int(y))]["study_role"]} for y in sorted(years)},
        "row_counts": {
            str(int(y)): int((feature_frame["survey_year"] == int(y)).sum()) for y in sorted(years)
        },
        "total_row_count": len(feature_frame),
        "primary_feature_count": len(feature_registry["feature_lists"]["primary_core_features"]),
        "expanded_feature_count": len(feature_registry["feature_lists"]["expanded_utilization_features"]),
        "feature_lists": feature_registry["feature_lists"],
        "processed_file": {
            "path": parquet_path.relative_to(repo_path).as_posix(),
            "sha256": compute_sha256(parquet_path) if parquet_path.is_file() else None,
            "row_count": len(feature_frame),
            "column_count": len(feature_frame.columns),
        },
        "audit_artifacts": {},
    }

    manifest["row_counts"]["total"] = len(feature_frame)

    for audit_key, a_path in audit_paths.items():
        ap = pathlib.Path(a_path).resolve()
        manifest["audit_artifacts"][audit_key] = {
            "path": ap.relative_to(repo_path).as_posix(),
            "sha256": compute_sha256(ap) if ap.is_file() else None,
        }

    return manifest


def _get_git_sha(repo_root: pathlib.Path) -> str | None:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return res.stdout.strip() or None
    except Exception:
        return None
