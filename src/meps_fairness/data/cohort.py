"""Cohort derivation, target construction, and variable harmonization for MEPS longitudinal data.

This module implements:
1. Deterministic cohort filtering:
   - YEARIND == 1 (Both years eligibility)
   - ALL5RDS == 1 (Completed all 5 rounds)
   - LONGWT > 0 (Positive longitudinal analysis weight)
   - 18 <= AGEY1X <= 64 (Working-age at end of baseline Year 1)
   - Baseline continuous coverage: INS...Y1X == 1 for all 12 baseline months
2. Strict target outcome construction:
   - Primary: Any uninsured month in Year 2 (any INS...Y2X == 2); zero if all 12 months == 1.
     Fails closed on any invalid (non-1/2) code.
   - Sensitivity 1: Prolonged uninsurance (sum of INS...Y2X == 2 >= 3)
   - Sensitivity 2: Year-end uninsurance (INSDEY2X == 2)
3. Audit dimensions:
   - Primary: RACETHX (5-category OMB), SEX (Male/Female)
   - Secondary: POVCATY1 (Poverty category), POVLEVY1 (Continuous poverty %),
     Age bands (18-24, 25-44, 45-64), Disability status (Round 1 composite limitation screener)
4. Survey design variables:
   - LONGWT (analysis weight), VARSTR (stratum), VARPSU (PSU), DUID (dwelling unit)
5. Conservative baseline predictor feature selection (74 features: 19 continuous, 55 categorical)
   with strict zero-leakage enforcement (Rounds 3, 4, 5 rejected; Round 1, Round 2, Y1 permitted).
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd

# 12 Baseline and 12 Follow-Up Monthly Insurance Variable Names
BASELINE_INSURANCE_MONTHS: tuple[str, ...] = (
    "INSJAY1X", "INSFEY1X", "INSMAY1X", "INSAPY1X",
    "INSMYY1X", "INSJUY1X", "INSJLY1X", "INSAUY1X",
    "INSSEY1X", "INSOCY1X", "INSNOY1X", "INSDEY1X",
)

FOLLOWUP_INSURANCE_MONTHS: tuple[str, ...] = (
    "INSJAY2X", "INSFEY2X", "INSMAY2X", "INSAPY2X",
    "INSMYY2X", "INSJUY2X", "INSJLY2X", "INSAUY2X",
    "INSSEY2X", "INSOCY2X", "INSNOY2X", "INSDEY2X",
)

DESIGN_COLUMNS: tuple[str, ...] = (
    "DUID", "PID", "DUPERSID", "PANEL", "YEARIND", "ALL5RDS", "LONGWT", "VARSTR", "VARPSU"
)

AUDIT_COLUMNS: tuple[str, ...] = (
    "RACETHX", "SEX", "POVCATY1", "POVLEVY1", "AGEY1X"
)

# Baseline predictor library across domains (Verified Round 1, Round 2, and Annual Y1 only)
# Round 3 variables are fielded in Year 2 and are strictly excluded to prevent temporal leakage.
DEMOGRAPHIC_PREDICTORS_CONT: tuple[str, ...] = ("AGEY1X", "EDUCYR", "FAMSZEY1")
DEMOGRAPHIC_PREDICTORS_CAT: tuple[str, ...] = (
    "HIDEG", "MARRY1X", "MARRY2X", "MARRYY1X",
    "REGION1", "REGION2", "REGIONY1"
)

SOCIOECONOMIC_PREDICTORS_CONT: tuple[str, ...] = (
    "POVLEVY1", "FAMINCY1", "TTLPY1X", "WAGEPY1X",
    "HOUR1", "HOUR2", "NUMEMP1", "NUMEMP2"
)
SOCIOECONOMIC_PREDICTORS_CAT: tuple[str, ...] = (
    "POVCATY1", "EMPST1", "EMPST2",
    "OFFER1X", "OFFER2X", "HELD1X", "HELD2X",
    "CHOIC1", "CHOIC2", "SELFCM1", "SELFCM2",
    "UNION1", "UNION2"
)

HEALTH_STATUS_PREDICTORS_CAT: tuple[str, ...] = (
    "RTHLTH1", "RTHLTH2", "MNHLTH1", "MNHLTH2",
    "HIBPDXY1", "DIABDXY1_M18", "ASTHDXY1", "CHDDXY1", "ANGIDXY1", "MIDXY1",
    "OHRTDXY1", "STRKDXY1", "EMPHDXY1", "CHBRON1", "CHOLDXY1",
    "CANCERY1", "ARTHDXY1", "JTPAIN1_M18"
)

FUNCTIONAL_LIMITATION_PREDICTORS_CAT: tuple[str, ...] = (
    "ACTLIM1", "ADLHLP1", "IADLHP1",
    "WLKLIM1", "SOCLIM1", "COGLIM1"
)

UTILIZATION_PREDICTORS_CONT: tuple[str, ...] = (
    "OBTOTVY1", "OBDRVY1", "OPTOTVY1", "ERTOTY1", "IPDISY1", "RXTOTY1",
    "TOTEXPY1", "TOTSLFY1"
)
UTILIZATION_PREDICTORS_CAT: tuple[str, ...] = (
    "HAVEUS2", "LOCATN2", "PROVTY2_M18"
)

COVERAGE_TYPE_PREDICTORS_CAT: tuple[str, ...] = (
    "INSCOVY1", "PRIEUY1", "PRINGY1", "PUBY1X", "MCAIDY1X", "MCAREY1X",
    "TRICRY1X", "VAPROGY1"
)

ALL_BASELINE_PREDICTOR_COLUMNS: tuple[str, ...] = tuple(sorted(set(
    DEMOGRAPHIC_PREDICTORS_CONT + DEMOGRAPHIC_PREDICTORS_CAT +
    SOCIOECONOMIC_PREDICTORS_CONT + SOCIOECONOMIC_PREDICTORS_CAT +
    HEALTH_STATUS_PREDICTORS_CAT + FUNCTIONAL_LIMITATION_PREDICTORS_CAT +
    UTILIZATION_PREDICTORS_CONT + UTILIZATION_PREDICTORS_CAT +
    COVERAGE_TYPE_PREDICTORS_CAT
)))

ALL_CONTINUOUS_PREDICTORS: tuple[str, ...] = tuple(sorted(set(
    DEMOGRAPHIC_PREDICTORS_CONT + SOCIOECONOMIC_PREDICTORS_CONT + UTILIZATION_PREDICTORS_CONT
)))

ALL_CATEGORICAL_PREDICTORS: tuple[str, ...] = tuple(sorted(set(
    DEMOGRAPHIC_PREDICTORS_CAT + SOCIOECONOMIC_PREDICTORS_CAT +
    HEALTH_STATUS_PREDICTORS_CAT + FUNCTIONAL_LIMITATION_PREDICTORS_CAT +
    UTILIZATION_PREDICTORS_CAT + COVERAGE_TYPE_PREDICTORS_CAT
)))


def is_prohibited_temporal_feature(col: str) -> bool:
    """Check whether a column is a prohibited post-baseline (Rounds 3-5) or Year 2 temporal feature."""
    if re.search(r"[345](X|_M18)?$", col):
        return True
    if col.startswith("INS") and "Y2" in col:
        return True
    if col.startswith("PROVTY"):
        return False
    if col.startswith("MARRY"):
        return col in ("MARRY3X", "MARRY4X", "MARRY5X", "MARRYY2X")
    if re.search(r"(Y2|Y2X|YR2)$", col) or re.search(r"Y2_M18$", col):
        return True
    return False


@dataclasses.dataclass(frozen=True)
class CohortData:
    """Encapsulates extracted and verified cohort data components."""

    panel: int
    raw_record_count: int
    eligible_record_count: int
    X: pd.DataFrame
    y: pd.Series
    y_prolonged: pd.Series
    y_yearend: pd.Series
    design: pd.DataFrame
    audit: pd.DataFrame
    continuous_features: tuple[str, ...]
    categorical_features: tuple[str, ...]

    def validate_leakage_and_integrity(self) -> None:
        """Run strict anti-leakage and structural checks."""
        # 1. Assert no Round 3, Round 4, Round 5, or Year 2 features in X
        for col in self.X.columns:
            if is_prohibited_temporal_feature(col):
                raise ValueError(f"Data leakage detected: prohibited temporal column {col} in predictor matrix X")

        # 2. Assert no identifiers in X
        for col in ("DUPERSID", "DUID", "PID", "PANEL"):
            if col in self.X.columns:
                raise ValueError(f"Data leakage detected: identifier {col} in predictor matrix X")

        # 3. Assert no design columns in X
        for col in ("VARSTR", "VARPSU", "LONGWT", "YEARIND", "ALL5RDS"):
            if col in self.X.columns:
                raise ValueError(f"Data leakage detected: survey design column {col} in predictor matrix X")

        # 4. Assert no audit demographic variables in X
        for col in ("RACETHX", "SEX", "RACEV1X", "HISPANX"):
            if col in self.X.columns:
                raise ValueError(f"Protected attribute {col} must not be in predictor matrix X")

        # 5. Assert design integrity
        if (self.design["LONGWT"] <= 0).any():
            raise ValueError("All cohort records must have LONGWT > 0")
        if self.design["VARSTR"].isna().any() or self.design["VARPSU"].isna().any():
            raise ValueError("Design variables VARSTR and VARPSU must be non-missing")


def resolve_prior_round_inheritance(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve prior round inheritance codes (-2: DETERMINED IN PREVIOUS ROUND).

    In MEPS longitudinal panels:
    - HOUR2 == -2 inherits HOUR1
    - NUMEMP2 == -2 inherits NUMEMP1
    - CHOIC2 == -2 inherits CHOIC1
    - SELFCM2 == -2 inherits SELFCM1
    - UNION2 == -2 inherits UNION1
    """
    df_out = df.copy()
    if "HOUR2" in df_out.columns and "HOUR1" in df_out.columns:
        mask = (df_out["HOUR2"] == -2)
        df_out.loc[mask, "HOUR2"] = df_out.loc[mask, "HOUR1"]
    if "NUMEMP2" in df_out.columns and "NUMEMP1" in df_out.columns:
        mask = (df_out["NUMEMP2"] == -2)
        df_out.loc[mask, "NUMEMP2"] = df_out.loc[mask, "NUMEMP1"]
    if "CHOIC2" in df_out.columns and "CHOIC1" in df_out.columns:
        mask = (df_out["CHOIC2"] == -2)
        df_out.loc[mask, "CHOIC2"] = df_out.loc[mask, "CHOIC1"]
    if "SELFCM2" in df_out.columns and "SELFCM1" in df_out.columns:
        mask = (df_out["SELFCM2"] == -2)
        df_out.loc[mask, "SELFCM2"] = df_out.loc[mask, "SELFCM1"]
    if "UNION2" in df_out.columns and "UNION1" in df_out.columns:
        mask = (df_out["UNION2"] == -2)
        df_out.loc[mask, "UNION2"] = df_out.loc[mask, "UNION1"]
    return df_out


def derive_composite_disability(df: pd.DataFrame) -> pd.Series:
    """Derive composite baseline disability/functional limitation indicator using Round 1 screeners only.

    Returns 1 if any Round 1 screener == 1 (Yes), 0 if all screeners == 2 (No), 0 if otherwise non-positive.
    """
    limitation_vars = [
        "ACTLIM1", "ADLHLP1", "IADLHP1", "WLKLIM1", "SOCLIM1", "COGLIM1"
    ]
    avail_vars = [v for v in limitation_vars if v in df.columns]
    if not avail_vars:
        return pd.Series(0, index=df.index, name="DISABILITY")

    has_limitation = (df[avail_vars] == 1).any(axis=1)
    all_no_limitation = (df[avail_vars] == 2).all(axis=1)

    result = pd.Series(np.nan, index=df.index, name="DISABILITY")
    result[has_limitation] = 1.0
    result[~has_limitation & all_no_limitation] = 0.0
    # Fill remaining (mix of 2 and non-response -7/-8) as 0.0 if not having limitation
    result[result.isna() & ~has_limitation] = 0.0
    return result


def derive_age_band(age_series: pd.Series) -> pd.Series:
    """Derive standardized 3-band age category."""
    bands = pd.Series("25-44", index=age_series.index, name="AGE_BAND")
    bands[age_series < 25] = "18-24"
    bands[age_series >= 45] = "45-64"
    return bands


def extract_meps_cohort(
    df: pd.DataFrame,
    panel_number: int,
    allow_target: bool = True,
) -> CohortData:
    """Extract and construct the analytic cohort from a loaded MEPS DataFrame.

    Parameters:
        df: Loaded raw MEPS Stata DataFrame for HC-244 or HC-252.
        panel_number: Expected panel number (26 or 27).
        allow_target: Whether to compute target outcomes (False for locked holdout pre-eval).

    Returns:
        CohortData instance.
    """
    raw_count = len(df)

    # 1. Eligibility Filters
    mask = (
        (df["YEARIND"] == 1) &
        (df["ALL5RDS"] == 1) &
        (df["LONGWT"] > 0) &
        (df["AGEY1X"] >= 18) &
        (df["AGEY1X"] <= 64)
    )

    # Baseline Continuous Insurance: INS...Y1X == 1 for all 12 months
    for month_col in BASELINE_INSURANCE_MONTHS:
        mask = mask & (df[month_col] == 1)

    cohort_df = df[mask].copy().reset_index(drop=True)
    eligible_count = len(cohort_df)

    # 2. Target Construction with Strict Valid-Code Contracts
    if allow_target:
        followup_df = cohort_df[list(FOLLOWUP_INSURANCE_MONTHS)]
        # Strict validation: each follow-up month must be valid code (1 or 2).
        # Any missing / non-response / out-of-scope code (-1, -7, -8, -15, etc.) is a stop condition.
        invalid_mask = ~followup_df.isin([1, 2])
        if invalid_mask.any().any():
            bad_count = int(invalid_mask.any(axis=1).sum())
            raise ValueError(
                f"Target ambiguity stop condition: detected {bad_count} eligible cohort rows with "
                f"invalid follow-up insurance codes (non 1 or 2). Coercion is strictly prohibited."
            )

        # Primary Outcome: Y = 1 if any month == 2; Y = 0 only if all 12 months == 1
        y_pos = (followup_df == 2).any(axis=1)
        y_neg = (followup_df == 1).all(axis=1)
        y = pd.Series(np.nan, index=cohort_df.index, name="y")
        y[y_pos] = 1.0
        y[y_neg] = 0.0

        if y.isna().any():
            raise ValueError("Target outcome contains unresolved indeterminate rows.")

        # Sensitivity 1: Prolonged uninsurance (>= 3 months)
        unins_count = (followup_df == 2).sum(axis=1)
        y_prolonged = (unins_count >= 3).astype(float)
        y_prolonged.name = "y_prolonged"

        # Sensitivity 2: Year-end uninsurance (INSDEY2X == 2)
        if not cohort_df["INSDEY2X"].isin([1, 2]).all():
            raise ValueError("Sensitivity outcome INSDEY2X contains invalid non-1/2 codes")
        y_yearend = (cohort_df["INSDEY2X"] == 2).astype(float)
        y_yearend.name = "y_yearend"
    else:
        y = pd.Series(np.nan, index=cohort_df.index, name="y")
        y_prolonged = pd.Series(np.nan, index=cohort_df.index, name="y_prolonged")
        y_yearend = pd.Series(np.nan, index=cohort_df.index, name="y_yearend")

    # 3. Design Variables
    design = cohort_df[[c for c in DESIGN_COLUMNS if c in cohort_df.columns]].copy()

    # 4. Audit Dimensions
    audit = pd.DataFrame(index=cohort_df.index)
    audit["RACETHX"] = cohort_df["RACETHX"].astype(int)
    audit["SEX"] = cohort_df["SEX"].astype(int)
    audit["POVCATY1"] = cohort_df["POVCATY1"].astype(int)
    audit["POVLEVY1"] = cohort_df["POVLEVY1"].astype(float)
    audit["AGEY1X"] = cohort_df["AGEY1X"].astype(int)
    audit["AGE_BAND"] = derive_age_band(cohort_df["AGEY1X"])
    audit["DISABILITY"] = derive_composite_disability(cohort_df)

    # 5. Predictor Matrix X with Prior-Round Inheritance
    cohort_inherited = resolve_prior_round_inheritance(cohort_df)
    predictor_cols = [c for c in ALL_BASELINE_PREDICTOR_COLUMNS if c in cohort_inherited.columns]
    X = cohort_inherited[predictor_cols].copy()

    cont_cols = tuple(c for c in ALL_CONTINUOUS_PREDICTORS if c in X.columns)
    cat_cols = tuple(c for c in ALL_CATEGORICAL_PREDICTORS if c in X.columns)

    cohort = CohortData(
        panel=panel_number,
        raw_record_count=raw_count,
        eligible_record_count=eligible_count,
        X=X,
        y=y,
        y_prolonged=y_prolonged,
        y_yearend=y_yearend,
        design=design,
        audit=audit,
        continuous_features=cont_cols,
        categorical_features=cat_cols,
    )
    cohort.validate_leakage_and_integrity()
    return cohort
