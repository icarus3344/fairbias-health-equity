"""Data contracts and synthetic cohort generator for NHIS Benchmark V1.

Implements:
1. 4 Experimental Arms:
   - Arm 001: SEX_A (2 groups), 21 primary core features
   - Arm 002: HISPALLP_A (7 groups), 21 primary core features
   - Arm 003: DISAB3_A (2 groups), 21 primary core features
   - Arm 004: DISAB3_A (2 groups), 15 primary core features (excludes 6 components)
2. Master Survey Design Partitioning (2022 PSU-level F/C split):
   - Sorted unique (PSTRAT, PPSU) clusters
   - Reproducible PCG64 PRNG (seed=20260913)
   - Uniform u < 0.20 -> Set C (Calibration), else Set F (Fitting)
   - 2023 -> Set S (Selection)
   - 2024 -> Set T (Frozen Test Evaluation)
3. Strict Record Identity:
   - record_key = f"{source}:{int(year)}:{id_norm}"
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

PRIMARY_CORE_21 = (
    "agep_a",
    "educp_a",
    "region",
    "pcnt18uptc",
    "pcntlt18tc",
    "ratcat_a",
    "empwrklsw1_a",
    "empwrkft1_a",
    "notcov_a",
    "phstat_a",
    "hypev_a",
    "chlev_a",
    "dibev_a",
    "asev_a",
    "visiondf_a",
    "hearingdf_a",
    "diff_a",
    "comdiff_a",
    "uppslfcr_a",
    "cogmemdff_a",
    "usualpl_a",
)

DISABILITY_COMPONENTS_6 = (
    "visiondf_a",
    "hearingdf_a",
    "diff_a",
    "comdiff_a",
    "uppslfcr_a",
    "cogmemdff_a",
)

PRIMARY_CORE_15 = tuple(f for f in PRIMARY_CORE_21 if f not in DISABILITY_COMPONENTS_6)

ARM_SPECS = {
    "arm_001": {
        "arm_id": "arm_001",
        "protected_attribute": "SEX_A",
        "expected_categories": [1, 2],
        "category_labels": {1: "Male", 2: "Female"},
        "features": PRIMARY_CORE_21,
        "feature_count": 21,
    },
    "arm_002": {
        "arm_id": "arm_002",
        "protected_attribute": "HISPALLP_A",
        "expected_categories": [1, 2, 3, 4, 5, 6, 7],
        "category_labels": {
            1: "Hispanic",
            2: "NH White",
            3: "NH Black",
            4: "NH Asian",
            5: "NH AIAN",
            6: "NH AIAN & Other",
            7: "NH Other / Multiple",
        },
        "features": PRIMARY_CORE_21,
        "feature_count": 21,
    },
    "arm_003": {
        "arm_id": "arm_003",
        "protected_attribute": "DISAB3_A",
        "expected_categories": [1, 2],
        "category_labels": {1: "With Disability", 2: "Without Disability"},
        "features": PRIMARY_CORE_21,
        "feature_count": 21,
    },
    "arm_004": {
        "arm_id": "arm_004",
        "protected_attribute": "DISAB3_A",
        "expected_categories": [1, 2],
        "category_labels": {1: "With Disability", 2: "Without Disability"},
        "features": PRIMARY_CORE_15,
        "feature_count": 15,
    },
}


@dataclasses.dataclass(frozen=True)
class AnnualSurveyDesign:
    """Full annual respondent design, including zero-contribution domain PSUs."""

    year: int
    record_keys: np.ndarray
    strata: np.ndarray
    psus: np.ndarray
    weights: np.ndarray
    domain_mask: np.ndarray

    def expand(self, domain_values: Any, fill: float = 0.0) -> np.ndarray:
        values = np.asarray(domain_values)
        if values.ndim != 1 or len(values) != int(self.domain_mask.sum()):
            raise ValueError("Domain values do not align with annual design rows")
        expanded = np.full(len(self.record_keys), fill, dtype=float)
        expanded[self.domain_mask] = values
        return expanded


@dataclasses.dataclass(frozen=True)
class PartitionDataset:
    """Immutable benchmark partition conforming to contract schema."""

    role: str  # 'fitting_F', 'calibration_C', 'selection_S', 'evaluation_T'
    year: int
    record_keys: np.ndarray
    X_semantic: pd.DataFrame
    y: np.ndarray
    A: np.ndarray
    WTFA_A: np.ndarray
    PSTRAT: np.ndarray
    PPSU: np.ndarray
    feature_names: Tuple[str, ...]
    arm_id: str
    metadata: Dict[str, Any]
    annual_design: Optional[AnnualSurveyDesign] = None

    def __len__(self) -> int:
        return len(self.y)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "year": int(self.year),
            "record_keys": self.record_keys,
            "X_semantic": self.X_semantic,
            "y": self.y,
            "A": self.A,
            "WTFA_A": self.WTFA_A,
            "PSTRAT": self.PSTRAT,
            "PPSU": self.PPSU,
            "feature_names": list(self.feature_names),
            "arm_id": self.arm_id,
            "metadata": dict(self.metadata),
        }


def make_record_key(source: str, year: int, id_norm: Union[int, str]) -> str:
    """Generate canonical record key: source:int(year):str(id_norm)."""
    if isinstance(year, (bool, np.bool_)) or not np.isfinite(float(year)) or float(year) != int(year):
        raise ValueError("Record year must be a finite integer")
    if pd.isna(id_norm) or not str(id_norm).strip():
        raise ValueError("A source respondent identifier is required; row positions are not identifiers")
    year_int = int(year)
    return f"{source}:{year_int}:{str(id_norm)}"


def partition_master_psus(
    strata: np.ndarray,
    psus: np.ndarray,
    *,
    seed: int = 20260913,
    calib_fraction: float = 0.20,
) -> Dict[Tuple[int, int], str]:
    """Execute master PSU-level F/C assignment on unique (PSTRAT, PPSU) clusters.

    Cluster assignment is atomic: no PSU cluster is split between F and C.
    """
    strata = _design_integer_vector(strata, "strata")
    psus = _design_integer_vector(psus, "psus")
    if len(strata) != len(psus):
        raise ValueError("Strata and PSU lengths differ")
    if not np.isfinite(calib_fraction) or not 0 < calib_fraction < 1:
        raise ValueError("Calibration fraction must lie strictly between 0 and 1")
    unique_clusters = sorted(
        {(int(s), int(p)) for s, p in zip(strata, psus)},
        key=lambda x: (x[0], x[1]),
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    cluster_mapping = {}
    for cluster in unique_clusters:
        u = float(rng.uniform(0.0, 1.0))
        cluster_mapping[cluster] = "calibration_C" if u < calib_fraction else "fitting_F"
    return cluster_mapping


def _design_integer_vector(values: Any, label: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    numeric = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    if not np.all(np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
        raise ValueError(f"{label} must contain finite integer identifiers")
    return numeric.astype(np.int64)


def generate_synthetic_nhis_cohort(
    *,
    n_records_per_year: int = 2000,
    years: Sequence[int] = (2022, 2023, 2024),
    n_strata: int = 20,
    psus_per_stratum: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a realistic synthetic NHIS multi-year survey cohort.

    Guarantees:
    - Zero microdata read or disclosure.
    - All 21 primary core features generated within legal CDC substantive code ranges.
    - Complex survey variables: PSTRAT, PPSU, positive finite WTFA_A.
    - Demographics: SEX_A, HISPALLP_A (7 classes), DISAB3_A.
    - Outcome: MEDDL12M_A (0/1 binary).
    """
    rng = np.random.default_rng(seed)
    dfs = []

    for yr in years:
        n = n_records_per_year
        # Survey design variables
        strata = rng.choice(np.arange(101, 101 + n_strata), size=n)
        psu_draws = rng.choice(np.arange(1, 1 + psus_per_stratum), size=n)
        # Survey weights: lognormal distribution with mean ~ 1000, strictly positive
        weights = rng.lognormal(mean=7.0, sigma=0.6, size=n)

        # Protected attributes
        # SEX_A: 1 = Male (48%), 2 = Female (52%)
        sex = rng.choice([1, 2], size=n, p=[0.48, 0.52])
        # HISPALLP_A: 7 groups with realistic disparities
        hisp = rng.choice(
            [1, 2, 3, 4, 5, 6, 7],
            size=n,
            p=[0.18, 0.60, 0.12, 0.06, 0.015, 0.010, 0.015],
        )
        # DISAB3_A: 1 = With disability (15%), 2 = Without (85%)
        disab = rng.choice([1, 2], size=n, p=[0.15, 0.85])

        # Semantic Predictor Features (21 primary core)
        # 1. agep_a: 18..85
        age = rng.integers(18, 86, size=n)
        # 2. pcnt18uptc: 1, 2, 3
        pcnt_adults = rng.choice([1, 2, 3], size=n, p=[0.30, 0.50, 0.20])
        # 3. pcntlt18tc: 0, 1, 2, 3
        pcnt_children = rng.choice([0, 1, 2, 3], size=n, p=[0.60, 0.20, 0.15, 0.05])
        # 4. educp_a: 1..10
        educ = rng.integers(1, 11, size=n)
        # 5. region: 1..4
        region = rng.choice([1, 2, 3, 4], size=n, p=[0.18, 0.22, 0.38, 0.22])
        # 6. ratcat_a: 1..14
        ratcat = rng.integers(1, 15, size=n)
        # 7. empwrklsw1_a: 1 (worked), 2 (did not work)
        emp_work = rng.choice([1, 2], size=n, p=[0.65, 0.35])
        # 8. empwrkft1_a: 1 (full time), 2 (part time), -1 (structural not in universe)
        emp_ft = np.where(emp_work == 1, rng.choice([1, 2], size=n, p=[0.80, 0.20]), -1)
        # 9. notcov_a: 1 (not covered / uninsured), 2 (covered / insured)
        # correlated with poverty (ratcat) and employment
        unins_p = 0.05 + 0.15 * (ratcat <= 5) + 0.10 * (emp_work == 2)
        unins = rng.binomial(1, np.clip(unins_p, 0.02, 0.40))
        notcov = np.where(unins == 1, 1, 2)
        # 10. phstat_a: 1 (excellent) .. 5 (poor)
        phstat = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.25, 0.35, 0.25, 0.10, 0.05])
        # 11-14. chronic conditions (1=yes, 2=no)
        hypev = rng.choice([1, 2], size=n, p=[0.32, 0.68])
        chlev = rng.choice([1, 2], size=n, p=[0.28, 0.72])
        dibev = rng.choice([1, 2], size=n, p=[0.12, 0.88])
        asev = rng.choice([1, 2], size=n, p=[0.10, 0.90])
        # 15-20. 6 disability components (1=no difficulty .. 4=cannot do at all)
        # tied to DISAB3_A
        disab_severe = (disab == 1)
        visiondf = np.where(disab_severe, rng.choice([1, 2, 3, 4], size=n, p=[0.2, 0.4, 0.3, 0.1]), 1)
        hearingdf = np.where(disab_severe, rng.choice([1, 2, 3, 4], size=n, p=[0.3, 0.4, 0.2, 0.1]), 1)
        diff_walk = np.where(disab_severe, rng.choice([1, 2, 3, 4], size=n, p=[0.1, 0.4, 0.3, 0.2]), 1)
        comdiff = np.where(disab_severe, rng.choice([1, 2, 3, 4], size=n, p=[0.4, 0.4, 0.15, 0.05]), 1)
        uppslfcr = np.where(disab_severe, rng.choice([1, 2, 3, 4], size=n, p=[0.4, 0.3, 0.2, 0.1]), 1)
        cogmemdff = np.where(disab_severe, rng.choice([1, 2, 3, 4], size=n, p=[0.3, 0.4, 0.2, 0.1]), 1)
        # 21. usualpl_a: 1 (yes), 2 (no), 3 (more than one place)
        usualpl = rng.choice([1, 2, 3], size=n, p=[0.82, 0.12, 0.06])

        # Latent outcome probability (cost-delayed care: MEDDL12M_A)
        # Driven by uninsured status, poor health, disability, poverty, and systemic demographic disparity
        logit_y = (
            -2.2
            + 1.5 * (notcov == 1)
            + 0.6 * (phstat >= 4)
            + 0.5 * (disab == 1)
            + 0.4 * (ratcat <= 4)
            + 0.3 * (sex == 2)  # known female reporting gap
            + 0.25 * (hisp == 1)
            + 0.20 * (hisp == 3)
            + 0.01 * (age - 45)
        )
        prob_y = 1.0 / (1.0 + np.exp(-logit_y))
        y = rng.binomial(1, prob_y)

        records = pd.DataFrame({
            "HHX": [f"{yr}_{i:06d}" for i in range(n)],
            "year": yr,
            "WTFA_A": weights,
            "PSTRAT": strata,
            "PPSU": psu_draws,
            "MEDDL12M_A": y,
            "SEX_A": sex,
            "HISPALLP_A": hisp,
            "DISAB3_A": disab,
            "agep_a": age,
            "educp_a": educ,
            "region": region,
            "pcnt18uptc": pcnt_adults,
            "pcntlt18tc": pcnt_children,
            "ratcat_a": ratcat,
            "empwrklsw1_a": emp_work,
            "empwrkft1_a": emp_ft,
            "notcov_a": notcov,
            "phstat_a": phstat,
            "hypev_a": hypev,
            "chlev_a": chlev,
            "dibev_a": dibev,
            "asev_a": asev,
            "visiondf_a": visiondf,
            "hearingdf_a": hearingdf,
            "diff_a": diff_walk,
            "comdiff_a": comdiff,
            "uppslfcr_a": uppslfcr,
            "cogmemdff_a": cogmemdff,
            "usualpl_a": usualpl,
        })
        dfs.append(records)

    return pd.concat(dfs, ignore_index=True)


def load_arm_partitions(
    df: pd.DataFrame,
    arm_id: str,
    *,
    psu_master_seed: int = 20260913,
) -> Dict[str, PartitionDataset]:
    """Partition cohort into F (Fit), C (Calibration), S (Selection), T (Test) for a specified arm."""
    if arm_id not in ARM_SPECS:
        raise ValueError(f"Unknown arm_id: {arm_id}. Valid arms: {list(ARM_SPECS.keys())}")

    spec = ARM_SPECS[arm_id]
    prot_col = spec["protected_attribute"]
    features = spec["features"]

    required = set(features) | {prot_col, "year", "HHX", "MEDDL12M_A", "PSTRAT", "PPSU", "WTFA_A"}
    if not df.columns.is_unique or required - set(df.columns):
        raise ValueError("Cohort has duplicated or missing required columns")
    master = df.copy().reset_index(drop=True)
    master["year"] = _design_integer_vector(master["year"], "year")
    if not set(master["year"]).issubset({2022, 2023, 2024}):
        raise ValueError("Benchmark years must be 2022, 2023, 2024")
    master["PSTRAT"] = _design_integer_vector(master["PSTRAT"], "PSTRAT")
    master["PPSU"] = _design_integer_vector(master["PPSU"], "PPSU")
    w = pd.to_numeric(master["WTFA_A"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("Complete annual design requires finite positive WTFA_A")
    master["WTFA_A"] = w
    master["_record_key"] = [make_record_key("nhis", yr, hhx) for yr, hhx in zip(master.year, master.HHX)]
    if master["_record_key"].duplicated().any():
        raise ValueError("Duplicate source respondent identifiers within a survey year")
    y = pd.to_numeric(master["MEDDL12M_A"], errors="coerce")
    if (master["MEDDL12M_A"].notna() & ~y.isin([0, 1])).any():
        raise ValueError("Benchmark outcome must be recoded 0/1 or missing, never raw NHIS codes")
    master["MEDDL12M_A"] = y

    # Allocate all annual PSUs BEFORE any outcome or protected-domain exclusions.
    df_2022 = master[master["year"] == 2022]
    psu_mapping = partition_master_psus(
        df_2022["PSTRAT"].values,
        df_2022["PPSU"].values,
        seed=psu_master_seed,
    )

    assignment_hash = hashlib.sha256(json.dumps(
        [[s, p, role] for (s, p), role in sorted(psu_mapping.items())], separators=(",", ":")
    ).encode()).hexdigest()

    partitions = {}
    for role, yr in [
        ("fitting_F", 2022),
        ("calibration_C", 2022),
        ("selection_S", 2023),
        ("evaluation_T", 2024),
    ]:
        annual = master[master["year"] == yr].copy()
        domain = (annual[prot_col].isin(spec["expected_categories"]) & annual["MEDDL12M_A"].isin([0, 1])).to_numpy()
        if yr == 2022:
            assigned = np.array([psu_mapping[(s, p)] for s, p in zip(annual.PSTRAT, annual.PPSU)])
            domain &= assigned == role
        frame = annual.loc[domain]
        rec_keys = frame["_record_key"].to_numpy(copy=True)
        design = AnnualSurveyDesign(
            year=yr, record_keys=annual["_record_key"].to_numpy(copy=True),
            strata=annual.PSTRAT.to_numpy(copy=True), psus=annual.PPSU.to_numpy(copy=True),
            weights=annual.WTFA_A.to_numpy(dtype=float, copy=True), domain_mask=domain.copy(),
        )
        p = PartitionDataset(
            role=role,
            year=yr,
            record_keys=rec_keys,
            X_semantic=frame[list(features)].copy().reset_index(drop=True),
            y=frame["MEDDL12M_A"].values.astype(int),
            A=frame[prot_col].values.astype(int),
            WTFA_A=frame["WTFA_A"].values.astype(float),
            PSTRAT=frame["PSTRAT"].values.astype(int),
            PPSU=frame["PPSU"].values.astype(int),
            feature_names=features,
            arm_id=arm_id,
            metadata={
                "protected_attribute": prot_col,
                "expected_categories": spec["expected_categories"],
                "category_labels": spec["category_labels"],
                "psu_assignment_sha256": assignment_hash,
                "split_seed": psu_master_seed,
                "full_annual_rows": len(annual),
                "domain_rows": len(frame),
                "source_provenance": dict(df.attrs.get("source_provenance", {})),
            },
            annual_design=design,
        )
        partitions[role] = p

    return partitions


def load_processed_nhis_cohort(parquet_path: Optional[str] = None) -> pd.DataFrame:
    """Load identified annual data without dropping non-domain design records.

    The historical parquet omits HHX. The default therefore rebuilds harmonized
    columns from local official CSVs, preserving source IDs, without changing
    historical artifacts. A caller-provided parquet must already carry HHX.
    """
    if parquet_path is None:
        return load_local_nhis_cohort()
    df = pd.read_parquet(parquet_path)
    if "HHX" not in df.columns:
        raise ValueError("Parquet lacks source HHX; positional record IDs are prohibited. Use local raw-source loader.")
    if "meddl12m" in df:
        df["MEDDL12M_A"] = df["meddl12m"]
    if "survey_year" in df:
        df["year"] = df["survey_year"]
    return df


def load_local_nhis_cohort(repo_root: Optional[Union[str, pathlib.Path]] = None, *,
                           years: Sequence[int] = (2022, 2023, 2024)) -> pd.DataFrame:
    """Stateless official-code harmonization; no fitted statistics or output files."""
    from ..features import harmonize_features_year, load_feature_registry
    from ..preprocessing import construct_empwrkft_series

    root = pathlib.Path(repo_root) if repo_root else pathlib.Path(__file__).resolve().parents[3]
    config_path = root / "configs/nhis/study.json"
    feature_path = root / "configs/nhis/features.json"
    study = json.loads(config_path.read_text())
    registry = load_feature_registry(feature_path)
    frames = []
    sources = {}
    needed = {"HHX", "SRVY_YR", "WTFA_A", "PSTRAT", "PPSU", "MEDDL12M_A", "MEDNG12M_A",
              "SEX_A", "HISPALLP_A", "DISAB3_A"}
    for tier in ("primary_core", "expanded_utilization"):
        needed.update(s["official_name"] for s in registry[tier].values())
    if not years or len(set(years)) != len(years) or not set(years).issubset({2022, 2023, 2024}):
        raise ValueError("years must be a nonempty unique subset of registered NHIS years")
    for yr in years:
        spec = study["years"][str(yr)]
        source = root / spec["local_csv_file"]
        h = hashlib.sha256()
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(block)
        raw = pd.read_csv(source, usecols=lambda name: name in needed, dtype={"HHX": "string"}, low_memory=False)
        if needed - set(raw.columns) or len(raw) != spec["expected_raw_rows"]:
            raise ValueError(f"Raw NHIS {yr} schema/row-count contract failed")
        if not np.all(_design_integer_vector(raw.SRVY_YR, "SRVY_YR") == yr):
            raise ValueError(f"Raw NHIS {yr} contains another survey year")
        if raw.HHX.isna().any() or raw.HHX.duplicated().any():
            raise ValueError(f"Raw NHIS {yr} respondent identifiers are missing or duplicated")
        frame = harmonize_features_year(raw, year=yr, study_role=spec["study_role"], feature_registry=registry)
        frame["HHX"] = raw.HHX.copy()
        frame["year"] = yr
        frame["MEDDL12M_A"] = frame["meddl12m"]
        frame["empwrkft1_a"] = construct_empwrkft_series(raw.EMPWRKLSW1_A, raw.EMPWRKFT1_A)
        frames.append(frame)
        sources[str(yr)] = {"path": spec["local_csv_file"], "sha256": h.hexdigest(), "rows": len(raw)}
    combined = pd.concat(frames, ignore_index=True)
    combined.attrs["source_provenance"] = {"raw_sources": sources, "identity": "official HHX + survey year",
        "feature_registry_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        "study_registry_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}
    return combined
