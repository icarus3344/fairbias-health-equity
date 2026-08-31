"""DUID-grouped stratified partitioning and partition integrity verification for MEPS Panel 26."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd

from meps_fairness.data.cohort import CohortData


@dataclasses.dataclass(frozen=True)
class PartitionData:
    """Encapsulates data for a single split partition."""

    name: str
    indices: np.ndarray
    duids: tuple[int, ...]
    X: pd.DataFrame
    y: pd.Series
    design: pd.DataFrame
    audit: pd.DataFrame

    @property
    def record_count(self) -> int:
        return len(self.indices)

    @property
    def duid_count(self) -> int:
        return len(self.duids)

    @property
    def positive_count(self) -> int:
        return int((self.y == 1.0).sum())

    @property
    def negative_count(self) -> int:
        return int((self.y == 0.0).sum())

    @property
    def total_weight(self) -> float:
        return float(self.design["LONGWT"].sum())

    @property
    def kish_effective_n(self) -> float:
        w = self.design["LONGWT"].values
        sum_w = np.sum(w)
        sum_w_sq = np.sum(w ** 2)
        if sum_w_sq == 0:
            return 0.0
        return float((sum_w ** 2) / sum_w_sq)


@dataclasses.dataclass(frozen=True)
class PartitionedCohort:
    """Encapsulates the 60/20/20 DUID-grouped split and refit partition."""

    seed: int
    split_assignment_hash: str
    train: PartitionData
    val: PartitionData
    cal: PartitionData
    train_val: PartitionData

    def validate_zero_leakage(self) -> None:
        """Verify strict zero-leakage across household (DUID) and row indices."""
        train_du = set(self.train.duids)
        val_du = set(self.val.duids)
        cal_du = set(self.cal.duids)

        if train_du & val_du:
            raise ValueError(f"DUID overlap between Train and Validation: {train_du & val_du}")
        if train_du & cal_du:
            raise ValueError(f"DUID overlap between Train and Calibration: {train_du & cal_du}")
        if val_du & cal_du:
            raise ValueError(f"DUID overlap between Validation and Calibration: {val_du & cal_du}")

        train_idx = set(self.train.indices)
        val_idx = set(self.val.indices)
        cal_idx = set(self.cal.indices)

        if train_idx & val_idx:
            raise ValueError("Row index overlap between Train and Validation")
        if train_idx & cal_idx:
            raise ValueError("Row index overlap between Train and Calibration")
        if val_idx & cal_idx:
            raise ValueError("Row index overlap between Validation and Calibration")


def split_panel26_duid_grouped(
    cohort: CohortData,
    seed: int = 20260828,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    cal_ratio: float = 0.2,
) -> PartitionedCohort:
    """Split Panel 26 cohort into 60% Train, 20% Validation, and 20% Calibration by DUID.

    Uses group-level stratification balancing positive outcomes and weight deciles.
    """
    if abs(train_ratio + val_ratio + cal_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    df_design = cohort.design
    df_y = cohort.y

    # Group summary by DUID
    duids = df_design["DUID"].unique()
    rng = np.random.RandomState(seed)

    # Compute group-level stratification features
    du_stats = []
    for du in duids:
        mask = (df_design["DUID"] == du)
        n_persons = mask.sum()
        n_pos = (df_y[mask] == 1.0).sum()
        tot_wt = df_design.loc[mask, "LONGWT"].sum()
        du_stats.append({
            "DUID": du,
            "n_persons": n_persons,
            "has_pos": int(n_pos > 0),
            "tot_wt": tot_wt,
        })

    df_du = pd.DataFrame(du_stats)

    # Compute weight deciles for stratification
    df_du["wt_bin"] = pd.qcut(df_du["tot_wt"], q=min(5, len(df_du)), labels=False, duplicates="drop")
    df_du["strat_key"] = df_du["has_pos"].astype(str) + "_" + df_du["wt_bin"].astype(str)

    train_duids: list[int] = []
    val_duids: list[int] = []
    cal_duids: list[int] = []

    for _, group in df_du.groupby("strat_key"):
        g_duids = group["DUID"].values.copy()
        rng.shuffle(g_duids)

        n_g = len(g_duids)
        n_tr = int(round(train_ratio * n_g))
        n_va = int(round(val_ratio * n_g))
        # Ensure at least 0 and not exceeding
        n_tr = min(n_tr, n_g)
        n_va = min(n_va, n_g - n_tr)

        train_duids.extend(g_duids[:n_tr])
        val_duids.extend(g_duids[n_tr:n_tr + n_va])
        cal_duids.extend(g_duids[n_tr + n_va:])

    # Map DUIDs to row indices
    train_mask = df_design["DUID"].isin(train_duids).values
    val_mask = df_design["DUID"].isin(val_duids).values
    cal_mask = df_design["DUID"].isin(cal_duids).values

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    cal_idx = np.where(cal_mask)[0]
    train_val_idx = np.where(train_mask | val_mask)[0]

    train_part = PartitionData(
        name="train",
        indices=train_idx,
        duids=tuple(sorted(train_duids)),
        X=cohort.X.iloc[train_idx].copy().reset_index(drop=True),
        y=cohort.y.iloc[train_idx].copy().reset_index(drop=True),
        design=cohort.design.iloc[train_idx].copy().reset_index(drop=True),
        audit=cohort.audit.iloc[train_idx].copy().reset_index(drop=True),
    )

    val_part = PartitionData(
        name="validation",
        indices=val_idx,
        duids=tuple(sorted(val_duids)),
        X=cohort.X.iloc[val_idx].copy().reset_index(drop=True),
        y=cohort.y.iloc[val_idx].copy().reset_index(drop=True),
        design=cohort.design.iloc[val_idx].copy().reset_index(drop=True),
        audit=cohort.audit.iloc[val_idx].copy().reset_index(drop=True),
    )

    cal_part = PartitionData(
        name="calibration",
        indices=cal_idx,
        duids=tuple(sorted(cal_duids)),
        X=cohort.X.iloc[cal_idx].copy().reset_index(drop=True),
        y=cohort.y.iloc[cal_idx].copy().reset_index(drop=True),
        design=cohort.design.iloc[cal_idx].copy().reset_index(drop=True),
        audit=cohort.audit.iloc[cal_idx].copy().reset_index(drop=True),
    )

    train_val_part = PartitionData(
        name="train_val",
        indices=train_val_idx,
        duids=tuple(sorted(set(train_duids) | set(val_duids))),
        X=cohort.X.iloc[train_val_idx].copy().reset_index(drop=True),
        y=cohort.y.iloc[train_val_idx].copy().reset_index(drop=True),
        design=cohort.design.iloc[train_val_idx].copy().reset_index(drop=True),
        audit=cohort.audit.iloc[train_val_idx].copy().reset_index(drop=True),
    )

    # Compute deterministic assignment hash
    assignment_record = [
        {"duid": int(du), "partition": "train" if du in train_duids else ("val" if du in val_duids else "cal")}
        for du in sorted(duids)
    ]
    assignment_json = json.dumps(assignment_record, separators=(",", ":"))
    split_hash = hashlib.sha256(assignment_json.encode("utf-8")).hexdigest()

    partitioned = PartitionedCohort(
        seed=seed,
        split_assignment_hash=split_hash,
        train=train_part,
        val=val_part,
        cal=cal_part,
        train_val=train_val_part,
    )
    partitioned.validate_zero_leakage()
    return partitioned
