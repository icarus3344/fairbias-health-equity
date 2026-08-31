"""Complex survey design-aware stratified-PSU bootstrap inference for paired comparisons."""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import numpy as np
import pandas as pd

from meps_fairness.evaluation.metrics import (
    primary_fairness_endpoint,
    weighted_auprc,
    weighted_auroc,
)


@dataclasses.dataclass(frozen=True)
class EndpointInference:
    """Statistical inference result for a single scalar endpoint."""

    point_estimate_unmitigated: float | None
    point_estimate_mitigated: float | None
    paired_difference: float | None
    std_error: float | None
    ci_95_lower: float | None
    ci_95_upper: float | None
    bootstrap_replicates_count: int
    status: str = "ESTIMABLE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "unmitigated": self.point_estimate_unmitigated,
            "mitigated": self.point_estimate_mitigated,
            "paired_difference": self.paired_difference,
            "std_error": self.std_error,
            "ci_95_lower": self.ci_95_lower,
            "ci_95_upper": self.ci_95_upper,
            "bootstrap_replicates": self.bootstrap_replicates_count,
            "status": self.status,
        }


@dataclasses.dataclass(frozen=True)
class BootstrapInferenceResult:
    """Comprehensive design-aware bootstrap inference results across primary and secondary endpoints."""

    utility_auprc: EndpointInference
    fairness_max_tpr_gap: EndpointInference
    utility_auroc: EndpointInference
    n_replicates: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "utility_auprc": self.utility_auprc.to_dict(),
            "fairness_max_tpr_gap": self.fairness_max_tpr_gap.to_dict(),
            "utility_auroc": self.utility_auroc.to_dict(),
            "n_replicates": self.n_replicates,
            "seed": self.seed,
        }


def stratified_psu_bootstrap_inference(
    y_true: Sequence[float] | np.ndarray,
    probs_unmit: Sequence[float] | np.ndarray,
    probs_mit: Sequence[float] | np.ndarray,
    design_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    n_bootstraps: int = 50,
    seed: int = 20260828,
) -> BootstrapInferenceResult:
    """Execute paired complex survey design-aware stratified-PSU bootstrap inference.

    Resamples PSUs with replacement within each VARSTR stratum and adjusts survey weights.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    p_unmit = np.asarray(probs_unmit, dtype=float).ravel()
    p_mit = np.asarray(probs_mit, dtype=float).ravel()
    w_base = np.asarray(design_df["LONGWT"].values, dtype=float)
    strata = design_df["VARSTR"].values
    psus = design_df["VARPSU"].values

    # Base point estimates
    base_auprc_unmit = weighted_auprc(y_t, p_unmit, sample_weight=w_base)
    base_auprc_mit = weighted_auprc(y_t, p_mit, sample_weight=w_base)
    base_auroc_unmit = weighted_auroc(y_t, p_unmit, sample_weight=w_base)
    base_auroc_mit = weighted_auroc(y_t, p_mit, sample_weight=w_base)

    base_fairness_unmit = primary_fairness_endpoint(y_t, p_unmit, audit_df, sample_weight=w_base)["primary_fairness_max_tpr_gap"]
    base_fairness_mit = primary_fairness_endpoint(y_t, p_mit, audit_df, sample_weight=w_base)["primary_fairness_max_tpr_gap"]

    # Pre-index stratum PSUs
    unique_strata = np.unique(strata)
    strata_psu_map: dict[Any, list[Any]] = {}
    strata_record_map: dict[Any, dict[Any, np.ndarray]] = {}

    for st in unique_strata:
        st_mask = (strata == st)
        unique_psus = list(np.unique(psus[st_mask]))
        strata_psu_map[st] = unique_psus
        strata_record_map[st] = {}
        for ps in unique_psus:
            strata_record_map[st][ps] = np.where(st_mask & (psus == ps))[0]

    rng = np.random.RandomState(seed)

    diffs_auprc: list[float] = []
    diffs_fairness: list[float] = []
    diffs_auroc: list[float] = []

    for _ in range(n_bootstraps):
        rep_w = np.zeros_like(w_base)

        for st in unique_strata:
            psu_list = strata_psu_map[st]
            m_h = len(psu_list)

            if m_h <= 1:
                # Single PSU in stratum: copy weights unchanged
                for ps in psu_list:
                    idxs = strata_record_map[st][ps]
                    rep_w[idxs] = w_base[idxs]
            else:
                # Resample (m_h - 1) PSUs with replacement
                sample_count = m_h - 1
                drawn_psus = rng.choice(psu_list, size=sample_count, replace=True)
                counts: dict[Any, int] = {}
                for dp in drawn_psus:
                    counts[dp] = counts.get(dp, 0) + 1

                scale_factor = float(m_h) / float(m_h - 1)
                for ps in psu_list:
                    k_hi = counts.get(ps, 0)
                    idxs = strata_record_map[st][ps]
                    rep_w[idxs] = w_base[idxs] * scale_factor * k_hi

        if rep_w.sum() <= 0:
            continue

        # Evaluate replicate metrics
        rep_auprc_unmit = weighted_auprc(y_t, p_unmit, sample_weight=rep_w)
        rep_auprc_mit = weighted_auprc(y_t, p_mit, sample_weight=rep_w)
        diffs_auprc.append(rep_auprc_mit - rep_auprc_unmit)

        rep_auroc_unmit = weighted_auroc(y_t, p_unmit, sample_weight=rep_w)
        rep_auroc_mit = weighted_auroc(y_t, p_mit, sample_weight=rep_w)
        diffs_auroc.append(rep_auroc_mit - rep_auroc_unmit)

        rep_fair_unmit = primary_fairness_endpoint(y_t, p_unmit, audit_df, sample_weight=rep_w)["primary_fairness_max_tpr_gap"]
        rep_fair_mit = primary_fairness_endpoint(y_t, p_mit, audit_df, sample_weight=rep_w)["primary_fairness_max_tpr_gap"]
        if rep_fair_unmit is not None and rep_fair_mit is not None:
            diffs_fairness.append(rep_fair_mit - rep_fair_unmit)

    def summarize_inference(pt_unmit: float | None, pt_mit: float | None, diff_list: list[float]) -> EndpointInference:
        if pt_unmit is None or pt_mit is None:
            return EndpointInference(
                point_estimate_unmitigated=None,
                point_estimate_mitigated=None,
                paired_difference=None,
                std_error=None,
                ci_95_lower=None,
                ci_95_upper=None,
                bootstrap_replicates_count=len(diff_list),
                status="NOT_ESTIMABLE_SUPPRESSED",
            )

        pt_diff = float(pt_mit - pt_unmit)
        if len(diff_list) >= 5:
            se = float(np.std(diff_list, ddof=1))
            ci_low = float(np.percentile(diff_list, 2.5))
            ci_upp = float(np.percentile(diff_list, 97.5))
        else:
            se = 0.0
            ci_low = pt_diff
            ci_upp = pt_diff

        return EndpointInference(
            point_estimate_unmitigated=float(pt_unmit),
            point_estimate_mitigated=float(pt_mit),
            paired_difference=float(pt_diff),
            std_error=se,
            ci_95_lower=ci_low,
            ci_95_upper=ci_upp,
            bootstrap_replicates_count=len(diff_list),
            status="ESTIMABLE",
        )

    return BootstrapInferenceResult(
        utility_auprc=summarize_inference(base_auprc_unmit, base_auprc_mit, diffs_auprc),
        fairness_max_tpr_gap=summarize_inference(base_fairness_unmit, base_fairness_mit, diffs_fairness),
        utility_auroc=summarize_inference(base_auroc_unmit, base_auroc_mit, diffs_auroc),
        n_replicates=len(diffs_auprc),
        seed=seed,
    )

