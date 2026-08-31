# Decision 0006: MEPS Pre-Experiment Repair Gate

**Date:** 2026-08-30  
**Status:** IMPLEMENTED_PENDING_INDEPENDENT_REVIEW — no commit authorization  
**Branch:** `research/meps-hc252-longitudinal`  
**Starting HEAD:** `e5e11f5e2621253bebe8309038a57c6c13cdc9aa`

## Gate objective

Repair the configuration, methodological naming, power-gating, and calibration-evidence boundaries that currently prevent independent review of the uncommitted MEPS Gates 6–13 work. This is a pre-experiment governance and implementation repair only.

## Governance lineage

This gate is executed under the continuous-batch authorization recorded in
[`Decision 0004`](0004-continuous-gemini-batch-implementation.md), which is part
of the final Gates 6–13 evidence closure. Decision 0006 narrows that batch
authorization to the repair scope and explicit prohibitions below; it does not
replace the canonical protocol or authorize a commit, experiment, download,
bootstrap, or Panel 27 access.

## Authorized work

1. Reconcile `configs/study.json`, `docs/research/RESEARCH_PROTOCOL.md`, and `docs/research/STATISTICAL_ANALYSIS_PLAN.md` with the verified Gate 7 variable dictionary and current source constants.
2. Replace overclaiming FairBias/Tang method-arm labels with the implementation's honest name: an exploratory survey-weighted group-aware centering heuristic that requires the protected group at inference.
3. Remove literal event-count logic and status text from the pipeline. Power status must be generated from runtime counts and a named threshold.
4. Freeze a development-panel policy before any additional outcome access. Earlier panel identities and years must be verified from AHRQ primary metadata; no outcome counts may be inspected in this gate and no naive cross-panel pooling is authorized.
5. State exactly that calibration metrics computed on the same partition used to fit the Platt calibrator are apparent calibration-fit diagnostics, not out-of-sample validation.
6. Add non-data unit/regression tests and a Section 9 report for these repairs.

## Explicitly prohibited

- No model training, pipeline execution, empirical metric generation, or new bootstrap run.
- No access to Panel 27 outcome values, protected-group distributions, predictor distributions, prevalence, or metrics.
- No new MEPS microdata download or extraction.
- No change to the primary estimand, target definition, suppression thresholds, or Panel 27 unlock rule.
- No commit, push, staging with broad pathspecs, or modification of the 14 frozen root files and `.gitignore`.
- No claim that the current heuristic implements Tang et al. or FairBias.

## Network allowlist

Read-only metadata lookup is permitted only from official AHRQ MEPS HTTPS pages under `meps.ahrq.gov`, solely to verify earlier longitudinal PUF identifiers, panel numbers, and covered years. Data archives and microdata must not be downloaded.

## Acceptance criteria

- All exact study-variable mappings are non-null and mechanically consistent with `configs/cohort_and_variables.json` and `src/meps_fairness/data/cohort.py`.
- The only implemented mitigated arm is honestly named and its inference-time protected-attribute requirement is explicit.
- No source or active test hard-codes the observed value `136` as pipeline logic or expected runtime status.
- Power-gate helpers are data-independent, tested with multiple synthetic event counts, and keep Panel 27 locked even when the numeric power threshold is met pending separate Codex authorization.
- Calibration evidence labels distinguish fit partition, diagnostic partition, and evidence nature.
- Earlier development panels are frozen from official metadata without accessing their outcomes; pooling/weight normalization remains a separately gated statistical decision.
- Targeted non-data tests pass; frozen baseline files remain unchanged; the final report uses the Section 9 headings and ends with `STOP — waiting for Codex review.`

## Official metadata used for the frozen candidate sequence

- AHRQ HC-217 details: Panel 23 two-year longitudinal PUF, 2018–2019 — `https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-217`
- AHRQ HC-225 listing: Panel 24 two-year longitudinal PUF, 2019–2020 — `https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_results.jsp?SearchTitle=Longitudinal&buttonYearandDataType=Search&cboDataTypeY=1%2CHousehold+Full+Year+File&cboDataYear=All&cboPufNumber=All`
- AHRQ HC-234 details/documentation: Panel 25 two-year longitudinal PUF, 2020–2021 — `https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-234`
- AHRQ HC-244 listing: Panel 26 two-year longitudinal PUF, 2021–2022 — same official longitudinal-file search listing.
- AHRQ identifies HC-226 as Panel 23's **three-year** 2018–2020 file; it is intentionally excluded from this common two-year candidate sequence.
- AHRQ's HC-234 documentation warns that pandemic-era phone collection and low Panel 25 Round 1 response raise data-quality and pooling/comparison concerns. This gate records that warning and does not authorize pooling.
