# Multi-Panel Development Plan (Gate 14A)

## Decision

`DESIGN_APPROVED_FOR_SYNTHETIC_IMPLEMENTATION`

This document freezes the statistical design for HC-217 (Panel 23), HC-225
(Panel 24), HC-234 (Panel 25), and HC-244 (Panel 26) before any new development
panel outcome access. It authorizes only a later pure-synthetic implementation
gate. It does not authorize a download, a microdata read, a real pipeline run,
model training, bootstrap inference, or any Panel 27 operation.

The machine-readable contract is
[`configs/development_panels.json`](../../configs/development_panels.json).

## 1. Fixed panel set and inclusion rule

The candidate sequence is fixed in advance as HC-217, HC-225, HC-234, and
HC-244. The sequence is based on the existing official-metadata records in the
source registry, `configs/study.json`, `configs/cohort_and_variables.json`, and
Decision 0006. The recorded facts establish the candidate PUF identity, panel
number, and consecutive two-calendar-year span only. They do not establish that
the early panels have interchangeable field names, codes, universes, weights,
or collection modes.

A candidate is included only if official AHRQ documentation and the following
predefined conditions support it:

1. It is the specified two-year longitudinal PUF, with the documented panel
   number and consecutive calendar years.
2. Its person-level unit, two-year eligibility concept, and five-round concept
   match the canonical cohort contract.
3. The documentation supports semantic equivalence for eligibility, baseline
   coverage, follow-up coverage, the 74 baseline predictors, protected audit
   variables, `LONGWT`, `VARSTR`, and `VARPSU`.
4. The panel-specific `LONGWT` target population can be stated from the
   official documentation and related to the frozen panel-specific estimand.
5. Collection-mode, response, instrument, and calendar-year differences do not
   invalidate the prespecified descriptive comparison.
6. Cross-panel keys can be namespaced without record linkage or
   re-identification.

No panel is selected, removed, or reweighted after seeing event counts. If one
of these conditions fails, the fixed multi-panel design stops and the failure is
escalated; it is not repaired by choosing a more convenient panel.

## 2. Common estimand and semantic contract

The unit is one person in one panel-specific two-year longitudinal PUF. For
each panel, the target is the civilian noninstitutionalized population
represented by that panel's official longitudinal target population after the
frozen eligibility restrictions. It is not an unrestricted claim about all US
adults and it is not a claim that the four panels form one common population.

The same two-year estimand is used for every candidate:

- baseline: the first calendar year of the panel;
- follow-up: the immediately following calendar year;
- age: `18 <= AGEY1X <= 64` at baseline Year 1 end;
- both-years eligibility: `YEARIND == 1`;
- five-round participation: `ALL5RDS == 1`;
- positive longitudinal weight: `LONGWT > 0`;
- baseline coverage: all 12 `INS<month>Y1X` fields carry the documented
  insured code;
- primary outcome: at least one of the 12 `INS<month>Y2X` fields carries the
  documented uninsured code;
- invalid or unknown outcome codes: fail closed.

The exact monthly fields, the 74 baseline predictors (19 continuous and 55
categorical), protected dimensions, and survey-design variables are referenced
from the Gate 7 canonical contract rather than re-invented here. Matching a
column name is not enough. Gate 15 must verify the label, meaning, timing,
universe, valid/missing codes, type, and scale for every panel. The predictor
matrix remains baseline-only and excludes identifiers, design variables,
protected audit variables, and all follow-up information.

The protected dimensions remain audit variables, not primary predictors:
primary race/ethnicity and sex; secondary poverty category, age band, and the
Round-1 limitation composite. Existing suppression rules remain unchanged.

## 3. `LONGWT` and cross-panel aggregation

`LONGWT` is treated as a panel-specific official two-year longitudinal person
weight. It is not treated as a frequency count, annual cross-sectional weight,
or common scale across panels. Gate 15 must confirm the exact target population
for each PUF from its documentation.

The primary estimand is therefore panel-stratified:

- calculate each panel's point estimates with its original `LONGWT`;
- calculate design-aware uncertainty within that panel using its namespaced
  strata and PSU structure;
- report panel-specific utility, calibration, fairness, and suppression status;
- do not publish a population estimate from raw row concatenation.

Raw concatenation is prohibited because row binding does not reconcile
panel-specific target populations or repeated calendar years. Dividing every
`LONGWT` by the number of panels is also prohibited; it has no frozen population
interpretation.

For development model fitting only, a panel-aware objective is frozen so that a
panel's weight scale cannot dominate the optimization. Within each panel and
each loss partition, define

\[
v_{pi} = \frac{LONGWT_{pi}}{\sum_{j\in p} LONGWT_{pj}}.
\]

The development loss is the explicit mean of the panel losses:

\[
L = \frac{1}{K}\sum_{p=1}^{K}
    \frac{\sum_{i\in p} v_{pi}\,\ell_i}{\sum_{i\in p}v_{pi}}.
\]

This is a panel-balanced model-selection criterion, not a pooled population
estimator. The implementation must compute panel losses separately and then
aggregate them. It must not substitute raw `LONGWT` on concatenated rows or
`LONGWT / K`.

## 4. Repeated calendar years and Panel 25

The candidate sequence necessarily contains overlapping calendar years: for
example, 2019 is HC-225 follow-up and HC-234 baseline. These overlaps are
historical context, not duplicate independent observations. Panels remain
separately labeled; no person-level cross-panel linkage or deduplication is
attempted.

HC-234 / Panel 25 receives a mandatory special review of pandemic-era
collection mode, instrument and response documentation, especially Round 1.
The comparison cannot be rescued by post-outcome exclusion or an ad hoc weight
adjustment. An unresolved difference stops the fixed design.

## 5. Development-validation structure

The selected structure is `panel_aware_pooling`:

1. Within each accepted panel, split by the namespaced household key
   `("PANEL", "DUID")` using the frozen 60/20/20 train/validation/calibration
   ratios.
2. Fit preprocessing and initial models on the union of panel training
   partitions using the explicit panel-balanced loss above. `PANEL` is a
   namespace and validation stratum, never a predictor.
3. Freeze model, preprocessing, mitigation, and selection choices using only
   the prespecified panel-balanced validation criterion and panel-specific
   diagnostics.
4. Refit only after those choices are frozen. Use calibration partitions only
   for calibration. Diagnostics on the same partition used to fit a Platt
   calibrator remain apparent calibration-fit evidence, not independent
   validation.
5. Report final development quantities separately for every panel. Any
   panel-balanced macro-summary is descriptive and must not be called a pooled
   population estimate.

The following alternatives are retained as prespecified checks:

- `panel_stratified_models`: a separate model per panel, used as a feasibility
  and sensitivity reference;
- `per_panel_sensitivity`: the common model evaluated separately with original
  panel weights and the frozen unweighted/outcome-definition sensitivities;
- `leave_one_panel_out`: train without one development panel and evaluate only
  on that panel, without tuning on it.

These structures do not authorize using a panel's event count to change the
candidate set. Any proposed change to threshold transfer or to the original
estimand requires a new explicit decision.

## 6. Power and estimability

Power is a vector of checks, not a single total. A later authorized data gate
must report all of the following simultaneously:

- total eligible positive events across the fixed development set;
- positive and negative events for each panel;
- for every primary audit cell, unweighted `n`, positive events, negative
  events, and Kish effective `n`;
- secondary-cell suppression under the existing rules.

The prespecified total minimum is 200 positive events. Reaching 200 is
necessary but not sufficient: it does not prove panel comparability, adequate
per-panel contribution, valid survey structure, or subgroup estimability.
Primary cells must satisfy the existing `n >= 100`, positive `>= 20`, negative
`>= 20`, and Kish effective `n >= 50` rules. Secondary cells are suppressed,
not merged after inspecting results. Insufficient power or an unestimable
required primary cell stops development and leaves Panel 27 locked.

## 7. Panel 27 lock and next gates

HC-252 / Panel 27 (2022–2023) remains a locked temporal holdout. Gate 14A
does not authorize outcome values, protected-group distributions, predictor
distributions, performance metrics, tuning, or selection on Panel 27.

Gate 14B may implement only this contract with pure synthetic fixtures,
including collision, semantic-conflict, weight-anomaly, low-power, subgroup,
and Panel-27-lock cases. It must not read or download real MEPS data. After
synthetic implementation is independently accepted, a separate authorization
check must decide whether to permit the source/schema work of Gate 15. That
authorization is not included here.

## Gate 14A report

Gate: Gate 14A — Multi-Panel Development Statistical Design
Status: `DESIGN_APPROVED_FOR_SYNTHETIC_IMPLEMENTATION` — metadata-only design; no new data or results access
Files changed:
- `configs/development_panels.json`
- `docs/decisions/0007-multi-panel-development-design.md`
- `docs/research/MULTI_PANEL_DEVELOPMENT_PLAN.md`
- `tests/test_development_panels_config.py`
Commands executed:
- Read the canonical execution protocol, existing source registry, study configuration, Gate 7 variable contract, and Decision 0006.
- Performed read-only consistency checks against existing metadata and configuration; no external source or MEPS data was accessed.
Permissions requested: None in Gate 14A. A separate data authorization checkpoint is required before Gate 15.
Tests executed: Pure JSON/configuration consistency tests only; no real-data, pipeline, training, bootstrap, or Panel 27 test.
Exact test results: `Ran 9 tests in 0.001s` / `OK`; `json.tool` validation passed; `git diff --check` passed for the four Gate 14A files.
Input hashes: Existing Gate 6–13 commit `2a685dea3736b0a60083b401e127a63084d65728`; no MEPS microdata input.
Output hashes: `configs/development_panels.json` — `8860a0a71c66b7184dc873a783b1ec5939f8c2902053acf7b8d9fa1c2222382d`; `tests/test_development_panels_config.py` — `8e14aa9901450b2572d45ec5bd03da08847046ced4c28cb83b5415be318c9430`; the two design documents are self-referential report artifacts and their hashes are intentionally not embedded.
Row counts: Not applicable; no data rows were read or generated.
Assumptions: Early-panel metadata is inherited from the repository's official-source records and remains unverified for semantic comparability until Gate 15.
Unresolved issues: Gate 15 must verify per-panel sources, schemas, semantics, LONGWT targets, design variables, and Panel 25 comparability; no outcome count is available or authorized in Gate 14A.
Git diff summary: Additive design/configuration/test files only; protected baseline files, `.gitignore`, MEPS data, and Panel 27 remain untouched.
Proposed next step: Codex independently reviews this design and, if accepted, authorizes Gate 14B pure-synthetic implementation. Do not download or read early-panel outcomes before the separate authorization checkpoint.
STOP — waiting for Codex review.
