# Decision 0007: Multi-Panel Development Statistical Design

Date: 2026-08-31
Gate: Gate 14A — Multi-Panel Development Statistical Design
Status: `DESIGN_APPROVED_FOR_SYNTHETIC_IMPLEMENTATION`

## Decision

The fixed four-panel development design is approved for pure-synthetic
implementation only. The candidate sequence is HC-217 / Panel 23,
HC-225 / Panel 24, HC-234 / Panel 25, and HC-244 / Panel 26. This approval is
conditional on Gate 15's later, separately authorized, panel-by-panel source
and comparability verification. It is not permission to download, read, or
summarize any new MEPS outcome data.

HC-252 / Panel 27 remains a completely locked temporal holdout. No Gate 14A or
Gate 14B action may inspect its outcomes, protected-group distributions,
predictor distributions, or performance results.

## Rationale and frozen decisions

1. **Panel inclusion is fixed before outcomes.** The four candidates are chosen
   from existing official AHRQ metadata records and a predefined two-year
   comparability contract. Event counts cannot select, remove, or reorder a
   panel. A failed source, semantic, weight, mode, or design check stops the
   fixed design rather than causing outcome-driven substitution.
2. **The common estimand is panel-specific.** Each panel represents its own
   documented civilian noninstitutionalized two-year longitudinal target after
   the frozen age, both-years, five-round, positive-weight, and continuous
   baseline-coverage restrictions. The primary outcome is any documented
   uninsured month in Year 2. Baseline predictors, protected audit variables,
   and survey-design fields retain their existing roles.
3. **`LONGWT` is not pooled by default.** Primary estimates use the original
   `LONGWT` within each panel and are reported separately. Raw row concatenation
   and simple division by the number of panels are prohibited because they do
   not establish a common target population.
4. **Development pooling is loss-level only.** If Gate 15 accepts all four
   panels, a common development model may use within-panel normalized `LONGWT`
   and an explicit mean of panel losses. This balances panel weight scales for
   model development; it is not a pooled population estimator. `PANEL` is used
   to namespace identities and stratify validation, never as a predictor.
5. **Repeated years and Panel 25 are explicit risks.** Overlapping calendar
   years remain labeled panel domains, without cross-panel linkage or
   deduplication. HC-234 receives a mandatory review of pandemic-era collection
   mode, response, instrument, and Round 1 comparability. Unresolved differences
   fail closed.
6. **Validation is frozen before implementation.** The selected structure is
   panel-aware pooling with panel-internal namespaced household splits and
   panel-stratified reporting. Separate panel models, per-panel sensitivities,
   and leave-one-panel-out checks are prespecified secondary structures. No
   structure may use observed event counts to revise the candidate set.
7. **Power is multidimensional.** Total events, per-panel events, and primary
   subgroup estimability must all be reported. The 200-positive-event minimum
   is necessary but not sufficient. Existing subgroup suppression rules remain
   in force, with no ad hoc post-outcome merging.

## Gate boundaries

Gate 14A changes no existing estimand, threshold, data authorization, or
Panel 27 lock. It adds a machine-readable design contract and a pure
configuration test. Gate 14B, if separately accepted, may implement only the
synthetic contract and fail-closed cases. A new authorization checkpoint is
required before Gate 15 may download or inspect early development-panel files.

## Evidence boundary

The design relies on existing repository records of official AHRQ metadata; no
new network lookup, archive download, microdata read, outcome count, or result
was performed in Gate 14A. The early-panel mappings and `LONGWT` target
interpretations are intentionally not promoted to verified facts until Gate 15.

## Standard Gate 14A report

Gate: Gate 14A — Multi-Panel Development Statistical Design
Status: `DESIGN_APPROVED_FOR_SYNTHETIC_IMPLEMENTATION`
Files changed:
- `configs/development_panels.json`
- `docs/decisions/0007-multi-panel-development-design.md`
- `docs/research/MULTI_PANEL_DEVELOPMENT_PLAN.md`
- `tests/test_development_panels_config.py`
Commands executed:
- Read-only inspection of the canonical execution protocol and existing local metadata/configuration records.
- No external source, MEPS microdata, outcome, result, pipeline, training, bootstrap, or Panel 27 operation.
Permissions requested: None for Gate 14A; a separate authorization checkpoint is required before Gate 15.
Tests executed: Pure configuration consistency tests only.
Exact test results: `Ran 9 tests in 0.001s` / `OK`; `json.tool` validation passed; `git diff --check` passed for the four Gate 14A files.
Input hashes: `2a685dea3736b0a60083b401e127a63084d65728`; no MEPS microdata.
Output hashes: `configs/development_panels.json` — `8860a0a71c66b7184dc873a783b1ec5939f8c2902053acf7b8d9fa1c2222382d`; `tests/test_development_panels_config.py` — `8e14aa9901450b2572d45ec5bd03da08847046ced4c28cb83b5415be318c9430`; the two design documents are self-referential report artifacts and their hashes are intentionally not embedded.
Row counts: Not applicable.
Assumptions: Existing official-source records establish metadata candidates, while early-panel semantic and weight comparability remain Gate 15 questions.
Unresolved issues: Gate 15 source/schema and comparability verification; separate new data authorization; Panel 27 remains locked.
Git diff summary: Additive files only; no frozen root file or `.gitignore` modification.
Proposed next step: Independent Codex review; if accepted, implement Gate 14B using only synthetic data.
STOP — waiting for Codex review.
