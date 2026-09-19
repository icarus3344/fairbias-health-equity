# FairBias A3/A4 analysis-readiness review

Gate: A3/A4 bounded source review  
Status: BLOCKED for final frozen paper export pending the findings below  
Scope: `experiment_evaluation.py`, `survey_linearization.py`, `survey_batch.py`, `paper_reporting.py`, and registered statistical rules. No NHIS data or model files were opened.

## Blocking findings

1. **Withdrawn: the production paper-manifest hash is present.** A previous read stopped before the complete output block. `experiment_evaluation.py:291-294` writes `evaluation_manifest_sha256`, and `paper_reporting.py:167-174` verifies it. The new production-to-report synthetic probe passes this real handoff; the earlier finding was a false positive.

2. **The current evaluation/reporting path is single-run, while the 2026-09-17 plan explicitly requires cross-run identity resolution.** `experiment_evaluation.py:146-165` binds one `run`, one `registration.json`, and relative `selected_artifacts`; `paper_reporting.py:182-204` likewise expects one registration and an exact `(candidate_id, seed)` set under one job-result list. This cannot represent the separately versioned FRAPPÉ correction, EG repair, BM MDS retry, or Joint identities without first creating an audited merged index. It must not be solved by concatenating job files with duplicate candidate/seed keys.

3. **Frozen analysis identity is incomplete at the public paper-export boundary.** `freeze_study` records analysis source hashes in its output (`experiment_evaluation.py:112-122`), and evaluation checks them (`:142-149`), but `paper_reporting.py:159-164` checks only the adjacent evaluation-manifest and selection/summary linkage; it does not require a study-freeze hash or verify the analysis-file hash set. A report can therefore be generated from a syntactically linked aggregate package without independently binding the full frozen analysis implementation.

4. **Withdrawn: untouched-base risk is connected in the complete aggregation block.** `experiment_evaluation.py:238-240` converts `base_risk` to `untouched_base_*`, which `paper_reporting.py` exports. The new production-to-report probe verifies the generated field; the earlier finding was caused by reading an incomplete source fragment.

## Statistical consistency findings

- **BA and paired BA:** `experiment_evaluation.py:247-252` averages frozen seed `q` vectors only for BA, which is valid because fixed-design BA is linear in q. Paired BA at `:274-282` uses the shared mean-policy q vectors and `linearized_survey_inference`; this matches the registered Taylor contrast. It must not be reused for EO.
- **EO/DP seed handling:** `experiment_evaluation.py:242-245` projects each model’s jointly covered gap band and averages endpoints. `survey_linearization.py:164-168` uses a rate-coordinate Bonferroni family multiplied by the number of supplied models, and `:195-207` projects the nonsmooth ranges. This follows the registered “average projected seed bands” rule; averaging q before EO would be a different estimand and is not done here.
- **p/q identity:** T fairness metrics consume `bundle.q_decision` (`experiment_evaluation.py:189-194`); risk metrics consume the bundle separately. `paper_reporting.py:78-91` labels risk columns with `_p`. The boundary still relies on every upstream `compute_risk_metrics` record preserving p-unavailable states; the exporter does not independently inspect provenance.
- **Multiplicity:** The registered family remains 20. `experiment_evaluation.py:210-211` uses `.05/20` for the primary-family EO projection and `:280-282` applies `.05/20` to paired BA. The linearization function additionally applies its internal rate-coordinate Bonferroni factor for EO, as required for method×rate joint coverage. BA remains a single-metric t interval. The resulting labels must distinguish the primary-family BA interval from the EO rate-coordinate projection; no max-gap normal p-value is produced.
- **Domain design:** Full annual arrays and `domain_mask` are passed to both linearization and bootstrap (`experiment_evaluation.py:208-214`). `survey_linearization.py:116-130` retains all annual strata/PSUs, and `survey_batch.py:29-33` constructs denominators and numerators from the domain. Singleton strata return `DESIGN_NOT_ESTIMABLE` in linearization. The final package still needs an aggregate domain-support table so unsupported groups/outcomes are not hidden by a valid row count.
- **Bootstrap role:** `survey_batch.py:9-18` describes the B=2000 output as shared paired replicate arrays; plan and evaluation label it descriptive. It must not be promoted to the formal BA/EO interval or used to change selection.

## Synthetic evidence and recommended probes

Existing bounded probes passed: `tests/benchmark/test_survey_linearization.py` (including multi-method critical-value behavior), `tests/benchmark/test_frozen_evaluation_contract.py` (`3 passed`), the new `tests/benchmark/test_analysis_readiness_probes.py` (`1 passed`), and the M=200 production Taylor coverage script. The latter used df=3 and no FPC, so it is not evidence for NHIS df≈500.

Before final freeze, add synthetic checks for:

- production `summary_T.json` → paper exporter manifest binding (covered by the new probe);
- duplicate candidate/seed across two run identities is rejected;
- changed analysis source hash is rejected by paper export, not only by evaluation;
- untouched-base p is emitted under the exporter’s expected field (covered by the new probe);
- one unavailable p/q method cannot populate AP/Brier from q;
- complete annual design with a zero-contribution PSU retains that PSU in the domain report.

## A4 boundary

The paper narrative may state the registered years, PSU-level F/C assignment, 2023 S selection, retrospective frozen 2024 T evaluation, four-arm relationship, method output semantics, survey weighting, Taylor/EO projection rules, B=2000 descriptive sensitivity, and failure/version disclosure. It must leave real cohort counts, event counts, completed-condition totals, and performance values as placeholders until an audited aggregate export exists.

STOP — waiting for Codex review.
