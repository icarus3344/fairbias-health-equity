# Manuscript-readiness review acceptance — 2026-09-18

Decision: **ACCEPT the bounded descriptive, transformation, literature and reporting package. Manuscript submission readiness remains conditional.** The user's chosen direction is a medical-informatics application paper. No server, training, model reselection, prediction, new performance inference, Git staging/commit, publication or third-party communication was performed by this gate.

## Scope and deliverables

Gate: `docs/plans/FAIRBIAS_MANUSCRIPT_READINESS_EXECUTION_20260918.md`. Main entry: `docs/paper/manuscript_readiness_20260918/READINESS_REPORT.md`. New integrated draft: `MANUSCRIPT_METHODS_RESULTS_v2.md`. The accepted missingness data are under `missingness/v2/`; the accepted new figure/caption are under `missingness/figures_v3/`. Earlier descriptive/figure versions remain historical and are excluded from the new delivery bundle.

The package supplies 336 variable/domain rows (312 included;24 Arm004 N/A),16 eligibility rows,12 omitted-code checks,1,680 final model-variable rows,42 mode-variable summaries and1,425 committed transformation events. Literature review contains ten method identities, eleven software-source entries and a bibliography with only verified DOI/URL fields. Submission preparation maps all52 TRIPOD+AI numbered items and22 STROBE main items; the structured abstract is300 whitespace-delimited words, below the verified350-word BMC template limit.

## Independent verification

`scripts/verify_nhis_manuscript_readiness_20260918.py` executed successfully and wrote `artifacts/nhis/manuscript_readiness_20260918/supervisor_verification.json`: **19,821 assertions passed**. These are integrity/arithmetic assertions, not19,821 independent statistical tests.

- All99 archived training sources and95 current evaluation sources retain their frozen hashes. The accepted admission/study identifiers match.
- All80 result and80 policy files were rehashed. Policy binaries were not deserialized.
- An independently written replay reconstructed every final transformation, with865 AE parent/candidate state pairs checked; BM560 andAE865 committed events reconcile.
- Independent code-domain grouping verifies final counts:820 merged,80 powered,60 dropped,600 unchanged;120 unavailable by design.80 models mix missing/substantive values,60 mix structural NIU/substantive values,70 have noncontiguous ordinal bins.
- Exact public raw source hashes and prepared-membership hashes were checked before bounded descriptive reads. All312 included rows were recomputed from raw columns with distinct routing/nonresponse/null/unknown classes, including weighted sums/fractions.24 N/A rows and16 existing Table1 domains' counts, events and weight sums reconcile.
- All12 omitted-code observations across2022–2024 are zero. This does not validate unsupported future codes, but rules out demonstrated impact from those omissions in this batch.
- Five meaningful synthetic missingness tests pass independently: `5 passed in1.78s`, covering routing, null/unknown, unequal weights, excluded predictors and annual/domain denominators.
- Reviewed source-text hashes and worker output hashes match, except for the explicitly verified append-only gate-host amendment described below. Primary-codebook employment universe text was independently checked. Journal/reporting references are tracked in the submission-source ledger.
- The old47-member paper bundle hash remains `460b0b78469b632c8ddc5e791c2d8c07215fd32bf477881388ddae25a93c669d`. No accepted performance table or old draft was overwritten.

Supervisor reviewed the plotted aggregate paths and rendered the missingness figure. Its new caption distinguishes44.31% unweighted from38.39% weighted combined non-observed/routed states inT, and correctly sums the **three non-observed states**, not all four displayed states. The earlier caption issue was a reporting defect, not a changed data value. Only figures_v3 is current. The image remains an analysis figure; final journal sizing/layout awaits a chosen target.

## Repairs and provenance exceptions retained

1. Initial missingness export conflated some unresolved employment source-null states with item nonresponse. The new v2 descriptive exporter preserves them. Initial outputs remain untouched; no frozen modeling preprocessing changed.
2. Figure-caption v1 omitted weighted/unweighted qualification and v2 incorrectly described the number of states being combined. Supervisor produced v3 with the correct wording and refreshed provenance; old figure versions are preserved and excluded from delivery.
3. The supervisor verifier's own development runs exposed a CSV-column-name mismatch, different manifest container shapes, and a plan-hash difference from a newly authorized source-host appendix. The verifier was corrected before acceptance; no failed verification was labelled PASS or used to update scientific conclusions.
4. Literature-agent baseline equality initially failed because `.gitignore` has pre-existing committed drift at `b595e59`. The14 inherited files still match the baseline; working-tree `.gitignore` is clean. Supervisor authorized additive documentation closure only and preserved the false full-baseline flag. No baseline repair is implied.
5. After the literature review's hash freeze, supervisor added permission for the author institutional repository to the gate. The exact pre-append prefix hashes to the worker's recorded value. The institutional article page opened, but its PDF returned403; the supplement remains unverified. No access challenge was bypassed.

## Scientific and publication boundaries

- Domain review is technical plausibility screening, not clinician approval. Traceability does not imply preservation of clinical meaning, causal explanation or patient benefit.
- Missingness is descriptive, not MCAR/MAR/MNAR identification. Complete released income categories include official single imputation; multiple-imputation uncertainty was not propagated.
- All80 completed local BM+AE/Joint policies have accepted formal evaluation. This does not establish exact reproduction of every original-paper algorithm, the fullJoint tuning grid, uniform superiority, or a new untouched external test.
- Comparator variants and output types are explicitly differentiated; FairGBM/fairret/TabM are not part of this completion comparison. Unavailable historical comparisons, computation differences and known-T status remain disclosed.
- FairBias supplementary algorithm correspondence and some complete primary-text access remain unresolved. Verified bibliographic identity is not claimed as full-method fidelity verification.
- Ethics/consent requirements, human authorship, actual funding/conflicts, patient involvement and final approval require truthful human/institutional confirmation. No facts have been invented.
- Full risk calibration, risk-metric paired intervals, subgroup absolute TPR/FPR displays and all-variable distributions are retained as concrete potential supplementation; they were not silently added to the frozen inferential family.
- A publishable source release must bind the actual running source, dependencies and applicable licenses; the oldGit HEAD alone does not identify all current code. This gate did not publish or authorize a license for inherited code.

## Handoff

The user can review the new integrated draft, methods bibliography, full report and author-facts table locally. No server is required for the completed work or manuscript editing. Automation stays paused. Next work should prioritize human application/ethics facts, selected-journal formatting and independently specified frozen-prediction reporting gaps; any new model variant requires separate development/validation boundaries.

New bundle membership and SHA verification are recorded separately in `artifacts/nhis/manuscript_readiness_20260918/bundle_verification.json`; this report does not create a self-referential archive hash.
