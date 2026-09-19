Gate:
Completed80 evaluation — independent read-only four-arm artifact review, 2026-09-18.

Status:
The bounded artifact and arithmetic review found no mismatch in the four completed arm evaluations. A total of 17,084 independent assertions passed. This is worker evidence for supervisor review, not self-approval of the experiment, merge, or scientific claims.

Files changed:
Only this new report for the real-artifact review. No frozen source, research parameter, original artifact, model, study, selection, or evaluation output was changed. The earlier synthetic test and design report were already finalized separately.

Commands executed:
Read the explicitly authorized `artifacts/nhis/completion_evaluation_20260918/control/study_v1.json`, its admission/release metadata, and four `evaluation_v1/arm_00*/evaluation_manifest.json`, `summary_T.json`, individual metric JSON, and saved bootstrap metric arrays. Rehashed current analysis sources, frozen training sources, bound completed80 policy/result/prepared files, 269 parent prediction files, and every artifact listed in each new arm manifest. Hashing did not deserialize models or prepared containers.

Two temporary Python commands ran with `.venv311/bin/python`, `PYTHONHASHSEED=0`, and one BLAS/OpenMP thread. The first independently checked metadata, identities, point/seed/bootstrap summaries, pair mappings, intervals, and hashes. The second reconstructed every nominal and family-adjusted seed DP/EO projection from saved rate point estimates and standard errors. These commands did not call model prediction, fit, selection, or cohort loading. No raw microdata or individual prediction values were printed.

Permissions requested:
The parent explicitly authorized this read-only review of actual T summaries and necessary prediction artifacts after all four arm evaluations completed. No additional permissions requested; no remote access or deployment.

Tests executed:
- Hash-bound study, selection, admission, release, source, new-model/result/input, parent-prediction, and arm-artifact checks.
- Exact 349-model union and disjoint arm coverage; exact 80 new plus 269 reused parent models; original five-seed identities for each of 16 new cells.
- Exact 96 selection slots and 168 contrast slots, with model/method/selection IDs and `reference_minus_comparison` direction bound to the study.
- BA/DP/EO means and seed standard deviations recomputed from individual metrics; row and paired bootstrap standard errors/intervals recomputed from common stored B=2000 replicate arrays.
- BA Taylor point agreement, fixed paired SE across nominal/family/union intervals, and t critical values for global endpoint counts 20/316/336.
- DP/EO bounds independently reconstructed for every seed, then averaged; pair EO intervals recomputed by interval subtraction.
- Non-model and S-infeasible slots retained; fixed-BM controls and all registered pairs retained.
- Existing semantic provenance validator applied to each observed manifest versus the frozen study; full annual row-order identity and per-arm domain identity checked.

Exact test results:
Both review commands exited 0. The first passed 7,910 assertions; maximum absolute difference between saved and recomputed seed means was 0.0. The second passed 9,174 assertions; maximum absolute numerical difference across reconstructed projections/BA identities was `6.661338147750939e-16`.

Seven actual selection rows exhibit a nonzero difference between mean seed EO and EO of the mean policy; the saved summaries use mean seed EO. This is an estimand validation, not a performance claim. All nominal and primary/secondary/union projections matched seedwise reconstruction.

| Arm | Models (new + parent) | Selection rows | Contrast slots | Evaluation rows | Evaluated pairs | Domain / annual rows | df |
|---|---:|---:|---:|---|---|---:|---:|
| arm_001 | 96 (20 + 76) | 24 | 42 | 24 VALID | 42 VALID | 32,350 / 32,629 | 610 |
| arm_002 | 66 (20 + 46) | 24 | 42 | 18 VALID; 6 NO_VALID_MODEL | 30 VALID; 12 NO_VALID_MODEL | 32,355 / 32,629 | 610 |
| arm_003 | 96 (20 + 76) | 24 | 42 | 24 VALID | 42 VALID | 32,354 / 32,629 | 610 |
| arm_004 | 91 (20 + 71) | 24 | 42 | 24 VALID | 42 VALID | 32,354 / 32,629 | 610 |

Arm002's six missing-model slots are exactly four NOT_SUPPORTED slots and two NO_VALID_FIXED_ABLATION slots. Its 12 unavailable pairs remain in the registry. S NO_FEASIBLE_CONFIGURATION rows remain present: arm002 12, arm003 7, arm004 6. S feasibility is not relabeled as T validity or used to remove a registered comparison. Each arm records 20 primary / 316 secondary / 336 union endpoint slots, rather than shrinking the family to its own available pairs.

Input hashes:
```text
study_v1.json     3c4572ec327d8d97a2f108fe3c82a8619370a78981c8fe1abdb496677b2933fb
selection        46e58b19a02ea5144b9c8fe11459c38b25db2fe9e503129125114a7e7167e69d
admission        2a83612b23048a54ce09196ec553b9881ee87bf5b5930f36fc53bc0e52d1a628
t_release_v1     64d34e601996ddcda1bf687b33fd9db2f6e259f94a29ae0ada41b08bef9964ff
```
All 95 declared analysis-source entries and 99 declared training-source entries matched their corresponding files. All four arms share annual record-order SHA `48157948cf960585feccd0c90455d77acbdbc41c14c0674bf36ad36e2df22947`. Domain hashes differ as expected; arms003/004 share `3b20eb4aaa2a24e019098f7131ee2fc6c904aa67c7fe4ae3647ece7d221415d0`.

Output hashes:
The reviewed immutable outputs are:
```text
arm_001 manifest 469a52dcf493ad5548629b31cb431eb8bbb447716fd0d9b52bd33af894303f72
arm_001 summary  3869566e1379553763f37299c5faeca4ef51981dbf1ca5a20c1ef67c078e040d
arm_002 manifest 576138591c3c9fa66bfefa00332ec12eef2ebb1aa35527aba643d201832aed07
arm_002 summary  bc4cde89c667e650e5e90cf678aef769795ec6ee364499ea7d43c27d1357b81f
arm_003 manifest a88e83ff1e2f73e4fa6e3e9cd4dcc339774b99bdea5f8d5efccdc80375b9ad29
arm_003 summary  97f71ed3bc966ef4909d61a048b170a56b68e406e0503872fba8a336e3b02a27
arm_004 manifest a3b75f69f830db4a40330f6e649102638edbf5b77cccfb35baa51d29d9e516a1
arm_004 summary  d57543a95e6e8f89b5b4e486875c8fa9a5899a96ac81782c3799b60c4b316930
```
This report's SHA is supplied separately to the parent. Every manifest-listed file was rehashed successfully; summaries independently bind their manifest hashes.

Row counts:
Raw microdata rows opened or emitted by this reviewer: 0. Aggregate T counts are in the table. Reviewed saved metric arrays contain 349 model identities × 2,000 replicates × three metrics; these are aggregate survey replicate outputs, not individual records. No model inference was repeated.

Assumptions:
The prior source and synthetic review establishes the common annual PSU bootstrap implementation. This review verifies saved replicate aggregation and survey-rate projection arithmetic, but does not recompute PSU covariance from raw T respondents. Individual metric point estimates and rate SEs are inputs to this arithmetic check. Model/prepared byte hashes were verified without deserialization. Parent selection bytes were rehashed without inspecting S metric values. Original model artifacts for the 269 parent predictions were not reloaded or independently rehashed; the study/admission identities and all 269 referenced prediction hashes were checked. All outputs retain the separate `fairbias_adaptive_completion_v1` branch and `known_T=True`; no untouched-holdout claim is supported.

Unresolved issues:
The frozen `merge_completion` checks exact dictionary equality of observed versus expected T provenance. Here the observed provenance adds `raw_sources.2024.rows=32629`, while the frozen study intentionally declares only the source path/hash. Removing only that additional observed row-count field yields the frozen declaration, and the pre-existing `_validate_observed_provenance` accepts every arm. This explains `SHARD_IDENTITY_CHANGED` in the first merge attempt; it is an export-contract mismatch, not evidence of a failed prediction or invalid arm artifact. The parent is preparing a separately identified additive exporter using the existing semantic validator. This report does not accept that exporter or its future merged output. The failed merge log and all original arm files should remain immutable.

Git diff summary:
One new report for this task. No production, existing test, or artifact modifications; no staging or commit.

Proposed next step:
Supervisor review the four-arm evidence, independently verify the additive exporter's restricted provenance normalization and exact output coverage, then decide archive/report acceptance. No S/T rerun or model inference is needed to resolve the observed metadata-only merge mismatch.

STOP — waiting for Codex review.
