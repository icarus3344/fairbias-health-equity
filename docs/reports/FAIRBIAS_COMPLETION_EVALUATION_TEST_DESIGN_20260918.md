Gate:
COMPLETED80 formal evaluation — independent synthetic contracts and integration, 2026-09-18.

Status:
Worker implementation and local synthetic verification complete; supervisor review required. No gate acceptance, real S/T evaluation, or empirical fairness finding is claimed by this report.

Files changed:
- `tests/benchmark/test_completion_evaluation_contracts.py` — new, synthetic-only contracts.
- `docs/reports/FAIRBIAS_COMPLETION_EVALUATION_TEST_DESIGN_20260918.md` — this new report.

No production, frozen training, historical evaluation, STATUS, or other agent-owned file was edited by this worker.

Commands executed:
Read the canonical protocol and current CPU-replica STATUS, completion evaluator/statistics, catalog evaluator/inference, data contracts, survey linearization, and reusable generated test fixtures. Inspected only source and administrative documents; generated fixture artifacts were created in pytest temporary directories. Verified the active branch is `research/nhis-fairbias`.

Final test command:
```text
PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv311/bin/python -m pytest tests/benchmark/test_completion_evaluation_contracts.py -q --disable-warnings --maxfail=1
```
Computed source/test SHA-256 values with `shasum -a 256`; inspected Git status for the two owned paths. No staging or commit.

Permissions requested:
None beyond the parent's explicit bounded implementation instruction. No network, remote access, actual NHIS/prepared/model access, fitting, deployment, or real S/T I/O occurred in this worker task.

Tests executed:
The new 95-case suite uses generated metadata and the existing `_annual` fixture. It exercises the production admission, selection, study, release, and arm-evaluation functions, with generated prediction/prepared/cohort seams for integration. All provenance verification and formal B=2000 statistical routines remain active in the successful integration test.

| Boundary | Verified synthetic witness |
|---|---|
| Exact new matrix | 80 unique jobs; two methods × four arms × two backbones × original seeds 0, 7, 19, 37, 73; 16 cells with exactly one identical original configuration per cell |
| Seed/config rejection | Missing, duplicate, extra, wrong, boolean, or floating seeds; config seed-list faults; second candidate or changed config within a cell; wrong method/arm/backbone/job ID |
| Parent registry | Registration SHA required; registered seed0-only comparator accepted; missing or wrong registered seed rejected; four registered NOT_SUPPORTED slots synthesized and retained; unexplained or removed unsupported slot rejected |
| Provenance | Corrupted result, policy, prepared input, training manifest, audit, parent study/metrics, runtime source, or admission SHA rejected before unpickling; metadata-only validation guards model, prior prediction, and cohort loaders |
| Mapping integrity | Rehashed comparator, contrast, family, T provenance, or C reload changes rejected against bound upstream evidence |
| Selection/study | Actual 80 generated C/S predictions, 80 exact C-q and C-p byte checks, four prepared loads, 16 reporting selections and 48 tau rows; 96 study slots and 168 contrasts; source/environment binding retained |
| Release/leakage | New branch, known_T=True, release schema/decision, study/selection SHA, arm membership, and fresh output required; bad release never reaches model/prepared/prior-prediction/T-input loading |
| Frozen policy | Rehashed study cannot change branch, known-T flag, arm set, bootstrap count, or bootstrap seed |
| Annual alignment | Domain vectors align with full-year record keys, strata, PSUs, and weights; out-of-domain PSUs contribute to the design; misalignment rejected |
| Paired inference | Identical predictions produce zero paired BA variance despite nonzero marginal variance; opposing seed EO gaps are averaged per seed rather than computed from ensemble q |
| Multiplicity | Primary 20 / secondary 316 / union 336 endpoint slots retained, including missing-model pairs; invalid union or undeclared fixed-BM tau exception rejected |
| Non-estimability | Singleton and missing support remain structured in statistical output; missing model pairs remain present; missing S support fails closed rather than producing a valid freeze |
| Arm integration | Actual selection→study→explicit release→arm001 evaluation; 20 new and one registered seed0 parent model; one common full-year bootstrap call with B=2000; 24 rows and 42 contrast slots; final artifact hashes, known-T marker, and output ownership verified |

Exact test results:
Final command exited 0:
```text
........................................................................ [ 75%]
.......................                                                  [100%]
95 passed in 14.69s
```
Earlier failures were used to improve the parent-owned implementation's runtime-source recheck and deterministic mapping checks. A later synthetic failure used NumPy integer record keys from an older survey unit fixture: production `make_record_key` generates strings. The integration fixture now follows that existing string contract for both domain and annual keys; no production code was changed to accommodate the test. The successful S/study-only intermediate run was 88 passed in 3.54s.

Input hashes:
Final local source snapshot:
```text
completion_evaluation.py  c0f35ac4fe9abcfbd6cd2d58bdd190da2a195157a8ed87a077ae3de3c97a347b
completion_statistics.py  cfde4e6e900388b777d9674124bb0c74a90c6014756a9a240a3881799d97c236
catalog_evaluation.py     6bf830e4a387776002a8731daeaee803f5a03920023cf65fa50a9db42349c103
catalog_inference.py      e64f65bc7052ba4b550b0029907a0fef4e34fb97146903b0b326b34d4d981784
 data_contracts.py        acdb9b163ea217b5db6b17596b8da342560433ebe0338f9db3c108f0a15977f5
survey_linearization.py   f8270032d26f03e5eca4aa84e8813e1b11a15a18922534c9afaa64b3232ab925
test_frozen_evaluation_contract.py 2c1744264dd7b54256fb676d8c31ee4189b3af3b49cba20bf33e5387f79061ea
```
Synthetic fixture input identities are computed inside each temporary run. No actual dataset or fitted-model bytes were opened to obtain these results.

Output hashes:
`tests/benchmark/test_completion_evaluation_contracts.py`:
`20eb0afe1510a1e9b57a8b55b5d998a1e59b70c929863d2a19d06c9c2dd30a67`.
This report's final SHA is supplied separately to the parent, avoiding a self-referential hash. The successful integration independently rehashes every artifact listed in its generated evaluation manifest.

Row counts:
Real S/T rows: 0. Generated new-model matrix: 80 jobs / 16 original configuration cells / five seeds each. Generated parent matrix: 80 comparator slots, including four unsupported slots and one valid seed0-only model. Successful arm001 integration: eight domain rows, 12 annual-design rows, three strata, six annual PSUs, df=3, 21 model predictions, 24 selection rows, 42 paired contrast slots, B=2000.

Assumptions:
The integration intentionally substitutes in-memory generated policies and prepared/cohort loaders; it does not independently re-unpickle actual completion policies or validate their cross-platform numerical behavior. The bare-policy schema and pre-unpickle package checks were inspected, while actual-model validation remains the supervisor's separate responsibility. Statistical tests exercise the real production calculations. No method/seed selection uses real S/T metrics. The branch is explicitly retrospective (`known_T=True`), not an untouched holdout or replacement of the original main analysis.

Unresolved issues:
The frozen implementation rejects nonfinite S metrics during strict JSON serialization. The generated missing-group test confirms that the partial output cannot become a valid study. It does not emit a usable NOT_ESTIMABLE selection freeze for that case; supporting such a freeze needs a separately reviewed version. The supervisor reported the actual S run has complete support and completed, but this worker did not read or independently verify those real outputs. Generic NumPy-integer record-key serialization is unsupported by the integration entry point; the tested NHIS path uses canonical string keys. This suite tests one full arm and does not replace independent four-arm result acceptance, full merge/report acceptance, or actual model loading.

Git diff summary:
Exactly two newly added, untracked worker-owned paths above. No edits to existing production files; no stage/commit. Shared workspace changes belonging to other agents are outside this report.

Proposed next step:
Supervisor independently run the final suite against the frozen source hashes, review the bounded limitations, and separately decide real T release and later four-arm acceptance. Worker does not issue that release.

STOP — waiting for Codex review.
