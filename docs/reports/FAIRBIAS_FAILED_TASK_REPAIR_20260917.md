# Failed-task repair and rerun readiness — 2026-09-17

Supervisor gate: `docs/plans/FAIRBIAS_FAILED_TASK_RECOVERY_20260917.md`.
Original source identity: `818f01dbe1fcaa119493efa78acabf165961d1fdeaabb8e23f36a86dce7d5891`.
Original registration SHA256: `340cd60f6ab0d5919e2ac7bc22b62bdb3cc758faaa986650abdb3978b964732a`.

## Verified inventory

The metadata-only inventory independently identifies **1,242** failed/incomplete seed jobs, excluding the superseded FRAPPÉ runtime. This is not 1,242 independent expensive optimizations.

| Family | Original incomplete jobs | Root cause / recovery |
|---|---:|---|
| LFR | 960 | 120 shared representation keys; finite-difference objective evaluation budget exhaustion. Analytic derivative of the unchanged objective. |
| EG_DP / EG_EO | 23 | Five of six representative configurations exhibit tiny C probability endpoint violations; sixth is valid on C. Bounded convex-mixture validation and endpoint roundoff repair. |
| FairBias BM | 198 | Eight undefined-geometry keys (176 dependent jobs), one MDS-cap key (22 jobs). One direct undefined-geometry root now confirmed as feature exhaustion on its search path. |
| BM_AE | 21 | 14 AE commit-cap exits, two MDS-cap exits, five time limits. Explicit cap10→40 retry, exact kernel acceleration and separately identified MDS budget retry. |
| Joint | 40 | 38 time limits, two MDS-cap exits. Exact kernel / strict-search pilot; scheduled search remains a separate method variant. |

Inventory and hashes: `fairbias_recovery_20260917/evidence/failure_inventory.json`. All 1,242 original job/result pairs were transferred as metadata only and checked against this inventory. Historical results/cache remain untouched.

## Completed independent F/C verification

All **nine** prespecified numerical pilots completed with exact policy reload and matching independently rehashed model artifacts. No S/T evaluation or comparative ranking was used in these pilots. Existing prepared F/C/S containers were deserialized, but S contents were not used; this is not a claim that S bytes were never loaded.

| Pilot | Original configuration | New elapsed seconds | Verified outcome |
|---|---|---:|---|
| LFR arm003 | k10, Az0.1, seed0 | 158.23 | 428 evaluations / 405 iterations; projected gradient 8.92e-6; optimizer gradient stopping criterion met. |
| LFR arm001 | k5, Az50, seed0 | 7.21 | 79 evaluations; relative-loss termination; projected gradient 0.0542. |
| LFR arm004 | k5, Az50, seed0 | 8.58 | 131 evaluations; relative-loss termination; projected gradient 0.2646. |
| EG six candidates | All six affected configs, seed0 | 27.10–288.92 | All passed C contract and reload. Five repaired 3–28 endpoint values by 2.22e-16; interior q values unchanged. |

The old arm003 LFR example exhausted 5,805 evaluations after four iterations in 331.78s. The two durations were measured on different hosts, so their ratio is not a controlled hardware speedup estimate. Reduced evaluation count and verified termination are the relevant evidence. LFR relative-loss termination does **not** certify stationarity or a global optimum; the large projected gradients for the high-Az pilots remain disclosed.

Evidence: `evidence/numerical_pilot_verification.json`, `evidence/numerical_model_hash_check.json` and per-pilot JSON under `evidence/numerical_pilots/`.

## BM root cause and honest failure status

F-only replay of `00b3d9db420e1187bb46_s0` completed in 446.43s with 255 geometry calls. Call254 still had one finite feature; call255 had **zero columns and zero d_phi entries**, with no negative or nonfinite numerical entries. The old combined error text incorrectly obscures this distinction. The historical diagnostic did not explicitly fix Python's hash seed and is structural evidence, not a bitwise replay certificate.

New `adapter_bm_recovery.py` raises the explicit `SearchFeatureExhausted` outcome, retains failure and admits no empty model. It does not invent zero bias or declare the entire constraint mathematically infeasible. The conclusion is limited to the observed search path; the other seven undefined-geometry roots require their own confirmation.

The MDS-retry pilot for `0ea73545d90f6c351057_s0` completed in **63.81s**, reached the bias constraint and saved/reloaded the representation. Independent remote model hashing matches its receipt. Of256 logical MDS fits, three required and completed the prescribed doubled-cap retry; those three share **one distinct distance matrix**, not three independent problems. Python hash seed0 and the exact NumPy kernel were used. This is F-only representation evidence; the legacy status name `FC_REPRESENTATION_PASS` does not imply C prediction/calibration was evaluated. The22 dependent formal prediction/evaluation jobs remain unreleased. Evidence: `evidence/bm_mds/independent_verification.json`.

## AE / Joint budget handling

`adapter_joint_ae_recovery.py` preserves strict search by default and transports geometry failures past legacy exception-swallowing. Complete feasible search, feasible budget-limited incumbents and incomplete geometry have distinct statuses. None is automatically a formal-study result.

The 14 AE-cap failures stop immediately after the tenth commit, before checking another no-improvement step. Exact acceleration alone cannot establish convergence. The opt-in recovery retains the original cap10 attempt and retries from the same seed with cap40 only for that exact exit; utility500, geometry20000 and BM50 remain unchanged. A cap40 failure remains failure. This is a new budget policy, not an unchanged original-budget result.

Both1800s CPU F/C pilots reached terminal states and were independently checked at12:15:34:

- Strict Joint `cc84d0d23c1bc5af275d_s0` completed in1705.90s (external supervisor1710.47s,exit0). Final maximumdφ0.002312795 is withinε0.003017468; termination is `STRICT_FEASIBLE_SEARCH_EXHAUSTED`, and all17,344 recorded MDS fits report convergence before cap. Exact C-policy reload was performed by the pilot; an independent rehash of the remote saved model matched `0f595746d4588f55acdfdf1cbf765a94b27e522cd3bdd80829a89d03e705beb7`. Search order and budgets were unchanged. This validates one F/C case, not all38 historical wall-time failures or formal S/T performance.
- BM_AE `3c8badc6d7eeaa99adce_s0` was externally stopped without a model. Its cap10 attempt consumed619.23s and reached the AE commit cap. The fresh cap40 attempt then ran1177.59s before the combined1800s wall limit. This is **external wall-budget interruption**, not evidence that40 commits were exhausted. A longer, explicitly identified wall-budget sensitivity pilot or resumable search design is still needed; no invalid result is promoted.

The verification binds80 project-source entries,3 script sources,7 component fingerprints,70 registered sources and42 loaded module origins. Summarized evidence: `evidence/joint_ae_pilots/independent_verification_20260917T041534.765143_0000.json`.

## Formal rerun release

The independently tested `run_nhis_numerical_recovery_worker.py` reuses the frozen production worker through a process-local adapter factory. Original job/config/seed/data/registration hashes are bound to every rerun. The complete runtime identity names the fresh run/cache; cache namespace and per-representation sidecars bind model/status hashes. Old caches and legacy EG VALID reuse are rejected.

The LFR queue was released on the CPU host at12:07:14 Asia/Shanghai, PID515609, with **960 jobs /120 representations**,28 workers,48GiB sampled RSS ceiling and12GiB host memory reserve. Runtime `d5d7ec7388a65a950f71cdd290daae6b4b79003a10d41d8e74acfc297d7aa002` binds85 files; all85 match the supervisor checkout, and all70 original registered source files remain unchanged.

At12:11:30,105 completed jobs were independently reverified against both worker and scheduler receipts and all listed output hashes. The subsequent live snapshot recorded107 finalized,28 active and0 failures. A two-second surviving-process sample accounted for25.89 CPU cores for LFR alone; short-lived jobs and the two pilots are excluded from that lower-bound sample. Cgroup memory was27.31GiB of60GiB. Evidence: `evidence/lfr_queue_independent_snapshot.json`.

Controller provenance caveat: the controller started before the final malformed-evidence hardening was copied to disk (file ctime12:08:12). The in-memory running controller retains its earlier code; that last change strengthens rejection/archival of malformed receipt types and result identities. The frozen scientific worker and all85 runtime-bound files were unchanged. The active queue has independently verified valid receipts; it must not be described as running the final controller revision. Final controller SHA256 `1349d35864161ee67ae7773d239397aba3939284d29f4570a0da69bdd1e82eca` passes28 tests on Linux, including real concurrent success/budget failure and malformed-evidence cases, and will be used for the subsequent EG queue. Do not restart or overwrite completed LFR work merely to update this orchestration-only change.

The separate23 EG reruns share the numerical worker/runtime and use a distinct namespace after LFR completes. All1,536 original EG job/result metadata pairs have been hash-verified on the CPU; exact original FAILED coverage isDP15+EO8. This copies metadata only, not fitted models or predictions. Pooling the original1,513 valid EG results requires a later compatibility audit. EG waiting controllerPID533441 was launched at12:13:05, and its live log confirms it is waiting for LFR completion. No EG fit was claimed at launch. EG scheduler tests16 passed on Linux in7.07s, including an actual generated-data production-worker fit and reload. Evidence: `evidence/eg_queue_launch.json`.

## Verification and boundaries

Independent Linux checks completed: numerical objective/adapter/pilot suite73 passed; formal numerical worker34 passed; Joint/AE facade21 passed; Joint/AE wrapper4 passed; BM outcome adapter5 passed. Local BM diagnostics/outcome suite11 passed. Final numerical queue suite28 passed locally (5.43s) and Linux (7.99s). A group-swap loss test was corrected from bitwise equality to an8-epsilon symmetry bound because reversing the pooled-loss summation order differed by one ULP on Linux; fixed-order upstream objective parity tests remain intact.

The14 inherited root files match the protected baseline. `.gitignore` is unchanged against HEAD; its historical committed difference from the baseline is preserved. No stage/commit, no local NHIS fit, no new T2024 evaluation. This report establishes repair implementation and bounded execution evidence, not FairBias superiority or paper readiness.


## GPU node retirement steering — 12:17

User requested letting the present GPU-node batch finish and moving subsequent work to CPU. No new source-node fits or BM diagnostic waiters were launched. The proposed8-root BM diagnostic launcher is offline only and has not been supervisor-released. At12:16:41, source FRAPPE had95 of100 finalized and five actual workers (the event-based active_count6 included a just-finalized worker). Their ages were498–536s; prior source jobs ranged449–788s. Training tail estimate3–6min, conservatively5–10min, is conditional and excludes backup. Low utilization is consistent with only five jobs remaining on a16-core allocation; it is not evidence of GPU hardware malfunction.

A separate archival transfer is authorized before retirement. Preserve the original run and all source-node artifacts, models, prediction files, inputs and source/environment metadata in a fresh CPU/local archive. Final hash verification must occur after the source queue is COMPLETE and no training worker remains. Do not shut down the instance or call the transfer complete from a launch record alone.

### Completed FRAPPÉ accounting

At12:20 the source100 queue reached COMPLETE with0failed/0active. Parent independently verified all100 source receipts/output hashes and the70 original source hashes. A subsequent cross-host verification covered the CPU300 and80 shards: **480 unique VALID receipts exactly match all480 registered corrected FRAPPÉ jobs**. No missing jobs or overlap. The CPU300 latest session reports236 completions, with64 earlier verified receipts; cumulative coverage comes from receipts rather than the latest session counter. Evidence: `evidence/frappe480_full_receipt_verification.json`. No comparative S metrics were used in this check. Transfer/retirement remains a separate pending operation.

### Direct retirement transfer started

The initial Mac relay was too slow and its orchestration was superseded. Its two local processes were stopped without deleting the partial snapshot or changing research output. The accepted transfer now runs **directly from source to CPU**, PID590936, using a temporary read-only rsync authorization restricted to the source project and a pinned SSH host key. Private keys remain on the CPU host. No shutdown is performed by the transfer helper.

A source manifest frozen after training completion covers **44,909 files /5,468,502,929 bytes**, SHA256 `45090455627b6957238651ae169874fc23f5dd882be704f64c29ef5d9daf24db`. Research artifacts, data, source, scripts, configuration, documentation, tests and selected root dependency/governance files are included; reusable installed environments and package caches are explicitly excluded. The deployed CPU environment already exists and source pip freeze is preserved separately. `backup_nhis_source_retirement.py` passed a generated local real-rsync test covering file hashes, symlinks, exclusions and the verified receipt. Final CPU hash agreement is required before declaring retirement safe. Local backup can proceed from CPU afterward, independent of the GPU instance.

### Retirement acceptance — 13:20 Asia/Shanghai

The direct transfer completed at13:08:03. At13:14:03 the supervisor independently rehashed all44,909 archived files, checked sizes/types/symlinks against the frozen manifest and found0 mismatches. The accepted CPU snapshot contains5,468,502,929 bytes. Source manifest identity was unchanged; the source queue was COMPLETE100/100 with no remaining training workers. Evidence: `evidence/retirement_supervisor_verification.json`, status `SUPERVISOR_ARCHIVE_ACCEPTED`.

The GPU instance can now be stopped by the user; no shutdown or deletion was performed. Only the specifically tagged temporary read-only SSH authorization and newly created transfer key were removed, with other authorizations preserved (`evidence/retirement_auth_cleanup.json`). The supplemental local copy launched at13:16 from CPU and is still in progress, not a completed second backup. Its completion is independent of the GPU instance. The existing heartbeat returned to a2-hour cadence and treats expected GPU retirement as normal. This acceptance establishes archival integrity, not completion of the remaining FairBias repair or scientific analysis gates.

### Numerical recovery completion verified — 13:25 Asia/Shanghai

LFR finalized960/960 at12:38:01 and its automatic EG successor finalized23/23 at12:43:09, both with0failed and0active. Live process inspection found no remaining training workers. Independent supervisor verification covered all983 worker and scheduler receipts, exact failure-inventory coverage, all receipt-listed output hashes, original job/result and data identities, recorded module-source hashes,85 runtime sources,70 original sources and4 prepared files. All983 execution receipts were VALID;9,920 unique file hashes were checked. Evidence: `evidence/numerical_983_final_verification_20260917.json`. Prediction arrays and comparative metric rankings were not inspected.

The empty queue explains the current low compute demand. FairBias BM198/BM_AE21/Joint40 historical incomplete cases still need diagnostic/recovery admission and are not currently scheduled. EG legacy-result compatibility, LFR optimizer diagnostics, scientific aggregation and subsequent selection/inference gates remain separate from this execution acceptance. Do not claim the whole benchmark or paper is finished.

### CPU continuation and eight BM root outcomes — 13:40 Asia/Shanghai

The supervisor launched eight distinct BM representation-root F-only replays and one fresh BM_AE cap10-to40 F/C pilot at13:32:37. All nine actual children were verified running. The BM replay budget is900s per root; BM_AE is an explicitly identified7200s combined wall-budget sensitivity run. Sources and inputs are hash-bound; no original result is overwritten. Local19 pilot-related tests passed7.75s and the same Linux19 passed7.00s.

All eight BM diagnostics have now exited in271.66–329.69s with the explicit `SEARCH_FEATURE_EXHAUSTED` outcome and no successful model. Supervisor process exits are0 because the diagnostic script records outcomes; they are **not successful fits**. Parent checked every terminal result and supervisor receipt, original job SHA and all83 recorded project-source hashes. These eight representation keys map exactly to176 historical dependent failures (22 each). Evidence: `evidence/fairbias_followup_pilot_progress_20260917.json` and `evidence/bm_176_failed_path_mapping_20260917.json`. This diagnoses the reviewed deterministic search paths; it does not prove that every possible representation is infeasible. Retain all176 original failures in accounting rather than endlessly rerunning the same paths or inventing successful models.

The new independent formal worker supports BM MDS retry and strict exact-kernel Joint only, retaining BM's original mixed weighted/unweighted predictor semantics. Worker SHA256 `5365453ef2a1ec1eafa18cbd922206b84d33b2ac2605d30e1505906b9ab7a4ff`;34 generated-data tests passed locally,68 new/old worker tests passed together, and parent independently reran34 on Linux (13.10s). Tests include actual production F/C/S fitting, weighted/unweighted BM shared representation cache, policy serialization/reload and failed-fit rejection. All85 earlier runtime-bound sources remained unchanged on deployment. Formal BM22/Joint38 queue release is still pending scheduler verification at this snapshot; BM_AE remains a running pilot. The remaining83 historical FairBias cases comprise BM22, BM_AE21 andJoint40.

### Formal BM22 and Joint38 release — 13:49 Asia/Shanghai

Supervisor decision: **ACCEPT FOR BOUNDED EXECUTION**, not scientific-result acceptance. The new queue passed23 tests locally and the parent independently reran the same23 onLinux (13.18s), including actual generated-data production children, shared BM cache, strict Joint, metadata integrity, duplicate ownership, PID reuse and external-resource accounting. Queue SHA256 `da731f0c1fbf6d3d4c20afac229ed65f7f721181ef021cd8119e315c34ea32ca`. Read-only preflight verified exactBM22 andJoint38 membership and89 runtime-source hashes against the supervisor checkout. BM22 includes14 unweighted and8 weighted predictors over one shared representation key. No comparative metrics were used for release.

The detached CPU controllerPID616131 started at13:48:46. At13:49:16 its first BM workerPID616195 was actively calculating (28.86 CPU seconds,718.7MiB RSS); the BM phase wasRUNNING with22 scheduled and0 finalized. AfterBM22, the same controller startsJoint38 automatically. A fresh namespace, separate policy-bound runtime identities and caches preserve historical artifacts. BM uses `a214eadb62ad8b480703335aa9aca655438fe2327ea160f757c1fb32ef2c2d45`; Joint uses `d9f80f1ddfef2fdf066f909791b5b2a721b0d12d2c69a55cc7dc3f4035aa7bf9`.

The28-slot/48-GiB resource policy reserves4GiB and one slot for each still-active diagnostic child, plus12GiB cgroup headroom. With onlyBM_AE active, formal capacity is at most27 workers/44GiB. Each formal worker retains the registered1800s wall and4GiB RSS allowance. BM's explicitly declared2x MDS iteration retry remains a distinct budget variant; Joint retains strict search without MDS retry or feasible-incumbent substitution. These are sampled memory safeguards, not kernel memory caps. NoGPU tasks were launched. Evidence: `evidence/formal_fairbias_preflight.json`, `evidence/formal_fairbias_launch_20260917.json`. Final outcome/receipt acceptance, BM_AE21 andJoint2MDS-cap remain outstanding.

### First formal phase complete — 13:51:46 Asia/Shanghai

BM22 reachedCOMPLETE22/22 with0failed/0active. Parent independently rehashed all files listed in its22 worker and scheduler receipts and checked matching VALID terminal statuses. Joint38 then started automatically under the same controller, with0 finalized at this snapshot. BM_AE long-wall pilot remained active. Evidence: `evidence/formal_fairbias_initial_execution_verification.json`. These22 completed fits belong to the explicitly identified BM MDS-retry runtime; no comparative metric ranking or unchanged-budget equivalence is claimed.
