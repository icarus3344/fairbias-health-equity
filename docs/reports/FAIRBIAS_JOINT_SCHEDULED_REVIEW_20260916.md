# Joint timeout remediation: supervisor review

This review covers the user's explicit authorization to use subagents, optimize FairBias, and resolve Joint timeouts. The existing formal NHIS run and inherited files remain unchanged. New work is isolated in versioned modules and independent F/C diagnostics. This is not a claim of comparative predictive or fairness superiority.

## Findings that explain the timeout

1. **Repeated MDS work dominates.** The preceding 180-second profile spent about 95.8% inside geometry. MDS embeds approximately 22 feature/origin nodes, not all respondents. Elbow selection evaluates up to 15 dimensions; the final embedding uses four initializations. Small distance computations run millions of times. Repeated general-purpose validation and array dispatch consume substantial CPU despite the small matrices.
2. **Strict Joint repeatedly attempts a whole AE search from infeasible states.** AE requires every feature's geometry to satisfy the global epsilon. The prior 600-second diagnostic had 288 AE audit candidates, all rejected for exceeding the fairness cap, with four BM commits and no AE commits. This is progress through an expensive search, not a deadlock.
3. **Wall time and iteration caps are different failures.** At 2026-09-16 15:12 UTC the original run had all 40 Joint receipts: 38 wall-time limits and 2 MDS iteration-cap failures. Faster arithmetic cannot remove a fixed iteration-cap failure. Arm 002 / seed 0 has cap records at geometry calls 35 and 96, selected dimension 3, final MDS `n_init=4`, `max_iter=10000`; it is not an initial-reference-epsilon failure.
4. **A legacy exception boundary can misstate completion.** `FairAccuracyEnhancement._is_fairness_acceptable` catches ordinary exceptions from geometry and treats them as candidate rejection. A resource/numerical interruption means a candidate was not fully assessed. It cannot prove that candidate is inferior or that the whole search is exhausted. The new controller transports typed stops through that boundary and fails on ordinary evaluation errors.
5. **The old adapter discards otherwise usable feasible progress at budget exhaustion.** `max_outer_iterations` actually bounds AE commits. A fully verified feasible state can exist before the next search exceeds a budget. A separate budget-aware termination contract can retain it, but must not claim the search converged.

## Implemented changes

### Exact arithmetic acceleration

`src/nhis_fairbias/benchmark/joint_mds_numpy.py` specializes the reviewed small float64 ndarray self-distance calculation. It uses the same installed NumPy arithmetic order: einsum row norms, matrix multiplication, two ordered additions, clipping, diagonal fill and square root. It retains per-request finite-input validation and falls back to the original public implementation for unsupported types, shapes and arguments.

Admission checks sklearn 1.9.1 and eleven source fingerprints. The scoped context restores hooks even on interruption and rejects nesting. It preserves the original distance matrix, MDS dimensions, initializations, random stream, tolerance, iteration limits, elbow rule, candidates and logical budgets. Full-table geometry memoization remains disabled because the previous diagnostic spent more time hashing than it saved.

Independent parent verification: **206 tests passed locally in 32.18 seconds and on Linux in 46.88 seconds**. Coverage includes 540 distance layout/scale cases, complete MDS coordinates/stress/iteration receipts/RNG, validation fallbacks, cap behavior, and full BM_AE/Joint paths with both deletion and categorical-merge witnesses and budget exits.

Real F/C identical-work observation: original 16-geometry prefix **94.2515 seconds**, prior v2 **35.9097 seconds**, new NumPy specialization **21.2253 seconds**. All 16 ordered geometry hashes and the one returned BM state hash agree with the original. The new observed ratio is **4.44x** versus the original. This is a single sequential timing under a live server workload, not an end-to-end or universal speed guarantee. See `fairbias_joint_scheduled_20260916/prefix_verification.json` and retained executed sources.

### Scheduled Joint and explicit incumbent handling

`src/nhis_fairbias/benchmark/adapters/adapter_fairbias_scheduled.py` defines `ScheduledJointAdapter`, variant `scheduled_joint_v1`. It is an algorithm extension of the current NHIS adaptation, not an equivalent implementation of the old Joint search and not a new claim of full paper reproduction.

- Attempt AE after three successful BM commits, immediately upon global feasibility, or when BM cannot progress. Interval 1 is available as a control. `DEFERRED` is recorded separately from `NO_CANDIDATE`.
- Preserve the same reference epsilon, F/C roles, unweighted C balanced-accuracy utility, LR/GBDT settings, BM/AE candidate rules and geometry.
- Commit a proposed transform only after a complete full-state geometry refresh and budget checks. Save the last verified globally feasible committed state.
- On resource exhaustion, fit that state as `FEASIBLE_BUDGET_LIMITED`, with `convergence_verified=False`. If a candidate hits the MDS cap, use `FEASIBLE_GEOMETRY_INCOMPLETE`. With no feasible incumbent, fail explicitly. Ordinary model/numerical failures and user interruptions never turn into successful budget exits.
- Preserve original conservative commit limits. Reaching ten AE commits does not establish that no further improvement exists.
- Verify partition immutability before every final refit, including a budget exit. Record selected versus committed proposals separately and correct the adapter's AE audit metric to `balanced_accuracy`.

Independent parent verification: **26 tests passed locally in 4.01 seconds and on Linux in 7.70 seconds**. Tests cover scheduling, rescue, deadlines before/after operations, exact evaluation limits, atomicity, feasible/no-feasible exits, cap propagation, ordinary failures, hook restoration, final-fit failures and real LR/GBDT interval-1 parity. The real categorical-merge fixture covers both a verified feasible budget exit and completed search.

### Independent execution and monitoring

`scripts/pilot_scheduled_joint.py` verifies the old registration and prepared-data hashes, snapshots and verifies loaded project module origins, fits only F/C, and writes aggregate receipts. It never invokes the formal worker or evaluates S/T. A strict-adapter fit containing unresolved MDS cap records is explicitly inadmissible even if the old adapter returns a model.

`scripts/supervise_joint_pilot.py` watches the entire child lifetime externally, with a sampled 4 GiB main-child RSS threshold and wall limit, then terminates only that process group if needed. It records a separate receipt even if the child cannot write its own result. The RSS threshold is sampled, not a kernel-enforced allocation cap, and it does not sum descendants. Actual adapters here are single-process. Supervisor exit success alone never means model-fit success.

Independent parent verification: **10 supervisor tests passed locally in 1.03 seconds and on Linux in 1.12 seconds**, including real short subprocess timeouts, SIGTERM resistance/SIGKILL, memory-threshold handling, monitor failures and immutable output paths.

Three additional local source-origin admission tests passed in 0.03 seconds: a valid on-disk hash cannot admit a shadowed module, a source change after the snapshot is rejected, and unknown/unverifiable modules fail closed.

Baseline check: all 14 inherited files match the protected tag. `.gitignore` matches the starting HEAD and was not edited here, but already contains five additional lines relative to that tag, introduced by historical commit `b595e59`. Those existing private-file exclusion rules were preserved; this review does not falsely certify `.gitignore` as byte-identical to the baseline tag.

## Pilot results

The 16-geometry parity prefix is complete and verified.

The exact accelerated strict arm 003 / LR / seed 0 pilot reached its 600-second alarm at **600.037 seconds**, without a fitted model. It made 494 geometry requests, returned six BM proposals, and accepted zero AE proposals. Its 459 completed AE audit candidates were all `EXCEEDS_FAIRNESS_CAP`; 7,893 completed MDS fit receipts reported the selected initialization converging before its cap. The interrupted last request is not represented as converged. Geometry accounted for 478.30 seconds; AE including geometry accounted for 563.25 seconds. These inclusive times must not be added together. Observed maximum RSS was about 727 MiB. Loaded module origins and both old/new source hashes verified successfully. The external supervisor recorded normal process exit because the child handled its own alarm; the child receipt correctly records `PILOT_HARD_LIMIT`, not success.

The scheduled arm 003 / LR / seed 0 pilot **completed its declared search and final fit in 587.132 seconds**. It committed 12 BM and 6 AE transforms, deferred AE eight times, and ended with `BM: NOOP_FEASIBLE` followed by `AE: NO_CANDIDATE`. Termination is `SEARCH_EXHAUSTED`, `convergence_verified=True`, with no resource-stop reason. Final maximum d_phi **0.0023127952466163444** is below epsilon **0.003017467969656895**. It made 446 geometry requests and 129 utility requests, with 7,136 completed MDS receipts and no selected-initialization cap. Maximum RSS was about 733 MiB. Supervisor and source-origin/hash checks passed; S/T were not evaluated.

The new LR final transform-state fingerprint, epsilon threshold and final maximum d_phi equal those of the already completed original BM_AE / arm 003 / LR / seed 0 fit (`c92bc5dfb91d19f3cee4`, 1305.120 seconds). This is a single final-state comparison, not a new cross-method prediction test or proof of schedule equivalence. It gives no evidence that the scheduled algorithm improves clinical fairness or prediction over BM_AE. Its observed benefit here is reaching the same verified representation within a smaller elapsed budget.

The scheduled arm 003 / GBDT / seed 0 pilot **returned a fitted, verified feasible incumbent in 902.750 seconds**, after 12 BM and 9 AE commits. Its 900-second search deadline was reached after a utility evaluation; final refitting used the reserved time. Termination is `FEASIBLE_BUDGET_LIMITED`, `budget_reason="SEARCH_TIME_LIMIT: after utility"`, `convergence_verified=False`. Final maximum d_phi **0.0022110902937514656** remains below the same epsilon **0.003017467969656895**. The interrupted candidate was not committed: the last trace remains `AE: STARTED`, with `selected=False`, `committed=False`. This is successful incumbent recovery, not completed search.

GBDT made 486 geometry and 150 utility requests. Its 7,776 completed MDS receipts had no selected-initialization caps, and maximum RSS was about 735 MiB. Geometry consumed 387.16 seconds and utility/model evaluation 390.85 seconds; the remaining cost is other processing. Thus further MDS acceleration alone cannot remove most of the remaining runtime. Supervisor, module-origin and source checks passed, and S/T were not evaluated.

| Diagnostic on arm 003 / seed 0 | Elapsed | Outcome | BM / AE commits | Search completed |
|---|---:|---|---:|---|
| Original registered Joint / LR | 1800-second limit | Timeout, no valid fitted result | Not retained by old failure receipt | No |
| Exact accelerated strict Joint / LR | 600.04 s | Diagnostic time limit | 6 BM engine returns / 0 AE returns | No |
| Scheduled Joint / LR | 587.13 s | Fitted feasible model | 12 / 6 | Yes |
| Original registered Joint / GBDT | 1800-second limit | Timeout, no valid fitted result | Not retained by old failure receipt | No |
| Scheduled Joint / GBDT | 902.75 s | Fitted feasible incumbent | 12 / 9 | No; declared search budget |

The scheduled rows change the search policy and cannot be used as an exact end-to-end speed ratio against original Joint. Neither row supplies S/T predictive metrics, population fairness estimates or confidence intervals.

## Selected-initialization MDS cap recovery

The separate module `src/nhis_fairbias/benchmark/mds_budget_retry.py` permits one fresh, same-seed fit with twice the iteration cap, preserving other MDS parameters. It records both the initial incomplete attempt and the retry, including identical-input distance hashes. No-cap outputs use the original call; nonfinite output, mutable/unset RNG and explicit initialization receive no retry. An unresolved second cap still reaches the original audit and fails. The wrapper `scripts/pilot_joint_mds_retry.py` preserves the base result and adds a separate receipt identifying the combined budget policy.

Parent verification: **30 tests passed locally in 1.42 seconds and on Linux in 3.26 seconds**. They cover real nested retry/NumPy/audit execution, fresh seeded fitting, unchanged no-cap outputs, unresolved caps, source/version checks, interruption records and both-receipt handling.

The separate arm 002 / LR / seed 0 real F/C diagnostic returned a fitted feasible incumbent in **361.313 seconds**, with three BM and eight AE commits. Its 360-second search limit was reached after a geometry evaluation. Final maximum d_phi **0.0020424583938788905** is below epsilon **0.002670203396539078**. Termination is `FEASIBLE_BUDGET_LIMITED`, with `convergence_verified=False`; this is not completed search.

All 2,912 logical MDS fits completed before their effective selected-initialization caps. Three calls required a second attempt, giving 2,915 recorded attempts. These are three repeated evaluations of **one unique distance matrix**, not three independent cap cases. Each first attempt reached 10,000 iterations at stress **2.868674738594887e-7**. Each fresh same-seed retry converged after **10,426** iterations under a 20,000 cap, with stress **2.8674344486685906e-7**. The paired input distance hashes match exactly. This directly verifies recovery for the observed capped state, while retaining the incomplete first attempts in the audit trail. It does not show that every capped state or seed will recover.

The child made 182 geometry requests and 79 utility requests; maximum RSS was approximately 741 MiB. Both source-identity layers and loaded-module origin checks passed. The retry wrapper's source hashes were unchanged, its base-result hash matches the preserved base result, and the external supervisor recorded normal exit without a forced termination. S/T were not evaluated. Consumers must read both `result.json` and `mds_retry_receipt.json`: the effective variant is `scheduled_joint_v1_with_mds_budget_retry_v1`, not the unmodified scheduled controller alone.

## Scientific interpretation and next admission

The strict accelerated adapter is the preferred route for retaining the current primary FairBias method. The scheduled version belongs in a separately labelled extension/ablation until complete paired comparisons exist. Better runtime or a returned feasible model does not imply better prediction or better population-level fairness.

For any expanded run, retain all 4 arms x 2 backbones x 5 seeds, every timeout/cap/failure, actual compute use and final feasibility. Separate fully searched solutions from budget-limited incumbents. Compare the original strict Joint, exact accelerated Joint, existing BM_AE and scheduled extension under declared budgets; do not combine successful seeds from different versions. Keep S/T out of decisions about interval, tolerance or budget, and preserve the existing survey-inference and selection-freeze gates.

The MDS audit exposes the selected best initialization's `n_iter_`; summing it is not total work across all initializations. Search completion is not a mathematical proof of global optimum. An increase in MDS iteration caps or a tolerance change requires its own version and F/C numerical sensitivity assessment; it is not part of the exact speedup implemented here.

## Supervisor disposition and evidence

**Accepted for bounded F/C pilot use:** the exact arithmetic specialization, scheduled controller with explicit incumbent termination, external supervision, and separately versioned MDS retry. The parent independently passed **275 targeted local tests** and **272 Linux tests**; the three module-origin tests were run locally only. These counts are specific to the new components, not a claim that every repository test was rerun.

**Not admitted as completed primary-study evidence:** the revised controller has not undergone the full paired 4-arm x 2-backbone x 5-seed experiment, S selection, frozen T evaluation or survey-weighted inference. Old Joint failures remain failures in their original run. No pilot result was inserted into that run. A new formal registration must identify schedule, retry policy, budget and code versions before producing comparable study results.

The final live check at **2026-09-16 15:56:51 UTC** confirmed all **70 registered source files** unchanged both locally and remotely, the original registration hash unchanged, and the existing dispatcher still alive. All 14 inherited files match the protected baseline; the preexisting `.gitignore` deviation is documented above. No source was staged or committed in this work.

The evidence directory is `docs/reports/fairbias_joint_scheduled_20260916/`. It contains original child/supervisor receipts, executed source snapshots, hash-verified transfers, test logs, the same-work prefix comparison, the LR/BM_AE final-state comparison, and the main-run identity snapshot. `verification.json` records the parent's acceptance checks; `MANIFEST.json` binds the evidence files. The plan is `docs/plans/FAIRBIAS_JOINT_SCHEDULED_PILOT_20260916.md`.

Code entry points: exact specialization at `src/nhis_fairbias/benchmark/joint_mds_numpy.py:97`; controller admission and finalization at `src/nhis_fairbias/benchmark/adapters/adapter_fairbias_scheduled.py:150`, state commit checks at line 267, and schedule at line 310; retry policy at `src/nhis_fairbias/benchmark/mds_budget_retry.py:94`; process monitoring at `scripts/supervise_joint_pilot.py:66`.
