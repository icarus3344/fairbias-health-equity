# Joint compute and scheduled-search pilot

## Authority and frozen inputs

The user explicitly authorized subagent discussion and FairBias algorithm optimization on 2026-09-16. Codex supervises implementation, verification and pilot admission. The existing AutoDL execution authorization applies: NHIS F/C only for diagnostic fitting, no local real-data training, no MEPS transfer, no S/T evaluation or selection. No stage/commit or changes to inherited files.

The registered run `benchmark_autodl_20260916_075038Z` is a read-only input to these pilots. Its 70 registered sources and historical results stay unchanged. New implementation files and a fresh independent output directory are mandatory; check registered source and prepared-data hashes before and after fitting. Prepared joblib contains F/C/S, but the pilot retains/accesses F and C only and drops the enclosing object. All output is aggregate-only.

## Predeclared variants

1. `strict_numpy`: the original registered `FairBiasAEAdapter`, with an opt-in NumPy specialization of the same MDS arithmetic. Same geometry, candidate order, random initializations, elbow search, tolerances and logical budgets. Require byte-exact distance, MDS and whole-adapter witnesses on macOS and Linux. Runtime speed remains empirical, not guaranteed.
2. `scheduled_joint_v1`: a separately named algorithm extension, using the same geometry and backbone contract. Attempt AE after every **3 successful BM commits**, immediately when the current state is globally feasible, or when BM makes no progress. Reset the schedule after an AE attempt. A deferred AE is neither a rejected candidate nor evidence of completed search. Interval 1 is a control for the scheduling rule.

The scheduled variant retains a snapshot only after a complete global geometry verification. On wall/utility/geometry/commit budget exhaustion it may fit the last verified feasible state, explicitly labelled `FEASIBLE_BUDGET_LIMITED`, with `convergence_verified=False`. MDS iteration-cap interruptions receive a separate incomplete-geometry status. If no such state exists, the attempt fails. Ordinary implementation/numerical errors must fail; they cannot promote an incumbent into a completed search. Final prediction-model fitting must succeed before any fitted model is declared available. This is a different stopping contract from the original strict adapter and must be reported separately in a paper.

## Fixed parameters and resource admission

- Preserve original F/C identities, mean-F reference epsilon multiplied by 1.0, unweighted C balanced accuracy, LR/GBDT parameters, official BM power stream/restart, AE six exponents, strict global epsilon and all numerical settings.
- Preserve 50 BM commits, 10 AE commits, 500 utility requests and 20,000 geometry requests. State explicitly that the old `max_outer_iterations` parameter bounds AE commits.
- Initial diagnostic targets: arm 003, seed 0, LR candidate `cc84d0d23c1bc5af275d` and GBDT candidate `394a639c29db43ad6ea6`. Both already have original-run timeout records, making them useful failure witnesses. They do not constitute representative performance evidence.
- Strict NumPy prefix: 16 geometry requests for comparison with the prior identical-work diagnostic. Bounded strict continuation: at most 600 seconds. Scheduled fit: 900 seconds search, 960 seconds hard process alarm. If needed, a second explicitly recorded 1500/1560-second attempt is permitted without changing scientific parameters. Never overwrite a prior attempt.
- One additional diagnostic process at a time, nice 19, one BLAS/OpenMP thread, 4 GiB RSS monitoring, Linux environment pinned to the recorded dependency set. The existing formal dispatcher remains active. Longer fits use an external supervisor: sample main-child RSS every 0.5 seconds; stop at the child alarm budget plus 15 seconds measured from process spawn, then allow 5 seconds before SIGKILL. The RSS threshold is sampled, not a kernel-enforced allocation limit, and does not sum hypothetical descendants. The actual adapters run in one process.

## Required verification and decision

Synthetic contracts cover numerical parity, fallback validation, exception restoration, scheduling/rescue, commit atomicity, budget boundaries, failure propagation and provenance. Supervisor independently runs the tests and reviews source. Linux witnesses are required before real-data pilots.

Pilot receipts record code/registration/prepared hashes, source unchanged checks, loaded project module origins and their source hashes, configuration, elapsed/user CPU/peak RSS, logical geometry/utility requests, MDS convergence/cap counts, committed BM/AE counts, scheduling decisions and termination reason. No per-person predictions or S/T metrics. A full fitted object is optional private runtime output and must not be published as aggregate evidence.

`convergence_verified` describes the declared discrete search termination and the returned MDS solution's checked stopping criterion. It is not proof of a global optimum. sklearn's exposed `n_iter_` describes the selected best initialization, so summed recorded iterations are not the total work of all initializations. A strict-adapter fit with an earlier unresolved MDS cap is explicitly inadmissible in the diagnostic, even if the legacy adapter returns a fitted object.

A successful pilot establishes implementability and a runtime observation. It does not establish FairBias superiority, original-paper equivalence, adequate survey inference, or readiness of the whole benchmark. Production integration requires a new registered method/run with the same complete arm/backbone/seed matrix; retain failures in the denominator and compare original strict Joint, BM_AE, exact-engineering Joint and the scheduled extension separately. No selecting favorable seeds or merging new results into original receipts.

## Additional selected-initialization MDS budget diagnostic

Two original arm 002 / seed 0 Joint jobs failed on a selected final MDS initialization reaching 10,000 iterations. After identifying this distinct failure mechanism, the supervisor admits a separately versioned diagnostic, `scheduled_joint_v1_with_mds_budget_retry_v1`, before any new arm 002 fitting.

On finite MDS output whose selected initialization hits its cap, retry once from the same original integer seed and the same parameters, increasing only `max_iter` to twice its original value (10,000 to 20,000 for the final embedding; 20,000 to 40,000 for an elbow fit). Do not warm-start from the incomplete embedding. Nonfinite output, explicit initialization arrays and mutable RNG objects receive no retry. Uncapped fits preserve their original outputs. Retain both attempt receipts and the cap actually used; a second capped result remains incomplete and must not be admitted as convergence. This is an increased numerical work budget, not a bit-equivalence claim for previously capped paths.

Implementation and synthetic verification must finish before a single arm 002 / LR / seed 0 F/C diagnostic (`7fc06dd6f9ea53dbccf1`) is admitted. Its search limit is 360 seconds and its in-process alarm 420 seconds, with the same external supervisor policy. This short run tests whether the previously blocking cap is recoverable; it need not complete the whole search. The wrapper writes a separate retry receipt and never rewrites the base runner's result. Read both receipts to identify the effective variant. No existing running source or result is modified.
