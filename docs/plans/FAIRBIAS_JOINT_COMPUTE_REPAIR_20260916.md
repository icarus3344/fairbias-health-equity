# Joint compute diagnosis and isolated repair gate

User authority: investigate and try to solve Joint's compute-budget issue; existing full-study authorization remains in force. Codex supervises admission. Existing registered workers, budgets, sources and sealed results remain unchanged; no staging or commits.

## Evidence and scope

The registered run is `benchmark_autodl_20260916_075038Z`, registration SHA-256 `340cd60f6ab0d5919e2ac7bc22b62bdb3cc758faaa986650abdb3978b964732a`. Joint uses 30 minutes per job, 50 BM commits, 10 AE commits, 500 utility evaluations and 20,000 logical geometry evaluations. An attempted job reaching a limit is not a successful converged method.

The first diagnostic is the pre-existing failed arm_003 / LR / seed 0 (`cc84d0d23c1bc5af275d`). It is selected as a representative timeout, without consulting S performance. It uses the existing F/C split, original settings, and one low-priority CPU process. Registered prepared files contain F/C/S; the diagnostic deserializes that payload, retains F/C only, and does not inspect S or produce S predictions. T is absent. Only aggregate timings and state/result hashes are exported.

## Registered bounded diagnostic sequence

1. Profile the unmodified fit for 180 seconds; distinguish profiler overhead from ordinary timing.
2. Run an unprofiled original prefix of exactly 16 geometry requests, with a 180-second protective limit.
3. Verify pure synthetic contracts for the opt-in optimization on macOS and the actual Linux environment. Reject unreviewed sklearn sources before installing hooks.
4. Run the same 16-request prefix with the optimized path. Require identical ordered geometry hashes, BM state hashes, logical geometry counts and utility counts. Treat timing as one live-server observation, not a universal speed claim.
5. If steps 3–4 pass, attempt the same F/C-only job with the optimized path for at most the existing 1,800 seconds. Do not raise scientific budgets, modify the power stream/restart rule, change epsilon, reduce MDS iterations/initializations, change dimensionality, or select by S/T outcomes.
6. Record precise failure type if it still fails. An AE commit limit, MDS cap and a wall-time limit require separate decisions; never relabel a limit as convergence. A full frozen benchmark replacement requires a separately versioned run and complete seed handling, not overwriting or merging the old failed receipts.

## Implementation admission

`joint_budget_optimization.py` is an opt-in module that the current registered worker does not import. Its hooks exist only inside the independent diagnostic process and restore even on BaseException.

- MDS fast path: same reviewed sklearn 1.9.1 `_euclidean_distances` arithmetic; specialize validation only for a plain finite float64 ndarray of at most 129 points by 64 dimensions and the default self-distance call. Other inputs go through the original public validation. Check finiteness on every iteration. Keep MDS stress, points, iteration counts, RNG, initializations and caps unchanged. Pin source fingerprints for public distance, private kernel and SMACOF; no global assume-finite option.
- Optional exact geometry memo: process-scoped, at most 128 entries; hash complete data, indices, schema, protected values, weights, seed and geometry options. Cache only complete successful finite results; copy results on return; bypass mutable or unspecified RNG. Preserve logical geometry request budgets and expose cache-hit counts separately from actual MDS fits. Final v2 defaults this OFF: the 600-second pilot measured about 28.48 seconds in hashing versus only 9.46 seconds of recorded reused computation.
- Optional Shapley-plan prototype remains disabled: profiling finds it accounts for little of the runtime, so introducing it offers no material benefit here.
- Durable line-flushed aggregate progress records explain failures even if the experiment worker is stopped. A completed diagnostic is not a benchmark result or evidence of fairness superiority.

## Required evidence

Retain original/optimized source hashes, registration and prepared-file hashes, environment/function identities, test logs, timing scope, per-step geometry/state hashes, resource use, final termination status and post-run checks of all registered sources. No individual records, model predictions or raw fit data in the report.
