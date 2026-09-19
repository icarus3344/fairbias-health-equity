# FairBias live progress check — 2026-09-16 08:50 CST

Supervisor read-only inspection of the running registered study, plus this additive status note. No registered source, budget, result or selection was changed.

Run: `artifacts/nhis/benchmark_codex_20260916_014500/`.

## Verified state

- 1,341 / 8,054 seed jobs have receipts: 1,021 VALID and 320 BUDGET_EXHAUSTED. This is 16.65% of attempted-job coverage, not elapsed-runtime coverage or paper readiness.
- All completed jobs so far belong to arm_001. VALID counts: FairBias BM 640; unmitigated 48; reweighing 24; EG_DP 192; EG_EO 97; TO_EO 4; OxonFair EO 16. Recent-method formal jobs and later arms remain pending.
- Main development PID 19210, completion controller PID 23343 and its idle-sleep inhibitor are alive. A current EG_EO/GBDT worker uses approximately one CPU core and 450 MiB RSS. Development has run about 7 hours 10 minutes.
- The completion admission verified all 85 bound files. Neither `selection_freeze.json` nor `evaluation_T/` exists; the current run has not begun 2024 model evaluation.

## LFR convergence incident

All 320 excluded seed jobs are LFR: 40 original representation fits and 280 downstream configurations excluded through the representation cache. All 40 original diagnostics report `STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT`, with 5,229–6,966 function calls, only 3–7 optimizer iterations, and training_n 21,869. The original fits consumed about 29.2 minutes total; this is not 320 independently timed-out heavy fits.

The registered adapter sets maxiter=5000 and maxfun=5000. The optimizer's iteration and function-evaluation limits have different meanings. Existing evidence establishes lack of convergence within the registered function-evaluation budget; it does not establish inferior LFR fairness or prediction. Formal full-data convergence admission was insufficient for this method. Strict rejection of unconverged outputs is functioning as intended.

`experiment_selection.py` deliberately marks an all-failed method condition NOT_ESTIMABLE and permits other completed methods to advance. The current completion controller has not been stopped. Thus its eventual completion would not, by itself, establish a usable LFR comparison. A separate training-only feasibility/budget assessment and explicitly versioned amendment would be needed for a recovered comparison; never overwrite existing failures, silently change this run's budget, relax convergence based on performance, or count absence as a FairBias win.

## Runtime estimate and limitations

Remaining GBDT jobs in EG_DP / EG_EO: 480 / 575. Observed mean runtimes in arm_001: 90.46 / 62.21 seconds. A simple same-cost extrapolation gives roughly 22 additional hours for those two subsets alone. Later arms may differ substantially, and recent methods, FairBias ablations and final survey inference add further work. Earlier expectations of finishing within one night were optimistic. Job-count percentages cannot be converted directly to time because representation-cache hits are cheap.

No S performance values or new T data were inspected for this progress check. No superiority conclusion is available.
