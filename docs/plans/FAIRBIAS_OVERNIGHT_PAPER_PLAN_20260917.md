# FairBias overnight paper delivery — 2026-09-17

> Execution correction: first actual activation was 2026-09-17 08:31 Asia/Shanghai, approximately eight hours later than the first intended slot. The overnight work packages did not execute at the written local times. The remaining heartbeat was paused during morning delivery; the cadence below records original intent, not current scheduling. See `docs/reports/FAIRBIAS_MORNING_DELIVERY_20260917.md` and the latest STATUS. The user subsequently explicitly authorized immediate execution of the unstarted registered tasks, including LFR under original budgets with failures preserved.

## Authority and execution cadence

The user authorizes autonomous diagnosis, implementation, testing, remote NHIS experiments and supervisor stage acceptance, with the first applied paper as the priority. Their latest constraint overrides continuous work: conserve tokens, use scheduled batches, do not repeatedly monitor, idle, or keep a goal running. Do not restart a continuous goal during these scheduled batches. Read this file and the short STATUS file before looking up older history.

One thread heartbeat (`fairbias`) has three scheduled activations, Asia/Shanghai: **00:30, 04:30, 07:30 on September 17**, with three total occurrences. Each activation performs useful work and then ends. Long training runs independently on the server. No sleeping or repeated polling to wait for them. The 07:30 activation produces the morning report by approximately 08:00 even if experiments remain unfinished. Do not create duplicate automations or another task.

Read `docs/AI_EXECUTION_PROTOCOL.md`. Preserve preexisting working-tree changes and protected baseline files. No staging, commit, public upload, local real-data fitting or MEPS access. Worker subagents are explicitly authorized for concrete independent work; use Luna for bounded implementation where appropriate, then independently verify. Do not delegate merely to monitor or duplicate reading.

## First-pass context

- Local: `/Users/lkc/Downloads/code_v_0_3`; branch `research/nhis-fairbias`; HEAD `67e6659fa65249a8842e34af5d8969629efe4bca`.
- Remote: SSH alias `fairbias-autodl`, project `/root/autodl-tmp/fairbias`, Python `.venv311/bin/python`.
- Original run: `artifacts/nhis/benchmark_autodl_20260916_075038Z` on the server.
- Registration SHA256 `340cd60f6ab0d5919e2ac7bc22b62bdb3cc758faaa986650abdb3978b964732a`; source identity `818f01dbe1fcaa119493efa78acabf165961d1fdeaabb8e23f36a86dce7d5891`; 70 registered source files.
- Actual cgroup limits: 16 CPU equivalents, 80 GiB; never allocate against host totals (~192 cores / 1 TiB). Existing queue has 15 single-thread workers. Original dispatcher PID 48932 is a historical hint; verify current process identity, not PID existence alone.
- Data disk: 50 GB, 18 GB used at initial check. No need to fill RAM or use GPU for appearance. Leave resource reserve and consider native thread pools.
- Real microdata and individual predictions remain on the server. Transfer only aggregate evidence with SHA verification and temporary `.part` files. Diagnostic fitting uses F/C only.

## 00:30 batch — paper-critical diagnosis and execution

1. Read `docs/reports/fairbias_overnight_20260917/STATUS.md`. Take one compact remote status/failure snapshot. Verify registration, actual source identity and completion receipts. `jobs/*/receipt.json` binds output files but has no top-level fit status; `jobs/*/result.json` contains the status. Scheduler-session counts differ from total-run counts. Avoid dumping thousands of records.
2. Verify which registered families have not been attempted. In particular LFR (960 seed jobs) and FRAPPE (480) were absent from the initial snapshot. Inspect the approved execution scripts and runtime policies before launching them. Preserve complete attempts and failures; do not call an unsupported family a FairBias victory. The main paper is FairBias-BM; Joint/BM_AE are fixed-capacity mechanism ablations, not the whole study.
3. Investigate the 23 EG-DP/EO failures: `adapter decision output must lie in [0, 1]`. Determine using F/C or a synthetic witness whether these are tiny numerical overshoots of a legitimate probability mixture or a substantive invalid output. Do not blindly clip arbitrary values or substitute hard predictions. A repair requires a narrow tolerance contract, meaningful regression tests, source/version separation, and preservation of the old failed records. Do not edit source imported by running registered jobs.
4. Audit FairBias-BM failures and cached propagation. Initial counts: six direct nonfinite/empty intermediate geometry failures propagated to 126 dependent jobs; one direct MDS cap propagated to 21. Determine distinct representation roots and whether invalid geometry originates from a valid no-active-feature boundary, numerical failure, or a bug. Never turn undefined d_phi into zero. Fix only with an actual witness; other cases remain failed and reported.
5. Advance Joint recovery proportionately. Previous work accepted exact NumPy MDS acceleration, a separately labelled scheduled controller and one same-seed 2x-cap retry. Read `docs/reports/FAIRBIAS_JOINT_SCHEDULED_REVIEW_20260916.md` and its `verification.json`, not the entire old audit tree. New LR arm003 seed0 finished in 587s; GBDT returned a feasible budget-limited solution in 903s; arm002 recovered one unique cap state and returned a feasible budget-limited solution in 361s. None proves full-matrix success or superior prediction. If feasible, register a bounded paired F/C matrix across all 4 arms x 2 backbones x 5 seeds using an explicit algorithm version and budget; start independent server work and exit rather than wait. Do not select convenient seeds or mix pilot results into the original run. Do not sacrifice the core comparison completion for an open-ended Joint rewrite.
6. Preserve the existing statistical and data boundaries. Source/algorithm/selection changes must be fixed using F/C and execution evidence before inspecting comparative S metrics. S is for registered model selection; T is for frozen evaluation. Earlier historical 2024 results were seen, so describe retrospective temporal validation, not globally blind external validation. Do not use new S/T rankings to redesign methods.
7. Run only tests appropriate to actual changes; do not rerun already-passed broad suites without a reason. Independently review worker changes. Save code identities, meaningful test output and a compact decision. Keep source variants and complete original receipts separable.

Useful entry points: `scripts/run_nhis_benchmark_parallel.py`, `scripts/handoff_nhis_parallel.py`, `scripts/check_frappe_runtime.py`, `src/nhis_fairbias/benchmark/experiment_selection.py`, `experiment_evaluation.py`, `predictions.py`, `adapters/adapter_reductions.py`, `adapters/adapter_fairbias.py`.

## 04:30 batch — one stage check

Read STATUS and take one snapshot. Verify newly completed evidence; start the next previously specified stage only when ready. Fix a newly actionable failure with a bounded test/repro. If jobs are healthy and still running, update STATUS and end immediately. No unchanged progress chatter. Do not expand into new models or literature exploration overnight.

## 07:30 batch — deliver even if incomplete

Write `docs/reports/FAIRBIAS_MORNING_DELIVERY_20260917.md` with:

- An exact timestamp, registered/attempted/valid/failed/pending counts by family and arm where useful; distinguish original and recovery variants.
- What changed, why, actual validation and remote runs; clearly state unresolved errors and unchanged failures.
- Any results admissible under the frozen selection and evaluation plan, their F/C/S/T roles, prediction versus decision probability, weighted versus unweighted metrics, complete seed support and survey estimability. Do not compare only successful seeds or label diagnostic geometry clinical fairness.
- The shortest remaining path to a first applied paper: completing declared comparators, fixed selection, frozen 2024 evaluation, survey intervals and paired differences, figures, results narrative and limitations. Existing inference guidance is in `docs/plans/FAIRBIAS_BENCHMARK_REGISTERED_EXTENSIONS_20260916.md` and `docs/reports/FAIRBIAS_CODEX_INTEGRATION_REVIEW_20260916.md`.
- A concise follow-on priority list with observed runtime/resource estimates where available. No claim of FairBias superiority, paper readiness or guaranteed acceptance without the relevant results.

If the full selection/evaluation gate is not satisfied, produce a useful development/validity report rather than force T evaluation. Separate future ideas in `docs/plans/FAIRBIAS_FUTURE_RESEARCH_BACKLOG_20260917.md`. Send the Chinese summary and report link in the current task. Finish the final scheduled activation without further recurring checks.

## Handoff and cost discipline

After each activation update `docs/reports/fairbias_overnight_20260917/STATUS.md` with concise current evidence, changed file names, active remote PID/run/log identities and the next executable step. Report failed checks honestly. Never rewrite frozen run evidence. Avoid full-history reads, repeated broad searches, duplicate tests, polling loops and idle agent waits.

The agent created a continuous goal before the user's cost-saving correction. The tools do not expose goal pause/cancel, and Computer Use explicitly prohibits operating the Codex app itself. The user must pause that goal using the UI if it remains active. Do not mark the unfinished research goal complete to silence it. The scheduled heartbeat is separate and already configured.
