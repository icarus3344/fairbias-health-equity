# FairBias completion queue: parallelism increase, 2026-09-17

Supervisor decision: resource handoff accepted. Training remains in progress; this does not accept the full method family or its predictive/fairness performance.

## Cause and resulting behavior

The server provides 64 CPU cores and 120 GiB of cgroup memory. The previous controller held the queue at ten single-thread F/C pilot jobs, while reserving another seventy registered jobs. Ten busy cores account for approximately 16% utilization. The requested resource cap of 56 did not override this pilot-stage gate.

The user explicitly requested higher utilization. The supervisor authorized early parallel computation of the remaining registered jobs, with unchanged algorithm, inputs, parameters, seeds, and final acceptance requirements. This relaxes the scheduling dependency on completion of every pilot; it can incur redundant computation if a common defect is subsequently found. It does not constitute early scientific acceptance.

Controller PID 2077 was replaced by PID 2303. Only the controller was stopped. Three original jobs had completed before handoff; the other seven retained their original PID and process start time. The replacement adopted those seven processes and launched 49 previously unstarted jobs. There were no duplicate job identities in the verified snapshot.

At **22:39:08 Asia/Shanghai**, the bounded measurement found:

| Measure | Verified value |
| --- | ---: |
| Active jobs / actual live worker processes | 56 / 56 |
| Completed F/C jobs inherited from the original controller | 3 |
| Pending jobs | 21 |
| Registered total | 80 |
| CPU utilization, 5-second interval relative to 64-core quota | 87.26% |
| Cgroup memory | 13.20 GiB / 120 GiB |
| Free data disk | 48.84 GiB |

This demonstrates higher resource utilization, not a measured end-to-end speedup. Long individual fits can still determine final completion time. Memory is not deliberately filled; the current tasks do not require the remaining capacity.

## Ownership and reproducibility

- Frozen runtime: `/root/autodl-tmp/fairbias_completion_v2_20260917`.
- New controller and live state: `/root/autodl-tmp/fairbias_completion_parallel_20260917/{run_nhis_completion_takeover.py,state.json}`.
- New outputs: `/root/autodl-tmp/fairbias_completion_parallel_20260917/jobs/<job_id>`.
- Original ten outputs remain in the frozen runtime's `runs/full80_v2/<job_id>`. Its old `state.json` is historical, not the current scheduler state.
- Final queue receipt will be in the new controller directory as `queue_receipt.json`.
- Runtime manifest seal remains `6607256f5efa027e0c954e7e0a0021d886074bf6dbfde0a52f1e5c5ba69ba0f0`. The 99 bound runtime files were not edited.
- The new external controller is separately bound by SHA-256 `8e35a9cc340a1f37531f1a0627246eabe1d5cd6f29b5e3b7f17b5dc5f85ad791`.

The controller maintains a total concurrency limit of 56, pauses new dispatch above 80% cgroup memory or below 5 GiB free disk, and has no wall-clock training timeout. Existing fits continue independently of the Mac or SSH connection. The app automation remains paused; the server controller itself advances the finite queue.

For inherited processes, the replacement cannot observe the original parent-child exit code. It records that value as unknown, checks bound artifacts and model hashes, and requires subsequent supervisor acceptance; it does not invent a successful exit code. New child processes require an observed zero exit code. All completed jobs still require independent full-family provenance and reload review before new selection/evaluation. No S/T performance was used for this change.

## Verification and handoff

New controller tests: **8 passed locally (0.12 s)** and **8 passed on Linux (0.40 s)**. Coverage includes disjoint ownership, duplicate rejection, PID reuse/zombie handling, truthful inherited exit-code semantics, artifact identity and model hash checks, and a synthetic full 80-job adoption/dispatch cycle.

Local evidence: `artifacts/nhis/full_method_completion_20260917/westb_35430/parallel_control/`. `efficiency_change_verified.json` independently checks the downloaded handoff hash, source hashes, seven inherited process identities, 56 unique active jobs/PIDs, unchanged completed set, and 80-job accounting. `handoff.json`, `transition_receipt.json`, and the before/after snapshots preserve the transition.

Next action is a bounded check of the **new** controller state or final receipt. Do not dispatch duplicate jobs, mutate frozen sources, or read the old state as live. On completion, independently audit all 80 job identities, input/source/model/result hashes and reload evidence. These 80 fixed anchors do not close the outstanding BM root failures or automatically cover an expanded tuned Joint grid. Preserve the historical study and failures; formal evaluation remains a separately frozen step.

No Git staging or commit was performed. Protected inherited files were not changed.
