# Synthetic Statistics Resource Probe — 2026-09-17

Gate: A3 statistical-stage resource measurement

Status: COMPLETE — synthetic-only bounded measurement; no real NHIS data, models, S/T predictions, fitting, source edits, or scheduling.

Files changed:

- This report.
- `FAIRBIAS_SYNTHETIC_STATISTICS_RESOURCE_PROBE_20260917.json` metadata evidence.

Commands executed: A remote scratch probe was run once on `fairbias-cpu` with `nice -n 10`, single-process Python, after a smaller interface smoke measurement. The formal run used N=30,000, B=2,000, 600 strata, 1,200 PSUs, seven groups, and 100 generated finite q vectors.

Permissions requested: None.

Tests executed: Interface smoke measurement followed by one formal synthetic resource measurement using `linearized_survey_inference` and `bootstrap_metric_arrays`.

Exact test results:

- Smoke, N=2,400 / B=20 / 3 models: linearized 0.2629 s; bootstrap 0.1547 s; peak RSS 150,097,920 bytes; COMPLETE.
- Formal, N=30,000 / B=2,000 / 100 models: linearized 19.9064 s; bootstrap 14.7761 s; peak RSS 289,558,528 bytes; COMPLETE.
- The formal process finished within the 120-second CPU bound. No individual-level arrays or metric values were written to the report.

Input hashes:

- `survey_batch.py`: `907225f0ac5083180303501571ff350e8c67f5378a30dc2da6decd7258ac68dd`
- `survey_linearization.py`: `f8270032d26f03e5eca4aa84e8813e1b11a15a18922534c9afaa64b3232ab925`
- Remote hashes matched the local source hashes exactly.

Output hashes:

- Scratch probe script hash: `886ec6d62222c3ce095526302cab8d7c1bf6775211fe2a0a24e4e47bcad8971c`
- Aggregate metadata is recorded in the accompanying JSON file.

Row counts: Formal synthetic input contained 30,000 rows, 600 strata, 1,200 PSUs, seven groups, 100 frozen q vectors, and 2,000 shared bootstrap replicates.

Assumptions: Synthetic rows used finite positive weights, complete two-PSU strata, all seven groups supported, and generated bounded q values. The measurement describes this implementation and hardware/runtime environment only; it is not a real T ETA or a claim about NHIS production memory behavior.

Unresolved issues: The probe does not measure model loading, preprocessing, I/O, worker orchestration, or real annual-domain sparsity. A production run should retain its own resource receipt.

Git diff summary: Additive report and metadata only; no source, environment, training, or scheduler files changed.

Proposed next step: Use approximately 20 seconds linearization and 15 seconds bootstrap as a synthetic planning reference for this stated workload, then independently measure any production arm under its own resource gate.

STOP — waiting for Codex review.
