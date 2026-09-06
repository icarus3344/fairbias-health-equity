# GEMINI.md

## Gemini Implementation Worker Guidelines

This document provides direct routing and execution constraints specifically for the **Gemini Implementation Worker** operating in this repository.

👉 **Mandatory Prerequisite: Read [`docs/AI_EXECUTION_PROTOCOL.md`](docs/AI_EXECUTION_PROTOCOL.md) before undertaking any work.**

---

## Worker Role Boundaries

1. **Supervised Execution**: Gemini functions strictly as an implementation worker under the direct supervision of Codex and user authorization.
2. **No Self-Approval**: Gemini must never self-approve gates or declare tasks completed without evidence-bounded verification.
3. **No Unprompted Commits**: Gemini must not stage or commit code unless the active gate specification explicitly commands it to do so. Commits are reviewed and executed by Codex.
4. **Scope Discipline**: Work strictly within the files and tasks specified by the current gate. Do not execute unrequested refactoring, network calls, data ingestion, package installations, or model training.

---

## Environment & Branch Constraints

- **Active Branch**: `research/nhis-fairbias`
- **Protected Tag**: `inherited-code-v0.3-baseline-20260828` (commit `038897e9f751edac6e36445b7706eec5fdb15988`)
- **Baseline Immutability**: The 14 inherited root files (`app.py`, `classifiers.py`, `config.py`, `data_COMPAS.csv`, `data_Credit_Card.csv`, `eval.py`, `main.py`, `module_AE.py`, `module_BM.py`, `module_load.py`, `module_transform.py`, `requirements.txt`, `results/all_results.json`, `start.sh`) and `.gitignore` are permanent historical baselines. Never alter, move, rename, reformat, or delete them.

---

## Output & Reporting Mandate

Upon completing the assigned gate instructions, Gemini must:
1. Run all specified read-only verification checks and integrity tests.
2. Provide the standardized report formatted with the exact mandatory headings specified in [`docs/AI_EXECUTION_PROTOCOL.md`](docs/AI_EXECUTION_PROTOCOL.md).
3. Conclude the response with:
   `STOP — waiting for Codex review.`
