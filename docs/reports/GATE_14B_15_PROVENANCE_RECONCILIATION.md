# Gate 14B / Gate 15 Provenance Reconciliation Report

## 1. Executive Summary and Scope of Acceptance

- **Active Branch**: `research/meps-hc252-longitudinal`
- **Prior Commit SHA**: `3ad9e1c53d8cccb6d2b2097f033c9ceae19d2314` (`wip(meps): sync Gate 14B and Gate 15 preflight`)
- **Supervisor Verdict**: `GATE 14B SUPERVISOR VERDICT: ACCEPT`
- **Accepted Scope**: The supervisor's acceptance applies strictly and exclusively to the pure-synthetic multi-panel implementation artifacts for Gate 14B contained in commit `3ad9e1c53d8cccb6d2b2097f033c9ceae19d2314` (`src/meps_fairness/multi_panel.py`, `tests/test_gate14b_multi_panel_synthetic.py`, and `docs/reports/GATE_14B_SYNTHETIC_IMPLEMENTATION.md`).
- **Explicit Exclusions**: Commit `3ad9e1c53d8cccb6d2b2097f033c9ceae19d2314` as a whole is **NOT** supervisor-approved. Gate 15 is **NOT** accepted by this verdict. No authorization is granted for HC-217 data, schema, codebook, documentation download, outcome inspection, microdata rows, or Panel 27 access.

## 2. Commit Provenance and Historical Report Reconciliation

- Commit `3ad9e1c53d8cccb6d2b2097f033c9ceae19d2314` existed before independent Gate 14B review and acceptance, bundling the pure-synthetic Gate 14B implementation with the Gate 15 preflight WIP contracts.
- As a consequence of this sequencing, the historical worker reports (`docs/reports/GATE_14B_SYNTHETIC_IMPLEMENTATION.md` and `docs/reports/GATE_15_HC217_SOURCE_PREFLIGHT.md`) contain stale commit-state statements describing their files as unstaged/uncommitted at the time of writing.
- In accordance with repository audit and provenance rules, those historical reports are preserved as immutable snapshots rather than rewritten.
- The supervisor independently audited and reviewed the exact Gate 14B pure-synthetic software contracts after the WIP commit `3ad9e1c53d8cccb6d2b2097f033c9ceae19d2314`, confirming 29/29 passing unit tests and complete compliance with Decision 0007.
- Gate 14B acceptance is strictly limited to pure-synthetic software-contract evidence.
- No Git rollback, reset, rebase, history rewrite, or force push was performed. The repository history remains linear and additive.

## 3. State-Machine Repair: Decoupling Gate 14B Fact from Gate 15 Authority

- **Identified Defect**: The initial Gate 15 artifact permission validator (`validate_gate15_artifact_permission` in `src/meps_fairness/gate15_source_review.py`) incorrectly coupled `permission_status = PENDING_SUPERVISOR_AUTHORIZATION` with `gate14b_prerequisite_status = PENDING_CODEX_ACCEPTANCE`. This prevented recording the factual completion of Gate 14B (`ACCEPTED_AND_COMMITTED`) while Gate 15 remains pending authorization (`PENDING_SUPERVISOR_AUTHORIZATION`).
- **Remediation**: `validate_gate15_artifact_permission` is repaired to decouple these states:
  1. `permission_status = PENDING_SUPERVISOR_AUTHORIZATION` may now coexist with either `gate14b_prerequisite_status = PENDING_CODEX_ACCEPTANCE` or `gate14b_prerequisite_status = ACCEPTED_AND_COMMITTED`.
  2. Recording `gate14b_prerequisite_status = ACCEPTED_AND_COMMITTED` does not authorize Gate 15 or schema access: `active_stage` MUST remain `SOURCE_METADATA_PREFLIGHT_ONLY`, `schema_stage_authorized` MUST remain `false`, schema stage `enabled` and `local_artifact_read_allowed` MUST remain `false`.
  3. `permission_status = SUPERVISOR_AUTHORIZED` continues to strictly require `gate14b_prerequisite_status = ACCEPTED_AND_COMMITTED`, `active_stage = SCHEMA_CODEBOOK_ONLY`, and explicit schema-stage authorization.
  4. Any attempt to authorize Gate 15 while Gate 14B remains `PENDING_CODEX_ACCEPTANCE` fails closed with `GATE15_PREREQUISITE_NOT_ACCEPTED`.

## 4. Configuration Update

- In `configs/gate15_hc217_artifact_permissions.json`, the prerequisite fact is updated:
  ```json
  "gate14b_prerequisite_status": "ACCEPTED_AND_COMMITTED"
  ```
- All other fields remain strictly unchanged:
  - `permission_status = PENDING_SUPERVISOR_AUTHORIZATION`
  - `active_stage = SOURCE_METADATA_PREFLIGHT_ONLY`
  - `schema_stage_authorized = false`
  - `outcome_values_read = false`
  - `microdata_rows_read = false`
  - `panel_27_accessed = false`
  - `stages.SCHEMA_CODEBOOK_ONLY.enabled = false`
  - `stages.SCHEMA_CODEBOOK_ONLY.local_artifact_read_allowed = false`

## 5. Security and Access Boundary Audit

- **HC-217 Download Authorization**: False / Unauthorized.
- **HC-217 Schema Access**: False / Unauthorized.
- **Worker/local HC-217 Documentation or Codebook Artifact Read During This Repair**: False. No HC-217 documentation or codebook artifact was opened or saved locally by the worker during this repair.
- **Prior Supervisor Online Source Verification**: Previously disclosed in `docs/reports/GATE_15_HC217_SOURCE_PREFLIGHT.md`. The supervisor read the official HC-217 documentation PDF online as text for source verification only; it was not saved locally and did not access microdata, outcome values, or Panel 27.
- **MEPS Microdata Rows Read**: 0 (False / Prohibited).
- **Outcome Values Read**: False / Prohibited.
- **Panel 27 Holdout Accessed**: False / Locked.
- **`configs/data_access.json` Modified**: False / Unmodified.
- **Untracked Directories**: `archive/baseline_v0.3/` remains a pre-existing untracked directory and is untouched.
- **Baseline Immutability**: All 14 root baseline files and `.gitignore` remain untouched.
