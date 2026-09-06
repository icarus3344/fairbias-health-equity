# AGENTS.md

## Agent Routing & Operational Protocol

This repository operates under a strict gated AI supervision model. All AI agents (including supervisors, planners, and implementation workers) must adhere to the rules defined in the canonical protocol:

👉 **Read [`docs/AI_EXECUTION_PROTOCOL.md`](docs/AI_EXECUTION_PROTOCOL.md) before performing any action.**

---

## Role Boundaries & Governance

- **Codex Supervisor**: Holds supervisory authority over task gating, requirements validation, code review, test result verification, and Git commit authorization.
- **Worker Agents**: Subordinate implementation agents (e.g., Gemini). Responsible for executing the exact tasks defined in the active gate specification, running verification tests, and submitting standard audit reports. Worker agents must never self-approve gates and must not commit without explicit supervisor instruction.

---

## Branch & Baseline Integrity

- **Active Research Branch**: `research/nhis-fairbias`
- **Protected Baseline Tag**: `inherited-code-v0.3-baseline-20260828` (resolving to commit `038897e9f751edac6e36445b7706eec5fdb15988`)
- **Immutability Rule**: The 14 inherited root files and `.gitignore` are frozen and must never be modified, moved, renamed, reformatted, or deleted. All new work is strictly additive.

---

## Reporting Requirement

At the conclusion of each gate, worker agents must produce a complete, evidence-bounded report using the exact headings mandated in Section 9 of [`docs/AI_EXECUTION_PROTOCOL.md`](docs/AI_EXECUTION_PROTOCOL.md), concluding with the explicit line:
`STOP — waiting for Codex review.`
