# Decision 0005: Round 4 Commit Boundary Audit (4565487)

Date: 2026-08-30
Status: Accepted (retroactive authorization recorded)

## Context

The Round 4 worker report (`docs/reports/REWORK_FAIRBIAS_ROUND4_TERMINAL_STATE_SEPARATION.md`)
was issued with `Status: COMPLETE` against `HEAD=f57fe84` and explicitly stated
"worker 未执行任何 git commit" — all round-4 changes were uncommitted workspace
modifications at report time.

After the report was issued, commit `4565487` ("feat(fairbias): 分离终态并修复终止语义")
appeared on `research/meps-hc252-longitudinal` containing that round-4 work. No
contemporaneous Codex Accept record for that commit exists in the repository.

## Facts Established by the Codex Round-4 Review (2026-08-30)

The Codex supervisor reviewed the content of `4565487` and established:

1. The terminal-state separation and termination-semantics fixes are correct
   (greedy terminal state and validation Pareto checkpoint separated;
   `converged=false` recorded for Credit budget exhaustion).
2. COMPAS/Credit artifact hashes match the report.
3. 27 targeted tests not involving MEPS training were independently re-run
   and all passed.
4. The 14 frozen inherited root files and `.gitignore` are unchanged in
   `4565487`.
5. Four defects were identified (REPAIR verdict, not Reject):
   - P0: the golden fixture `tests/fixtures/official_uciadult/distance_matrix_step_0.csv`
     was NOT included in `4565487` (only `PROVENANCE.json` was tracked), so a
     clean checkout fails the MDS golden tests with `FileNotFoundError`.
   - P1: the `paper_strict` state name overclaimed paper alignment (automatic
     MDS dim=3 vs official fixed dim=2; six-value power grid vs the official
     interleaved stream; finite iteration budget).
   - P1: `configs/study.json` lists mitigation arms not implemented as claimed
     (deferred to the MEPS gates gate, out of round-4.1 scope).
   - P2: hardcoded event count `136` in the MEPS pipeline lock-state string
     (deferred to the MEPS gates gate, out of round-4.1 scope).

## Decision

1. Commit `4565487` is retroactively recognized as the round-4 delivery
   boundary: its content has been audited by the Codex review above (facts
   1–4) and its known defects (fact 5) are exactly the round-4.1 REPAIR
   scope. No rollback is required.
2. Round 4.1 (this gate) is authorized to commit directly upon completion
   of the repair and clean-checkout verification, per the supervisor's
   explicit instruction accompanying the REPAIR verdict.
3. Rule going forward: every gate commit must reference the dated Codex
   verdict (Accept or Repair-with-authorization) that authorizes it, either
   in the commit message or in a decision record like this one. A commit
   without a contemporaneous authorization record is an audit exception
   that must be documented here.

## Consequences

- The commit-history gap between the round-4 report and `4565487` is now
  documented rather than silent.
- Round 4.1 adds the missing fixture, replaces `paper_strict` with the two
  explicit algorithm modes (subsequently renamed across the Round 4.1
  repair rounds: the official-derived mode is now
  `official_code_derived_monotone_cursor_unweighted` — an official-code-
  derived variant with a termination-safety extension, not an
  official-code equivalence claim — alongside `engineering_bounded`;
  see the Round 4.1 repair reports), and verifies a clean checkout (see
  `docs/reports/REWORK_FAIRBIAS_ROUND4_1_MODE_SEPARATION.md`).
