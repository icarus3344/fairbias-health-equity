# Independent Review — A0–A2 Repair Tranche 4 (V3)

**Review date:** 2026-09-19  
**Reviewer role:** independent gate reviewer; no production analysis authority  
**Review scope:** synthetic contracts and their bound verification artifacts only  
**Final status:** `INDEPENDENT_REVIEW_TRANCHE_4_COMPLETE__A2_ACCEPTED_FOR_SYNTHETIC_CONTRACTS__A3_NO_GO__2025_LOCKED`

## 1. Verdict

**A2 is restored to `ACCEPTED_FOR_SYNTHETIC_CONTRACTS`.** Tranche 4 closes the two P1 failures in the preceding independent review and rejects every requested identity, universe-completeness, and receipt-tampering counterexample. I found no additional reproducible gate-critical defect in the reviewed synthetic-contract scope that would incorrectly authorize A3.

This is a deliberately narrow acceptance. It does **not** approve scientific registry contents, the production survey-inference implementation, a real 2023 performance scan, A3, A4, the paper's empirical claims, or any 2025 access. **A3 remains NO-GO, and 2025 remains locked.** The current registry and checklist correctly retain review/freeze blockers; this memo does not edit them or silently convert their state.

## 2. Materials and independent checks

I read the complete tranche-4 report and verification receipt, current V3 module, both V3 synthetic test files, pre-freeze registry, interface/family specification, freeze checklist, and the preceding independent review. I did not read any real outcome/performance artifact or any 2025 microdata.

### 2.1 Test replay

The exact six-file synthetic/benchmark suite completed independently:

```text
100 passed, 3 warnings in 2.13s
```

The three warnings are the already visible numerical warnings in `survey_batch.py:60` (divide by zero, overflow, and invalid matrix multiplication); they are not evidence of a failing assertion in the reviewed contract suite, but must not be treated as validation of production survey inference.

A targeted replay of the tranche-4 identity/universe cases completed:

```text
10 passed, 60 deselected in 1.64s
```

A separate targeted check for 2025-lock and repair-receipt/supersession coverage completed:

```text
3 passed, 74 deselected in 1.42s
```

`git diff --check` reported no whitespace errors.

### 2.2 Independent adversarial replay

I also reconstructed the counterexamples directly in memory, independently of merely counting named tests. The current verifiers produced these fail-closed outcomes:

| Counterexample | Result | Independent rejection evidence |
|---|---|---|
| `audit_role=expanded` row inserted into an anchor family | **CLOSED** | Rejected because scope attributes do not exactly match the frozen contrast universe. |
| Same attribute placed in both anchor and new-O sets | **CLOSED** | Rejected because the new-O row identity contradicts the frozen contrast universe; the universe builder also requires disjoint sets. |
| Same O relabelled with a different group rule across endpoints | **CLOSED** | Rejected because the altered slot is outside the frozen contrast universe. |
| One required contrast-universe slot omitted | **CLOSED** | Rejected because anchor-native rows omit or add frozen-universe slots. |
| One unregistered slot added | **CLOSED** | Rejected by the same exact-slot equality check. |
| Receipt contents changed and its self-hash recomputed | **CLOSED** | Rejected because the attribute records no longer match the declared O sets. |

These checks cover the two previous P1 mechanisms: role/partition relabelling and a self-consistent but incomplete or enlarged family universe. Both are now closed in the reviewed code path.

### 2.3 Hash and receipt verification

I independently recalculated the receipt's declared input/output hashes. All matched the live files reviewed. In particular:

- module: `7d7961bb2697489ebbd5750d0426e1354fa034e232d38bdd11a970453703c4ea`
- main V3 synthetic test: `c6beac8a74be0aec6abbbbe0e6fa2375122e35ae285f5936495e419a071bc64f`
- V3 registry test: `f739202788f4e98befb0407c4c8a74dad2128db20d299f025855c87b88caa4b7`
- pre-freeze registry: `ec0dd67fcc602b6337d425a742d1e9abf3d5b9245fc77ca38b6bffbfd00c7bdb`
- interface/family specification: `eea366962b7f9f6322c02de9c423bdb6970df6a10c72b95692dcbdd596cd03f6`
- freeze checklist: `a6a470f61f18c82b38a69ca1688fba5fbc9648aa76665c90ac7e597208f7d0dd`
- frozen comparison-panel draft hash: `2a63adc9e43829af676570553d8dbedc7a9209ca59a832089e77479bfe6a8d0f`
- tranche-4 report: `b2c80712a641d9dd5a4d9bdea9a365ad93c96a36644755b553ffcd94e1714b4a`
- tranche-4 verification receipt self-hash: `46a24a950f868b87132d95e96b4f2a6a98211c9fadc8b68b337a282ca3fa703d`

The earlier planning receipt remains explicitly superseded and is not current execution authority. The current report terminates with the stated review stop rather than claiming self-approval.

## 3. Contract-level findings

### 3.1 Frozen contrast universe — acceptable for the synthetic gate

The universe receipt now fixes and re-verifies:

- the ordered anchor set and the sorted, unique, disjoint new-O set;
- the O-registry version/hash and selection-rule identity;
- the panel and family identities;
- per-attribute audit role, group-rule hash, role hash, registry identity, and expected contrasts;
- exact and unique universe contrast IDs.

The verifier reconstructs these relationships instead of trusting a receipt that is merely self-consistent. A hollow, missing, enlarged, or role-swapped universe therefore fails before Q2 aggregation.

### 3.2 Family and Q2 binding — acceptable for the synthetic gate

Family generation and verification now bind the embedded universe receipt and exact expected slots, including their allowed roles. The Q2 path binds the family/question, endpoint, model or method pair, panel, delta identity fields, audit role and role hash, group-rule identity, O-registry identity, seed policy, and contrast-universe hash. It requires the full exact slot set for the selected family scope. The no-new-O branch is separately tied to the same empty-new-O universe and provenance rather than being an unbound shortcut.

These protections are sufficient to accept the isolated synthetic contracts. They do not establish that the eventual production registry contains scientifically appropriate contrasts or that a production result table was derived correctly.

### 3.3 Registry and checklist truthfulness

The live registry remains `DRAFT_NOT_FROZEN`; real performance scanning is false; the comparison panel is still technically bound but pending member-contract and coverage preflight; and A3 remains blocked pending A0–A2 acceptance plus scientific freeze. All inspected 2025 freeze, verification, recording, and micro-performance-read flags remain false.

The checklist reports 52 items, with 32 complete and 20 incomplete, and does not claim A1 freeze, A3 authorization, or 2025 release. Its pending independent-review item may be updated in a later Worker tranche to cite this memo, but this reviewer did not alter that production artifact.

## 4. Remaining A3 blockers

Restoring A2 synthetic acceptance does not satisfy the A3 entry gate. The following must be resolved and independently recorded before any authentic performance scan.

### 4.1 Human/scientific facts

1. Freeze the application/use context that justifies each endpoint-specific tolerance `delta`; record the value, units, rationale, approver, and approval time without consulting outcomes.
2. Approve the formal new-O set, group definitions, reference groups, nesting/correlation handling, estimand language, and allowed/non-allowed claims.
3. Approve the small set of confirmatory method-versus-matched-baseline comparisons; leave the larger migration matrix descriptive unless separately registered.
4. Record ethics, authorship, scientific-responsibility, data-access, and any restricted-data facts. Later decide whether A4 is warranted from the A3.5 paper package; A4 remains optional.

### 4.2 Production and provenance controls

1. Complete the 2022/2023 official-codebook harmonization and codebook-wide eligibility ledger for candidate O variables.
2. Freeze every main-panel member's model, source, feature/preprocessing, threshold, prediction, hard-label, record-order, and record-key contract and hashes; complete blind non-performance coverage preflight on the common domain. A later low-coverage method must enter an additional panel and must not shrink the main panel.
3. Freeze the training-randomness/seed aggregation estimand and demonstrate it in the actual model-production path, not only in synthetic rows.
4. Freeze a production inferential registry/run manifest that hash-binds endpoint-specific **delta value, units, and rationale**, not only a rationale identifier, and that derives the three-state conclusion from the validated interval rather than trusting a supplied conclusion label.
5. Validate the chosen production simultaneous-CI or resampling implementation, including coverage/reference tests and correct treatment of the maximum EO gap. Ordinary CIs and Holm-adjusted p-values must remain distinct from simultaneous intervals.
6. Freeze and hash the actual contrast universe, statistical families, source files, survey design variables, support/ESS/PSU rules, precision/MDE outputs, and output schema. Verify record order and prediction alignment before inference.
7. Obtain a separate independent A3 authorization after the above artifacts are complete. The 2024 run, if later authorized, remains retrospective and rule-bound; it cannot repair a failed prospective freeze.

## 5. Gate decision

| Gate or action | Decision | Scope |
|---|---|---|
| A2 synthetic contracts | **GO — `ACCEPTED_FOR_SYNTHETIC_CONTRACTS`** | Code-level synthetic identity, universe, receipt, family, Q2 aggregation, alignment/hash, and lock contracts only. |
| Scientific A1 freeze | **NO-GO** | Human facts and final inferential registry remain incomplete. |
| Authentic 2023 performance scan / A3 | **NO-GO** | Production provenance, panel, survey-inference, simultaneous-CI, delta, and authorization blockers remain. |
| Retrospective 2024 evaluation | **NO-GO** | Cannot precede the declared A3 freeze and authorization sequence. |
| 2025 microdata/outcome/performance access | **LOCKED / NO-GO** | No release condition is satisfied. |
| A4 or paper-readiness claim | **NO-GO** | Requires A3 evidence, A3.5 claim-evidence package, and a separate scientific decision. |

The permissible next step is to update the planning artifacts to record this narrow independent A2 acceptance while continuing only the non-performance scientific and production freeze work listed above. This review must not be cited as permission to inspect 2023 outcomes, to unlock 2025, or to claim that the paper's empirical thesis has been established.
