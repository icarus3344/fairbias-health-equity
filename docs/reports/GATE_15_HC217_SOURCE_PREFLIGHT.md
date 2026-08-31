# Gate 15: HC-217 Source Metadata Preflight

Gate 15 remains `SOURCE_METADATA_PREFLIGHT_ONLY`.  The current repair does
not authorize a local HC-217 archive, documentation, codebook, schema, or
microdata read, and it does not modify `configs/data_access.json`.

## What is fixed in the preflight contract

- HC-217 is locked to Panel 23 and the exact 2018–2019 period.
- The details page and all three listed artifact URLs must match the exact
  AHRQ endpoints recorded in the source metadata.
- The separate artifact permission contract names exactly the Stata archive,
  documentation PDF, and codebook PDF.  It maps Gate 15's `archive` and
  `codebook` labels to the downloader's `data_archives` and `codebooks`
  labels, and requires the Stata ZIP to allow exactly `.dta` members.
- The permission is currently
  `PENDING_SUPERVISOR_AUTHORIZATION`; a broad PUF entry alone is not enough.
  The scoped downloader rejects this pending permission before opening a
  network connection.
- Its explicit Gate 14B prerequisite is currently
  `PENDING_CODEX_ACCEPTANCE`; a future Gate 15 authorization must record
  `ACCEPTED_AND_COMMITTED` after independent Gate 14B review.
- `structural_checks_complete` is false until the semantic review is complete,
  even if a separately authorized synthetic record supplies schema hashes and
  column metadata.

## Evidence boundary

Worker-side Gate 15 evidence remains metadata-only: no worker archive or PDF
was opened locally, no local PDF was saved, no MEPS row or outcome value was
read, and Panel 27 was not accessed.  Separately and transparently, the
supervisor's independent check opened and read the official
[HC-217 documentation PDF](https://meps.ahrq.gov/mepsweb/data_stats/download_data/pufs/h217/h217doc.pdf)
as online text for source verification; it was not saved locally and did not
access microdata, outcomes, or Panel 27.

The three HEAD metadata values already recorded in the source JSON remain
metadata evidence only.  Their SHA-256 fields remain null because no
authorized local download has occurred.

## Section 9 Standard Worker Report

Gate: Gate 15 — HC-217 source metadata preflight repair
Status: SOURCE_METADATA_PREFLIGHT_ONLY; STOPPED_PENDING_EXACT_HC217_AUTHORIZATION
Files changed:
- configs/gate15_hc217_source_review.json
- configs/gate15_hc217_artifact_permissions.json
- docs/reports/GATE_15_HC217_SOURCE_PREFLIGHT.md
- src/meps_fairness/data/__init__.py
- src/meps_fairness/data/download.py
- src/meps_fairness/gate15_source_review.py
- tests/test_gate15_source_review.py
- tests/test_gate15_scoped_downloader.py
Commands executed:
- Read-only inspection of `docs/AI_EXECUTION_PROTOCOL.md`, the Gate 14A
  decision, current data-access policy, source preflight, and downloader.
- `PYTHONPATH=src .venv/bin/python -m unittest tests.test_gate15_source_review tests.test_gate15_scoped_downloader`
- `PYTHONPATH=src .venv/bin/python -m unittest tests.test_download_meps`
- `.venv/bin/python -m json.tool configs/gate15_hc217_artifact_permissions.json`
- `.venv/bin/python -m py_compile src/meps_fairness/data/download.py src/meps_fairness/gate15_source_review.py tests/test_gate15_source_review.py tests/test_gate15_scoped_downloader.py`
Permissions requested: None.  HC-217 schema/codebook authorization, download,
and PUF-level `data_access.json` expansion remain prohibited.
Tests executed: Gate 15 source and scoped-downloader offline tests; existing
downloader regression tests.  All inputs were synthetic objects, JSON, or
temporary test files.
Exact test results: `Ran 26 tests` / `OK`; existing downloader regression
suite `Ran 87 tests` / `OK`.  No network opener was used by the new Gate 15
tests.
Input hashes: `configs/gate15_hc217_source_review.json` —
`cbae69df7168b4730d9de4c264beeb6e51dcee198124cb2301e277c0a755828e`;
`configs/gate15_hc217_artifact_permissions.json` —
`11483e4cd53ce7fa621de9d6318369c49ed9d28119da2306b3bcc0ae490b6e38`;
current `configs/data_access.json` —
`353bda2a7952b317051ab6e318a0b61edc2b1965bb21b2e667463436aed7d440`;
no MEPS microdata or local HC-217 artifact input.
Output hashes: `src/meps_fairness/gate15_source_review.py` —
`c2dd3e9fb863c09c7c76e32365c28726e3db297b957310b5c1898cfa830db9a8`;
`src/meps_fairness/data/download.py` —
`a71d30b5a0fcaa01665519ea56ab85f970264bf68f9dcf53209d4b151abd3cde`;
`src/meps_fairness/data/__init__.py` —
`f3d5a6484effa84c5a8398a00db2c56ff5b469ff80ab2fe69854448f041eda3b`;
`tests/test_gate15_source_review.py` —
`ef223e27d47f8b6ce2cd89f964c641197896a7192007b45e8938e4c15aa8057e`;
`tests/test_gate15_scoped_downloader.py` —
`fdeecc9a02c3f7e839c79a68b6f8dba03d8a504744099abf2377c9dbafee92f4`;
no HC-217 artifact hashes produced.  The source record retains null SHA-256
fields pending explicit authorization and a local download.
Row counts: Not applicable; no MEPS rows were read.
Assumptions: The exact HC-217 endpoint metadata supplied by the supervisor is
the frozen source contract; broad PUF authorization must not substitute for
artifact-specific, stage-specific authorization.
Unresolved issues: Independent Codex review and Gate 15 acceptance remain
pending; the exact permission is not authorized; local SHA-256, schema,
codebook semantics, LONGWT meaning, and cross-panel comparability remain
unverified.
Git diff summary: Additive Gate 15 repair files plus scoped downloader changes;
no staged files, no commit, no protected-baseline change, and no HC-217 file
under `data/`.  The pre-existing untracked `archive/baseline_v0.3/` remains
outside this gate and is disclosed rather than silently omitted.
Proposed next step: Independent supervisor review of the exact permission
contract and regression evidence.  Keep the active state
`SOURCE_METADATA_PREFLIGHT_ONLY` until the project owner separately authorizes
the exact three artifacts and schema/codebook-only stage.
STOP — waiting for Codex review.
