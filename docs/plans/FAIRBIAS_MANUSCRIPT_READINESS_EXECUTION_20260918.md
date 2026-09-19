# FairBias manuscript-readiness execution gate, 2026-09-18

Supervisor scope: execute the user's requested per-variable missingness descriptions, domain plausibility review of transformations, literature verification and submission preparation. Target direction confirmed by user: medical-informatics application paper. Specific journal is not yet selected. Parallel agents are authorized; supervisor independently accepts outputs.

## Fixed boundaries

- Read `docs/AI_EXECUTION_PROTOCOL.md`. Keep inherited baseline files, all frozen training/evaluation sources, predictions and statistical outputs unchanged. Preserve prior reports and paper bundle; deliver a new version of manuscript-readiness artifacts.
- No training, tuning, reselection, new T predictions, altered comparators or statistical family. No server connection, Git staging/commit, submission or messages to others.
- Descriptive raw-variable and missingness aggregation is authorized by this gate, including existing local NHIS 2022/2023/2024 sources. Verify against frozen annual input hashes before reading. T access is solely for predefined descriptive columns/counts and masks; do not calculate model performance or make algorithm changes. Never print or export individual records, identifiers, model predictions or secrets.
- Distinguish item nonresponse, structural not-in-universe, unexpected/unmapped values and true observed values; report numerator, denominator, unweighted and survey-weighted fractions. Actual frozen preprocessing policies remain unchanged. Do not infer MCAR/MAR/MNAR from descriptive counts.
- Domain review is evidence-based technical/clinical plausibility screening, not fabricated clinician endorsement or formal clinical validation. Distinguish BM history from final representations after AE.
- Do not invent author contributions, funding, ethics approval/exemption, institutional determinations, conflicts, journal requirements or study preregistration.

## Public-source retrieval authorization

User-requested literature/reporting verification authorizes read-only HTTPS searches and primary-source retrieval. Initial allowlist: `cdc.gov`, `www.cdc.gov`, `ftp.cdc.gov`, `stacks.cdc.gov`, `spj.science.org`, `doi.org`, `proceedings.mlr.press`, `proceedings.neurips.cc`, `papers.nips.cc`, `proceedings.iclr.cc`, `openreview.net`, `arxiv.org`, `github.com`, `raw.githubusercontent.com`, `fairlearn.org`, `aif360.readthedocs.io`, `oxonfair.readthedocs.io`, `jmlr.org`, `link.springer.com`, `springer.com`, `nature.com`, `www.bmj.com`, `bmj.com`, `informatics.bmj.com`, `authors.bmj.com`, `tripod-statement.org`, `www.tripod-statement.org`, `strobe-statement.org`, `www.strobe-statement.org`, `equator-network.org`, `www.equator-network.org`, `icmje.org`, `www.icmje.org`, `academic.oup.com`, `elsevier.com`, `www.sciencedirect.com`, `jmir.org`, `medinform.jmir.org`, `support.jmir.org`, `ncbi.nlm.nih.gov`, `pubmed.ncbi.nlm.nih.gov`, `pmc.ncbi.nlm.nih.gov`. Official linked primary sources may be added with a documented reason. Do not send private project data to search services. If downloading, use .part, record size/hash/URL and rename only after verification. Web extraction references are not local download authenticity certificates.

## Workstreams and acceptance criteria

1. **Missingness** — produce an independent additive exporter, synthetic boundary checks, frozen input/source manifest, full aggregate CSV and readable summary. Cover all registered core predictors plus outcome/protected eligibility in F/C/S/T and full-year context as feasible; Arm004 exclusions must be N/A, never zero missingness. Verify denominator reconciliation, structural routing and saved cohort counts. Preserve unavailable distinctions explicitly.
2. **Transformation review** — extract all80 completed models' JSON trace/final-state evidence with hashes, map operations to official variable meanings, catalogue dropped/collapsed/merged/powered features and structural-code handling. Produce a variable-level review matrix, severity-ranked concerns, manuscript wording and future-only mitigation proposals. No performance-driven edits.
3. **Literature** — verify identity/year/venue/DOI or stable primary URL of FairBias and all main comparators; cover modern supplementary methods separately. Check claims against actual local adapter/code paths, binary/multigroup support and p/q semantics. Deliver reference file plus claim–source–implementation matrix with verified/unresolved labels. Explicitly correct inherited inaccurate citations without rewriting frozen files.
4. **Submission** — map applicable TRIPOD+AI and STROBE/reporting items to evidence and gaps; verify medical-informatics journal author instructions from official sources. Deliver conservative application-paper positioning, manuscript supplement text, declarations template and action register. Distinguish verified journal requirements from suggestions and pending author-only facts.
5. **Integration/acceptance** — supervisor reviews source bindings, reproduces meaningful aggregates, checks literature links and material claims, and writes a new integrated manuscript-readiness report and bundle. Preserve old accepted package. Record exactly what is complete and what requires an actual author/clinician/institution decision.

Worker outputs go under `docs/paper/manuscript_readiness_20260918/` in owned subdirectories, scripts under `scripts/`, tests under `tests/`. Reports follow protocol Section9. All closure is supervisor-owned.

## Primary-source host addition

2026-09-18: allow read-only `research.cuhk.edu.hk` and its directly linked institutional full-text file host. Reason: the author institution indexes a licensed version-of-record PDF of Tang et al. (2024), enabling a bounded full-text check when the publisher's direct page fails. This author-deposited paper is a primary source, not a substitute bibliographic aggregator. Do not bypass access challenges or infer supplement access from the main PDF.
