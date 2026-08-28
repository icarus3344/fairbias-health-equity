# 3. MEPS Data Use Authorization and Compliance Recording

Date: 2026-08-28

## Context

Conducting empirical research using Public Use Files (PUFs) from the Agency for Healthcare Research and Quality (AHRQ) Medical Expenditure Panel Survey (MEPS) requires adherence to the terms and data-use requirements published in the AHRQ MEPS Data Use Agreement (`https://meps.ahrq.gov/data_stats/data_use.jsp`).

Prior to authorizing network ingestion (Gate 5) or data processing on branch `research/meps-hc252-longitudinal`, the project requires an explicit, immutable governance record that:
1. Formally records the user's explicit acknowledgment and authorization provided in the supervising conversation.
2. Defines the exact research purpose (evaluating risk-prediction and fair-allocation methods for a hypothetical beneficial retention-outreach use case in a retrospective public-use-data study; not an operational system or deployment) and strictly prohibits harmful, punitive, or re-identifying uses.
3. Establishes a machine-readable data access configuration (`configs/data_access.json`) to govern Gate 5 network operations (HTTPS to `meps.ahrq.gov` exclusively).
4. Explicitly bounds the audit status of the record (internal project compliance record, not a legal opinion or claim of individual AHRQ approval).

---

## Decision

1. **Record User-Provided Authorization**:
   - Formally document the exact statement provided by the user in the supervising conversation on 2026-08-28:
     > “我同意遵守 MEPS 数据使用协议，仅用于统计分析，不尝试重新识别、不与可识别记录连接，并同意在成果中引用 AHRQ/MEPS；请开始 Gate 0。”
   - Bounded Scope: This repository and official MEPS HC-244 (Panel 26) and HC-252 (Panel 27) public-use files.
   - Clarified Legal Status: This document and associated compliance files constitute an internal project audit record of user-provided acknowledgment. They do not constitute a legal opinion, do not represent an AHRQ signature, and make no claim of individual project approval by AHRQ.

2. **Define Authorized Purpose and Explicit Prohibitions**:
   - **Authorized Purpose**: Statistical reporting and analysis for a retrospective public-use-data study evaluating risk-prediction and fair-allocation methods for a hypothetical beneficial retention-outreach use case; it is not an operational system or deployment.
   - **Strictly Prohibited Uses**:
     - Underwriting, risk-based pricing, coverage denial, eligibility determination, benefit reduction, or any punitive actions.
     - Any attempt to learn, disclose, or re-identify individual respondents, establishments, or households.
     - Any attempt to link MEPS data with external individually identifiable records (with MEPS-NHIS linkage restricted to Federal Research Data Centers).
     - Printing raw microdata rows or individual records to stdout, logs, reports, or artifacts.
     - Committing raw/interim microdata to Git or uploading microdata to third-party services.

3. **Establish Gate 5 Network Ingestion Allowlist**:
   - **Allowed Host**: `meps.ahrq.gov` exclusively.
   - **Allowed Scheme**: `https` only.
   - **Redirect Policy**: Redirects must remain strictly on `meps.ahrq.gov`; cross-host redirects are strictly forbidden.
   - **Download Authorization**: True only for HC-244 and HC-252 official data archives (`.zip` packages), documentation PDFs, codebook PDFs, and official programming statements.

4. **Mandate Citation & Machine-Readable Configuration**:
   - Enforce formal citation of AHRQ and MEPS in all project reports and publications.
   - Maintain `configs/data_access.json` as the machine-readable source of truth for downstream data ingestion, pipeline scripts, and audit tests.

---

## Consequences & Trade-offs

- **Positive (Regulatory Compliance & Auditability)**: Establishes a verifiable, immutable record of user authorization prior to any data transfer.
- **Positive (Enforceable Guardrails)**: Encodes network and artifact allowlists into machine-readable JSON, enabling automated preflight checks in Gate 5.
- **Positive (Privacy Protection)**: Precludes harmful commercial or punitive uses, re-identification risks, and microdata exposure.
- **Operational Requirement**: Ingestion scripts in Gate 5 must strictly parse `configs/data_access.json` and enforce host/scheme restrictions before initiating HTTP requests.
