# MEPS Data Use Agreement (DUA) User Acknowledgment & Compliance Record

**Date**: 2026-08-28
**Scope**: Repository `research/meps-hc252-longitudinal` and AHRQ MEPS HC-244 (Panel 26) and HC-252 (Panel 27) Public Use Files
**Status**: Acknowledged and Recorded (Pre-Data Ingestion)
**Governance Function**: Project Compliance Audit Record

---

## 1. Regulatory Status & Audit Disclaimer

This document serves solely as an internal project audit record of the user-provided data use authorization recorded in the supervising conversation for this repository.

> **CRITICAL DISCLAIMER**:
> - This document is a project audit record of user-provided acknowledgment.
> - This document is **not** a legal opinion.
> - This document is **not** an AHRQ signature or executed government instrument.
> - This document makes **no claim** that AHRQ individually reviewed or approved this research project.

---

## 2. User-Provided Authorization Statement

In the supervising conversation on **2026-08-28**, the user explicitly affirmed agreement to the AHRQ MEPS Data Use Agreement under the following exact statement:

> “我同意遵守 MEPS 数据使用协议，仅用于统计分析，不尝试重新识别、不与可识别记录连接，并同意在成果中引用 AHRQ/MEPS；请开始 Gate 0。”

---

## 3. Scope of Authorization

This authorization applies strictly to:
1. This repository and its associated research pipelines on branch `research/meps-hc252-longitudinal`.
2. Official public-use data files from the Agency for Healthcare Research and Quality (AHRQ):
   - **MEPS HC-244**: Panel 26 Longitudinal Data Public Use File (2021–2022).
   - **MEPS HC-252**: Panel 27 Longitudinal Data Public Use File (2022–2023).

---

## 4. Authorized Purpose

Data access and analysis under this agreement are restricted exclusively to:
- **Statistical Reporting and Analysis**: Conducting academic, methodology-focused statistical modeling and fairness evaluation for a hypothetical beneficial retention-outreach use case using retrospective public-use data.
- **Beneficial Research Objective**: This retrospective public-use-data study evaluates risk-prediction and fair-allocation methods for a hypothetical beneficial retention-outreach use case; it is not an operational system or deployment.

---

## 5. Explicitly Prohibited Uses & Privacy Guardrails

In compliance with the AHRQ MEPS Data Use Agreement (`https://meps.ahrq.gov/data_stats/data_use.jsp`) and repository governance protocols, the following activities are strictly prohibited:

1. **Adverse Operational Decision-Making**:
   - Individual underwriting or premium risk-rating.
   - Actuarial pricing or discriminatory tiering.
   - Health insurance coverage denial or benefit reduction.
   - Individual eligibility determination.
   - Any punitive, disciplinary, or exclusionary actions.
2. **Re-Identification & Disclose Attempts**:
   - Any attempt to learn, infer, or disclose the identity of any individual survey respondent, establishment, or sample household.
3. **External Identifiable Record Linkage**:
   - Any attempt to link MEPS public-use data with other individually identifiable records or external registries.
   - *Note on NHIS Linkage*: In accordance with official HC-252 documentation, MEPS-NHIS linkage is restricted to authorized Federal Research Data Centers (AHRQ Data Center, NCHS RDC, or U.S. Census RDC network) and is strictly forbidden within this local repository.
4. **Data Exposure & Logging Prohibitions**:
   - Printing raw microdata rows, individual person records, or identifiable data to standard output, execution logs, reports, or documentation.
   - Staging, committing, or tracking raw or processed MEPS microdata in the Git repository.
   - Uploading MEPS microdata or individual-level records to third-party services, external cloud APIs, or unauthorized remote storage.

---

## 6. Mandatory Citation Requirement

All publications, technical reports, presentations, and empirical findings resulting from the use of MEPS data must formally cite the Agency for Healthcare Research and Quality (AHRQ) and the Medical Expenditure Panel Survey (MEPS):

> Agency for Healthcare Research and Quality. Medical Expenditure Panel Survey (MEPS) Public Use Files (HC-244, HC-252). U.S. Department of Health and Human Services.

---

## 7. Gate 5 Network Allowlist & Data Ingestion Boundary

Network operations planned for Gate 5 (Official Data Ingestion & Provenance Recording) are subject to the following frozen constraints:

1. **Domain Allowlist**: `meps.ahrq.gov` strictly.
2. **Protocol Scheme**: `https` exclusively.
3. **Redirect Restriction**: All HTTP redirects must remain strictly on the `meps.ahrq.gov` host. Any redirect attempting to traverse to an off-host domain will be immediately aborted.
4. **Authorized Artifact Types**:
   - Official HC-244 and HC-252 data archives (e.g., `.zip` archive files containing `.ssp`/`.xpt` transport data).
   - Official documentation PDFs (`h244doc.pdf`, `h252doc.pdf`).
   - Official codebook PDFs (`h244cb.pdf`, `h252cb.pdf`).
   - Official AHRQ SAS/Stata/R programming statements and variable layout files.
5. **Provenance & Checksum Policy**:
   - Exact target URLs, retrieval timestamps, file sizes, and locally computed SHA-256 hashes will be recorded in an immutable provenance manifest upon download.
   - Local SHA-256 hashes verify local storage immutability and reproducibility; they do not constitute proof of publisher authenticity when no upstream checksum is published.
