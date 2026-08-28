# MEPS Source Registry & Data Provenance Specification

**Date**: 2026-08-28  
**Status**: Pre-Data Ingestion Registry (No Local Microdata Present)  
**Access Date for Upstream Evidence**: 2026-08-28

---

## 1. Local Storage Status Notice

> **IMPORTANT**: As of Gate 3, **NO MEPS MICRODATA FILES HAVE BEEN DOWNLOADED OR ARE PRESENT LOCALLY**. All directories (`data/raw/`, `data/interim/`, `data/processed/`) contain only `.gitkeep` placeholder files. Network requests and data ingestion are strictly prohibited until authorized under subsequent execution gates.

---

## 2. Upstream Official Source File Registry

The following table documents official survey file metadata verified directly against Agency for Healthcare Research and Quality (AHRQ) MEPS documentation:

| Dimension / Property | MEPS HC-244 Panel 26 Longitudinal Data Public Use File | MEPS HC-252 Panel 27 Longitudinal Data Public Use File |
|---|---|---|
| **Panel Number** | Panel 26 | Panel 27 |
| **Survey Years** | 2021 (Year 1) – 2022 (Year 2) | 2022 (Year 1) – 2023 (Year 2) |
| **Release Date** | September 2024 | September 2025 |
| **Total Person Records** | 6,741 | 8,292 |
| **Total Variables** | 2,737 | 2,648 |
| **Complete 5 Rounds (`ALL5RDS=1`)** | 6,295 records | 7,812 records |
| **Both-Years In-Scope Variable** | `YEARIND` (`YEARIND=1`) | `YEARIND` (`YEARIND=1`) |
| **Rounds Structure** | 5 rounds across 2 years | 5 rounds across 2 years |
| **Variable Suffix Convention** | 2021/2022 annual suffixes become Y1/Y2 | 2022/2023 annual suffixes become Y1/Y2 |
| **Longitudinal Analysis Weight** | `LONGWT` | `LONGWT` |
| **Survey Design Strata Variable** | `VARSTR` | `VARSTR` |
| **Survey Design PSU Variable** | `VARPSU` | `VARPSU` |
| **Official Details URL** | [HC-244 Details Page](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-244) | [HC-252 Details Page](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-252) |
| **Documentation PDF** | [HC-244 Documentation](https://meps.ahrq.gov/data_stats/download_data/pufs/h244/h244doc.pdf) | [HC-252 Documentation](https://meps.ahrq.gov/data_stats/download_data/pufs/h252/h252doc.pdf) |
| **Codebook PDF** | [HC-244 Codebook](https://meps.ahrq.gov/data_stats/download_data/pufs/h244/h244cb.pdf) | [HC-252 Codebook](https://meps.ahrq.gov/data_stats/download_data/pufs/h252/h252cb.pdf) |
| **Candidate Download Endpoint (SSP/XPT ZIP)** | `https://meps.ahrq.gov/data_files/pufs/h244/h244ssp.zip` *(To be HTTP-verified in Gate 5)* | `https://meps.ahrq.gov/mepsweb/data_files/pufs/h252/h252ssp.zip` *(To be HTTP-verified in Gate 5)* |
| **Publisher Checksum** | *Not listed (none verified upstream)* | *Not listed (none verified upstream)* |

---

## 3. General Longitudinal Survey References

- **AHRQ MEPS Longitudinal Data Guidance**: Overview of design, weight construction, and longitudinal pooling principles.  
  URL: [https://meps.ahrq.gov/mepsweb/data_stats/more_info_download_data_files.jsp](https://meps.ahrq.gov/mepsweb/data_stats/more_info_download_data_files.jsp) (Accessed 2026-08-28).

---

## 4. Data Use Agreement & Regulatory Guardrails

All research activities must strictly comply with the **AHRQ Data Use Agreement for MEPS Public Use Files**:
- **Data Use Agreement URL**: [https://meps.ahrq.gov/data_stats/data_use.jsp](https://meps.ahrq.gov/data_stats/data_use.jsp) (Accessed 2026-08-28).

### 4.1 Key Provisions
1. **Statistical Reporting and Analysis Only**: Data must be used exclusively for statistical reporting and health-related analysis.
2. **No Re-Identification**: Absolute prohibition against attempting to learn or disclose the identity of any survey respondent, establishment, or household.
3. **No Attempts to Link with Individually Identifiable Records**: Prohibited from making any attempt to link MEPS public-use data with other individually identifiable records.
4. **MEPS-NHIS Linkage Restriction**: In accordance with HC-252 documentation, MEPS-NHIS linkage may occur only within the AHRQ Data Center, NCHS Research Data Center, or U.S. Census Research Data Center network (not a blanket statement that linkage is never allowed).
5. **Formal Citation**: All reports and publications must cite the Agency for Healthcare Research and Quality (AHRQ) and the Medical Expenditure Panel Survey.

> *Note*: Project participants must strictly follow these stipulations without expanding legal interpretations or quoting unverified external penalties.
