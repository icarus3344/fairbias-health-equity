# FAIRBIAS-BENCHMARK-B0-R1 Preflight Audit Report

**Document ID**: `docs/plans/fairbias_benchmark_b0_r1_20260914/B0_PREFLIGHT.md`  
**Execution Timestamp**: `2026-09-14T15:55:00+08:00`  
**Active Gate**: `FAIRBIAS-BENCHMARK-B0-R1`  
**Governance Authority**: [AI Execution Protocol](file:///Users/lkc/Downloads/code_v_0_3/docs/AI_EXECUTION_PROTOCOL.md) | [AGENTS.md](file:///Users/lkc/Downloads/code_v_0_3/AGENTS.md) | [GEMINI.md](file:///Users/lkc/Downloads/code_v_0_3/GEMINI.md)  
**Supervisor**: Codex Supervisor  
**Implementation Worker**: Gemini Implementation Worker  

---

## 1. Git Repository & Working Tree Identity

### 1.1 Branch and Commit Hashes
- **Active Branch**: `research/nhis-fairbias`.
  - Local Remote-Tracking Reference: Tracking branch is `origin/research/nhis-fairbias`. Note: In accordance with the offline/read-only constraints of Gate B0-R1, no `git fetch` was performed; this status reflects the local repository state at preflight.
- **Active HEAD Commit**: `67e6659fa65249a8842e34af5d8969629efe4bca`
- **Protected Baseline Tag**: `inherited-code-v0.3-baseline-20260828`
  - **Resolving Commit**: `038897e9f751edac6e36445b7706eec5fdb15988`
  - **Git Tree Object**: `9e43047f69326a844cec1e7acdb6726af555dff3`

### 1.2 Baseline Immutability Audit
Per Section 4 of the [AI Execution Protocol](file:///Users/lkc/Downloads/code_v_0_3/docs/AI_EXECUTION_PROTOCOL.md), the 14 inherited root files and `.gitignore` are permanent historical baseline artifacts and must never be altered, moved, renamed, reformatted, or deleted:
1. `app.py`
2. `classifiers.py`
3. `config.py`
4. `data_COMPAS.csv`
5. `data_Credit_Card.csv`
6. `eval.py`
7. `main.py`
8. `module_AE.py`
9. `module_BM.py`
10. `module_load.py`
11. `module_transform.py`
12. `requirements.txt`
13. `results/all_results.json`
14. `start.sh`
15. `.gitignore`

**Audit Verification Command**:
```bash
git diff HEAD -- app.py classifiers.py config.py data_COMPAS.csv data_Credit_Card.csv eval.py main.py module_AE.py module_BM.py module_load.py module_transform.py requirements.txt results/all_results.json start.sh .gitignore
```
**Verification Result**: Exact diff against HEAD is completely empty (0 insertions, 0 deletions across all 15 protected baseline files).  
*Historical Context on `.gitignore`*: `.gitignore` contains 5 historical lines added in commit `b595e59` (`refactor: organize repository, decommission MEPS track, and focus purely on fairbias-health-equity`), which predated HEAD `67e6659fa65249a8842e34af5d8969629efe4bca`. In the current working tree, `.gitignore` has zero uncommitted modifications against HEAD.

---

## 2. Pre-existing Working Tree Candidate State

### 2.1 Distinction of Working Tree Modifications
The working tree currently contains 14 unstaged modified files and 1 untracked test file. As established in the [Fourth Review](file:///Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_FOURTH_REVIEW_20260913.md), [Consolidated Audit](file:///Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_CONSOLIDATED_AUDIT_AND_APPLICATION_PLAN_20260913.md), and [Supervisor Review of B0](file:///Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_B0_SUPERVISOR_REVIEW_20260914.md):
- These 15 files constitute **pre-existing, unreviewed/unaccepted candidate modifications** from the preceding remediation round (verdict: `REPAIR`).
- These 15 candidate files are **NOT** modifications introduced by Gate B0 or Gate B0-R1.
- HEAD commit `67e6659fa65249a8842e34af5d8969629efe4bca` does **NOT** contain these modifications.
- They must not be staged, committed, reverted, or cleaned up during Gate B0-R1.
- Their SHA-256 hashes match identically to the verified hashes in `docs/reports/FAIRBIAS_FOURTH_REVIEW_20260913_evidence/final_verification.json` and `docs/reports/FAIRBIAS_B0_SUPERVISOR_REVIEW_20260914_evidence/verification.json`.

### 2.2 Pre-existing 15 Candidate Files Hash Table
| File Path | Status | Byte Count | SHA-256 Hash |
|---|---|---|---|
| [`docs/reports/NHIS_D8_R4B_SUPERVISOR_ERRATUM.md`](file:///Users/lkc/Downloads/code_v_0_3/docs/reports/NHIS_D8_R4B_SUPERVISOR_ERRATUM.md) | Modified (unstaged) | 9,633 | `61c59c662a1a455cc3b6e356a3ca6e99785831eda7e51bb1641347b40c7ab412` |
| [`scripts/run_nhis_d8_r4_substantive.py`](file:///Users/lkc/Downloads/code_v_0_3/scripts/run_nhis_d8_r4_substantive.py) | Modified (unstaged) | 42,047 | `2fe3d0fbd25d1dbad0620145fb4f1ec30e389b7209f061fdfbd1880c110721d5` |
| [`src/fairbias/bias_metric.py`](file:///Users/lkc/Downloads/code_v_0_3/src/fairbias/bias_metric.py) | Modified (unstaged) | 25,533 | `498f849fd390aa3d7cc152a9c0ef177f11593aa5b7a6e4cb1c489dcad1147644` |
| [`src/fairbias/enhancement.py`](file:///Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py) | Modified (unstaged) | 49,561 | `60ac98971803893f2f747ee37f0f7582f5fb06b8ad3a899ee5f03181ac0dd700` |
| [`src/fairbias/enhancement_contracts.py`](file:///Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py) | Modified (unstaged) | 24,920 | `60565855fb6a6dfc3df2d9435f00288c3c124737e1b3b285948fdf0893dcb851` |
| [`src/fairbias/enhancement_state.py`](file:///Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_state.py) | Modified (unstaged) | 10,938 | `287f2fe08fc0879c90d3633b76a4b4ff0b0c177b8f07d03d171bd1499d67c469` |
| [`src/nhis_fairbias/adapter.py`](file:///Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/adapter.py) | Modified (unstaged) | 12,721 | `10dbc9213651bcb44c63e40dcd06b3617058d0c84632be41cf57b4078ab2d21b` |
| [`src/nhis_fairbias/d8_enhancement_runner.py`](file:///Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py) | Modified (unstaged) | 58,377 | `1b1677258fffe83f57de1657aef248a9417091c6650b6902df9397a356f1952c` |
| [`src/nhis_fairbias/preprocessing.py`](file:///Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py) | Modified (unstaged) | 22,289 | `e96dd44a82ae4f17cc7821d134ed85cd1e4d4beb2e2d8ab4e9c040d7ddca66d5` |
| [`src/nhis_fairbias/survey.py`](file:///Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/survey.py) | Modified (unstaged) | 8,648 | `6984311469ab5f7de978627371ab23d5a0b86841f525087791dd2ff8712a2dbb` |
| [`tests/test_audit_remediation_probes.py`](file:///Users/lkc/Downloads/code_v_0_3/tests/test_audit_remediation_probes.py) | Untracked | 28,942 | `553b2eea728c35e3cc3b833fabdec536b4869f1d95514494be8a97f185c285cc` |
| [`tests/test_fairbias_enhancement_contracts.py`](file:///Users/lkc/Downloads/code_v_0_3/tests/test_fairbias_enhancement_contracts.py) | Modified (unstaged) | 38,689 | `af3a496504e684ecc6004993bbcd3df38cb6a46c7cc1b5e8cf0fa7184cbe290d` |
| [`tests/test_nhis_d6_temporal.py`](file:///Users/lkc/Downloads/code_v_0_3/tests/test_nhis_d6_temporal.py) | Modified (unstaged) | 64,137 | `6e8fe1f061b66741a41498d0326a2172ab659392ec002760c5a38f56afaa68d2` |
| [`tests/test_nhis_d8_synthetic_contracts.py`](file:///Users/lkc/Downloads/code_v_0_3/tests/test_nhis_d8_synthetic_contracts.py) | Modified (unstaged) | 98,468 | `37a3a1e55b23798246bf4841ba9e310e67b0f198cff804ba2f6317da78a53e8a` |
| [`tests/test_nhis_survey.py`](file:///Users/lkc/Downloads/code_v_0_3/tests/test_nhis_survey.py) | Modified (unstaged) | 3,020 | `cff76d823df2981915b557802c30ba4dae1c697d5d1a89dc4ba3730c758303d2` |

---

## 3. Read-Only Environment & Dependency Inventory

Per the Gate B0-R1 specification and Supervisor guidance (R12), the Python environment was audited exclusively via read-only metadata introspection (`importlib.metadata`). No modules were imported to execute model initialization or pipeline code, no packages were installed, and no network connections were established.

**Important Scope Qualification**: The findings below apply strictly to the **four (4) specifically inspected Python interpreters** and do not constitute a statement regarding all possible Python environments on the host system.

### 3.1 Four Inspected Local Interpreters
1. **System Python (`/opt/homebrew/opt/python@3.13/bin/python3.13`)**:
   - Python Version: `3.13.5 (main, Jun 11 2025, 15:36:57) [Clang 17.0.0 (clang-1700.0.13.3)]`
   - Platform: `macOS-26.6.2-arm64-arm-64bit-Mach-O`
   - `xgboost`: `3.4.1`
   - `scikit-learn`, `numpy`, `pandas`, `scipy`, `fairlearn`, `aif360`, `pytest`: `NOT_INSTALLED`

2. **Framework Python (`/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`)** *(Designated Test Environment)*:
   - Python Version: `3.13.2 (v3.13.2:4f8bb3947cf, Feb 4 2025, 11:51:10) [Clang 15.0.0 (clang-1500.3.9.4)]`
   - `scikit-learn`: `1.7.1`
   - `numpy`: `2.2.6`
   - `pandas`: `2.3.1`
   - `scipy`: `1.16.1`
   - `xgboost`: `3.4.1`
   - `pytest`: `8.4.1`
   - `fairlearn`: **`NOT_INSTALLED`**
   - `aif360`: **`NOT_INSTALLED`**

3. **Repository Virtual Environment (`/Users/lkc/Downloads/code_v_0_3/.venv/bin/python`)**:
   - Python Version: `3.11.15 (main, May 10 2026, 19:23:07) [Clang 22.1.3 ]`
   - `scikit-learn`: `1.9.0`
   - `numpy`: `2.4.6`
   - `pandas`: `2.3.3`
   - `scipy`: `1.17.1`
   - `fairlearn`: **`NOT_INSTALLED`**
   - `aif360`: **`NOT_INSTALLED`**
   - `xgboost`, `pytest`: **`NOT_INSTALLED`**

4. **Legacy Virtual Environment (`/Users/lkc/Downloads/code_v_0_3/.venv311/bin/python`)**:
   - Python Version: `3.11.15 (main, May 10 2026, 19:23:07) [Clang 22.1.3 ]`
   - `scikit-learn`: `1.3.0`
   - `numpy`: `1.24.3`
   - `pandas`: `2.1.1`
   - `scipy`: `1.10.1`
   - `xgboost`: `1.7.6`
   - `fairlearn`: **`NOT_INSTALLED`**
   - `aif360`: **`NOT_INSTALLED`**
   - `pytest`: **`NOT_INSTALLED`**

### 3.2 Dependency Gap & Resolution Policy
- **Identified Gap**: Neither `fairlearn` nor `aif360` is present in any of the four inspected Python environments.
- **Strict B0-R1 Constraint**: In Gate B0-R1, installing packages or downloading wheels is strictly prohibited.
- **Future Resolution in Gate B3**:
  - Gate B3 will construct a dedicated benchmark virtual environment.
  - Required packages (`fairlearn`, `aif360`) and compatible wheels will be installed strictly under an HTTPS domain allowlist (`https://pypi.org`, `https://files.pythonhosted.org`).
  - Downloads must follow the atomic protocol (`.part` download $\to$ checksum verification $\to$ atomic rename).
  - The inherited baseline root file `requirements.txt` will remain completely frozen and unmodified.

---

## 4. Protected Objects & Microdata Isolation

1. **Microdata Protection & DUA Compliance**:
   - Zero individual-level records, raw survey microdata files (`data/raw/`, `data/interim/`, `data/processed/`, or `scratch/`), or historical individual predictions in `runs/` were opened, read, or processed.
   - All MEPS and NHIS data use agreements are strictly upheld. Zero microdata rows are output in any report or artifact.

2. **Network Isolation**:
   - The network was completely disabled. Zero HTTP/HTTPS requests were dispatched during Gate B0-R1.

3. **No Implementation Code Modification, No Model Training, No Git Mutation**:
   - No source code in `src/`, `configs/`, `scripts/`, or `tests/` was modified.
   - No models were trained or instantiated; no benchmark pipelines or test suites were executed.
   - No Git staging (`git add`), committing (`git commit`), or pushing was performed.
   - All original Gate B0 deliverables in `docs/plans/fairbias_benchmark_b0_20260914/` remain completely untouched. All B0-R1 revisions are placed strictly in the new additive directory `docs/plans/fairbias_benchmark_b0_r1_20260914/`.
