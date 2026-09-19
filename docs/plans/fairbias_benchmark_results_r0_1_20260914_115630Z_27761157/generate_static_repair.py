#!/usr/bin/env python3
"""Standard library-only static fact generator for RESULTS-R0.1.

Computes exact file fingerprints, AST test function inventories, non-finite JSON tokens,
and cohort/metric extractions from historical run artifacts without project imports or model execution.
Outputs strictly valid FACTS.json.
"""
import ast
import collections
import hashlib
import json
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
OUTPUT_DIR = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent

# Explicit Allowlist of files to fingerprint
ALLOWLIST_FILES = [
    # Protocol & Agent Governance
    "AGENTS.md",
    "GEMINI.md",
    "docs/AI_EXECUTION_PROTOCOL.md",
    # Plans
    "docs/plans/FAIRBIAS_PRIMARY_BENCHMARK_MASTER_PLAN_20260913.md",
    "docs/plans/GEMINI_FAIRBIAS_BENCHMARK_RESULTS_REPAIR_PROMPT_20260914.md",
    "docs/plans/GEMINI_FAIRBIAS_RESULTS_R0_TARGETED_REPAIR_PROMPT_20260914.md",
    "docs/plans/fairbias_benchmark_b0_r1_20260914/EXPERIMENT_REGISTRY_DRAFT.md",
    "docs/plans/fairbias_benchmark_b0_r1_20260914/ISSUE_CLOSURE_MATRIX.md",
    # Previous Gate RESULTS-R0 Deliverables
    "docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/AUTOMATED_STATIC_MANIFEST.json",
    "docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/AUTOMATED_STATIC_TABLES.md",
    "docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/FINDING_RESPONSE.md",
    "docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/REPAIR_IMPLEMENTATION_SPEC.md",
    "docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/RESULT_STATUS_ADDENDUM.md",
    "docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/WORKER_REPORT.md",
    # Supervisor Reviews & Verified Evidence
    "docs/reports/FAIRBIAS_FULL_BENCHMARK_SUPERVISOR_REVIEW_20260914.md",
    "docs/reports/FAIRBIAS_FULL_BENCHMARK_SUPERVISOR_REVIEW_20260914_evidence/verification.json",
    "docs/reports/FAIRBIAS_RESULTS_R0_SUPERVISOR_REVIEW_20260914.md",
    "docs/reports/FAIRBIAS_RESULTS_R0_SUPERVISOR_REVIEW_20260914_evidence/verified/verification.json",
    "docs/reports/FAIRBIAS_RESULTS_R0_SUPERVISOR_REVIEW_20260914_evidence/verified/corrected_aggregate.json",
    "docs/reports/FAIRBIAS_RESULTS_R0_SUPERVISOR_REVIEW_20260914_evidence/verified/corrected_static_tables.md",
    "docs/reports/FAIRBIAS_CONSOLIDATED_AUDIT_AND_APPLICATION_PLAN_20260913.md",
    "docs/reports/FAIRBIAS_FOURTH_REVIEW_20260913.md",
    # Provenance releases
    "docs/releases/NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7/D6_ARM_003/input_provenance.json",
    "docs/releases/NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609/D6_ARM_003/input_provenance.json",
    "docs/releases/NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609/preprocessing_provenance.json",
    # Historical Benchmark Run Artifact
    "runs/sequential_full_benchmark_20260914_103657Z/integrated_full_benchmark_summary.json",
    # Configs
    "configs/nhis/features.json",
    # Benchmark Implementation Components
    "src/nhis_fairbias/benchmark/__init__.py",
    "src/nhis_fairbias/benchmark/adapters/__init__.py",
    "src/nhis_fairbias/benchmark/adapters/base.py",
    "src/nhis_fairbias/benchmark/adapters/adapter_fairbias.py",
    "src/nhis_fairbias/benchmark/adapters/adapter_lfr.py",
    "src/nhis_fairbias/benchmark/adapters/adapter_reductions.py",
    "src/nhis_fairbias/benchmark/adapters/adapter_reweighing.py",
    "src/nhis_fairbias/benchmark/adapters/adapter_threshold_optimizer.py",
    "src/nhis_fairbias/benchmark/adapters/adapter_unmitigated.py",
    "src/nhis_fairbias/benchmark/data_contracts.py",
    "src/nhis_fairbias/benchmark/metrics.py",
    "src/nhis_fairbias/benchmark/preprocessing.py",
    "src/nhis_fairbias/benchmark/runner.py",
    "src/nhis_fairbias/benchmark/selection.py",
    "src/nhis_fairbias/benchmark/survey_inference.py",
    "scripts/run_sequential_full_benchmark.py",
    "scripts/run_benchmark_demo.py",
    "tests/benchmark/test_adapters.py",
    "tests/benchmark/test_data_contracts.py",
    "tests/benchmark/test_metrics.py",
    "tests/benchmark/test_selection.py",
    # Core legacy files referenced in C01-C06 / E01
    "src/fairbias/config.py",
    "src/fairbias/mitigation.py",
    "src/fairbias/evaluator.py",
    "src/fairbias/bias_metric.py",
    "src/fairbias/enhancement.py",
    "src/fairbias/enhancement_contracts.py",
    "src/fairbias/enhancement_state.py",
    "src/nhis_fairbias/adapter.py",
    "src/nhis_fairbias/preprocessing.py",
    "src/nhis_fairbias/survey.py",
    "src/nhis_fairbias/d8_enhancement_runner.py",
    "scripts/run_nhis_d8_r4_substantive.py",
    "tests/test_audit_remediation_probes.py",
    "tests/test_fairbias_enhancement_contracts.py",
    "tests/test_nhis_d6_temporal.py",
    "tests/test_nhis_d8_synthetic_contracts.py",
    "tests/test_nhis_survey.py",
    "docs/reports/NHIS_D8_R4B_SUPERVISOR_ERRATUM.md"
]


def fingerprint(path: pathlib.Path) -> dict:
    data = path.read_bytes()
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "lines": len(data.splitlines()),
    }


def git_cmd(*args):
    argv = ["git", *args]
    res = subprocess.run(argv, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return res.stdout.strip()


def run_facts_generation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fingerprint Allowlisted Files
    fingerprints = {}
    missing_files = []
    for rel_path in sorted(ALLOWLIST_FILES):
        p = REPO_ROOT / rel_path
        if p.exists():
            fingerprints[rel_path] = fingerprint(p)
        else:
            missing_files.append(rel_path)

    # 2. Git Environment State
    protected_14 = [
        "app.py", "classifiers.py", "config.py", "data_COMPAS.csv", "data_Credit_Card.csv",
        "eval.py", "main.py", "module_AE.py", "module_BM.py", "module_load.py",
        "module_transform.py", "requirements.txt", "results/all_results.json", "start.sh"
    ]
    git_state = {
        "branch": git_cmd("branch", "--show-current"),
        "head": git_cmd("rev-parse", "HEAD"),
        "status_porcelain": git_cmd("status", "--porcelain"),
        "staged_files": git_cmd("diff", "--cached", "--name-only"),
        "root14_diff_vs_baseline_tag": git_cmd("diff", "--name-only", "inherited-code-v0.3-baseline-20260828", "--", *protected_14),
        "gitignore_diff_vs_head": git_cmd("diff", "--name-only", "HEAD", "--", ".gitignore"),
        "gitignore_diff_vs_baseline_tag": git_cmd("diff", "--name-only", "inherited-code-v0.3-baseline-20260828", "--", ".gitignore"),
    }

    # 3. Non-finite tokens in run summary JSON
    run_json_path = REPO_ROOT / "runs/sequential_full_benchmark_20260914_103657Z/integrated_full_benchmark_summary.json"
    nonfinite_tokens = []

    def handle_constant(value):
        nonfinite_tokens.append(value)
        return None

    raw_run_text = run_json_path.read_text(encoding="utf-8")
    run_data = json.loads(raw_run_text, parse_constant=handle_constant)

    # 4. Method and Arm Extractions
    historical_status_counts = collections.Counter()
    arm_extractions = {}
    for arm_id in ["arm_001", "arm_002", "arm_003", "arm_004"]:
        arm_obj = run_data[arm_id]
        methods = arm_obj["methods"]
        for m in methods.values():
            historical_status_counts[m["status"]] += 1

        fb = methods["FAIRBIAS_BM"]
        base = methods["UNMITIGATED"]
        rw = methods["REWEIGHING"]

        differences = {
            "fb_minus_baseline_balanced_accuracy": fb["balanced_accuracy"] - base["balanced_accuracy"],
            "fb_minus_baseline_eo_gap": fb["eo_gap"] - base["eo_gap"],
            "fb_minus_rw_balanced_accuracy": fb["balanced_accuracy"] - rw["balanced_accuracy"],
            "fb_minus_rw_eo_gap": fb["eo_gap"] - rw["eo_gap"],
        }
        rw_dominates_ba_eo = (rw["balanced_accuracy"] > fb["balanced_accuracy"]) and (rw["eo_gap"] < fb["eo_gap"])

        arm_extractions[arm_id] = {
            "sample_sizes": arm_obj["sample_sizes"],
            "all_years_sum": sum(arm_obj["sample_sizes"].values()),
            "methods": methods,
            "differences": differences,
            "rw_point_dominates_fb_ba_eo": rw_dominates_ba_eo,
            "paired_contrasts": arm_obj["paired_contrasts"],
        }

    # 5. AST Test Functions Inventory
    test_files = [
        "tests/benchmark/test_adapters.py",
        "tests/benchmark/test_data_contracts.py",
        "tests/benchmark/test_metrics.py",
        "tests/benchmark/test_selection.py",
    ]
    ast_inventory = {}
    for tf_rel in test_files:
        tf_path = REPO_ROOT / tf_rel
        content = tf_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=tf_rel)
        funcs = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                funcs[node.name] = {
                    "line": node.lineno,
                    "end_line": node.end_lineno,
                }
        ast_inventory[tf_rel] = funcs

    # 6. Previous Gate Output Audit Check
    r0_files = [
        "docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/AUTOMATED_STATIC_MANIFEST.json",
        "docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/AUTOMATED_STATIC_TABLES.md",
        "docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/FINDING_RESPONSE.md",
        "docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/REPAIR_IMPLEMENTATION_SPEC.md",
        "docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/RESULT_STATUS_ADDENDUM.md",
        "docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/WORKER_REPORT.md",
    ]
    r0_total_bytes = sum(fingerprints[f]["bytes"] for f in r0_files)
    r0_total_lines = sum(fingerprints[f]["lines"] for f in r0_files)

    # 7. Supervisor Input Hash Verification
    sup_review_path = "docs/reports/FAIRBIAS_FULL_BENCHMARK_SUPERVISOR_REVIEW_20260914.md"
    sup_verif_path = "docs/reports/FAIRBIAS_FULL_BENCHMARK_SUPERVISOR_REVIEW_20260914_evidence/verification.json"
    actual_review_hash = fingerprints[sup_review_path]["sha256"]
    actual_verif_hash = fingerprints[sup_verif_path]["sha256"]
    expected_review_hash = "5ee400543c11ee5a7c559dc2e17ed82c8e5fa6c19cd6eca36b45843e4f96c8a2"
    expected_verif_hash = "9a55c87665a19746c4917f39dadc3b4e2a38d4146cd2274aee3241978256e89d"

    hash_verification = {
        "supervisor_review_actual_sha256": actual_review_hash,
        "supervisor_review_expected_sha256": expected_review_hash,
        "supervisor_review_matches": actual_review_hash == expected_review_hash,
        "supervisor_verification_actual_sha256": actual_verif_hash,
        "supervisor_verification_expected_sha256": expected_verif_hash,
        "supervisor_verification_matches": actual_verif_hash == expected_verif_hash,
    }

    facts = {
        "manifest_id": "RESULTS_R0_1_FACTS",
        "generator_script": "docs/plans/fairbias_benchmark_results_r0_1_20260914_115630Z_27761157/generate_static_repair.py",
        "git_state": git_state,
        "hash_verification": hash_verification,
        "previous_gate_r0_summary": {
            "file_count": len(r0_files),
            "total_bytes": r0_total_bytes,
            "total_lines": r0_total_lines,
        },
        "nonfinite_token_audit": {
            "token_count": len(nonfinite_tokens),
            "tokens": nonfinite_tokens,
            "source_file": "runs/sequential_full_benchmark_20260914_103657Z/integrated_full_benchmark_summary.json",
        },
        "historical_status_counts": dict(historical_status_counts),
        "arms": arm_extractions,
        "ast_test_inventory": ast_inventory,
        "ast_total_test_functions": sum(len(v) for v in ast_inventory.values()),
        "fingerprints": fingerprints,
    }

    facts_path = OUTPUT_DIR / "FACTS.json"
    with open(facts_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")

    print(f"Generated FACTS.json successfully at {facts_path}")
    print(f"  Total fingerprinted files: {len(fingerprints)}")
    print(f"  Nonfinite NaN tokens in summary JSON: {len(nonfinite_tokens)}")
    print(f"  Total AST test functions: {sum(len(v) for v in ast_inventory.values())}")
    print(f"  R0 Deliverable bytes: {r0_total_bytes}, lines: {r0_total_lines}")
    print(f"  Supervisor review hash matches expected: {actual_review_hash == expected_review_hash}")
    print(f"  Supervisor verification hash matches expected: {actual_verif_hash == expected_verif_hash}")


if __name__ == "__main__":
    run_facts_generation()
