import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PLAN_ROOT = ROOT / "docs/plans/nhis_bias_attribute_discovery_20260918"
REGISTRY_PATH = PLAN_ROOT / "PRE_FREEZE_REGISTRY_V3_DRAFT.json"
CHECKLIST_PATH = PLAN_ROOT / "A1_FREEZE_CHECKLIST_V3_DRAFT.json"
REPAIR_VERIFICATION_PATH = PLAN_ROOT / "repair_verification_v3.json"
HISTORICAL_VERIFICATION_PATH = PLAN_ROOT / "planning_verification_v3.json"


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def test_v3_registry_preserves_v2_and_binds_current_static_sources():
    registry = _json(REGISTRY_PATH)
    expected = {
        "docs/AI_EXECUTION_PROTOCOL.md": "1a3d38929d65493f8352d6b46d32f65fa836502d08a3c2d2704dd713e5c3beac",
        "docs/plans/nhis_bias_attribute_discovery_20260918/RESEARCH_PLAN_V2.md": "1a90183fe1b3248be86bdbce3b4461443c77c6f13815282033e5de66e9f31152",
        "docs/plans/nhis_bias_attribute_discovery_20260918/IMPLEMENTATION_HANDOFF_V2.md": "bb6aeac0d81419fdf58a5be716589959de5a9d95b1a8cd00f4c6446239ae3600",
        "configs/nhis/variables.json": registry["source_bindings"]["variables_registry_sha256"],
        "configs/nhis/features.json": registry["source_bindings"]["features_registry_sha256"],
        "configs/nhis/study.json": registry["source_bindings"]["study_registry_sha256"],
        "docs/paper/manuscript_readiness_20260918/transformations/primary_sources/adult-codebook-2024.pdf": registry["source_bindings"]["official_2024_codebook_sha256"],
        "artifacts/nhis/completion_evaluation_20260918/control/admission_v1.json": registry["source_bindings"]["completion_admission_sha256"],
        "artifacts/nhis/completion_evaluation_20260918/control/study_v1.json": registry["source_bindings"]["completion_study_sha256"],
    }
    for relative, digest in expected.items():
        assert _sha(ROOT / relative) == digest


def test_candidate_registry_is_a_frozen_size_seed_subset_of_the_larger_atlas():
    registry = _json(REGISTRY_PATH)
    candidates = registry["candidate_registry"]["candidates"]
    registered = [row["variable"] for row in candidates]
    with (PLAN_ROOT / "VARIABLE_ATLAS_V3_DRAFT.csv").open(newline="", encoding="utf-8") as stream:
        atlas_rows = list(csv.DictReader(stream))
    atlas = [row["variable"] for row in atlas_rows]
    assert len(registered) == len(set(registered)) == 24
    assert len(atlas) == len(set(atlas)) == 31
    assert registered == atlas[:24]
    assert set(registered).issubset(atlas)
    assert registered[:3] == ["SEX_A", "HISPALLP_A", "DISAB3_A"]
    assert registry["audit_scopes"]["anchor"] == registered[:3]
    extensions = atlas_rows[24:]
    assert {row["variable"] for row in extensions} == {
        "URBRRL23",
        "MARSTAT_A",
        "NATUSBORN_A",
        "CITZNSTP_A",
        "FDSCAT3_A",
        "ANXEV_A",
        "DEPEV_A",
    }
    assert {row["admission_status"] for row in extensions} == {
        "METADATA_ATLAS_ONLY_PENDING_DOWNSELECT"
    }
    registered_extensions = registry["candidate_registry"]["metadata_atlas_extensions_verified_2024"]
    assert {row["variable"] for row in registered_extensions} == {
        row["variable"] for row in extensions
    }
    assert all(row["admission"] == "METADATA_ATLAS_ONLY_PENDING_DOWNSELECT_AND_CROSS_YEAR_REVIEW" for row in registered_extensions)


def test_main_panel_members_match_admission_manifest_without_result_selection():
    registry = _json(REGISTRY_PATH)
    panel = registry["comparison_panels"]["main_risk_panel"]
    admission = _json(ROOT / "artifacts/nhis/completion_evaluation_20260918/control/admission_v1.json")
    parent_index = {
        (row["arm_id"], row["backbone"], row["method"], row["status"]): row
        for row in admission["comparators"]
    }
    job_groups = {}
    for job in admission["jobs"]:
        key = (job["config"]["arm_id"], job["config"]["backbone"], job["config"]["method"])
        job_groups.setdefault(key, []).append(job["model_id"])
    members = {row["member_id"]: row for row in panel["members"]}
    for method in ("UNMITIGATED", "REWEIGHING", "FAIRBIAS_BM"):
        source = parent_index[("arm_001", "GBDT", method, "FEASIBLE")]
        assert members[method]["selection_id"] == source["selection_id"]
        assert set(members[method]["model_ids"]) == set(source["model_ids"])
        assert {row["output_type"] for row in source["prediction_refs"]} == {"event_probability_p"}
        source_by_seed = {row["seed"]: row for row in source["prediction_refs"]}
        for binding in members[method]["seed_model_bindings"]:
            expected = source_by_seed[binding["seed"]]
            assert binding["model_id"] == expected["model_id"]
            assert binding["model_sha256"] == expected["model_sha256"]
    for method in ("FAIRBIAS_BM_AE", "FAIRBIAS_JOINT"):
        assert set(members[method]["model_ids"]) == set(job_groups[("arm_001", "GBDT", method)])
        assert len(members[method]["model_ids"]) == 5
        assert all(binding["model_sha256"] is None for binding in members[method]["seed_model_bindings"])
        assert members[method]["output_type_status"] == "REGISTERED_PENDING_SOURCE_CONFIRMATION"
    required_member_fields = {
        "source_binding",
        "O_train",
        "seed_policy_sha256",
        "seed_model_bindings",
        "feature_sha256",
        "preprocessing_sha256",
        "training_source_sha256",
        "prediction_source_code_sha256",
        "backbone",
        "output_type",
        "threshold_policy",
        "O_at_inference",
        "member_contract_complete",
        "pending_fields",
    }
    for member in members.values():
        assert required_member_fields.issubset(member)
        assert member["O_train"] == "SEX_A"
        assert member["member_contract_complete"] is False
        assert set(member["source_binding"]) == {"path", "sha256"}
        assert member["source_binding"]["path"] == panel["source_manifest"]["path"]
        assert member["source_binding"]["sha256"] == panel["source_manifest"]["sha256"]
        assert member["backbone"] == panel["deployment_contract"]["backbone"]
        assert member["threshold_policy"]["status"] == "PENDING"
        assert member["threshold_policy"]["selection_role"] is None
        assert member["threshold_policy"]["threshold_policy_sha256"] is None
    assert panel["baseline_member_id"] == "UNMITIGATED"
    assert panel["O_train"] == "SEX_A"
    assert panel["seed_policy"]["status"] == "HUMAN_DECISION_REQUIRED"
    assert panel["model_id_uniqueness_policy"] == "UNIQUE_ACROSS_PANEL_MEMBERS_AND_SEEDS"
    assert panel["deployment_contract"]["output_type"] == "event_probability_p"
    assert panel["deployment_contract"]["feature_sha256"] is None
    assert panel["deployment_contract"]["preprocessing_sha256"] is None
    assert panel["deployment_contract"]["threshold_policy_mode"] == "REUSE_EACH_MEMBER_FROZEN_POLICY_NO_RETUNING"
    assert panel["common_domain_contract"]["scope"] == "per_O"
    assert panel["common_domain_contract"]["global_all_O_complete_case_prohibited"] is True
    freeze_contract = registry["comparison_panels"]["freeze_validator_contract"]
    assert freeze_contract["annual_design_rows_retained"] is True
    assert freeze_contract["global_all_O_complete_case_prohibited"] is True
    assert freeze_contract["false_nonboolean_or_missing_domain_flags_rejected"] is True
    payload = dict(panel)
    claimed_hash = payload.pop("panel_sha256")
    assert panel["status"] == "TECHNICALLY_BOUND_PENDING_MEMBER_CONTRACT_AND_COVERAGE_PREFLIGHT"
    assert claimed_hash == _canonical_sha(payload)


def test_registry_exposes_q2_prediction_family_and_precision_contracts_without_freezing_content():
    registry = _json(REGISTRY_PATH)
    q2 = registry["q2_scope_aggregator"]
    assert q2["priority_high_to_low"] == [
        "SUPPORTED_EXCEEDS_TOLERANCE",
        "INSUFFICIENT_EVIDENCE",
        "SUPPORTED_WITHIN_TOLERANCE",
    ]
    assert q2["global_all_O_complete_case_prohibited"] is True
    assert set(q2["required_scientific_contrast_identity"]) >= {
        "year",
        "evidence_role",
        "panel_id",
        "panel_sha256",
        "O_train",
        "O_audit",
        "audit_role",
        "audit_role_sha256",
        "O_group_rule_sha256",
        "O_registry_version",
        "O_registry_sha256",
        "contrast_universe_sha256",
        "group",
        "reference_group",
        "endpoint",
        "seed_policy_sha256",
        "delta",
        "delta_units",
        "delta_rationale_id",
        "contrast_id",
    }
    assert q2["no_new_O_path"]["empty_new_O_rows_allowed_only_with_hash_bound_receipt"] is True
    assert q2["no_new_O_path"]["affirmative_new_O_change_prohibited"] is True
    assert q2["no_new_O_path"]["codebook_wide_eligibility_ledger_complete"] is False
    assert q2["no_new_O_path"]["contrast_universe_new_O_set_must_be_empty"] is True
    assert q2["no_new_O_path"]["candidate_universe_scope"] == "PROJECT_ATLAS_NOT_CODEBOOK_WIDE_ELIGIBILITY_LEDGER"
    assert {"year", "panel_id", "model_id_or_method_matched_baseline_pair", "delta"}.issubset(
        q2["expanded_rows_shared_scope_identity"]
    )
    predictions = registry["prediction_decision_contract"]
    assert predictions["decision_output_type"] == "binary_decision_y_hat"
    assert predictions["soft_probability_as_error_rate_prohibited"] is True
    assert set(predictions["required_hashes"]) == {
        "prediction_sha256",
        "threshold_policy_sha256",
        "y_hat_sha256",
        "record_order_sha256",
    }
    families = registry["statistical_families"]
    assert families["status"] == "MANIFEST_SCHEMA_IMPLEMENTED_CONTENT_NOT_FROZEN"
    assert families["predeclared_NA_slots_retained"] is True
    assert {"year", "evidence_role", "scope", "panel_sha256", "O_train", "O_audit", "family_id", "family_sha256"}.issubset(
        families["required_identity_fields"]
    )
    assert "manifest_sha256" in families["required_manifest_fields"]
    assert families["top_level_verifier_contract"]["manifest_sha256_required"] is True
    assert "O_group_rule_sha256" in families["required_identity_fields"]
    assert "audit_role_sha256" in families["required_identity_fields"]
    assert "contrast_universe_sha256" in families["required_identity_fields"]
    assert "contrast_universe_receipt" in families["required_manifest_fields"]
    universe = registry["contrast_universe_contract"]
    assert universe["fixed_ordered_anchor_O_set"] == ["SEX_A", "HISPALLP_A", "DISAB3_A"]
    assert universe["same_O_in_anchor_and_new_O_prohibited"] is True
    assert universe["family_and_q2_must_bind_same_receipt"] is True
    assert universe["scope_rules"]["anchor"] == "EXACT_FIXED_ANCHOR_UNIVERSE_ONLY"
    assert families["seed_policy_verifier_contract"]["seeds_nonempty_unique_non_boolean_integers"] is True
    threshold = predictions["threshold_policy_frozen_schema"]
    assert threshold["selection_role"] == "FROZEN_WITHOUT_EVALUATION_RESULT_ACCESS"
    assert threshold["partial_self_hashed_policy_prohibited"] is True
    precision = registry["precision"]
    assert precision["minimum_detectable_effect_status"].startswith("NOT_IMPLEMENTED")
    assert "precision_sufficient_for_delta" not in precision["required_outputs"]


def test_registry_cannot_authorize_a3_or_2025_in_draft_state():
    registry = _json(REGISTRY_PATH)
    assert registry["status"] == "DRAFT_NOT_FROZEN"
    assert registry["real_performance_scan_authorized"] is False
    assert registry["delta_decision"]["status"] == "HUMAN_DECISION_REQUIRED"
    assert registry["a3_gate"]["status"].startswith("BLOCKED")
    assert registry["a3_gate"]["technical_panel_frozen"] is False
    release = registry["validation_2025_lock"]
    assert release["status"] == "LOCKED"
    assert release["micro_outcome_or_performance_read"] is False
    assert not any(value is True for key, value in release.items() if key.endswith("_frozen") or key.endswith("_verified") or key.endswith("_recorded"))


def test_repair_checklist_records_software_work_without_self_approval():
    checklist = _json(CHECKLIST_PATH)
    items = {row["id"]: row for row in checklist["items"]}
    assert checklist["status"].startswith("REPAIR_TRANCHE_4_WORKER_IMPLEMENTED")
    assert len(items) == 52
    assert sum(row["complete"] for row in items.values()) == 32
    assert items["A1-015"]["complete"] is False
    assert items["A1-027"]["complete"] is False
    assert all(items[f"A1-{index:03d}"]["complete"] is True for index in range(33, 53))
    assert checklist["a1_freeze_ready"] is False
    assert checklist["a3_authorized"] is False
    assert checklist["validation_2025_authorized"] is False
    assert checklist["readiness_scopes"]["a2_repair_tranche_4"]["ready"] is False


def test_repair_receipt_supersedes_historical_planning_receipt_without_self_approval():
    historical = _json(HISTORICAL_VERIFICATION_PATH)
    receipt = _json(REPAIR_VERIFICATION_PATH)
    assert historical["lifecycle_status"] == "SUPERSEDED_BY_REPAIR_VERIFICATION_V3"
    assert historical["superseded_by"] == "repair_verification_v3.json"
    assert historical["historical_snapshot_only"] is True
    assert receipt["schema_version"] == "nhis_bias_attribute_audit_repair_verification_v3"
    assert receipt["status"]["a2"] == "REPAIR_TRANCHE_4_WORKER_COMPLETE_REVIEW_REQUIRED"
    assert receipt["status"]["a3"] == "BLOCKED_NOT_AUTHORIZED"
    assert receipt["status"]["validation_2025"] == "LOCKED_NOT_READ"
    assert receipt["reviewer_input_sha256"]["INDEPENDENT_REVIEW_A0_A2_REPAIR_V3.md"] == _sha(
        PLAN_ROOT / "INDEPENDENT_REVIEW_A0_A2_REPAIR_V3.md"
    )
    assert receipt["reviewer_input_sha256"]["INDEPENDENT_REVIEW_A0_A2_REPAIR_TRANCHE_2_V3.md"] == _sha(
        PLAN_ROOT / "INDEPENDENT_REVIEW_A0_A2_REPAIR_TRANCHE_2_V3.md"
    )
    assert receipt["reviewer_input_sha256"]["INDEPENDENT_REVIEW_A0_A2_REPAIR_TRANCHE_3_V3.md"] == _sha(
        PLAN_ROOT / "INDEPENDENT_REVIEW_A0_A2_REPAIR_TRANCHE_3_V3.md"
    )
    for relative, digest in receipt["output_sha256"].items():
        assert _sha(ROOT / relative) == digest
    for name, digest in receipt["tranche_reports_sha256"].items():
        assert _sha(PLAN_ROOT / name) == digest
    assert str(REPAIR_VERIFICATION_PATH.relative_to(ROOT)) not in receipt["output_sha256"]
    assert receipt["output_hash_note"] == "This verification file deliberately does not hash itself to avoid self-reference."
    assert receipt["real_performance_scan_authorized"] is False
    assert receipt["2025_micro_outcome_or_performance_read"] is False
