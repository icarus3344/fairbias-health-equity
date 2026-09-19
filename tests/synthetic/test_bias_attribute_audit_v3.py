import copy

import numpy as np
import pytest

from nhis_fairbias.benchmark.survey_linearization import linearized_survey_inference
from nhis_fairbias.bias_attribute_audit_v3 import (
    AuditAttributeSpec,
    AuditContractError,
    CONCLUSION_EXCEEDS,
    CONCLUSION_INSUFFICIENT,
    CONCLUSION_WITHIN,
    aggregate_q2_scope_conclusions,
    audit_role_sha256,
    assert_2025_performance_access_allowed,
    bind_thresholded_decisions,
    canonical_sha256,
    classify_method_tradeoff,
    classify_tolerance_interval,
    freeze_comparison_panel,
    freeze_contrast_universe_receipt,
    freeze_no_new_O_receipt,
    freeze_threshold_policy,
    generate_statistical_families,
    holm_adjust,
    per_attribute_common_domain,
    precision_diagnostics,
    project_max_absolute_interval,
    record_order_sha256,
    simultaneous_intervals_from_replicates,
    validate_supplemental_panel,
    validate_binary_decisions,
    validate_metric_input,
    verify_frozen_panel,
    verify_record_alignment,
    verify_statistical_family_manifest,
    verify_contrast_universe_receipt,
)


_O_REGISTRY_VERSION = "synthetic_O_registry_v1"
_O_REGISTRY_SHA256 = "9" * 64
_SEED_POLICY = {
    "status": "FROZEN",
    "seeds": [0, 7],
    "aggregation_estimand": "mean_over_registered_training_seeds",
    "training_randomness_inference": "report_seed_distribution_not_survey_replicates",
}


def _attribute(variable, groups, reference, role, rule_character):
    return AuditAttributeSpec(
        variable=variable,
        groups=tuple(groups),
        reference_group=reference,
        audit_role=role,
        O_group_rule_sha256=rule_character * 64,
        audit_role_sha256=audit_role_sha256(
            variable,
            role,
            _O_REGISTRY_VERSION,
            _O_REGISTRY_SHA256,
        ),
        O_registry_version=_O_REGISTRY_VERSION,
        O_registry_sha256=_O_REGISTRY_SHA256,
    )


def _anchor_attributes():
    return [
        _attribute("SEX_A", (1, 2), 1, "anchor", "1"),
        _attribute("HISPALLP_A", (1, 2), 1, "anchor", "2"),
        _attribute("DISAB3_A", (1, 2), 1, "anchor", "3"),
    ]


def _new_attributes():
    return [_attribute("VISIONDF_A", (1, 2), 1, "expanded_candidate", "4")]


def _contrast_universe(
    *,
    include_new=True,
    seed_policy=None,
    endpoints=("fnr", "fpr"),
    new_attributes=None,
):
    policy = _SEED_POLICY if seed_policy is None else seed_policy
    selected_new = _new_attributes() if new_attributes is None else new_attributes
    return freeze_contrast_universe_receipt(
        _anchor_attributes(),
        selected_new if include_new else [],
        O_registry_version=_O_REGISTRY_VERSION,
        O_registry_sha256=_O_REGISTRY_SHA256,
        selection_rule_id="synthetic_selection_v1",
        selection_rule_sha256="8" * 64,
        q1_model_ids=["fixed_gbdt"],
        confirmatory_method_pairs=[("reweighing", "unmitigated")],
        descriptive_method_pairs=[("joint", "unmitigated")],
        endpoints=endpoints,
        year=2023,
        evidence_role="discovery",
        panel_id="synthetic_panel",
        panel_sha256="a" * 64,
        O_train="SEX_A",
        seed_policy_sha256=canonical_sha256(policy),
        delta_rationale_id="synthetic_delta",
    )


def _family_kwargs(*, scope, include_new=True, seed_policy=None, endpoints=("fnr", "fpr")):
    policy = _SEED_POLICY if seed_policy is None else seed_policy
    receipt = _contrast_universe(
        include_new=include_new,
        seed_policy=policy,
        endpoints=endpoints,
    )
    attributes = _anchor_attributes() + (_new_attributes() if scope == "expanded" and include_new else [])
    return attributes, {
        "contrast_universe_receipt": receipt,
        "q1_model_ids": ["fixed_gbdt"],
        "confirmatory_method_pairs": [("reweighing", "unmitigated")],
        "descriptive_method_pairs": [("joint", "unmitigated")],
        "endpoints": endpoints,
        "year": 2023,
        "evidence_role": "discovery",
        "scope": scope,
        "panel_id": "synthetic_panel",
        "panel_sha256": "a" * 64,
        "O_train": "SEX_A",
        "seed_policy": policy,
        "delta_rationale_id": "synthetic_delta",
    }


def test_record_alignment_binds_exact_order_and_rejects_duplicates():
    keys = ["nhis:2023:001", "nhis:2023:002", "nhis:2023:003"]
    digest = record_order_sha256(keys)
    result = verify_record_alignment(keys, list(keys), expected_order_sha256=digest)
    assert result == {"status": "ALIGNED", "rows": 3, "record_order_sha256": digest}
    with pytest.raises(AuditContractError, match="position 0"):
        verify_record_alignment(keys, list(reversed(keys)))
    with pytest.raises(AuditContractError, match="duplicate"):
        record_order_sha256([keys[0], keys[0]])
    with pytest.raises(AuditContractError, match="canonical string"):
        record_order_sha256([1, "1"])
    with pytest.raises(AuditContractError, match="canonical string"):
        verify_record_alignment(["1"], [1])


def test_each_attribute_has_its_own_panel_common_domain():
    outcome_valid = np.ones(6, dtype=bool)
    prediction_masks = {
        "baseline": np.array([1, 1, 1, 1, 1, 0], dtype=bool),
        "method": np.array([1, 1, 1, 1, 0, 1], dtype=bool),
    }
    o_one = np.array([1, 2, 1, 2, 9, 1])
    o_two = np.array([1, 1, 7, 2, 2, 2])
    domain_one, meta_one = per_attribute_common_domain(outcome_valid, o_one, [1, 2], prediction_masks)
    domain_two, meta_two = per_attribute_common_domain(outcome_valid, o_two, [1, 2], prediction_masks)
    assert domain_one.tolist() == [True, True, True, True, False, False]
    assert domain_two.tolist() == [True, True, False, True, False, False]
    assert meta_one["domain_rows"] == 4
    assert meta_two["domain_rows"] == 3


def test_q1_and_method_change_are_disjoint_families_with_distinct_nulls():
    attributes, kwargs = _family_kwargs(scope="expanded")
    families = generate_statistical_families(attributes, **kwargs)
    assert families["family_sizes"] == {
        "q1_confirmatory": 8,
        "q3_q4_confirmatory": 8,
        "q3_q4_descriptive": 8,
    }
    assert {row["null_hypothesis"] for row in families["q1_confirmatory"]} == {"d_model_O_group = 0"}
    assert {row["null_hypothesis"] for row in families["q3_q4_confirmatory"]} == {
        "d_method_O_group - d_matched_baseline_O_group = 0"
    }
    q1_ids = {row["contrast_id"] for row in families["q1_confirmatory"]}
    method_ids = {row["contrast_id"] for row in families["q3_q4_confirmatory"]}
    assert q1_ids.isdisjoint(method_ids)
    assert all(row["year"] == 2023 and row["O_train"] == "SEX_A" for row in families["q1_confirmatory"])
    assert all(row["slot_status"] == "PREDECLARED_NOT_YET_ESTIMATED" for row in families["q1_confirmatory"])
    for name, manifest in families["family_manifests"].items():
        assert manifest["family_id"]
        assert len(manifest["family_sha256"]) == 64
        assert manifest["slot_count_including_not_estimable"] == families["family_sizes"][name]
        assert manifest["na_slots_retained"] is True
    verified = verify_statistical_family_manifest(families)
    assert set(verified) == {"q1_confirmatory", "q3_q4_confirmatory", "q3_q4_descriptive"}
    tampered = copy.deepcopy(families)
    tampered["q1_confirmatory"][0]["O_audit"] = "TAMPERED"
    payload = dict(tampered)
    payload.pop("manifest_sha256")
    tampered["manifest_sha256"] = canonical_sha256(payload)
    with pytest.raises(AuditContractError, match="frozen contrast universe"):
        verify_statistical_family_manifest(tampered)


@pytest.mark.parametrize(
    "mutator,error",
    [
        (lambda manifest: manifest["manifest_identity"].update(year=2099), "identity contradicts"),
        (lambda manifest: manifest["family_sizes"].update(q1_confirmatory=999), "family_sizes"),
    ],
)
def test_family_verifier_rejects_rehashed_top_level_identity_and_size_tampering(mutator, error):
    attributes, kwargs = _family_kwargs(scope="anchor", include_new=False)
    manifest = generate_statistical_families(attributes, **kwargs)
    tampered = copy.deepcopy(manifest)
    mutator(tampered)
    payload = dict(tampered)
    payload.pop("manifest_sha256")
    tampered["manifest_sha256"] = canonical_sha256(payload)
    with pytest.raises(AuditContractError, match=error):
        verify_statistical_family_manifest(tampered)


def test_family_verifier_rejects_unrehased_top_level_tampering():
    attributes, kwargs = _family_kwargs(scope="anchor", include_new=False)
    manifest = generate_statistical_families(attributes, **kwargs)
    manifest["manifest_identity"]["year"] = 2099
    with pytest.raises(AuditContractError, match="manifest_sha256 mismatch"):
        verify_statistical_family_manifest(manifest)


@pytest.mark.parametrize(
    "seed_policy,error",
    [
        (
            {
                "status": "FROZEN",
                "seeds": [],
                "aggregation_estimand": "mean",
                "training_randomness_inference": "distribution",
            },
            "nonempty ordered seeds",
        ),
        (
            {
                "status": "FROZEN",
                "seeds": [0, 0],
                "aggregation_estimand": "mean",
                "training_randomness_inference": "distribution",
            },
            "unique",
        ),
        (
            {
                "status": "FROZEN",
                "seeds": [False],
                "aggregation_estimand": "mean",
                "training_randomness_inference": "distribution",
            },
            "non-boolean integers",
        ),
        (
            {
                "status": "FROZEN",
                "seeds": ["0"],
                "aggregation_estimand": "mean",
                "training_randomness_inference": "distribution",
            },
            "non-boolean integers",
        ),
        (
            {
                "status": "FROZEN",
                "seeds": [0],
                "aggregation_estimand": "",
                "training_randomness_inference": "distribution",
            },
            "aggregation_estimand",
        ),
    ],
)
def test_family_generator_rejects_invalid_seed_policy_semantics(seed_policy, error):
    attributes, kwargs = _family_kwargs(
        scope="anchor",
        include_new=False,
        seed_policy=seed_policy,
    )
    with pytest.raises(AuditContractError, match=error):
        generate_statistical_families(attributes, **kwargs)


def _rehash_family_manifest_after_shared_identity_change(manifest):
    shared = manifest["manifest_identity"]
    for family_name, family in manifest["family_manifests"].items():
        for field, value in shared.items():
            family[field] = copy.deepcopy(value)
        family_identity = {
            key: value
            for key, value in family.items()
            if key not in {"family_id", "family_sha256", "na_slots_retained"}
        }
        family_id = f"family_{canonical_sha256(family_identity)[:20]}"
        stripped_slots = []
        for slot in manifest[family_name]:
            for field, value in shared.items():
                slot[field] = copy.deepcopy(value)
            stripped_slots.append(
                {
                    key: value
                    for key, value in slot.items()
                    if key not in {"family_id", "family_sha256"}
                }
            )
        family_hash = canonical_sha256({"identity": family_identity, "slots": stripped_slots})
        family["family_id"] = family_id
        family["family_sha256"] = family_hash
        for slot in manifest[family_name]:
            slot["family_id"] = family_id
            slot["family_sha256"] = family_hash
    payload = dict(manifest)
    payload.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_sha256(payload)


def test_family_verifier_rejects_self_consistent_hollow_seed_policy():
    attributes, kwargs = _family_kwargs(scope="anchor", include_new=False)
    manifest = generate_statistical_families(attributes, **kwargs)
    manifest["manifest_identity"]["seed_policy"] = {"status": "FROZEN"}
    manifest["manifest_identity"]["seed_policy_sha256"] = canonical_sha256({"status": "FROZEN"})
    _rehash_family_manifest_after_shared_identity_change(manifest)
    with pytest.raises(AuditContractError, match="seed_policy fields"):
        verify_statistical_family_manifest(manifest)


def _rehash_family_and_manifest(manifest, family_name):
    family = manifest["family_manifests"][family_name]
    family_identity = {
        key: value
        for key, value in family.items()
        if key not in {"family_id", "family_sha256", "na_slots_retained"}
    }
    family_id = f"family_{canonical_sha256(family_identity)[:20]}"
    stripped_slots = [
        {key: value for key, value in slot.items() if key not in {"family_id", "family_sha256"}}
        for slot in manifest[family_name]
    ]
    family_hash = canonical_sha256({"identity": family_identity, "slots": stripped_slots})
    family["family_id"] = family_id
    family["family_sha256"] = family_hash
    for slot in manifest[family_name]:
        slot["family_id"] = family_id
        slot["family_sha256"] = family_hash
    payload = dict(manifest)
    payload.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_sha256(payload)


def test_audit_attribute_role_is_hash_bound_to_registry_identity():
    with pytest.raises(AuditContractError, match="audit_role_sha256"):
        AuditAttributeSpec(
            variable="SEX_A",
            groups=(1, 2),
            reference_group=1,
            audit_role="anchor",
            O_group_rule_sha256="1" * 64,
            audit_role_sha256=audit_role_sha256(
                "SEX_A",
                "expanded_candidate",
                _O_REGISTRY_VERSION,
                _O_REGISTRY_SHA256,
            ),
            O_registry_version=_O_REGISTRY_VERSION,
            O_registry_sha256=_O_REGISTRY_SHA256,
        )


def test_family_generator_rejects_expanded_role_in_anchor_and_incomplete_expanded_scope():
    receipt = _contrast_universe()
    _, kwargs = _family_kwargs(scope="anchor")
    kwargs["contrast_universe_receipt"] = receipt
    with pytest.raises(AuditContractError, match="scope attributes"):
        generate_statistical_families(_new_attributes(), **kwargs)
    expanded_attributes, expanded_kwargs = _family_kwargs(scope="expanded")
    with pytest.raises(AuditContractError, match="scope attributes"):
        generate_statistical_families(expanded_attributes[:-1], **expanded_kwargs)


def test_contrast_universe_rejects_same_O_as_anchor_and_new_O():
    duplicate = _attribute("SEX_A", (1, 2), 1, "expanded_candidate", "7")
    with pytest.raises(AuditContractError, match="unique and disjoint"):
        freeze_contrast_universe_receipt(
            _anchor_attributes(),
            [duplicate],
            O_registry_version=_O_REGISTRY_VERSION,
            O_registry_sha256=_O_REGISTRY_SHA256,
            selection_rule_id="synthetic_selection_v1",
            selection_rule_sha256="8" * 64,
            q1_model_ids=["fixed_gbdt"],
            confirmatory_method_pairs=[("reweighing", "unmitigated")],
            descriptive_method_pairs=[("joint", "unmitigated")],
            year=2023,
            evidence_role="discovery",
            panel_id="synthetic_panel",
            panel_sha256="a" * 64,
            O_train="SEX_A",
            seed_policy_sha256=canonical_sha256(_SEED_POLICY),
            delta_rationale_id="synthetic_delta",
        )


def test_family_verifier_rejects_same_O_cross_endpoint_group_rule_drift_after_rehash():
    attributes, kwargs = _family_kwargs(scope="expanded")
    manifest = generate_statistical_families(attributes, **kwargs)
    slot = next(
        row
        for row in manifest["q1_confirmatory"]
        if row["O_audit"] == "SEX_A" and row["endpoint"] == "fpr"
    )
    slot["O_group_rule_sha256"] = "f" * 64
    expected_fields = next(
        row
        for row in manifest["contrast_universe_receipt"]["expected_contrasts"]["q1_confirmatory"]
        if row["O_audit"] == "SEX_A" and row["endpoint"] == "fpr"
    )
    body = {field: slot[field] for field in expected_fields if field != "contrast_id"}
    slot["contrast_id"] = f"q1_{canonical_sha256(body)[:20]}"
    _rehash_family_and_manifest(manifest, "q1_confirmatory")
    with pytest.raises(AuditContractError, match="frozen contrast universe"):
        verify_statistical_family_manifest(manifest)


def test_contrast_universe_receipt_tamper_fails_even_when_outer_hash_is_recomputed():
    receipt = _contrast_universe()
    receipt["new_O_set"] = []
    payload = dict(receipt)
    payload.pop("contrast_universe_sha256")
    receipt["contrast_universe_sha256"] = canonical_sha256(payload)
    with pytest.raises(AuditContractError, match="attribute records"):
        verify_contrast_universe_receipt(receipt)


def _scope_row(
    receipt,
    attribute,
    state,
    *,
    layer,
    group=2,
    annual=100,
    domain=80,
    estimation_status="VALID",
    endpoint="fnr",
    model_id="fixed_gbdt",
    delta=0.05,
):
    expected = next(
        row
        for row in receipt["expected_contrasts"]["q1_confirmatory"]
        if row["O_audit"] == attribute
        and row["group"] == group
        and row["endpoint"] == endpoint
        and row["model_id"] == model_id
    )
    return {
        **copy.deepcopy(expected),
        "contrast_key": expected["contrast_id"],
        "contrast_universe_sha256": receipt["contrast_universe_sha256"],
        "delta": delta,
        "delta_units": "absolute_rate_difference",
        "audit_layer": layer,
        "conclusion_state": state,
        "estimation_status": estimation_status,
        "annual_rows": annual,
        "domain_rows": domain,
    }


def _q2_rows(receipt, *, layer, default_state=CONCLUSION_WITHIN):
    role = "anchor" if layer == "anchor" else "expanded_candidate"
    return [
        _scope_row(receipt, row["O_audit"], default_state, layer=layer, group=row["group"])
        for row in receipt["expected_contrasts"]["q1_confirmatory"]
        if row["audit_role"] == role and row["endpoint"] == "fnr" and row["model_id"] == "fixed_gbdt"
    ]


def _replace_state(rows, attribute, state, *, domain=None, estimation_status="VALID"):
    output = copy.deepcopy(rows)
    for row in output:
        if row["O_audit"] == attribute:
            row["conclusion_state"] = state
            row["estimation_status"] = estimation_status
            if domain is not None:
                row["domain_rows"] = domain
    return output


def test_q2_scope_aggregator_separates_content_multiplicity_and_per_O_coverage():
    receipt = _contrast_universe()
    native = _q2_rows(receipt, layer="anchor")
    expanded = _replace_state(native, "SEX_A", CONCLUSION_INSUFFICIENT, domain=95)
    native = _replace_state(native, "SEX_A", CONCLUSION_WITHIN, domain=95)
    native = _replace_state(native, "HISPALLP_A", CONCLUSION_WITHIN, domain=70)
    expanded = _replace_state(expanded, "HISPALLP_A", CONCLUSION_WITHIN, domain=70)
    new_o = _replace_state(
        _q2_rows(receipt, layer="new_O"),
        "VISIONDF_A",
        CONCLUSION_EXCEEDS,
        domain=60,
    )
    result = aggregate_q2_scope_conclusions(
        native,
        expanded,
        new_o,
        contrast_universe_receipt=receipt,
    )
    assert result["scope_priority_high_to_low"] == [
        CONCLUSION_EXCEEDS,
        CONCLUSION_INSUFFICIENT,
        CONCLUSION_WITHIN,
    ]
    assert result["anchor_native_conclusion"] == CONCLUSION_WITHIN
    assert result["anchor_under_expanded_family_conclusion"] == CONCLUSION_INSUFFICIENT
    assert result["expanded_conclusion"] == CONCLUSION_EXCEEDS
    assert result["multiplicity_precision_change"]["changed"] is True
    assert result["new_O_content_change"]["changed"] is True
    assert result["new_limiting_O"] == ["VISIONDF_A"]
    coverage = {row["O_audit"]: row for row in result["per_O_coverage"]}
    assert coverage["SEX_A"]["coverage_fraction"] == pytest.approx(0.95)
    assert coverage["HISPALLP_A"]["coverage_fraction"] == pytest.approx(0.70)
    assert coverage["VISIONDF_A"]["coverage_fraction"] == pytest.approx(0.60)
    assert result["coverage_is_per_O_not_global_complete_case"] is True


def test_q2_scope_aggregator_exceeds_takes_priority_over_insufficient_and_rejects_domain_drift():
    receipt = _contrast_universe()
    anchor = _replace_state(_q2_rows(receipt, layer="anchor"), "SEX_A", CONCLUSION_EXCEEDS)
    anchor_expanded = copy.deepcopy(anchor)
    new_o = _replace_state(
        _q2_rows(receipt, layer="new_O"),
        "VISIONDF_A",
        CONCLUSION_INSUFFICIENT,
        estimation_status="NOT_ESTIMABLE",
    )
    assert aggregate_q2_scope_conclusions(
        anchor,
        anchor_expanded,
        new_o,
        contrast_universe_receipt=receipt,
    )["expanded_conclusion"] == CONCLUSION_EXCEEDS
    drifted = copy.deepcopy(anchor_expanded)
    drifted[0]["domain_rows"] = 79
    with pytest.raises(AuditContractError, match="must not change"):
        aggregate_q2_scope_conclusions(
            anchor,
            drifted,
            new_o,
            contrast_universe_receipt=receipt,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("endpoint", "fpr"),
        ("model_id", "different_model"),
        ("panel_id", "different_panel"),
        ("delta", 0.10),
        ("O_group_rule_sha256", "e" * 64),
    ],
)
def test_q2_rejects_full_scientific_identity_mismatch(field, value):
    receipt = _contrast_universe()
    native = _q2_rows(receipt, layer="anchor")
    expanded = copy.deepcopy(native)
    expanded[0]["conclusion_state"] = CONCLUSION_INSUFFICIENT
    expanded[0][field] = value
    new_o = _q2_rows(receipt, layer="new_O")
    with pytest.raises(AuditContractError, match="scientific contrast identity"):
        aggregate_q2_scope_conclusions(
            native,
            expanded,
            new_o,
            contrast_universe_receipt=receipt,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("year", 2024),
        ("panel_id", "different_panel"),
        ("model_id", "different_model"),
        ("seed_policy_sha256", "f" * 64),
        ("delta", 0.10),
    ],
)
def test_q2_rejects_new_O_cross_layer_shared_scope_drift(field, value):
    receipt = _contrast_universe()
    native = _q2_rows(receipt, layer="anchor")
    expanded = copy.deepcopy(native)
    new_o = _q2_rows(receipt, layer="new_O")
    new_o[0][field] = value
    with pytest.raises(AuditContractError, match="anchor shared-scope identity|frozen contrast universe"):
        aggregate_q2_scope_conclusions(
            native,
            expanded,
            new_o,
            contrast_universe_receipt=receipt,
        )


def test_q2_rejects_inconsistent_group_rule_hash_within_one_new_O():
    vision = [_attribute("VISIONDF_A", (1, 2, 3), 1, "expanded_candidate", "4")]
    receipt = _contrast_universe(new_attributes=vision)
    native = _q2_rows(receipt, layer="anchor")
    expanded = copy.deepcopy(native)
    new_o = _q2_rows(receipt, layer="new_O")
    new_o[1]["O_group_rule_sha256"] = "2" * 64
    with pytest.raises(AuditContractError, match="group-rule identity"):
        aggregate_q2_scope_conclusions(
            native,
            expanded,
            new_o,
            contrast_universe_receipt=receipt,
        )


def test_q2_rejects_same_O_relabelled_as_new_O():
    receipt = _contrast_universe()
    native = _q2_rows(receipt, layer="anchor")
    expanded = copy.deepcopy(native)
    new_o = _q2_rows(receipt, layer="new_O")
    new_o[0]["O_audit"] = "SEX_A"
    new_o[0]["audit_role_sha256"] = audit_role_sha256(
        "SEX_A",
        "expanded_candidate",
        _O_REGISTRY_VERSION,
        _O_REGISTRY_SHA256,
    )
    with pytest.raises(AuditContractError, match="frozen contrast universe"):
        aggregate_q2_scope_conclusions(
            native,
            expanded,
            new_o,
            contrast_universe_receipt=receipt,
        )


def test_q2_rejects_missing_and_extra_slots_relative_to_frozen_universe():
    receipt = _contrast_universe()
    native = _q2_rows(receipt, layer="anchor")
    expanded = copy.deepcopy(native)
    new_o = _q2_rows(receipt, layer="new_O")
    with pytest.raises(AuditContractError, match="omit or add"):
        aggregate_q2_scope_conclusions(
            native[:-1],
            expanded[:-1],
            new_o,
            contrast_universe_receipt=receipt,
        )
    extra = copy.deepcopy(native[0])
    extra["contrast_id"] = extra["contrast_key"] = "q1_" + "e" * 20
    with pytest.raises(AuditContractError, match="omit or add"):
        aggregate_q2_scope_conclusions(
            native + [extra],
            expanded + [copy.deepcopy(extra)],
            new_o,
            contrast_universe_receipt=receipt,
        )


def _no_new_o_receipt(universe, shared):
    return freeze_no_new_O_receipt(
        shared_scope_identity={
            field: shared[field]
            for field in (
                "year",
                "evidence_role",
                "family_name",
                "question",
                "panel_id",
                "panel_sha256",
                "O_train",
                "O_registry_version",
                "O_registry_sha256",
                "contrast_universe_sha256",
                "endpoint",
                "seed_policy_sha256",
                "delta",
                "delta_units",
                "delta_rationale_id",
                "model_id",
            )
        },
        atlas_source={"path": "synthetic/atlas.csv", "sha256": "1" * 64, "variable_count": 31},
        candidate_registry_source={
            "path": "synthetic/candidates.json",
            "sha256": "2" * 64,
            "candidate_denominator": 24,
        },
        excluded_candidate_count=20,
        not_estimable_candidate_count=4,
        precision_interpretation="MIXED_PRECISION_ACROSS_CANDIDATES",
        precision_evidence_source={
            "path": "synthetic/precision.json",
            "sha256": "3" * 64,
            "schema_version": "synthetic_precision_v1",
        },
        selection_rule={
            "selection_rule_id": "synthetic_selection_v1",
            "path": "synthetic/selection.json",
            "schema_version": "synthetic_selection_v1",
            "sha256": "8" * 64,
        },
        O_registry={
            "version": _O_REGISTRY_VERSION,
            "path": "synthetic/O_registry.json",
            "sha256": _O_REGISTRY_SHA256,
        },
        contrast_universe_sha256=universe["contrast_universe_sha256"],
    )


def test_q2_accepts_hash_bound_no_new_O_negative_path_without_affirmative_change():
    receipt = _contrast_universe(include_new=False)
    native = _q2_rows(receipt, layer="anchor")
    expanded = copy.deepcopy(native)
    result = aggregate_q2_scope_conclusions(
        native,
        expanded,
        [],
        contrast_universe_receipt=receipt,
        no_new_O_receipt=_no_new_o_receipt(receipt, native[0]),
    )
    assert result["expanded_conclusion"] == result["anchor_under_expanded_family_conclusion"]
    assert result["expanded_equals_anchor_when_no_new_O"] is True
    assert result["new_O_content_conclusion"] == "NOT_APPLICABLE_NO_NEW_O"
    assert result["new_O_content_change"]["changed"] is False
    assert result["new_O_content_change"]["affirmative_new_O_change_allowed"] is False


def test_q2_no_new_O_path_rejects_missing_tampered_or_inconsistent_receipts():
    receipt = _contrast_universe(include_new=False)
    native = _q2_rows(receipt, layer="anchor")
    expanded = copy.deepcopy(native)
    with pytest.raises(AuditContractError, match="hash-bound no-new-O receipt"):
        aggregate_q2_scope_conclusions(native, expanded, [], contrast_universe_receipt=receipt)
    tampered = _no_new_o_receipt(receipt, native[0])
    tampered["excluded_candidate_count"] = 19
    with pytest.raises(AuditContractError, match="hash mismatch"):
        aggregate_q2_scope_conclusions(
            native,
            expanded,
            [],
            contrast_universe_receipt=receipt,
            no_new_O_receipt=tampered,
        )
    inconsistent = _no_new_o_receipt(receipt, native[0])
    inconsistent["excluded_candidate_count"] = 19
    payload = dict(inconsistent)
    payload.pop("receipt_sha256")
    inconsistent["receipt_sha256"] = canonical_sha256(payload)
    with pytest.raises(AuditContractError, match="do not reconcile"):
        aggregate_q2_scope_conclusions(
            native,
            expanded,
            [],
            contrast_universe_receipt=receipt,
            no_new_O_receipt=inconsistent,
        )


def test_q2_no_new_O_receipt_requires_bound_provenance_and_current_scope():
    receipt = _contrast_universe(include_new=False)
    native = _q2_rows(receipt, layer="anchor")
    expanded = copy.deepcopy(native)
    missing = _no_new_o_receipt(receipt, native[0])
    missing.pop("O_registry")
    payload = dict(missing)
    payload.pop("receipt_sha256")
    missing["receipt_sha256"] = canonical_sha256(payload)
    with pytest.raises(AuditContractError, match="registered schema"):
        aggregate_q2_scope_conclusions(
            native,
            expanded,
            [],
            contrast_universe_receipt=receipt,
            no_new_O_receipt=missing,
        )
    wrong_scope = _no_new_o_receipt(receipt, native[0])
    wrong_scope["shared_scope_identity"]["year"] = 2024
    payload = dict(wrong_scope)
    payload.pop("receipt_sha256")
    wrong_scope["receipt_sha256"] = canonical_sha256(payload)
    with pytest.raises(AuditContractError, match="anchor shared-scope identity"):
        aggregate_q2_scope_conclusions(
            native,
            expanded,
            [],
            contrast_universe_receipt=receipt,
            no_new_O_receipt=wrong_scope,
        )
    assert _no_new_o_receipt(receipt, native[0])["codebook_wide_eligibility_ledger_complete"] is False


def test_holm_preserves_not_estimable_slots_and_does_not_claim_ci():
    adjusted = holm_adjust({"a": 0.01, "b": None, "c": 0.04})
    assert adjusted["family_size_including_not_estimable"] == 3
    rows = {row["contrast_id"]: row for row in adjusted["rows"]}
    assert rows["b"]["raw_status"] == "NOT_ESTIMABLE"
    assert rows["b"]["family_p_for_adjustment"] == 1.0
    assert rows["a"]["holm_adjusted_p"] == pytest.approx(0.03)
    assert "No confidence interval" in adjusted["confidence_interval_note"]


def test_simultaneous_intervals_are_stored_separately_and_project_full_family():
    rng = np.random.default_rng(20260919)
    points = np.array([0.05, -0.02, 0.08, 0.01])
    replicates = rng.normal(points, [0.02, 0.03, 0.025, 0.015], size=(500, 4))
    result = simultaneous_intervals_from_replicates(
        points, replicates, labels=["fnr_g2", "fpr_g2", "fnr_g3", "fpr_g3"]
    )
    assert result["family_size"] == 4
    assert result["simultaneous_critical"] >= result["ordinary_critical"]
    for row in result["rows"]:
        ordinary_width = row["ordinary_ci"][1] - row["ordinary_ci"][0]
        simultaneous_width = row["simultaneous_ci"][1] - row["simultaneous_ci"][0]
        assert simultaneous_width >= ordinary_width
    projected = project_max_absolute_interval(result["rows"])
    assert projected["component_count"] == 4
    assert projected["posthoc_selected_pair_interval"] is False
    assert projected["simultaneous_ci"][1] >= projected["point_estimate"]
    with pytest.raises(AuditContractError, match="full predeclared"):
        project_max_absolute_interval([result["rows"][2]])


def test_three_state_tolerance_and_precision_are_not_significance_shortcuts():
    assert classify_tolerance_interval(-0.02, 0.03, 0.05) == CONCLUSION_WITHIN
    assert classify_tolerance_interval(0.07, 0.12, 0.05) == CONCLUSION_EXCEEDS
    assert classify_tolerance_interval(0.01, 0.09, 0.05) == CONCLUSION_INSUFFICIENT
    assert classify_tolerance_interval(0.0, 0.04, 0.05, nonnegative_estimand=True) == CONCLUSION_WITHIN
    assert classify_tolerance_interval(0.06, 0.12, 0.05, nonnegative_estimand=True) == CONCLUSION_EXCEEDS
    precision = precision_diagnostics(0.01, -0.02, 0.04, 0.05, standard_error=0.015, critical_value=2.0)
    assert precision["supplied_critical_margin_of_error"] == pytest.approx(0.03)
    assert precision["minimum_detectable_effect"] is None
    assert precision["rules_out_effects_with_absolute_magnitude_above_delta"] is True
    assert precision["support_thresholds_are_precision_claim"] is False
    counterexample = precision_diagnostics(0.30, 0.28, 0.32, 0.05)
    assert counterexample["ci_half_width"] == pytest.approx(0.02)
    assert counterexample["rules_out_effects_with_absolute_magnitude_above_delta"] is False
    assert counterexample["rules_out_within_tolerance_region"] is True
    assert counterexample["tolerance_conclusion_state"] == CONCLUSION_EXCEEDS
    assert "precision_sufficient_for_delta" not in counterexample


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (
            dict(
                gap_change=-0.04,
                disadvantaged_group_performance_change=0.03,
                advantaged_group_performance_change=0.0,
                overall_ba_change=0.01,
                overall_risk_quality_change=-0.01,
                any_other_O_worsened=False,
            ),
            "GAP_NARROWED_DISADVANTAGED_GROUP_IMPROVED",
        ),
        (
            dict(
                gap_change=-0.04,
                disadvantaged_group_performance_change=0.0,
                advantaged_group_performance_change=-0.04,
                overall_ba_change=-0.02,
                overall_risk_quality_change=-0.01,
                any_other_O_worsened=False,
            ),
            "GAP_NARROWED_BY_ADVANTAGED_GROUP_LOSS",
        ),
        (
            dict(
                gap_change=-0.01,
                disadvantaged_group_performance_change=-0.03,
                advantaged_group_performance_change=-0.02,
                overall_ba_change=-0.03,
                overall_risk_quality_change=-0.02,
                any_other_O_worsened=False,
            ),
            "ALL_GROUPS_WORSE",
        ),
        (
            dict(
                gap_change=-0.04,
                disadvantaged_group_performance_change=0.03,
                advantaged_group_performance_change=0.0,
                overall_ba_change=0.01,
                overall_risk_quality_change=0.0,
                any_other_O_worsened=True,
            ),
            "TARGET_O_IMPROVED_OTHER_O_WORSENED",
        ),
    ],
)
def test_method_tradeoff_categories_preserve_absolute_changes(kwargs, expected):
    result = classify_method_tradeoff(**kwargs)
    assert result["category"] == expected
    assert "overall_ba_change" in result
    assert "overall_risk_quality_change" in result


@pytest.mark.parametrize(
    "disadvantaged,advantaged,other_worsened,expected",
    [
        (-0.02, -0.05, True, "ALL_GROUPS_WORSE"),
        (-0.02, -0.05, False, "ALL_GROUPS_WORSE"),
        (0.03, 0.00, True, "TARGET_O_IMPROVED_OTHER_O_WORSENED"),
        (0.03, 0.00, False, "GAP_NARROWED_DISADVANTAGED_GROUP_IMPROVED"),
        (0.00, -0.05, True, "GAP_NARROWED_BY_ADVANTAGED_GROUP_LOSS"),
        (0.00, -0.05, False, "GAP_NARROWED_BY_ADVANTAGED_GROUP_LOSS"),
    ],
)
def test_method_tradeoff_overlap_truth_table(disadvantaged, advantaged, other_worsened, expected):
    result = classify_method_tradeoff(
        gap_change=-0.03,
        disadvantaged_group_performance_change=disadvantaged,
        advantaged_group_performance_change=advantaged,
        overall_ba_change=-0.01,
        overall_risk_quality_change=None,
        any_other_O_worsened=other_worsened,
    )
    assert result["category"] == expected


def test_threshold_contract_separates_soft_risk_mean_from_binary_error_rates():
    keys = ["r1", "r2", "r3", "r4"]
    p_event = np.array([0.60, 0.60, 0.40, 0.40])
    y = np.array([1, 1, 0, 0])
    policy = freeze_threshold_policy("synthetic_fixed_half", 0.5)
    y_hat, receipt = bind_thresholded_decisions(keys, p_event, policy)
    assert y_hat.tolist() == [True, True, False, False]
    assert receipt["prediction_sha256"] == canonical_sha256(p_event.tolist())
    assert receipt["threshold_policy_sha256"] == policy["threshold_policy_sha256"]
    assert receipt["y_hat_sha256"] == canonical_sha256([1, 1, 0, 0])
    assert receipt["record_order_sha256"] == record_order_sha256(keys)
    assert float(p_event[y == 1].mean()) == pytest.approx(0.60)
    assert float(y_hat[y == 1].mean()) == pytest.approx(1.0)
    validate_metric_input("fnr", y_hat, output_type="binary_decision_y_hat")
    validate_metric_input("brier", p_event, output_type="event_probability_p")
    with pytest.raises(AuditContractError, match="binary_decision_y_hat"):
        validate_metric_input("fpr", p_event, output_type="event_probability_p")
    with pytest.raises(AuditContractError, match="not scores"):
        validate_binary_decisions(p_event)


def _digest(label):
    return canonical_sha256({"label": label})


def _panel():
    seed_policy = {
        "status": "FROZEN",
        "seeds": [0],
        "aggregation_estimand": "mean_over_registered_training_seeds",
        "training_randomness_inference": "report_seed_distribution_not_survey_replicates",
    }
    threshold = freeze_threshold_policy("synthetic_fixed_half", 0.5)

    def member(member_id, model_id):
        return {
            "member_id": member_id,
            "model_ids": [model_id],
            "source_binding": {
                "path": f"synthetic/{member_id}.json",
                "sha256": _digest(f"source-binding:{member_id}"),
                "json_pointer": f"/members/{member_id}",
            },
            "O_train": "SEX_A",
            "seed_policy_sha256": canonical_sha256(seed_policy),
            "seed_model_bindings": [
                {"seed": 0, "model_id": model_id, "model_sha256": _digest(f"model:{model_id}")}
            ],
            "feature_sha256": _digest("features"),
            "preprocessing_sha256": _digest("preprocessing"),
            "training_source_sha256": _digest("training-source"),
            "prediction_source_code_sha256": _digest(f"prediction-code:{member_id}"),
            "backbone": "GBDT",
            "output_type": "event_probability_p",
            "threshold_policy": threshold,
            "O_at_inference": False,
            "member_contract_complete": True,
        }

    return {
        "panel_id": "main_risk_gbdt_arm001_v3",
        "baseline_member_id": "unmitigated",
        "O_train": "SEX_A",
        "seed_policy": seed_policy,
        "model_id_uniqueness_policy": "UNIQUE_ACROSS_PANEL_MEMBERS_AND_SEEDS",
        "deployment_contract": {
            "backbone": "GBDT",
            "feature_sha256": _digest("features"),
            "preprocessing_sha256": _digest("preprocessing"),
            "output_type": "event_probability_p",
            "O_at_inference": False,
            "threshold_policy_mode": "REUSE_EACH_MEMBER_FROZEN_POLICY_NO_RETUNING",
            "threshold_comparison_operator": ">=",
        },
        "common_domain_contract": {
            "scope": "per_O",
            "all_panel_members_required": True,
            "annual_design_rows_retained": True,
            "global_all_O_complete_case_prohibited": True,
        },
        "members": [
            member("unmitigated", "m0"),
            member("reweighing", "m1"),
        ],
    }


def test_main_panel_is_immutable_and_late_methods_use_supplemental_panel():
    frozen = freeze_comparison_panel(_panel())
    frozen_hash = verify_frozen_panel(frozen)
    mutated = copy.deepcopy(frozen)
    mutated["members"].append({"member_id": "late", "model_ids": ["m2"], "source_binding": "sha2"})
    with pytest.raises(AuditContractError, match="hash mismatch"):
        verify_frozen_panel(mutated)
    supplemental = {
        "panel_id": "supplemental_low_coverage_v3",
        "parent_panel_id": frozen["panel_id"],
        "parent_panel_sha256": frozen_hash,
        "may_redefine_parent_domain": False,
    }
    validate_supplemental_panel(frozen, supplemental)
    supplemental["may_redefine_parent_domain"] = True
    with pytest.raises(AuditContractError, match="must not redefine"):
        validate_supplemental_panel(frozen, supplemental)
    incomplete = _panel()
    incomplete["members"][0]["model_sha256"] = None
    incomplete["members"][0]["member_contract_complete"] = False
    with pytest.raises(AuditContractError, match="explicitly complete"):
        freeze_comparison_panel(incomplete)


def test_panel_verifier_revalidates_schema_without_mutating_input():
    forged = {"panel_id": "forged", "status": "FROZEN", "members": []}
    forged["panel_sha256"] = canonical_sha256(forged)
    before = copy.deepcopy(forged)
    with pytest.raises(AuditContractError, match="missing required fields"):
        verify_frozen_panel(forged)
    assert forged == before


@pytest.mark.parametrize(
    "mutator,error",
    [
        (lambda panel: panel["members"][0].update(O_at_inference=True), "O_at_inference differs"),
        (lambda panel: panel["members"][0].update(output_type="binary_decision_y_hat"), "output_type differs"),
        (lambda panel: panel["members"][0].update(source_binding="unstructured"), "structured path"),
        (
            lambda panel: panel["members"][1].update(model_ids=[panel["members"][0]["model_ids"][0]])
            or panel["members"][1]["seed_model_bindings"][0].update(
                model_id=panel["members"][0]["model_ids"][0]
            ),
            "unique across panel",
        ),
    ],
)
def test_panel_freeze_rejects_deployment_source_and_model_identity_mismatches(mutator, error):
    panel = _panel()
    mutator(panel)
    with pytest.raises(AuditContractError, match=error):
        freeze_comparison_panel(panel)


@pytest.mark.parametrize(
    "field,mode,value,error",
    [
        ("annual_design_rows_retained", "set", False, "retain annual design rows"),
        ("annual_design_rows_retained", "set", 1, "retain annual design rows"),
        ("annual_design_rows_retained", "drop", None, "registered schema"),
        ("global_all_O_complete_case_prohibited", "set", False, "prohibit global"),
        ("global_all_O_complete_case_prohibited", "set", 1, "prohibit global"),
        ("global_all_O_complete_case_prohibited", "drop", None, "registered schema"),
        ("all_panel_members_required", "set", False, "require every member"),
    ],
)
def test_panel_freeze_requires_full_per_O_survey_domain_contract(field, mode, value, error):
    panel = _panel()
    if mode == "drop":
        panel["common_domain_contract"].pop(field)
    else:
        panel["common_domain_contract"][field] = value
    with pytest.raises(AuditContractError, match=error):
        freeze_comparison_panel(panel)


def test_panel_rejects_partial_self_hashed_threshold_policy():
    panel = _panel()
    partial = {"status": "FROZEN", "threshold": 0.5, "comparison_operator": ">="}
    partial["threshold_policy_sha256"] = canonical_sha256(partial)
    panel["members"][0]["threshold_policy"] = partial
    with pytest.raises(AuditContractError, match="threshold policy fields"):
        freeze_comparison_panel(panel)


@pytest.mark.parametrize(
    "mutator,error",
    [
        (lambda panel: panel.update(panel_id=1), "panel_id must be a nonempty string"),
        (lambda panel: panel.update(O_train=True), "O_train must be a nonempty string"),
        (lambda panel: panel["members"][0].update(member_id=1), "member_id must be a nonempty string"),
        (
            lambda panel: panel["members"][0].update(model_ids=[1])
            or panel["members"][0]["seed_model_bindings"][0].update(model_id=1),
            "model_id must be a nonempty string",
        ),
    ],
)
def test_panel_rejects_non_string_identifiers(mutator, error):
    panel = _panel()
    mutator(panel)
    with pytest.raises(AuditContractError, match=error):
        freeze_comparison_panel(panel)


def test_panel_rejects_non_string_threshold_policy_id_and_boolean_seed():
    panel = _panel()
    policy = copy.deepcopy(panel["members"][0]["threshold_policy"])
    policy["policy_id"] = 1
    payload = dict(policy)
    payload.pop("threshold_policy_sha256")
    policy["threshold_policy_sha256"] = canonical_sha256(payload)
    panel["members"][0]["threshold_policy"] = policy
    with pytest.raises(AuditContractError, match="policy_id must be a nonempty string"):
        freeze_comparison_panel(panel)

    panel = _panel()
    panel["seed_policy"]["seeds"] = [False]
    for member in panel["members"]:
        member["seed_policy_sha256"] = canonical_sha256(panel["seed_policy"])
        member["seed_model_bindings"][0]["seed"] = False
    with pytest.raises(AuditContractError, match="non-boolean integers"):
        freeze_comparison_panel(panel)


def test_survey_domain_inference_keeps_zero_contribution_psus_in_full_design():
    # Two strata, two PSUs per stratum. Domain rows contribute from every PSU,
    # while the annual design retains four additional zero-contribution rows.
    strata = np.repeat([1, 2], 4)
    psus = np.array([10, 10, 11, 11, 20, 20, 21, 21])
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    group = np.array([1, 9, 2, 9, 1, 9, 2, 9])
    domain = group != 9
    p_event = np.array([0.9, 0.2, 0.7, 0.1, 0.8, 0.3, 0.6, 0.2])
    record_keys = [f"row-{index}" for index in range(8)]
    q, receipt = bind_thresholded_decisions(
        record_keys,
        p_event,
        freeze_threshold_policy("synthetic_domain_half", 0.5),
    )
    assert receipt["decision_output_type"] == "binary_decision_y_hat"
    result = linearized_survey_inference(
        y,
        {"fixed_model": q},
        group,
        strata,
        psus,
        np.ones(8),
        [1, 2],
        domain_mask=domain,
    )
    assert result["status"] == "VALID"
    assert result["domain_n"] == 4
    assert result["degrees_of_freedom"] == 2


def test_2025_access_remains_locked_until_every_freeze_and_human_release():
    receipt = {
        "validation_year": 2025,
        "status": "LOCKED",
        "candidate_O_frozen": True,
        "delta_approved_with_domain_rationale": False,
        "models_frozen": True,
        "metrics_frozen": True,
        "families_frozen": True,
        "source_code_frozen": True,
        "source_hashes_verified": True,
        "record_order_contract_frozen": True,
        "human_release_decision_recorded": False,
    }
    with pytest.raises(AuditContractError, match="remains locked"):
        assert_2025_performance_access_allowed(receipt)
    receipt["delta_approved_with_domain_rationale"] = True
    receipt["human_release_decision_recorded"] = True
    receipt["status"] = "APPROVED_FOR_ONE_SHOT_2025_EVALUATION"
    assert_2025_performance_access_allowed(receipt)
