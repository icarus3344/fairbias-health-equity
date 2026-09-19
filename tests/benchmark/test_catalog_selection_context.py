"""Study metadata reuse must preserve the public resolver's full validation."""
import copy

import pytest

from test_catalog_selection import case, write, sha
from nhis_fairbias.benchmark import catalog_selection as selection


def test_context_resolver_verifies_catalog_once_and_matches_public_api(case, monkeypatch):
    artifact = selection.freeze_catalog_selection(case['admission_path'])
    path = case['root'] / 'selection_context.json'
    write(path, artifact)
    expected = selection.resolve_selected_models(path, sha(path))
    calls = []
    original = selection.catalog.build_catalog
    def counted(manifest):
        calls.append(manifest)
        return original(manifest)
    monkeypatch.setattr(selection.catalog, 'build_catalog', counted)
    actual, context = selection.resolve_selected_models_context(path, sha(path))
    assert actual == expected and len(calls) == 1
    assert context['rows'].keys() == case['admission']['resolved_jobs'].keys()
    assert not actual['evaluation_authorized'] and not actual['models_deserialized']


@pytest.mark.parametrize('change', ['wrong_hash', 'missing_selected_model'])
def test_context_resolver_cannot_bypass_artifact_guards(case, change):
    artifact = selection.freeze_catalog_selection(case['admission_path'])
    path = case['root'] / 'selection_context.json'
    if change == 'missing_selected_model':
        artifact = copy.deepcopy(artifact)
        artifact['selected_models'].pop()
    write(path, artifact)
    with pytest.raises(selection.SelectionAdmissionError):
        selection.resolve_selected_models_context(path, '0'*64 if change == 'wrong_hash' else sha(path))
