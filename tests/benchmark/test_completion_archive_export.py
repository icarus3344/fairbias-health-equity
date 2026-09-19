"""Generated archive export checks; no model, predictions or NHIS reads."""
import copy
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def archive(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2]/'scripts/merge_nhis_completion_archive.py'
    spec = importlib.util.spec_from_file_location('completion_archive_export', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    provenance = {'raw_sources': {'2024': {'path': 'data/raw/nhis/2024/adult24.csv', 'sha256': '1'*64}},
                  'feature_registry_sha256': '2'*64, 'study_registry_sha256': '3'*64,
                  'identity': 'official HHX + survey year'}
    selections, contrasts, dirs = [], [], []
    for arm in m.ARMS:
        selections.extend({'arm_id': arm, 'selection_id': arm+str(i), 'model_ids': []} for i in range(24))
        contrasts.extend({'contrast_id': arm+str(i), 'reference_selection_id': arm+str(i%24)} for i in range(42))
    study = {'selections': selections, 'contrasts': contrasts, 't_source_provenance': provenance,
             'study_branch': m.BRANCH, 'known_T': True, 'selection': {'sha256': '4'*64}}
    sp = tmp_path/'study.json'
    m.fresh(sp, study)
    study_sha = m.sha(sp)
    monkeypatch.setattr(m, 'validate_study', lambda *args: (study, {}))
    rp = tmp_path/'release.json'
    m.fresh(rp, {'schema_version': 'completion_t_release_v1', 'decision': 'SUPERVISOR_RELEASED_COMPLETION_T',
                 'study_branch': m.BRANCH, 'known_T': True, 'study_sha256': study_sha,
                 'selection_sha256': '4'*64, 'arm_ids': list(m.ARMS)})
    for arm in m.ARMS:
        d = tmp_path/arm
        d.mkdir()
        artifact = d/'statistics.json'
        m.fresh(artifact, {'generated': True})
        observed = copy.deepcopy(provenance)
        observed['raw_sources']['2024']['rows'] = 12
        manifest = {'arm_id': arm, 'status': 'COMPLETE', 'known_T': True, 'study_branch': m.BRANCH,
                    'study_sha256': study_sha, 'source_provenance': observed, 'family_sizes': m.FAMILY_SIZES,
                    'release': m.bound(rp), 'models': [], 'artifacts': {artifact.name: m.sha(artifact)}}
        m.fresh(d/'evaluation_manifest.json', manifest)
        m.fresh(d/'summary_T.json', {'arm_id': arm, 'known_T': True, 'study_branch': m.BRANCH,
                'study_sha256': study_sha, 'evaluation_manifest_sha256': m.sha(d/'evaluation_manifest.json'),
                'selections': [{**r, 'design_n': 12} for r in selections if r['arm_id'] == arm],
                'paired_contrasts': [r for r in contrasts if r['reference_selection_id'].startswith(arm)]})
        dirs.append(d)
    return m, sp, study_sha, dirs, tmp_path/'merged.json'


def test_observed_rows_are_preserved_without_changing_frozen_provenance(archive):
    m, study, digest, dirs, output = archive
    before = study.read_bytes()
    result = m.merge_completion(study, digest, dirs, output)
    assert result['known_T'] is True and len(result['selections']) == 96 and len(result['paired_contrasts']) == 168
    assert result['export_version'] == 'completion_archive_merge_v2' and m.verify(result['export_source']).is_file()
    assert study.read_bytes() == before
    with pytest.raises(ValueError):
        m.merge_completion(study, digest, dirs, output)


@pytest.mark.parametrize('fault', ['raw_hash', 'row_count', 'unknown_metadata', 'artifact_hash', 'duplicate_arm', 'missing_pair'])
def test_export_still_rejects_identity_and_coverage_errors(archive, fault):
    import json
    m, study, digest, dirs, output = archive
    d = dirs[0]
    manifest = m.read(d/'evaluation_manifest.json')
    summary = m.read(d/'summary_T.json')
    if fault == 'raw_hash': manifest['source_provenance']['raw_sources']['2024']['sha256'] = 'f'*64
    elif fault == 'row_count': manifest['source_provenance']['raw_sources']['2024']['rows'] = 13
    elif fault == 'unknown_metadata': manifest['source_provenance']['unfrozen'] = True
    elif fault == 'artifact_hash': (d/'statistics.json').write_text('changed')
    elif fault == 'duplicate_arm': dirs[-1] = dirs[0]
    elif fault == 'missing_pair': summary['paired_contrasts'].pop()
    (d/'evaluation_manifest.json').write_text(json.dumps(manifest))
    summary['evaluation_manifest_sha256'] = m.sha(d/'evaluation_manifest.json')
    (d/'summary_T.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        m.merge_completion(study, digest, dirs, output)
    assert not output.exists()
