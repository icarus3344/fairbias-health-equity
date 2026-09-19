"""Archive relocation must preserve all frozen identity and integrity checks."""
import json
from pathlib import Path

import pytest

from nhis_fairbias.benchmark.catalog_archive import ArchiveCatalog
from nhis_fairbias.benchmark import catalog_reporting as reporting
from nhis_fairbias.benchmark.result_catalog import CatalogError
from test_catalog_reporting import case, selection_case, _export, sha


def _move(case):
    paths = [case[k] for k in ('study_path', 'selection_path', 'evaluation_path', 'summary_path')]
    hashes = [sha(p) for p in paths]
    old = case['root']
    backup = old.with_name(old.name + '_backup')
    old.rename(backup)
    out = old.with_name(old.name + '_report')
    archive = ArchiveCatalog([{'recorded_prefix': str(old), 'local_root': str(backup)}],
        local_input_roots=[case['study']['analysis_root']], output_dir=out)
    return paths, hashes, backup, out, archive


def _relocated(paths, hashes, out, archive, **kwargs):
    return reporting.write_catalog_reporting(*paths, out, expected_study_sha256=hashes[0],
        expected_selection_sha256=hashes[1], expected_summary_sha256=hashes[3], archive=archive, **kwargs)


def test_moved_archive_exports_identical_tables_without_rewriting_frozen_json(case, monkeypatch):
    original = _export(case)
    prior = {p.name: p.read_bytes() for p in original['csv'].parent.glob('*.csv')}
    paths, hashes, backup, out, archive = _move(case)
    import joblib
    import numpy as np
    monkeypatch.setattr(joblib, 'load', lambda *a, **k: pytest.fail('Model read forbidden'))
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('Prediction read forbidden'))
    result = _relocated(paths, hashes, out, archive)
    assert prior == {p.name: p.read_bytes() for p in out.glob('*.csv')}
    proof = json.loads(result['manifest'].read_text())
    assert proof['denominators']['registered_jobs'] == 12
    assert proof['denominators']['catalog_attempts'] == 13
    assert [sha(archive.physical(p)) for p in paths] == hashes
    assert proof['selected_models'] == case['study']['selected_models']


def test_moved_historical_failure_tamper_still_fails_closed(case):
    paths, hashes, backup, out, archive = _move(case)
    failed = archive.physical(case['base']['jobs']['eg_s0'] / 'result.json')
    failed.write_bytes(failed.read_bytes() + b' ')
    with pytest.raises(reporting.ReportingError, match='FILE_HASH_MISMATCH'):
        _relocated(paths, hashes, out, archive)
    assert not out.exists()


def test_moved_archive_does_not_accept_changed_frozen_analysis_sources(case):
    paths, hashes, backup, out, archive = _move(case)
    empty = backup / 'empty_sources'
    empty.mkdir()
    archive.local_input_roots.append(empty)
    with pytest.raises(reporting.ReportingError, match='ANALYSIS_SOURCE_CLOSURE_MISMATCH'):
        _relocated(paths, hashes, out, archive, analysis_root=empty)
    assert not out.exists()


def test_mapping_is_explicit_longest_prefix_and_confined(tmp_path):
    broad, specific = tmp_path / 'broad', tmp_path / 'specific'
    broad.mkdir(); specific.mkdir()
    mappings = [{'recorded_prefix': '/recorded/server', 'local_root': str(broad)},
                {'recorded_prefix': '/recorded/server/special', 'local_root': str(specific)}]
    archive = ArchiveCatalog(mappings, local_input_roots=[], output_dir=tmp_path / 'report')
    assert archive.physical('/recorded/server/file.json') == broad / 'file.json'
    assert archive.physical('/recorded/server/special/file.json') == specific / 'file.json'
    for invalid, code in [('/unmapped/file', 'UNMAPPED_ARCHIVE_PATH'),
                          ('/recorded/server/../escape', 'INVALID_ARCHIVE_PATH')]:
        with pytest.raises(CatalogError, match=code): archive.physical(invalid)
    (specific / 'escape').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(CatalogError, match='ARCHIVE_PATH_ESCAPE'):
        archive.physical('/recorded/server/special/escape/private')
    with pytest.raises(CatalogError, match='DUPLICATE_ARCHIVE_PREFIX'):
        ArchiveCatalog(mappings + [mappings[0]], local_input_roots=[], output_dir=tmp_path / 'report')
    with pytest.raises(CatalogError, match='OUTPUT_INSIDE_EVIDENCE'):
        ArchiveCatalog(mappings, local_input_roots=[], output_dir=specific / 'report')
    with pytest.raises(CatalogError, match='ARCHIVE_OUTPUT_MISMATCH'):
        archive.protect_output(tmp_path / 'another')
