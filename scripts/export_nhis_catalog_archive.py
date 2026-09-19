#!/usr/bin/env python3
"""Export previously evaluated, immutable study evidence from local backups."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

from nhis_fairbias.benchmark.catalog_archive import ArchiveCatalog
from nhis_fairbias.benchmark.catalog_reporting import write_catalog_reporting
from nhis_fairbias.benchmark.result_catalog import CatalogError, _Files


def export(plan_path, expected_plan_sha256):
    started = time.monotonic()
    plan_path = Path(plan_path).resolve()
    files = _Files({'plan': str(plan_path.parent)}, [])
    files.verify(plan_path, expected_plan_sha256)
    plan = json.loads(plan_path.read_text())
    if plan.get('schema_version') != 'nhis_local_archive_export_plan_v1':
        raise CatalogError('INVALID_EXPORT_PLAN')
    if plan.get('decision') != 'SUPERVISOR_AUTHORIZED_AGGREGATE_EXPORT':
        raise CatalogError('EXPORT_NOT_AUTHORIZED')
    runtime = Path(plan['export_runtime_root']).resolve()
    actual = {p.relative_to(runtime).as_posix() for package in ('nhis_fairbias', 'fairbias')
              for p in (runtime / 'src' / package).rglob('*.py')}
    actual.update({'scripts/evaluate_nhis_catalog.py', 'scripts/export_nhis_catalog_archive.py'})
    if actual != set(plan['export_runtime_files']):
        raise CatalogError('EXPORT_RUNTIME_CLOSURE_MISMATCH')
    # Importing an unbound installed package must not impersonate this runtime.
    import nhis_fairbias.benchmark.catalog_reporting as reporting
    if Path(reporting.__file__).resolve() != runtime / 'src/nhis_fairbias/benchmark/catalog_reporting.py':
        raise CatalogError('WRONG_EXPORT_RUNTIME')
    if Path(__file__).resolve() != runtime / 'scripts/export_nhis_catalog_archive.py':
        raise CatalogError('WRONG_EXPORT_ENTRYPOINT')
    files.source_manifest(runtime, plan['export_runtime_files'])
    out = Path(plan['output_dir']).resolve()
    archive = ArchiveCatalog(plan['mappings'], local_input_roots=[plan['frozen_analysis_root']], output_dir=out)
    inputs = plan['inputs']
    # The evaluation digest is also bound by the frozen summary; check the
    # caller's explicit declaration before creating any output.
    archive._Files({'input': str(Path(inputs['study']['path']).parent)}, []).verify(
        inputs['evaluation']['path'], inputs['evaluation']['sha256'])
    outputs = write_catalog_reporting(inputs['study']['path'], inputs['selection']['path'],
        inputs['evaluation']['path'], inputs['summary']['path'], out,
        expected_study_sha256=inputs['study']['sha256'], expected_selection_sha256=inputs['selection']['sha256'],
        expected_summary_sha256=inputs['summary']['sha256'], analysis_root=plan['frozen_analysis_root'], archive=archive)
    files.source_manifest(runtime, plan['export_runtime_files'])
    receipt = {'schema_version': 'nhis_local_archive_export_receipt_v1',
        'status': 'EXPORT_COMPLETE_REVIEW_REQUIRED', 'evaluation_authorized': False,
        'export_version': plan['export_version'], 'plan_path': str(plan_path), 'plan_sha256': expected_plan_sha256,
        'inputs': inputs, 'frozen_analysis_root': plan['frozen_analysis_root'],
        'export_runtime_files': plan['export_runtime_files'], 'mappings': plan['mappings'],
        'environment': {'python': sys.version, 'platform': platform.platform(), 'executable': sys.executable},
        'elapsed_seconds': time.monotonic() - started,
        'output_files': {p.name: files.sha(p) for p in sorted(out.iterdir())}}
    (out / 'export_receipt.json').write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    result = export(args.plan, args.plan_sha256)
    print(json.dumps({k: result[k] for k in ('status', 'export_version', 'elapsed_seconds')}))
