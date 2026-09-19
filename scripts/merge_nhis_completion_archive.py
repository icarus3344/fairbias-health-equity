#!/usr/bin/env python3
"""Export already-evaluated completion shards; no model or prediction loading.

v2 accepts the loader's observed annual row count while validating all frozen
source identity fields with the original evaluation provenance validator.
Frozen analysis files and v1 failure evidence are not modified.
"""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nhis_fairbias.benchmark.completion_evaluation import (
    validate_study, validate_release, require, read, verify, sha, bound, fresh,
    ev, ARMS, BRANCH, FAMILY_SIZES,
)

def merge_completion(study_path, study_sha, arm_directories, output):
    study, admission = validate_study(study_path, study_sha)
    require(len(arm_directories) == 4 and not Path(output).exists(), 'FOUR_NEW_ARM_SHARDS_REQUIRED')
    seen, selections, pairs, receipts = set(), [], [], []
    for directory in map(Path, arm_directories):
        manifest_path, summary_path = directory/'evaluation_manifest.json', directory/'summary_T.json'
        manifest, summary = read(manifest_path), read(summary_path)
        arm = manifest['arm_id']
        require(arm in ARMS and arm not in seen and summary['arm_id'] == arm, 'DUPLICATE_OR_WRONG_ARM')
        seen.add(arm)
        require(manifest['status'] == 'COMPLETE' and manifest['known_T'] is summary['known_T'] is True
                and manifest['study_branch'] == summary['study_branch'] == BRANCH
                and manifest['study_sha256'] == summary['study_sha256'] == study_sha
                and summary['evaluation_manifest_sha256'] == sha(manifest_path)
                and manifest['family_sizes'] == FAMILY_SIZES, 'SHARD_IDENTITY_CHANGED')
        ev._validate_observed_provenance(study['t_source_provenance'], manifest['source_provenance'])
        raw = manifest['source_provenance']['raw_sources']['2024']
        require(set(raw) == set(study['t_source_provenance']['raw_sources']['2024']) | {'rows'}
                and type(raw['rows']) is int and raw['rows'] > 0
                and all(r['design_n'] == raw['rows'] for r in summary['selections']), 'ANNUAL_ROWS_MISMATCH')
        validate_release(study, study_sha, read(verify(manifest['release'])), [arm])
        for name, value in manifest['artifacts'].items():
            require(Path(name).name == name and sha(directory/name) == value, 'SHARD_ARTIFACT_CHANGED')
        expected_rows = [r for r in study['selections'] if r['arm_id'] == arm]
        expected_pairs = [c for c in study['contrasts'] if c['reference_selection_id'] in {r['selection_id'] for r in expected_rows}]
        require(len(summary['selections']) == len(expected_rows) and len(summary['paired_contrasts']) == len(expected_pairs)
                and all(all(actual[k] == v for k, v in expected.items()) for actual, expected in zip(summary['selections'], expected_rows))
                and all(all(actual[k] == v for k, v in expected.items()) for actual, expected in zip(summary['paired_contrasts'], expected_pairs))
                and set(manifest['models']) == {m for r in expected_rows for m in r['model_ids']}, 'SHARD_COVERAGE_CHANGED')
        selections.extend(summary['selections'])
        pairs.extend(summary['paired_contrasts'])
        receipts.append({'manifest': bound(manifest_path), 'summary': bound(summary_path)})
    require(len(selections) == 96 and len(pairs) == 168 and seen == set(ARMS), 'MERGED_COVERAGE_CHANGED')
    merged = {'schema_version': 'completion_merged_summary_v1', 'status': 'COMPLETE', 'study_branch': BRANCH,
              'known_T': True, 'export_version': 'completion_archive_merge_v2',
              'export_source': bound(__file__), 'study': bound(study_path), 'shards': receipts, 'family_sizes': FAMILY_SIZES,
              'selections': selections, 'paired_contrasts': pairs,
              'limitations': ['Retrospective computational extension after original T was known',
                              'Fixed FairBias anchors and original comparator selection budgets differ',
                              'Inference conditions on the fitted models; seed variation is separate']}
    fresh(output, merged)
    return merged

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', required=True)
    p.add_argument('--study-sha256', required=True)
    p.add_argument('--arm-directories', nargs=4, required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    r = merge_completion(a.study, a.study_sha256, a.arm_directories, a.output)
    print({'status': r['status'], 'conditions': len(r['selections']), 'pairs': len(r['paired_contrasts']),
           'output': str(Path(a.output).resolve()), 'sha256': sha(a.output)})
