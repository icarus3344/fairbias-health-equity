"""Local NHIS schema/support preflight; never fits or evaluates a model."""
import argparse
import datetime
import json
from pathlib import Path
import resource
import sys
import numpy as np
from nhis_fairbias.benchmark.data_contracts import ARM_SPECS, load_arm_partitions, load_local_nhis_cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    frame = load_local_nhis_cohort()
    summary = {'started_utc': started, 'rows_complete_design': len(frame),
               'source_provenance': frame.attrs['source_provenance'], 'arms': {},
               'model_fits': 0, 'role': 'support_and_schema_only'}
    for arm, spec in ARM_SPECS.items():
        summary['arms'][arm] = {}
        for role, part in load_arm_partitions(frame, arm).items():
            design = part.annual_design
            clusters = set(zip(design.strata, design.psus))
            counts = {int(h): sum(s == h for s, _ in clusters) for h in set(design.strata)}
            summary['arms'][arm][role] = {
                'domain_n': len(part), 'annual_design_n': len(design.weights),
                'strata': len(counts), 'psus': len(clusters), 'singleton_strata': int(sum(c < 2 for c in counts.values())),
                'psu_assignment_sha256': part.metadata['psu_assignment_sha256'],
                'groups': {str(g): {'n': int((part.A == g).sum()),
                    'events': int(((part.A == g) & (part.y == 1)).sum()),
                    'nonevents': int(((part.A == g) & (part.y == 0)).sum())} for g in spec['expected_categories']},
            }
    summary['max_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    summary['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with output.open('x') as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print(json.dumps({'complete_rows': len(frame), 'model_fits': 0,
        'max_rss_bytes': summary['max_rss_bytes'], 'domain_counts': {
            a: {r: m['domain_n'] for r, m in p.items()} for a, p in summary['arms'].items()}}, indent=2))


if __name__ == '__main__':
    main()
