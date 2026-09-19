#!/usr/bin/env python3
"""Explicit catalog freeze, supervisor-released arm evaluation, or shard merge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='phase', required=True)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--selection', type=Path, required=True)
    freeze.add_argument('--selection-sha', required=True)
    freeze.add_argument('--t-provenance', type=Path, required=True)
    freeze.add_argument('--extension-resolutions', type=Path, required=True)
    freeze.add_argument('--supplemental-evidence', type=Path, required=True)
    freeze.add_argument('--output', type=Path, required=True)
    for phase in ('evaluate-arm', 'merge'):
        command = commands.add_parser(phase)
        command.add_argument('--study-freeze', type=Path, required=True)
        command.add_argument('--study-sha', required=True)
        command.add_argument('--release', type=Path, required=True)
        command.add_argument('--release-sha', required=True)
        command.add_argument('--output', type=Path, required=True)
        if phase == 'evaluate-arm':
            command.add_argument('--arm', required=True)
            command.add_argument('--data-root', type=Path, required=True)
        else:
            command.add_argument('--shards', type=Path, required=True,
                help='JSON list of explicit summary path/SHA bindings, one per complete arm')
    args = parser.parse_args(argv)
    from nhis_fairbias.benchmark import catalog_evaluation as evaluation
    read = lambda path: json.loads(path.read_text())
    if args.phase == 'freeze':
        evaluation.freeze_catalog_study(args.selection, args.selection_sha, args.output,
            t_source_provenance=read(args.t_provenance), extension_resolutions=read(args.extension_resolutions),
            supplemental_evidence=read(args.supplemental_evidence))
    elif args.phase == 'evaluate-arm':
        evaluation.evaluate_catalog_arm(args.study_freeze, args.study_sha, args.release, args.release_sha,
            args.output, arm_id=args.arm, data_root=args.data_root)
    else:
        evaluation.merge_catalog_evaluations(args.study_freeze, args.study_sha, read(args.shards), args.output,
            release_path=args.release, expected_release_sha256=args.release_sha)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
