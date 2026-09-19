#!/usr/bin/env python3
"""Separate phases for the additive, known-T completion evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from nhis_fairbias.benchmark import completion_evaluation as c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('admit')
    p.add_argument('--backup', required=True)
    p.add_argument('--parent-archive', required=True)
    p.add_argument('--parent-registration')
    p.add_argument('--output', required=True)
    for name in ('select', 'freeze', 'evaluate', 'merge'):
        p = commands.add_parser(name)
        p.add_argument('--input', required=True)
        p.add_argument('--sha256', required=True)
        p.add_argument('--output', required=True)
        if name == 'evaluate':
            p.add_argument('--arm', choices=c.ARMS, required=True)
            p.add_argument('--release', required=True)
            p.add_argument('--release-sha256', required=True)
        if name == 'merge':
            p.add_argument('--arm-directories', nargs=4, required=True)
    args = parser.parse_args()
    if args.command == 'admit':
        result = c.build_admission(args.backup, args.parent_archive, args.output, parent_registration=args.parent_registration)
    elif args.command == 'select':
        result = c.freeze_completion_selection(args.input, args.sha256, args.output)
    elif args.command == 'freeze':
        result = c.freeze_completion_study(args.input, args.sha256, args.output)
    elif args.command == 'evaluate':
        result = c.evaluate_completion_arm(args.input, args.sha256, args.release, args.release_sha256, args.output, arm_id=args.arm)
    else:
        result = c.merge_completion(args.input, args.sha256, args.arm_directories, args.output)
    print({'phase': args.command, 'output': str(Path(args.output).resolve()),
           'sha256': c.sha(args.output) if Path(args.output).is_file() else c.sha(Path(args.output)/'summary_T.json'),
           'study_branch': result['study_branch'], 'known_T': result['known_T']}, flush=True)


if __name__ == '__main__':
    main()
