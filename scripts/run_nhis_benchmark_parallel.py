#!/usr/bin/env python3
"""Bounded parallel development scheduler for the registered NHIS benchmark.

This command runs only the F/C/S development workers.  It never selects a
model or loads the 2024 evaluation partition.  Use a supervisor-authorized
parallel policy JSON when requesting more than one worker.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhis_fairbias.benchmark.parallel_execution import (  # noqa: E402
    ParallelBenchmarkScheduler,
    ResourceLimits,
    SchedulerError,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("develop",), help="run registered F/C/S development jobs")
    parser.add_argument("--run", required=True, type=Path, help="prepared benchmark run directory")
    parser.add_argument("--workers", type=int, default=8, help="concurrent workers (default: 8; maximum: 12)")
    parser.add_argument("--max-rss-gib", type=float, default=64.0,
                        help="aggregate process RSS ceiling in GiB (default: 64; leave host reserve)")
    parser.add_argument("--worker-rss-gib", type=float, default=4.0,
                        help="per-worker process RSS ceiling in GiB (default: 4)")
    parser.add_argument("--fit-seconds", type=float, default=1800.0,
                        help="per-worker wall-clock ceiling (default: 1800)")
    parser.add_argument("--poll-seconds", type=float, default=0.5,
                        help="scheduler polling interval (default: 0.5)")
    parser.add_argument("--methods", nargs="+", help="registered methods to include")
    parser.add_argument("--backbones", nargs="+", help="registered backbones to include")
    parser.add_argument("--max-jobs", type=int, help="bounded development smoke batch")
    parser.add_argument("--parallel-policy", type=Path,
                        help="supervisor-authorized policy JSON required when workers > 1")
    args = parser.parse_args(argv)
    if args.max_rss_gib <= 0 or args.worker_rss_gib <= 0:
        parser.error("memory limits must be positive")
    try:
        limits = ResourceLimits(
            workers=args.workers,
            total_rss_bytes=int(args.max_rss_gib * 1024**3),
            worker_rss_bytes=int(args.worker_rss_gib * 1024**3),
            fit_seconds=args.fit_seconds,
            poll_seconds=args.poll_seconds,
        )
        scheduler = ParallelBenchmarkScheduler(
            args.run.resolve(), runner=ROOT / "scripts/run_nhis_benchmark.py",
            source_script=Path(__file__).resolve(), limits=limits,
            methods=args.methods, backbones=args.backbones, max_jobs=args.max_jobs,
            policy_path=args.parallel_policy,
        )
        summary = scheduler.run()
    except (SchedulerError, ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"parallel scheduler refused to run: {exc}\n")
    print(summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
