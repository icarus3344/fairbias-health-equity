"""Precompile hash-checked bytecode for unchanged registered source only.

No fitting, model loading, data loading or changes to scientific Python source.
Fresh workers using -B can read this cache; -B only prevents automatic writes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registered_files(registration):
    value = json.loads(registration.read_text())
    result = []
    for rel, expected in value["source_files"].items():
        p = (ROOT / rel).resolve()
        if not p.is_relative_to(ROOT) or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ValueError("unsafe registered source path")
        if sha(p) != expected:
            raise ValueError(f"registered source differs: {rel}")
        if p.suffix == ".py":
            result.append(p)
    return result


def time_imports(count):
    values = []
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    for _ in range(count):
        start = time.perf_counter()
        p = subprocess.run([sys.executable, "-B", "-c",
            "from nhis_fairbias.benchmark.experiment_worker import execute_job"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        if p.returncode:
            raise RuntimeError(f"import probe failed: {p.stderr[-1500:]}")
        values.append(time.perf_counter()-start)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    registration = args.registration.resolve()
    out = args.output.resolve()
    if out.exists() or out.is_relative_to(registration.parent):
        parser.error("output must be a fresh directory outside the active run")
    if not 1 <= args.repeats <= 5:
        parser.error("repeats must be between 1 and 5")
    sources = registered_files(registration)
    reg_sha = sha(registration)
    out.mkdir(parents=True)
    original_cache = {}
    for p in sources:
        cached = Path(importlib.util.cache_from_source(str(p)))
        original_cache[str(cached.relative_to(ROOT))] = sha(cached) if cached.exists() else None
    before = time_imports(args.repeats)
    manifest = []
    for p in sources:
        cached = Path(py_compile.compile(str(p), doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH))
        content = cached.read_bytes()
        if (content[:4] != importlib.util.MAGIC_NUMBER or int.from_bytes(content[4:8], "little") != 3 or
                content[8:16] != importlib.util.source_hash(p.read_bytes())):
            raise RuntimeError("generated cache failed header/source-hash verification")
        manifest.append({"source": str(p.relative_to(ROOT)), "source_sha256": sha(p),
            "bytecode": str(cached.relative_to(ROOT)), "bytecode_sha256": sha(cached),
            "invalidation_mode": "CHECKED_HASH"})
    registered_files(registration)
    if sha(registration) != reg_sha:
        raise RuntimeError("registration changed during precompilation")
    after = time_imports(args.repeats)
    report = {"diagnostic_only": True, "optimization": "hash_checked_source_bytecode",
        "python": sys.executable, "python_version": sys.version, "timestamp": time.time(),
        "helper_sha256": sha(Path(__file__)), "registration_sha256": reg_sha,
        "all_registered_sources_unchanged": True, "fitting_or_microdata_access": False,
        "original_bytecode": original_cache, "compiled": manifest,
        "import_seconds_before": before, "import_seconds_after": after,
        "median_seconds_before": statistics.median(before), "median_seconds_after": statistics.median(after),
        "median_delta_seconds": statistics.median(before)-statistics.median(after),
        "measurement_caveat": "Small sequential sample while main queue runs; warm-cache and load effects are not isolated. Not an end-to-end speedup claim."}
    with (out / "bytecode_report.json").open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k:report[k] for k in ("all_registered_sources_unchanged", "median_seconds_before", "median_seconds_after", "median_delta_seconds")}))


if __name__ == "__main__":
    main()
