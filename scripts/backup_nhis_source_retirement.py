"""Frozen research-file manifest and verified, read-only server-to-server pull.

Installed environments/caches are excluded. The caller records pip freeze
separately. No command in this module shuts down a host or deletes source data.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time

DIRECTORIES = ('artifacts', 'data', 'src', 'scripts', 'configs', 'docs', 'tests')
ROOT_PATTERNS = ('requirements*.txt', 'environment*.yml', 'environment*.yaml',
                 'pyproject.toml', 'setup.cfg', 'Pipfile*', 'uv.lock', 'AGENTS.md')
EXCLUDED = ('.venv*', 'venv*', 'site-packages', '__pycache__', '.cache',
            'pip-cache', 'wheels', 'benchmark_dependencies*', 'frappe_env')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def excluded(name):
    return any(fnmatch.fnmatchcase(name, p) for p in EXCLUDED)


def entry(path):
    if path.is_symlink():
        return {'type': 'symlink', 'target': os.readlink(path)}
    if not path.is_file():
        raise ValueError('Unsupported backup file type: ' + str(path))
    before = path.stat()
    digest = sha(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('File changed during hashing: ' + str(path))
    return {'type': 'file', 'bytes': after.st_size, 'sha256': digest}


def manifest(root):
    root = Path(root).resolve()
    files = {}
    for top in DIRECTORIES:
        for folder, dirs, names in os.walk(root / top, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not excluded(d))
            symlinks = [d for d in dirs if (Path(folder) / d).is_symlink()]
            dirs[:] = [d for d in dirs if d not in symlinks]
            for name in sorted(names + symlinks):
                if excluded(name):
                    continue
                path = Path(folder) / name
                files[path.relative_to(root).as_posix()] = entry(path)
    for path in sorted(root.iterdir()):
        if any(fnmatch.fnmatchcase(path.name, p) for p in ROOT_PATTERNS):
            files[path.name] = entry(path)
    return {'schema': 'nhis_retirement_manifest_v1', 'created_unix': time.time(),
            'source_root': str(root), 'excluded_components': EXCLUDED, 'files': files,
            'total_bytes': sum(r.get('bytes', 0) for r in files.values())}


def pull(source, destination, control, ssh_command):
    destination, control = Path(destination), Path(control)
    destination.mkdir(parents=True, exist_ok=True)
    control.mkdir(parents=True, exist_ok=True)
    common = ['rsync', '-a', '--partial', '--stats', '-e', ssh_command]
    subprocess.run(common + [source + '.retirement_export/', str(control / 'source_export') + '/'], check=True)
    manifest_path = control / 'source_export' / 'manifest.json'
    recorded = json.loads(manifest_path.read_text())
    if recorded.get('schema') != 'nhis_retirement_manifest_v1' or not recorded.get('files'):
        raise ValueError('Missing frozen source manifest')
    for name in recorded['files']:
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\0' in name:
            raise ValueError('Unsafe manifest path')
    file_list = control / 'files.list'
    file_list.write_bytes(b'\0'.join(name.encode() for name in sorted(recorded['files'])) + b'\0')
    subprocess.run(common + ['--from0', '--files-from=' + str(file_list), source,
                            str(destination) + '/'], check=True)
    failures = []
    for name, expected in recorded['files'].items():
        try:
            if entry(destination / name) != expected:
                failures.append(name)
        except (OSError, ValueError):
            failures.append(name)
    if failures:
        raise ValueError('Backup verification failed: ' + repr(failures[:10]))
    result = {'status': 'CPU_ARCHIVE_VERIFIED', 'finished_unix': time.time(),
              'source_manifest_sha256': sha(manifest_path), 'file_count': len(recorded['files']),
              'total_bytes': recorded['total_bytes'], 'source': source,
              'destination': str(destination), 'source_deleted': False,
              'local_copy_complete': False, 'mismatches': 0}
    with (control / 'verification.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    m = sub.add_parser('manifest')
    m.add_argument('--root', type=Path, required=True)
    m.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('pull')
    for name in ('source', 'destination', 'control', 'ssh-command'):
        p.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if args.mode == 'manifest':
        result = manifest(args.root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as f:
            json.dump(result, f, indent=2)
        print(json.dumps({'files': len(result['files']), 'bytes': result['total_bytes'],
                          'manifest_sha256': sha(args.output)}), flush=True)
    else:
        pull(args.source, args.destination, args.control, args.ssh_command)


if __name__ == '__main__':
    main()
