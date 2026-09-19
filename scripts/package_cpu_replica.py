"""Package the authorized Linux runtime and frozen development inputs for a CPU replica."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import subprocess

BASE = Path('/root/autodl-tmp')
ROOT = BASE / 'fairbias'
RUN = ROOT / 'artifacts/nhis/benchmark_autodl_20260916_075038Z'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    registration = json.loads((RUN / 'registration.json').read_text())
    for relative, expected in registration['source_files'].items():
        assert sha(ROOT / relative) == expected, relative
    for item in registration['prepared'].values():
        assert sha(Path(item['path'])) == item['sha256']
    output = BASE / 'fairbias_deployment' / 'cpu_replica_20260917'
    output.mkdir(parents=True, exist_ok=False)
    paths = ['envs/fairbias311', 'fairbias/.venv311', 'fairbias/src',
             'fairbias/configs', 'fairbias/scripts', 'fairbias/tests',
             'fairbias/docs/AI_EXECUTION_PROTOCOL.md', 'fairbias/AGENTS.md',
             'fairbias/artifacts/nhis/benchmark_dependencies_20260916',
             str((RUN / 'registration.json').relative_to(BASE)),
             str((RUN / 'prepared').relative_to(BASE))]
    # Include registered files even if future registrations add other source directories.
    covered = lambda p: any(p == root or p.startswith(root + '/') for root in paths)
    paths += [f'fairbias/{p}' for p in registration['source_files'] if not covered(f'fairbias/{p}')]
    partial = output / 'replica.tar.gz.part'
    subprocess.run(['nice', '-n', '10', 'tar', '-I', 'gzip -1', '--exclude=__pycache__',
                    '-cf', str(partial), '-C', str(BASE), *paths], check=True)
    digest = sha(partial)
    target = output / 'replica.tar.gz'
    os.rename(partial, target)
    manifest = {'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'archive': str(target), 'sha256': digest, 'bytes': target.stat().st_size,
                'registration_sha256': sha(RUN / 'registration.json'),
                'source_files': registration['source_files'],
                'prepared': registration['prepared'], 'paths': paths,
                'scope': 'registered development input and exact installed environments; no historical jobs or T2024 evaluation'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'archive': str(target), 'sha256': digest, 'bytes': target.stat().st_size}), flush=True)

if __name__ == '__main__':
    main()
