"""Copy one explicitly sealed CPU manifest after the earlier archive finishes.

No training, source mutation, deletion, broad directory discovery or host shutdown.
The caller first prepares a manifest of completed runs on the CPU server.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(manifest_path, expected_sha, destination, predecessor):
    manifest_path, destination, predecessor = map(Path, (manifest_path, destination, predecessor))
    if sha(manifest_path) != expected_sha:
        raise ValueError('Completed-run manifest changed')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'nhis_completed_cpu_backup_v1' or not manifest.get('files'):
        raise ValueError('Invalid completed-run manifest')
    destination.mkdir(parents=True, exist_ok=False)
    control = destination / 'control'
    control.mkdir()
    (control / 'manifest.json').write_bytes(manifest_path.read_bytes())

    def state(value):
        temporary = control / 'state.json.part'
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(control / 'state.json')

    try:
        listing = []
        for name, record in manifest['files'].items():
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\0' in name or record.get('type') != 'file':
                raise ValueError('Unsafe or unsupported manifest entry')
            listing.append(name)
        files = control / 'files.list'
        files.write_bytes(b'\0'.join(name.encode() for name in sorted(listing)) + b'\0')
        deadline = time.monotonic() + 6 * 3600
        state({'status': 'WAITING_FOR_PREVIOUS_VERIFIED_COPY', 'predecessor': str(predecessor)})
        while not predecessor.exists():
            if (predecessor.parent / 'failure.json').exists() or time.monotonic() > deadline:
                raise RuntimeError('Previous copy failed or exceeded the dependency wait limit')
            time.sleep(60)
        previous = json.loads(predecessor.read_text())
        if (previous.get('status') != 'LOCAL_ARCHIVE_VERIFIED'
                or previous.get('mismatches') != 0
                or previous.get('source_manifest_sha256') != '45090455627b6957238651ae169874fc23f5dd882be704f64c29ef5d9daf24db'):
            raise ValueError('Previous archive completion is not the bound verified archive')
        snapshot = destination / 'snapshot'
        snapshot.mkdir()
        state({'status': 'TRANSFERRING', 'files': len(listing), 'bytes': manifest['total_bytes']})
        # Fixed authorized SSH alias/root; never use a hostname from a manifest.
        subprocess.run(['rsync', '-az', '--partial', '--checksum', '--stats', '--from0',
            '--files-from=' + str(files), '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
            'fairbias-cpu:/root/autodl-tmp/fairbias/', str(snapshot) + '/'], check=True)
        state({'status': 'VERIFYING', 'files': len(listing)})
        for name, record in manifest['files'].items():
            path = snapshot / name
            if (path.is_symlink() or not path.is_file() or path.stat().st_size != record['bytes']
                    or sha(path) != record['sha256']):
                raise ValueError('Completed-run local backup verification failed: ' + name)
        result = {'status': 'LOCAL_COMPLETED_CPU_BACKUP_VERIFIED', 'files': len(listing),
                  'bytes': manifest['total_bytes'], 'manifest_sha256': expected_sha,
                  'mismatches': 0, 'finished_unix': time.time(),
                  'covers_running_bmae_sensitivity': False}
        with (control / 'verification.json').open('x') as f:
            json.dump(result, f, indent=2)
        state(result)
    except BaseException as exc:
        state({'status': 'FAILED', 'error_type': type(exc).__name__})
        with (control / 'failure.json').open('x') as f:
            json.dump({'error_type': type(exc).__name__, 'time': time.time()}, f)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'manifest-sha256', 'destination', 'predecessor'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run(args.manifest, args.manifest_sha256, args.destination, args.predecessor)
