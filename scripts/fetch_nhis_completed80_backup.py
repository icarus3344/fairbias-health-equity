"""Download and independently verify a sealed completed-run backup from westb."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import subprocess
import tarfile
import time


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def run():
    destination = Path('artifacts/nhis/completed80_backup_20260918').resolve()
    control = destination/'control'
    source = '/root/autodl-tmp/fairbias_completion_delivery_20260918'
    host = 'root@connect.westb.seetacloud.com'
    key = str(Path.home()/'.ssh/fairbias_westb_35430')
    ssh = ['ssh', '-i', key, '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
           '-o', 'ConnectTimeout=15', '-p', '35430', host]
    response = subprocess.run(ssh+['cat '+source+'/archive_receipt.json'], check=True, capture_output=True)
    receipt = json.loads(response.stdout)
    assert receipt['status'] == 'REMOTE_BACKUP_SEALED'
    (control/'archive_receipt.json').write_bytes(response.stdout)
    for name, expected in [('files_manifest.json', receipt['files_manifest_sha256']),
                           ('audit.json', receipt['audit_sha256'])]:
        target = control/name
        assert not target.exists()
        r = subprocess.run(ssh+['cat '+source+'/'+name], check=True, capture_output=True)
        tmp = target.with_suffix('.part')
        tmp.write_bytes(r.stdout)
        assert sha(tmp) == expected
        tmp.replace(target)
    audit = json.loads((control/'audit.json').read_text())
    assert audit['status'] == 'ALL_80_FC_ARTIFACTS_AND_RELOADS_VERIFIED' and len(audit['jobs']) == 80
    assert audit['auditor_sha256'] == (control/'auditor_source_sha256.txt').read_text().strip()
    index = json.loads((control/'files_manifest.json').read_text())
    assert index['audit_sha256'] == receipt['audit_sha256']
    print(json.dumps({'phase': 'TRANSFER', 'compressed_bytes': receipt['archive_bytes'],
                      'files': receipt['files'], 'uncompressed_bytes': receipt['bytes']}), flush=True)
    archive = destination/'completed80.tar.gz'
    assert not archive.exists()
    tmp = destination/'completed80.tar.gz.part'
    subprocess.run(['scp', '-i', key, '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
                    '-o', 'ConnectTimeout=15', '-P', '35430', host+':'+source+'/completed80.tar.gz',
                    str(tmp)], check=True)
    assert tmp.stat().st_size == receipt['archive_bytes'] and sha(tmp) == receipt['archive_sha256']
    tmp.replace(archive)
    print(json.dumps({'phase': 'LOCAL_ARCHIVE_HASH_VERIFIED'}), flush=True)
    snapshot = destination/'snapshot'
    snapshot.mkdir(exist_ok=False)
    files = index['files']
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == len(files) == len({m.name for m in members})
        for member in members:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts
            assert member.name in files and member.size == files[member.name]['bytes']
            target = snapshot/member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix+'.part')
            h = hashlib.sha256()
            stream = tar.extractfile(member)
            assert stream is not None
            with temporary.open('wb') as output:
                for block in iter(lambda: stream.read(1048576), b''):
                    output.write(block)
                    h.update(block)
            assert h.hexdigest() == files[member.name]['sha256']
            temporary.replace(target)
    for name, record in files.items():
        p = snapshot/name
        assert p.stat().st_size == record['bytes'] and sha(p) == record['sha256']
    verified = {'status': 'LOCAL_ALL80_BACKUP_VERIFIED_STOP_READY', 'finished_unix': time.time(),
                'files': len(files), 'bytes': sum(v['bytes'] for v in files.values()),
                'archive_sha256': receipt['archive_sha256'],
                'manifest_sha256': receipt['files_manifest_sha256'], 'audit_sha256': receipt['audit_sha256'],
                'mismatches': 0, 'remote_host': 'connect.westb.seetacloud.com:35430',
                'source': source, 'snapshot': str(snapshot), 'shutdown_performed': False,
                'preserve_instance_and_disks': True, 'all_paper_work_complete': False}
    with (control/'verification.json').open('x') as f:
        json.dump(verified, f, indent=2)
        f.write('\n')
    print(json.dumps(verified), flush=True)


if __name__ == '__main__':
    run()
