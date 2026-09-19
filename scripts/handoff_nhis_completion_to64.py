"""One-shot scheduler replacement. Model processes are never signalled."""
from pathlib import Path
import os, sys, json, signal, time, subprocess

root = Path('/root/autodl-tmp/fairbias_completion_v2_20260917')
old_control = Path('/root/autodl-tmp/fairbias_completion_parallel_20260917')
control = Path('/root/autodl-tmp/fairbias_completion_parallel64_20260917')
sys.path.insert(0, str(root/'scripts'))
sys.path.insert(0, str(control))
from run_nhis_fairbias_completion import load_manifest
from run_nhis_completion_takeover_v2 import sha, atomic, process_identity, process_alive, check_artifact, owned_partition

m = load_manifest(root/'control/manifest.json')
jobs = {j['job_id']: j for j in m['jobs']}
assert '8 passed' in (control/'linux_tests.log').read_text()
assert sha(old_control/'run_nhis_completion_takeover.py') == '8e35a9cc340a1f37531f1a0627246eabe1d5cd6f29b5e3b7f17b5dc5f85ad791'
quota, period = map(int, Path('/sys/fs/cgroup/cpu.max').read_text().split())
assert quota/period >= 64
first = json.loads((old_control/'state.json').read_text())
pid = first['controller_pid']
old = process_identity(pid)
assert old and str(old_control/'run_nhis_completion_takeover.py') in old['argv']
assert first['manifest_sha256'] == m['manifest_sha256']
assert not (control/'handoff.json').exists()
frozen = terminated = False
try:
    os.kill(pid, signal.SIGSTOP)
    frozen = True
    for _ in range(20):
        now = process_identity(pid)
        if now and now['start_ticks'] == old['start_ticks'] and now['state'] == 'T':
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Previous scheduler did not suspend')
    state = json.loads((old_control/'state.json').read_text())
    assert state['controller_pid'] == pid and state['manifest_sha256'] == m['manifest_sha256']
    live = {}
    outputs = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        info = process_identity(int(proc.name))
        if info and str(root/'scripts/run_nhis_fairbias_completion.py') in info['argv'] and info['state'] != 'Z':
            argv = info['argv']
            job = argv[argv.index('--job-id')+1]
            assert job not in live and job in state['active']
            previous = state['active'][job]
            assert previous['pid'] == info['pid'] and previous['start_ticks'] == info['start_ticks']
            assert argv[argv.index('--output')+1] == previous['output']
            assert argv[argv.index('--manifest')+1] == str(root/'control/manifest.json')
            assert argv[argv.index('--prepared')+1] == str(root/'prepared'/(jobs[job]['config']['arm_id']+'.joblib'))
            live[job] = info
            outputs[job] = previous['output']
    completed = dict(state['completed'])
    for job in set(state['active'])-set(live):
        completed[job] = check_artifact(Path(state['active'][job]['output']), job, m['manifest_sha256'], None, adopted=True)
    pending = owned_partition(jobs, live, completed)
    assert len(pending) == len(state['pending']) and set(pending) == set(state['pending'])
    assert all(not (old_control/'jobs'/job).exists() for job in pending)
    assert all(not (root/'runs/full80_v2'/job).exists() for job in pending)
    atomic(control/'old_state_at_handoff.json', state)
    handoff = {'old_run': str(old_control), 'old_controller': old, 'adopted': live,
               'adopted_outputs': outputs, 'completed': completed, 'new_jobs': pending,
               'manifest_sha256': m['manifest_sha256'],
               'new_controller_sha256': sha(control/'run_nhis_completion_takeover_v2.py'),
               'time': time.time(), 'policy': 'User requested 64-way computation on confirmed 64 CPU quota; science and final acceptance unchanged',
               'no_model_process_signalled': True, 'no_frozen_source_changed': True}
    atomic(control/'handoff.json', handoff)
    atomic(control/'resources.json', {'concurrency': 64})
    os.kill(pid, signal.SIGTERM)
    os.kill(pid, signal.SIGCONT)
    for _ in range(30):
        if not process_alive(old):
            terminated = True
            break
        time.sleep(.1)
    if not terminated:
        raise RuntimeError('Previous controller did not exit')
    logfile = (control/'controller.log').open('xb')
    process = subprocess.Popen([sys.executable, '-B', str(control/'run_nhis_completion_takeover_v2.py'),
                                '--root', str(root), '--control', str(control)], cwd=control,
                               env=dict(os.environ, PYTHONPATH=str(root/'src')), stdin=subprocess.DEVNULL,
                               stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {'new_controller_pid': process.pid, 'old_controller_pid': pid, 'adopted_jobs': len(live),
               'completed_at_handoff': len(completed), 'pending_jobs': len(pending), 'time': time.time(),
               'handoff_sha256': sha(control/'handoff.json'), 'concurrency': 64}
    atomic(control/'transition_receipt.json', receipt)
    print(json.dumps(receipt))
finally:
    if frozen and not terminated and process_alive(old):
        os.kill(pid, signal.SIGCONT)
