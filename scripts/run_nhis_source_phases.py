"""Execute a frozen sequence of server-side benchmark phases without an AI monitor."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import socket
import subprocess
import time

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    if config['hostname']!=socket.gethostname():raise ValueError('wrong phase host')
    output=Path(config['audit_directory']);output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):env[name]='1'
    def status(value):
        tmp=output/'status.json.part';tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(output/'status.json')
    identity={'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
              'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'pid':os.getpid(),'hostname':socket.gethostname()}
    (output/'launch.json').write_text(json.dumps(identity,indent=2)+'\n')
    for phase in config['phases']:
        start=time.time();status({**identity,'state':'RUNNING','phase':phase['name'],'started':start})
        with (output/(phase['name']+'.log')).open('x') as log:
            result=subprocess.run(phase['command'],cwd=config['working_directory'],env=env,
                                  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        record={'name':phase['name'],'command':phase['command'],'start':start,'end':time.time(),'returncode':result.returncode}
        (output/(phase['name']+'_exit.json')).write_text(json.dumps(record,indent=2)+'\n')
        if result.returncode:
            status({**identity,'state':'FAILED','phase':phase['name'],'returncode':result.returncode})
            raise SystemExit(result.returncode)
    status({**identity,'state':'COMPLETE','finished':time.time()})

if __name__=='__main__':main()
