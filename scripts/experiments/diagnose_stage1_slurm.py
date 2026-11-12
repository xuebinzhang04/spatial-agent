"""Read-only scheduler/account probes and sbatch --test-only; never submit a job."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import socket
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'05_Experiment/results/stage1-slurm-diagnostic-20261019'


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT)
    OUT=parser.parse_args().output.resolve()
    OUT.mkdir(exist_ok=True)
    results=[]
    commands=[['hostname'],['id'],['sinfo','-s'],['squeue','--me'],
              ['hyakusage','-a','video'],['sbatch','--test-only',str(ROOT/'scripts/experiments/stage1_action_tours.slurm')]]
    for i,command in enumerate(commands):
        try:
            p=subprocess.run(command,text=True,capture_output=True,timeout=4)
            result={'command':command,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
        except subprocess.TimeoutExpired as e:
            def decode(s):return s.decode(errors='replace') if isinstance(s,bytes) else s or ''
            result={'command':command,'returncode':None,'timeout_seconds':4,'stdout':decode(e.stdout),'stderr':decode(e.stderr)}
        results.append(result)
        (OUT/f'probe-{i}.log').write_text('$ '+' '.join(command)+'\n'+result['stdout']+result['stderr'])
        print(command[0],result['returncode'],flush=True)
    sockets={}
    for name,family in [('AF_INET',socket.AF_INET),('AF_UNIX',socket.AF_UNIX)]:
        try:
            s=socket.socket(family,socket.SOCK_STREAM);s.close();sockets[name]={'creation_allowed':True}
        except OSError as e:
            sockets[name]={'creation_allowed':False,'errno':e.errno,'error':str(e)}
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'commands':results,'socket_creation':sockets,
            'job_submitted':False,'billable_allocation_requested':False,
            'skill_used':'/gpfs/scrubbed/xuebiz/SKILL.md',
            'conclusion':('Client networking is available; inspect scheduler test-only and account results. No actual job submitted.'
                          if sockets['AF_INET']['creation_allowed'] else
                          'Client socket denial prevents checking scheduler; live account/budget not verified.')}
    (OUT/'diagnostic.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(sockets,indent=2))


if __name__=='__main__':main()
