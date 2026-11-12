"""Prepare/reproduce one bounded live-VLM smoke job using existing assets.

Default is scheduler validation only. --submit submits a single billable GPU job.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    parser.add_argument('--submit',action='store_true')
    parser.add_argument('--native-only',action='store_true',help='Retry only missing/moved tasks; preserve earlier geometric results')
    args=parser.parse_args()
    script=ROOT/'scripts/experiments/stage1_vlm_smoke.slurm'
    usage=subprocess.run(['hyakusage','-a','video'],text=True,capture_output=True,check=True)
    if 'Usage Report for Account' not in usage.stdout or 'Permission denied' in usage.stdout+usage.stderr:
        raise RuntimeError('Account usage cannot be verified; no job submitted')
    tested=subprocess.run(['sbatch','--test-only',str(script)],text=True,capture_output=True)
    print(tested.stdout+tested.stderr,end='')
    tested.check_returncode()
    if not args.submit:
        print('Test only: no job submitted. Use --submit for a single 30-minute GPU smoke job.')
        return
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=(args.output or ROOT/'05_Experiment/results'/('stage1-vlm-smoke-'+stamp)).resolve()
    run.mkdir(parents=True,exist_ok=False)
    files=list((ROOT/'04_AgentTask/TaskFrameworks/spatial_interaction/spatial_interaction').rglob('*.py'))
    files += [script,Path(__file__),ROOT/'scripts/serve_qwen.sh',ROOT/'scripts/experiments/stage1_vlm_smoke.py']
    meta={'run_id':run.name,'project_name':'Spatial Interaction Learning Stage 1',
          'purpose':'Four live VLM episodes, no oracle fallback and no training',
          'created_at_utc':datetime.now(timezone.utc).isoformat(),'state':'PREPARED',
          'model':{'name':'Qwen3.8-27B','checkpoint':str(ROOT/'01_Models/Qwen3.8-27B'),'precision':'bfloat16'},
          'resources':{'account':'video','partition':'gpu-h200','qos':'debug','gpus':1,'cpus':8,'memory':'50G','time_limit':'00:30:00'},
          'parameters':{'max_steps_per_episode':20,'memory_budget':6,'temperature':0,'native_only':args.native_only,
                        'simulator_renderer':'CPU lavapipe; H200 used by VLM'},
          'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    path=run/'run_metadata.json'
    def save():path.write_text(json.dumps(meta,indent=2)+'\n')
    save()
    (run/'account-usage-before.txt').write_text(usage.stdout+usage.stderr)
    (run/'scheduler-test-only.txt').write_text(tested.stdout+tested.stderr)
    exports='ALL,STAGE1_VLM_NATIVE_ONLY='+str(int(args.native_only))+',STAGE1_VLM_RUN='+str(run)
    submitted=subprocess.run(['sbatch','--hold','--parsable','--export='+exports,str(script)],text=True,capture_output=True)
    (run/'submission.log').write_text(submitted.stdout+submitted.stderr)
    if submitted.returncode:
        meta.update(state='SUBMISSION_FAILED',error=submitted.stderr);save();sys.exit(submitted.returncode)
    job=submitted.stdout.strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Unexpected submission response; inspect submission.log')
    meta.update(job_id=job,state='HELD');save()
    released=subprocess.run(['scontrol','release',job],text=True,capture_output=True)
    (run/'release.log').write_text(released.stdout+released.stderr)
    if released.returncode:meta.update(state='RELEASE_FAILED');save();released.check_returncode()
    print(json.dumps({'job_id':job,'run_dir':str(run),'state':'released to scheduler'},indent=2))


if __name__=='__main__':main()
