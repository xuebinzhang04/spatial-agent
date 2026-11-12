"""Submit the authorized one-H200 meeting demo after live account/queue checks."""
import argparse
import json
from pathlib import Path
import subprocess
import uuid
ROOT = Path(__file__).resolve().parents[2]
PY = '/gpfs/scrubbed/xuebiz/conda/envs/spatialagent/bin/python'
p = argparse.ArgumentParser()
p.add_argument('--map', default='small-city-11')
a = p.parse_args()
for cmd in (['sinfo','-s'], ['squeue','--me'], ['hyakusage']):
    subprocess.run(cmd, cwd=ROOT, check=True, timeout=30)
(ROOT/'05_Experiment/logs').mkdir(exist_ok=True)
run = ROOT / '05_Experiment/results' / ('interaction-demo-prepared-' + uuid.uuid4().hex[:12])
subprocess.run([PY,str(ROOT/'scripts/experiments/interaction_demo_record.py'),'prepare','--run-dir',str(run),'--map',a.map],check=True)
job = None
try:
    # Hold prevents a race between run record creation and execution.
    result = subprocess.run(['sbatch','--hold','--parsable',f'--export=ALL,DEMO_MAP={a.map}',str(ROOT/'scripts/experiments/simworld_interaction_demo.slurm')],check=True,text=True,capture_output=True,cwd=ROOT)
    job = result.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError(result.stdout)
    target = run.parent / ('interaction-demo-' + job)
    run.rename(target)
    subprocess.run([PY,str(ROOT/'scripts/experiments/interaction_demo_record.py'),'submitted','--run-dir',str(target),'--job-id',job],check=True)
    subprocess.run(['scontrol','release',job],check=True)
    print(f'Submitted {job}; artifacts: {target}',flush=True)
except Exception:
    if job and job.isdigit():
        subprocess.run(['scancel',job])
    raise
