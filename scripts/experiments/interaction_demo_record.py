"""Prepare and update the required record for the manual simulator demo."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / '04_AgentTask/SpatialAgentBench/DeliveryBench'

def now():
    return datetime.now(timezone.utc).isoformat()

def command(*args, cwd=ROOT):
    p = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    return p.stdout.strip() if p.returncode == 0 else 'Not recorded'

p = argparse.ArgumentParser()
p.add_argument('mode', choices=['prepare', 'submitted', 'started', 'ended'])
p.add_argument('--run-dir', type=Path, required=True)
p.add_argument('--job-id')
p.add_argument('--map', default='small-city-11')
p.add_argument('--exit-code', type=int)
a = p.parse_args()
run = a.run_dir.resolve()
run.mkdir(parents=True, exist_ok=True)
f = run / 'run_metadata.json'
if a.mode == 'prepare':
    materials = [f'04_AgentTask/SpatialAgentBench/DeliveryBench/maps/{a.map}/{n}' for n in ('roads.json', 'buildings.json', 'progen_world_enriched.json')]
    materials += ['04_AgentTask/SpatialAgentBench/DeliveryBench/vlm_delivery/input/' + n for n in ('game_mechanics_config.json', 'food.json', 'special_notes.json', 'store_items.json')]
    for path in materials:
        if not (ROOT / path).is_file():
            raise FileNotFoundError(path)
    mechanics = json.loads((TASK / 'vlm_delivery/input/game_mechanics_config.json').read_text())
    d = {'run_id': run.name, 'project_id': 'agent-interaction-' + uuid.uuid4().hex,
         'project_name': 'SimWorld × DeliveryBench meeting demo', 'task_name': 'Manual Agent Interaction',
         'task_id': f'{a.map}-seed0-manual',
         'purpose': 'Demonstrate Environment variables, native DeliveryBench Task APIs, and manual/script Harness feedback with real UE images; verify turn/step/order interaction, not model performance.',
         'dataset': {'name': 'DeliveryBench', 'version': command('git', 'rev-parse', 'HEAD', cwd=TASK), 'split': 'demo', 'tasks': [a.map], 'materials': materials, 'sha256': {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in materials}},
         'experiment_type': 'interactive integration demo; scripted initial tour; no comparison baseline',
         'model': {'name': 'None: human/script policy', 'checkpoint': 'Not applicable', 'precision': 'Not applicable'},
         'environment': {'name': 'SimWorld/UE DeliveryBench-specific gym_citynav build', 'version': '64a4889d72a2c7b792d242083e53a926b43ceaa9', 'runtime': '02_Simulator/DeliveryBench/Linux/gym_citynav', 'mechanics': mechanics},
         'source': {'commit': command('git', 'rev-parse', 'HEAD'), 'task_commit': command('git', 'rev-parse', 'HEAD', cwd=TASK),
                    'local_patches': command('git', 'status', '--short', cwd=TASK),
                    'entrypoint': '04_AgentTask/Demos/spatial_agent/run.py',
                    'launch_script': 'scripts/experiments/simworld_interaction_demo.slurm', 'config': 'run-specific checkout/task/DeliveryBench/vlm_delivery/input/experiment_config.json'},
         'parameters': {'seed': 0, 'map': a.map, 'task_limit': 100, 'time_scale': 1.0,
                        'action_timeout_s': 900, 'sim_tick_ms': 100, 'vlm_pump_ms': 200,
                        'resolution': [640, 480], 'port': 8767, 'ue_port': 9000,
                        'policy': 'manual API calls with initial script: VIEW_ORDERS, TURN_AROUND(60,left), STEP_FORWARD, ACCEPT_ORDER(first available ID)',
                        'lifecycle': mechanics['lifecycle']},
         'slurm': {'job_id': 'Not recorded', 'partition': 'gpu-h200', 'account': 'video', 'qos': 'normal',
                   'resources': {'gpus': 1, 'cpus': 8, 'host_ram_gb': 50, 'wall_minutes': 15},
                   'state': 'prepared', 'created_at': now(), 'started_at': 'Not recorded', 'ended_at': 'Not recorded', 'exit_code': 'Not recorded', 'failure_reason': 'Not recorded'},
         'artifacts': {'result_dir': str(run.relative_to(ROOT)), 'primary_metrics': 'Not applicable: not a benchmark evaluation',
                       'report': 'reports/', 'trajectory': 'interaction_trace.jsonl', 'snapshots': 'snapshots/', 'logs': ['simulator.log', 'bridge.log']},
         'notes': 'Gym reward is always zero. Demo stop is a demonstration cutoff, not native benchmark completion. Isolated input checkout avoids changing shared configuration. Custom ability task cards are proposals, not implemented evaluators.'}
else:
    d = json.loads(f.read_text())
    if a.mode == 'submitted':
        d['run_id'] = run.name
        d['artifacts']['result_dir'] = str(run.relative_to(ROOT))
        d['slurm'].update(job_id=a.job_id, state='submitted', created_at=now())
        d['artifacts']['logs'].append(f'05_Experiment/logs/interaction-demo-{a.job_id}.log')
    elif a.mode == 'started':
        d['slurm'].update(state='running', started_at=now(), node=os.uname().nodename)
    else:
        d['slurm'].update(state='completed' if a.exit_code == 0 else 'failed', ended_at=now(), exit_code=a.exit_code,
                          failure_reason='Not applicable' if a.exit_code == 0 else 'Inspect bridge.log and simulator.log')
    if a.mode in ('started', 'ended'):
        out = command('scontrol', 'show', 'job', str(d['slurm']['job_id']))
        (run / f'slurm-{a.mode}.txt').write_text(out)
f.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')
print(run, flush=True)
