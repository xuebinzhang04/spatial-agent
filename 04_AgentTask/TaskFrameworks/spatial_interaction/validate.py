#!/usr/bin/env python3
"""Reproduce unit tests, oracle/random rollouts and replay; writes experiment metadata."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import uuid
from spatial_interaction import EpisodeConfig, make_env
from spatial_interaction.benchmarks import generate_episodes
from spatial_interaction.core.episode import save_episodes
from spatial_interaction.core.trajectory import replay
from spatial_interaction.environments.dynamic_memory.backend import workspace_root
from spatial_interaction.runner import run_episode


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,help='New run directory; existing artifacts are never overwritten')
    parser.add_argument('--thor',action='store_true',help='Include real Unity tasks and tests')
    parser.add_argument('--splits',action='store_true',help='Validate four recipes in each of all three splits')
    args=parser.parse_args()
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    task_root=Path(__file__).resolve().parent
    project_id=os.environ.get('SPATIAL_PROJECT_ID',str(uuid.uuid4()))
    configurations=[]
    for split in (('development','validation','test') if args.splits else ('development',)):
        configs=generate_episodes(4 if args.thor else 2,split,environment=None if args.thor else 'shape3d')
        configurations.extend(EpisodeConfig(**{**c.to_dict(),'task_config':{**c.task_config,'image_size':128}}) for c in configs)
    metadata={'run_id':output.name,'project_id':project_id,'project_name':'Stage 1 simulator correctness validation',
        'task_name':'Four spatial interaction task types' if args.thor else 'Shape3D tasks','task_id':'stage1-validation',
        'purpose':'Verify reset/actions/interventions/answers/rewards/replay and oracle solvability; no claim about model reasoning.',
        'dataset':{'name':'spatial-interaction-stage1','version':'0.1.0','split':'all' if args.splits else 'development',
                   'tasks':[c.episode_id+':'+c.task_type for c in configurations],
                   'materials':['procedural shape seeds',str(workspace_root()/'02_Simulator/AI2THOR/runtime/releases')]},
        'experiment_type':'functional validation','model':{'name':'oracle/random','checkpoint':'not applicable','precision':'not applicable'},
        'environment':{'name':'THOR 5.0.0 cached CloudRendering + NumPy/Pillow' if args.thor else 'NumPy/Pillow',
                       'version':sys.version,'python':sys.executable,'host':socket.gethostname(),
                       'software_vulkan':os.environ.get('SPATIAL_THOR_SOFTWARE','0')},
        'source':{'commit':'Not recorded (no workspace Git metadata)','entrypoint':str(Path(__file__).resolve()),
                  'config':'episodes.jsonl','patches':'source_hashes.json'},
        'parameters':{'episodes':[c.to_dict() for c in configurations], 'policies':['oracle','random'],
                      'tests':'unittest discover -s tests -v','thor_start_timeout':35,'thor_action_timeout':20,
                      'random_max_steps':16,'native_replay_max_pixel_error':3,'native_replay_max_mean_pixel_error':.5},
        'slurm':{'job_id':os.environ.get('SLURM_JOB_ID','not applicable'),
                 'partition':os.environ.get('SLURM_JOB_PARTITION','not applicable'),
                 'resources':'software CPU validation' if os.environ.get('SPATIAL_THOR_SOFTWARE')=='1' else 'Not recorded',
                 'state':'RUNNING','created_at':now(),'started_at':now(),'ended_at':'Not recorded',
                 'exit_code':'Not recorded','failure_reason':'Not recorded'},
        'artifacts':{'result_dir':str(output),'primary_metrics':'validation.json','report':'validation.json',
                     'trajectory':'episodes/','logs':['tests.log']},
        'notes':'Oracle reads privileged state. Random policy and model adapter receive observations only. '
                'Missing means native deactivation. Native RGB has bounded raster variation; shape pixels are exact. '
                'No model inference/training or downloads.'}
    meta_path=output/'run_metadata.json'
    meta_path.write_text(json.dumps(metadata,indent=2))
    save_episodes(configurations,output/'episodes.jsonl')
    hashes={str(p.relative_to(task_root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(task_root.rglob('*')) if p.is_file() and
            (p.suffix in ('.py','.toml') or 'scene_states' in p.parts)}
    (output/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    report={'completed':False,'unit_tests':{},'oracles':[],'random':[],'replays':[]}
    try:
        environment={**os.environ,'SPATIAL_TEST_THOR':'1' if args.thor else '0'}
        with (output/'tests.log').open('w') as log:
            result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=task_root,
                                  env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        report['unit_tests']={'exit_code':result.returncode,'log':'tests.log'}
        if result.returncode:
            raise RuntimeError('Unit/integration suite failed; inspect tests.log')
        print('Unit/integration tests passed',flush=True)
        for config in configurations:
            env=make_env(config,**({'runtime_dir':output/'thor-runtime'} if config.environment=='dynamic_memory' else {}))
            try:
                directory=output/'episodes'/config.episode_id/'oracle'
                result=run_episode(config,directory,'oracle',True,env)
                report['oracles'].append(result)
                if not result['metrics']['success']:
                    raise RuntimeError(f'Oracle failed: {config.episode_id}')
                report['replays'].append({'episode_id':config.episode_id,**replay(env,directory)})
                # Bound random validation time and exercise automatic phase expiry.
                random_config=EpisodeConfig(**{**config.to_dict(),'max_steps':16,
                    'task_config':{**config.task_config,**({'exploration_budget':4} if config.environment=='dynamic_memory' else {})}})
                env.config=random_config
                report['random'].append(run_episode(random_config,output/'episodes'/config.episode_id/'random','random',False,env))
                print(config.episode_id,config.task_type,'oracle/replay/random passed',flush=True)
            finally:
                env.close()
            (output/'validation.json').write_text(json.dumps(report,indent=2))
        for split in ('development','validation','test'):
            configs=[c for c in configurations if c.episode_id.startswith(split+'_')]
            if configs:
                save_episodes(configs,output/f'{split}.jsonl')
        report['completed']=True
        metadata['slurm'].update(state='COMPLETED',exit_code=0)
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'
        metadata['slurm'].update(state='FAILED',exit_code=1,failure_reason=report['error'])
        raise
    finally:
        (output/'validation.json').write_text(json.dumps(report,indent=2))
        metadata['slurm']['ended_at']=now()
        meta_path.write_text(json.dumps(metadata,indent=2))
    print(json.dumps({'completed':True,'oracles':len(report['oracles']),'replays':len(report['replays']),
                      'random_rollouts':len(report['random']),'output':str(output)},indent=2))


if __name__=='__main__':
    main()
