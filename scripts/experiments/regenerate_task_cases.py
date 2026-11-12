"""Freeze fresh eight-task recipes, then execute and inspect them in an allocation.

Preparation is lightweight. Execution retains all outcomes, including invalid resets
and ineffective model behavior; it never substitutes Oracle actions for the model.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'04_AgentTask/TaskFrameworks'))
sys.path.insert(0,str(ROOT/'04_AgentTask/Demos/spatial_agent'))
from task_support import SingleTask
from analyze_interaction import write_analysis

TASKS=('missing_object','moved_object','shape_identification','pose_alignment',
       'memory_maze','dependency_doors','relational_rearrangement','dynamic_replanning')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix('.pending.json')
    pending.write_text(json.dumps(value,indent=2)+'\n')
    pending.replace(path)


def runtime_sources():
    paths=list((ROOT/'04_AgentTask/TaskFrameworks/spatial_interaction/spatial_interaction').rglob('*.py'))
    paths += [ROOT/'04_AgentTask/CustomAgentTasks'/task/'env.py' for task in TASKS]
    paths += [ROOT/'04_AgentTask/Demos/spatial_agent/session.py',Path(__file__),
              ROOT/'04_AgentTask/TaskFrameworks/task_support.py',ROOT/'scripts/experiments/analyze_interaction.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}


def prepare(output,count,seed):
    if output.exists():raise FileExistsError('Choose a new output directory; previous cases are preserved')
    output.mkdir(parents=True)
    expectation_path=ROOT/'04_AgentTask/CustomAgentTasks/expectations.json'
    expectations=json.loads(expectation_path.read_text())
    plan=[]
    for index,task in enumerate(TASKS):
        definition=SingleTask(ROOT/'04_AgentTask/CustomAgentTasks'/task)
        for case in range(count):
            level=1+case%2
            config={'image_size':224}
            if task in ('missing_object','moved_object'):
                config.update(memory_protocol='active_scan_v1',exploration_budget=96)
            if task=='shape_identification':
                config.update(require_initial_pixel_ambiguity=True,initial_rotation=0,voxels=6)
            if task=='memory_maze':
                config.update(memory_protocol='explicit_recall_v1',exploration_budget=120)
            specification=definition.generate(seed=seed+index*100+case,difficulty=level,
                max_steps=240 if definition.definition['environment']!='shape3d' else 60,task_config=config)
            name=f'{task}-seed{specification.seed}-L{level}'
            write(output/'recipes'/f'{name}.json',specification.to_dict())
            plan.append({'case':name,'task':task,'recipe':f'recipes/{name}.json',
                         'environment':'shape3d' if specification.environment=='shape3d' else 'native',
                         'status':'recipe_generated_not_executed','expectation':expectations['tasks'][task]})
    write(output/'plan.json',{'schema':'spatial-case-regeneration/1','cases':plan,
        'expectation_sha256':hashlib.sha256(expectation_path.read_bytes()).hexdigest(),
        'runtime_sha256':runtime_sources(),
        'no_object_id_overlays':True,'status':'recipes_only_not_demo_ready'})
    now=datetime.now(timezone.utc).isoformat()
    write(output/'run_metadata.json',{'run_id':output.name,'project_id':str(uuid.uuid4()),
        'project_name':'Interaction-first case regeneration','task_name':'Eight spatial task families','task_id':'eight-task-regeneration-v1',
        'purpose':'Regenerate and inspect cases against explicit ability expectations; preserve ineffective behavior for diagnosis.',
        'dataset':{'name':'Fresh procedural development recipes','version':'expectations/1','split':'development','tasks':list(TASKS),'materials':['plan.json','recipes/']},
        'experiment_type':'Recipe generation; simulator/model execution pending',
        'model':{'name':'Qwen3.8-27B planned, not executed','checkpoint':'01_Models/Qwen3.8-27B','precision':'Not loaded'},
        'environment':{'name':'AI2THOR and Shape3D planned','version':'No fresh simulator run'},
        'source':{'entrypoint':'scripts/experiments/regenerate_task_cases.py','config':'plan.json','commit':'Not recorded',
                  'sha256':{str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}},
        'parameters':{'seed':seed,'cases_per_task':count,'total_cases':len(plan),'native_max_steps':240,'shape_max_steps':60},
        'slurm':{'job_id':'Not submitted','partition':'Not applicable to recipe preparation','resources':'Light CPU preparation','state':'NOT_SUBMITTED',
                 'created_at':now,'started_at':'Not started','ended_at':'Not completed','exit_code':'Not run','failure_reason':'Native/model execution requires an allocation and reachable model service'},
        'artifacts':{'result_dir':str(output.relative_to(ROOT)),'primary_metrics':'execution.json (pending)','report':'plan.json','trajectory':'runs/ (pending)'},
        'notes':'Recipes are not valid scenes, successful rollouts or new demo media. No old examples are relabelled as new.'})
    return {'recipes':len(plan),'tasks':len(TASKS),'output':str(output),'status':'not_executed'}


def inspect_setting(env):
    """Privileged diagnostics only; never added to policy observations."""
    task=env.config.task_type
    checks={}
    if env.config.environment=='dynamic_memory':
        from spatial_interaction.environments.dynamic_memory.interventions import objects
        checks['target_outside_initial_view']=not objects(env.controller)[env.target].get('visible',False)
        checks['explicit_commit_protocol']=env.active_memorization
        checks['survey_budget_feasible']='Unverified: requires replay of a complete room survey'
        checks['translation_necessary']='Unverified: stationary-view visibility sweep required'
    elif task=='shape_identification':
        import numpy as np
        from spatial_interaction.environments.shape3d.renderer import render
        target=render(env.cells,env.orientation,env.camera,env.yaw,env.pitch,env.size*2,env.occlusion)
        twins=[c for i,c in enumerate(env.candidates) if f'candidate_{i+1}'!=env.correct and
               np.array_equal(target,render(c,env.orientation,env.camera,env.yaw,env.pitch,env.size*2,env.occlusion))]
        checks['initial_render_ambiguous']=bool(twins)
        checks['distinguishing_translation_exists']=bool(env.ambiguity_witness)
    elif task=='pose_alignment':
        from spatial_interaction.environments.shape3d.geometry import rotation_error,shortest_alignment
        checks['initial_pose_unsolved']=rotation_error(env.orientation,env.reference,env.cells)>1e-6
        checks['rotation_witness_within_budget']=len(shortest_alignment(env.orientation,env.reference,env.cells,env.axes))+1<=env.config.max_steps
    elif task=='memory_maze':
        checks['starts_in_exploration']=env.phase=='exploration'
        checks['explicit_commit_protocol']=env.explicit_recall
        checks['memory_benefit']='Unverified: matched memory-retention ablation required'
    elif task=='dependency_doors':
        from spatial_interaction.environments.dynamic_memory.interventions import objects
        state=objects(env.controller)
        checks['closed_sources']=all(not state[o['source']]['isOpen'] for o in env.orders)
        checks['immediate_finish_fails']=not env.evaluate()['success']
    elif task=='relational_rearrangement':
        checks['initial_relation_unsolved']=not env.goal_matches()
        checks['immediate_finish_fails']=not env.evaluate()['success']
        checks['distinct_valid_invalid_slots']=env.goal_slot!=env.wrong_slot
    elif task=='dynamic_replanning':
        checks['blocked_edge_witness']=bool(env.blocked_edge)
        checks['longer_feasible_detour']=env.new_route_length>=env.old_route_length+2
        checks['immediate_finish_fails']=not env.evaluate()['success']
    return {'checks':checks,'failed_checks':[k for k,v in checks.items() if v is False],
            'unverified_checks':[k for k,v in checks.items() if isinstance(v,str)]}


def execute(output,model,model_url):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Case execution requires a Slurm allocation; preparation is available without one')
    from session import Session
    from spatial_interaction import EpisodeConfig
    plan=json.loads((output/'plan.json').read_text())
    expectation_path=ROOT/'04_AgentTask/CustomAgentTasks/expectations.json'
    if plan.get('runtime_sha256')!=runtime_sources():
        raise ValueError('Runtime source changed; prepare a fresh batch rather than changing frozen cases')
    if plan['expectation_sha256']!=hashlib.sha256(expectation_path.read_bytes()).hexdigest():
        raise ValueError('Expectations changed; prepare a new frozen plan')
    metadata=json.loads((output/'run_metadata.json').read_text())
    metadata['experiment_type']='Uncorrected VLM case inspection; native admission incomplete'
    metadata['model']['name']=model
    metadata['slurm'].update(job_id=os.environ['SLURM_JOB_ID'],state='RUNNING',
        started_at=datetime.now(timezone.utc).isoformat(),resources='See allocated job record',failure_reason='Not recorded')
    write(output/'run_metadata.json',metadata)
    results=[]
    write(output/'summary.json',{'model':model,'scope':'Fresh case inspection; not a calibrated benchmark'})
    for case in plan['cases']:
        run=output/'runs'/case['case']
        if run.exists():raise FileExistsError('Preserve existing attempts; prepare a new batch to retry')
        session=Session(run,model_url,model)
        cfg=EpisodeConfig(**json.loads((output/case['recipe']).read_text()))
        recorder=None
        try:
            state=session.reset({'environment':case['environment'],'task':cfg.task_type,
                'seed':cfg.seed,'difficulty':cfg.difficulty,'budget':cfg.max_steps,'scene':cfg.scene_id},specification=cfg)
            session.metadata['project_id']=json.loads((output/'run_metadata.json').read_text())['project_id']
            session.metadata['source']['entrypoint']='scripts/experiments/regenerate_task_cases.py'
            session.metadata['purpose']='Uncorrected VLM case inspection against frozen interaction expectations'
            session.save_metadata()
            setting=inspect_setting(session.adapter.env)
            write(run/'setting-review.json',setting)
            if setting['failed_checks']:
                raise ValueError('Setting does not support expected interaction: '+', '.join(setting['failed_checks']))
            from spatial_interaction.core.trajectory import TrajectoryRecorder
            recorder=TrajectoryRecorder(run/'recording',cfg,debug=True)
            recorder.record(session.adapter.observation,info=session.adapter.info,state=session.adapter.env.privileged_state())
            while not state['terminated'] and not state['truncated']:
                result=session.infer(True)
                state=result['state']
                recorder.record(session.adapter.observation,result['action'],state['reward'],state['terminated'],state['truncated'],
                                session.adapter.info,session.adapter.env.privileged_state())
                write_analysis(run,run/'interaction-analysis.json')
            write(run/'recording/report.json',{'completed':True,'policy':'vlm','oracle_uses_privileged_state':False,
                'metrics':state['evaluation'],'reward':state['total_reward'],'terminated':state['terminated'],'truncated':state['truncated']})
            write(run/'recording/ground_truth.json',session.adapter.env.privileged_state())
            (run/'recording/model_calls.jsonl').symlink_to('../model_calls.jsonl')
            result={'case':case['case'],'status':'executed_not_admitted','evaluation':state.get('evaluation'),
                    'review':{'expected':case['expectation']['expected'],
                              'actual':'See interaction-analysis.json and complete session.jsonl',
                              'shortcut_controls':setting,
                              'demo_ready':False}}
            session.metadata['parameters'].update(frozen_recipe=cfg.to_dict())
        except Exception as error:
            result={'case':case['case'],'status':'failed','error':str(error),'demo_ready':False}
            (run/'traceback.log').write_text(traceback.format_exc())
            session.metadata['slurm'].update(state='FAILED',exit_code=1,failure_reason=str(error))
        finally:
            if recorder:recorder.close()
            session.close()
        write(run/'case-review.json',result)
        results.append(result)
        write(output/'execution.json',{'cases':results,'demo_ready':False,
            'note':'Actual model attempts only. Shortcut controls, native/public replay and review must precede promotion.'})
    metadata['slurm'].update(state='COMPLETED' if all(r['status']=='executed_not_admitted' for r in results) else 'PARTIAL',
        ended_at=datetime.now(timezone.utc).isoformat(),exit_code=0)
    write(output/'run_metadata.json',metadata)
    return {'cases':len(results),'executed':sum(r['status']=='executed_not_admitted' for r in results),'demo_ready':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--count',type=int,default=3)
    parser.add_argument('--seed',type=int,default=1042)
    parser.add_argument('--model',default='Qwen3.8-27B')
    parser.add_argument('--model-url',default='http://127.0.0.1:8000/v1')
    args=parser.parse_args()
    if args.count<1 or args.seed<0:parser.error('count must be positive and seed nonnegative')
    result=prepare(args.output.resolve(),args.count,args.seed) if args.mode=='prepare' else execute(args.output.resolve(),args.model,args.model_url)
    print(json.dumps(result))


if __name__=='__main__':main()
