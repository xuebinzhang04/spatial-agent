"""Record clear, fixed Stage 1 meeting cases using existing environments.

Oracle diagnostic traces and privileged explanatory views are explicitly separate.
No model inference or training is performed.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import uuid
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT/'04_AgentTask/TaskFrameworks/spatial_interaction'
sys.path.insert(0, str(ENGINE))
from spatial_interaction import EpisodeConfig, make_env
from spatial_interaction.agents.policies import OracleAgent
from spatial_interaction.core.trajectory import TrajectoryRecorder, export_gif, export_debug_page

OUT = ROOT/'05_Experiment/results/stage1-meeting-demo-20261019'
TASKS = ('missing_object','moved_object','shape_identification','pose_alignment')


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def native_view(env, oid):
    c = env.controller
    target = env.privileged_state()['current_state'][oid]['position']
    event = c.step(action='GetReachablePositions')
    if not event.metadata['lastActionSuccess']:
        return None
    positions = sorted(event.metadata['actionReturn'], key=lambda p:abs(math.hypot(p['x']-target['x'],p['z']-target['z'])-1.15))
    best = None
    for position in positions[:10]:
        yaw = math.degrees(math.atan2(target['x']-position['x'],target['z']-position['z']))%360
        for horizon in (20.,35.,50.):
            event = c.step(action='TeleportFull',position=position,rotation={'x':0.,'y':yaw,'z':0.},horizon=horizon,standing=True)
            if not event.metadata['lastActionSuccess']:
                continue
            obj = next((o for o in event.metadata['objects'] if o['objectId']==oid),{})
            box = event.instance_detections2D.get(oid)
            if not obj.get('visible') or box is None:
                continue
            area = int((box[2]-box[0])*(box[3]-box[1]))
            if best is None or area > best['area']:
                best = {'area':area,'rgb':env.render().copy(),'bbox':list(map(int,box)),
                        'camera':event.metadata['agent']}
            if area >= 3000:
                return best
    return best


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'summary.json').exists():
        raise FileExistsError('Meeting recordings already exist; preserve them')
    configs=[]
    for i,task in enumerate(TASKS):
        dynamic = i < 2
        cfg = {'image_size':448,'exploration_budget':12,'min_target_pixels':1400} if dynamic else {'image_size':224,'split':'development'}
        configs.append(EpisodeConfig(episode_id='meeting_'+task,environment='dynamic_memory' if dynamic else 'shape3d',
                         task_type=task,scene_id='FloorPlan1' if dynamic else 'procedural',seed=42+i,
                         difficulty=3 if task=='moved_object' else 1,max_steps=50,task_config=cfg))
    timestamp = datetime.now(timezone.utc).isoformat()
    metadata={'run_id':OUT.name,'project_id':str(uuid.uuid4()),'project_name':'Stage 1 progress meeting demo',
        'task_name':'Four fixed demonstration episodes','task_id':'meeting-recordings',
        'purpose':'Create readable real simulator recordings for a progress meeting; illustrative cases are separate from benchmark validation statistics.',
        'dataset':{'name':'Stage 1 meeting cases','version':'2026-10-19','split':'illustrative development cases',
                   'tasks':list(TASKS),'materials':[c.to_dict() for c in configs]},
        'experiment_type':'oracle visualization capture','model':{'name':'privileged oracle diagnostic','checkpoint':'not applicable','precision':'not applicable'},
        'environment':{'name':'cached THOR 5.0.0 software Vulkan and NumPy shape renderer','version':'existing Stage 1 implementation'},
        'source':{'commit':'Not recorded: no workspace Git metadata','entrypoint':str(Path(__file__).relative_to(ROOT)),'config':'episode.json in each case'},
        'parameters':{'seeds':[42,43,44,45],'native_rgb':448,'shape_view_size':224,'policy':'oracle',
                      'privileged_inspection':'extra evaluator camera views, excluded from agent trajectories'},
        'slurm':{'job_id':'not applicable','partition':'not applicable','resources':'local software CPU render, LP_NUM_THREADS=2',
                 'state':'RUNNING','created_at':timestamp,'started_at':timestamp,'ended_at':'Not recorded','exit_code':'Not recorded','failure_reason':'Not recorded'},
        'artifacts':{'result_dir':str(OUT),'primary_metrics':'summary.json','report':'summary.json','trajectory':'cases/','logs':['capture.log']},
        'notes':'These recordings demonstrate task mechanics, not model reasoning. No VLM calls, learning or GPU allocation. Missing means native deactivation; moved inspection camera is evaluator-only.'}
    write(OUT/'run_metadata.json',metadata)
    summary={'completed':False,'cases':[]}
    try:
        for config in configs:
            folder = OUT/'cases'/config.task_type
            env = make_env(config,**({'runtime_dir':OUT/'thor-runtime'} if config.environment=='dynamic_memory' else {}))
            recorder = None
            try:
                obs, info = env.reset()
                initial_state = env.privileged_state()
                evaluator = {'privileged':True,'excluded_from_agent_observations':True,'initial':{},'changed':{}}
                target = getattr(env,'target',None)
                if target:
                    box = env.controller.last_event.instance_detections2D.get(target)
                    evaluator['initial']={'bbox':list(map(int,box)) if box is not None else None,
                                          'object_type':initial_state['initial_state'][target]['objectType']}
                recorder = TrajectoryRecorder(folder,config,debug=True)
                recorder.record(obs,info=info,state=initial_state)
                policy = OracleAgent(env)
                total = 0.
                while not env.done:
                    action = policy.act(obs,env.available_actions)
                    obs,reward,terminated,truncated,info = env.step(action)
                    total += reward
                    recorder.record(obs,action,reward,terminated,truncated,info,env.privileged_state())
                    if target and env.phase=='detection' and not evaluator['changed']:
                        box = env.controller.last_event.instance_detections2D.get(target)
                        evaluator['changed']={'bbox':list(map(int,box)) if box is not None else None,
                                              'frame':f'frame_{env.steps:04d}.png'}
                recorder.close();recorder=None
                state = env.privileged_state()
                write(folder/'ground_truth.json',state)
                if target:
                    Image.fromarray(env.initial_rgb).save(folder/'debug_initial.png')
                    Image.fromarray(env.changed_rgb).save(folder/'debug_changed.png')
                    change = state['ground_truth']['changes'][0]
                    evaluator['change'] = change
                    reverse = {v:k for k,v in state['id_map'].items()}
                    evaluator['location_types']={}
                    for key in ('previous_location','current_location'):
                        stable = change.get(key)
                        if stable and stable in reverse:
                            obj = state['initial_state'].get(reverse[stable],state['current_state'].get(reverse[stable],{}))
                            evaluator['location_types'][key]=obj.get('objectType',stable)
                    if config.task_type=='moved_object':
                        view = native_view(env,target)
                        if view:
                            Image.fromarray(view.pop('rgb')).save(folder/'evaluator_current_location.png')
                            evaluator['inspection']={**view,'frame':'evaluator_current_location.png',
                                'caption':'评估者查看移动后位置；相机切换不属于代理轨迹',
                                'not_agent_action':True,'privileged':True}
                write(folder/'evaluator_views.json',evaluator)
                result={'episode_id':config.episode_id,'task_type':config.task_type,'seed':config.seed,'policy':'oracle',
                        'terminated':terminated,'truncated':truncated,'reward':total,'metrics':env.metrics,
                        'oracle_uses_privileged_state':True,'illustrative_only':True}
                assert result['metrics']['success'],config.task_type
                write(folder/'report.json',result)
                export_gif(folder);export_debug_page(folder)
                summary['cases'].append(result)
                write(OUT/'summary.json',summary)
                print(config.task_type,'recorded',obs['rgb'].shape,'success',flush=True)
            finally:
                if recorder:recorder.close()
                env.close()
        summary['completed']=True
        metadata['slurm'].update(state='COMPLETED',exit_code=0,failure_reason='not applicable')
    except Exception as error:
        summary['error']=f'{type(error).__name__}: {error}'
        metadata['slurm'].update(state='FAILED',exit_code=1,failure_reason=summary['error'])
        raise
    finally:
        metadata['slurm']['ended_at']=datetime.now(timezone.utc).isoformat()
        write(OUT/'summary.json',summary);write(OUT/'run_metadata.json',metadata)
    print('Meeting recordings complete',flush=True)


if __name__=='__main__':
    main()
