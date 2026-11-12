"""Long, evaluator-scripted action tours; these are not model benchmarks.

Reuse the fixed meeting recipes, preserve all earlier captures, and record every
agent action. Privileged controller queries never silently change the camera.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np
from PIL import Image, ImageDraw

import prepare_stage1_meeting_cases as original

ROOT = original.ROOT
from spatial_interaction import EpisodeConfig, make_env
from spatial_interaction.core.trajectory import TrajectoryRecorder, export_debug_page, replay
from spatial_interaction.environments.shape3d.geometry import shortest_alignment


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def annotated_gif(folder):
    rows = [json.loads(s) for s in (folder/'trajectory.jsonl').read_text().splitlines()]
    images, durations = [], []
    for row in rows:
        with Image.open(folder/row['frame']) as raw:
            image = Image.new('RGB', (raw.width, raw.height+62), '#132432')
            image.paste(raw, (0,62))
        draw = ImageDraw.Draw(image)
        name = row['action']['action'] if row['action'] else 'RESET'
        phase = row['observation']['phase'].upper()
        draw.text((10,8), f"SCRIPTED / PRIVILEGED  |  {phase}  |  step {row['step']}/{len(rows)-1}", fill='#8fe0d3')
        draw.text((10,30), name + ('  - submitted' if row['terminated'] else ''), fill='white')
        images.append(image)
        durations.append(1800 if row['terminated'] or name=='MEMORIZE' else 650)
    images[0].save(folder/'replay.gif',save_all=True,append_images=images[1:],duration=durations,loop=0)
    for image in images:
        image.close()


def record(config, out, verify_replay):
    folder = out/'cases'/config.task_type
    env = make_env(config, **({'runtime_dir':out/'thor-runtime'} if config.environment=='dynamic_memory' else {}))
    recorder = None
    try:
        obs, info = env.reset()
        initial = env.privileged_state()
        target = getattr(env,'target',None)
        evaluator = {'privileged':True,'excluded_from_agent_observations':True,'initial':{},'changed':{}}
        if target:
            box = env.controller.last_event.instance_detections2D.get(target)
            evaluator['initial'] = {'bbox':list(map(int,box)) if box is not None else None,
                                    'object_type':initial['initial_state'][target]['objectType']}
        recorder = TrajectoryRecorder(folder,config,debug=True)
        recorder.record(obs,info=info,state=initial)
        total, invalid = 0., []

        def step(action):
            nonlocal obs, info, total
            if isinstance(action,str): action={'action':action}
            obs,reward,terminated,truncated,info = env.step(action)
            recorder.record(obs,action,reward,terminated,truncated,info,env.privileged_state())
            total += reward
            if not info['action_valid']: invalid.append({'step':env.steps,'action':action,'error':info.get('error')})
            if target and env.phase=='detection' and not evaluator['changed']:
                box=env.controller.last_event.instance_detections2D.get(target)
                evaluator['changed']={'bbox':list(map(int,box)) if box is not None else None,
                                      'frame':f'frame_{env.steps:04d}.png'}
            print(config.task_type, env.steps,action['action'],info['action_valid'],flush=True)
            return info['action_valid']

        if target:
            def scan():
                # A full 360-degree native panorama, with pitch changes en route.
                for i in range(12):
                    step('RotateRight')
                    if i==2: step('LookDown')
                    if i==5: step('LookUp')
                # Translations in a verified free neighborhood; no camera teleport.
                event=env.controller.step(action='GetReachablePositions')
                if event.metadata['lastActionSuccess']:
                    agent=event.metadata['agent']
                    p=agent['position'];yaw=np.radians(agent['rotation']['y'])
                    back=np.array([p['x']-.25*np.sin(yaw),p['z']-.25*np.cos(yaw)])
                    points=event.metadata['actionReturn']
                    if points and min(np.linalg.norm(back-np.array([q['x'],q['z']])) for q in points)<.15:
                        if step('MoveBack'): step('MoveAhead')
                step('LookDown');step('LookUp')
            scan()
            step({'action':'MEMORIZE','memory':'Scripted tour checkpoint; no visual inference was performed.'})
            scan()
            step({'action':'SUBMIT','answer':env.privileged_state()['ground_truth']})
        else:
            camera_tour=['CAMERA_RIGHT']*3+['CAMERA_YAW_LEFT','CAMERA_FORWARD',
                'CAMERA_PITCH_DOWN','CAMERA_PITCH_UP','CAMERA_BACKWARD',
                'CAMERA_YAW_RIGHT']+['CAMERA_LEFT']*3
            for action in camera_tour: step(action)
            if config.task_type=='shape_identification':
                for action in ['CAMERA_LEFT']*2+['CAMERA_YAW_RIGHT','CAMERA_YAW_LEFT']+['CAMERA_RIGHT']*2:
                    step(action)
                step({'action':'SUBMIT_SHAPE','candidate_id':env.ground_truth()['candidate_id']})
            else:
                for action in ['OBJECT_ROTATE_X_POS','OBJECT_ROTATE_Y_POS','OBJECT_ROTATE_Z_POS',
                               'OBJECT_ROTATE_X_NEG','OBJECT_ROTATE_Y_NEG','OBJECT_ROTATE_Z_NEG']:
                    step(action)
                for action in shortest_alignment(env.orientation,env.reference,env.cells,env.axes): step(action)
                step('SUBMIT_POSE')
        recorder.close();recorder=None
        assert env.done and env.metrics['success'], (config.task_type,env.metrics)
        state=env.privileged_state()
        write(folder/'ground_truth.json',state)
        if target:
            Image.fromarray(env.initial_rgb).save(folder/'debug_initial.png')
            Image.fromarray(env.changed_rgb).save(folder/'debug_changed.png')
            change=state['ground_truth']['changes'][0];evaluator['change']=change
            reverse={v:k for k,v in state['id_map'].items()}
            evaluator['location_types']={key:state['initial_state'][reverse[change[key]]]['objectType']
                for key in ('previous_location','current_location') if change.get(key) in reverse}
            if config.task_type=='moved_object':
                view=original.native_view(env,target)
                if view:
                    Image.fromarray(view.pop('rgb')).save(folder/'evaluator_current_location.png')
                    evaluator['inspection']={**view,'frame':'evaluator_current_location.png',
                        'caption':'评估者查看移动后位置；相机切换不属于代理轨迹',
                        'not_agent_action':True,'privileged':True}
        write(folder/'evaluator_views.json',evaluator)
        report={'episode_id':config.episode_id,'task_type':config.task_type,'seed':config.seed,
            'policy':'scripted privileged action tour','terminated':True,'truncated':False,'reward':total,
            'metrics':env.metrics,'oracle_uses_privileged_state':True,'illustrative_only':True,
            'invalid_actions':invalid,'demonstrates':'Action effects, not autonomous perception or navigation'}
        if verify_replay:
            report['replay']=replay(env,folder)
        write(folder/'report.json',report)
        annotated_gif(folder);export_debug_page(folder)
        return report
    finally:
        if recorder: recorder.close()
        env.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'05_Experiment/results/stage1-action-tours-20261019')
    parser.add_argument('--tasks',nargs='+',choices=original.TASKS,default=list(original.TASKS))
    parser.add_argument('--native-size',type=int,default=320)
    parser.add_argument('--verify-replay',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    configs=[]
    for task in args.tasks:
        i=original.TASKS.index(task);dynamic=i<2
        cfg={'image_size':args.native_size,'exploration_budget':40,'min_target_pixels':int(1400*(args.native_size/448)**2)} if dynamic else {'image_size':224,'split':'development'}
        configs.append(EpisodeConfig(episode_id='action_tour_'+task,environment='dynamic_memory' if dynamic else 'shape3d',
            task_type=task,scene_id='FloorPlan1' if dynamic else 'procedural',seed=42+i,
            difficulty=3 if task=='moved_object' else 1,max_steps=80,task_config=cfg))
    meta={'run_id':out.name,'purpose':'Long action-effect demonstrations; no model or training claims',
          'created_at_utc':datetime.now(timezone.utc).isoformat(),'state':'RUNNING',
          'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'python':sys.executable,
          'policy':'Privileged scripted panorama / viewpoint / rotation tour',
          'configs':[c.to_dict() for c in configs],
          'source_sha256':{str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    write(out/'run_metadata.json',meta)
    summary={'completed':False,'cases':[]}
    try:
        for config in configs:
            summary['cases'].append(record(config,out,args.verify_replay));write(out/'summary.json',summary)
        summary['completed']=True;meta['state']='COMPLETED'
    except Exception as error:
        meta['state']='FAILED';summary['error']=f'{type(error).__name__}: {error}'
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        meta['ended_at_utc']=datetime.now(timezone.utc).isoformat()
        write(out/'run_metadata.json',meta);write(out/'summary.json',summary)


if __name__=='__main__':main()
