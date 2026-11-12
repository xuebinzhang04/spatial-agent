"""Native simulator admission checks, per-task artifacts, and oracle diagnostics."""
import argparse
from datetime import datetime,timezone
import json
import hashlib
import shutil
from pathlib import Path
import sys
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'04_AgentTask/TaskFrameworks/spatial_interaction'))
from spatial_interaction import EpisodeConfig,make_env
from spatial_interaction.agents.policies import RandomAgent
from spatial_interaction.runner import run_episode
from spatial_interaction.core.trajectory import replay


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def checked_agent_step(env,action):
    result=env.step(action)
    assert result[-1]['action_valid'],(action,result[-1])
    return result


def navigate(env,pose):
    for _ in range(160):
        action=env.nav_action(pose)
        if action is None:return
        checked_agent_step(env,action)
    raise AssertionError('Native scripted navigation exceeded bound')


def relational_negatives(env):
    env.reset();navigate(env,env.pickup_view)
    checked_agent_step(env,{'action':'PickupObject','object_id':env.sid(env.target)})
    navigate(env,env.wrong_view)
    checked_agent_step(env,{'action':'PLACE','slot_id':env.wrong_slot})
    result=checked_agent_step(env,{'action':'FINISH'})
    assert not result[-1]['metrics']['success'] and not result[-1]['metrics']['relation_satisfied']
    env.reset()
    while not env.done:
        action=env.oracle_action()
        if action['action']=='FINISH':break
        checked_agent_step(env,action)
    assert env.goal_matches()
    navigate(env,env.view_of(env.anchor,36))
    checked_agent_step(env,{'action':'PickupObject','object_id':env.sid(env.anchor)})
    result=checked_agent_step(env,{'action':'FINISH'})
    assert not result[-1]['metrics']['success'] and not result[-1]['metrics']['protected_objects_preserved']



def replanning_negatives(env):
    env.reset();env._teleport(env.goal_view)
    result=env.step({'action':'MARK','object_id':env.sid(env.target)})
    assert not result[-1]['action_valid'] and env.progress==0
    env.reset();navigate(env,env.checkpoint_view)
    assert env.changed and env.progress==1
    a,b=env.blocked_edge
    import math
    pose={'position':{'x':a[0],'y':env.checkpoint_view['position']['y'],'z':a[1]},
          'rotation':{'x':0.,'y':math.degrees(math.atan2(b[0]-a[0],b[1]-a[1]))%360,'z':0.},
          'cameraHorizon':30.,'isStanding':True}
    env._teleport(pose)
    before=env.controller.last_event.metadata['agent']['position'].copy()
    result=env.step({'action':'MoveAhead'})
    assert not result[-1]['action_valid']
    after=env.controller.last_event.metadata['agent']['position']
    assert np.linalg.norm(np.array([before[a]-after[a] for a in 'xyz']))<.02
    result=checked_agent_step(env,{'action':'FINISH'})
    assert not result[-1]['metrics']['success']



def repair_negatives(env):
    env.reset()
    dirty=next(oid for oid in env.faults if env.initial_state[oid].get('dirtyable'))
    env._teleport(env.views[dirty])
    result=env.step({'action':'CLEAN','object_id':env.sid(dirty)})
    assert not result[-1]['action_valid'] and env.controller.last_event.metadata['objects']
    checked_agent_step(env,{'action':'INSPECT','object_id':env.sid(dirty)})
    result=env.step({'action':'CLEAN','object_id':env.sid(dirty)})
    assert not result[-1]['action_valid']
    result=env.step({'action':'ACTIVATE'})
    assert not result[-1]['action_valid'] and not env.activated
    env.reset();healthy=next(iter(env.healthy));env._teleport(env.views[healthy])
    checked_agent_step(env,{'action':'PickupObject','object_id':env.sid(healthy)})
    result=checked_agent_step(env,{'action':'FINISH'})
    assert not result[-1]['metrics']['success'] and not result[-1]['metrics']['healthy_preserved']


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=('memory_maze','dependency_doors','relational_rearrangement','dynamic_replanning'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',type=int,default=3);p.add_argument('--levels',type=int,nargs='+',choices=(1,2,3));args=p.parse_args();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    custom=ROOT/'04_AgentTask/CustomAgentTasks'
    task_dir=custom/args.task
    sources=list(task_dir.glob('*.py'))+[task_dir/'task.json',Path(__file__)]
    sources+=list((ROOT/'04_AgentTask/TaskFrameworks/spatial_interaction/spatial_interaction').rglob('*.py'))
    hashes={}
    for source in sources:
        relative=source.relative_to(ROOT);hashes[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
        target=out/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    write(out/'runtime-source-sha256.json',hashes)
    report={'completed':False,'task':args.task,'oracles':[],'replays':[],'checks':[],
            'scope':'Native simulator mechanical/negative checks, not VLM accuracy'}
    env=None
    try:
        indices=[level-1 for level in args.levels] if args.levels else range(args.cases)
        for i in indices:
            cfg=EpisodeConfig(episode_id=f'{args.task}_L{i+1}_seed{42+i}',environment='thor_tasks',task_type=args.task,
                scene_id=f'FloorPlan{i+1}',seed=42+i,difficulty=i+1,max_steps=360,
                task_config={'split':('development','validation','test')[i],'image_size':128,'exploration_budget':12})
            print('Validating',cfg.episode_id,cfg.scene_id,flush=True)
            env=make_env(cfg,runtime_dir=out/'thor-runtime');env.privileged_references=True  # scripted negatives use private IDs
            obs,info=env.reset();state=env.privileged_state()
            if i==0:
                write(out/'simulator_capabilities.json',list(state['current_state'].values()))
            assert obs['rgb'].dtype==np.uint8 and obs['rgb'].shape==(128,128,3)
            assert not {'ground_truth','current_state','agent','id_map','verified_views'}&set(obs)
            _,_,terminated,truncated,info=env.step({'action':'FINISH'})
            assert terminated and not truncated and not info['metrics']['success']
            report['checks'].append(cfg.episode_id+': premature FINISH rejected by verifier')
            result=run_episode(cfg,out/'episodes'/cfg.episode_id,policy='oracle',debug=True,env=env)
            report['oracles'].append(result)
            assert result['metrics']['success'],result
            report['replays'].append(replay(env,out/'episodes'/cfg.episode_id))
            write(out/'validation.json',report)
            if i==0:
                if args.task=='relational_rearrangement':
                    relational_negatives(env)
                    report['checks'].append('actual wrong-slot placement and protected-anchor pickup both rejected')
                if args.task=='dynamic_replanning':
                    replanning_negatives(env)
                    report['checks'].append('goal marking before checkpoint, real blocked movement and FINISH before goal all fail')
                if args.task=='signal_repair':
                    repair_negatives(env)
                    report['checks'].append('repair before inspection, repair without tool, early activation and healthy-object manipulation all fail')
                obs,_=env.reset();before=env.progress
                _,reward,t,tr,info=env.step({'action':'not_an_action'})
                assert not info['action_valid'] and reward<0 and env.progress==before
                random=RandomAgent(73)
                while not env.done:
                    obs,_,_,_,_=env.step(random.act(obs,env.available_actions))
                    if env.steps>=20 and not env.done:env.step({'action':'FINISH'})
                report['checks'].append('invalid action handling and bounded observation-only random rollout')
                env.close();env=make_env(cfg,runtime_dir=out/'thor-runtime');env.privileged_references=True
                report['fresh_instance_replay']=replay(env,out/'episodes'/cfg.episode_id)
                report['checks'].append('fresh simulator instance regenerates the same seeded episode')
            env.close();env=None
        report['completed']=True
    except Exception as error:
        if env is not None and env.controller is not None:
            write(out/'failed_scene_objects.json',env.controller.last_event.metadata.get('objects',[]))
            write(out/'failed_generation.json',getattr(env,'generation_debug',[]))
        report['error']=f'{type(error).__name__}: {error}'
        (out/'failure.log').write_text(traceback.format_exc());raise
    finally:
        if env:env.close()
        write(out/'validation.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
