"""Re-render historical native trajectories without ID overlays; never re-infer a policy."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

from build import HERE, ROOT, digest, dump, read


def assert_scene(expected, actual):
    if expected['id_map'] != actual['id_map']:
        raise AssertionError('Object identity binding changed')
    before, after = expected['current_state'], actual['current_state']
    if set(before) != set(after): raise AssertionError('Scene object set changed')
    for oid, obj in before.items():
        for field in ('position','rotation'):
            if not np.allclose([obj[field][a] for a in 'xyz'],[after[oid][field][a] for a in 'xyz'],atol=.0001,rtol=0):
                raise AssertionError('Scene geometry changed')
        for field in ('isOpen','isPickedUp','isDirty','isToggled','isBroken','parentReceptacles'):
            if obj.get(field) != after[oid].get(field): raise AssertionError('Object state changed')
    key = 'agent' if 'agent' in expected else 'current_pose'
    for field in ('position','rotation'):
        if not np.allclose([expected[key][field][a] for a in 'xyz'],[actual[key][field][a] for a in 'xyz'],atol=.0001,rtol=0):
            raise AssertionError('Camera pose changed')
    if abs(expected[key]['cameraHorizon']-actual[key]['cameraHorizon']) > .0001:
        raise AssertionError('Camera horizon changed')
    # Older native logs omit task_state; its empty default has identical semantics.
    if expected.get('task_state', {}) != actual.get('task_state', {}):
        raise AssertionError('Task state changed')
    for field in ('progress','events'):
        if expected.get(field) != actual.get(field): raise AssertionError('Task history changed: '+field)


def render_episode(episode, output, render_size=None):
    sys.path.insert(0,str(ROOT/'04_AgentTask/TaskFrameworks/spatial_interaction'))
    from spatial_interaction import EpisodeConfig, make_env
    source = ROOT/episode['provenance']['directory']
    if digest(source/'trajectory.jsonl') != episode['provenance']['trajectory_sha256']:
        raise ValueError('Source trajectory hash changed')
    rows = [json.loads(line) for line in (source/'trajectory.jsonl').read_text().splitlines()]
    states = [json.loads(line)['state'] for line in (source/'privileged.jsonl').read_text().splitlines()]
    if len(rows) != len(states): raise ValueError('Incomplete private state evidence')
    directory = output/episode['id'];directory.mkdir()
    env = make_env(EpisodeConfig(**read(source/'episode.json')),runtime_dir=directory/'runtime')
    # Recorded privileged (oracle) actions use private object IDs; replay must accept them,
    # like trajectory.replay() does for privileged traces.
    env.privileged_references = True
    # This is a rendering-only replay of recorded privileged actions. It neither
    # invents new public goals nor re-evaluates the old policy on unnumbered input.
    env._observation = lambda: {'rgb':env.render(), 'phase':env.phase}
    frames = []
    try:
        obs, _ = env.reset()
        for i,row in enumerate(rows):
            if i:
                obs,reward,terminated,truncated,_ = env.step(row['action'])
                if not np.isclose(reward,row['reward']) or terminated != row['terminated'] or truncated != row['truncated']:
                    raise AssertionError('Recorded transition changed')
            actual = env.privileged_state()
            try:
                assert_scene(states[i],actual)
            except AssertionError as error:
                dump(directory/'replay-mismatch.json',{'step':i,'error':str(error),
                     'expected':states[i],'actual':actual})
                raise
            if obs['phase'] != row['observation']['phase']: raise AssertionError('Task phase changed')
            rgb=obs['rgb']
            if render_size:
                # Render more pixels at the verified pose, then restore the policy's
                # original resolution before taking the next recorded action.
                height,width=rgb.shape[:2]
                event=env.controller.step(action='ChangeResolution',x=render_size,y=render_size)
                if not event.metadata['lastActionSuccess']:raise RuntimeError('High-resolution capture failed')
                assert_scene(states[i],env.privileged_state())
                rgb=event.frame.copy()
                event=env.controller.step(action='ChangeResolution',x=width,y=height)
                if not event.metadata['lastActionSuccess']:raise RuntimeError('Policy resolution restore failed')
                assert_scene(states[i],env.privileged_state())
            path = directory/f'frame_{i:04d}.png'
            Image.fromarray(rgb).save(path)
            frames.append({'path':str(path.relative_to(ROOT)), 'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()})
        video = output/(episode['id']+'.mp4')
        ffmpeg = ROOT.parent/'conda/envs/spatialagent/bin/ffmpeg'
        subprocess.run([str(ffmpeg),'-hide_banner','-loglevel','error','-y','-framerate','2',
                        '-i',str(directory/'frame_%04d.png'),'-c:v','libx264','-threads','2',
                        '-preset','veryfast','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
        return {'source_trajectory_sha256':digest(source/'trajectory.jsonl'),
                'source_privileged_sha256':digest(source/'privileged.jsonl'),
                'native_state_verified':True,'policy_reinferred':False,'frames':frames,
                'video':str(video.relative_to(ROOT)),'video_sha256':digest(video)}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--render-size',type=int,choices=(320,512,640))
    parser.add_argument('--presentation-only',action='store_true')
    parser.add_argument('--showcase',action='store_true',help='Only the task-card cases listed under showcase in demo.json')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Native rendering requires a Slurm allocation and the registered AI2-THOR Python runtime')
    output = args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    catalog = read(HERE/'site/data/catalog.json')
    episodes = [e for e in catalog['episodes'] if e['task'] in {t['id'] for t in catalog['tasks'] if t['environment']=='ai2thor'}]
    if args.presentation_only:
        selected={e['episode'] for e in read(HERE/'presentation/examples.json')['examples']}
        episodes=[e for e in episodes if e['id'] in selected]
    if args.showcase:
        directories=set(read(HERE/'demo.json')['showcase'].values())
        episodes=[e for e in episodes if e['provenance']['directory'] in directories]
    sources = [Path(__file__), ROOT/'scripts/experiments/rerender_demo.slurm']
    sources += list((ROOT/'04_AgentTask/TaskFrameworks/spatial_interaction/spatial_interaction').rglob('*.py'))
    sources += [ROOT/'04_AgentTask/CustomAgentTasks'/e['task']/'env.py' for e in episodes if (ROOT/'04_AgentTask/CustomAgentTasks'/e['task']/'env.py').exists()]
    hashes = {}
    for path in set(sources):
        key = str(path.relative_to(ROOT));hashes[key]=digest(path)
        target = output/'source_snapshot'/key;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    metadata = {'run_id':output.name,'project_id':'unnumbered-replay-'+os.environ['SLURM_JOB_ID'],
        'project_name':'SpatialAgent unnumbered media','task_name':'Native display re-rendering','task_id':'native-clean-render',
        'purpose':'State-verified rendering without object ID overlays; not a new model rollout',
        'experiment_type':'Native rendering and physical replay validation',
        'dataset':{'tasks':[e['task'] for e in episodes],'materials':[e['provenance']['directory'] for e in episodes]},
        'model':{'name':'Original recorded actions; no inference'},
        'environment':{'name':'Registered AI2-THOR 5.0.0 CloudRendering'},
        'source':{'entrypoint':str(Path(__file__).relative_to(ROOT)),'sha256':hashes},
        'parameters':{'episodes':[e['id'] for e in episodes],'render_size':args.render_size},
        'slurm':{'job_id':os.environ['SLURM_JOB_ID'],'state':'RUNNING','started_at':datetime.now(timezone.utc).isoformat()},
        'artifacts':{'result_dir':str(output.relative_to(ROOT)),'manifest':'clean-frames.json'}}
    dump(output/'run_metadata.json',metadata)
    (output/'unity-config').mkdir()
    target = HERE/'presentation/clean-frames.json'
    # A subset render merges into the existing manifest; a full render replaces it.
    manifest = read(target) if (args.presentation_only or args.showcase) and target.exists() else {}
    try:
        for episode in episodes:
            print('Rendering '+episode['id'],flush=True)
            manifest[episode['id']] = render_episode(episode,output,args.render_size)
            dump(output/'clean-frames.json',manifest)
        if target.exists(): shutil.copy2(target,output/'previous-clean-frames.json')
        pending = target.with_suffix('.pending');dump(pending,manifest);pending.replace(target)
        python = ROOT.parent/'conda/envs/spatialagent/bin/python'
        subprocess.run([str(python),str(HERE/'build.py'),'--record',str(output/'site-build')],check=True)
        metadata['slurm'].update(state='COMPLETED',exit_code=0)
    except BaseException as error:
        metadata['slurm'].update(state='FAILED',exit_code=1,failure_reason=str(error))
        raise
    finally:
        metadata['slurm']['ended_at'] = datetime.now(timezone.utc).isoformat()
        dump(output/'run_metadata.json',metadata)


if __name__ == '__main__':
    main()
