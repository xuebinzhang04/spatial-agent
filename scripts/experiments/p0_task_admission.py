"""P0 admission and shortcut controls for the frozen interaction-case batch.

For every recipe this checks, under the public reference protocol:
  * fresh reset (generation) succeeds and public observations contain no private IDs;
  * a privileged oracle solves it when every object reference is converted to a
    view:STEP:X:Y pixel reference (the environment runs with private IDs rejected);
  * the public oracle trajectory replays exactly in a fresh environment (first case per task);
  * shortcut controls fail: immediate commit/finish, and for missing/moved objects a
    stationary-scan policy (rotation only, privileged answer) that tests whether
    translation is necessary;
  * reset and step wall-clock time.

Shape3D additionally gets a seed sweep of ambiguity admission and revealing sides.
Oracle and control policies read privileged state; none of this measures model ability.
Run with the AI2-THOR runtime interpreter inside a Slurm allocation (see p0_task_admission.slurm).
"""
import argparse
import json
import os
import re
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '04_AgentTask/TaskFrameworks/spatial_interaction'))
from spatial_interaction import EpisodeConfig, make_env  # noqa: E402
from spatial_interaction.agents.policies import OracleAgent  # noqa: E402
from spatial_interaction.core.trajectory import TrajectoryRecorder, replay  # noqa: E402

PRIVATE = re.compile(r'object_\d+')
NATIVE = ('dynamic_memory', 'thor_tasks')


def utc():
    return datetime.now(timezone.utc).isoformat()


def public_fields(observation):
    return {k: v for k, v in observation.items() if k != 'rgb'}


def pixel_reference(env, oid, steps=None):
    """Return view:STEP:X:Y on unambiguous visible pixels of a private object, or None.

    Searches the given observed steps (latest first); avoids the bottom feedback banner.
    """
    frames = env.visual_frames
    for step in sorted(frames if steps is None else steps, reverse=True):
        labels, ids = frames[step]
        if oid not in ids:
            continue
        ys, xs = np.nonzero(labels == ids.index(oid))
        keep = ys < labels.shape[0] - 26
        if keep.any():
            ys, xs = ys[keep], xs[keep]
        if len(ys) == 0:
            continue
        # The pixel closest to the mask centroid stays inside thin or concave masks.
        k = int(np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2))
        return f'view:{step}:{(xs[k] + .5) / labels.shape[1]:.4f}:{(ys[k] + .5) / labels.shape[0]:.4f}'
    return None


def to_public(env, action):
    """Convert an oracle action's private object reference into a pixel reference."""
    if 'object_id' not in action or action['object_id'].startswith('view:'):
        return action
    oid = env.reverse_ids[action['object_id']]
    reference = pixel_reference(env, oid, [env.steps])
    if reference is None:
        raise RuntimeError(f"Oracle target {action['object_id']} has no visible pixels at step {env.steps}")
    return {**action, 'object_id': reference}


def run_policy(env, choose, directory=None, label='public_oracle'):
    """Step env until done with choose(observation) actions; optionally record a trajectory."""
    observation = env._observation()
    recorder = TrajectoryRecorder(directory, env.config) if directory else None
    if recorder:
        recorder.record(observation)
    total, info, durations, invalid = 0., {}, [], []
    terminated = truncated = False
    try:
        while not env.done:
            action = choose(observation)
            started = time.perf_counter()
            observation, reward, terminated, truncated, info = env.step(action)
            durations.append(time.perf_counter() - started)
            total += reward
            if not info.get('action_valid', True):
                invalid.append({'step': env.steps, 'action': action, 'error': info.get('error')})
            if recorder:
                recorder.record(observation, action, reward, terminated, truncated)
    finally:
        if recorder:
            recorder.close()
    metrics = info.get('metrics') or env.metrics
    if directory:
        (Path(directory) / 'report.json').write_text(json.dumps(
            {'policy': label, 'reward': total, 'terminated': terminated, 'truncated': truncated, 'metrics': metrics}, indent=2))
    return {'success': bool(metrics.get('success')), 'reward': total, 'terminated': terminated, 'truncated': truncated,
            'steps': env.steps, 'metrics': metrics, 'invalid_actions': invalid,
            'mean_step_seconds': float(np.mean(durations)) if durations else None}


def dynamic_oracle(env):
    """Rotation-only survey, commit, then submit the privileged answer with pixel references.

    This doubles as the stationary-scan shortcut control: if it succeeds,
    the case does not require translation.
    """
    state = {'turns': 0, 'post': 0}

    def choose(observation):
        if env.phase == 'memorization':
            if state['turns'] < 12:  # a full 360-degree in-place scan at 30 degrees per turn
                state['turns'] += 1
                return {'action': 'RotateLeft'}
            return {'action': 'MEMORIZE', 'memory': 'privileged control scan'}
        change = deepcopy(env.truth['changes'][0])
        before = sorted(env.memorization_view_steps)
        answer = {'type': change['type'],
                  'object_id': pixel_reference(env, env.reverse_ids[change['object_id']], before),
                  'previous_location': pixel_reference(env, env.reverse_ids.get(change['previous_location'], ''), before)}
        if change['type'] == 'moved':
            after = [s for s in env.visual_frames if s not in env.memorization_view_steps]
            current = pixel_reference(env, env.reverse_ids.get(change['current_location'], ''), after)
            if current is None and state['post'] < 12:
                state['post'] += 1
                return {'action': 'RotateLeft'}
            answer['current_location'] = current
        if None in answer.values():
            # Unreachable by rotation alone: submit an explicit miss so the control records failure.
            answer = {k: (v if v is not None else 'view:0:0.5:0.5') for k, v in answer.items()}
        return {'action': 'SUBMIT', 'answer': {'changes': [answer]}}
    return choose


def initial_twins(env):
    """Wrong candidates whose initial target rendering is pixel-identical to the answer."""
    from spatial_interaction.environments.shape3d.renderer import render
    def image(cells):
        return render(cells, env.orientation, env.camera, env.yaw, env.pitch, env.size * 2, env.occlusion)
    base = image(env.cells)
    return [f'candidate_{i+1}' for i, cells in enumerate(env.candidates)
            if f'candidate_{i+1}' != env.correct and np.array_equal(base, image(cells))]


def immediate_control(env):
    if env.config.environment == 'dynamic_memory':
        actions = iter([{'action': 'MEMORIZE'}, {'action': 'SUBMIT', 'answer': {'changes': [
            {'type': 'missing' if env.config.task_type == 'missing_object' else 'moved',
             'object_id': 'view:0:0.5:0.5', 'previous_location': 'view:0:0.5:0.7',
             **({'current_location': 'view:0:0.4:0.7'} if env.config.task_type == 'moved_object' else {})}]}}])
    elif env.config.task_type == 'shape_identification':
        # Submit a pixel-identical wrong twin: the guess the initial view cannot rule out.
        wrong = next(f'candidate_{i+1}' for i in range(len(env.candidates)) if f'candidate_{i+1}' != env.correct)
        actions = iter([{'action': 'SUBMIT_SHAPE', 'candidate_id': (initial_twins(env) or [wrong])[0]}])
    elif env.config.task_type == 'pose_alignment':
        actions = iter([{'action': 'SUBMIT_POSE'}])
    elif env.config.task_type == 'memory_maze':
        actions = iter([{'action': 'MEMORIZE'}, {'action': 'FINISH'}])
    else:
        actions = iter([{'action': 'FINISH'}])
    return lambda observation: next(actions, {'action': 'FINISH'})


def check_case(path, output, do_replay):
    recipe = json.loads(path.read_text())
    config = EpisodeConfig(**recipe)
    case = {'case': path.stem, 'task': config.task_type, 'environment': config.environment, 'scene': config.scene_id}
    runtime = {'runtime_dir': str(output / 'thor-runtime' / path.stem)} if config.environment in NATIVE else {}
    env = make_env(config, **runtime)
    try:
        started = time.perf_counter()
        try:
            observation, _ = env.reset()
        except Exception as error:  # generation rejection is a result, not a crash
            case.update(admitted=False, reset_error=f'{type(error).__name__}: {error}',
                        reset_seconds=time.perf_counter() - started)
            return case
        case['reset_seconds'] = time.perf_counter() - started
        case['admitted'] = True
        leaked = PRIVATE.findall(json.dumps(public_fields(observation), default=str))
        case['public_observation_private_ids'] = sorted(set(leaked))
        case['instruction'] = observation['instruction']

        # Public-reference oracle: privileged planning, public execution.
        if config.environment == 'shape3d':
            agent = OracleAgent(env)
            env.privileged_references = False
            choose = lambda observation: agent.act(observation, env.available_actions)  # noqa: E731
        elif config.environment == 'dynamic_memory':
            choose = dynamic_oracle(env)
        else:
            choose = lambda observation: to_public(env, env.oracle_action())  # noqa: E731
        directory = output / 'episodes' / path.stem
        try:
            case['public_oracle'] = run_policy(env, choose, directory if do_replay else None)
        except Exception as error:
            case['public_oracle'] = {'success': False, 'error': f'{type(error).__name__}: {error}'}
        if config.environment == 'dynamic_memory':
            # The dynamic oracle is the stationary scan: success means translation was unnecessary.
            case['stationary_scan_solves'] = case['public_oracle'].get('success')

        if do_replay and case['public_oracle'].get('steps'):
            fresh = make_env(config, **runtime)
            try:
                case['public_replay'] = replay(fresh, directory)
            except Exception as error:
                case['public_replay'] = {'verified': False, 'error': f'{type(error).__name__}: {error}'}
            finally:
                fresh.close()

        env.reset()
        env.privileged_references = False
        if config.task_type == 'shape_identification':
            # Best no-interaction policy: eliminate distinguishable candidates, guess among twins.
            case['initial_view_guess_success'] = 1 / (1 + len(initial_twins(env)))
        case['immediate_control'] = run_policy(env, immediate_control(env))
        case['shortcut_controls_fail'] = not case['immediate_control']['success']
        return case
    finally:
        env.close()


def reveal_sides(env):
    """Sides (-1 left, +1 right) whose camera translation reveals the hidden difference."""
    from spatial_interaction.environments.shape3d.renderer import render, camera_basis
    twins = [c for i, c in enumerate(env.candidates) if f'candidate_{i+1}' != env.correct]
    base = render(env.cells, env.orientation, env.camera, env.yaw, env.pitch, env.size * 2, env.occlusion)
    twins = [c for c in twins if np.array_equal(base, render(c, env.orientation, env.camera, env.yaw, env.pitch,
                                                              env.size * 2, env.occlusion))]
    right, _, _ = camera_basis(env.yaw, 0.)
    found = set()
    for sign in (-1, 1):
        for count in range(1, 6):
            camera = env.camera + sign * .6 * count * right
            yaw = env.yaw + 15 * int(round(((np.degrees(np.arctan2(-camera[0], -camera[2])) - env.yaw + 180) % 360 - 180) / 15))
            a = render(env.cells, env.orientation, camera, yaw, env.pitch, env.size * 2, env.occlusion)
            if any(np.count_nonzero(np.max(np.abs(a.astype(int) - render(t, env.orientation, camera, yaw, env.pitch,
                   env.size * 2, env.occlusion).astype(int)), axis=2) > 10) for t in twins):
                found.add(sign)
                break
    return sorted(found)


def shape_sweep(seeds):
    rows = []
    for seed in seeds:
        config = EpisodeConfig(seed=seed, max_steps=60, difficulty=2, task_config={
            'split': 'development', 'image_size': 96, 'require_initial_pixel_ambiguity': True,
            'initial_rotation': 0, 'voxels': 6})
        env = make_env(config)
        try:
            env.reset()
        except ValueError as error:
            rows.append({'seed': seed, 'admitted': False, 'reason': str(error)})
            continue
        sides = reveal_sides(env)
        rows.append({'seed': seed, 'admitted': True, 'candidates': len(env.candidates), 'reveal_sides': sides,
                     'witness_first_action': env.ambiguity_witness['actions'][0]['action']})
    admitted = [r for r in rows if r['admitted']]
    count = lambda pred: sum(pred(r) for r in admitted)  # noqa: E731
    return {'seeds': len(rows), 'admitted': len(admitted),
            'reveal_left_only': count(lambda r: r['reveal_sides'] == [-1]),
            'reveal_right_only': count(lambda r: r['reveal_sides'] == [1]),
            'reveal_both': count(lambda r: r['reveal_sides'] == [-1, 1]),
            'witness_left_first': count(lambda r: r['witness_first_action'] == 'CAMERA_LEFT'),
            'random_guess_success_expectation': float(np.mean([1 / r['candidates'] for r in admitted])) if admitted else None,
            'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--recipes', type=Path,
                        default=ROOT / '05_Experiment/results/interaction-cases-active-20261025/recipes')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', nargs='*', help='Restrict to these task types')
    parser.add_argument('--shape-seeds', type=int, default=60)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose a fresh output directory; existing evidence is preserved')
    args.output.mkdir(parents=True, exist_ok=True)
    job = os.environ.get('SLURM_JOB_ID')
    metadata = {
        'run_id': args.output.name, 'project_id': args.output.name, 'project_name': 'SpatialAgent RL environment P0',
        'task_name': 'p0-task-admission', 'task_id': 'p0-admission',
        'purpose': 'Verify that frozen interaction cases admit, are solvable through public view:STEP:X:Y references, '
                   'replay, and that immediate-commit and stationary-scan shortcuts fail. No model is evaluated.',
        'dataset': {'name': 'interaction-cases-active-20261025', 'version': 'frozen recipes', 'split': 'development',
                    'tasks': sorted(p.stem for p in args.recipes.glob('*.json')), 'materials': [str(args.recipes)]},
        'experiment_type': 'environment admission and shortcut controls',
        'model': {'name': 'none (privileged oracle and scripted controls)', 'checkpoint': None, 'precision': None},
        'environment': {'name': 'AI2-THOR 5.0.0 CloudRendering via CPU lavapipe; Shape3D NumPy renderer',
                        'version': 'cached f0825767cd50d69f666c7f282e54abfe58f1e917'},
        'source': {'commit': 'workspace (not a git repository); see archive record 2026-11-08',
                   'entrypoint': 'scripts/experiments/p0_task_admission.py', 'config': str(args.recipes)},
        'parameters': {'tasks': args.tasks, 'shape_seeds': args.shape_seeds, 'replay': 'first case per task'},
        'slurm': {'job_id': job, 'partition': os.environ.get('SLURM_JOB_PARTITION'),
                  'resources': f"cpus={os.environ.get('SLURM_CPUS_PER_TASK')}", 'state': 'RUNNING',
                  'created_at': utc(), 'started_at': utc(), 'ended_at': None, 'exit_code': None,
                  'failure_reason': 'Not recorded'},
        'artifacts': {'result_dir': str(args.output), 'primary_metrics': str(args.output / 'summary.json'),
                      'report': str(args.output / 'cases.jsonl'), 'trajectory': str(args.output / 'episodes'), 'logs': []},
        'notes': 'Oracle/control policies use privileged state; results certify environment mechanics only.'}
    (args.output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))

    cases, replayed = [], set()
    exit_code, failure = 0, 'Not recorded'
    try:
        for path in sorted(args.recipes.glob('*.json')):
            task = json.loads(path.read_text())['task_type']
            if args.tasks and task not in args.tasks:
                continue
            try:
                case = check_case(path, args.output, task not in replayed)
            except Exception as error:
                case = {'case': path.stem, 'task': task, 'crash': f'{type(error).__name__}: {error}',
                        'traceback': traceback.format_exc()}
            replayed.add(task)
            cases.append(case)
            with (args.output / 'cases.jsonl').open('a') as handle:
                handle.write(json.dumps(case, default=str) + '\n')
            print(json.dumps({k: case.get(k) for k in ('case', 'admitted', 'reset_error', 'crash')}
                             | {'oracle': case.get('public_oracle', {}).get('success'),
                                'stationary': case.get('stationary_scan_solves'),
                                'controls_fail': case.get('shortcut_controls_fail')}), flush=True)
        shape = shape_sweep(range(2000, 2000 + args.shape_seeds)) if args.shape_seeds else None
        if shape:
            (args.output / 'shape_sweep.json').write_text(json.dumps(shape, indent=2))
        summary = {'cases': len(cases), 'admitted': sum(bool(c.get('admitted')) for c in cases),
                   'crashed': [c['case'] for c in cases if 'crash' in c],
                   'rejected': {c['case']: c['reset_error'] for c in cases if c.get('admitted') is False},
                   'public_oracle_success': [c['case'] for c in cases if c.get('public_oracle', {}).get('success')],
                   'public_oracle_failure': {c['case']: c.get('public_oracle') for c in cases
                                             if c.get('admitted') and not c.get('public_oracle', {}).get('success')},
                   'private_id_leaks': {c['case']: c['public_observation_private_ids'] for c in cases
                                        if c.get('public_observation_private_ids')},
                   'replays': {c['case']: c['public_replay'] for c in cases if 'public_replay' in c},
                   'shortcut_controls_succeeded': [c['case'] for c in cases if c.get('shortcut_controls_fail') is False],
                   'stationary_scan_solves': [c['case'] for c in cases if c.get('stationary_scan_solves')],
                   'reset_seconds': {c['case']: round(c['reset_seconds'], 2) for c in cases if 'reset_seconds' in c},
                   'mean_step_seconds': {c['case']: c['public_oracle'].get('mean_step_seconds') for c in cases
                                         if c.get('public_oracle', {}).get('mean_step_seconds')},
                   'shape_sweep': {k: v for k, v in (shape or {}).items() if k != 'rows'}}
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
        print(json.dumps({k: summary[k] for k in ('cases', 'admitted', 'crashed', 'shortcut_controls_succeeded',
                                                    'stationary_scan_solves')}, indent=2))
    except BaseException as error:
        exit_code, failure = 1, f'{type(error).__name__}: {error}'
        raise
    finally:
        metadata['slurm'].update(state='COMPLETED' if exit_code == 0 else 'FAILED', ended_at=utc(),
                                 exit_code=exit_code, failure_reason=failure)
        (args.output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
