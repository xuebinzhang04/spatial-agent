"""Record the two Shape3D showcase cases for the demo with the current generator.

shape_identification: an admitted initial-pixel-ambiguity case (a wrong candidate renders
identically from the start view); the policy follows the generator's verified revealing
camera path, then submits. pose_alignment: an L2 case (exact depth 2) from the canonical
start; the policy moves the camera once (the object does not turn), then rotates along a
shortest path. Both policies read privileged state: they illustrate case design, not ability.
CPU only (NumPy renderer); runs in about a minute on one core.

--native records the four AI2-THOR showcase cases (audit-clean cases of job 448983) with the
current generator at 512 px using the privileged task oracle. It needs the AI2-THOR runtime
inside an allocation (record_showcase_native.slurm).
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '04_AgentTask/TaskFrameworks/spatial_interaction'))
from spatial_interaction import EpisodeConfig, make_env  # noqa: E402
from spatial_interaction.core.trajectory import TrajectoryRecorder  # noqa: E402
from spatial_interaction.environments.shape3d.geometry import shortest_alignment  # noqa: E402


# (task, level, seed): generation-audited cases without warnings in task-generation-audit-native-448983.
NATIVE = (('memory_maze', 3, 3001), ('dependency_doors', 2, 3000),
          ('relational_rearrangement', 2, 3000), ('dynamic_replanning', 2, 3000))


def native_case(task, level, seed, directory):
    from spatial_interaction.runner import run_episode
    config = EpisodeConfig(episode_id=f'showcase_{task}_L{level}_seed{seed}', environment='thor_tasks', task_type=task,
                           scene_id='FloorPlan1', seed=seed, difficulty=level, max_steps=240,
                           task_config={'split': 'development', 'image_size': 512})
    return run_episode(config, directory, 'oracle', debug=True)


def utc():
    return datetime.now(timezone.utc).isoformat()


def shape_case(seeds):
    """Admitted ambiguity case with the shortest verified revealing path among `seeds`."""
    best = None
    for seed in seeds:
        config = EpisodeConfig(episode_id=f'showcase_shape_identification_seed{seed}', seed=seed, difficulty=2,
                               max_steps=60, task_config={'split': 'development', 'image_size': 224,
                                                          'require_initial_pixel_ambiguity': True,
                                                          'initial_rotation': 0, 'voxels': 6})
        env = make_env(config)
        try:
            env.reset()
        except ValueError:
            continue
        length = len(env.ambiguity_witness['actions'])
        if best is None or length < best[1]:
            best = (config, length)
    if best is None:
        raise RuntimeError('No admitted ambiguity case among the given seeds')
    return best[0]


def record(config, directory):
    env = make_env(config)
    env.privileged_references = True
    observation, info = env.reset()
    if config.task_type == 'shape_identification':
        plan = list(env.ambiguity_witness['actions']) + [{'action': 'SUBMIT_SHAPE', 'candidate_id': env.correct}]
    else:
        plan = [{'action': 'CAMERA_RIGHT'}] + [{'action': a} for a in
                shortest_alignment(env.orientation, env.reference, env.cells, env.axes)] + [{'action': 'SUBMIT_POSE'}]
    recorder = TrajectoryRecorder(directory, config, debug=True)
    recorder.record(observation, info=info, state=env.privileged_state())
    total = 0.
    for action in plan:
        observation, reward, terminated, truncated, info = env.step(action)
        total += reward
        recorder.record(observation, action, reward, terminated, truncated, info, env.privileged_state())
        if env.done:
            break
    recorder.close()
    report = {'episode_id': config.episode_id, 'task_type': config.task_type, 'policy': 'oracle',
              'seed': config.seed, 'terminated': terminated, 'truncated': truncated, 'reward': total,
              'metrics': env.metrics, 'oracle_uses_privileged_state': True,
              'plan': 'verified revealing camera path, then privileged answer' if config.task_type == 'shape_identification'
                      else 'one camera move, then shortest object rotation path'}
    (directory / 'report.json').write_text(json.dumps(report, indent=2))
    (directory / 'ground_truth.json').write_text(json.dumps(env.privileged_state(), indent=2, default=str))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shape-seeds', type=int, nargs='*', default=list(range(2000, 2012)))
    parser.add_argument('--pose-seed', type=int, default=4000)
    parser.add_argument('--native', action='store_true', help='Record the AI2-THOR showcase cases instead')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh output directory; existing evidence is preserved')
    args.output.mkdir(parents=True)
    started = utc()
    pose = EpisodeConfig(episode_id=f'showcase_pose_alignment_seed{args.pose_seed}', task_type='pose_alignment',
                         seed=args.pose_seed, difficulty=2, max_steps=50,
                         task_config={'split': 'development', 'image_size': 224, 'initial_rotation': 0})
    reports = {}
    if args.native:
        for task, level, seed in NATIVE:
            reports[task] = native_case(task, level, seed, args.output / 'cases' / task)
            print(json.dumps({task: reports[task]['metrics'], 'seed': seed}), flush=True)
    else:
        for config in (shape_case(args.shape_seeds), pose):
            reports[config.task_type] = record(config, args.output / 'cases' / config.task_type)
            print(json.dumps({config.task_type: reports[config.task_type]['metrics'], 'seed': config.seed}))
    metadata = {
        'run_id': args.output.name, 'project_id': args.output.name, 'project_name': 'SpatialAgent task demo',
        'task_name': 'showcase-case-recording', 'task_id': 'showcase-cases',
        'purpose': 'Record Shape3D example cases whose design (initial ambiguity; exact rotation depth) the demo explains.',
        'dataset': {'name': 'generated Shape3D cases', 'version': 'current generator', 'split': 'development',
                    'tasks': list(reports), 'materials': []},
        'experiment_type': 'presentation recording with privileged scripted policies',
        'model': {'name': 'none', 'checkpoint': None, 'precision': None},
        'environment': {'name': 'AI2-THOR 5.0.0 CPU lavapipe' if args.native else 'Shape3D NumPy renderer',
                        'version': 'workspace'},
        'source': {'commit': 'workspace (not a git repository)', 'entrypoint': 'scripts/experiments/record_showcase_cases.py',
                   'config': {'native': list(NATIVE)} if args.native else
                             {'shape_seeds': args.shape_seeds, 'pose_seed': args.pose_seed}},
        'parameters': {}, 'slurm': {'job_id': os.environ.get('SLURM_JOB_ID') or 'Not applicable: light CPU recording',
                                    'state': 'COMPLETED', 'created_at': started, 'started_at': started,
                                    'ended_at': utc(), 'exit_code': 0, 'failure_reason': 'Not applicable'},
        'artifacts': {'result_dir': str(args.output), 'trajectory': str(args.output / 'cases'), 'logs': []},
        'notes': 'Privileged scripted policies; illustrates case design only.'}
    (args.output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
