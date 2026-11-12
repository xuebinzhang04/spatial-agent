"""Audit task generation: controls -> generated environment variables -> expectations.

Every generated case is summarized with the variable model of
04_AgentTask/CustomAgentTasks/task_variables.json:
  controls       the EpisodeConfig that produced it,
  local_state    public observation fields (what a policy sees at one step),
  global_memory  latent world variables the policy must build, update or keep frozen.
Each case is checked against its task's control effects and expectations
(fail = generator contract broken; warn = admissible but weak for memory training).
Live generation also checks determinism (same recipe -> same variables and first
frame). Cross-case checks report variable diversity, answer balance and start poses.

Sources:
  --source live      reset environments now. Shape3D runs on CPU in the main env;
                     native tasks need the AI2-THOR runtime inside a Slurm allocation
                     (task_generation_audit.slurm).
  --source recorded  read ground_truth.json of the demo's recorded native episodes;
                     no simulator. Older recordings may predate current protocols.
Oracle-free: no policy acts. Results certify generation, not task difficulty.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TASKS_DIR = ROOT / '04_AgentTask/CustomAgentTasks'
sys.path.insert(0, str(ROOT / '04_AgentTask/TaskFrameworks/spatial_interaction'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spatial_interaction import EpisodeConfig, make_env  # noqa: E402
from spatial_interaction.environments.dynamic_memory.env import DynamicMemoryEnv  # noqa: E402

VARIABLES = json.loads((TASKS_DIR / 'task_variables.json').read_text())
PRIVATE = re.compile(r'object_\d{3}')
SHAPE = ('shape_identification', 'pose_alignment')
DYNAMIC = ('missing_object', 'moved_object')
THOR = ('memory_maze', 'dependency_doors', 'relational_rearrangement', 'dynamic_replanning')


def utc():
    return datetime.now(timezone.utc).isoformat()


def planar(a, b):
    return math.hypot(a['x'] - b['x'], a['z'] - b['z'])


def describe(oid, state):
    # public_description only reads the supplied scene, so no live environment is needed.
    return DynamicMemoryEnv.public_description(None, oid, state)


# ---------------------------------------------------------------- variable extraction

def shape_variables(env):
    from spatial_interaction.environments.shape3d.geometry import canonical, rotation_error, shortest_alignment
    identity = bool(np.array_equal(env.orientation, np.eye(3)))
    if env.config.task_type == 'pose_alignment':
        return {'voxels': len(env.cells), 'axes': env.axes,
                'alignment_length': len(shortest_alignment(env.orientation, env.reference, env.cells, env.axes)),
                '_generator_depth': getattr(env, 'alignment_length', None),
                'initially_aligned': rotation_error(env.orientation, env.reference, env.cells) < 1e-6,
                'initial_rotation_identity': identity, 'occlusion': env.occlusion}
    from p0_task_admission import initial_twins
    return {'voxels': len(env.cells), 'candidates': len(env.candidates),
            'correct_index': int(env.correct.split('_')[1]),
            'distinct_candidates': len({canonical(c) for c in env.candidates}) == len(env.candidates),
            'ambiguous_distractor': bool(env.ambiguous), 'initial_pixel_twins': len(initial_twins(env)),
            'witness_actions': len(env.ambiguity_witness['actions']) if env.ambiguity_witness else 0,
            'initial_rotation_identity': identity, 'occlusion': env.occlusion}


def dynamic_variables(task, p):
    state, reverse = p['initial_state'], {v: k for k, v in p['id_map'].items()}
    if p['ground_truth']['changes']:
        change = p['ground_truth']['changes'][0]
    else:
        # Before MEMORIZE the change is only planned: reconstruct it as _intervene() will record it.
        target_oid = reverse[p['target_id']]
        parents = sorted(state[target_oid].get('parentReceptacles') or [])
        change = {'object_id': p['target_id'], 'previous_location': p['id_map'][parents[0]]}
        if task == 'moved_object':
            change['current_location'] = p['id_map'][p['plan']['receptacle_id']]
    target, support = state[reverse[change['object_id']]], state[reverse[change['previous_location']]]
    exposed = [o for o in state.values() if o.get('pickupable') and o.get('parentReceptacles')
               and not o.get('receptacleObjectIds')]
    result = {'target_type': target['objectType'], 'support_type': support['objectType'],
              'target_visible_at_start': bool(target.get('visible')),
              'target_distance_m': round(planar(p['initial_pose']['position'], target['position']), 2),
              'exposed_objects': len(exposed), 'support_count': len(target.get('parentReceptacles') or [])}
    if task == 'moved_object':
        result['new_support_type'] = state[reverse[change['current_location']]]['objectType']
        result['relocation_distance_m'] = round(planar(target['position'], p['plan']['position']), 2)
        result['_receptacles_distinct'] = change['current_location'] != change['previous_location']
    return result


def thor_variables(task, p, start):
    state, reverse, truth = p['initial_state'], {v: k for k, v in p['id_map'].items()}, p['ground_truth']
    kind = lambda oid: state[oid]['objectType']  # noqa: E731
    if task == 'memory_maze':
        order = [reverse[s] for s in truth['ordered_landmarks']]
        distinct = list(dict.fromkeys(order))
        spacing = [planar(state[a]['position'], state[b]['position'])
                   for i, a in enumerate(distinct) for b in distinct[i + 1:]]
        return {'landmark_types': [kind(o) for o in order], 'distinct_landmarks': len(distinct),
                'returns_to_first': len(order) > 1 and order[-1] == order[0],
                'publicly_describable': all(describe(o, state) is not None for o in distinct),
                'visible_at_start': sum(bool(state[o].get('visible')) for o in distinct),
                'min_landmark_spacing_m': round(min(spacing), 2) if spacing else None,
                'start_position': [round(start['position']['x'], 2), round(start['position']['z'], 2)] if start else None}
    if task == 'dependency_doors':
        orders = truth['orders']
        return {'orders': len(orders), 'object_types': [kind(o['object']) for o in orders],
                'source_types': [kind(o['source']) for o in orders],
                'destination_types': [kind(o['destination']) for o in orders],
                'sources_closed_at_start': all(not state[o['source']].get('isOpen') for o in orders),
                'object_inside_source': all(o['source'] in (state[o['object']].get('parentReceptacles') or [])
                                            for o in orders),
                'closed_pickup_rejected': bool(truth.get('native_closed_pickup_rejected'))}
    if task == 'relational_rearrangement':
        from spatial_interaction.environments.thor_tasks.relations import relation_holds
        anchor = state[reverse[truth['anchor']]]['position']
        axes = [tuple(a) for a in truth['world_axes']]
        holds = {s: relation_holds(pos, anchor, axes, truth['near_metres']) for s, pos in truth['surface_slots'].items()}
        return {'target_type': kind(reverse[truth['target']]), 'anchor_type': kind(reverse[truth['anchor']]),
                'relation_axes': truth['world_axes'], 'near_metres': truth['near_metres'],
                'slots': len(holds), 'goal_witness_holds': holds[truth['goal_slot_witness']],
                'wrong_witness_fails': not holds[truth['incorrect_slot_witness']],
                'satisfying_slots': sum(holds.values())}
    if task == 'dynamic_replanning':
        return {'target_type': kind(reverse[truth['target']]), 'obstacle_type': kind(truth['intervention']['object']),
                'baseline_route_steps': truth['baseline_route_steps'], 'detour_route_steps': truth['detour_route_steps'],
                'detour_extra_steps': truth['detour_route_steps'] - truth['baseline_route_steps'],
                'intervention_planned': bool(truth.get('intervention'))}
    raise ValueError(f'No variable extractor for {task}')


# ---------------------------------------------------------------- expectations

def expectations(task, config, v):
    """Return [(check, status, detail)] with status pass/fail/warn."""
    level, cfg, out = config['difficulty'], config['task_config'], []
    # Underscored entries are consistency facts used only by these checks, not task variables.
    private = {k: v.pop(k) for k in list(v) if k.startswith('_')}

    def check(name, ok, detail='', weak=False):
        out.append({'check': name, 'status': 'pass' if ok else ('warn' if weak else 'fail'), 'detail': detail})

    if task == 'shape_identification':
        ambiguity = cfg.get('require_initial_pixel_ambiguity', False)
        want = cfg.get('candidates', min(5, 2 + level))
        check('candidate count follows control', v['candidates'] == want, f"{v['candidates']} vs {want}")
        want = cfg.get('voxels', 6 if ambiguity else 5 + level)
        check('voxel count follows control', v['voxels'] == want, f"{v['voxels']} vs {want}")
        want = cfg.get('occlusion', .35 if level >= 4 else 0.)
        check('occlusion follows control', v['occlusion'] == want, f"{v['occlusion']} vs {want}")
        if 'initial_rotation' not in cfg and level == 1:
            check('L1 starts in canonical orientation', v['initial_rotation_identity'])
        check('candidates distinct up to rotation', v['distinct_candidates'])
        if ambiguity:
            check('initial view has a pixel-identical twin', v['initial_pixel_twins'] >= 1, str(v['initial_pixel_twins']))
            check('revealing camera path fits budget', 0 < v['witness_actions'] <= config['max_steps'] - 1,
                  f"{v['witness_actions']} actions")
        else:
            check('initial view separates candidates (no ambiguity required)', v['initial_pixel_twins'] == 0,
                  f"{v['initial_pixel_twins']} twins; interaction is optional for this case", weak=True)
    elif task == 'pose_alignment':
        check('start pose differs from reference', not v['initially_aligned'])
        want = min(cfg.get('alignment_length', level), 3)
        if v['axes'] == 'XYZ':  # every depth up to the group diameter 3 is reachable
            check('alignment depth = min(L, 3) quarter turns', v['alignment_length'] == want,
                  f"{v['alignment_length']} vs {want}")
        else:
            check('alignment depth within 1..min(L, 3)', 1 <= v['alignment_length'] <= want, str(v['alignment_length']))
        if private.get('_generator_depth') is not None:  # absent for recordings
            check('generator depth equals independent shortest path', private['_generator_depth'] == v['alignment_length'],
                  f"{private['_generator_depth']} vs {v['alignment_length']}")
        check('alignment fits action budget', v['alignment_length'] + 1 <= config['max_steps'])
        if 'axes' in cfg:
            check('axes follow control', v['axes'] == cfg['axes'], v['axes'])
    elif task in DYNAMIC:
        active = cfg.get('memory_protocol') == 'active_scan_v1'
        if active or level == 2:
            check('target hidden from start view', not v['target_visible_at_start'])
        if active:
            check('target has one supporting receptacle', v['support_count'] == 1, str(v['support_count']))
        check('scene offers a nontrivial memory load (>= 5 exposed objects)', v['exposed_objects'] >= 5,
              str(v['exposed_objects']), weak=True)
        if task == 'moved_object':
            check('original and new receptacles differ', private['_receptacles_distinct'])
            check('relocation is at least 0.5 m', v['relocation_distance_m'] >= .5,
                  f"{v['relocation_distance_m']} m", weak=True)
    elif task == 'memory_maze':
        want = min(4, 1 + level)
        check('distinct landmarks = min(4, 1+L)', v['distinct_landmarks'] == want, f"{v['distinct_landmarks']} vs {want}")
        check('return-to-first follows L>=2', v['returns_to_first'] == (level >= 2))
        check('landmarks publicly describable', v['publicly_describable'], ', '.join(v['landmark_types']))
        check('not every landmark visible at start', v['visible_at_start'] < v['distinct_landmarks'],
              f"{v['visible_at_start']}/{v['distinct_landmarks']} visible", weak=True)
        if v['min_landmark_spacing_m'] is not None:
            check('landmarks at least 1 m apart', v['min_landmark_spacing_m'] >= 1.,
                  f"{v['min_landmark_spacing_m']} m", weak=True)
    elif task == 'dependency_doors':
        check('order count = L', v['orders'] == level, f"{v['orders']} vs {level}")
        check('sources start closed', v['sources_closed_at_start'])
        check('objects start inside sources', v['object_inside_source'])
        check('closed-container pickup natively rejected', v['closed_pickup_rejected'])
    elif task == 'relational_rearrangement':
        want = 1 if level < 3 else 2
        check('relation axes follow L', len(v['relation_axes']) == want, f"{len(v['relation_axes'])} vs {want}")
        check('near constraint follows L>=2', (v['near_metres'] is None) == (level == 1), str(v['near_metres']))
        check('goal witness satisfies relation (recomputed)', v['goal_witness_holds'])
        check('incorrect witness violates relation (recomputed)', v['wrong_witness_fails'])
        check('relation is selective (some slots fail)', v['satisfying_slots'] < v['slots'],
              f"{v['satisfying_slots']}/{v['slots']} slots satisfy")
        check('at most half of the slots satisfy the relation', v['satisfying_slots'] * 2 <= v['slots'],
              f"{v['satisfying_slots']}/{v['slots']} slots satisfy; random placement often succeeds", weak=True)
    elif task == 'dynamic_replanning':
        check('baseline route >= 8+2L', v['baseline_route_steps'] >= 8 + 2 * level,
              f"{v['baseline_route_steps']} vs {8 + 2 * level}")
        want = cfg.get('min_detour_steps', 2)
        check('detour >= baseline + min_detour_steps', v['detour_extra_steps'] >= want,
              f"+{v['detour_extra_steps']} vs +{want}")
        check('blocking intervention planned and verified', v['intervention_planned'])
        check('detour adds at least 4 steps (replanning pressure)', v['detour_extra_steps'] >= 4,
              f"+{v['detour_extra_steps']}", weak=True)
    declared = set(VARIABLES['tasks'][task]['global_memory'])
    check('extracted variables match task_variables.json', set(v) == declared,
          f"missing {sorted(declared - set(v))}, extra {sorted(set(v) - declared)}")
    return out


def local_state_checks(task, observation, protocol_active):
    out = []
    keys = set(observation) - {'rgb'}
    shared = set(VARIABLES['shared']['local_state'])
    extra = set(VARIABLES['tasks'][task]['local_state_extra'])
    if task in DYNAMIC and not protocol_active:
        extra = {'view_step'}
    missing = (shared | extra) - set(observation)
    out.append({'check': 'public observation has declared local-state fields',
                'status': 'pass' if not missing else 'fail', 'detail': f'missing {sorted(missing)}' if missing else ''})
    text = json.dumps({k: observation[k] for k in keys if k in ('instruction', 'feedback', 'memory_checkpoint')})
    leaks = sorted(set(PRIVATE.findall(text)))
    out.append({'check': 'no private object IDs in public text', 'status': 'pass' if not leaks else 'fail',
                'detail': ', '.join(leaks)})
    return out, sorted(keys)


# ---------------------------------------------------------------- sources

def frame_check(first, second, directory):
    """Compare two renders of the same recipe; save both when they differ.

    State variables are compared exactly; pixels tolerate renderer noise. A mean
    difference below 1/255 on fewer than 1% of pixels passes; up to 5% of pixels warns.
    """
    name = 'same recipe renders the same first frame'
    if first.shape != second.shape:
        return {'check': name, 'status': 'fail', 'detail': f'shape {first.shape} vs {second.shape}'}
    difference = np.abs(first.astype(int) - second.astype(int)).max(axis=2)
    mean, changed = float(difference.mean()), float(np.count_nonzero(difference > 8) / difference.size)
    if mean == 0:
        return {'check': name, 'status': 'pass', 'detail': 'bit-identical'}
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(first).save(directory / 'first_frame_a.png')
    Image.fromarray(second).save(directory / 'first_frame_b.png')
    status = 'pass' if mean < 1 and changed < .01 else ('warn' if changed < .05 else 'fail')
    return {'check': name, 'status': status,
            'detail': f'mean |diff| {mean:.2f}/255, {100 * changed:.2f}% pixels differ by >8; frames saved in {directory.name}'}


def live_case(config, determinism, runtime_root):
    task = config.task_type
    runtime = {'runtime_dir': str(runtime_root / config.episode_id)} if config.environment != 'shape3d' else {}
    record = {'task': task, 'case': config.episode_id, 'source': 'live', 'controls': config.to_dict()}
    signatures = []
    for attempt in range(2 if determinism else 1):
        env = make_env(config, **runtime)
        try:
            started = time.perf_counter()
            try:
                observation, _ = env.reset()
            except Exception as error:  # a rejected recipe is a generation result
                record.update(admitted=False, reset_error=f'{type(error).__name__}: {error}')
                return record
            record['reset_seconds'] = round(time.perf_counter() - started, 2)
            if config.environment == 'shape3d':
                variables = shape_variables(env)
            else:
                state = env.privileged_state()
                if task in DYNAMIC:
                    variables = dynamic_variables(task, state)
                else:
                    variables = thor_variables(task, state, env.initial_pose)
            frame = np.ascontiguousarray(observation['rgb']).copy()
            signatures.append((json.dumps(variables, sort_keys=True, default=str), frame))
            if attempt == 0:
                record['admitted'] = True
                active = config.task_config.get('memory_protocol') == 'active_scan_v1'
                local, keys = local_state_checks(task, observation, active)
                record['local_state'] = {'fields': keys, 'phase': observation.get('phase'),
                                         'instruction': observation.get('instruction', '')[:400]}
                record['global_memory'] = variables
                record['checks'] = expectations(task, config.to_dict(), variables) + local
                record['first_frame'] = hashlib.sha256(frame.tobytes()).hexdigest()[:16]
        finally:
            env.close()
    if determinism:
        (first_vars, first_frame), (second_vars, second_frame) = signatures
        a, b = json.loads(first_vars), json.loads(second_vars)
        changed = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        record['checks'].append({'check': 'same recipe regenerates identical global-memory variables',
                                 'status': 'pass' if not changed else 'fail',
                                 'detail': '; '.join(f'{k}: {a.get(k)} vs {b.get(k)}' for k in changed)})
        record['checks'].append(frame_check(first_frame, second_frame, runtime_root.parent / 'frames' / config.episode_id))
    return record


def recorded_shape(config, state):
    """Rebuild the attributes shape_variables() reads from a recording's step-0 privileged state."""
    from types import SimpleNamespace
    truth, cfg = state['ground_truth'], config['task_config']
    candidates = [np.array(c) for c in state['candidate_geometry']]
    correct = truth['candidate_id']
    return SimpleNamespace(
        config=SimpleNamespace(task_type=config['task_type']), cells=np.array(truth['shape']),
        orientation=np.array(state['orientation']), reference=np.array(truth['reference_orientation']),
        candidates=candidates, correct=correct or 'candidate_1', ambiguous=truth['ambiguous_front_silhouette'],
        ambiguity_witness=state.get('ambiguity_witness'), axes=cfg.get('axes', 'XYZ'),
        camera=np.array(state['camera_position']), yaw=state['camera_yaw'], pitch=state['camera_pitch'],
        size=int(cfg.get('image_size', 224)), occlusion=cfg.get('occlusion', .35 if config['difficulty'] >= 4 else 0.))


def recorded_case(directory):
    directory = Path(directory)
    config = json.loads((directory / 'episode.json').read_text())
    task = config['task_type']
    p = json.loads((directory / 'ground_truth.json').read_text())
    first = None
    if (directory / 'privileged.jsonl').exists():
        with (directory / 'privileged.jsonl').open() as handle:
            first = json.loads(handle.readline())['state']
    if task in SHAPE:
        variables = shape_variables(recorded_shape(config, first))
    elif task in DYNAMIC:
        variables = dynamic_variables(task, p)
    else:
        variables = thor_variables(task, p, first.get('agent') if first else None)
    with (directory / 'trajectory.jsonl').open() as handle:
        row = json.loads(handle.readline())
    observation = {**row['observation'], 'rgb': row.get('frame')}  # recordings store RGB as a frame file
    active = config['task_config'].get('memory_protocol') == 'active_scan_v1'
    local, keys = local_state_checks(task, observation, active)
    for item in local:  # recordings may predate the current public protocol
        if item['status'] == 'fail':
            item.update(status='warn', detail=item['detail'] + ' (historical recording protocol)')
    return {'task': task, 'case': directory.name, 'source': 'recorded', 'evidence': str(directory.relative_to(ROOT)),
            'controls': config, 'admitted': True,
            'local_state': {'fields': keys, 'phase': observation.get('phase'),
                            'instruction': observation.get('instruction', '')[:400]},
            'global_memory': variables, 'checks': expectations(task, config, variables) + local}


def live_grid(tasks, seeds, levels_override=None):
    """Seed x difficulty grid plus explicit control overrides for every requested task."""
    grid = []

    def add(task, seed, level, task_config=None, max_steps=None, tag=''):
        environment = 'shape3d' if task in SHAPE else ('dynamic_memory' if task in DYNAMIC else 'thor_tasks')
        cfg = {'split': 'development', 'image_size': 160 if environment != 'shape3d' else 96, **(task_config or {})}
        steps = max_steps or (60 if environment == 'shape3d' else 240)
        grid.append(EpisodeConfig(episode_id=f'{task}-L{level}-seed{seed}{tag}', environment=environment,
                                  task_type=task, scene_id='procedural' if environment == 'shape3d' else 'FloorPlan1',
                                  seed=seed, difficulty=level, max_steps=steps, task_config=cfg))

    for task in tasks:
        levels = levels_override or ((1, 2, 3, 4, 5) if task in SHAPE else
                                     ((1, 2) if task == 'missing_object' else (1, 2, 3)))
        for level in levels:
            for seed in seeds:
                add(task, seed, level)
        first = seeds[0]
        if task == 'shape_identification':
            add(task, first, 2, {'candidates': 4, 'voxels': 8}, tag='-override')
            add(task, first, 2, {'occlusion': .5}, tag='-occlusion')
            for seed in seeds[:2]:
                add(task, seed, 2, {'require_initial_pixel_ambiguity': True, 'initial_rotation': 0, 'voxels': 6},
                    tag='-ambiguous')
        elif task == 'pose_alignment':
            add(task, first, 3, {'axes': 'Z'}, tag='-axesZ')
            add(task, first, 2, {'initial_rotation': 5}, tag='-rotation5')
        elif task in DYNAMIC:
            add(task, first, 1, {'memory_protocol': 'active_scan_v1', 'exploration_budget': 96}, tag='-active')
        elif task == 'memory_maze':
            add(task, first, 2, {'memory_protocol': 'explicit_recall_v1', 'exploration_budget': 120}, tag='-recall')
        elif task == 'dynamic_replanning':
            add(task, first, 2, {'min_detour_steps': 4}, tag='-detour4')
        elif task == 'relational_rearrangement':
            add(task, first, 1, {'max_satisfying_fraction': 1.0}, tag='-unordered')
            for seed in seeds:
                add(task, seed, 2, {'fixture_order': 'seeded'}, tag='-fixture')
        if task in THOR:
            for seed in seeds:
                add(task, seed, 1, {'start_pose': 'seeded'}, tag='-start')
    return grid


def shape_balance(seeds):
    """Correct-candidate position over many seeds: a fixed position would be a policy shortcut."""
    counts = Counter()
    for seed in seeds:
        env = make_env(EpisodeConfig(seed=seed, difficulty=3, max_steps=60,
                                     task_config={'split': 'development', 'image_size': 64}))
        env.reset()
        counts[env.correct] += 1
    expected = len(seeds) / max(len(counts), 1)
    chi2 = sum((c - expected) ** 2 / expected for c in counts.values())
    return {'seeds': len(seeds), 'difficulty': 3, 'counts': dict(sorted(counts.items())), 'chi_square': round(chi2, 2),
            'note': 'Five positions at L3; chi-square above 13.3 (df=4, p=.01) would indicate positional bias.'}


def cross_case(records):
    summary = {}
    for task in sorted({r['task'] for r in records}):
        rows = [r for r in records if r['task'] == task]
        admitted = [r for r in rows if r.get('admitted')]
        statuses = Counter(c['status'] for r in admitted for c in r['checks'])
        signatures = Counter(json.dumps(r['global_memory'], sort_keys=True, default=str) for r in admitted)
        item = {'cases': len(rows), 'admitted': len(admitted),
                'rejected': {r['case']: r['reset_error'] for r in rows if not r.get('admitted')},
                'checks': dict(statuses),
                'failed': sorted({f"{r['case']}: {c['check']}" for r in admitted for c in r['checks'] if c['status'] == 'fail'}),
                'warned': sorted({f"{r['case']}: {c['check']} ({c['detail']})" for r in admitted
                                  for c in r['checks'] if c['status'] == 'warn'}),
                'distinct_variable_sets': len(signatures)}
        starts = {json.dumps(r['global_memory'].get('start_position')) for r in admitted
                  if r['global_memory'].get('start_position')}
        if starts:
            item['distinct_start_positions'] = len(starts)
        summary[task] = item
    return summary


def write_result(args, records, complete):
    result = {'schema': 'task-generation-audit/1', 'source': args.source, 'generated_at': utc(),
              'complete': complete, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'variable_model': VARIABLES,
              'summary': cross_case(records), 'cases': records}
    if complete and args.source == 'live' and 'shape_identification' in args.tasks and args.balance_seeds:
        result['shape_answer_balance'] = shape_balance(range(5000, 5000 + args.balance_seeds))
    (args.output / 'task_generation_audit.json').write_text(json.dumps(result, indent=2, default=str))
    return result


def write_metadata(output, args, state, failure='Not recorded', exit_code=None):
    job = os.environ.get('SLURM_JOB_ID')
    metadata = {
        'run_id': output.name, 'project_id': output.name, 'project_name': 'SpatialAgent task demo',
        'task_name': 'task-generation-audit', 'task_id': 'task-generation-audit',
        'purpose': 'Check that generated tasks follow their controls, are deterministic, and that local-state and '
                   'global-memory variables match task_variables.json expectations. No policy or model acts.',
        'dataset': {'name': 'generated recipes' if args.source == 'live' else 'demo recorded episodes',
                    'version': 'task_variables.json spatial-task-variables/1', 'split': 'development',
                    'tasks': args.tasks, 'materials': [str(TASKS_DIR / 'task_variables.json')]},
        'experiment_type': 'environment generation audit',
        'model': {'name': 'none', 'checkpoint': None, 'precision': None},
        'environment': {'name': 'Shape3D NumPy renderer; AI2-THOR 5.0.0 (live native only, CPU lavapipe)',
                        'version': 'workspace'},
        'source': {'commit': 'workspace (not a git repository)', 'entrypoint': 'scripts/experiments/task_generation_audit.py',
                   'config': vars(args) | {'output': str(args.output)}},
        'parameters': {'source': args.source, 'seeds': args.seeds, 'determinism': not args.no_determinism,
                       'balance_seeds': args.balance_seeds},
        'slurm': {'job_id': job, 'partition': os.environ.get('SLURM_JOB_PARTITION'),
                  'resources': f"cpus={os.environ.get('SLURM_CPUS_PER_TASK')}", 'state': state,
                  'created_at': utc(), 'started_at': utc(), 'ended_at': None if exit_code is None else utc(),
                  'exit_code': exit_code, 'failure_reason': failure},
        'artifacts': {'result_dir': str(output), 'primary_metrics': str(output / 'task_generation_audit.json'),
                      'report': str(output / 'cases.jsonl'), 'trajectory': None, 'logs': []},
        'notes': 'Recorded mode reads historical evidence that may predate current protocols; live native mode '
                 'requires an allocation. warn = admissible but weak for memory training.'}
    (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--source', choices=('live', 'recorded'), required=True)
    parser.add_argument('--tasks', nargs='*', default=None)
    parser.add_argument('--seeds', type=int, nargs='*', default=[3000, 3001, 3002])
    parser.add_argument('--balance-seeds', type=int, default=60, help='Shape answer-position sweep (0 disables)')
    parser.add_argument('--no-determinism', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose a fresh output directory; existing evidence is preserved')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.tasks is None:
        args.tasks = list(SHAPE) if args.source == 'live' else list(SHAPE + DYNAMIC + THOR)
    write_metadata(args.output, args, 'RUNNING')
    records, exit_code, failure = [], 0, 'Not recorded'
    try:
        if args.source == 'live':
            units = live_grid(args.tasks, args.seeds)
            run = lambda config: live_case(config, not args.no_determinism, args.output / 'thor-runtime')  # noqa: E731
        else:
            sources = json.loads((ROOT / '04_AgentTask/Demos/spatial_agent/demo.json').read_text())['episode_globs']
            units = sorted({str(Path(p).parent) for pattern in sources for p in glob.glob(str(ROOT / pattern))
                            if (Path(p).parent / 'ground_truth.json').exists()})
            units = [u for u in units if json.loads((Path(u) / 'episode.json').read_text())['task_type'] in args.tasks]
            run = recorded_case
        for unit in units:
            try:
                record = run(unit)
            except Exception as error:
                name = getattr(unit, 'episode_id', str(unit))
                record = {'task': getattr(unit, 'task_type', None) or json.loads((Path(unit) / 'episode.json').read_text())['task_type'],
                          'case': name, 'admitted': False, 'reset_error': f'crash {type(error).__name__}: {error}',
                          'traceback': traceback.format_exc()}
            records.append(record)
            with (args.output / 'cases.jsonl').open('a') as handle:
                handle.write(json.dumps(record, default=str) + '\n')
            # Rewrite the summary after every case so a time-limited job leaves usable partial results.
            write_result(args, records, complete=False)
            bad = [c['check'] for c in record.get('checks', []) if c['status'] == 'fail']
            print(f"{record['case']}: admitted={record.get('admitted')} fail={bad or '-'} "
                  f"{record.get('reset_error', '')}", flush=True)
        result = write_result(args, records, complete=True)
        print(json.dumps({t: {k: s[k] for k in ('cases', 'admitted', 'checks', 'distinct_variable_sets')}
                          for t, s in result['summary'].items()}, indent=1))
    except BaseException as error:
        exit_code, failure = 1, f'{type(error).__name__}: {error}'
        raise
    finally:
        write_metadata(args.output, args, 'COMPLETED' if exit_code == 0 else 'FAILED', failure, exit_code)


if __name__ == '__main__':
    main()
