"""Inspect completed or growing episode/session logs without controlling the agent.

Repeated actions and image hashes are diagnostic signals, not proof of uselessness.
Private simulator state, when present, is used only for the researcher report.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def read_rows(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines)-1 or line.endswith('\n'):
                raise
    return rows


def analyze(directory):
    directory = Path(directory)
    trajectory = directory/'trajectory.jsonl'
    if trajectory.exists():
        rows = read_rows(trajectory)
        private = {r['step']: r for r in read_rows(directory/'privileged.jsonl')}
    else:
        # Inference/state-read events may repeat an already completed step.
        by_step = {}
        for event in read_rows(directory/'session.jsonl'):
            state = event.get('state', {})
            if 'step' in state:
                by_step[state['step']] = state
        rows = [by_step[k] for k in sorted(by_step)]
        private = {}
    stages = {}
    commits = []
    for index, row in enumerate(rows):
        action = row.get('action')
        name = action.get('action') if isinstance(action, dict) else action
        if not name:
            continue
        # A recorded row contains the post-action observation; assign its action
        # to the preceding phase so MEMORIZE is counted as a memory-phase action.
        phase = rows[max(0,index-1)].get('observation', {}).get('phase', 'unknown')
        stage = stages.setdefault(phase, {'actions': [], 'counts': Counter(), 'known_invalid_steps': []})
        stage['actions'].append({'step':row['step'], 'action':name})
        stage['counts'][name] += 1
        info=private.get(row['step'],{}).get('info') or {}
        if info.get('action_valid') is False or row.get('action_valid') is False:
            stage['known_invalid_steps'].append(row['step'])
        if name == 'MEMORIZE':
            commits.append(row['step'])
    for stage in stages.values():
        names = [a['action'] for a in stage['actions']]
        alternating = best = 0
        for i, name in enumerate(names):
            alternating = (alternating+1 if i >= 2 and name == names[i-2] and name != names[i-1] else min(i+1,2))
            best = max(best,alternating)
        stage['longest_two_action_alternation'] = best if best >= 4 else 0
        stage['counts'] = dict(stage['counts'])
    translations = yaw_changes = None
    distance = None
    target_visible_initially = None
    if private:
        poses = [(private[r['step']]['state'].get('current_pose') or private[r['step']]['state'].get('agent')) for r in rows if r['step'] in private]
        poses = [p for p in poses if p]
        distance = sum(math.dist([a['position'][k] for k in ('x','z')], [b['position'][k] for k in ('x','z')]) for a,b in zip(poses,poses[1:]))
        translations = sum(math.dist([a['position'][k] for k in ('x','z')], [b['position'][k] for k in ('x','z')]) > .01 for a,b in zip(poses,poses[1:]))
        yaw_changes = sum(abs((b['rotation']['y']-a['rotation']['y']+180)%360-180) > .1 for a,b in zip(poses,poses[1:]))
        first = private.get(0, {}).get('state', {})
        target = next((oid for oid,ref in first.get('id_map',{}).items() if ref == first.get('target_id')), None)
        if target:
            target_visible_initially = first.get('current_state',{}).get(target,{}).get('visible')
    warnings = []
    if any(s['known_invalid_steps'] for s in stages.values()):
        warnings.append('Invalid actions consumed budget; inspect public feedback and action prerequisites.')
    if commits and commits[0] == 1:
        warnings.append('Memory committed immediately, before any exploration action.')
    if any(s['longest_two_action_alternation'] >= 6 for s in stages.values()):
        warnings.append('Long two-action cycle: inspect whether observations or task evidence changed.')
    if target_visible_initially:
        warnings.append('Target visible initially: this case does not require discovering it outside the initial view.')
    if translations == 0:
        warnings.append('Recorded simulator poses show no translation; travel ability was not exercised.')
    final = rows[-1] if rows else {}
    report = directory/'report.json'
    outcome = json.loads(report.read_text()).get('metrics',{}) if report.exists() else final.get('evaluation')
    files = [p for p in (trajectory,directory/'session.jsonl',directory/'privileged.jsonl',report) if p.exists()]
    return {'generated_at':datetime.now(timezone.utc).isoformat(), 'input':str(directory),
            'observed_frames':len(rows), 'memory_commit_steps':commits, 'stages':stages,
            'pose_diagnostics':{'translation_transitions':translations,'yaw_transitions':yaw_changes,
                                'recorded_pose_path_length_m':distance,'target_visible_initially':target_visible_initially},
            'outcome':outcome,'warnings':warnings,
            'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'limits':'Snapshot of recorded evidence; not proof a process is live. Pose path may include camera resets. '
                     'Repeated actions alone do not prove zero information gain. No policy correction or target disclosure.'}


def write_analysis(directory, output):
    output = Path(output)
    result = analyze(directory)
    output.parent.mkdir(parents=True,exist_ok=True)
    pending = output.with_suffix('.pending.json')
    pending.write_text(json.dumps(result,indent=2)+'\n')
    pending.replace(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = write_analysis(args.run_dir, args.output)
    print(json.dumps({'frames':result['observed_frames'],'warnings':result['warnings']}))


if __name__ == '__main__':
    main()
