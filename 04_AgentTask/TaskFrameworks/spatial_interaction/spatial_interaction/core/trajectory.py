"""Public trajectory plus optional, separate privileged evaluator log."""
from pathlib import Path
from PIL import Image
import hashlib
import json
import numpy as np


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def frame_hash(rgb):
    return hashlib.sha256(rgb.tobytes()).hexdigest()


class TrajectoryRecorder:
    def __init__(self, directory, config, debug=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / 'episode.json').write_text(json.dumps(config.to_dict(), indent=2))
        self.trace = (self.directory / 'trajectory.jsonl').open('w')
        self.debug = (self.directory / 'privileged.jsonl').open('w') if debug else None
        self.index = 0

    def record(self, observation, action=None, reward=0.0, terminated=False, truncated=False, info=None, state=None,
               withheld=False):
        filename = f'frame_{self.index:04d}.png'
        Image.fromarray(observation['rgb']).save(self.directory / filename)
        row = {'step': self.index, 'action': action, 'reward': reward, 'terminated': terminated,
               'truncated': truncated, 'observation': {k: v for k, v in observation.items() if k != 'rgb'},
               'frame': filename, 'frame_sha256': frame_hash(observation['rgb'])}
        if withheld:
            row['withheld'] = True  # saved for researchers; never shown to the policy
        self.trace.write(json.dumps(row, default=jsonable) + '\n')
        self.trace.flush()
        if self.debug:
            self.debug.write(json.dumps({'step': self.index, 'info': info, 'state': state}, default=jsonable) + '\n')
            self.debug.flush()
        self.index += 1

    def close(self):
        self.trace.close()
        if self.debug:
            self.debug.close()


def export_gif(directory, duration=250):
    directory = Path(directory)
    rows = [json.loads(x) for x in (directory / 'trajectory.jsonl').read_text().splitlines()]
    frames = [Image.open(directory / row['frame']).convert('RGB') for row in rows]
    if not frames:
        raise ValueError('No frames')
    target = directory / 'replay.gif'
    frames[0].save(target, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    for frame in frames:
        frame.close()
    return target


def recorded_with_privileged_references(directory):
    """Replay must re-apply the reference policy that was active when recording.

    Oracle traces use private IDs. Model traces do not. Traces without a report or
    model log predate the public-reference gate and were recorded with private IDs allowed.
    """
    directory = Path(directory)
    if (directory / 'report.json').exists():
        return json.loads((directory / 'report.json').read_text()).get('policy') == 'oracle'
    return not (directory / 'model_calls.jsonl').exists()


def replay(env, directory, strict_pixels=False):
    rows = [json.loads(x) for x in (Path(directory) / 'trajectory.jsonl').read_text().splitlines()]
    if hasattr(env, 'privileged_references'):
        env.privileged_references = recorded_with_privileged_references(directory)
    max_pixel_error = 0
    max_outlier_pixels = 0
    max_mean_pixel_error = 0.
    native=env.config.environment in ('dynamic_memory','thor_tasks')
    private_path=Path(directory)/'privileged.jsonl'
    private_rows=[json.loads(line)['state'] for line in private_path.read_text().splitlines()] if native and private_path.exists() else None

    def check_state(index):
        if private_rows is None:return
        expected=private_rows[index];actual=env.privileged_state()
        if expected['id_map']!=actual['id_map']:raise AssertionError(f'Native object binding differs at step {index}')
        before=expected['current_state'];after=actual['current_state']
        if set(before)!=set(after):raise AssertionError(f'Native object set differs at step {index}')
        for oid,obj in before.items():
            for field in ('position','rotation'):
                if not np.allclose([obj[field][a] for a in 'xyz'],[after[oid][field][a] for a in 'xyz'],atol=.0001,rtol=0):
                    raise AssertionError(f'Native object {field} differs at step {index}')
            for field in ('isOpen','isPickedUp','isDirty','isToggled','isBroken','parentReceptacles'):
                if obj.get(field)!=after[oid].get(field):raise AssertionError(f'Native object state differs at step {index}')
        key='agent' if 'agent' in expected else 'current_pose'
        for field in ('position','rotation'):
            if not np.allclose([expected[key][field][a] for a in 'xyz'],[actual[key][field][a] for a in 'xyz'],atol=.0001,rtol=0):
                raise AssertionError(f'Native camera {field} differs at step {index}')
        if abs(expected[key]['cameraHorizon']-actual[key]['cameraHorizon'])>.0001:
            raise AssertionError(f'Native camera pitch differs at step {index}')
        if expected.get('task_state',{})!=actual.get('task_state',{}):raise AssertionError(f'Native task state differs at step {index}')
        for field in ('progress','events'):
            if expected.get(field)!=actual.get(field):raise AssertionError(f'Native task history differs at step {index}')

    def check_frame(rgb, row):
        nonlocal max_pixel_error, max_mean_pixel_error, max_outlier_pixels
        if frame_hash(rgb) == row['frame_sha256']:
            return
        if strict_pixels or env.config.environment not in ('dynamic_memory','thor_tasks'):
            raise AssertionError(f"Frame mismatch at step {row['step']}")
        # Across fresh Unity processes, equal physical state can have isolated
        # raster edge differences (measured: one of 16384 pixels, max 80).
        # Bound mean error and the number of pixels exceeding 8, rather than
        # accepting a broad similarity score. Native state is checked separately.
        with Image.open(Path(directory) / row['frame']) as saved:
            delta = np.abs(rgb.astype(np.int16)-np.asarray(saved).astype(np.int16))
        max_pixel_error = max(max_pixel_error, int(delta.max()))
        max_mean_pixel_error = max(max_mean_pixel_error, float(delta.mean()))
        outliers=int(np.count_nonzero(delta.max(axis=2)>8))
        max_outlier_pixels=max(max_outlier_pixels,outliers)
        if outliers>max(1,int(rgb.shape[0]*rgb.shape[1]*.001)) or delta.mean() > .5:
            stem=Path(directory)/f"replay_mismatch_{row['step']:04d}"
            Image.fromarray(rgb).save(str(stem)+'.png')
            Path(str(stem)+'.json').write_text(json.dumps({'max_pixel_error':int(delta.max()),
                'mean_pixel_error':float(delta.mean()),'outlier_pixels':outliers,'state':env.privileged_state()},default=jsonable,indent=2))
            raise AssertionError(f"Native frame differs beyond raster tolerance at step {row['step']}: max={delta.max()}, mean={delta.mean():.6f}")

    obs, _ = env.reset()
    check_state(0)
    check_frame(obs['rgb'], rows[0])
    for row in rows[1:]:
        obs, reward, terminated, truncated, _ = env.step(row['action'])
        if row.get('withheld'):
            env.withhold_view(row['step'])
        check_state(row['step'])
        check_frame(obs['rgb'], row)
        if (not np.isclose(reward, row['reward']) or terminated != row['terminated'] or truncated != row['truncated'] or
                {k:v for k,v in obs.items() if k != 'rgb'} != row['observation']):
            raise AssertionError(f"Replay mismatch at step {row['step']}")
    return {'replayed_steps': len(rows) - 1, 'verified': True,
            'exact_pixels': max_pixel_error == 0, 'native_state_verified':private_rows is not None, 'max_pixel_error': max_pixel_error,
            'max_mean_pixel_error': max_mean_pixel_error,'max_outlier_pixels_over_8':max_outlier_pixels,
            'native_outlier_fraction_limit':.001 if native else 0.}


def export_debug_page(directory):
    """Offline evaluator view. This page must never be supplied as an observation."""
    from html import escape
    directory = Path(directory)
    rows = [json.loads(x) for x in (directory/'trajectory.jsonl').read_text().splitlines()]
    truth = (directory/'ground_truth.json').read_text()
    first = 'debug_initial.png' if (directory/'debug_initial.png').exists() else rows[0]['frame']
    last = 'debug_changed.png' if (directory/'debug_changed.png').exists() else rows[-1]['frame']
    html = ('<!doctype html><meta charset="utf-8"><title>Evaluator debug view</title>'
            '<style>body{background:#172031;color:#eee;font:16px sans-serif;margin:24px}'
            'img{max-width:46%;vertical-align:top;margin:8px}pre{white-space:pre-wrap}</style>'
            '<h1>Evaluator debug view — privileged</h1><p>Initial / changed or final observations</p>'
            f'<img src="{first}"><img src="{last}"><p><a href="replay.gif">Animation</a> · '
            '<a href="trajectory.jsonl">Public trajectory</a></p><h2>Ground truth and simulator state</h2>'
            '<pre>'+escape(truth)+'</pre>')
    (directory/'debug.html').write_text(html)
    return directory/'debug.html'
