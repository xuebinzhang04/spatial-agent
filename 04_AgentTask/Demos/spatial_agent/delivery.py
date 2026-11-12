"""DeliveryBench environment adapter, migrated from the original tested bridge."""
from __future__ import annotations
import base64
import json
from pathlib import Path
import shutil
import threading
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / '04_AgentTask/SpatialAgentBench/DeliveryBench'

def utc():
    return datetime.now(timezone.utc).isoformat()

def jsonable(v):
    if hasattr(v, 'tolist'):
        return v.tolist()
    if hasattr(v, 'item'):
        return v.item()
    if isinstance(v, Path):
        return str(v)
    raise TypeError(type(v).__name__)

def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=jsonable))
    temporary.replace(path)


def isolated_checkout(run, map_name):
    """Keep upstream input files and old output paths untouched."""
    checkout = run / 'checkout' / 'task' / 'DeliveryBench'
    checkout.mkdir(parents=True, exist_ok=True)
    package = checkout / 'vlm_delivery'
    package.mkdir(exist_ok=True)
    for item in TASK.iterdir():
        if item.name not in ('vlm_delivery', '.git'):
            target = checkout / item.name
            if not target.exists():
                target.symlink_to(item, target_is_directory=item.is_dir())
    for item in (TASK / 'vlm_delivery').iterdir():
        if item.name not in ('input', '__pycache__'):
            target = package / item.name
            if not target.exists():
                target.symlink_to(item, target_is_directory=item.is_dir())
    inputs = package / 'input'
    inputs.mkdir(exist_ok=True)
    for item in (TASK / 'vlm_delivery/input').glob('*.json'):
        shutil.copy2(item, inputs / item.name)
    config = {'gym_env': {'map_name': map_name, 'base_dir': str(checkout), 'ue_ip': '127.0.0.1', 'ue_port': 9000},
              'lifecycle': {'export_path': str(run / 'reports')},
              'trajectory_output_dir': str(run / 'trajectories'), 'agent_count': 1, 'multi_agent': False}
    write_json(inputs / 'experiment_config.json', config)
    return checkout


class Harness:
    """Serialize decisions; observe real fields rather than the Gym fallback state."""
    def __init__(self, env, run, args):
        self.env, self.run, self.args = env, run, args
        self.lock = threading.Lock()
        self.counter = 0
        self.ended = False

    def observe(self):
        def capture():
            from vlm_delivery.utils.vlm_runtime import vlm_collect_images
            dm = self.env.dms[0]
            dm._viewer_agent_id = 1
            images = vlm_collect_images(dm)
            return {'map': self.args.map, 'seed': self.args.seed, 'sampled_at': utc(),
                    'mode': 'live UE / manual-script harness',
                    'position_cm': {'x': float(dm.x), 'y': float(dm.y)},
                    'energy_pct': float(dm.energy_pct),
                    'battery_pct': float(dm.e_scooter.battery_pct) if dm.e_scooter else None,
                    'earnings': float(dm.earnings_total), 'transport': dm.mode.value,
                    'sim_time_s': float(dm.clock.now_sim()),
                    'active_orders': [int(o.id) for o in dm.active_orders],
                    'order_pool': self.env.om.orders_text(),
                    'prompt': dm.build_vlm_input(),
                    'action_errors': dm.vlm_errors,
                    'lifecycle_done': bool(getattr(dm, '_lifecycle_done', False)),
                    'images': ['data:image/png;base64,' + base64.b64encode(img).decode() if img else None for img in images]}
        box = self.env._invoker.call(capture)
        if not box.get('ok'):
            raise RuntimeError(box.get('exc'))
        return box['result']

    def save(self, snapshot, command=None, before=None):
        self.counter += 1
        snap_dir = self.run / 'snapshots'
        snap_dir.mkdir(exist_ok=True)
        snapshot['snapshot_id'] = self.counter
        write_json(snap_dir / f'{self.counter:03d}.json', snapshot)
        write_json(self.run / 'latest_state.json', snapshot)
        for i, data in enumerate(snapshot['images']):
            if data:
                (snap_dir / f'{self.counter:03d}-{i}.png').write_bytes(base64.b64decode(data.split(',', 1)[1]))
        event = {k: v for k, v in snapshot.items() if k not in ('images', 'prompt', 'order_pool')}
        event.update(command=command, before=before)
        with (self.run / 'interaction_trace.jsonl').open('a') as f:
            f.write(json.dumps(event, default=jsonable, ensure_ascii=False) + '\n')

    def step(self, command):
        if self.ended:
            raise RuntimeError('Episode terminated/truncated; restart a new demo session.')
        if not isinstance(command, str) or len(command) > 4096:
            raise ValueError('Expected one API call under 4096 characters.')
        before = self.observe()
        obs, reward, terminated, truncated, info = self.env.step(command)
        self.ended = bool(terminated or truncated)
        snapshot = self.observe()
        snapshot['transition'] = {'observation': obs, 'reward': reward, 'terminated': terminated,
                                  'truncated': truncated, 'info': info,
                                  'action_error': snapshot['action_errors']}
        self.save(snapshot, command, {k: before[k] for k in ('position_cm', 'energy_pct', 'battery_pct', 'earnings')})
        return snapshot


