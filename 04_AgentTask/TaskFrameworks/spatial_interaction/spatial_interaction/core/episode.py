"""Portable public episode recipes. Answers and simulator state never live here."""
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
import json

TASKS = {'dynamic_memory': ('missing_object', 'moved_object'),
         'shape3d': ('shape_identification', 'pose_alignment'),
         'thor_tasks': ('memory_maze','dependency_doors','relational_rearrangement',
                        'dynamic_replanning')}


@dataclass(frozen=True)
class EpisodeConfig:
    episode_id: str = 'example_0001'
    environment: str = 'shape3d'
    task_type: str = 'shape_identification'
    scene_id: str = 'procedural'
    seed: int = 42
    difficulty: int = 1
    instruction: str = ''
    max_steps: int = 50
    task_config: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.environment not in TASKS or self.task_type not in TASKS[self.environment]:
            raise ValueError('Unsupported environment/task pair')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if type(self.max_steps) is not int or self.max_steps < 2:
            raise ValueError('max_steps must be >= 2')
        if type(self.difficulty) is not int or not 1 <= self.difficulty <= 5:
            raise ValueError('difficulty must be 1..5')
        if self.environment == 'dynamic_memory' and self.difficulty > 3:
            raise ValueError('Dynamic L4/L5 are not validated/supported in Stage 1')
        if not isinstance(self.task_config, dict):
            raise ValueError('task_config must be a mapping')
        if self.environment == 'thor_tasks' and self.difficulty > 3:
            raise ValueError('Migrated native tasks currently support structural L1-L3')

    def to_dict(self):
        return asdict(self)

    def with_seed(self, seed):
        return replace(self, seed=seed) if seed is not None else self


def save_episodes(episodes, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [e.to_dict() for e in episodes]
    path.write_text('\n'.join(json.dumps(e, sort_keys=True) for e in records) + '\n'
                    if path.suffix == '.jsonl' else json.dumps(records, indent=2))


def load_episodes(path):
    path = Path(path)
    records = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
               if path.suffix == '.jsonl' else json.loads(path.read_text()))
    if isinstance(records, dict):
        records = [records]
    return [EpisodeConfig(**r) for r in records]
