"""Split-aware deterministic recipes. Dynamic layouts and shape orbits are held out."""
from .core.episode import EpisodeConfig

SPLITS = {'development':(0, ('FloorPlan1','FloorPlan201')),
          'validation':(1_000_000, ('FloorPlan2','FloorPlan202')),
          'test':(2_000_000, ('FloorPlan3','FloorPlan203'))}


def generate_episodes(count=8, split='development', seed=42, environment=None, task_type=None):
    if split not in SPLITS or count < 1:
        raise ValueError('Unknown split or nonpositive episode count')
    offset, scenes = SPLITS[split]
    from .core.episode import TASKS
    if environment=='thor_tasks' or task_type in TASKS['thor_tasks']:
        if environment not in (None,'thor_tasks'):raise ValueError('Task does not belong to the selected environment')
        if task_type not in TASKS['thor_tasks']:raise ValueError('Select one native task_type')
        return [EpisodeConfig(episode_id=f'{split}_{task_type}_{i:04d}',environment='thor_tasks',task_type=task_type,
            scene_id=scenes[0],seed=seed+offset+i,max_steps=240,task_config={'split':split,'image_size':160}) for i in range(count)]
    tasks = [('dynamic_memory','missing_object',1), ('dynamic_memory','moved_object',3),
             ('shape3d','shape_identification',1), ('shape3d','pose_alignment',1)]
    if environment:
        tasks = [t for t in tasks if t[0] == environment]
        if not tasks:
            raise ValueError('Unknown environment')
    if task_type:
        tasks = [t for t in tasks if t[1] == task_type]
        if not tasks:
            raise ValueError('Task does not belong to the selected environment')
    episodes = []
    for i in range(count):
        env, task, level = tasks[i % len(tasks)]
        episode_id = f'{split}_{task}_{i:04d}' if task_type else f'{split}_{i:04d}'
        episodes.append(EpisodeConfig(episode_id=episode_id, environment=env, task_type=task,
            scene_id=scenes[(i//len(tasks))%len(scenes)] if env == 'dynamic_memory' else 'procedural',
            seed=seed+offset+i, difficulty=level, max_steps=80 if env == 'dynamic_memory' else 50,
            task_config={'split':split, 'image_size':160} if env == 'shape3d' else ({'exploration_budget':32, 'memory_protocol':'active_scan_v1'}
                  if env == 'dynamic_memory' else {'exploration_budget':12})))
    return episodes
