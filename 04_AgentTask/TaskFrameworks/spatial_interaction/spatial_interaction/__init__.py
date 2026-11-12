from .core.episode import EpisodeConfig


def make_env(config, **kwargs):
    if config.environment == 'thor_tasks':
        import importlib.util
        from pathlib import Path
        custom=Path(__file__).resolve().parents[3]/'CustomAgentTasks'
        path=custom/config.task_type/'env.py'
        if not path.exists():raise RuntimeError(f'Native task not migrated yet: {config.task_type}')
        spec=importlib.util.spec_from_file_location('_native_task_'+config.task_type,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module.TaskEnv(config,**kwargs)
    if config.environment == 'dynamic_memory':
        from .environments.dynamic_memory.env import DynamicMemoryEnv
        return DynamicMemoryEnv(config, **kwargs)
    from .environments.shape3d.env import Shape3DEnv
    return Shape3DEnv(config, **kwargs)
