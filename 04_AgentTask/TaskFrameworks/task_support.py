"""Shared plumbing for independent task definitions under CustomAgentTasks.

Task identity/defaults live in each task's task.json; shared engines live here.
This module is evaluator/research infrastructure, never a policy observation.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys

FRAMEWORKS = Path(__file__).resolve().parent
WORKSPACE = FRAMEWORKS.parents[1]


class SingleTask:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.definition = json.loads((self.directory / 'task.json').read_text())
        self.name = self.definition['name']
        self.backend = self.definition['backend']
        if self.directory.name != self.name or self.backend != 'spatial_interaction':
            raise ValueError('Task name must match its directory and use a supported shared backend')
        self.engine_root = FRAMEWORKS / self.backend
        if str(self.engine_root) not in sys.path:
            sys.path.insert(0, str(self.engine_root))

    def _check(self, specification):
        actual = getattr(specification, 'task_type', None)
        if actual != self.name:
            raise ValueError(f'{self.name} only accepts its own task specifications, received {actual!r}')

    def generate(self, seed=None, difficulty=None, split=None, task_config=None, **options):
        defaults = self.definition['defaults']
        seed = defaults['seed'] if seed is None else seed
        difficulty = defaults['difficulty'] if difficulty is None else difficulty
        split = defaults['split'] if split is None else split
        from spatial_interaction import EpisodeConfig
        from spatial_interaction.benchmarks import generate_episodes
        cfg = generate_episodes(1, split, seed, self.definition['environment'], self.name)[0].to_dict()
        cfg.update(difficulty=difficulty, **options)
        cfg['task_config'].update(task_config or {})
        specification = EpisodeConfig(**cfg)
        self._check(specification)
        if specification.environment != self.definition['environment']:
            raise ValueError('Task environment mismatch')
        return specification

    def make_env(self, specification=None, **kwargs):
        if specification is None:
            specification = self.generate()
        self._check(specification)
        from spatial_interaction import make_env
        return make_env(specification, **kwargs)

    def evaluate(self, env, answer=None):
        """Privileged evaluator; only return metrics to the evaluation caller."""
        specification = env.config
        self._check(specification)
        if self.definition['environment']=='thor_tasks':return env.evaluate()
        if self.definition['environment'] == 'dynamic_memory':
            from spatial_interaction.core.actions import parse_action
            from spatial_interaction.core.rewards import evaluate_changes
            if env.phase != 'detection':
                raise ValueError('Changes can only be evaluated after the intervention')
            action = parse_action({'action':'SUBMIT', 'answer':answer}, ('SUBMIT',))
            return env.evaluate_answer(action['answer'])
        if self.name == 'shape_identification':
            from spatial_interaction.core.actions import parse_action
            action = parse_action({'action':'SUBMIT_SHAPE', 'candidate_id':answer['candidate_id']}, ('SUBMIT_SHAPE',))
            success = action['candidate_id'] == env.ground_truth()['candidate_id']
            return {'success':success, 'shape_accuracy':float(success)}
        from spatial_interaction.environments.shape3d.geometry import rotation_error
        error = rotation_error(env.orientation, env.reference, env.cells)
        return {'success':error < 1e-6, 'rotation_error_degrees':error}

    def cli(self, argv=None):
        """Reuse existing CLIs with a hard task boundary, including loaded corpora."""
        module_name = f'_custom_task_cli_{self.name}'
        spec = importlib.util.spec_from_file_location(module_name, self.engine_root/'run.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.main(argv, task_type=self.name)
