"""Gymnasium-style signatures without a mandatory Gymnasium dependency."""
from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from .actions import parse_action, InvalidAction


class SpatialEnv(ABC):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 4}

    def __init__(self, config):
        self.config = deepcopy(config)
        self.done = True

    def _start(self, seed, options):
        if options:
            raise ValueError('Unsupported reset options; configure episodes through EpisodeConfig')
        self.config = self.config.with_seed(seed)
        self.rng = np.random.default_rng(self.config.seed)
        self.steps = 0
        self.history = []
        self.memory = ''
        self.done = False
        self.phase = 'interaction'
        self.metrics = {}

    @property
    @abstractmethod
    def available_actions(self): ...

    @abstractmethod
    def reset(self, seed=None, options=None): ...

    @abstractmethod
    def _execute(self, action):
        """Return (terminal score or None, evaluator/action metadata)."""

    def step(self, action):
        if self.done:
            raise RuntimeError('reset() required before step() or after episode end')
        self.steps += 1
        info = {'action_valid': True}
        score = None
        try:
            parsed = parse_action(action, self.available_actions)
            score, extra = self._execute(parsed)
            info.update(extra)
            self.memory = parsed.get('memory', self.memory)
            self.history.append(deepcopy(parsed))
        except InvalidAction as error:
            info.update(action_valid=False, error=str(error))
            self.history.append({'action': 'INVALID'})
        terminated = score is not None
        truncated = self.steps >= self.config.max_steps and not terminated
        self.done = terminated or truncated
        # Total action cost cannot exceed 0.1, including invalid submissions.
        reward = (float(score) if terminated else 0.0) - 0.1 / self.config.max_steps
        if truncated:
            self.metrics = self.timeout_metrics()
            info['metrics'] = self.metrics
        info.update(steps=self.steps, phase=self.phase)
        info.update(self._after_step() or {})
        info['phase'] = self.phase
        return self._observation(), reward, terminated, truncated, info

    def timeout_metrics(self):
        return {'success': False, 'environment_actions': self.steps}

    def _after_step(self):
        pass

    def withhold_view(self, step):
        """Mark the frame of `step` as never shown to the policy (multi-action turns).

        Environments that resolve references to observed views must forget it.
        """

    def _observation(self):
        return {'rgb': self.render().copy(), 'instruction': self.instruction,
                'phase': self.phase, 'action_history': deepcopy(self.history), 'memory': self.memory}

    @abstractmethod
    def render(self): ...

    @abstractmethod
    def privileged_state(self): ...

    def close(self):
        self.done = True
