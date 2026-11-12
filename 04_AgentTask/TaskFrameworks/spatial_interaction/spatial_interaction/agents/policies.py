"""Observation-only random baseline and explicitly privileged oracle diagnostics."""
import numpy as np
from ..core.actions import CAMERA, THOR
from ..environments.shape3d.geometry import shortest_alignment


class RandomAgent:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, observation, available_actions):
        names = [a for a in available_actions if a not in ('OpenObject','CloseObject','PickupObject')]
        name = str(self.rng.choice(names))
        if name in ('PutObject','MARK','INSPECT','CLEAN','POWER_ON','PUT_INSIDE'):
            # Public protocol: a random point on the current view, never a private ID.
            x, y = self.rng.random(2)
            return {'action':name,'object_id':f"view:{observation.get('view_step', 0)}:{x:.3f}:{y:.3f}"}
        if name=='PLACE':return {'action':name,'slot_id':'slot_1'}
        if name == 'SUBMIT':
            return {'action': name, 'answer': {'changes': []}}
        if name == 'SUBMIT_SHAPE':
            return {'action': name, 'candidate_id': 'candidate_1'}
        return {'action': name}


class OracleAgent:
    """Uses evaluator state to prove mechanical solvability; never a VLM benchmark score."""
    def __init__(self, env):
        self.env = env
        self.visited = False
        env.privileged_references = True

    def act(self, observation, available_actions):
        env = self.env
        if env.config.environment=='thor_tasks':return env.oracle_action()
        if env.config.environment == 'dynamic_memory':
            if observation['phase'] == 'memorization':
                if not self.visited:
                    self.visited = True
                    return {'action': 'LookDown', 'memory':'Diagnostic oracle memorization checkpoint'}
                return {'action':'MEMORIZE'}
            return {'action':'SUBMIT', 'answer':env.privileged_state()['ground_truth']}
        if not self.visited and env.config.max_steps > 5:
            self.visited = True
            return {'action':'CAMERA_RIGHT'}
        if env.config.task_type == 'shape_identification':
            return {'action':'SUBMIT_SHAPE', 'candidate_id':env.ground_truth()['candidate_id']}
        path = shortest_alignment(env.orientation, env.reference, env.cells, env.axes)
        return {'action':path[0] if path else 'SUBMIT_POSE'}
