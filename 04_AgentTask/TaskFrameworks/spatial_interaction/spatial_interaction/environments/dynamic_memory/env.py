"""Two-phase memory benchmark backed by real iTHOR native actions."""
from copy import deepcopy
from pathlib import Path
import math
import json
import tempfile
import re
import numpy as np
from ...core.base_env import SpatialEnv
from ...core.actions import THOR, InvalidAction
from ...core.rewards import evaluate_changes
from .backend import create_controller
from .interventions import InvalidEpisode, checked, objects, xyz, location, remove, move


class DynamicMemoryEnv(SpatialEnv):
    # Public policies must reference objects through observed pixels. Privileged
    # scripts (oracle, replay of oracle traces, construction validators) opt in.
    privileged_references = False

    def _start(self, seed, options):
        super()._start(seed, options)
        self.visual_frames = {}
        self.memory_checkpoint = ""
        self.feedback = ""
        self.memorization_view_steps = set()
        self.withheld_steps = set()

    def __init__(self, config, controller=None, runtime_dir=None):
        super().__init__(config)
        self.controller = controller
        self._runtime_temp = None
        if runtime_dir is None:
            self._runtime_temp = tempfile.TemporaryDirectory(prefix='spatial-thor-')
            runtime_dir = self._runtime_temp.name
        self.runtime_dir = str(runtime_dir)

    @property
    def active_memorization(self):
        return self.config.environment == 'dynamic_memory' and self.config.task_config.get('memory_protocol') == 'active_scan_v1'

    @property
    def available_actions(self):
        actions = THOR[:6] if self.active_memorization else THOR
        return actions + (('MEMORIZE',) if self.phase == 'memorization' else ('SUBMIT',))

    def _reset_scene(self):
        event = self.controller.reset(self.config.scene_id)
        if not event.metadata['lastActionSuccess']:
            raise InvalidEpisode('Scene reset failed')
        self.controller._spatial_inactive_ids = set()
        checked(self.controller, 'SetRandomSeed', seed=self.config.seed % (2**31))
        checked(self.controller, 'PausePhysicsAutoSim')
        path = Path(self.config.task_config.get('scene_state_path',
                    Path(__file__).parent / 'scene_states' / f'{self.config.scene_id}.json'))
        if not path.is_file():
            raise InvalidEpisode(f'Canonical scene snapshot missing: {path}; run capture_scenes.py first')
        snapshot = json.loads(path.read_text())
        from ai2thor.build import COMMIT_ID
        if snapshot['scene_id'] != self.config.scene_id or snapshot['build_id'] != COMMIT_ID:
            raise InvalidEpisode('Canonical snapshot scene/build mismatch')
        checked(self.controller, 'SetObjectPoses', objectPoses=snapshot['object_poses'], placeStationary=True)

    def _teleport(self, pose):
        checked(self.controller, 'TeleportFull', position=pose['position'], rotation=pose['rotation'],
                horizon=pose.get('cameraHorizon', 0.), standing=pose.get('isStanding', True))

    def _eligible(self):
        detections = self.controller.last_event.instance_detections2D
        found = []
        cfg = self.config.task_config
        minimum = float(cfg.get('min_target_pixels', 100 if self.config.difficulty == 1 else 8))
        maximum = float(cfg.get('max_target_pixels', 1800 if self.config.difficulty == 2 else self.size**2))
        for obj in objects(self.controller).values():
            box = detections.get(obj['objectId'])
            if (not obj.get('pickupable') or not obj.get('visible') or box is None or
                    not obj.get('parentReceptacles') or obj.get('receptacleObjectIds')):
                continue
            area = float((box[2]-box[0])*(box[3]-box[1]))
            if minimum <= area <= maximum and obj.get('distance', 0) <= cfg.get('max_target_distance', 4.5):
                if self.active_memorization:
                    parents = obj.get('parentReceptacles') or []
                    if len(parents) != 1:
                        continue
                    support = objects(self.controller).get(parents[0], {})
                    masks = self.controller.last_event.instance_masks
                    if (not support.get('visible') or parents[0] not in masks or
                            not np.any(masks[parents[0]]) or
                            (support.get('openable') and not support.get('isOpen'))):
                        continue
                found.append(obj)
        return sorted(found, key=lambda o: o['objectId'])

    def _find_initial_view(self):
        cfg = self.config.task_config
        initial = cfg.get('initial_pose')
        if initial:
            self._teleport(initial)
            if not self._eligible():
                raise InvalidEpisode('Configured initial pose has no eligible visible targets')
            return
        # Deterministically find a useful egocentric view; no overhead camera is shown.
        event = checked(self.controller, 'GetReachablePositions')
        positions = sorted(event.metadata['actionReturn'], key=lambda p: (p['x'],p['z'],p['y']))
        pickups = [o for o in objects(self.controller).values() if o.get('pickupable') and o.get('parentReceptacles')]
        order = self.rng.permutation(len(pickups))
        for index in order[:16]:
            obj = pickups[int(index)]
            target = xyz(obj)
            ranked = sorted(positions, key=lambda p: abs(math.hypot(p['x']-target[0], p['z']-target[2])-1.1))
            for position in ranked[:3]:
                yaw = math.degrees(math.atan2(target[0]-position['x'],target[2]-position['z'])) % 360
                event = self.controller.step(action='TeleportFull', position=position,
                    rotation={'x':0., 'y':yaw, 'z':0.}, horizon=20., standing=True)
                if event.metadata['lastActionSuccess'] and self._eligible():
                    return
        raise InvalidEpisode('No reachable view with an eligible target')

    def reset(self, seed=None, options=None):
        self._start(seed, options)
        self.phase = 'memorization'
        self.phase_steps = 0
        self.truth = {'changes': []}
        cfg = self.config.task_config
        if cfg.get('memory_protocol') not in (None, 'active_scan_v1'):
            raise ValueError('Unknown memory_protocol')
        self.size = int(cfg.get('image_size', 224))
        if not 64 <= self.size <= 512:
            raise ValueError('image_size must be 64..512')
        self.explore_budget = int(cfg.get('exploration_budget', min(15, self.config.max_steps//2)))
        if not 1 <= self.explore_budget < self.config.max_steps:
            raise ValueError('exploration_budget must leave time for detection')
        if self.config.difficulty == 3 and self.config.task_type != 'moved_object':
            raise ValueError('Dynamic L3 uses moved_object')
        if self.controller is None:
            self.controller = create_controller(self.config.scene_id, self.size, self.runtime_dir)
        self._reset_scene()
        self._find_initial_view()
        self.initial_pose = deepcopy(self.controller.last_event.metadata['agent'])
        initial = objects(self.controller)
        self.id_map = {oid: f'object_{i+1:03d}' for i,oid in enumerate(sorted(initial))}
        self.reverse_ids = {v:k for k,v in self.id_map.items()}
        eligible = self._eligible()
        self.plan = None
        # Preflight the exact intervention before offering an episode to an agent.
        errors = []
        for index in self.rng.permutation(len(eligible)):
            obj = eligible[int(index)]
            self.target = obj['objectId']
            try:
                if self.config.task_type == 'missing_object':
                    after = remove(self.controller, self.target)
                else:
                    after, self.plan = move(self.controller, self.target, self.id_map, self.rng)
                    self._verify_discoverable(after[self.target])
                break
            except InvalidEpisode as error:
                errors.append(str(error))
                self._reset_scene()
                self._teleport(self.initial_pose)
        else:
            raise InvalidEpisode('Intervention preflight failed: ' + '; '.join(errors))
        self._reset_scene()
        self._teleport(self.initial_pose)
        self.target_view_pose = deepcopy(self.initial_pose)
        if self.config.difficulty == 2 or self.active_memorization:
            # The target was verified from a reachable view, but L2 starts with
            # the camera turned away so recovering that view requires interaction.
            self.initial_pose = deepcopy(self.initial_pose)
            self.initial_pose['rotation']['y'] = (self.initial_pose['rotation']['y'] + 120.) % 360
            self._teleport(self.initial_pose)
            if objects(self.controller)[self.target].get('visible'):
                raise InvalidEpisode('Initial viewpoint must hide the target for this protocol')
        self.initial_state = objects(self.controller)
        if set(initial) != set(self.initial_state) or any(
                np.linalg.norm(xyz(initial[oid])-xyz(self.initial_state[oid])) > .025 for oid in initial):
            raise InvalidEpisode('Scene reset did not reproduce the preflight state')
        self.initial_rgb = self.render().copy()
        self.instruction = self.config.instruction or (
            'Memorize the scene, then use MEMORIZE to begin change detection. '
            'Objects have no identifier overlays. Reference an object or receptacle using '
            'view:STEP:X:Y, where STEP is an observed environment step and X,Y are normalized image coordinates '
            'in [0,1), measured from the top-left. The point must lie on its visible pixels. '
            'Report the changed object and previous receptacle (plus current receptacle for moved objects). '
            'Submit {"action":"SUBMIT","answer":{"changes":[{"type":"missing",'
            '"object_id":"view:0:0.5:0.5","previous_location":"view:0:0.6:0.7"}]}}. '
            'The environment performs one hidden change. Object interactions are available; '
            'your own interactions are not counted as hidden changes.')
        if self.active_memorization:
            self.instruction = self.active_instruction()
        return self._observation(), {'valid': True, 'intervention_preflight': True,
                                      'scene_id': self.config.scene_id, 'difficulty': self.config.difficulty}

    def active_instruction(self):
        missing = self.config.task_type == 'missing_object'
        operation = 'removed' if missing else 'relocated to another receptacle'
        submission = {'type':'missing' if missing else 'moved', 'object_id':'view:STEP:X:Y',
                      'previous_location':'view:STEP:X:Y'}
        if not missing:
            submission['current_location'] = 'view:STEP:X:Y'
        return (
            f'You will compare this room before and after exactly one exposed pickupable object is {operation}. '
            'The object that will change is not visible from the initial camera view; its identity is not disclosed. '
            'Survey the accessible room before committing: turn to inspect different surfaces, '
            'look up/down where useful, and move when a surface is occluded or too distant. '
            'Memorize object appearances AND their supporting receptacles, with spatial landmarks '
            'and view:STEP:X:Y references for both. Closed-container interiors are outside the task scope. '
            'Do not assume that looking at the initial image is enough. '
            'MoveAhead/MoveBack translate by 0.25 m; RotateLeft/RotateRight turn by 30 degrees. '
            'Object manipulation is disabled so only the environment changes object presence. '
            'Each action may include a memory string: it REPLACES the entire notebook, so preserve '
            'earlier object-location records and add new evidence rather than writing only your current intent. '
            'You receive only the current image, your action history, notebook and memory checkpoint; '
            'earlier images are not automatically replayed. '
            'Use {"action":"MEMORIZE","memory":"your complete object-location notebook"} '
            'when your survey is complete. This explicitly starts the hidden change and resets the camera. '
            'After the change, revisit the recorded locations to distinguish change from occlusion. '
            'Submit exactly one changed object and its ORIGINAL supporting receptacle, both referenced '
            'using pixels from previously observed images; moved objects also require the new receptacle: '
            + json.dumps({'action':'SUBMIT', 'answer':{'changes':[submission]}}) + '. '
            'STEP is the actual view_step; X,Y are normalized coordinates in [0,1) from the top-left. '
            'Use separate valid points on the object and its receptacle; placeholders are not literal values. '
            'Success requires the correct object and original receptacle, backed by pre-change observations. '
            + (('Additional episode instruction: ' + self.config.instruction) if self.config.instruction else ''))

    def _verify_discoverable(self, obj):
        # Validate existence of a reachable view of a moved object, then restore camera.
        pose = deepcopy(self.controller.last_event.metadata['agent'])
        event = checked(self.controller, 'GetReachablePositions')
        positions = event.metadata['actionReturn']
        target = xyz(obj)
        ranked = sorted(positions, key=lambda p: math.hypot(p['x']-target[0],p['z']-target[2]))
        for position in ranked[:12]:
            yaw = math.degrees(math.atan2(target[0]-position['x'],target[2]-position['z'])) % 360
            event = self.controller.step(action='TeleportFull', position=position,
                rotation={'x':0.,'y':yaw,'z':0.}, horizon=20., standing=True)
            if event.metadata['lastActionSuccess']:
                match = objects(self.controller).get(obj['objectId'], {})
                if match.get('visible') and obj['objectId'] in event.instance_detections2D:
                    self._teleport(pose)
                    return
        self._teleport(pose)
        raise InvalidEpisode('Moved object has no verified reachable visible viewpoint')

    def _intervene(self):
        before = objects(self.controller)
        # An agent may have picked up the target; restore held object through native
        # DropHandObject, then verify the pre-intervention location below.
        inventory = self.controller.last_event.metadata.get('inventoryObjects', [])
        if inventory:
            checked(self.controller, 'DropHandObject', forceAction=True)
            before = objects(self.controller)
        if self.target not in before:
            raise InvalidEpisode('Target absent before intervention')
        previous = location(before[self.target], self.id_map)
        if self.config.task_type == 'missing_object':
            after = remove(self.controller, self.target)
            change = {'type':'missing', 'object_id':self.id_map[self.target], 'previous_location':previous}
        else:
            after, _ = move(self.controller, self.target, self.id_map, self.rng, self.plan)
            change = {'type':'moved', 'object_id':self.id_map[self.target], 'previous_location':previous,
                      'current_location':location(after[self.target], self.id_map)}
        self.truth = {'changes':[change]}
        if self.config.task_config.get('reset_camera', True):
            self._teleport(self.initial_pose)
        self.changed_state = objects(self.controller)
        self.phase = 'detection'
        self.phase_steps = 0
        self.changed_rgb = self.render().copy()

    def _execute(self, action):
        name = action['action']
        if name == 'MEMORIZE':
            self.memory_checkpoint = action.get('memory', self.memory)
            self.memorization_view_steps = set(self.visual_frames)
            self._intervene()
            return None, {'phase_transition': 'memorization->detection', 'intervention_verified': True}
        if name == 'SUBMIT':
            self.metrics = self.evaluate_answer(action['answer'])
            score = float(self.metrics['success']) if self.active_memorization else self.metrics['change_f1']
            return score, {'metrics': self.metrics}
        kwargs = {}
        if name in ('OpenObject','CloseObject','PickupObject'):
            oid = self.resolve_reference(action['object_id'])
            obj = objects(self.controller).get(oid)
            capability = 'pickupable' if name == 'PickupObject' else 'openable'
            if not obj or not obj.get('visible') or not obj.get(capability):
                raise InvalidAction('Object is absent, not visible, or lacks required capability')
            kwargs['objectId'] = oid
        event = self.controller.step(action=name, **kwargs)
        if not event.metadata['lastActionSuccess']:
            # Simulator error strings can contain internal object IDs/coordinates;
            # only a generic error enters evaluator info, never observations.
            raise InvalidAction('Simulator rejected action (visibility, collision, state, or reach)')
        return None, {}

    def evaluate_answer(self, answer):
        answer = deepcopy(answer)
        for change in answer['changes']:
            for key in ('object_id', 'previous_location', 'current_location'):
                if key not in change:
                    continue
                if change[key].startswith('view:'):
                    change[key] = self.id_map[self.resolve_reference(change[key])]
                elif not self.privileged_references:
                    raise InvalidAction('Answers must reference objects as view:STEP:X:Y; private identifiers are rejected')
        if len({change['object_id'] for change in answer['changes']}) != len(answer['changes']):
            raise InvalidAction('Duplicate visual references to the same changed object')
        metrics = evaluate_changes(answer, self.truth, self.steps, self.config.max_steps)
        if self.active_memorization:
            change = self.truth['changes'][0]
            target = self.reverse_ids[change['object_id']]
            source = self.reverse_ids[change['previous_location']]
            def was_seen(oid):
                for step in self.memorization_view_steps:
                    labels, ids = self.visual_frames[step]
                    if oid in ids and np.any(labels == ids.index(oid)):
                        return True
                return False
            evidence = was_seen(target) and was_seen(source)
            metrics['pre_change_evidence'] = evidence
            metrics['success'] = metrics['success'] and evidence
            metrics['episode_reward'] = float(metrics['success']) - 0.1*self.steps/self.config.max_steps
            return metrics
        return metrics

    def _after_step(self):
        if self.phase == 'memorization' and not self.done:
            self.phase_steps += 1
            if self.phase_steps >= self.explore_budget and not self.active_memorization:
                self._intervene()
                return {'phase_transition': 'memorization->detection', 'intervention_verified': True,
                        'transition_reason': 'exploration_budget'}

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if not self.active_memorization:
            return obs, reward, terminated, truncated, info
        self.feedback = ('Action completed.' if info['action_valid'] else 'Action failed: ' + info['error'])
        if info.get('phase_transition'):
            self.feedback = 'Memory committed. One object has changed; revisit remembered locations.'
        if self.phase == 'memorization' and self.phase_steps >= self.explore_budget and not self.done:
            self.done = truncated = True
            self.metrics = self.timeout_metrics()
            self.metrics['failure_reason'] = 'memorization_deadline_without_commit'
            info['metrics'] = self.metrics
            self.feedback = 'Memorization deadline reached without MEMORIZE; episode ended without intervention.'
        return self._observation(), reward, terminated, truncated, info

    def render(self):
        event = self.controller.last_event
        frame = event.frame.copy()
        labels = np.full(frame.shape[:2], -1, dtype=np.int32)
        visible = {o['objectId'] for o in self.controller.last_event.metadata['objects'] if o.get('visible')}
        ids = []
        for oid, mask in event.instance_masks.items():
            if oid not in visible or oid not in self.id_map:
                continue
            mask = np.asarray(mask, dtype=bool)
            overlap = mask & (labels != -1)
            labels[mask & (labels == -1)] = len(ids)
            labels[overlap] = -2
            ids.append(oid)
        # Segmentation stays private; only unmodified RGB enters observations.
        self.visual_frames[self.steps] = (labels, ids)
        return frame

    def withhold_view(self, step):
        # An unseen intermediate frame can be neither referenced nor counted as evidence.
        self.visual_frames.pop(step, None)
        self.withheld_steps.add(step)

    def resolve_reference(self, reference):
        if not reference.startswith('view:'):
            if not self.privileged_references:
                raise InvalidAction('Object references must be view:STEP:X:Y; private identifiers are rejected')
            return self.reverse_ids.get(reference)
        try:
            _, step, x, y = reference.split(':')
            x, y = float(x), float(y)
            labels, ids = self.visual_frames[int(step)]
            if not (0 <= x < 1 and 0 <= y < 1):
                raise ValueError
            index = int(labels[int(y*labels.shape[0]), int(x*labels.shape[1])])
            if index < 0:
                raise ValueError
            return ids[index]
        except (ValueError, KeyError, IndexError):
            raise InvalidAction('Reference must select visible object pixels in an observed view:STEP:X:Y') from None

    def _observation(self):
        obs = super()._observation()
        obs['view_step'] = self.steps
        if self.active_memorization:
            remaining = max(0, self.config.max_steps - self.steps)
            deadline = max(0, self.explore_budget - self.phase_steps)
            stage = ('MEMORIZATION: actively survey the room and preserve object-location notes. '
                     f'{deadline} actions remain to issue MEMORIZE (commit also costs one action). '
                     'There is no automatic phase transition.' if self.phase == 'memorization' else
                     'DETECTION: compare revisited locations against your memory checkpoint, then SUBMIT. '
                     'Do not continue memorizing an unchanged initial view.')
            obs['instruction'] += f' Current stage: {stage} Total actions remaining: {remaining}.'
            obs['feedback'] = self.feedback
            obs['memory_checkpoint'] = self.memory_checkpoint
            obs['task_progress'] = {'steps_remaining': remaining,
                                    'memorization_actions_remaining': deadline if self.phase == 'memorization' else 0,
                                    'commit_required': self.phase == 'memorization'}
        def public_name(match):
            if match.group() == 'object_NNN':
                return 'view:0:0.5:0.5'
            description = self.public_description(self.reverse_ids.get(match.group()))
            if description is None:
                raise InvalidEpisode('Task goal needs a unique visual description; duplicate object types cannot be identified by private IDs')
            return description
        obs['instruction'] = re.sub(r'object_\d+|object_NNN', public_name, obs['instruction'])
        obs['instruction'] = obs['instruction'].replace('Object labels are stable visible IDs.', 'Objects have no identifier overlays.')
        obs['instruction'] += (' Object references use view:STEP:X:Y on previously observed pixels; '
                               'X,Y are normalized from the top-left. No object identifiers are supplied.')
        return obs

    def public_description(self, oid, scene=None):
        """Name an object for public instructions, or None when the type is ambiguous.

        Repeated types are described relative to a uniquely typed landmark when the
        nearest peer is at least 0.3 m closer than the next; otherwise no public
        description exists and generation must choose another object.
        """
        scene = self.initial_state if scene is None else scene
        name = scene.get(oid, {}).get('objectType')
        if name is None:
            return 'the visually specified object'
        peers = [obj for obj in scene.values() if obj.get('objectType') == name]
        if len(peers) == 1:
            return name
        for anchor in scene.values():
            anchor_type = anchor.get('objectType')
            if anchor_type == name or sum(o.get('objectType') == anchor_type for o in scene.values()) != 1:
                continue
            if 'position' not in anchor or any('position' not in obj for obj in peers):
                continue
            distances = sorted((float(np.linalg.norm(xyz(obj)-xyz(anchor))), obj['objectId']) for obj in peers)
            if distances[0][1] == oid and distances[1][0]-distances[0][0] >= .3:
                return name + ' nearest the ' + anchor_type
        return None

    def privileged_state(self):
        return {'ground_truth':deepcopy(self.truth), 'target_id':self.id_map[self.target],
                'missing_semantics':'native GameObject deactivation; not destruction',
                'inactive_object_ids': sorted(getattr(self.controller, '_spatial_inactive_ids', set())),
                'id_map':deepcopy(self.id_map), 'initial_state':deepcopy(self.initial_state),
                'current_state':objects(self.controller), 'initial_pose':deepcopy(self.initial_pose),
                'current_pose':deepcopy(self.controller.last_event.metadata['agent']), 'plan':deepcopy(self.plan)}

    def close(self):
        super().close()
        if self.controller is not None:
            self.controller.stop()
            self.controller = None
        if self._runtime_temp:
            self._runtime_temp.cleanup()
            self._runtime_temp = None
