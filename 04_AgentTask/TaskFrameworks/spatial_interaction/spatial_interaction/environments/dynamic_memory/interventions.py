"""Simulator-native verified interventions; metadata stays evaluator-only."""
from copy import deepcopy
import numpy as np


class InvalidEpisode(RuntimeError):
    pass


def objects(controller):
    inactive = getattr(controller, '_spatial_inactive_ids', set())
    return {o['objectId']: deepcopy(o) for o in controller.last_event.metadata['objects']
            if o['objectId'] not in inactive}


def checked(controller, action, **kwargs):
    event = controller.step(action=action, **kwargs)
    if not event.metadata['lastActionSuccess']:
        raise InvalidEpisode(f"{action}: {event.metadata.get('errorMessage', '')}")
    return event


def xyz(obj):
    return np.array([obj['position'][axis] for axis in 'xyz'])


def location(obj, id_map):
    parents = obj.get('parentReceptacles') or []
    return id_map[sorted(parents)[0]] if parents else 'floor'


def verify_unchanged(before, after, target_id, tolerance=.025):
    if set(before)-{target_id} != set(after)-{target_id}:
        raise InvalidEpisode('Intervention unexpectedly added/removed distractors')
    for oid in set(before) & set(after) - {target_id}:
        if np.linalg.norm(xyz(before[oid])-xyz(after[oid])) > tolerance:
            raise InvalidEpisode(f'Intervention moved distractor {oid}')


def remove(controller, target_id):
    before = objects(controller)
    # Cached 5.0.0's destructive RemoveFromScene produces a Unity metadata
    # NullReferenceException. DisableObject calls GameObject.SetActive(false):
    # renderers, colliders and interaction scripts are inactive. This is task-scene
    # deactivation, not destruction or a visual-only occlusion trick. EnableObject
    # is deliberately absent from the agent action protocol.
    checked(controller, 'DisableObject', objectId=target_id)
    checked(controller, 'Pass')
    raw_target = next((o for o in controller.last_event.metadata['objects'] if o['objectId'] == target_id), None)
    if raw_target and (raw_target.get('visible') or raw_target.get('isInteractable')):
        raise InvalidEpisode('Deactivated target remains visible/interactable')
    # THOR retains inactive objects in its registry/metadata. Keep a separate
    # evaluator record of deactivation and exclude them from the task scene.
    controller._spatial_inactive_ids = getattr(controller, '_spatial_inactive_ids', set()) | {target_id}
    after = objects(controller)
    if target_id in after:
        raise InvalidEpisode('Deactivated target remains in accessible task scene')
    if target_id in getattr(controller.last_event, 'instance_detections2D', {}):
        raise InvalidEpisode('Removed object still appears in segmentation')
    verify_unchanged(before, after, target_id)
    return after


def move(controller, target_id, id_map, rng, plan=None):
    before = objects(controller)
    target = before[target_id]
    if not (target.get('pickupable') or target.get('moveable')):
        raise InvalidEpisode('Target cannot be relocated')
    previous = location(target, id_map)
    candidates = []
    if plan is not None:
        candidates = [plan]
    else:
        receptacles = sorted((o for o in before.values() if o.get('receptacle') and
                              not o.get('openable') and o['objectId'] != target_id and
                              id_map[o['objectId']] != previous), key=lambda o: o['objectId'])
        for index in rng.permutation(len(receptacles)):
            receptacle = receptacles[int(index)]
            event = controller.step(action='GetSpawnCoordinatesAboveReceptacle',
                                    objectId=receptacle['objectId'], anywhere=True)
            if not event.metadata['lastActionSuccess']:
                continue
            points = event.metadata.get('actionReturn') or []
            for point_index in rng.permutation(len(points))[:24]:
                point = deepcopy(points[int(point_index)])
                # Native downward placement ray should begin above the surface.
                point['y'] += .1
                candidates.append({'position': point, 'receptacle_id': receptacle['objectId']})
    original = {'position': target['position'], 'rotation': target['rotation']}
    for candidate in candidates:
        event = controller.step(action='PlaceObjectAtPoint', objectId=target_id,
                                position=candidate['position'], rotation=target['rotation'], forceKinematic=True)
        if not event.metadata['lastActionSuccess']:
            continue
        checked(controller, 'Pass')
        after = objects(controller)
        actual = after[target_id]
        if (np.linalg.norm(xyz(actual)-xyz(target)) >= .4 and
                location(actual, id_map) not in (previous, 'floor')):
            verify_unchanged(before, after, target_id)
            return after, candidate
        checked(controller, 'PlaceObjectAtPoint', objectId=target_id, **original, forceKinematic=True)
    raise InvalidEpisode('No collision-free placement on a different open receptacle found')
