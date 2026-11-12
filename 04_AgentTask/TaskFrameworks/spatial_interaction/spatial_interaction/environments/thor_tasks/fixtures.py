"""Privileged native fixture helpers; never included in agent observations."""
from copy import deepcopy
import math
from .base import point_key
from ..dynamic_memory.interventions import objects,InvalidEpisode


def capture_poses(controller):
    # SetObjectPoses disables omitted movable objects: preserve ALL of them.
    return [{'objectName':o['name'],'position':deepcopy(o['position']),'rotation':deepcopy(o['rotation'])}
            for o in objects(controller).values() if o.get('pickupable') or o.get('moveable')]


def bind_ids(env):
    env.id_map={oid:f'object_{i+1:03d}' for i,oid in enumerate(sorted(objects(env.controller)))}
    env.reverse_ids={v:k for k,v in env.id_map.items()}


def seed_on_counter(env,oid,offset=0):
    """Generation-only native pickup/put. No unchecked position teleport."""
    surfaces=[o for o in objects(env.controller).values() if o['objectType']=='CounterTop']
    surfaces.sort(key=lambda o:o['axisAlignedBoundingBox']['size']['x']*o['axisAlignedBoundingBox']['size']['z'],reverse=True)
    for surface in surfaces:
        try:view=env.view_of(surface['objectId'],36)
        except InvalidEpisode:continue
        env._teleport(view)
        result=env.controller.step(action='PickupObject',objectId=oid,forceAction=True)
        if not result.metadata['lastActionSuccess']:continue
        result=env.controller.step(action='PutObject',objectId=surface['objectId'],forceAction=True,
                                   placeStationary=True,randomSeed=(env.config.seed+offset)%(2**31))
        if result.metadata['lastActionSuccess'] and surface['objectId'] in (objects(env.controller)[oid].get('parentReceptacles') or []):
            env.view_cache={};return surface['objectId']
        env.controller.step(action='DropHandObject',forceAction=True)
    raise InvalidEpisode('No native counter placement for fixture item')


def verify_oracle(env):
    """Execute the complete native solution during construction, including hand state.

    Native reachable-cell lists occasionally overconnect corners. The oracle may
    prune ONLY an edge whose actual movement failed; agent actions remain native.
    """
    env.privileged_references=True
    for _ in range(env.config.max_steps):
        action=env.oracle_action();before=deepcopy(env.controller.last_event.metadata['agent'])
        _,_,terminated,truncated,info=env.step(action)
        if not info['action_valid']:
            if action['action']!='MoveAhead' or getattr(env,'battery',1)<=0:
                raise InvalidEpisode('Native solution action failed: '+str(action)+' '+str(info.get('error')))
            start=point_key(before['position']);yaw=math.radians(before['rotation']['y'])
            end=(round(start[0]+.25*math.sin(yaw),3),round(start[1]+.25*math.cos(yaw),3))
            if end not in env.graph.get(start,[]):raise InvalidEpisode('Native solution has an invalid grid edge')
            env.graph[start].remove(end)
            if start in env.graph.get(end,[]):env.graph[end].remove(start)
        if terminated or truncated:
            if not info['metrics']['success']:raise InvalidEpisode('Complete native solution failed: '+str(info['metrics']))
            return deepcopy(info['metrics'])
    raise InvalidEpisode('Native solution exceeded action budget')
