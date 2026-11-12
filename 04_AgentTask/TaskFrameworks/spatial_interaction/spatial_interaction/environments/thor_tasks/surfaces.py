"""Privileged, collision-checked surface fixture generation for native tasks."""
from copy import deepcopy
import math
import numpy as np
from .placement import SurfaceTaskEnv
from ..dynamic_memory.interventions import objects,InvalidEpisode


def spread_points(raw,limit=96):
    """Cover x/z geometry; array-stride sampling can alias a receptacle grid."""
    keys=sorted({tuple(round(p[a],4) for a in 'xyz') for p in raw})
    if len(keys)<=limit:return [dict(zip('xyz',p)) for p in keys]
    coordinates=np.asarray([[p[0],p[2]] for p in keys])
    distances=np.sum((coordinates-coordinates.mean(axis=0))**2,axis=1)
    chosen=[int(np.argmin(distances))];nearest=np.full(len(keys),np.inf)
    for _ in range(limit-1):
        nearest=np.minimum(nearest,np.sum((coordinates-coordinates[chosen[-1]])**2,axis=1))
        nearest[chosen]=-1.
        chosen.append(int(np.argmax(nearest)))
    return [dict(zip('xyz',keys[i])) for i in chosen]


class SurfaceFixtureEnv(SurfaceTaskEnv):
    def _reset_scene(self):
        super()._reset_scene()
        for plan in getattr(self,'surface_setup',[]):
            result=self.controller.step(action='PlaceObjectAtPoint',objectId=plan['object'],position=plan['point'],forceKinematic=True)
            if not result.metadata['lastActionSuccess']:raise InvalidEpisode('Native surface fixture restoration failed')

    def surface_fixture(self):
        """Find two movable props and >=3 separate real placements around an anchor.

        Only episode construction uses privileged placement. Agent placement remains
        a nearby visible slot and the native held object, with collision checks.
        """
        self.generation_debug=[]
        initial=objects(self.controller)
        items=[o for o in initial.values() if o.get('pickupable') and not o.get('receptacleObjectIds') and
               max(o['axisAlignedBoundingBox']['size'].values())<.28 and
               np.prod(list(o['axisAlignedBoundingBox']['size'].values()))>.00005]
        # Anchor and target are named by type in the public instruction. Fixture
        # placement moves them, so only scene-unique types stay unambiguous.
        types=[o.get('objectType') for o in initial.values()]
        items=[o for o in items if types.count(o['objectType'])==1]
        items.sort(key=lambda o:(max(o['axisAlignedBoundingBox']['size'].values()),o['objectType']))
        order=self.config.task_config.get('fixture_order','size')
        if order=='seeded':items=[items[int(i)] for i in self.rng.permutation(len(items))]
        elif order!='size':raise ValueError("fixture_order must be 'size' or 'seeded'")
        surfaces=sorted((o for o in initial.values() if o['objectType']=='CounterTop'),
                        key=lambda o:o['axisAlignedBoundingBox']['size']['x']*o['axisAlignedBoundingBox']['size']['z'],reverse=True)
        errors=[];attempts=0
        for surface in surfaces[:6]:
            self.surface_setup=[];self._reset_scene();self.refresh_navigation();self.view_cache={}
            event=self.controller.step(action='GetSpawnCoordinatesAboveReceptacle',objectId=surface['objectId'],anywhere=True)
            raw=event.metadata.get('actionReturn') or []
            if not event.metadata['lastActionSuccess'] or len(raw)<4:continue
            # Spread the candidates across the complete surface rather than one grid corner.
            points=spread_points(raw)
            # Keep native centre-line samples as well as the 2D coverage points.
            for i in np.linspace(0,len(raw)-1,min(64,len(raw)),dtype=int):
                point={a:round(raw[int(i)][a],4) for a in 'xyz'}
                if point not in points:points.append(point)
            center=surface['axisAlignedBoundingBox']['center']
            anchors=sorted(points,key=lambda p:math.hypot(p['x']-center['x'],p['z']-center['z']))[:24]
            for sx,sz in ((1,1),(1,-1),(-1,1),(-1,-1)):
                extreme=max(points,key=lambda p:sx*p['x']+sz*p['z'])
                if extreme not in anchors:anchors.append(extreme)
            for anchor in items[:5]:
                for anchor_point in anchors:
                    self.surface_setup=[];self._reset_scene()
                    point=deepcopy(anchor_point);point['y']+=.12
                    attempts+=1
                    if attempts>400:raise InvalidEpisode('Surface fixture exceeded bounded native placement search')
                    event=self.controller.step(action='PlaceObjectAtPoint',objectId=anchor['objectId'],position=point,forceKinematic=True)
                    if not event.metadata['lastActionSuccess'] or surface['objectId'] not in (objects(self.controller)[anchor['objectId']].get('parentReceptacles') or []):
                        self.generation_debug.append({'stage':'anchor','type':anchor['objectType'],'point':point,'error':event.metadata.get('errorMessage','wrong support')});continue
                    self.surface_setup=[{'object':anchor['objectId'],'point':point}]
                    for target in [o for o in items if o['objectId']!=anchor['objectId']][:2]:
                        self._reset_scene();self.view_cache={}
                        valid=[]
                        local_points=sorted(points,key=lambda p:math.hypot(p['x']-point['x'],p['z']-point['z']))
                        for raw_point in local_points:
                            if math.hypot(raw_point['x']-point['x'],raw_point['z']-point['z'])<.22:continue
                            if any(math.hypot(raw_point['x']-p['x'],raw_point['z']-p['z'])<.2 for p in valid):continue
                            raised=deepcopy(raw_point);raised['y']+=.12
                            attempts+=1
                            if attempts>400:raise InvalidEpisode('Surface fixture exceeded bounded native placement search')
                            event=self.controller.step(action='PlaceObjectAtPoint',objectId=target['objectId'],position=raised,forceKinematic=True)
                            actual=objects(self.controller)[target['objectId']]
                            if not event.metadata['lastActionSuccess'] or surface['objectId'] not in (actual.get('parentReceptacles') or []):continue
                            box=actual['axisAlignedBoundingBox']
                            slot={'x':actual['position']['x'],'y':box['center']['y']-box['size']['y']/2+.015,'z':actual['position']['z']}
                            valid.append(slot)
                            if len(valid)>=8:break
                        if len(valid)<3:
                            self.generation_debug.append({'stage':'placement','anchor':anchor['objectType'],'target':target['objectType'],'valid_points':len(valid)});continue
                        self.surface=surface['objectId'];self.anchor=anchor['objectId'];self.target=target['objectId']
                        self.rng.shuffle(valid)
                        self.slots={f'slot_{i+1}':p for i,p in enumerate(valid)}
                        # Validate a reachable egocentric view of every slot.
                        self.surface_setup.append({'object':self.target,'point':dict(valid[0],y=valid[0]['y']+.1)})
                        self._reset_scene();self.refresh_navigation();self.view_cache={}
                        self.slot_views={}
                        for name in self.slots:
                            try:self.slot_views[name]=self.slot_view(name)
                            except InvalidEpisode:pass
                        self.generation_debug.append({'stage':'views','slots':len(self.slots),'visible_slots':len(self.slot_views)})
                        if len(self.slot_views)>=3:
                            self.slots={k:v for k,v in self.slots.items() if k in self.slot_views}
                            accepted=not hasattr(self,'fixture_usable') or self.fixture_usable()
                            self.generation_debug.append({'anchor':deepcopy(objects(self.controller)[self.anchor]['position']),
                                'slots':deepcopy(self.slots),'accepted':accepted,'surface':self.surface})
                            if accepted:return
                        self.surface_setup=self.surface_setup[:1]
            errors.append(surface['objectId'])
        raise InvalidEpisode('No collision-checked visible surface fixture: '+str(errors))
