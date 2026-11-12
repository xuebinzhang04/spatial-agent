"""Native container-access prerequisites, retrieval and ordered transport."""
from copy import deepcopy
import math
import numpy as np
from spatial_interaction.environments.thor_tasks.base import ThorTaskEnv
from spatial_interaction.environments.dynamic_memory.interventions import objects,InvalidEpisode


class TaskEnv(ThorTaskEnv):
    extra_actions=('OpenObject','CloseObject','PickupObject','PutObject')

    def _reset_scene(self):
        super()._reset_scene()
        if getattr(self,'generated_poses',None):
            result=self.controller.step(action='SetObjectPoses',objectPoses=self.generated_poses,placeStationary=True)
            if not result.metadata['lastActionSuccess']:raise InvalidEpisode('Generated native layout restoration failed')

    def augment_source(self):
        """Seed extra contents using collision-checked native pickup/put, not a symbolic relation."""
        initial=objects(self.controller)
        sources=[o for o in initial.values() if o.get('openable') and o.get('receptacle') and not o.get('isOpen')]
        for source in sources:
            try:view=self.access_view(source['objectId'])
            except InvalidEpisode:continue
            self._teleport(view)
            self.controller.step(action='OpenObject',objectId=source['objectId'])
            added=[]
            items=[o for o in initial.values() if o.get('pickupable') and not o.get('receptacleObjectIds') and
                   source['objectId'] not in (o.get('parentReceptacles') or []) and max(o['axisAlignedBoundingBox']['size'].values())<.3]
            for obj in items[:10]:
                result=self.controller.step(action='PickupObject',objectId=obj['objectId'],forceAction=True)
                if not result.metadata['lastActionSuccess']:continue
                result=self.controller.step(action='PutObject',objectId=source['objectId'],forceAction=True,placeStationary=True)
                if not result.metadata['lastActionSuccess']:
                    self.controller.step(action='DropHandObject',forceAction=True);continue
                if source['objectId'] not in (objects(self.controller)[obj['objectId']].get('parentReceptacles') or []):continue
                self.view_cache={}
                try:
                    self._teleport(self.view_of(obj['objectId'],36))
                    result=self.controller.step(action='PickupObject',objectId=obj['objectId'])
                    if not result.metadata['lastActionSuccess']:continue
                    result=self.controller.step(action='PutObject',objectId=source['objectId'],forceAction=True,placeStationary=True)
                    if result.metadata['lastActionSuccess']:added.append(obj['objectId'])
                except InvalidEpisode:continue
                if len(added)>=self.config.difficulty:break
            self.controller.step(action='CloseObject',objectId=source['objectId'],forceAction=True)
            if len(added)>=self.config.difficulty:
                current=objects(self.controller)
                self.generated_poses=[{'objectName':o['name'],'position':deepcopy(o['position']),'rotation':deepcopy(o['rotation'])}
                                      for o in current.values() if o.get('pickupable') or o.get('moveable')]
                self.setup=[{'source':source['objectId'],'object':oid} for oid in added]
                self._reset_scene();self.refresh_navigation();self.view_cache={}
                return
            self._reset_scene();self.refresh_navigation();self.view_cache={}
        raise InvalidEpisode('No accessible native source accepts enough retrievable objects')

    def access_view(self,oid):
        """Find a view that supports both native opening and closing, not visibility alone."""
        old=deepcopy(self.controller.last_event.metadata['agent']);obj=objects(self.controller)[oid]
        ranked=sorted(self.positions.values(),key=lambda p:abs(math.hypot(p['x']-obj['position']['x'],p['z']-obj['position']['z'])-1.3))
        try:
            for position in ranked[:48]:
                if math.hypot(position['x']-obj['position']['x'],position['z']-obj['position']['z'])>1.55:continue
                yaw=round(math.degrees(math.atan2(obj['position']['x']-position['x'],obj['position']['z']-position['z']))/90)*90%360
                for horizon in (30.,0.,60.):
                    event=self.controller.step(action='TeleportFull',position=position,rotation={'x':0.,'y':yaw,'z':0.},horizon=horizon,standing=True)
                    if not event.metadata['lastActionSuccess'] or not objects(self.controller)[oid].get('visible'):continue
                    opened=self.controller.step(action='OpenObject',objectId=oid)
                    if not opened.metadata['lastActionSuccess']:continue
                    closed=self.controller.step(action='CloseObject',objectId=oid)
                    if closed.metadata['lastActionSuccess']:return deepcopy(closed.metadata['agent'])
                    self.controller.step(action='CloseObject',objectId=oid,forceAction=True)
            raise InvalidEpisode('No native open/close access view for '+obj['objectType'])
        finally:self._teleport(old)

    def reset(self,seed=None,options=None):
        requested=self.config.seed if seed is None else seed
        cached=getattr(self,'_verified_episode',None)
        if cached and cached['seed']==requested:
            self.setup=deepcopy(cached['setup']);self.generated_poses=deepcopy(cached['poses']);self.prepare(seed,options)
            self.orders=deepcopy(cached['orders']);self.instruction=cached['instruction']
            self.collected=set();self.pending_close=None
            return self.reset_info()
        self.setup=[];self.generated_poses=[]
        self.prepare(seed,options);self.collected=set();self.pending_close=None
        initial=objects(self.controller)
        eligible=[o for o in initial.values() if o.get('pickupable') and not o.get('receptacleObjectIds') and
                  any(initial[p].get('openable') and not initial[p].get('isOpen') for p in o.get('parentReceptacles') or [])]
        if len(eligible)<self.config.difficulty:
            self.augment_source()
            # SetObjectPoses regenerates native IDs from positions. Bind public IDs
            # after the complete fixture exists; no observation has been emitted yet.
            self.id_map={oid:f'object_{i+1:03d}' for i,oid in enumerate(sorted(objects(self.controller)))}
            self.reverse_ids={v:k for k,v in self.id_map.items()}
        while True:
            initial=objects(self.controller)
            candidates=[]
            for obj in initial.values():
                if obj.get('pickupable') and not obj.get('receptacleObjectIds'):
                    for parent in obj.get('parentReceptacles') or []:
                        if initial[parent].get('openable') and initial[parent].get('receptacle') and not initial[parent].get('isOpen'):
                            if None not in (self.public_description(obj['objectId'],initial),self.public_description(parent,initial)):
                                candidates.append((obj['objectId'],parent))
                            break
            self.orders=[];errors=[]
            for index in self.rng.permutation(len(candidates)):
                oid,source=candidates[int(index)]
                try:
                    source_view=self.access_view(source);self._teleport(source_view)
                    event=self.controller.step(action='OpenObject',objectId=source)
                    if not event.metadata['lastActionSuccess']:raise InvalidEpisode('Container cannot be opened from verified view')
                    self.view_cache.pop(oid,None);item_view=self.view_of(oid,36);self._teleport(item_view)
                    event=self.controller.step(action='PickupObject',objectId=oid)
                    if not event.metadata['lastActionSuccess']:raise InvalidEpisode('Native pickup preflight failed')
                    surfaces=sorted((o for o in initial.values() if o['objectType']=='CounterTop' and
                                     self.public_description(o['objectId'],initial) is not None),key=lambda o:np.linalg.norm(np.array(list(o['position'].values()))-np.array(list(initial[source]['position'].values()))))
                    destination=None
                    for surface in surfaces:
                        try:
                            destination_view=self.view_of(surface['objectId']);self._teleport(destination_view)
                            result=self.controller.step(action='PutObject',objectId=surface['objectId'],placeStationary=True)
                            if result.metadata['lastActionSuccess'] and surface['objectId'] in (objects(self.controller)[oid].get('parentReceptacles') or []):
                                destination=surface['objectId'];break
                        except InvalidEpisode:continue
                    if destination is None:raise InvalidEpisode('No verified native destination placement')
                    self._reset_scene();self.refresh_navigation();self._teleport(item_view)
                    attempt=self.controller.step(action='PickupObject',objectId=oid)
                    if attempt.metadata['lastActionSuccess']:raise InvalidEpisode('Closed source did not enforce retrieval prerequisite')
                    self.orders.append({'object':oid,'source':source,'destination':destination,
                                        'source_view':source_view,'item_view':item_view,'destination_view':destination_view})
                except InvalidEpisode as error:errors.append(str(error))
                finally:
                    self._reset_scene();self.refresh_navigation();self.view_cache={}
                    self._teleport(self.initial_pose)
                if len(self.orders)>=self.config.difficulty:break
            if len(self.orders)>=self.config.difficulty or self.setup:break
            # Existing contents gave too few verified dependencies (e.g. inaccessible
            # containers): seed one accessible source once, then verify again.
            try:self.augment_source()
            except InvalidEpisode as error:errors.append(str(error));break
            self.id_map={oid:f'object_{i+1:03d}' for i,oid in enumerate(sorted(objects(self.controller)))}
            self.reverse_ids={v:k for k,v in self.id_map.items()}
        if len(self.orders)<self.config.difficulty:
            raise InvalidEpisode(f'Insufficient verified dependencies: {len(self.orders)}/{self.config.difficulty}; '
                                 f'{len(candidates)} candidates; setup={self.setup}; '+'; '.join(errors))
        orders='; then '.join(f'retrieve {self.sid(o["object"])} from {self.sid(o["source"])} and put it on {self.sid(o["destination"])}' for o in self.orders)
        self.instruction=('In order, '+orders+'. Open each source before retrieval. One native hand, native collision and visibility apply. '
            'Use OpenObject/CloseObject/PickupObject/PutObject with object_id set to a view:STEP:X:Y reference on its visible pixels in an observed image. PutObject references the destination surface. Close every used source before FINISH. '
            'Only nearby visible objects can be manipulated. Move 0.25 m, turn 90 degrees. '
            'This is physical container access, not simulated key locks.')
        self._verified_episode={'seed':self.config.seed,'setup':deepcopy(self.setup),'poses':deepcopy(self.generated_poses),'orders':deepcopy(self.orders),'instruction':self.instruction}
        return self.reset_info()

    def after_action(self):
        if self.progress>=len(self.orders):return
        order=self.orders[self.progress];current=objects(self.controller)
        inventory={o['objectId'] for o in self.controller.last_event.metadata.get('inventoryObjects',[])}
        if order['object'] in inventory:self.collected.add(order['object'])
        if order['object'] in self.collected and order['destination'] in (current[order['object']].get('parentReceptacles') or []):
            self.events.append({'event':'retrieved_and_delivered','object_id':self.sid(order['object']),'step':self.steps})
            self.progress+=1;self.pending_close=order

    def evaluate(self):
        current=objects(self.controller)
        delivered=sum(o['object'] in self.collected and o['destination'] in (current[o['object']].get('parentReceptacles') or []) for o in self.orders)
        closed=all(not current[o['source']]['isOpen'] for o in self.orders)
        return {'success':delivered==len(self.orders) and self.progress==len(self.orders) and closed and not self.controller.last_event.metadata.get('inventoryObjects'),
                'delivered':delivered,'required':len(self.orders),'sources_closed':closed,'environment_actions':self.steps,
                'native_prerequisite_verified':True}

    def oracle_action(self):
        current=objects(self.controller)
        if self.pending_close:
            o=self.pending_close
            if current[o['source']]['isOpen']:
                return self.nav_action(o['source_view']) or {'action':'CloseObject','object_id':self.sid(o['source'])}
            self.pending_close=None
        if self.progress>=len(self.orders):return {'action':'FINISH'}
        o=self.orders[self.progress]
        held={x['objectId'] for x in self.controller.last_event.metadata.get('inventoryObjects',[])}
        if o['object'] in held:
            return self.nav_action(o['destination_view']) or {'action':'PutObject','object_id':self.sid(o['destination'])}
        if not current[o['source']]['isOpen']:
            return self.nav_action(o['source_view']) or {'action':'OpenObject','object_id':self.sid(o['source'])}
        return self.nav_action(o['item_view']) or {'action':'PickupObject','object_id':self.sid(o['object'])}

    def truth(self):return {'orders':deepcopy(self.orders),'generated_container_placements':deepcopy(self.setup),'native_closed_pickup_rejected':True}
