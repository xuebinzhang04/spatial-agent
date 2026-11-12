"""Native one-hand spatial placement with protected anchors and visible slots."""
from copy import deepcopy
import itertools
import numpy as np
from PIL import Image,ImageDraw
from spatial_interaction.environments.thor_tasks.surfaces import SurfaceFixtureEnv
from spatial_interaction.environments.thor_tasks.relations import relation_holds
from spatial_interaction.environments.dynamic_memory.interventions import objects,InvalidEpisode
from spatial_interaction.core.actions import InvalidAction


class TaskEnv(SurfaceFixtureEnv):
    extra_actions=('PickupObject','PLACE')
    fields=('surface_setup','slots','slot_views','surface','anchor','target','goal_slot','wrong_slot',
            'axes','near','pickup_view','goal_view','wrong_view','protected','instruction')

    def reset(self,seed=None,options=None):
        requested=self.config.seed if seed is None else seed
        cached=getattr(self,'_verified_episode',None)
        if cached and cached['seed']==requested:
            for key in self.fields:setattr(self,key,deepcopy(cached[key]))
            self.prepare(seed,options)
            return self.reset_info()
        self.surface_setup=[];self.slots={}
        self.prepare(seed,options);self.surface_fixture()
        configurations=self.relation_configurations()
        if not configurations:raise InvalidEpisode('Surface has no nontrivial requested relation with an incorrect alternative')
        errors=[]
        # Try selective relations first: when most slots satisfy the relation, a random
        # placement usually succeeds and the case barely tests spatial grounding.
        limit=float(self.config.task_config.get('max_satisfying_fraction',.5))
        anchor=objects(self.controller)[self.anchor]['position']
        def selective(configuration):
            satisfied=sum(relation_holds(p,anchor,configuration[0],configuration[1]) for p in self.slots.values())
            return satisfied<=limit*len(self.slots)
        order=sorted(self.rng.permutation(len(configurations)),key=lambda i:not selective(configurations[int(i)]))
        for index in order[:24]:
            self.axes,self.near,self.goal_slot,self.wrong_slot=configurations[int(index)]
            point=deepcopy(self.slots[self.wrong_slot]);point['y']+=.1
            self.surface_setup[-1]={'object':self.target,'point':point}
            self._reset_scene();self.refresh_navigation();self.view_cache={}
            try:
                self.pickup_view=self.view_of(self.target,36);self._teleport(self.pickup_view)
                self.native('PickupObject',objectId=self.target)
                # The hand itself can occlude a slot; validate while holding the real target.
                self.goal_view=self.slot_view(self.goal_slot)
                self.wrong_view=self.slot_view(self.wrong_slot)
                self._teleport(self.goal_view)
                super().task_action({'action':'PLACE','slot_id':self.goal_slot})
                if not self.goal_matches():raise InvalidEpisode('Native final pose violates requested geometry')
                break
            except (InvalidEpisode,InvalidAction) as error:errors.append(str(error))
        else:raise InvalidEpisode('No verified pickup and relation placement: '+'; '.join(errors[-3:]))
        self._reset_scene();self.refresh_navigation();self.view_cache={};self._teleport(self.initial_pose)
        self.protected={oid:{'position':deepcopy(o['position']),'rotation':deepcopy(o['rotation'])}
                        for oid,o in objects(self.controller).items() if oid!=self.target}
        directions={('x',1):'east (+x)',('x',-1):'west (-x)',('z',1):'north (+z)',('z',-1):'south (-z)'}
        relation=' AND '.join(directions[tuple(c)]+' of' for c in self.axes)
        proximity=f' and within {self.near} metres of it' if self.near is not None else ''
        self.instruction=(f'Pick up {self.sid(self.target)} and place it {relation} anchor {self.sid(self.anchor)}{proximity}. '
            'Keep the anchor and every other object untouched and unmoved. The hand must be empty at FINISH. '
            'Use PickupObject with object_id set to a view:STEP:X:Y reference on its visible pixels in an observed image, and PLACE with slot_id (slot_N from cyan numbered surface markers). '
            'A marker is shown only when its surface point is in the current camera view and unoccluded. '
            'Compass heading is shown at the top: N=+z, E=+x, S=-z, W=-x. Directions use this fixed world frame, not screen left/right. '
            'Move 0.25 m and turn 90 degrees; placement requires a visible slot within 1.6 m. FINISH submits the real scene state.')
        self._verified_episode={'seed':self.config.seed,**{key:deepcopy(getattr(self,key)) for key in self.fields}}
        return self.reset_info()

    def fixture_usable(self):return bool(self.relation_configurations())

    def relation_configurations(self):
        current=objects(self.controller);anchor=current[self.anchor]['position']
        configurations=[]
        axes_sets=([('x',),('z',)] if self.config.difficulty<3 else [('x','z')])
        for axes in axes_sets:
            for signs in itertools.product((-1,1),repeat=len(axes)):
                constraints=list(zip(axes,signs));near=.85 if self.config.difficulty>=2 else None
                good=[s for s,p in self.slots.items() if relation_holds(p,anchor,constraints,near)]
                bad=[s for s,p in self.slots.items() if not relation_holds(p,anchor,constraints,near)]
                configurations.extend((constraints,near,g,b) for g in good for b in bad)
        return configurations

    def goal_matches(self):
        current=objects(self.controller);target=current[self.target];anchor=current[self.anchor]
        return self.surface in (target.get('parentReceptacles') or []) and relation_holds(target['position'],anchor['position'],self.axes,self.near)

    def evaluate(self):
        current=objects(self.controller)
        def unchanged(now,saved):
            moved=np.array([now['position'][a]-saved['position'][a] for a in 'xyz'])
            # Euler degrees wrap: 359.99 and 0.0 are the same orientation.
            turned=np.array([(now['rotation'][a]-saved['rotation'][a]+180.)%360.-180. for a in 'xyz'])
            return np.linalg.norm(moved)<.01 and np.linalg.norm(turned)<.01
        preserved=all(oid in current and oid not in self.touched and unchanged(current[oid],saved)
                      for oid,saved in self.protected.items())
        geometry=self.goal_matches();empty=not self.controller.last_event.metadata.get('inventoryObjects')
        return {'success':bool(geometry and preserved and empty and self.target in self.touched),
                'relation_satisfied':bool(geometry),'protected_objects_preserved':bool(preserved),
                'hand_empty':empty,'environment_actions':self.steps}

    def oracle_action(self):
        if self.goal_matches() and self.target in self.touched:return {'action':'FINISH'}
        held={o['objectId'] for o in self.controller.last_event.metadata.get('inventoryObjects',[])}
        if self.target in held:
            return self.nav_action(self.goal_view) or {'action':'PLACE','slot_id':self.goal_slot}
        return self.nav_action(self.pickup_view) or {'action':'PickupObject','object_id':self.sid(self.target)}

    def render(self):
        rgb=super().render();frame=Image.fromarray(rgb);draw=ImageDraw.Draw(frame)
        yaw=self.controller.last_event.metadata['agent']['rotation']['y']%360
        heading=('N','E','S','W')[int(round(yaw/90))%4]
        draw.rectangle((0,0,self.size,11),fill=(15,25,32));draw.text((2,0),'Heading '+heading,fill='white')
        return np.asarray(frame).copy()

    def truth(self):
        return {'target':self.sid(self.target),'anchor':self.sid(self.anchor),'world_axes':deepcopy(self.axes),
                'near_metres':self.near,'goal_slot_witness':self.goal_slot,'incorrect_slot_witness':self.wrong_slot,
                'surface_slots':deepcopy(self.slots),'native_fixture':deepcopy(self.surface_setup)}
