"""Native obstacle intervention with a physically verified alternate route."""
from copy import deepcopy
import math
import numpy as np
from PIL import Image,ImageDraw
from spatial_interaction.environments.thor_tasks.base import ThorTaskEnv,point_key
from spatial_interaction.environments.thor_tasks.placement import project
from spatial_interaction.environments.dynamic_memory.interventions import objects,InvalidEpisode
from spatial_interaction.core.actions import InvalidAction


class TaskEnv(ThorTaskEnv):
    needs_depth=True
    extra_actions=('MARK','WAIT')
    fields=('target','goal_view','checkpoint','checkpoint_view','change_plan','blocked_edge',
            'old_route_length','new_route_length','baseline_graph','changed_graph','instruction')

    def reset(self,seed=None,options=None):
        requested=self.config.seed if seed is None else seed
        cached=getattr(self,'_verified_episode',None)
        self.prepare(seed,options)
        self.generation_debug=[];self.changed=False
        if cached and cached['seed']==requested:
            for key in self.fields:setattr(self,key,deepcopy(cached[key]))
        else:
            self.generate_intervention()
            self.instruction=(f'Go to the cyan C checkpoint on the floor. When you reach it, the room will change. '
                f'Inspect your actual observations and find another route to {self.sid(self.target)}. '
                'MARK it with object_id set to a view:STEP:X:Y reference on its visible pixels in an observed image when it is visible within 1.6 metres, then FINISH while still there. '
                'Moving 0.25 m, turning 90 degrees, and looking are independent actions. '
                'You cannot move the obstacle. No map or correct route is provided. '
                'WAIT observes without moving; reaching the checkpoint is required before marking the goal.')
            self._verified_episode={'seed':self.config.seed,**{k:deepcopy(getattr(self,k)) for k in self.fields}}
        self._reset_scene();self.refresh_navigation();self.graph=deepcopy(self.baseline_graph)
        self._teleport(self.initial_pose);self.progress=0;self.events=[];self.touched=set();self.feedback=''
        self.phase='before_change'
        return self.reset_info()

    def walk_native(self,pose,limit=180):
        """Private construction witness; each edge is executed in native physics.

        GetReachablePositions can overconnect neighbouring cells at corners.
        Remove only edges whose native traversal actually failed, then replan.
        """
        for _ in range(limit):
            action=self.nav_action(pose)
            if action is None:return
            before=deepcopy(self.controller.last_event.metadata['agent'])
            kwargs={'degrees':90} if action['action'].startswith('Rotate') else {}
            event=self.controller.step(action=action['action'],**kwargs)
            if event.metadata['lastActionSuccess']:continue
            if action['action']!='MoveAhead':raise InvalidEpisode('Native witness camera action failed')
            start=point_key(before['position']);yaw=math.radians(before['rotation']['y'])
            end=(round(start[0]+.25*math.sin(yaw),3),round(start[1]+.25*math.cos(yaw),3))
            if end not in self.graph.get(start,[]):raise InvalidEpisode('Witness did not reach a valid grid edge')
            self.graph[start].remove(end)
            if start in self.graph.get(end,[]):self.graph[end].remove(start)
        raise InvalidEpisode('Native route witness exceeded bounded search')

    def place_barrier(self):
        plan=self.change_plan
        self.native('PlaceObjectAtPoint',objectId=plan['object'],position=plan['point'],
                    rotation=plan['rotation'],forceKinematic=True)
        actual=objects(self.controller)[plan['object']]['position']
        if math.hypot(actual['x']-plan['point']['x'],actual['z']-plan['point']['z'])>.04:
            raise InvalidEpisode('Native obstacle did not reach the intended floor point')

    def generate_intervention(self):
        initial=objects(self.controller)
        bodies=[o for o in initial.values() if o.get('moveable') and not o.get('pickupable') and
                o['objectType'] in ('GarbageCan','Stool','Chair') and
                .25<max(o['axisAlignedBoundingBox']['size'][a] for a in ('x','z'))<.8]
        bodies.sort(key=lambda o:(o['objectType']!='GarbageCan',o['objectId']))
        targets=[o for o in initial.values() if not o.get('pickupable') and
                 o['objectType'] in ('Fridge','Microwave','CoffeeMachine','CounterTop','Sink') and
                 self.public_description(o['objectId'],initial) is not None]
        targets.sort(key=lambda o:o['objectId'])
        if not bodies:raise InvalidEpisode('No supported native movable floor obstacle')
        candidates=[]
        for i in self.rng.permutation(len(targets))[:10]:
            obj=targets[int(i)]
            try:goal=self.view_of(obj['objectId'],24)
            except InvalidEpisode:continue
            end=point_key(goal['position'])
            for start in sorted(self.positions):
                route=self.route(start,end)
                if len(route)<8+2*self.config.difficulty:continue
                if len(self.route(point_key(self.initial_pose['position']),start))<4:continue
                # A straight-line gap keeps the intervention away from the agent.
                for k in range(4,len(route)-3):
                    if math.dist(route[k],start)<.9:continue
                    candidates.append((obj['objectId'],deepcopy(goal),start,route,k))
        if not candidates:raise InvalidEpisode('No checkpoint/goal with sufficient native route length')
        errors=[]
        for index in self.rng.permutation(len(candidates))[:40]:
            target,goal,start,route,k=candidates[int(index)]
            for body in bodies[:2]:
                try:
                    self._reset_scene();self.refresh_navigation()
                    checkpoint_view={'position':deepcopy(self.positions[start]),'rotation':{'x':0.,'y':0.,'z':0.},'cameraHorizon':30.,'isStanding':True}
                    # Demonstrate that the original full route and approach are physically walkable.
                    self._teleport(self.initial_pose);self.walk_native(checkpoint_view)
                    self._teleport(checkpoint_view);self.walk_native(goal)
                    baseline=deepcopy(self.graph);old_length=len(self.route(start,point_key(goal['position'])))-1
                    self._teleport(checkpoint_view)
                    point=deepcopy(self.positions[route[k]])
                    self.change_plan={'object':body['objectId'],'point':point,'rotation':deepcopy(body['rotation'])}
                    self.place_barrier();self.refresh_navigation()
                    if start not in self.positions or point_key(goal['position']) not in self.positions:
                        raise InvalidEpisode('Obstacle blocks checkpoint or all goal access')
                    self._teleport(checkpoint_view)
                    detour=self.route(start,point_key(goal['position']))
                    if len(detour)-1<old_length+int(self.config.task_config.get('min_detour_steps',2)):raise InvalidEpisode('Obstacle does not require a sufficiently longer alternate route')
                    # At least one previously traversable edge must now fail a real MoveAhead.
                    blocked=None
                    for edge_index in range(max(0,k-3),k):
                        a,b=route[edge_index:edge_index+2]
                        yaw=math.degrees(math.atan2(b[0]-a[0],b[1]-a[1]))%360
                        pose={'position':{'x':a[0],'y':checkpoint_view['position']['y'],'z':a[1]},
                              'rotation':{'x':0.,'y':yaw,'z':0.},'cameraHorizon':30.,'isStanding':True}
                        event=self.controller.step(action='TeleportFull',position=pose['position'],rotation=pose['rotation'],horizon=30.,standing=True)
                        if not event.metadata['lastActionSuccess']:continue
                        before=point_key(event.metadata['agent']['position'])
                        event=self.controller.step(action='MoveAhead')
                        if not event.metadata['lastActionSuccess'] and math.dist(before,point_key(event.metadata['agent']['position']))<.02:
                            blocked=(a,b);break
                    if blocked is None:raise InvalidEpisode('No native collision on formerly walkable route')
                    self._teleport(checkpoint_view);self.walk_native(goal)
                    if not objects(self.controller)[target].get('visible'):raise InvalidEpisode('Goal occluded after detour')
                    new_length=len(self.route(start,point_key(goal['position'])))-1
                    self.target=target;self.goal_view=deepcopy(goal);self.checkpoint_view=checkpoint_view
                    self.checkpoint=deepcopy(checkpoint_view['position']);self.checkpoint['y']-=.87
                    self.blocked_edge=blocked;self.old_route_length=old_length;self.new_route_length=new_length
                    self.baseline_graph=baseline;self.changed_graph=deepcopy(self.graph)
                    self.generation_debug.append({'accepted':True,'native_collision_edge':blocked,
                                                  'old_route_length':old_length,'new_route_length':new_length})
                    return
                except (InvalidEpisode,InvalidAction) as error:
                    errors.append(str(error));self.generation_debug.append({'accepted':False,'error':str(error),'obstacle':body['objectType']})
        raise InvalidEpisode('No physically verified replanning task: '+'; '.join(errors[-5:]))

    def after_action(self):
        if self.done:return
        if not self.changed and point_key(self.controller.last_event.metadata['agent']['position'])==point_key(self.checkpoint):
            self.place_barrier();self.refresh_navigation()
            if not set(self.changed_graph)<=set(self.positions):raise InvalidEpisode('Intervention did not reproduce validated reachable cells')
            self.graph=deepcopy(self.changed_graph);self.changed=True;self.phase='after_change';self.progress=1
            self.events.append({'type':'physical_intervention','step':self.steps})
            self.feedback='Checkpoint reached. The room changed; inspect and replan.'
            return {'phase_transition':'before_change->after_change'}

    def task_action(self,action):
        if action['action']!='MARK':return super().task_action(action)
        obj=self.object_for(action['object_id'])
        if not self.changed or obj['objectId']!=self.target:raise InvalidAction('Reach the checkpoint, then mark the requested goal')
        self.progress=2;self.events.append({'type':'goal_marked','step':self.steps})

    def evaluate(self):
        obj=objects(self.controller).get(getattr(self,'target',None),{})
        near=bool(obj.get('visible') and self.planar_distance(obj)<=1.6)
        return {'success':bool(self.changed and self.progress==2 and near),
                'checkpoint_reached':self.changed,'goal_marked':self.progress==2,'at_goal':near,
                'native_blocked_edge_verified':True,'baseline_route_steps':self.old_route_length,
                'detour_route_steps':self.new_route_length,'environment_actions':self.steps}

    def oracle_action(self):
        if not self.changed:return self.nav_action(self.checkpoint_view) or {'action':'WAIT'}
        if self.progress==1:return self.nav_action(self.goal_view) or {'action':'MARK','object_id':self.sid(self.target)}
        return {'action':'FINISH'}

    def render(self):
        rgb=super().render()
        if not hasattr(self,'checkpoint') or getattr(self,'changed',False):return rgb
        point=project(self.checkpoint,self.controller.last_event.metadata,self.size)
        if point is not None:
            x,y,z=point;depth=self.controller.last_event.depth_frame
            if depth is not None and float(depth[y,x])+.12>=z:
                image=Image.fromarray(rgb);draw=ImageDraw.Draw(image)
                draw.ellipse((x-6,y-4,x+6,y+4),fill=(10,80,90),outline=(80,240,240));draw.text((x-3,y-5),'C',fill='white')
                return np.asarray(image).copy()
        return rgb

    def truth(self):
        return {'target':self.sid(self.target),'checkpoint':deepcopy(self.checkpoint),
                'intervention':deepcopy(self.change_plan),'blocked_edge':deepcopy(self.blocked_edge),
                'baseline_route_steps':self.old_route_length,'detour_route_steps':self.new_route_length,
                'goal_view_witness':deepcopy(self.goal_view),'changed':self.changed}
