"""Native iTHOR task base. Navigation graphs and oracle plans stay privileged."""
from collections import deque
from copy import deepcopy
import math
import numpy as np
from PIL import Image, ImageDraw
from ..dynamic_memory.env import DynamicMemoryEnv
from ..dynamic_memory.backend import create_controller
from ..dynamic_memory.interventions import checked, objects, xyz, InvalidEpisode
from ...core.actions import InvalidAction

NAV=('MoveAhead','MoveBack','RotateLeft','RotateRight','LookUp','LookDown')


def point_key(p):
    return (round(p['x'],3),round(p['z'],3))


class ThorTaskEnv(DynamicMemoryEnv):
    extra_actions=()

    @property
    def available_actions(self):return NAV+self.extra_actions+('FINISH',)

    def prepare(self,seed=None,options=None):
        self._start(seed,options)
        self.phase='interaction';self.feedback='';self.events=[];self.view_cache={}
        self.size=int(self.config.task_config.get('image_size',160))
        if not 64<=self.size<=512:raise ValueError('image_size must be 64..512')
        if self.controller is None:self.controller=create_controller(self.config.scene_id,self.size,self.runtime_dir,depth=getattr(self,'needs_depth',False))
        self._reset_scene()
        current=objects(self.controller)
        self.id_map={oid:f'object_{i+1:03d}' for i,oid in enumerate(sorted(current))}
        self.reverse_ids={v:k for k,v in self.id_map.items()}
        self.refresh_navigation()
        order=sorted(self.positions);mode=self.config.task_config.get('start_pose','fixed')
        if mode=='fixed':first,yaw=self.positions[order[0]],0.
        elif mode=='seeded':
            # Separate stream: the task's own sampling (self.rng) is unchanged by this choice.
            picker=np.random.default_rng([self.config.seed,11])
            first,yaw=self.positions[order[int(picker.integers(len(order)))]],float(90*picker.integers(4))
        else:raise ValueError("start_pose must be 'fixed' or 'seeded'")
        self._teleport({'position':first,'rotation':{'x':0.,'y':yaw,'z':0.},'cameraHorizon':30.,'isStanding':True})
        self.initial_pose=deepcopy(self.controller.last_event.metadata['agent'])
        self.initial_state=objects(self.controller)
        self.touched=set();self.progress=0

    def refresh_navigation(self):
        event=checked(self.controller,'GetReachablePositions')
        self.positions={point_key(p):p for p in event.metadata['actionReturn']}
        self.graph={p:[q for q in ((round(p[0]+.25,3),p[1]),(round(p[0]-.25,3),p[1]),
                                 (p[0],round(p[1]+.25,3)),(p[0],round(p[1]-.25,3))) if q in self.positions]
                    for p in self.positions}
        if len(self.positions)<4:raise InvalidEpisode('Insufficient native reachable positions')

    def route(self,start,end):
        start=min(self.positions,key=lambda p:math.dist(p,start));end=min(self.positions,key=lambda p:math.dist(p,end))
        queue=deque([start]);previous={start:None}
        while queue:
            p=queue.popleft()
            if p==end:break
            for q in self.graph[p]:
                if q not in previous:previous[q]=p;queue.append(q)
        if end not in previous:raise InvalidEpisode('No collision-connected route to task goal')
        result=[end]
        while result[-1]!=start:result.append(previous[result[-1]])
        return list(reversed(result))

    def sid(self,oid):return self.id_map[oid]

    def object_for(self,stable,capability=None,near=True):
        oid=self.resolve_reference(stable);obj=objects(self.controller).get(oid)
        if not obj:raise InvalidAction('Unknown task object')
        if near and (not obj.get('visible') or self.planar_distance(obj)>1.6):
            raise InvalidAction('Object must be visible and within 1.6 metres')
        if capability and not obj.get(capability):raise InvalidAction('Object lacks required native capability')
        return obj

    def planar_distance(self,obj):
        p=self.controller.last_event.metadata['agent']['position']
        return math.hypot(p['x']-obj['position']['x'],p['z']-obj['position']['z'])

    def view_of(self,oid,max_candidates=12):
        if oid in self.view_cache:return deepcopy(self.view_cache[oid])
        pose=deepcopy(self.controller.last_event.metadata['agent']);obj=objects(self.controller)[oid]
        ranked=sorted(self.positions.values(),key=lambda p:abs(math.hypot(p['x']-obj['position']['x'],p['z']-obj['position']['z'])-1.))
        try:
            for position in ranked[:max_candidates]:
                if math.hypot(position['x']-obj['position']['x'],position['z']-obj['position']['z'])>1.55:continue
                yaw=round(math.degrees(math.atan2(obj['position']['x']-position['x'],obj['position']['z']-position['z']))/90)*90%360
                for horizon in (30.,0.,60.):
                    event=self.controller.step(action='TeleportFull',position=position,rotation={'x':0.,'y':yaw,'z':0.},horizon=horizon,standing=True)
                    if event.metadata['lastActionSuccess']:
                        actual=objects(self.controller).get(oid,{})
                        if actual.get('visible') and oid in event.instance_detections2D:
                            found=deepcopy(event.metadata['agent']);self.view_cache[oid]=found
                            return deepcopy(found)
            raise InvalidEpisode('No verified nearby cardinal view for '+obj['objectType'])
        finally:self._teleport(pose)

    def nav_action(self,pose):
        agent=self.controller.last_event.metadata['agent']
        path=self.route(point_key(agent['position']),point_key(pose['position']))
        wanted=(round(math.degrees(math.atan2(path[1][0]-path[0][0],path[1][1]-path[0][1]))/90)*90%360
                if len(path)>1 else pose['rotation']['y']%360)
        delta=(wanted-agent['rotation']['y'])%360
        if min(delta,360-delta)>1.:
            return {'action':'RotateRight' if delta<=180 else 'RotateLeft'}
        if len(path)>1:return {'action':'MoveAhead'}
        horizon=agent['cameraHorizon']
        if horizon>180:horizon-=360
        desired=pose.get('cameraHorizon',30.)
        if abs(horizon-desired)>1:return {'action':'LookDown' if horizon<desired else 'LookUp'}
        return None

    def native(self,action,**kwargs):
        event=self.controller.step(action=action,**kwargs)
        if not event.metadata['lastActionSuccess']:raise InvalidAction('Native action rejected: collision, state, visibility or reach')
        return event

    def step(self,action):
        obs,reward,terminated,truncated,info=super().step(action)
        if not info['action_valid']:
            self.feedback='Action failed: '+info['error']
            obs=self._observation()
        return obs,reward,terminated,truncated,info

    def _execute(self,action):
        name=action['action'];self.feedback=''
        if name=='FINISH':
            self.metrics=self.evaluate()
            return float(self.metrics['success']),{'metrics':self.metrics}
        if name in NAV:
            kwargs={'degrees':90} if name.startswith('Rotate') else {}
            self.native(name,**kwargs)
        elif name in ('OpenObject','CloseObject','PickupObject'):
            obj=self.object_for(action['object_id'],'pickupable' if name=='PickupObject' else 'openable')
            self.native(name,objectId=obj['objectId']);self.touched.add(obj['objectId'])
            if name in ('OpenObject','CloseObject'):self.refresh_navigation()
        elif name=='PutObject':
            obj=self.object_for(action['object_id'],'receptacle')
            if obj.get('openable') and not obj.get('isOpen'):raise InvalidAction('Open the destination container first')
            if not self.controller.last_event.metadata.get('inventoryObjects'):raise InvalidAction('The hand is empty')
            self.native('PutObject',objectId=obj['objectId'],placeStationary=True)
        elif name=='WAIT':self.native('Pass')
        else:self.task_action(action)
        return None,{}

    def _after_step(self):return self.after_action()

    def after_action(self):pass

    def task_action(self,action):raise InvalidAction('Action not implemented by this task')

    def evaluate(self):raise NotImplementedError

    def timeout_metrics(self):
        metrics=self.evaluate();metrics['success']=False;return metrics

    def reset_info(self):
        self.initial_state=objects(self.controller)
        return self._observation(),{'valid':True,'simulator':'AI2-THOR 5.0.0','task_type':self.config.task_type}

    def render(self):
        rgb=super().render()
        if self.feedback:
            image=Image.fromarray(rgb);draw=ImageDraw.Draw(image)
            draw.rectangle((0,self.size-24,self.size,self.size),fill=(15,25,32))
            draw.text((3,self.size-20),self.feedback[:65],fill='white');rgb=np.asarray(image).copy()
        return rgb

    def _observation(self):
        obs=super()._observation();obs['feedback']=self.feedback
        return obs

    def privileged_state(self):
        return {'initial_state':deepcopy(self.initial_state),'current_state':objects(self.controller),
                'id_map':deepcopy(self.id_map),'agent':deepcopy(self.controller.last_event.metadata['agent']),
                'events':deepcopy(self.events),'progress':self.progress,'task_state':deepcopy(self.task_state()),'ground_truth':self.truth()}

    def task_state(self):return {}

    def truth(self):return {}
