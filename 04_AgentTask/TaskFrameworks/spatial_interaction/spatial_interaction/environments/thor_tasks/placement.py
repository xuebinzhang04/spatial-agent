"""Observed surface slots and native hand placement; no relation-solving action."""
from copy import deepcopy
import math
import numpy as np
from PIL import Image,ImageDraw
from .base import ThorTaskEnv
from ..dynamic_memory.interventions import objects,InvalidEpisode
from ...core.actions import InvalidAction


def project(point,metadata,size):
    camera=metadata['cameraPosition'];agent=metadata['agent']
    delta=np.array([point[a]-camera[a] for a in 'xyz'])
    yaw=math.radians(agent['rotation']['y']);pitch=math.radians(agent['cameraHorizon'])
    right=np.array([math.cos(yaw),0.,-math.sin(yaw)])
    forward=np.array([math.sin(yaw)*math.cos(pitch),-math.sin(pitch),math.cos(yaw)*math.cos(pitch)])
    up=np.cross(forward,right);z=float(delta@forward)
    if z<=.05:return None
    focal=size/(2*math.tan(math.radians(metadata.get('fov',90))/2))
    x=int(round(size/2+focal*float(delta@right)/z));y=int(round(size/2-focal*float(delta@up)/z))
    return (x,y,z) if 6<=x<size-6 and 6<=y<size-26 else None


class SurfaceTaskEnv(ThorTaskEnv):
    needs_depth=True

    def visible_slots(self):
        found={};event=self.controller.last_event
        for name,point in getattr(self,'slots',{}).items():
            location=project(point,event.metadata,self.size)
            if location is None:continue
            x,y,z=location;depth=event.depth_frame
            if depth is not None and float(depth[y,x])+.12>=z:found[name]=location
        return found

    def render(self):
        rgb=super().render()
        if not hasattr(self,'slots'):return rgb
        image=Image.fromarray(rgb);draw=ImageDraw.Draw(image)
        for name,(x,y,z) in self.visible_slots().items():
            draw.ellipse((x-5,y-5,x+5,y+5),fill=(15,75,90),outline=(80,240,240))
            draw.text((x-5,y-5),name.split('_')[1],fill='white')
        return np.asarray(image).copy()

    def task_action(self,action):
        if action['action']!='PLACE':return super().task_action(action)
        slot=action['slot_id']
        if slot not in self.slots or slot not in self.visible_slots():raise InvalidAction('Placement slot must be currently visible')
        # The task surface is environment-owned, not a policy reference: check it directly
        # (the public reference path rejects private identifiers).
        if not objects(self.controller).get(self.surface,{}).get('receptacle'):
            raise InvalidAction('Task surface is not an available native receptacle')
        agent=self.controller.last_event.metadata['agent']['position'];point=self.slots[slot]
        if math.hypot(point['x']-agent['x'],point['z']-agent['z'])>1.6:raise InvalidAction('Placement slot is out of reach')
        inventory=self.controller.last_event.metadata.get('inventoryObjects',[])
        if len(inventory)!=1:raise InvalidAction('Pick up one object before placement')
        point=deepcopy(self.slots[slot]);point['y']+=.1
        self.native('PlaceObjectAtPoint',objectId=inventory[0]['objectId'],position=point,
                    rotation=self.initial_state[inventory[0]['objectId']]['rotation'],forceKinematic=True)

    def slot_view(self,slot):
        old=deepcopy(self.controller.last_event.metadata['agent']);point=self.slots[slot]
        ranked=sorted(self.positions.values(),key=lambda p:abs(math.hypot(p['x']-point['x'],p['z']-point['z'])-.9))
        try:
            for position in ranked[:16]:
                if math.hypot(position['x']-point['x'],position['z']-point['z'])>1.45:continue
                yaw=round(math.degrees(math.atan2(point['x']-position['x'],point['z']-position['z']))/90)*90%360
                for horizon in (30.,60.,0.):
                    event=self.controller.step(action='TeleportFull',position=position,rotation={'x':0.,'y':yaw,'z':0.},horizon=horizon,standing=True)
                    if event.metadata['lastActionSuccess'] and slot in self.visible_slots():return deepcopy(event.metadata['agent'])
            raise InvalidEpisode('No verified visible placement slot')
        finally:self._teleport(old)
