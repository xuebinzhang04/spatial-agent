"""Fixed task fixtures and evidence collection over the existing native API.

No route simulation, order settlement or food physics is reimplemented here.
"""
import ast
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

from cases import ROOT,public_observation,verify


class NativeCase:
    def __init__(self,case,run):
        if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Native execution requires a Slurm allocation')
        if case['family']=='visual_handoff':
            raise RuntimeError('Visual handoff still needs UE visibility and primitive-position synchronization admission')
        sys.path.insert(0,str(ROOT/'04_AgentTask/Demos/spatial_agent'))
        from delivery import isolated_checkout,Harness
        checkout=isolated_checkout(run,case['map'])
        os.chdir(checkout)
        sys.path.insert(0,str(checkout));sys.path.insert(0,str(checkout/'simworld'))
        os.environ['QT_QPA_PLATFORM']='offscreen';os.environ['DELIVERYBENCH_MULTI_AGENT']='0'
        from vlm_delivery.gym_like_interface.gym_like_interface import DeliveryBenchGymEnv
        self.case,self.run=case,run
        self.env=DeliveryBenchGymEnv(base_dir=str(checkout),map_name=case['map'],
            max_steps=case['budget']['actions']+4,action_timeout_s=600,enable_viewer=True,time_scale=1.)
        self.trace=[];self.commands=[];self.done=False
        self.env.bootstrap_qt()
        try:
            self.env.reset(seed=case['seed'])
            self._install_fixture()
        except BaseException:
            self.env.close();raise
        self.harness=Harness(self.env,run,SimpleNamespace(map=case['map'],seed=case['seed']))
        self.origin=self.env.clock.now_sim()
        self.wall_start=time.monotonic()
        self.outcome=None

    def _install_fixture(self):
        from vlm_delivery.entities.order import OrderManager

        class FixedOrderPool(OrderManager):
            def fill_pool(self,*args,**kwargs):
                # Native completion still runs; the case deliberately has no new orders.
                return None

        manager=FixedOrderPool(capacity=len(self.case['private']['orders']),clock=self.env.clock,seed=self.case['seed'])
        self.orders=[]
        for specification in self.case['private']['orders']:
            self.orders.append(self._make_order(specification))
        manager._orders=self.orders.copy();manager._id_counter=max(o.id for o in self.orders)
        self.env.om=manager
        dm=self.env.dms[0];dm.set_order_manager(manager)
        dm.active_orders.clear();dm.carrying.clear()
        dm.energy_pct=self.case['initial']['energy_pct']
        if dm.e_scooter is None:raise RuntimeError('Native fixture has no scooter')
        dm.e_scooter.battery_pct=self.case['initial']['battery_pct']
        dm.x,dm.y=self.case['initial']['position_cm']
        self.env.ue.teleport_xy(dm.agent_id,dm.x,dm.y)
        # Native MapObserver owns UE position polling and movement resource costs.
        # Seed its baseline so the initial fixture teleport is not charged as travel.
        self.env._viewer.set_agent_xy(dm.agent_id,dm.x,dm.y)
        self.env._viewer.attach_order_manager(manager)
        name=self.env.ue.get_delivery_man_name(dm.agent_id)
        self.env.ue.set_orientation_safe(name,[0.,float(self.case['initial']['yaw_deg']),0.])
        dm._viewer_agent_id=1
        actual=self.env.ue.get_position_and_direction(dm.agent_id).get(str(dm.agent_id))
        if actual is None or math.dist([actual[0].x,actual[0].y],[dm.x,dm.y])>100:
            raise RuntimeError('UE actor did not reproduce the case initial position')

    def _make_order(self,specification):
        from vlm_delivery.entities.order import Order,FoodItem
        from vlm_delivery.base.types import Vector
        food=dict(specification['food']);food['temp_c']=float(food.get('serving_temp_c',25))
        order=Order(city_map=self.env.map,
            pickup_address=Vector(*specification['pickup']['center_cm']),
            delivery_address=Vector(*specification['dropoff']['center_cm']),items=[FoodItem(**food)])
        order.id=specification['order_id'];order.rng=random.Random(self.case['seed']+order.id)
        order.prep_longest_s=specification['prep_time_s'];order.compute()
        order.time_limit_s=specification['deadline_s']
        order.special_note=specification['customer_note']
        order.allowed_delivery_methods=[specification['required_method']]
        for field,node in (('pickup',order.pickup_node),('dropoff',order.dropoff_node)):
            if math.dist([node.position.x,node.position.y],specification[field]['position_cm'])>1:
                raise RuntimeError('Native order binding differs from the generated map recipe')
        return order

    def _capture(self):
        dm=self.env.dms[0]
        pose=self.env.ue.get_position_and_direction(dm.agent_id).get(str(dm.agent_id))
        if pose is None:raise RuntimeError('No authoritative UE position returned')
        state={'source':'native','sim_time_s':float(self.env.clock.now_sim()-self.origin),
            'position_cm':[float(dm.x),float(dm.y)],'ue_position_cm':[float(pose[0].x),float(pose[0].y)],
            'yaw_deg':float(pose[1]),'battery_pct':float(dm.e_scooter.battery_pct),
            'energy_pct':float(dm.energy_pct),'earnings':float(dm.earnings_total),'transport':dm.mode.value,
            'feedback':str(dm.vlm_errors),'orders':{},'bag':{},'bag_order_ids':[],
            'pending_food_orders':[int(k) for k,v in dm._pending_food_by_order.items() if v]}
        for o in self.orders:
            state['orders'][str(o.id)]={'is_accepted':bool(o.is_accepted),'has_picked_up':bool(o.has_picked_up),
                'has_delivered':bool(o.has_delivered),'delivery_method':getattr(o,'delivery_method',None),
                'remaining_prep_s':float(o.remaining_prep_s()),'elapsed_active_s':float(o.sim_elapsed_active_s),
                'foods':[{'name':it.name,'temperature_c':float(it.temp_c),'damage':int(it.damage_level),
                          'odor_contamination':float(it.odor_contamination)} for it in o.items]}
        if dm.insulated_bag:
            state['bag']={label:[{'order_id':getattr(it,'order_id',None),'name':it.name,'temperature_c':float(it.temp_c)} for it in comp]
                          for label,comp in zip(dm.insulated_bag.labels,dm.insulated_bag._comps)}
            state['bag_order_ids']=sorted({int(it.order_id) for comp in dm.insulated_bag._comps for it in comp if hasattr(it,'order_id')})
        if all(o.has_delivered for o in self.orders):state['phase']='complete'
        elif any(state['pending_food_orders']):state['phase']='pack'
        elif not all(o.is_accepted for o in self.orders):state['phase']='dispatch'
        else:state['phase']='pickup_and_delivery'
        state['steps_remaining']=max(0,self.case['budget']['actions']-len(self.commands))
        state['seconds_remaining']=max(0,self.case['budget']['sim_seconds']-state['sim_time_s'])
        return state

    def record(self,command=None,transition=None):
        box=self.env._invoker.call(self._capture)
        if not box.get('ok'):raise RuntimeError(box.get('exc'))
        state=box['result'];state['command']=command or ''
        if transition is not None:state['native_transition']=transition
        self.trace.append(state)
        with (self.run/'native-trace.jsonl').open('a') as file:file.write(json.dumps(state)+'\n')
        observation=public_observation(self.case,state)
        (self.run/'public-observation.json').write_text(json.dumps(observation,indent=2)+'\n')
        snapshot=self.harness.observe()
        snapshot['prompt']=json.dumps(observation)
        snapshot['case_id']=self.case['case_id'];snapshot['phase']=state['phase']
        self.harness.save(snapshot,command)
        return state

    def step(self,command):
        if self.done:raise RuntimeError('Case ended')
        if len(self.commands)>=self.case['budget']['actions']:raise RuntimeError('Action budget exhausted')
        if time.monotonic()-self.wall_start>self.case['budget']['wall_seconds']:raise TimeoutError('Case wall-clock budget exhausted')
        call=ast.parse(command,mode='eval').body
        if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name):raise ValueError('Expected one native API call')
        allowed={'VIEW_ORDERS','ACCEPT_ORDER','MOVE','WAIT','PICKUP','PLACE_FOOD_IN_BAG','DROP_OFF'}
        if self.case['family']=='recharge_delivery':allowed|={'CHARGE','SWITCH'}
        if call.func.id not in allowed:raise ValueError('Action is outside the declared case API')
        if call.func.id=='MOVE' and self.case['public']['constraints'].get('accept_all_before_travel'):
            state=self.trace[-1]
            if not all(o['is_accepted'] for o in state['orders'].values()):raise ValueError('Accept every assigned order before travel')
        self.commands.append(command)
        _,reward,terminated,truncated,info=self.env.step(command)
        state=self.record(command,{'gym_reward':reward,'terminated':terminated,'truncated':truncated,'info':info})
        if info.get('error'):raise RuntimeError('Native API failure: '+str(info['error']))
        if truncated or terminated and not all(o['has_delivered'] for o in state['orders'].values()):
            self.done=True;raise RuntimeError('Native runtime ended before case completion')
        if state['sim_time_s']>self.case['budget']['sim_seconds']:raise TimeoutError('Case simulation budget exhausted')
        return state

    def reference(self):
        """Privileged mechanical admission witness; no VLM is called or corrected."""
        self.record()
        for order in self.case['private']['orders']:self.step(f"ACCEPT_ORDER({order['order_id']})")
        charge=self.case['private']['charge']
        if charge:
            x,y=charge['station']['position_cm'];self.step(f'MOVE({x},{y})')
            self.step(f"CHARGE(target_pct={charge['target_pct']})");self.step('WAIT(until="charge_done")')
            self.step('SWITCH(to="e-scooter")')
        orders={o['order_id']:o for o in self.case['private']['orders']}
        for event in self.case['private']['serial_witness']['events']:
            order=orders[event['order_id']];oid=order['order_id']
            if event['operation']=='pickup':
                x,y=order['pickup']['position_cm'];self.step(f'MOVE({x},{y})')
                remaining=self.trace[-1]['orders'][str(oid)]['remaining_prep_s']
                if remaining>0:self.step(f'WAIT(seconds={math.ceil(remaining)+1})')
                self.step(f'PICKUP(orders=[{oid}])')
                if not self.trace[-1]['orders'][str(oid)]['has_picked_up']:raise RuntimeError('Native pickup did not succeed')
                self.step(f'PLACE_FOOD_IN_BAG("order {oid}: 1 -> A")')
            else:
                x,y=order['dropoff']['position_cm'];self.step(f'MOVE({x},{y})')
                self.step(f"DROP_OFF({oid}, method={order['required_method']!r})")
        self.done=True;self.outcome=verify(self.case,self.trace)
        if self.outcome['success']:
            self.outcome['negative_controls']=self._check_negative_controls()
        return self.outcome

    def policy(self,model,url,max_calls):
        """Execute actual model decisions using public state and the three RGB views."""
        import re
        from urllib.request import Request,urlopen
        from cases import public_observation
        self.record()
        for index in range(max_calls):
            if all(o['has_delivered'] for o in self.trace[-1]['orders'].values()):break
            snapshot=self.harness.observe()
            instruction=('Complete the assigned fixed courier case. Use only the public task, state, views and your action history. '
                'Return exactly one JSON object {"command":"ONE_NATIVE_API_CALL()"}. Do not return a plan instead of an action. '
                'Wait for preparation when needed. Check feedback and order state after each action. Use the declared API and units.\n')
            public=public_observation(self.case,self.trace[-1])
            text=instruction+json.dumps({'observation':public,'your_actions':self.commands[-12:]})
            content=[{'type':'text','text':text}]
            content += [{'type':'image_url','image_url':{'url':value}} for value in snapshot['images'] if value]
            payload={'model':model,'temperature':0,'max_tokens':1024,'chat_template_kwargs':{'enable_thinking':False},
                     'messages':[{'role':'user','content':content}]}
            record={'call':index+1,'input_step':len(self.commands),'model':model,'request':payload}
            started=time.monotonic()
            try:
                request=Request(url.rstrip('/')+'/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                with urlopen(request,timeout=180) as result:response=json.load(result)
                record.update(response=response,latency_seconds=time.monotonic()-started)
                if response.get('model')!=model:raise ValueError('Inference response model identity mismatch')
                output=response['choices'][0]['message']['content'];record['model_output']=output
                clean=re.sub(r'^```(?:json)?\s*|\s*```$','',output.strip()).strip()
                action=json.loads(clean)
                if not isinstance(action,dict) or set(action)!={'command'} or not isinstance(action['command'],str):
                    raise ValueError('Expected exactly one command string')
                record['command']=action['command']
            except Exception as error:
                record['error']=str(error)
                raise
            finally:
                with (self.run/'model-calls.jsonl').open('a') as file:file.write(json.dumps(record)+'\n')
            self.step(action['command'])
        self.done=True
        return verify(self.case,self.trace)

    def _check_negative_controls(self):
        """A separate ready-order fixture avoids confounding distance with cooking."""
        from copy import deepcopy
        last=self.case['private']['serial_witness']['events'][-1]['order_id']
        specification=deepcopy(next(o for o in self.case['private']['orders'] if o['order_id']==last))
        specification['order_id']=10000+last
        specification['prep_time_s']=0.;specification['food']['prep_time_s']=0
        probe=self._make_order(specification)
        def install():
            self.env.om._orders.append(probe)
            self.env.om._id_counter=max(self.env.om._id_counter,probe.id)
        box=self.env._invoker.call(install)
        if not box.get('ok'):raise RuntimeError(box.get('exc'))
        self.env.step(f'ACCEPT_ORDER({probe.id})')
        box=self.env._invoker.call(lambda:self.env.ue.get_position_and_direction(self.env.dms[0].agent_id))
        if not box.get('ok'):raise RuntimeError(box.get('exc'))
        pose=box['result'].get(str(self.env.dms[0].agent_id))
        if pose is None:raise RuntimeError('No UE pose for negative controls')
        xy=[float(pose[0].x),float(pose[0].y)]
        if not probe.is_accepted or not probe.is_ready_for_pickup():raise RuntimeError('Control order is not accepted and ready')
        if math.dist(xy,specification['pickup']['position_cm'])<=600:raise RuntimeError('Control pickup is not remote')
        if math.dist(xy,specification['dropoff']['position_cm'])>500:raise RuntimeError('Control delivery is not at the destination')
        _,_,_,truncated,info=self.env.step(f'PICKUP(orders=[{probe.id}])')
        if truncated or info.get('error') or probe.has_picked_up:raise RuntimeError('Ready remote-pickup control failed')
        _,_,_,truncated,info=self.env.step(f"DROP_OFF({probe.id}, method={specification['required_method']!r})")
        if truncated or info.get('error') or probe.has_delivered:raise RuntimeError('Delivery-without-pickup control failed')
        result={'source':'native','separate_control_fixture':True,'accepted_and_ready':True,
                'ue_position_cm':xy,'ready_remote_pickup_rejected':True,
                'delivery_without_pickup_at_destination_rejected':True}
        (self.run/'negative-controls.json').write_text(json.dumps(result,indent=2)+'\n')
        return result

    def close(self):self.env.close()
