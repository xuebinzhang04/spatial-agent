"""Reproducible DeliveryBench recipes using the upstream city graph and menus.

Graph witnesses are construction evidence, not proof of UE traversability.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[3]
UPSTREAM=ROOT/'04_AgentTask/SpatialAgentBench/DeliveryBench'
sys.path.insert(0,str(UPSTREAM))
sys.path.insert(0,str(UPSTREAM/'simworld'))
from vlm_delivery.map.map import Map

FAMILIES=('pickup_delivery','batch_schedule','recharge_delivery','visual_handoff')
METHODS={'leave_at_door':'Leave the food at the door without knocking.',
         'knock':'Knock when you arrive.', 'call':'Call when you arrive.',
         'hand_to_customer':'Hand the food directly to the customer outside the building.'}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class City:
    def __init__(self,name):
        if name not in [p.name for p in (UPSTREAM/'maps').iterdir() if p.is_dir()]:
            raise ValueError('Unknown installed map')
        self.name=name
        self.paths=[UPSTREAM/'maps'/name/n for n in ('roads.json','progen_world_enriched.json')]
        self.paths += [UPSTREAM/'vlm_delivery/input'/n for n in ('game_mechanics_config.json','food.json')]
        self.mechanics=json.loads(self.paths[2].read_text())
        self.menu=json.loads(self.paths[3].read_text())['items']
        self.map=Map(self.mechanics['map'])
        self.map.import_roads(str(self.paths[0]));self.map.import_pois(str(self.paths[1]))
        self.by_type={kind:[m for m in self.map.poi_meta if self.map._t_of(m['node'])==kind]
                      for kind in ('restaurant','building','charging_station')}
        self.distances={}

    @staticmethod
    def node(meta):return meta['door_node'] or meta['node']

    @classmethod
    def xy(cls,meta):
        p=cls.node(meta).position
        return [float(p.x),float(p.y)]

    def route(self,a,b):
        key=(self.node(a),self.node(b))
        if key not in self.distances:
            path,distance=self.map.shortest_path_nodes(*key)
            if not path or not math.isfinite(distance):raise ValueError('Disconnected native graph endpoints')
            self.distances[key]={'distance_cm':round(distance,3),
                'endpoints_cm':[self.xy(a),self.xy(b)]}
        return self.distances[key]

    def describe(self,meta):
        return {'name':getattr(meta['node'],'display_name',self.map._t_of(meta['node'])),'position_cm':self.xy(meta),
                'center_cm':list(meta['center']),'road':meta['road_name']}


def simulate_serial(city,start,orders,sequence,speed_cm_s=600.,initial_time=0.):
    """One feasible serial pickup/delivery order, never an optimality claim."""
    here=start;elapsed=initial_time;events=[];completion={};distance=0.
    for index in sequence:
        order=orders[index]
        for operation,target in (('pickup',order['_pickup']),('deliver',order['_dropoff'])):
            leg=city.route(here,target);distance+=leg['distance_cm'];elapsed+=leg['distance_cm']/speed_cm_s
            if operation=='pickup':elapsed=max(elapsed,float(order['prep_time_s']))
            elapsed+=3. # Explicit action-handling allowance; must be calibrated in UE.
            events.append({'operation':operation,'order_id':order['order_id'],
                           'arrival_s':elapsed,'graph_leg':leg})
            if operation=='deliver':completion[order['order_id']]=elapsed
            here=target
    return {'events':events,'completion_s':completion,'duration_s':elapsed,'distance_cm':distance}


def generate(city,family,seed,level):
    if family not in FAMILIES or level not in (1,2,3):raise ValueError('Unknown family or level')
    rng=random.Random(seed)
    count=min(3,level+1) if family=='batch_schedule' else 1
    speed=float(city.mechanics['avg_speed_cm_s']['e-scooter'])
    rate=float(city.mechanics['scooter_batt_decay_pct_per_m'])
    ambient=[f for f in city.menu if f['name'] in ('Sandwich','Donut','Cake','BlueCheese')]
    for attempt in range(150):
        start=rng.choice(city.by_type['building']);orders=[]
        for index in range(count):
            pickup=rng.choice(city.by_type['restaurant']);dropoff=rng.choice(city.by_type['building'])
            food=rng.choice(ambient)
            method=rng.choice(('leave_at_door','knock','call')) if family=='batch_schedule' else 'leave_at_door'
            if family=='visual_handoff':method='hand_to_customer'
            orders.append({'order_id':index+1,'pickup':city.describe(pickup),'dropoff':city.describe(dropoff),
                'food':dict(food),'prep_time_s':float(food['prep_time_s']),
                'required_method':method,'customer_note':METHODS[method],
                '_pickup':pickup,'_dropoff':dropoff})
        if any(math.dist(city.xy(start),city.xy(o['_pickup']))<=1200 or
               math.dist(city.xy(o['_pickup']),city.xy(o['_dropoff']))<=1200 for o in orders):continue
        try:
            serial=[simulate_serial(city,start,orders,p,speed) for p in itertools.permutations(range(count))]
        except ValueError:continue
        best=min(serial,key=lambda w:w['duration_s']);worst=max(serial,key=lambda w:w['duration_s'])
        trip_m=best['distance_cm']/100
        if not (50*level<=trip_m<=600+450*level):continue
        public_constraints={'complete_all_orders':True,'on_time':True,'use_required_delivery_method':True,
                            'pack_before_delivery':True,'scooter_only_except_charging':True,
                            'pickup_radius_cm':600.,'delivery_radius_cm':500.,'position_sync_tolerance_cm':200.}
        control={'remote_pickup_distance_cm':min(math.dist(city.xy(start),city.xy(o['_pickup'])) for o in orders),
                 'initial_delivery_without_pickup_must_fail':True}
        charge=None;initial_battery=100.
        if family=='batch_schedule':
            gaps={k:worst['completion_s'][k]-t for k,t in best['completion_s'].items()}
            critical=max(gaps,key=gaps.get)
            if gaps[critical]<90:continue
            deadlines={o['order_id']:best['completion_s'][o['order_id']]+max(60,.25*best['duration_s']) for o in orders}
            deadlines[critical]=best['completion_s'][critical]+gaps[critical]*.5
            control.update(reversed_serial_completion_s=worst['completion_s'],
                critical_order=critical,alternative_serial_misses_deadline=True)
            public_constraints['accept_all_before_travel']=True
        else:deadlines={1:best['duration_s']+max(120,.5*best['duration_s'])}
        if family=='recharge_delivery':
            options=[]
            for station in city.by_type['charging_station']:
                try:
                    approach=city.route(start,station)['distance_cm']
                    after=simulate_serial(city,station,orders,[0],speed)
                except ValueError:continue
                reach_cost=approach/100*rate;direct_cost=best['distance_cm']/100*rate
                if direct_cost<=reach_cost+4 or direct_cost>=90:continue
                battery=reach_cost+(direct_cost-reach_cost)*(.8,.5,.25)[level-1]
                target=min(95.,after['distance_cm']/100*rate+10.)
                remaining=battery-reach_cost
                charging_s=max(0,target-remaining)/city.mechanics['escooter_defaults']['charge_rate_pct_per_min']*60
                route=simulate_serial(city,station,orders,[0],speed,approach/speed+charging_s+6)
                options.append((route['duration_s'],station,battery,target,route,reach_cost,direct_cost))
            if not options:continue
            _,station,initial_battery,target,best,reach_cost,direct_cost=min(options,key=lambda x:x[0])
            charge={'station':city.describe(station),'target_pct':target,
                    'approach_graph_leg':city.route(start,station)}
            deadlines={1:best['duration_s']+max(120,.3*best['duration_s'])}
            control.update(battery_to_station_pct=reach_cost,battery_without_recharge_pct=direct_cost,
                           initial_battery_pct=initial_battery)
            public_constraints['no_battery_depletion']=True
        for o in orders:o['deadline_s']=deadlines[o['order_id']]
        public_orders=[{k:v for k,v in o.items() if not k.startswith('_')} for o in orders]
        # Native binding centers and exact reference routes are construction data.
        public=[{k:v for k,v in o.items() if k!='food'} for o in public_orders]
        for o in public:
            o['pickup']={k:v for k,v in o['pickup'].items() if k!='center_cm'}
            o['dropoff']={k:v for k,v in o['dropoff'].items() if k!='center_cm'}
        api={
            'VIEW_ORDERS':'VIEW_ORDERS() reads available order information.',
            'ACCEPT_ORDER':'ACCEPT_ORDER(1) or ACCEPT_ORDER([1,2]) starts preparation for the assigned order IDs.',
            'MOVE':'MOVE(x,y) uses centimeters and the native route planner; this is not a low-level navigation action.',
            'WAIT':'WAIT(seconds=30) waits for simulated time to pass.',
            'PICKUP':'PICKUP(orders=[1]) requires proximity and completed preparation.',
            'PLACE_FOOD_IN_BAG':'PLACE_FOOD_IN_BAG("order 1: 1 -> A") puts the first pending item of that order in compartment A. Preserve existing items when reorganizing.',
            'DROP_OFF':'DROP_OFF(1, method="leave_at_door") requires pickup and proximity; obey the required method.'}
        if family=='recharge_delivery':
            api.update(CHARGE='CHARGE(target_pct=80) starts charging near a station.',
                       WAIT='WAIT(until="charge_done") or WAIT(seconds=30).',
                       SWITCH='SWITCH(to="e-scooter") resumes riding after charging.')
        if family=='visual_handoff':
            api.update(TURN_AROUND='TURN_AROUND(angle=30,direction="left") changes heading.',
                       STEP_FORWARD='STEP_FORWARD() requests one native forward step; native/logical position agreement must be admitted first.')
        for order in public:
            source=next(o for o in public_orders if o['order_id']==order['order_id'])
            order['food']={key:source['food'][key] for key in ('name','category','serving_temp_c','motion_sensitive')}
        instruction=('Complete the assigned courier orders. Accept each order, reach its restaurant, wait until preparation '
            'is complete, PICKUP(orders=[...]), place food into the insulated bag, reach the destination and DROP_OFF '
            'using the customer\'s required method. Confirm outcomes from environment feedback; issuing a command is '
            'not proof it succeeded. Coordinates use centimeters in API calls (100 cm = 1 m). '
            'Keep scooter transport except when charging; battery packs, cars and bus shortcuts are excluded. '
            'Deadlines below are measured from case start; all orders must finish before their deadline. '
            'Order IDs are public workflow handles; no object IDs are drawn on first-person images. ')
        if family=='batch_schedule':instruction+='Accept all assigned orders before travel. Plan preparation, pickup and delivery order together. '
        if family=='recharge_delivery':instruction+='Your initial battery cannot finish the direct delivery; find a reachable charging station and restore enough charge. '
        if family=='visual_handoff':instruction+=('After arriving at the building, inspect the first-person view and use TURN_AROUND/STEP_FORWARD '
            'to find the customer. Their exact position is hidden; a door-only delivery fails this task. ')
        private={'orders':public_orders,'serial_witness':best,'shortcut_checks':control,'charge':charge,
                 'native_requirements':['collision-free route replay','Python/UE position agreement','pickup/packing/dropoff state transitions']}
        if family=='visual_handoff':
            private['native_requirements']+=['customer spawned outside arrival view','customer visible from an executable later view',
                                             'primitive motion updates position used by handoff evaluator']
        result={'schema':'delivery-cases/1','family':family,'seed':seed,'level':level,'map':city.name,
            'status':'graph_screened_native_pending' if family!='visual_handoff' else 'native_visual_admission_required',
            'simulator':'simworld_delivery','initial':{'position_cm':city.xy(start),'yaw_deg':rng.choice((0,90,180,270)),
                'battery_pct':initial_battery,'energy_pct':100.,'transport':'e-scooter'},
            'structural_difficulty':{'orders':count,'graph_distance_m':round(best['distance_cm']/100,2),
                'initial_battery_pct':initial_battery,'empirically_calibrated':False},
            'budget':{'actions':50+20*count,'sim_seconds':max(deadlines.values())+60,'wall_seconds':1200},
            'public':{'instruction':instruction,'orders':public,'api':api,'constraints':public_constraints,
                      'charging_stations':[{k:v for k,v in city.describe(m).items() if k!='center_cm'} for m in city.by_type['charging_station']] if family=='recharge_delivery' else [],
                      'observation_views':['Global map','Local map','First-person RGB']},
            'private':private,
            'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in city.paths},
            'limitations':['Graph connectivity does not establish UE reachability.',
                          'Difficulty levels are structural and have not been calibrated with VLM runs.',
                          'Serial witness is not an optimal planning certificate.']}
        result['case_id']=family+'-'+content_hash(result)[:12]
        return result
    raise ValueError(f'No graph-screened {family} case after 150 attempts: {city.name} seed {seed}')


def public_observation(case,state):
    """Allowlist the policy state; never serialize the private case certificate."""
    fields=('sim_time_s','position_cm','yaw_deg','energy_pct','battery_pct','transport','feedback',
            'orders','bag','phase','steps_remaining','seconds_remaining')
    return {'case_id':case['case_id'],'task':case['public'],
            'state':{k:state[k] for k in fields if k in state}}


def verify(case,trace):
    """Verify trusted native snapshots, never agent-supplied completion claims."""
    reasons=[];per_order={}
    if not trace:return {'success':False,'reasons':['No native evidence'],'orders':{}}
    if trace[0].get('source')!='native':reasons.append('Initial snapshot is not native evidence')
    first=trace[0]
    if (first.get('ue_position_cm') is None or math.dist(first['ue_position_cm'],case['initial']['position_cm'])>100 or
            abs(float(first.get('battery_pct',-1000))-case['initial']['battery_pct'])>.5 or
            abs(float(first.get('energy_pct',-1000))-case['initial']['energy_pct'])>.5 or
            first.get('transport')!=case['initial']['transport']):
        reasons.append('Native initial conditions do not match recipe')
    for expected in case['private']['orders']:
        oid=str(expected['order_id']);accepted=picked=delivered=packing=None
        for i,row in enumerate(trace):
            o=row.get('orders',{}).get(oid,{})
            if accepted is None and o.get('is_accepted'):accepted=i
            if picked is None and o.get('has_picked_up'):picked=i
            if packing is None and picked is not None and i>picked and not o.get('has_delivered') and oid in [str(x) for x in row.get('bag_order_ids',[])]:packing=i
            if delivered is None and o.get('has_delivered'):delivered=i
        errors=[]
        if any(x is None for x in (accepted,picked,delivered)):errors.append('Incomplete native order transitions')
        elif not 0<accepted<picked<delivered:errors.append('Invalid accept/pickup/delivery ordering')
        else:
            start=trace[0].get('orders',{}).get(oid,{})
            if any(start.get(k) for k in ('is_accepted','has_picked_up','has_delivered')):errors.append('Initial order already progressed')
            for index,target in ((picked,expected['pickup']['position_cm']),(delivered,expected['dropoff']['position_cm'])):
                if expected['required_method']=='hand_to_customer' and index==delivered:
                    target=trace[index].get('handoff_positions_cm',{}).get(oid)
                pose=trace[index].get('ue_position_cm')
                if pose is None or target is None or math.dist(pose,target)>case['public']['constraints']['pickup_radius_cm' if index==picked else 'delivery_radius_cm']:errors.append('Missing or remote physical interaction evidence')
            actual=trace[delivered]['orders'][oid]
            if actual.get('delivery_method')!=expected['required_method']:errors.append('Wrong delivery method')
            if trace[delivered]['sim_time_s']>expected['deadline_s']:errors.append('Case-relative deadline missed')
            if packing is None or not picked<packing<delivered:errors.append('No native bag-placement evidence')
        per_order[oid]={'success':not errors,'reasons':errors}
        reasons.extend(f'Order {oid}: {e}' for e in errors)
    for row in trace:
        pose=row.get('ue_position_cm');logical=row.get('position_cm')
        if (row.get('source')!='native' or pose is None or logical is None or
                not all(math.isfinite(float(v)) for v in pose+logical) or math.dist(pose,logical)>case['public']['constraints']['position_sync_tolerance_cm']):
            reasons.append('Unverified or unsynchronized native position');break
        if any(not math.isfinite(float(row.get(k,float('nan')))) for k in ('sim_time_s','battery_pct','energy_pct')):
            reasons.append('Missing/nonfinite resource or time evidence');break
    if any(row.get('transport') not in ('e-scooter','walk') for row in trace):
        reasons.append('Unsupported transport shortcut')
    if any(b.get('sim_time_s',-1)<a.get('sim_time_s',0) for a,b in zip(trace,trace[1:])):
        reasons.append('Native clock moved backwards')
    if case['family']=='recharge_delivery':
        if min(float(r.get('battery_pct',0)) for r in trace)<=0:reasons.append('Battery depleted')
        stations=[p['position_cm'] for p in case['public']['charging_stations']]
        if any(row.get('transport')=='walk' and (row.get('ue_position_cm') is None or not any(math.dist(row['ue_position_cm'],p)<=600 for p in stations)) for row in trace):
            reasons.append('Walking used outside charging station')
        charged=False
        for a,b in zip(trace,trace[1:]):
            if float(b.get('battery_pct',0))-float(a.get('battery_pct',0))>1:
                if b.get('ue_position_cm') and any(math.dist(b['ue_position_cm'],p)<=600 for p in stations):charged=True
        if not charged:reasons.append('No actual charging at a station')
    if case['public']['constraints'].get('accept_all_before_travel'):
        moving=next((i for i,r in enumerate(trace) if r.get('command','').startswith('MOVE(')),len(trace))
        if moving<len(trace) and not all(trace[moving-1].get('orders',{}).get(str(o['order_id']),{}).get('is_accepted')
                                        for o in case['private']['orders']):reasons.append('Travel began before accepting every order')
    if len(trace)-1>case['budget']['actions']:reasons.append('Action budget exceeded')
    if trace[-1].get('sim_time_s',math.inf)>case['budget']['sim_seconds']:reasons.append('Episode time budget exceeded')
    return {'success':not reasons,'reasons':reasons,'orders':per_order,'environment_actions':max(0,len(trace)-1),
            'native_reward_used':False}
