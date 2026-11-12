"""Native-map construction and verifier boundary tests; no UE process is launched."""
from copy import deepcopy
import json
import math
import os
import tempfile
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cases import City,generate,verify,public_observation,content_hash


class CaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.city=City('small-city-11')

    def case(self,family='pickup_delivery',level=1):return generate(self.city,family,20261026,level)

    def trace(self,case):
        order=case['private']['orders'][0];oid=str(order['order_id'])
        initial={'source':'native','sim_time_s':0.,'position_cm':case['initial']['position_cm'],
            'ue_position_cm':case['initial']['position_cm'],'transport':'e-scooter','energy_pct':100.,'battery_pct':case['initial']['battery_pct'],
            'orders':{oid:{'is_accepted':False,'has_picked_up':False,'has_delivered':False}},'bag_order_ids':[]}
        rows=[initial]
        for i,changes in enumerate(({'is_accepted':True},{'has_picked_up':True},{},
                                   {'has_delivered':True,'delivery_method':order['required_method']}),1):
            row=deepcopy(rows[-1]);row['sim_time_s']=float(i);row['orders'][oid].update(changes)
            point=order['dropoff']['position_cm'] if i==4 else order['pickup']['position_cm']
            row['position_cm']=point;row['ue_position_cm']=point
            if i==3:row['bag_order_ids']=[int(oid)]
            if i==4:row['bag_order_ids']=[]
            rows.append(row)
        return rows

    def test_seed_reproduces_full_recipe_across_new_map_objects(self):
        a=self.case();b=generate(City('small-city-11'),'pickup_delivery',20261026,1)
        self.assertEqual(a,b)
        identity=dict(a);identity.pop('case_id')
        self.assertTrue(a['case_id'].endswith(content_hash(identity)[:12]))

    def test_public_input_excludes_routes_certificates_and_private_centers(self):
        case=self.case('visual_handoff')
        public=public_observation(case,{'sim_time_s':1,'private':'SECRET','handoff_positions_cm':'SECRET'})
        text=json.dumps(public)
        for token in ('SECRET','serial_witness','source_sha256','center_cm','shortcut_checks','handoff_positions_cm'):
            self.assertNotIn(token,text)
        self.assertEqual(case['status'],'native_visual_admission_required')

    def test_all_families_and_structural_levels_construct_from_real_map(self):
        for family in ('pickup_delivery','batch_schedule','recharge_delivery','visual_handoff'):
            for level in (1,2,3):
                case=generate(self.city,family,20261026+level,level)
                self.assertEqual(case['level'],level)
                self.assertGreater(case['private']['serial_witness']['distance_cm'],0)
                self.assertFalse(case['structural_difficulty']['empirically_calibrated'])
                self.assertGreater(case['private']['shortcut_checks']['remote_pickup_distance_cm'],1200)

    def test_batch_witness_and_alternative_have_different_deadline_outcomes(self):
        case=self.case('batch_schedule',2);private=case['private'];control=private['shortcut_checks']
        for o in private['orders']:
            self.assertLessEqual(private['serial_witness']['completion_s'][o['order_id']],o['deadline_s'])
        target=next(o for o in private['orders'] if o['order_id']==control['critical_order'])
        self.assertGreater(control['reversed_serial_completion_s'][target['order_id']],target['deadline_s'])

    def test_recharge_is_reachable_but_direct_trip_exceeds_initial_battery(self):
        for level in (1,2,3):
            case=self.case('recharge_delivery',level);control=case['private']['shortcut_checks']
            self.assertLess(control['battery_to_station_pct'],case['initial']['battery_pct'])
            self.assertLess(case['initial']['battery_pct'],control['battery_without_recharge_pct'])
            cost=case['private']['serial_witness']['distance_cm']/100*self.city.mechanics['scooter_batt_decay_pct_per_m']
            self.assertLess(cost,case['private']['charge']['target_pct'])

    def test_strict_state_and_physical_evidence_succeeds(self):
        case=self.case();result=verify(case,self.trace(case))
        self.assertTrue(result['success'],result)
        self.assertFalse(result['native_reward_used'])

    def test_remote_wrong_method_unpacked_and_unsynchronized_traces_fail(self):
        case=self.case();valid=self.trace(case)
        variants=[]
        remote=deepcopy(valid);remote[-1]['position_cm']=remote[0]['position_cm'];remote[-1]['ue_position_cm']=remote[0]['position_cm'];variants.append(remote)
        wrong=deepcopy(valid);wrong[-1]['orders']['1']['delivery_method']='call';variants.append(wrong)
        unpacked=deepcopy(valid);unpacked[-2]['bag_order_ids']=[];variants.append(unpacked)
        mismatch=deepcopy(valid);mismatch[2]['ue_position_cm']=[1e8,1e8];variants.append(mismatch)
        for trace in variants:self.assertFalse(verify(case,trace)['success'])

    def test_reward_flags_and_nonfinite_positions_cannot_forge_success(self):
        case=self.case();trace=self.trace(case)
        self.assertFalse(verify(case,[{'source':'native','reward':1.,'success':True}])['success'])
        trace[2]['ue_position_cm']=[math.nan,0.]
        self.assertFalse(verify(case,trace)['success'])
        trace=self.trace(case);trace[-1]['sim_time_s']=1e8
        self.assertFalse(verify(case,trace)['success'])

    def test_initial_battery_cannot_be_silently_replaced(self):
        case=self.case();trace=self.trace(case);trace[0]['battery_pct']=90
        self.assertFalse(verify(case,trace)['success'])

    def test_charging_at_another_public_station_is_allowed(self):
        case=self.case('recharge_delivery');trace=self.trace(case)
        chosen=case['private']['charge']['station']['position_cm']
        other=next(p['position_cm'] for p in case['public']['charging_stations'] if p['position_cm']!=chosen)
        charge=deepcopy(trace[1]);charge['position_cm']=other;charge['ue_position_cm']=other
        charge['battery_pct']=min(100.,case['initial']['battery_pct']+10)
        charge['sim_time_s']=1.5;charge['command']='WAIT(until="charge_done")';trace.insert(2,charge)
        self.assertTrue(verify(case,trace)['success'])

    def test_native_order_fixture_preserves_bindings_and_control_readiness(self):
        from types import SimpleNamespace
        from native import NativeCase
        from vlm_delivery.entities.order import OrderManager
        holder=object.__new__(NativeCase);holder.env=SimpleNamespace(map=self.city.map);holder.case=self.case()
        specification=holder.case['private']['orders'][0]
        order=holder._make_order(specification)
        self.assertEqual([order.pickup_node.position.x,order.pickup_node.position.y],specification['pickup']['position_cm'])
        self.assertEqual(order.prep_longest_s,specification['prep_time_s'])
        ready=deepcopy(specification);ready['order_id']=10001;ready['prep_time_s']=0.;ready['food']['prep_time_s']=0
        control=holder._make_order(ready)
        manager=OrderManager(capacity=1,seed=1);manager._orders=[control]
        self.assertFalse(control.is_ready_for_pickup())
        self.assertTrue(manager.accept_order(control.id))
        self.assertTrue(control.is_ready_for_pickup())

    def test_reference_command_syntax_matches_native_parser(self):
        from types import SimpleNamespace
        original=Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            working=Path(directory)/'task/DeliveryBench';working.mkdir(parents=True)
            try:
                # The upstream logger resolves ../../log relative to its checkout.
                os.chdir(working)
                from vlm_delivery.gameplay.action_space import parse_action
                for command in ('MOVE(100,200)','WAIT(seconds=10)','CHARGE(target_pct=80)',
                                'WAIT(until="charge_done")','SWITCH(to="e-scooter")',
                                'PLACE_FOOD_IN_BAG("order 1: 1 -> A")'):
                    action,_=parse_action(command,SimpleNamespace())
                    self.assertIsNotNone(action)
            finally:os.chdir(original)



if __name__=='__main__':unittest.main(verbosity=2)
