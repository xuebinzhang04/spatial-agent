"""CPU tests by default. SPATIAL_TEST_THOR=1 adds real Unity integration tests."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from spatial_interaction import EpisodeConfig, make_env
from spatial_interaction.core.actions import parse_action, InvalidAction, CAMERA
from spatial_interaction.core.episode import save_episodes, load_episodes
from spatial_interaction.core.rewards import evaluate_changes
from spatial_interaction.core.trajectory import TrajectoryRecorder, export_gif, replay, frame_hash
from spatial_interaction.environments.shape3d.geometry import (
    ROTATIONS, quarter_turn, rotation_error, symmetry_group, canonical, generate_shape, shape_split, shortest_alignment)
from spatial_interaction.environments.shape3d.renderer import render
from spatial_interaction.agents.policies import RandomAgent, OracleAgent
from spatial_interaction.agents.vlm_adapter import VLMAdapter
from spatial_interaction.benchmarks import generate_episodes


def shape_config(task='shape_identification', **kwargs):
    return EpisodeConfig(task_type=task, **kwargs)


class ProtocolTests(unittest.TestCase):
    def test_action_schema(self):
        self.assertEqual(parse_action('{"action":"CAMERA_LEFT"}', CAMERA), {'action':'CAMERA_LEFT'})
        for raw in [None, [], {}, 'x', {'action':'CAMERA_LEFT','x':0}, {'action':'CAMERA_LEFT','memory':{}},
                    {'action':'SUBMIT','answer':{'changes':[{'type':'missing'}]}},
                    {'action':'OpenObject'}, {'action':['CAMERA_LEFT']}]:
            with self.subTest(raw=raw), self.assertRaises(InvalidAction):
                parse_action(raw, CAMERA+('SUBMIT','OpenObject'))

    def test_duplicate_change_rejected(self):
        change = {'type':'missing','object_id':'a','previous_location':'b'}
        with self.assertRaises(InvalidAction):
            parse_action({'action':'SUBMIT','answer':{'changes':[change,change]}}, ('SUBMIT',))

    def test_config_validation(self):
        for fields in [{'environment':'x'}, {'seed':-1}, {'max_steps':1}, {'difficulty':0},
                       {'environment':'dynamic_memory','task_type':'missing_object','difficulty':4}]:
            with self.assertRaises(ValueError):
                EpisodeConfig(**fields)

    def test_episode_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            configs = generate_episodes(4)
            for name in ('episodes.json','episodes.jsonl'):
                path = Path(d)/name
                save_episodes(configs,path)
                self.assertEqual(configs,load_episodes(path))

    def test_change_metrics(self):
        truth = {'changes':[{'type':'missing','object_id':'a','previous_location':'table'},
                            {'type':'moved','object_id':'b','previous_location':'shelf','current_location':'table'}]}
        metrics = evaluate_changes(truth,truth,3,50)
        self.assertTrue(metrics['success'])
        self.assertAlmostEqual(metrics['episode_reward'], .994)
        wrong = {'changes':[{'type':'missing','object_id':'b','previous_location':'wrong'}]}
        metrics = evaluate_changes(wrong,truth)
        self.assertAlmostEqual(metrics['change_f1'],2/3)
        self.assertEqual(metrics['change_type_accuracy'],0.)
        self.assertEqual(metrics['current_location_accuracy'],0.)
        self.assertFalse(metrics['success'])
        self.assertEqual(evaluate_changes({'changes':[]},truth)['change_f1'],0.)


class GeometryTests(unittest.TestCase):
    def test_rotation_group(self):
        self.assertEqual(len(ROTATIONS),24)
        for r in ROTATIONS:
            np.testing.assert_array_equal(r.T@r,np.eye(3))
            self.assertAlmostEqual(np.linalg.det(r),1.)
        for axis in 'XYZ':
            r=quarter_turn(axis)
            np.testing.assert_array_equal(np.linalg.matrix_power(r,4),np.eye(3))
            np.testing.assert_array_equal(quarter_turn(axis,-1)@r,np.eye(3))

    def test_composition_order(self):
        x,y=quarter_turn('X'),quarter_turn('Y')
        self.assertFalse(np.array_equal(x@y,y@x))
        cells=generate_shape(np.random.default_rng(8))
        self.assertEqual(rotation_error(x@x,x@x,cells),0.)
        self.assertAlmostEqual(rotation_error(x,np.eye(3),cells),90.)

    def test_rotational_symmetry(self):
        cube=np.array([[0,0,0]])
        self.assertEqual(len(symmetry_group(cube)),24)
        self.assertEqual(rotation_error(quarter_turn('X'),np.eye(3),cube),0.)
        cells=generate_shape(np.random.default_rng(0),symmetry='asymmetric')
        self.assertEqual(len(symmetry_group(cells)),1)
        self.assertGreater(rotation_error(quarter_turn('Y'),np.eye(3),cells),0.)
        for r in ROTATIONS:
            self.assertEqual(canonical(cells),canonical(cells@r.T+12))

    def test_shape_splits_disjoint(self):
        keys={}
        for split in ('development','validation','test'):
            keys[split]=set()
            for seed in range(4):
                cells=generate_shape(np.random.default_rng(seed),split=split)
                self.assertEqual(shape_split(cells),split)
                keys[split].add(canonical(cells))
        self.assertFalse(keys['development']&keys['test'])
        self.assertFalse(keys['development']&keys['validation'])
        self.assertFalse(keys['validation']&keys['test'])

    def test_procedural_symmetric_shapes(self):
        for n in (4,5,6,9,10):
            cells=generate_shape(np.random.default_rng(n),voxels=n,symmetry='symmetric')
            self.assertEqual(len(cells),n)
            self.assertGreater(len(symmetry_group(cells)),1)

    def test_renderer_depth_and_camera(self):
        cells=np.array([[0,0,0]])
        rgb=render(cells,np.eye(3),[0,0,-5],size=80)
        self.assertEqual(rgb.shape,(80,80,3))
        self.assertEqual(rgb.dtype,np.uint8)
        self.assertFalse(np.array_equal(rgb[40,40],rgb[0,0]))
        self.assertGreater(np.count_nonzero(rgb!=render(cells,np.eye(3),[2,0,-5],size=80)),0)
        occluded=render(cells,np.eye(3),[0,0,-5],size=80,occlusion=.8)
        self.assertTrue(np.array_equal(occluded[40,40],[113,105,91]))


class ShapeEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env=make_env(shape_config(task_config={'image_size':64}))

    def tearDown(self):
        self.env.close()

    def test_reset_and_seeding(self):
        a,info=self.env.reset(seed=17)
        truth=self.env.ground_truth()
        b,_=self.env.reset(seed=17)
        self.assertTrue(info['valid'])
        np.testing.assert_array_equal(a['rgb'],b['rgb'])
        self.assertEqual(truth,self.env.ground_truth())
        c,_=self.env.reset(seed=18)
        self.assertNotEqual(truth['shape'],self.env.ground_truth()['shape'])
        self.assertEqual(set(c),{'rgb','instruction','phase','action_history','memory'})

    def test_partial_observation_no_labels(self):
        obs,_=self.env.reset()
        serialized=json.dumps({k:v for k,v in obs.items() if k!='rgb'})
        for forbidden in ('candidate_id": "candidate_', 'reference_orientation','ground_truth','camera_position'):
            self.assertNotIn(forbidden,serialized)
        obs['action_history'].append('mutate')
        self.assertEqual(self.env.history,[])

    def test_invalid_and_termination(self):
        self.env=make_env(shape_config(max_steps=2,task_config={'image_size':64}))
        self.env.reset()
        _,r,t,tr,info=self.env.step({'action':'BAD'})
        self.assertFalse(info['action_valid'])
        self.assertFalse(t or tr)
        _,r2,t,tr,_=self.env.step(None)
        self.assertTrue(tr)
        self.assertFalse(t)
        self.assertAlmostEqual(r+r2,-.1)
        with self.assertRaises(RuntimeError): self.env.step({'action':'CAMERA_LEFT'})

    def test_shape_reward_and_memory(self):
        self.env.reset()
        self.env.step({'action':'CAMERA_LEFT','memory':'I saw a side face'})
        obs,reward,t,tr,info=self.env.step({'action':'SUBMIT_SHAPE','candidate_id':self.env.correct})
        self.assertTrue(t)
        self.assertFalse(tr)
        self.assertTrue(info['metrics']['success'])
        self.assertAlmostEqual(reward,.998)
        self.assertEqual(obs['memory'],'I saw a side face')

    def test_camera_object_independent(self):
        self.env=make_env(shape_config('pose_alignment',task_config={'image_size':64}))
        self.env.reset()
        orientation=self.env.orientation.copy()
        self.env.step({'action':'CAMERA_YAW_LEFT'})
        np.testing.assert_array_equal(orientation,self.env.orientation)
        camera=self.env.camera.copy()
        yaw=self.env.yaw
        self.env.step({'action':'OBJECT_ROTATE_X_POS'})
        np.testing.assert_array_equal(camera,self.env.camera)
        self.assertEqual(yaw,self.env.yaw)
        np.testing.assert_array_equal(self.env.orientation,quarter_turn('X')@orientation)

    def test_pose_oracle_and_restricted_axes(self):
        for axes in ('X','YZ','XYZ'):
            self.env=make_env(shape_config('pose_alignment',task_config={'image_size':64,'axes':axes}))
            obs,_=self.env.reset()
            agent=OracleAgent(self.env)
            while not self.env.done:
                obs,_,_,_,_=self.env.step(agent.act(obs,self.env.available_actions))
            self.assertTrue(self.env.metrics['success'])
            self.assertEqual(self.env.metrics['rotation_error_degrees'],0.)

    def test_difficulty_controls_alignment_depth(self):
        # Depth = min(L, 3) with all axes; restricted axes cap it at their deepest pose.
        for level in range(1, 6):
            for seed in (1, 2):
                self.env=make_env(shape_config('pose_alignment',seed=seed,difficulty=level,task_config={'image_size':64}))
                self.env.reset()
                path=shortest_alignment(self.env.orientation,self.env.reference,self.env.cells)
                self.assertEqual(len(path),min(level,3))
                self.assertEqual(self.env.alignment_length,len(path))
        self.env=make_env(shape_config('pose_alignment',difficulty=5,task_config={'image_size':64,'axes':'Z'}))
        self.env.reset()
        self.assertEqual(len(shortest_alignment(self.env.orientation,self.env.reference,self.env.cells,'Z')),2)
        self.env=make_env(shape_config('pose_alignment',difficulty=1,task_config={'image_size':64,'alignment_length':3}))
        self.env.reset()
        self.assertEqual(self.env.alignment_length,3)

    def test_wrong_pose(self):
        self.env=make_env(shape_config('pose_alignment',task_config={'image_size':64}))
        self.env.reset()
        _,r,t,_,info=self.env.step({'action':'SUBMIT_POSE'})
        self.assertTrue(t)
        self.assertLess(r,0)
        self.assertFalse(info['metrics']['success'])

    def test_generated_oracles_and_random(self):
        for split in ('development','validation','test'):
            for cfg in generate_episodes(4,split,environment='shape3d'):
                cfg=EpisodeConfig(**{**cfg.to_dict(),'task_config':{**cfg.task_config,'image_size':64}})
                self.env=make_env(cfg)
                obs,_=self.env.reset()
                agent=OracleAgent(self.env)
                while not self.env.done:
                    obs,*_=self.env.step(agent.act(obs,self.env.available_actions))
                self.assertTrue(self.env.metrics['success'])
        for task in ('shape_identification','pose_alignment'):
            self.env=make_env(shape_config(task,max_steps=8,task_config={'image_size':64}))
            obs,_=self.env.reset()
            random=RandomAgent(18)
            while not self.env.done:
                obs,*_=self.env.step(random.act(obs,self.env.available_actions))

    def test_trajectory_and_replay(self):
        obs,_=self.env.reset()
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d)/'episode'
            recorder=TrajectoryRecorder(directory,self.env.config,True)
            recorder.record(obs,state=self.env.privileged_state())
            action={'action':'CAMERA_RIGHT'}
            obs,r,t,tr,info=self.env.step(action)
            recorder.record(obs,action,r,t,tr,info,self.env.privileged_state())
            recorder.close()
            self.assertTrue(export_gif(directory).is_file())
            self.assertTrue(replay(self.env,directory)['verified'])
            text=(directory/'trajectory.jsonl').read_text()
            self.assertNotIn('ground_truth',text)
            self.assertIn('ground_truth',(directory/'privileged.jsonl').read_text())


class AdapterTests(unittest.TestCase):
    def test_model_parser_and_observation_boundary(self):
        env=make_env(shape_config(task_config={'image_size':64}))
        try:
            obs,_=env.reset()
            obs['ground_truth']='SECRET_MARKER'
            calls=[]
            def fake(payload):
                calls.append(payload)
                return {'choices':[{'message':{'content':'```json\n{"action":"CAMERA_LEFT","memory":"notes"}\n```'}}]}
            adapter=VLMAdapter(transport=fake)
            for _ in range(2):
                action=adapter.act(obs,env.available_actions)
                obs,*_=env.step(action)
            self.assertEqual(len(calls),2)
            self.assertNotIn('SECRET_MARKER',json.dumps(calls))
            self.assertEqual(env.memory,'notes')
            self.assertTrue(calls[0]['messages'][0]['content'][1]['image_url']['url'].startswith('data:image/png;base64,'))
        finally:
            env.close()

    def test_invalid_model_output(self):
        adapter=VLMAdapter(transport=lambda p:{'choices':[{'message':{'content':'nonsense'}}]})
        obs={'rgb':np.zeros((10,10,3),np.uint8),'instruction':'test','phase':'interaction','memory':'','action_history':[]}
        self.assertEqual(adapter.act(obs,CAMERA),{'action':'INVALID'})
        self.assertIsNotNone(adapter.calls[0]['parse_error'])


@unittest.skipUnless(os.environ.get('SPATIAL_TEST_THOR')=='1','requires real cached Unity/Vulkan runtime')
class ThorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory(prefix='stage1-thor-tests-')
        cls.env=make_env(EpisodeConfig(environment='dynamic_memory',task_type='missing_object',
            scene_id='FloorPlan1',max_steps=16,task_config={'image_size':128,'exploration_budget':3}),
            runtime_dir=cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()
        cls.directory.cleanup()

    def test_missing_native_intervention(self):
        env=self.env
        env.config=EpisodeConfig(environment='dynamic_memory',task_type='missing_object',scene_id='FloorPlan1',
                                 max_steps=16,task_config={'image_size':128,'exploration_budget':3})
        obs,info=env.reset(seed=42)
        self.assertTrue(info['intervention_preflight'])
        self.assertEqual(obs['phase'],'memorization')
        initial=frame_hash(obs['rgb'])
        target=env.target
        obs,_,_,_,info=env.step({'action':'MEMORIZE','memory':'my own notes'})
        self.assertEqual(obs['phase'],'detection')
        self.assertTrue(info['intervention_verified'])
        self.assertNotIn(target,env.privileged_state()['current_state'])
        self.assertNotEqual(initial,frame_hash(obs['rgb']))
        _,reward,terminated,_,info=env.step({'action':'SUBMIT','answer':env.truth})
        self.assertTrue(terminated)
        self.assertTrue(info['metrics']['success'])
        self.assertGreater(reward,.9)

    def test_moved_native_intervention(self):
        env=self.env
        env.config=EpisodeConfig(environment='dynamic_memory',task_type='moved_object',scene_id='FloorPlan1',
                                 difficulty=3,max_steps=16,task_config={'image_size':128,'exploration_budget':3})
        obs,_=env.reset(seed=43)
        before=env.initial_state[env.target]['position']
        obs,_,_,_,_=env.step({'action':'MEMORIZE'})
        after=env.privileged_state()['current_state'][env.target]['position']
        self.assertNotEqual(before,after)
        self.assertNotEqual(env.truth['changes'][0]['previous_location'],env.truth['changes'][0]['current_location'])
        _,_,t,_,info=env.step({'action':'SUBMIT','answer':env.truth})
        self.assertTrue(t)
        self.assertTrue(info['metrics']['success'])

    def test_reset_determinism_invalid_and_auto_phase(self):
        env=self.env
        env.config=EpisodeConfig(environment='dynamic_memory',task_type='missing_object',scene_id='FloorPlan1',
                                 max_steps=8,task_config={'image_size':128,'exploration_budget':2})
        a,_=env.reset(seed=7)
        state=env.privileged_state()
        b,_=env.reset(seed=7)
        self.assertEqual(state['target_id'],env.privileged_state()['target_id'])
        current=env.privileged_state()
        for oid in state['initial_state']:
            np.testing.assert_allclose(list(state['initial_state'][oid]['position'].values()),
                                       list(current['initial_state'][oid]['position'].values()),atol=1e-5)
        delta=np.abs(a['rgb'].astype(np.int16)-b['rgb'].astype(np.int16))
        self.assertLessEqual(delta.max(),3)
        self.assertLessEqual(delta.mean(),.5)
        obs,_,_,_,info=env.step({'action':'OpenObject','object_id':'unknown'})
        self.assertFalse(info['action_valid'])
        self.assertEqual(obs['phase'],'memorization')
        obs,_,_,_,_=env.step({'action':'RotateRight'})
        self.assertEqual(obs['phase'],'detection')
        agent=RandomAgent(33)
        while not env.done:
            obs,*_=env.step(agent.act(obs,env.available_actions))

    def test_l2_requires_viewpoint_adjustment(self):
        env=self.env
        env.config=EpisodeConfig(environment='dynamic_memory',task_type='missing_object',scene_id='FloorPlan1',
                                 difficulty=2,max_steps=30,task_config={'image_size':128,'exploration_budget':12})
        obs,_=env.reset(seed=42)
        self.assertFalse(env.privileged_state()['current_state'][env.target]['visible'])
        for _ in range(4):
            obs,*_=env.step({'action':'RotateLeft'})
        self.assertTrue(env.privileged_state()['current_state'][env.target]['visible'])
        env.step({'action':'MEMORIZE'})
        _,_,t,_,info=env.step({'action':'SUBMIT','answer':env.truth})
        self.assertTrue(t and info['metrics']['success'])

    def test_native_trajectory_replay(self):
        env=self.env
        env.config=EpisodeConfig(environment='dynamic_memory',task_type='missing_object',scene_id='FloorPlan1',
                                 max_steps=16,task_config={'image_size':128,'exploration_budget':3})
        obs,_=env.reset(seed=42)
        with tempfile.TemporaryDirectory() as d:
            recorder=TrajectoryRecorder(Path(d)/'episode',env.config)
            recorder.record(obs)
            for action in ({'action':'MEMORIZE'},):
                obs,r,t,tr,info=env.step(action)
                recorder.record(obs,action,r,t,tr,info)
            action={'action':'SUBMIT','answer':env.truth}
            obs,r,t,tr,info=env.step(action)
            recorder.record(obs,action,r,t,tr,info)
            recorder.close()
            self.assertTrue(replay(env,Path(d)/'episode')['verified'])

    def test_native_motion_and_interaction_dispatch(self):
        env=self.env
        env.config=EpisodeConfig(environment='dynamic_memory',task_type='missing_object',scene_id='FloorPlan1',
                                 max_steps=40,task_config={'image_size':128,'exploration_budget':25})
        env.reset(seed=42)
        for action in ('MoveAhead','MoveBack','RotateLeft','RotateRight','LookUp','LookDown'):
            obs,_,t,tr,info=env.step({'action':action})
            self.assertFalse(t or tr)
            self.assertEqual(obs['rgb'].dtype,np.uint8)
        current=env.privileged_state()['current_state']
        for action,capability in (('OpenObject','openable'),('CloseObject','openable'),('PickupObject','pickupable')):
            candidates=[o for o in current.values() if o.get(capability) and o.get('visible')]
            if candidates:
                oid=env.id_map[candidates[0]['objectId']]
                obs,_,t,tr,info=env.step({'action':action,'object_id':oid})
                self.assertIn('action_valid',info)
                self.assertFalse(t or tr)


if __name__=='__main__':
    unittest.main(verbosity=2)
