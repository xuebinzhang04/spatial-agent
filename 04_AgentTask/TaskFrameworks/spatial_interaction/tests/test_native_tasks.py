"""Task-independent route/action boundaries; real scenes are checked in Slurm."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial_interaction import EpisodeConfig,make_env
from spatial_interaction.core.actions import parse_action,InvalidAction
from spatial_interaction.environments.thor_tasks.base import ThorTaskEnv
from spatial_interaction.environments.dynamic_memory.interventions import InvalidEpisode


class NativeTaskProtocolTests(unittest.TestCase):
    def visual_env(self):
        from types import SimpleNamespace
        import numpy as np
        from spatial_interaction.environments.dynamic_memory.env import DynamicMemoryEnv
        env = DynamicMemoryEnv(EpisodeConfig(), runtime_dir='/tmp')
        env.steps, env.visual_frames = 0, {}
        env.id_map = {'Mug|private': 'object_001', 'Table|private': 'object_002'}
        env.reverse_ids = {v:k for k,v in env.id_map.items()}
        mug = np.zeros((8,8), dtype=bool); mug[1:3,1:3] = True
        table = np.zeros((8,8), dtype=bool); table[4:7,1:7] = True
        env.initial_state = {oid: {'objectType':oid.split('|')[0]} for oid in env.id_map}
        event = SimpleNamespace(frame=np.full((8,8,3), 100, dtype=np.uint8),
            instance_masks={'Mug|private':mug, 'Table|private':table},
            metadata={'objects':[{'objectId':oid,'visible':True} for oid in env.id_map]})
        env.controller = SimpleNamespace(last_event=event)
        return env

    def test_raw_rgb_and_observed_pixel_references(self):
        import numpy as np
        env = self.visual_env()
        np.testing.assert_array_equal(env.render(), env.controller.last_event.frame)
        self.assertEqual(env.resolve_reference('view:0:0.2:0.2'), 'Mug|private')
        env.steps = 1
        env.controller.last_event.instance_masks = {}
        env.render()
        self.assertEqual(env.resolve_reference('view:0:0.2:0.2'), 'Mug|private')
        for ref in ('view:1:0.2:0.2','view:99:0.2:0.2','view:0:1:0.2','view:0:nan:0.2','view:0:0.9:0.9'):
            with self.subTest(ref=ref), self.assertRaises(InvalidAction):
                env.resolve_reference(ref)
        env = self.visual_env()
        env.controller.last_event.instance_masks['Table|private'] |= env.controller.last_event.instance_masks['Mug|private']
        env.render()
        with self.assertRaises(InvalidAction): env.resolve_reference('view:0:0.2:0.2')
        env = self.visual_env()
        env.controller.last_event.metadata['objects'][0]['visible'] = False
        env.render()
        with self.assertRaises(InvalidAction): env.resolve_reference('view:0:0.2:0.2')

    def test_public_goal_has_no_object_number_and_rejects_ambiguous_goal(self):
        env = self.visual_env()
        env.instruction = 'Visit object_001. MARK with object_NNN.'
        env.phase, env.history, env.memory = 'interaction', [], ''
        obs = env._observation()
        self.assertNotIn('object_001', obs['instruction'])
        self.assertNotIn('object_NNN', obs['instruction'])
        self.assertIn('Mug', obs['instruction'])
        env.initial_state['other'] = {'objectType':'Mug'}
        with self.assertRaises(InvalidEpisode): env._observation()

    def test_visual_submission_resolves_to_private_evaluator_identity(self):
        env = self.visual_env()
        env.render()
        env.truth = {'changes':[{'type':'missing','object_id':'object_001','previous_location':'object_002'}]}
        env.steps = 1
        _, info = env._execute({'action':'SUBMIT','answer':{'changes':[
            {'type':'missing','object_id':'view:0:0.2:0.2','previous_location':'view:0:0.5:0.7'}]}})
        self.assertTrue(info['metrics']['success'])
        with self.assertRaises(InvalidAction):
            env._execute({'action':'SUBMIT','answer':{'changes':[
                {'type':'missing','object_id':'view:0:0.2:0.2','previous_location':'view:0:0.5:0.7'},
                {'type':'missing','object_id':'view:0:0.25:0.25','previous_location':'view:0:0.5:0.7'}]}})

    def test_native_action_schema_is_strict(self):
        for action in ('MARK','INSPECT','CLEAN','POWER_ON','PUT_INSIDE'):
            with self.assertRaises(InvalidAction):parse_action({'action':action},(action,))
            self.assertEqual(parse_action({'action':action,'object_id':'object_001'},(action,))['object_id'],'object_001')
        with self.assertRaises(InvalidAction):parse_action({'action':'PLACE','position':[0,0,0]},('PLACE',))
        self.assertEqual(parse_action({'action':'PLACE','slot_id':'slot_1'},('PLACE',))['slot_id'],'slot_1')

    def test_routes_do_not_cross_missing_physical_cells(self):
        env=object.__new__(ThorTaskEnv)
        env.positions={(0.,0.):{},(.25,0.):{},(.25,.25):{}}
        env.graph={(0.,0.):[(.25,0.)],(.25,0.):[(0.,0.),(.25,.25)],(.25,.25):[(.25,0.)]}
        self.assertEqual(env.route((0,0),(.25,.25)),[(0.,0.),(.25,0.),(.25,.25)])
        env.graph[(.25,0.)]=[(0.,0.)]
        with self.assertRaises(InvalidEpisode):env.route((0,0),(.25,.25))

    def test_surface_projection_tracks_camera_yaw_and_pitch(self):
        from spatial_interaction.environments.thor_tasks.placement import project
        metadata={'cameraPosition':{'x':0.,'y':1.,'z':0.},
                  'agent':{'rotation':{'y':0.},'cameraHorizon':0.},'fov':90.}
        self.assertEqual(project({'x':0.,'y':1.,'z':2.},metadata,128),(64,64,2.))
        self.assertIsNone(project({'x':0.,'y':1.,'z':-2.},metadata,128))
        metadata['agent']['rotation']['y']=90.
        projected=project({'x':2.,'y':1.,'z':0.},metadata,128)
        self.assertEqual(projected[:2],(64,64))
        metadata['agent']['cameraHorizon']=30.
        projected=project({'x':1.7320508075688772,'y':0.,'z':0.},metadata,128)
        self.assertEqual(projected[:2],(64,64))

    def test_surface_markers_respect_depth_occlusion(self):
        from types import SimpleNamespace
        import numpy as np
        from spatial_interaction.environments.thor_tasks.placement import SurfaceTaskEnv
        env=object.__new__(SurfaceTaskEnv);env.size=128
        env.slots={'slot_1':{'x':0.,'y':1.,'z':2.}}
        metadata={'cameraPosition':{'x':0.,'y':1.,'z':0.},
                  'agent':{'rotation':{'y':0.},'cameraHorizon':0.},'fov':90.}
        event=SimpleNamespace(metadata=metadata,depth_frame=np.ones((128,128)))
        env.controller=SimpleNamespace(last_event=event)
        self.assertEqual(env.visible_slots(),{})
        event.depth_frame[:]=2.
        self.assertIn('slot_1',env.visible_slots())

    def test_native_generation_rejects_foreign_backends_and_splits_layouts(self):
        from spatial_interaction.benchmarks import generate_episodes
        with self.assertRaises(ValueError):
            generate_episodes(1,environment='shape3d',task_type='memory_maze')
        scenes=[]
        for split in ('development','validation','test'):
            configurations=generate_episodes(2,split,42,'thor_tasks','dependency_doors')
            self.assertTrue(all(c.task_type=='dependency_doors' for c in configurations))
            scenes.append({c.scene_id for c in configurations})
        self.assertFalse(scenes[0]&scenes[1] or scenes[1]&scenes[2] or scenes[0]&scenes[2])

    def test_metric_relations_require_every_conjunct_and_distance(self):
        from spatial_interaction.environments.thor_tasks.relations import relation_holds
        anchor={'x':1.,'z':1.}
        self.assertTrue(relation_holds({'x':1.3,'z':.7},anchor,[('x',1),('z',-1)],near=.5))
        self.assertFalse(relation_holds({'x':1.3,'z':1.3},anchor,[('x',1),('z',-1)],near=.5))
        self.assertFalse(relation_holds({'x':2.,'z':0.},anchor,[('x',1),('z',-1)],near=.5))
        self.assertFalse(relation_holds({'x':1.01,'z':.7},anchor,[('x',1),('z',-1)]))

    def test_native_replay_checks_geometry_even_with_identical_rgb(self):
        from copy import deepcopy
        from types import SimpleNamespace
        import tempfile
        import numpy as np
        from spatial_interaction.core.trajectory import TrajectoryRecorder,replay
        observation={'rgb':np.zeros((64,64,3),dtype=np.uint8),'instruction':'task','phase':'interaction',
                     'action_history':[],'memory':''}
        zero=dict(x=0.,y=0.,z=0.)
        state={'id_map':{'native':'object_001'},'current_state':{'native':{'position':dict(zero),'rotation':dict(zero)}},
               'agent':{'position':dict(zero),'rotation':dict(zero),'cameraHorizon':0.},'progress':0,'events':[]}
        current=deepcopy(state)
        env=SimpleNamespace(config=SimpleNamespace(environment='thor_tasks'),
                            reset=lambda:(observation,{}),privileged_state=lambda:current)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'trace'
            recorder=TrajectoryRecorder(path,EpisodeConfig(environment='thor_tasks',task_type='memory_maze'),True)
            recorder.record(observation,state=state);recorder.close()
            observation['rgb'][0,0,0]=6
            observation['rgb'][1,1,0]=80
            self.assertTrue(replay(env,path)['native_state_verified'])
            observation['rgb'][2,:5,0]=80
            with self.assertRaisesRegex(AssertionError,'raster tolerance'):
                replay(env,path)
            observation['rgb'][2,:5,0]=0
            current['current_state']['native']['position']['x']=.01
            with self.assertRaisesRegex(AssertionError,'object position'):
                replay(env,path)
            current['current_state']['native']['position']['x']=0.
            current['task_state']={'battery':5}
            with self.assertRaisesRegex(AssertionError,'task state'):
                replay(env,path)

    def test_surface_sampling_covers_both_axes_without_array_order_aliasing(self):
        from spatial_interaction.environments.thor_tasks.surfaces import spread_points
        points=[{'x':x/10,'y':1.,'z':z/10} for x in range(30) for z in range(10)]
        selected=spread_points(points,16)
        self.assertEqual(selected,spread_points(list(reversed(points)),16))
        self.assertGreater(len({p['z'] for p in selected}),3)
        self.assertGreater(len({p['x'] for p in selected}),3)
        self.assertEqual(len(selected),16)

    def test_unvalidated_levels_are_rejected(self):
        with self.assertRaises(ValueError):EpisodeConfig(environment='thor_tasks',task_type='memory_maze',difficulty=4)


if __name__=='__main__':unittest.main()
