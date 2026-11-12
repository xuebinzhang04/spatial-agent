"""Protocol checks with synthetic events; no claim of native scene admission."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatial_interaction import EpisodeConfig, make_env
from spatial_interaction.benchmarks import generate_episodes
from spatial_interaction.environments.dynamic_memory.env import DynamicMemoryEnv
from spatial_interaction.core.actions import InvalidAction
from spatial_interaction.agents.policies import RandomAgent


class ProtocolEnv(DynamicMemoryEnv):
    def _intervene(self):
        self.phase='detection'
        self.phase_steps=0
        self.truth={'changes':[{'type':'missing','object_id':'object_001','previous_location':'object_002'}]}


class ActiveMemoryTests(unittest.TestCase):
    def env(self, active=True):
        cfg=EpisodeConfig(environment='dynamic_memory',task_type='missing_object',max_steps=8,
                          task_config={'exploration_budget':3, **({'memory_protocol':'active_scan_v1'} if active else {})})
        env=ProtocolEnv(cfg,runtime_dir='/tmp')
        env._start(None,None)
        env.phase='memorization';env.phase_steps=0;env.explore_budget=3
        env.id_map={'mug':'object_001','table':'object_002'}
        env.reverse_ids={v:k for k,v in env.id_map.items()}
        env.initial_state={'mug':{'objectType':'Mug'},'table':{'objectType':'Table'}}
        mask=np.zeros((8,8),bool);mask[1:3,1:3]=True
        support=np.zeros((8,8),bool);support[4:7,1:7]=True
        env.controller=SimpleNamespace(last_event=SimpleNamespace(frame=np.zeros((8,8,3),np.uint8),
            instance_masks={'mug':mask,'table':support},
            metadata={'objects':[{'objectId':'mug','visible':True},{'objectId':'table','visible':True}]}))
        env.instruction=env.active_instruction()
        env.render()
        return env

    def test_deadline_fails_without_silently_committing(self):
        env=self.env()
        for _ in range(3):obs,reward,terminated,truncated,info=env.step({'action':'INVALID'})
        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertEqual(obs['phase'],'memorization')
        self.assertEqual(info['metrics']['failure_reason'],'memorization_deadline_without_commit')
        self.assertIn('episode ended without intervention',obs['feedback'])

    def test_explicit_commit_freezes_notebook_and_updates_prompt(self):
        env=self.env()
        obs,*_=env.step({'action':'MEMORIZE','memory':'Mug on table; retain both pixel references.'})
        self.assertEqual(obs['phase'],'detection')
        self.assertEqual(obs['memory_checkpoint'],obs['memory'])
        self.assertIn('Current stage: DETECTION',obs['instruction'])
        obs,*_=env.step({'action':'INVALID','memory':'discarded invalid note'})
        self.assertIn('Mug on table',obs['memory_checkpoint'])
        self.assertIn('Action failed',obs['feedback'])
        self.assertNotIn('object_001',str({k:v for k,v in obs.items() if k!='rgb'}))

    def test_public_policy_cannot_submit_private_identifiers(self):
        env=self.env()
        env.step({'action':'MEMORIZE'})
        _,reward,terminated,_,info=env.step({'action':'SUBMIT','answer':deepcopy(env.truth)})
        self.assertFalse(terminated)
        self.assertFalse(info['action_valid'])
        self.assertIn('private identifiers are rejected',info['error'])
        with self.assertRaises(InvalidAction):env.resolve_reference('object_001')

    def test_guessing_private_correct_answer_without_observation_fails(self):
        env=self.env()
        env.privileged_references=True  # isolates the pre-change evidence gate
        env.visual_frames={}
        env.step({'action':'MEMORIZE'})
        _,reward,terminated,_,info=env.step({'action':'SUBMIT','answer':deepcopy(env.truth)})
        self.assertTrue(terminated)
        self.assertFalse(info['metrics']['success'])
        self.assertFalse(info['metrics']['pre_change_evidence'])
        self.assertLess(reward,0)

    def test_visual_answer_and_task_verifier_share_strict_result(self):
        env=self.env()
        env.step({'action':'MEMORIZE'})
        answer={'changes':[{'type':'missing','object_id':'view:0:0.2:0.2','previous_location':'view:0:0.5:0.7'}]}
        self.assertTrue(env.evaluate_answer(answer)['success'])
        _,reward,terminated,_,info=env.step({'action':'SUBMIT','answer':answer})
        self.assertTrue(terminated and info['metrics']['success'])
        self.assertGreater(reward,.9)
        self.assertEqual(answer['changes'][0]['object_id'],'view:0:0.2:0.2')

    def test_legacy_recipe_preserves_automatic_transition(self):
        env=self.env(active=False)
        for _ in range(3):obs,*_=env.step({'action':'INVALID'})
        self.assertEqual(obs['phase'],'detection')

    def test_generated_memory_cases_use_explicit_protocol(self):
        for task in ('missing_object','moved_object'):
            cfg=generate_episodes(1,task_type=task)[0]
            self.assertEqual(cfg.task_config['memory_protocol'],'active_scan_v1')
            self.assertEqual(cfg.max_steps,80)
            env=DynamicMemoryEnv(cfg,runtime_dir='/tmp')
            prompt=env.active_instruction()
            self.assertIn('REPLACES the entire notebook',prompt)
            self.assertIn('MEMORIZE',prompt)
            self.assertNotIn('object_001',prompt)
            if task=='moved_object':self.assertIn('current_location',prompt)


class PublicDescriptionTests(unittest.TestCase):
    def scene(self):
        return {'a':{'objectId':'a','objectType':'StoveBurner','position':{'x':0.,'y':0.,'z':0.}},
                'b':{'objectId':'b','objectType':'StoveBurner','position':{'x':.2,'y':0.,'z':0.}},
                'f':{'objectId':'f','objectType':'Fridge','position':{'x':3.,'y':0.,'z':0.}},
                'm':{'objectId':'m','objectType':'Mug','position':{'x':-2.,'y':0.,'z':0.}}}

    def test_unique_and_ambiguous_types(self):
        env=object.__new__(DynamicMemoryEnv)
        scene=self.scene()
        self.assertEqual(env.public_description('f',scene),'Fridge')
        # Burners 0.2 m apart cannot be separated by the 0.3 m landmark margin.
        self.assertIsNone(env.public_description('a',scene))
        self.assertIsNone(env.public_description('b',scene))

    def test_landmark_disambiguation(self):
        env=object.__new__(DynamicMemoryEnv)
        scene=self.scene();scene['b']['position']['x']=2.
        self.assertEqual(env.public_description('b',scene),'StoveBurner nearest the Fridge')

    def test_random_policy_uses_public_references(self):
        action=RandomAgent(0).act({'view_step':4},('MARK',))
        self.assertRegex(action['object_id'],r'^view:4:0\.\d{3}:0\.\d{3}$')


class ActivePerceptionTests(unittest.TestCase):
    def test_actual_initial_pixel_twins_and_revealing_action_sequence(self):
        from spatial_interaction.environments.shape3d.renderer import render
        env=make_env(EpisodeConfig(seed=1242,max_steps=24,task_config={'image_size':64,
            'require_initial_pixel_ambiguity':True,'initial_rotation':0,'voxels':6,'split':'development'}))
        try:
            env.reset()
            def image(cells):return render(cells,env.orientation,env.camera,env.yaw,env.pitch,env.size*2,env.occlusion)
            first=image(env.cells)
            twins=[c.copy() for i,c in enumerate(env.candidates) if f'candidate_{i+1}'!=env.correct and np.array_equal(first,image(c))]
            self.assertTrue(twins)
            witness=env.ambiguity_witness.copy()
            for action in witness['actions']:env.step(action)
            self.assertGreaterEqual(witness['changed_foreground_fraction'],.10)
            self.assertGreaterEqual(witness['retained_target_area_ratio'],.60)
            self.assertTrue(any(not np.array_equal(image(env.cells),image(c)) for c in twins))
            self.assertNotIn('ambiguity_witness',env._observation())
            # The instruction may flag insufficiency but must not prescribe the solving action.
            self.assertNotIn('Translate',env.instruction)
        finally:env.close()

    def test_wrong_initial_render_is_rejected_even_with_shared_front_geometry(self):
        env=make_env(EpisodeConfig(seed=1244,max_steps=12,task_config={'image_size':64,
            'require_initial_pixel_ambiguity':True,'initial_rotation':0,'voxels':6,'split':'development'}))
        try:
            with self.assertRaisesRegex(ValueError,'actual initial pixels'):env.reset()
        finally:env.close()


if __name__=='__main__':unittest.main(verbosity=2)
