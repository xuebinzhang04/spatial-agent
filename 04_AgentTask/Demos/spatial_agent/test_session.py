"""Real lightweight environments and model/HTTP contracts, without GPU or sockets."""
import io
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run import handler
from session import DeliveryAdapter, ObservedSession, ROOT, Session, TaskAdapter


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.session = Session(self.directory.name)

    def tearDown(self):
        self.session.close()
        self.directory.cleanup()

    def reset_shape(self):
        return self.session.reset({"environment": "shape3d", "task": "shape_identification", "seed": 42, "budget": 20})

    def test_frozen_recipe_is_used_without_regenerating_defaults(self):
        import sys
        sys.path.insert(0,str(ROOT/'04_AgentTask/TaskFrameworks/spatial_interaction'))
        from spatial_interaction import EpisodeConfig
        spec=EpisodeConfig(seed=17,max_steps=9,task_config={'image_size':64,'candidates':2})
        state=self.session.reset({'environment':'shape3d','task':'shape_identification','seed':17,'difficulty':1,'budget':9},specification=spec)
        self.assertEqual(self.session.adapter.env.size,64)
        self.assertEqual(state['budget'],9)
        self.assertEqual(self.session.metadata['parameters']['frozen_recipe'],spec.to_dict())
        with self.assertRaisesRegex(ValueError,'Frozen recipe'):
            TaskAdapter('shape3d','pose_alignment',17,1,9,'procedural',specification=spec)

    def test_real_geometry_step_and_terminal_guard(self):
        before = self.reset_shape()
        after = self.session.step({"action": "CAMERA_RIGHT"})
        self.assertNotEqual(before["views"][0]["image"], after["views"][0]["image"])
        self.assertEqual(after["step"], 1)
        final = self.session.step({"action": "SUBMIT_SHAPE", "candidate_id": "candidate_1"})
        self.assertTrue(final["terminated"])
        with self.assertRaisesRegex(ValueError, "Episode ended"):
            self.session.step({"action": "CAMERA_RIGHT"})
        rows = [json.loads(line) for line in (Path(self.directory.name)/"session.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 3)

    def test_removed_graph_environment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown environment/task pair"):
            self.session.reset({"environment": "symbolic", "task": "signal_repair", "seed": 3})

    def test_pose_initialization(self):
        state = self.session.reset({"environment": "shape3d", "task": "pose_alignment", "seed": 4})
        self.assertIn("OBJECT_ROTATE_X_POS", state["actions"])

    def test_local_model_catalog_is_independent_of_service(self):
        capabilities = self.session.capabilities()
        self.assertIsNone(capabilities['model'])
        self.assertFalse(capabilities['model_verified'])
        qwen = next(m for m in capabilities['models'] if m['id']=='Qwen3.8-27B')
        self.assertEqual(qwen['checkpoint'],'01_Models/Qwen3.8-27B')

    def test_clean_replay_guard_rejects_camera_and_object_state_changes(self):
        from rerender import assert_scene
        source = ROOT/'05_Experiment/results/stage1-action-tours-20261019/cases/missing_object/privileged.jsonl'
        with source.open() as file: expected=json.loads(file.readline())['state']
        actual=deepcopy(expected);assert_scene(expected,actual)
        actual['current_pose']['position']['x']+=.25
        with self.assertRaises(AssertionError):assert_scene(expected,actual)
        actual=deepcopy(expected)
        obj=next(iter(actual['current_state'].values()))
        obj['isOpen']=not obj.get('isOpen',False)
        with self.assertRaises(AssertionError):assert_scene(expected,actual)

    def test_clean_replay_accepts_only_empty_task_state_schema_difference(self):
        from rerender import assert_scene
        source=ROOT/'05_Experiment/results/native-migration-dependency_doors-371892/episodes/dependency_doors_L1_seed42/privileged.jsonl'
        with source.open() as file:expected=json.loads(file.readline())['state']
        self.assertNotIn('task_state',expected)
        actual=deepcopy(expected);actual['task_state']={}
        assert_scene(expected,actual)
        actual['task_state']={'changed':True}
        with self.assertRaisesRegex(AssertionError,'Task state changed'):assert_scene(expected,actual)
        actual=deepcopy(expected);actual['progress']+=1
        with self.assertRaisesRegex(AssertionError,'Task history changed'):assert_scene(expected,actual)
        actual=deepcopy(expected);actual['events'].append({'event':'unrecorded'})
        with self.assertRaisesRegex(AssertionError,'Task history changed'):assert_scene(expected,actual)

    def test_connect_verifies_served_identity_before_changing_model(self):
        with patch('session.urlopen', return_value=io.BytesIO(b'{"data":[{"id":"Qwen3.8-27B"}]}')):
            code, result = self.request('/api/model', {'model':'Qwen3.8-27B','url':'http://127.0.0.1:8000/v1'})
        self.assertEqual(code,200)
        self.assertTrue(result['model_verified'])
        with patch('session.urlopen', return_value=io.BytesIO(b'{"data":[{"id":"other"}]}')):
            code, result = self.request('/api/model', {'model':'wrong','url':'http://127.0.0.1:9000/v1'})
        self.assertEqual(code,400)
        self.assertEqual(self.session.model,'Qwen3.8-27B')

    def test_worker_publishes_inputs_and_outputs_for_read_only_viewer(self):
        self.session.close()
        self.session = Session(self.directory.name, 'http://127.0.0.1:8000/v1', 'Qwen3.8-27B', transport=lambda payload:{
            'model':'Qwen3.8-27B','choices':[{'message':{'content':'{"action":"SUBMIT_SHAPE","candidate_id":"candidate_1"}'}}]})
        self.reset_shape()
        self.session.infer(True)
        viewer = ObservedSession(self.directory.name)
        state = viewer.state()
        self.assertEqual(state['step'],1)
        self.assertIn('image_streams',state['model_input'])
        self.assertIn('SUBMIT_SHAPE',state['model_output'])
        self.assertTrue(viewer.capabilities()['read_only'])
        self.assertEqual(len(state['events']),3)
        before = (Path(self.directory.name)/'run_metadata.json').read_bytes()
        cls = handler(viewer,lambda:None)
        request = object.__new__(cls); output=[]
        request.reply=lambda value,code=200:output.append((code,value))
        for operation in ('reset','action','infer','model'):
            request.perform(operation,{})
            self.assertEqual(output[-1][0],409)
        viewer.close()
        self.assertEqual(before,(Path(self.directory.name)/'run_metadata.json').read_bytes())

    def test_headless_rollout_runs_without_web_server(self):
        import rollout
        run = Path(self.directory.name)/'worker'
        def worker(*args):
            return Session(*args,transport=lambda payload:{'choices':[{'message':{
                'content':'{"action":"SUBMIT_SHAPE","candidate_id":"candidate_1"}'}}]})
        with patch('rollout.Session',side_effect=worker),patch('sys.argv',['rollout.py','--run-dir',str(run),'--budget','5']):
            rollout.main()
        self.assertEqual(json.loads((run/'run_metadata.json').read_text())['slurm']['state'],'COMPLETED')
        self.assertTrue(ObservedSession(run).state()['terminated'])

    def test_model_proposal_and_execution_use_only_public_inputs(self):
        captured = []
        def model(payload):
            captured.append(payload)
            return {"model": "test-model", "choices": [{"message": {"content": '{"action":"CAMERA_RIGHT"}'}}]}
        self.session.close()
        self.session = Session(self.directory.name, "http://127.0.0.1:8000/v1", "test-model", transport=model)
        self.reset_shape()
        result = self.session.infer(False)
        self.assertFalse(result["executed"])
        self.assertEqual(self.session.adapter.steps, 0)
        result = self.session.infer(True)
        self.assertTrue(result["executed"])
        self.assertEqual(result["state"]["policy"], "test-model")
        self.assertEqual(self.session.adapter.steps, 1)
        text = captured[-1]["messages"][0]["content"][0]["text"]
        for forbidden in ("ground_truth", "privileged_state", '"evaluation"', '"correct"', 'presentation_phase_notes'):
            self.assertNotIn(forbidden, text)
        self.assertTrue(any(item["type"] == "image_url" for item in captured[-1]["messages"][0]["content"]))

    def test_model_identity_and_bad_json_do_not_act(self):
        self.session.close()
        self.session = Session(self.directory.name, "http://127.0.0.1/v1", "configured", transport=lambda _: {
            "model": "different", "choices": [{"message": {"content": '{}'}}]})
        self.reset_shape()
        with self.assertRaisesRegex(ValueError, "different model"):
            self.session.infer(True)
        self.assertEqual(self.session.adapter.steps, 0)
        self.session.transport = lambda _: {"choices": [{"message": {"content": 'not JSON'}}]}
        with self.assertRaises(ValueError):
            self.session.infer(True)
        self.assertEqual(self.session.adapter.steps, 0)

    def test_native_launch_requires_allocation(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "Slurm"):
            TaskAdapter("native", "missing_object", 0, 1, 20, "FloorPlan1")

    def test_public_native_actions_reject_private_numbered_references(self):
        adapter = object.__new__(TaskAdapter)
        adapter.kind = 'native'
        with self.assertRaisesRegex(ValueError, 'visual references'):
            adapter.step({'action':'PickupObject','object_id':'object_001'})

    def test_delivery_snapshot_preserves_native_fields_and_views(self):
        state = json.loads((ROOT / "05_Experiment/results/interaction-demo-345795/latest_state.json").read_text())
        adapter = object.__new__(DeliveryAdapter)
        adapter.steps, adapter.api = 7, "VIEW_ORDERS()"
        result = adapter.normalize(state)
        self.assertEqual([view["name"] for view in result["views"]], ["Global map", "Local map", "First-person RGB"])
        self.assertEqual(result["observation"]["battery_pct"], state["battery_pct"])
        self.assertEqual(result["observation"]["order_pool"], state["order_pool"])
        self.assertIsNone(result["evaluation"])
        self.assertEqual(result["environment"], "simworld_delivery")

    def request(self, path, value, origin="http://localhost"):
        cls = handler(self.session, lambda: None)
        request = object.__new__(cls)
        body = json.dumps(value).encode()
        request.path = path
        request.headers = {"Origin": origin, "Host": "localhost", "Content-Type": "application/json", "Content-Length": str(len(body))}
        request.rfile = io.BytesIO(body)
        output = []
        request.reply = lambda value, code=200: output.append((code, value))
        request.do_POST()
        return output[0]

    def test_http_reset_step_and_cross_origin(self):
        code, state = self.request("/api/reset", {"environment": "shape3d", "task": "shape_identification"})
        self.assertEqual(code, 200)
        code, state = self.request("/api/action", {"action": {"action": "CAMERA_RIGHT"}})
        self.assertEqual(code, 200)
        self.assertEqual(state["step"], 1)
        code, _ = self.request("/api/action", {}, origin="http://another-host")
        self.assertEqual(code, 403)
        self.assertEqual(self.session.adapter.steps, 1)

    def test_http_serializes_operations(self):
        self.session.lock.acquire()
        try:
            code, _ = self.request("/api/reset", {})
            self.assertEqual(code, 409)
        finally:
            self.session.lock.release()

    def test_selected_task_must_match_running_environment(self):
        self.reset_shape()
        code, _ = self.request("/api/action", {"action": {"action": "CAMERA_RIGHT"}, "expected_environment": "simworld_delivery", "expected_task": "deliverybench"})
        self.assertEqual(code, 400)
        self.assertEqual(self.session.adapter.steps, 0)

    def test_stop_is_available_during_long_operation(self):
        self.session.lock.acquire()
        try:
            code, result = self.request("/api/stop", {})
            self.assertEqual(code, 200)
            self.assertEqual(result["status"], "stopping")
        finally:
            self.session.lock.release()


if __name__ == "__main__":
    unittest.main(verbosity=2)
