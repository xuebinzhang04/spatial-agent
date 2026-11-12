"""Multi-action turns, rollout/memory history modes and withheld intermediate frames."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatial_interaction import EpisodeConfig, make_env
from spatial_interaction.agents.vlm_adapter import VLMAdapter
from spatial_interaction.core.actions import InvalidAction
from spatial_interaction.core.trajectory import replay
from spatial_interaction.runner import run_episode
from test_active_memory import ActiveMemoryTests


def scripted(*replies):
    """Fake transport returning the given replies in order and recording each request."""
    requests, queue = [], list(replies)

    def transport(payload):
        requests.append(payload)
        return {'choices':[{'message':{'content':queue.pop(0)}}]}
    return transport, requests


def images(message):
    return sum(part.get('type') == 'image_url' for part in message['content']) if isinstance(message['content'], list) else 0


def shape_config(max_steps=10):
    return EpisodeConfig(seed=7, max_steps=max_steps, task_config={'image_size':64})


class RolloutHarnessTests(unittest.TestCase):
    def run_shape(self, replies, **options):
        transport, requests = scripted(*replies)
        directory = Path(tempfile.mkdtemp(prefix='harness-test-')) / 'episode'
        report = run_episode(shape_config(), directory, 'vlm', agent_options={'transport':transport, **options})
        rows = [json.loads(line) for line in (directory / 'trajectory.jsonl').read_text().splitlines()]
        return report, requests, rows, directory

    def test_rollout_keeps_past_turns_and_presents_one_image_per_turn(self):
        first = '[{"action":"CAMERA_LEFT"},{"action":"CAMERA_RIGHT"}]'
        report, requests, rows, _ = self.run_shape([first, '{"action":"SUBMIT_SHAPE","candidate_id":"candidate_1"}'])
        self.assertEqual((report['turns'], report['environment_steps']), (2, 3))
        self.assertEqual(report['harness'], {'history':'rollout', 'max_actions':3, 'image_window':None})
        second = requests[1]['messages']
        self.assertEqual([m['role'] for m in second], ['system', 'user', 'assistant', 'user'])
        self.assertEqual(second[2]['content'], first)  # raw reply stays in context
        self.assertEqual([images(m) for m in second], [0, 1, 0, 1])
        feedback = second[3]['content'][0]['text']
        self.assertIn('1. {"action": "CAMERA_LEFT"} -> ok', feedback)
        self.assertTrue(feedback.endswith('Observation:'))  # Shape3D has no view references
        self.assertNotIn('Task instruction', feedback)  # unchanged phase: instruction not repeated
        self.assertEqual([r.get('withheld', False) for r in rows], [False, True, False, False])

    def test_image_window_drops_older_images_but_keeps_text(self):
        replies = ['{"action":"CAMERA_LEFT"}', '{"action":"CAMERA_RIGHT"}',
                   '{"action":"SUBMIT_SHAPE","candidate_id":"candidate_1"}']
        _, requests, _, _ = self.run_shape(replies, image_window=1)
        last = requests[2]['messages']
        self.assertEqual(sum(images(m) for m in last), 1)
        self.assertIn('[earlier image omitted]', json.dumps(last))
        self.assertEqual(sum(m['role'] == 'user' for m in last), 3)

    def test_memory_mode_is_a_single_stateless_message(self):
        replies = ['[{"action":"CAMERA_LEFT","memory":"left side seen"}]',
                   '{"action":"SUBMIT_SHAPE","candidate_id":"candidate_1"}']
        _, requests, _, _ = self.run_shape(replies, history='memory')
        second = requests[1]['messages']
        self.assertEqual(len(second), 1)
        self.assertEqual(images(second[0]), 1)
        self.assertIn('left side seen', second[0]['content'][0]['text'])
        self.assertIn('last_turn', second[0]['content'][0]['text'])

    def test_failed_action_ends_turn_and_reports_skipped(self):
        replies = ['[{"action":"NOT_AN_ACTION"},{"action":"CAMERA_LEFT"}]',
                   '{"action":"SUBMIT_SHAPE","candidate_id":"candidate_1"}']
        report, requests, rows, _ = self.run_shape(replies)
        self.assertEqual(report['environment_steps'], 2)
        feedback = requests[1]['messages'][3]['content'][0]['text']
        self.assertIn('failed:', feedback)
        self.assertIn('1 later action(s) were not executed', feedback)
        self.assertNotIn('withheld', rows[1])  # a failed action ends the turn and is presented

    def test_extra_actions_are_truncated(self):
        adapter = VLMAdapter(transport=lambda p:None, max_actions=2)
        actions, error, truncated = adapter.parse_turn('[{"action":"A"},{"action":"B"},{"action":"C"}]')
        self.assertEqual((len(actions), error, truncated), (2, None, True))
        self.assertEqual(adapter.parse_turn('not json')[0], [{'action':'INVALID'}])

    def test_rollout_trajectory_with_withheld_rows_replays(self):
        replies = ['[{"action":"CAMERA_LEFT"},{"action":"CAMERA_YAW_RIGHT"}]',
                   '{"action":"SUBMIT_SHAPE","candidate_id":"candidate_1"}']
        _, _, _, directory = self.run_shape(replies)
        env = make_env(shape_config())
        try:
            self.assertTrue(replay(env, directory)['verified'])
        finally:
            env.close()


class WithheldViewTests(unittest.TestCase):
    def test_withheld_frame_is_not_referable_or_evidence(self):
        env = ActiveMemoryTests.env(ActiveMemoryTests())
        env.step({'action':'INVALID'})  # still renders and registers frame 1
        env.withhold_view(env.steps)
        with self.assertRaises(InvalidAction):
            env.resolve_reference('view:1:0.2:0.2')
        self.assertEqual(env.resolve_reference('view:0:0.2:0.2'), 'mug')
        env.step({'action':'MEMORIZE'})
        self.assertNotIn(1, env.memorization_view_steps)

    def test_phase_change_ends_the_turn(self):
        from spatial_interaction.runner import run_turn
        env = ActiveMemoryTests.env(ActiveMemoryTests())
        *_, results = run_turn(env, [{'action':'MEMORIZE'}, {'action':'INVALID'}])
        self.assertEqual(env.steps, 1)
        self.assertEqual(env.phase, 'detection')
        self.assertEqual(results[-1], {'skipped':1})


if __name__ == '__main__':
    unittest.main(verbosity=2)
