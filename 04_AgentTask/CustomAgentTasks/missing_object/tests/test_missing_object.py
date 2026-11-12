"""Task identity, generation and task-boundary regression."""
import importlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
task = importlib.import_module('missing_object')


class TaskTests(unittest.TestCase):
    def test_definition_and_seeded_generation(self):
        self.assertEqual(task.TASK_NAME, 'missing_object')
        self.assertEqual(task.TASK.directory.name, task.TASK_NAME)
        a, b = task.generate(), task.generate()
        self.assertEqual(a.to_dict(), b.to_dict())
        actual = getattr(a, 'family', getattr(a, 'task_type', None))
        self.assertEqual(actual, task.TASK_NAME)

    def test_rejects_foreign_task_before_simulator_start(self):
        class Foreign:
            family = task_type = 'another_task'
        with self.assertRaises(ValueError):
            task.make_env(Foreign())


if __name__ == '__main__':
    unittest.main()
