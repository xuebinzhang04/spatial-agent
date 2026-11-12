"""moved_object: 记忆真实 iTHOR 场景，识别被移动物体及其前后 receptacle。"""
from pathlib import Path
import sys

_frameworks = Path(__file__).resolve().parents[2] / 'TaskFrameworks'
if str(_frameworks) not in sys.path:
    sys.path.insert(0, str(_frameworks))
from task_support import SingleTask

TASK = SingleTask(Path(__file__).resolve().parent)
TASK_NAME = 'moved_object'
generate = TASK.generate
make_env = TASK.make_env
evaluate = TASK.evaluate
