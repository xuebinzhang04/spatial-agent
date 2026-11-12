"""missing_object: 记忆真实 iTHOR 场景，经过隐藏干预后识别被停用的物体及原位置。"""
from pathlib import Path
import sys

_frameworks = Path(__file__).resolve().parents[2] / 'TaskFrameworks'
if str(_frameworks) not in sys.path:
    sys.path.insert(0, str(_frameworks))
from task_support import SingleTask

TASK = SingleTask(Path(__file__).resolve().parent)
TASK_NAME = 'missing_object'
generate = TASK.generate
make_env = TASK.make_env
evaluate = TASK.evaluate
