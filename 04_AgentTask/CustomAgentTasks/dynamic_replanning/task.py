"""dynamic_replanning: 发现路径封闭后重新规划并完成有序导航目标。"""
from pathlib import Path
import sys

_frameworks = next(parent / 'TaskFrameworks' for parent in Path(__file__).resolve().parents
                   if (parent / 'TaskFrameworks' / 'task_support.py').is_file())
if str(_frameworks) not in sys.path:
    sys.path.insert(0, str(_frameworks))
from task_support import SingleTask

TASK = SingleTask(Path(__file__).resolve().parent)
TASK_NAME = 'dynamic_replanning'
generate = TASK.generate
make_env = TASK.make_env
evaluate = TASK.evaluate
