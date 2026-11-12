"""dependency_doors: 按依赖顺序收集钥匙、解锁门并到达出口。"""
from pathlib import Path
import sys

_frameworks = next(parent / 'TaskFrameworks' for parent in Path(__file__).resolve().parents
                   if (parent / 'TaskFrameworks' / 'task_support.py').is_file())
if str(_frameworks) not in sys.path:
    sys.path.insert(0, str(_frameworks))
from task_support import SingleTask

TASK = SingleTask(Path(__file__).resolve().parent)
TASK_NAME = 'dependency_doors'
generate = TASK.generate
make_env = TASK.make_env
evaluate = TASK.evaluate
