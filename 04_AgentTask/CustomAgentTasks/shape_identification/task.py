"""shape_identification: 主动移动相机，从候选参考视图中识别目标多方块形状。"""
from pathlib import Path
import sys

_frameworks = Path(__file__).resolve().parents[2] / 'TaskFrameworks'
if str(_frameworks) not in sys.path:
    sys.path.insert(0, str(_frameworks))
from task_support import SingleTask

TASK = SingleTask(Path(__file__).resolve().parent)
TASK_NAME = 'shape_identification'
generate = TASK.generate
make_env = TASK.make_env
evaluate = TASK.evaluate
