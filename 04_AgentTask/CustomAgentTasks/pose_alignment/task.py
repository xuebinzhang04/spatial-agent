"""pose_alignment: 独立控制相机和物体，将目标绕三个轴旋转以匹配参考姿态。"""
from pathlib import Path
import sys

_frameworks = Path(__file__).resolve().parents[2] / 'TaskFrameworks'
if str(_frameworks) not in sys.path:
    sys.path.insert(0, str(_frameworks))
from task_support import SingleTask

TASK = SingleTask(Path(__file__).resolve().parent)
TASK_NAME = 'pose_alignment'
generate = TASK.generate
make_env = TASK.make_env
evaluate = TASK.evaluate
