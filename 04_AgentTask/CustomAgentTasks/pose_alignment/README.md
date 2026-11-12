# pose_alignment — 三维姿态对齐

独立控制相机和物体，将目标绕三个轴旋转以匹配参考姿态。

此目录只定义和运行 `pose_alignment`。任务身份及默认参数在 `task.json`；Python 入口为 `task.py`，命令行入口为 `run.py`。共享模拟器、生成算法和记录工具来自 `../../TaskFrameworks/spatial_interaction/`。

从 SpatialAgent 根目录运行，输出目录必须是新的：

```bash
task_python=/gpfs/scrubbed/xuebiz/conda/envs/spatialagent/bin/python
"$task_python" 04_AgentTask/CustomAgentTasks/pose_alignment/run.py run --policy oracle --debug --output 05_Experiment/results/pose_alignment-demo
"$task_python" 04_AgentTask/CustomAgentTasks/pose_alignment/run.py generate --count 2 --output 05_Experiment/results/pose_alignment-corpus/development.jsonl
```

Python API（从本目录执行）：

```python
from task import generate, make_env, evaluate

specification = generate()
env = make_env(specification)
try:
    observation, info = env.reset()
    # 使用当前任务的动作协议执行 env.step(action)。
finally:
    env.close()
```

接口拒绝其他任务的配置、命令行任务选择和混入其他任务的 corpus/episode。

```bash
cd 04_AgentTask/CustomAgentTasks/pose_alignment
"$task_python" -m unittest discover -s tests -v
```

`benchmarks/` 中每个 split 只包含本任务的默认配置。模型适配、奖励、回放、缺失物体停用语义及渲染限制见 [共享后端说明](../../TaskFrameworks/spatial_interaction/README.md)。
默认基准来自重构前验证通过的 Stage 1 记录；迁移后的回归结果见 CustomAgentTasks/README.md。
