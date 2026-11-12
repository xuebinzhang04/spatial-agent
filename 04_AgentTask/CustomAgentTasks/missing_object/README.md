# missing_object — 物体缺失检测

记忆真实 iTHOR 场景，经过隐藏干预后识别被停用的物体及原位置。

此目录只定义和运行 `missing_object`。任务身份及默认参数在 `task.json`；Python 入口为 `task.py`，命令行入口为 `run.py`。共享模拟器、生成算法和记录工具来自 `../../TaskFrameworks/spatial_interaction/`。

从 SpatialAgent 根目录运行，输出目录必须是新的：

```bash
task_python=/gpfs/scrubbed/xuebiz/SpatialAgent/02_Simulator/AI2THOR/runtime/python/bin/python
export SPATIAL_THOR_SOFTWARE=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
export LP_NUM_THREADS=2
export XDG_CONFIG_HOME=/tmp/spatial-task-unity
export XDG_CACHE_HOME=/gpfs/scrubbed/xuebiz/SpatialAgent/.cache
mkdir -p "$XDG_CONFIG_HOME"
"$task_python" 04_AgentTask/CustomAgentTasks/missing_object/run.py run --policy oracle --debug --output 05_Experiment/results/missing_object-demo
"$task_python" 04_AgentTask/CustomAgentTasks/missing_object/run.py generate --count 2 --output 05_Experiment/results/missing_object-corpus/development.jsonl
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
cd 04_AgentTask/CustomAgentTasks/missing_object
"$task_python" -m unittest discover -s tests -v
```

`benchmarks/` 中每个 split 只包含本任务的默认配置。模型适配、奖励、回放、缺失物体停用语义及渲染限制见 [共享后端说明](../../TaskFrameworks/spatial_interaction/README.md)。
默认基准来自重构前验证通过的 Stage 1 记录；迁移后的回归结果见 CustomAgentTasks/README.md。


## Active memorization protocol

New generated configurations use `memory_protocol: active_scan_v1`. The target starts outside the initial view; its identity stays private. The public instruction requests an active room survey and cumulative object/support notes. `MEMORIZE` explicitly commits the notebook and triggers the hidden change. Reaching the memory deadline without commitment truncates the episode without an intervention. Detection exposes the frozen notebook, remaining budget and action feedback. Only navigation/camera actions are available before submission, keeping the environment's hidden change distinct from agent manipulation.

Strict success checks the changed object, original support and their actual pre-change visibility records. Missing-only identity F1 is a diagnostic, not the active protocol's success reward. The task API and environment use the same answer verifier. Legacy benchmark JSON files retain their previous semantics; they are historical baselines, not the revised case suite.

The active protocol is development code with synthetic-event tests. Turning the camera away establishes only an initial-view gap. It does not prove necessary travel, complete room coverage, an adequate VLM budget or a memory benefit. The fresh 24-case batch, native/model execution status and review criteria are tracked in the [case review](../../../05_Experiment/results/active-memory-review-20261025/CASE_REVIEW.md) and [expectation registry](../expectations.json).
