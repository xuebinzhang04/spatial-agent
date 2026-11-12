# 真实室内地标记忆与顺序重访

从 Interaction Lab 的 memory_maze 迁移到真实 AI2-THOR，使用第一人称 RGB、原生移动、转向与碰撞。保留历史名称，不宣称 iTHOR 厨房是程序生成迷宫。

训练设计见 DESIGN.md。每个 episode 包含探索与顺序重访两个阶段；MEMORIZE 或探索预算耗尽后，相机回到起点，代理须实际导航并依次 MARK 地标，最后 FINISH。目标必须可见且距离不超过 1.6 米。L1 两个地标，L2 三个地标并返回第一个，L3 四个地标并返回第一个。

## 验证

真实场景 FloorPlan1/2/3、种子 42/43/44、L1/L2/L3 的 oracle 全部成功，分别 41/107/73 步；三条回放通过。还检查了提前提交失败、非法动作与随机策略交互。证据：05_Experiment/results/native-migration-memory_maze-371491/validation.json。Oracle 读取私有导航图，只说明机械可解性，不代表 VLM 能力。

## 运行

在已分配的计算节点，从仓库根目录执行；复用已有 Python 和模拟器：

```bash
export CUDA_VISIBLE_DEVICES=''
export SPATIAL_THOR_SOFTWARE=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
export LP_NUM_THREADS=2
02_Simulator/AI2THOR/runtime/python/bin/python 04_AgentTask/CustomAgentTasks/memory_maze/run.py run --policy oracle --max-steps 360 --debug --output 05_Experiment/results/my-memory-maze
```

用 --policy random 测试随机交互；真实模型使用 --policy vlm 并预先启动本地推理服务。启动模拟器时清空 CUDA 可见设备只针对该进程，不能清空已启动模型服务的 GPU 设置。

benchmarks 下三个 JSONL 分别来自本次实际通过验证的开发、验证和测试场景。固定种子可复现，THOR 回放允许很小的光栅变化。旧符号入口备份在 archive/agent-interaction/2026-10-19-symbolic-task-entries/memory_maze；原符号引擎保留用于历史复现。没有 RL 训练或泛化结论。
