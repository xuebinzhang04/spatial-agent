# dependency_doors: 真实模拟器任务

设计、训练价值、难度与语义变化见 [DESIGN.md](DESIGN.md)。使用 AI2-THOR 原生场景、第一人称 RGB、移动与碰撞；目标通过真实场景状态和操作历史评分。观察不包含私有导航图、答案或完整对象状态。

## 已执行验证

FloorPlan1/2/3，种子 42/43/44，L1/L2/L3：3 个 oracle 成功（48/60/146 步），3 条轨迹回放通过。另有提前 FINISH 失败、非法动作和随机策略交互检查。
证据：`05_Experiment/results/native-migration-dependency_doors-371892/validation.json`。Oracle 读取私有信息，只证明物理可解性，不代表 VLM 自主成功率。benchmarks 中的配置来自实际通过的任务。

## 复现

在已分配的计算节点、仓库根目录执行：

```bash
export CUDA_VISIBLE_DEVICES=''
export SPATIAL_THOR_SOFTWARE=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
export LP_NUM_THREADS=2
02_Simulator/AI2THOR/runtime/python/bin/python 04_AgentTask/CustomAgentTasks/dependency_doors/run.py run --policy oracle --max-steps 360 --debug --output 05_Experiment/results/my-dependency_doors
```

`--policy random` 检查普通交互；`--policy vlm` 连接已启动的本地推理服务，不在每一步加载模型。模拟器进程使用 CPU 渲染；模型服务使用独立 GPU 设置。

重新提交完整验证：`python3 scripts/experiments/launch_native_task_migration.py --task dependency_doors --submit`。记录含逐帧 RGB、动作 JSONL、GIF、私有调试状态；回放允许极小的原生渲染像素误差。旧符号入口已备份，原符号引擎已退役并移入 archive/agent-interaction/legacy-demos/symbolic/。
