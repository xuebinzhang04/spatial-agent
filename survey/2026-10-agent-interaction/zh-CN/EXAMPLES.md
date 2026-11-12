# 十二种交互任务机制

以下是**本项目的任务构造实例**，并非复现相关基准的官方任务，也不表示已运行这些系统。[文献矩阵](LITERATURE.md)区分阅读深度与项目推论。

| 机制与来源 | 直观交互实例 | 生成与验证 | 对 SpatialAgent 的启示 |
| --- | --- | --- | --- |
| 1. 终端修复——[TMax](https://arxiv.org/abs/2606.23321)、[SWE-Gym](https://arxiv.org/abs/2412.21139) | 检查格式不一的 CSV → 合并 → 测试 → 修复错误 → 输出汇总。 | 采样表结构、错误和依赖；检查结果、边界例及输入文件未被改动。 | 世界状态与执行反馈比自报完成更重要。 |
| 2. 跨应用绑定——[AppWorld](https://arxiv.org/abs/2407.18901)、[AutoForge](https://arxiv.org/abs/2512.22857) | 查参会者 → 匹配联系人 → 建清单 → 将清单 ID 写入事件。 | 构造类型化依赖 DAG；检查终态与受保护状态，接受不同合法调用。 | 钥匙、门、物体和订单也需要跨动作绑定。 |
| 3. 澄清与双向控制——[ToolSandbox](https://arxiv.org/abs/2408.04682)、[τ²-Bench](https://arxiv.org/abs/2506.07982) | 搜索两位 Alex → 向用户确认 → 设置权限 → 发给正确收件人。 | 区分隐藏、用户掌握和工具状态；验证前置条件、收件人及重复运行可靠性。 | 询问也可成为空间行动；V1 尚无模拟用户。 |
| 4. 动态事件——[ARE](https://arxiv.org/abs/2509.17158)、[Gaia2](https://arxiv.org/abs/2602.11964) | 会议时间变化 → 重查冲突 → 更新预订。 | 固定事件时间与可见性；按最新约束检查终态及过时操作。 | 道路封闭应迫使 Agent 重新观察和规划。 |
| 5. 网页／桌面产物——[WebArena](https://arxiv.org/abs/2307.13854)、[OSWorld](https://arxiv.org/abs/2404.07972) | 筛选订单 → 导出 CSV → 计算总额 → 附到项目页面。 | 重置应用和文件；检查产物内容及附件关系，而非回答文本。 | 比较时固定动作粒度和预算。 |
| 6. 隐藏规则实验——[ScienceWorld](https://arxiv.org/abs/2203.07540)、[TextWorld](https://arxiv.org/abs/1806.11532) | 获取样品和仪器 → 配置实验 → 观察导电性 → 选材料。 | 采样隐藏属性及可重复的测量规则；验证合法实验与结果。 | 信号修复环境实现了该机制：必须到场检查隐藏故障类型，再匹配工具修复。 |
| 7. 指令组合——[BabyAI](https://arxiv.org/abs/1810.08272)、[gSCAN](https://arxiv.org/abs/2003.05161) | 找钥匙 → 开门 → 把红物体放在蓝物体左侧。 | 将目标语法编译为状态规则；保留组合，验证次序与几何关系。 | 当前环境分别隔离机制；该跨环境任务链仍是后续工作。 |
| 8. 程序生成规则——[Procgen](https://arxiv.org/abs/1912.01588)、[XLand](https://arxiv.org/abs/2107.12808) | 学习钥匙与门的对应关系，再在新地图中测试变化后的规则。 | 分别变化布局、实体、规则和目标，并审计各类保留测试。 | 新随机种子本身不代表新能力。 |
| 9. 共享资源——[Crafter](https://arxiv.org/abs/2109.06780)、[Craftax](https://arxiv.org/abs/2402.16801) | 收集 → 制作工具 → 采矿，同时维持生存资源。 | 强制配方与资源守恒；分别报告完整链成功率和成就覆盖率。 | 配送子任务间的电量、时间、容量必须共享。 |
| 10. 自动课程——[PAIRED](https://arxiv.org/abs/2012.02096)、[ACCEL](https://arxiv.org/abs/2203.01302)、[EnvGen](https://arxiv.org/abs/2403.12014) | 发现二层依赖失败 → 多采样已验证二层任务 → 在固定保留集重测。 | 分开记录准入失败、策略失败和学习进展；变异后重新验证。 | 先确认任务有效，再调能力参数。 |
| 11. 主动空间信念——[Theory of Space](https://arxiv.org/abs/2602.07055)、[VAGEN](https://arxiv.org/abs/2510.16907) | 探索房间 → 目标移动 → 重新取证 → 送往当前位置。 | 固定状态语义、成对改变观察；任务得分与信念诊断分开。 | 符号 SVG 不是真实视觉训练；任务成功仍是主要指标。 |
| 12. DeliveryBench 扩展——[DeliveryBench](https://arxiv.org/abs/2512.19234) | 接两单 → 等备餐 → 安排取送顺序 → 充电 → 准时送达。 | 导入道路／POI；验证图移动、订单状态、容量、电量、时限和证书。 | 图规则及充电点是合成的，迁移前须与原生 UE 状态比对。 |

统一流程为**采样机制／依赖 → 实例化世界和目标程序 → 可行证书 → 受限观察轨迹 → 实测难度 → 更新课程**。生成器错误与策略失败须分开审计。

**演示：**[五环境真实证书轨迹](../../../04_AgentTask/Demos/spatial_agent/site/index.html#explorer)、[真实地图上的图轨迹](../../../04_AgentTask/Demos/spatial_agent/site/index.html#explorer)、[交互机制示意](../../../04_AgentTask/Demos/spatial_agent/site/index.html#environment-examples)，以及含 UE 录制快照的[中文会议演示归档](../../../archive/agent-interaction/zh-original/meeting-demo.html)。官方视觉实例另见 [ProcTHOR](https://procthor.allenai.org/)、[BEHAVIOR](https://behavior.stanford.edu/)、[EnvGen](https://envgen-llm.github.io/) 和 [OMNI-EPIC](https://omni-epic.vercel.app/)。本地证书依赖特权状态，图导入不是 UE 执行，机制页面仅为示意。
