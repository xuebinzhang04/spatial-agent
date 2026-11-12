# Agent Interaction 研究基础

[English version](../README.md)

**核心问题：**低成本、可控的交互训练，能否提升空间决策能力并迁移至外部基准？研究链条为 **Environment（状态、动力学、观察）→ Task（基于 API 的能力要求、生成器、验证器）→ Agent（Harness 与 RL）→ 独立评估**。空间问答得分与交互任务成功率应分别衡量。

| 文档 | 内容 |
| --- | --- |
| [文献综述](LITERATURE.md) · [文献数据](../sources.json) | 64 项一手资料、阅读深度、研究动机、环境构造、评估及项目启示 |
| [RL 分类](RL_TAXONOMY.md) · [任务实例](EXAMPLES.md) | 环境与训练方法分类、12 个具体交互机制 |
| [设计协议](DESIGN.md) · [研究构想](BRAINSTORM.md) | 生成与验证规则、三个可证伪研究方向 |
| [迁移实验协议](TRANSFER_PROTOCOL.md) | 内部划分、消融、外部迁移和统计指标 |
| [Interaction Lab](../../../04_AgentTask/CustomAgentTasks/interaction_lab/zh-CN/README.md) | 五个可运行环境、API、命令、验证证据及限制 |
| [实例展示](../../../04_AgentTask/CustomAgentTasks/interaction_lab/zh-CN/CASE_GENERATION.md) | 每类一个真实 L3 任务及完整的 case 生成流程 |

**演示：**[五类环境轨迹](../../../04_AgentTask/Demos/spatial_agent/site/index.html#explorer)展示当前五类默认环境的 15 条通过验证的构造证书；[DeliveryBench 路网轨迹](../../../04_AgentTask/Demos/spatial_agent/site/index.html#explorer)使用真实道路与 POI 文件；[交互机制演示](../../../04_AgentTask/Demos/spatial_agent/site/index.html#environment-examples)用于教学。[验证记录](../../../archive/agent-interaction/legacy-demos/symbolic/validation.json)包含模拟 DOM 检查及文件哈希；尚未进行真实浏览器检查。更早的中文会议演示存于[归档](../../../archive/agent-interaction/zh-original/README.md)。

本综述服务于环境构造阶段，并非穷尽式系统综述，也没有复现引用论文的结果。默认五类环境为记忆迷宫、依赖门、关系重排、动态重规划和信号修复。DeliveryBench 资源配送仅是可选迁移基线，不属于默认环境。这些仍是符号／图世界，不是新建 3D 场景。构造证书证明在特权状态下存在可行轨迹，不保证受限观察策略能在相同预算内完成。L1–L5 是结构参数，不是实测模型难度。模型 RL、UE 状态一致性和迁移收益仍需实验。
