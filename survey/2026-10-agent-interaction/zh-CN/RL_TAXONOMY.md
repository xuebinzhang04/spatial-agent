# Agent RL 环境与任务分类

收集的 64 项是相关工作，不是 64 个已证明可用于 RL 训练的环境。统一接口、评估基准、任务生成器和 RL 算法是不同产物。来源与阅读深度见[文献综述](LITERATURE.md)，具体流程见[任务实例](EXAMPLES.md)。以下分类是本项目的分析，不代表文献已形成共识。

## 可执行世界的选择

| 类型 | 动机 | 构造与状态 | 验证要求 | 项目用途与实例 |
| --- | --- | --- | --- | --- |
| 确定性代码／数据库 | 重置廉价、转移稳定、事实精确 | API 前置条件、实体状态、可执行程序 | 终态谓词、时序、保护约束、负例轨迹 | V1 机制层；AppWorld、ToolSandbox、AutoForge |
| 学习式／语言模型模拟器 | 扩展接口和任务种类 | 生成响应或世界函数；部分事实由模型推断 | 一致性、可重复性、非法状态、模拟器错误归因 | 辅助生成，不能取代确定性评分事实；CompoWorld |
| 真实应用／容器 | 保留真实工具行为和交付物 | 网站、桌面、文件、仓库、服务快照 | 执行／产物评估、重置和运行成本 | 工作流迁移；WebArena、OSWorld、SWE-Gym |
| 具身物理／游戏引擎 | 感知、动作和长期世界交互 | 场景、碰撞、物体属性、规则、资源 | 物理有效性、可达性、任务状态、行为和成本 | UE／Habitat／THOR；ProcTHOR、BEHAVIOR、DeliveryBench |
| 轻量程序化世界 | 高吞吐、可控因素、组合保留测试 | 网格、文本逻辑、参数化游戏 | 精确目标、规则保留、学习曲线 | 构造阶段优先；BabyAI、TextWorld、Procgen、XLand-MiniGrid |

## 从能力要求到训练任务

| 能力 | 生成变量 | 必需交互 | 主要评估 | 代表来源 |
| --- | --- | --- | --- | --- |
| 执行与调试 | 文件、错误、依赖、工具链 | 检查、运行、读反馈、修复 | 正确产物、测试、执行成本 | [TMax](https://arxiv.org/abs/2606.23321)、[SWE-smith](https://arxiv.org/abs/2504.21798) |
| 跨工具规划 | 类型绑定、DAG、分支 | 将前次输出用于后续 API | 状态目标、受保护事实、顺序 | [AutoForge](https://arxiv.org/abs/2512.22857)、[AppWorld](https://arxiv.org/abs/2407.18901) |
| 对话与协作 | 歧义、用户信息、策略、共享控制 | 澄清、操作工具、协调 | 环境成功率和重复可靠性 | [τ²-Bench](https://arxiv.org/abs/2506.07982) |
| 主动空间理解 | 可见范围、记忆、参照系、变化 | 探索、整合、修正、运用 | 任务成功和结构化诊断 | [Theory of Space](https://arxiv.org/abs/2602.07055)、[VAGEN](https://arxiv.org/abs/2510.16907) |
| 长程资源管理 | 配方／订单、备餐时间、容量、电量 | 获取、满足依赖、调度 | 完整任务成功率、资源与时间约束 | [Crafter](https://arxiv.org/abs/2109.06780)、[DeliveryBench](https://arxiv.org/abs/2512.19234) |
| 系统性组合 | 规则、谓词、实体绑定、目标程序 | 解决已见原语的未见组合 | 保留结构上的成功率 | [BabyAI](https://arxiv.org/abs/1810.08272)、[gSCAN](https://arxiv.org/abs/2003.05161)、[XLand](https://arxiv.org/abs/2107.12808) |

## 不同研究产物的作用

- 多轮／RL 训练与诊断：TMax、RAGEN、VAGEN、AgentGym-RL、SWE-Gym、AutoForge、EnvGen；其训练方式并非同一种算法。
- 任务与环境评估：AppWorld、ToolSandbox、τ／τ²、WebArena、OSWorld、Theory of Space、DeliveryBench。可执行评分使其可能成为轨迹来源，但不意味着每篇原论文都做了 RL 训练。
- 程序生成与组合分布：BabyAI、TextWorld、Procgen、XLand、Reasoning Gym。静态问题生成器不一定是持续交互世界。
- 课程／环境设计：PAIRED、PLR、ACCEL、ALP-GMM、POET、EnvGen、OMNI-EPIC。需与固定随机分布比较，测试集不能参与课程选择。
- 接口／平台：GEM、OpenEnv、BrowserGym、ARE。它们便于替换环境和 Harness，任务正确性与组合性仍需单独保证。

## 当前设计决定

先在低成本、可重置、可审计的符号层验证能力需求；同步生成任务描述、世界和评分语义，接受满足约束的不同解。分别评估固定任务、新布局、未见组合、干预和新后端。对生成错误、可行性失败、策略失败及学习进展分别记录反馈。

扩大训练前先测步数／token、运行时间、重置成本和吞吐。当前 Python 图环境不声称具备 JAX 或向量化吞吐。外部迁移比较必须固定感知接口、动作粒度、Harness 和预算；见[迁移协议](TRANSFER_PROTOCOL.md)。
