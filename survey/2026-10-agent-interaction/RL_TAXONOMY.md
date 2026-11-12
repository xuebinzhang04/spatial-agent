# Research Taxonomy of Agent RL Environments and Tasks

The 64 collected items are related works, not 64 environments demonstrated to train with RL. Unified interfaces, evaluation benchmarks, task generators, and RL algorithms are different outputs. See [LITERATURE.md](LITERATURE.md) for primary sources and reading depth, and [EXAMPLES.md](EXAMPLES.md) for concrete flows. The choices below are this project's analysis, not a consensus proven by the cited literature.

## Choice of executable world

| Type | Motivation | Construction and state | Verification needs | Project use and examples |
| --- | --- | --- | --- | --- |
| Deterministic code/database | Cheap reset, stable transitions, exact task facts | API preconditions, entity state, executable programs | Final predicates, order, preservation, negative traces | V1 mechanism layer; AppWorld, ToolSandbox, AutoForge |
| Learned/language-model simulator | Broader interfaces and task variety | Generated responses or world functions; some facts inferred by a model | Consistency, repeatability, illegal states, simulator error attribution | Assist generation without replacing deterministic scoring facts; CompoWorld |
| Real app/container | Preserve tool behavior and deliverables | Websites, desktops, files, repositories, service snapshots | Execution/artifact evaluators, reset and runtime cost | Workflow transfer; WebArena, OSWorld, SWE-Gym |
| Embodied physics/game engine | Perception, action, long-term world interaction | Scenes, collisions, object properties, rules, resources | Physical validity, reachability, task state, behavior, cost | UE/Habitat/THOR backends; ProcTHOR, BEHAVIOR, DeliveryBench |
| Lightweight procedural world | Throughput, controlled factors, composition holdouts | Grids, text logic, parameterized games | Exact goals, rule holdouts, learning curves | Construction-stage priority; BabyAI, TextWorld, Procgen, XLand-MiniGrid |

## From ability requirements to training tasks

| Ability | Generation variables | Required interaction | Main evaluation | Representative sources |
| --- | --- | --- | --- | --- |
| Execution/debugging | Files, errors, dependencies, toolchains | Inspect, run, read feedback, repair | Correct artifact, tests, execution cost | [TMax](https://arxiv.org/abs/2606.23321), [SWE-smith](https://arxiv.org/abs/2504.21798) |
| Cross-tool planning | Typed bindings, DAGs, branches | Use outputs in later API calls | State goals, protected facts, order | [AutoForge](https://arxiv.org/abs/2512.22857), [AppWorld](https://arxiv.org/abs/2407.18901) |
| Conversation/collaboration | Ambiguity, user information, policies, shared control | Clarify, operate tools, coordinate | Environment success and repeat reliability | [τ²-Bench](https://arxiv.org/abs/2506.07982) |
| Active spatial understanding | Visibility, memory, reference frame, change | Explore, integrate, revise, use | Task success and structured diagnostics | [Theory of Space](https://arxiv.org/abs/2602.07055), [VAGEN](https://arxiv.org/abs/2510.16907) |
| Long-horizon resources | Recipes/orders, preparation time, capacity, battery | Acquire, satisfy dependencies, schedule | Whole-task success, resource/time limits | [Crafter](https://arxiv.org/abs/2109.06780), [DeliveryBench](https://arxiv.org/abs/2512.19234) |
| Systematic composition | Rules, predicates, object bindings, goal programs | Solve unseen combinations of seen primitives | Held-out structural success | [BabyAI](https://arxiv.org/abs/1810.08272), [gSCAN](https://arxiv.org/abs/2003.05161), [XLand](https://arxiv.org/abs/2107.12808) |

## Roles of different outputs

- Multi-turn/RL training and diagnosis: TMax, RAGEN, VAGEN, AgentGym-RL, SWE-Gym, AutoForge, EnvGen. Their training forms are not one algorithm.
- Task and environment evaluation: AppWorld, ToolSandbox, τ/τ², WebArena, OSWorld, Theory of Space, DeliveryBench. Executable scoring makes them potential rollout sources; it does not mean every original paper trained with RL.
- Procedural generation and compositional distributions: BabyAI, TextWorld, Procgen, XLand, Reasoning Gym. A static question generator is not automatically a persistent interactive world.
- Curriculum/environment design: PAIRED, PLR, ACCEL, ALP-GMM, POET, EnvGen, OMNI-EPIC. Compare with a fixed random distribution and keep curricula away from tests.
- Interfaces/platforms: GEM, OpenEnv, BrowserGym, ARE. They ease swapping environments and Harnesses; correctness and composition remain task properties.

## Current design decision

Validate ability requirements first in a cheap, resettable, auditable symbolic layer. Generate descriptions, worlds, and scoring semantics together; accept alternative solutions satisfying constraints. Evaluate fixed tasks, new layouts, unseen compositions, interventions, and new backends separately. Record generation errors, feasibility failures, policy failures, and learning progress as different feedback to generator and policy.

Before scaling training, measure step/token cost, wall time, reset cost, and throughput. The current Python graph environments make no JAX/vectorized throughput claim. External transfer must hold perception interface, action granularity, Harness, and budget fixed; see [TRANSFER_PROTOCOL.md](TRANSFER_PROTOCOL.md).
