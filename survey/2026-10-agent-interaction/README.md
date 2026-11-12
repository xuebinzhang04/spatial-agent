# Agent Interaction Research Foundation

[中文版](zh-CN/README.md)

**Question:** Can cheap, controlled interaction training improve spatial decisions and transfer to external benchmarks? The pipeline is **Environment (state, dynamics, observations) → Task (API-grounded ability, generator, verifier) → Agent (Harness and RL) → independent evaluation**. Spatial QA and interaction success are separate outcomes.

| Read | Purpose |
| --- | --- |
| [LITERATURE.md](LITERATURE.md) · [sources.json](sources.json) | 64 primary sources, reading depth, motivation, construction, evaluation, and project implications |
| [RL_TAXONOMY.md](RL_TAXONOMY.md) · [EXAMPLES.md](EXAMPLES.md) | Environment/RL taxonomy and 12 concrete mechanism examples |
| [DESIGN.md](DESIGN.md) · [BRAINSTORM.md](BRAINSTORM.md) | Shared generation/verification protocol and three testable research directions |
| [TRANSFER_PROTOCOL.md](TRANSFER_PROTOCOL.md) | Internal splits, ablations, external transfer, and metrics |
| [INTERACTION_PRESENTATION.md](INTERACTION_PRESENTATION.md) | Research presentation mechanisms from DeliveryBench, VLM-3R, and relevant PHAI projects; current interaction stories and proposed diagnostic experiments |
| [Interaction Lab](../../04_AgentTask/CustomAgentTasks/interaction_lab/README.md) | Five executable environments, API, commands, evidence, and limitations |
| [Case gallery](../../04_AgentTask/CustomAgentTasks/interaction_lab/CASE_GENERATION.md) | One actual L3 task per family and the exact case-generation pipeline |

The maintained [unified research demo](../../04_AgentTask/Demos/spatial_agent/README.md) now imports these original trajectories alongside native examples and provides simulator/task/model selectors, definitions, variables, recorded media and online APIs. Source HTML and historical validation are preserved under `archive/agent-interaction/legacy-demos/`.

**Demos:** [five-family traces](../../04_AgentTask/Demos/spatial_agent/site/index.html#explorer) contain 15 verified construction certificates from the current five default families; [DeliveryBench graph trace](../../04_AgentTask/Demos/spatial_agent/site/index.html#explorer) imports a real road/POI map; [mechanism illustrations](../../04_AgentTask/Demos/spatial_agent/site/index.html#environment-examples) are teaching examples. [Validation](../../archive/agent-interaction/legacy-demos/symbolic/validation.json) records simulated-DOM checks and file hashes. No real-browser check is claimed. The earlier Chinese meeting viewer is preserved in [archive](../../archive/agent-interaction/zh-original/README.md).

This is a construction-stage survey, not an exhaustive review or a reproduction of reported results. The original five prototype environments are Memory Maze, Dependency Doors, Relational Rearrangement, Dynamic Replanning, and Signal Repair. DeliveryBench resource delivery is an optional transfer baseline. The legacy demos above are symbolic/graph worlds; subsequent native migrations and their actual status are documented in [CustomAgentTasks](../../04_AgentTask/CustomAgentTasks/README.md). The current [research presentation](../../04_AgentTask/Demos/spatial_agent/PRESENTATION.md) uses real AI2-THOR and procedural geometry recordings. Feasibility witnesses do not establish observation-limited policy success. Structural levels are not measured model difficulty. RL and transfer effects remain untested.
