# Research Directions and Five Environments

These are testable hypotheses, not established novelty. Existing overlap includes dependency-task generation ([AutoForge](https://arxiv.org/abs/2512.22857)), belief revision ([Theory of Space](https://arxiv.org/abs/2602.07055)), and adaptive curricula ([EnvGen](https://arxiv.org/abs/2403.12014)). The contribution to test is whether controlled interaction mechanisms improve held-out spatial behavior under equal training and evaluation budgets.

| Hypothesis | Construction and comparison | Falsification test |
| --- | --- | --- |
| **H1: dependency composition** improves unseen chains. | Generate ability graph → spatial world/API preconditions → goal program and feasible certificate. Compare primitive-only, random composition, and dependency composition with equal primitive coverage, entities, and interactions. Measure whole-task success, failed stage, invalid actions, and cost. | Gain disappears after matching trajectory length/training budget or on held-out binding/depth. |
| **H2: paired interventions** improve recovery. | Hold map and goal fixed while changing heading, key position, closure, or battery. Compare no intervention, static augmentation, and during-interaction intervention. Measure paired success, repeated stale actions, and recovery delay. | Gain exists only at generator-selected event positions; fails with held-out timing/location. |
| **H3: information under resources** helps unknown-city delivery. | Cross full/local view × limited/unlimited resources × static/dynamic topology. Exploration costs time/battery but can serve multiple goals. | At equal actions, policy only charges/waits more without better discovery, routing, or completion. |

V1 supports H1 in linear doors and diagnostic tool-repair chains, H2 through counterfactual instances and one triggered road closure, and only the exploration side of H3. Resource delivery remains an optional transfer baseline. It does **not** implement a combined partial-observation delivery world, all cross-family chains, or these training comparisons.

## Five implemented families

| Family | Required interaction | Generated task and verifier | Main level controls / transfer target |
| --- | --- | --- | --- |
| Memory Maze | Explore, remember, revisit | Ordered beacon visits in a generated maze; true visit order | Map, goals, budget / Theory of Space |
| Dependency Doors | Acquire keys, unlock, cross | Room-chain barriers with enforced key preconditions | Chain depth / indoor sequences |
| Relational Rearrangement | Pick, carry, place while preserving anchors | Geometric conjunctions and touch-history preservation; `left_of+above` held out | Objects, relations, budget / RoboSpatial |
| Dynamic Replanning | Observe and reroute after closure | Checkpoint triggers edge closure; event and visits verified | Map, waypoints, budget / dynamic navigation |
| Signal Repair | Inspect hidden faults, use matching tools, preserve healthy relays, activate | Local diagnosis, tool preconditions, mixed A+B fault holdout, final preservation | Relay count, fault count, map, budget / active spatial diagnosis |

All five default environments are new symbolic graph/2D worlds. Delivery remains an optional transfer baseline that imports roads/POIs under synthetic rules. All share TaskSpec, action format, generator, verifier, RL interface, and Harness. [Environment cards](../../04_AgentTask/CustomAgentTasks/interaction_lab/ENVIRONMENTS.md) give exact L1–L5 settings.

## Design rules and next experiments

- Admit tasks with explicit preconditions, reachable goals, a replayed feasible certificate, negative checks, and separately audited splits. A zero-success policy does not prove a task invalid; a certificate does not prove partial-observation solvability.
- Let true state and temporal constraints decide success; allow multiple legal solutions. Keep world, policy, and event randomness distinct. Use ablations to test whether memory, preconditions, replanning, and resource sharing are necessary rather than shortcuts.
- Calibrate difficulty with a fixed model, Harness, observation, action granularity, and budget. Then compare single-family versus mixture RL on structural holdouts and paired interventions before testing UE state consistency and graph-to-3D transfer. New 3D worlds require scene compilation, collision, and reachability checks.
