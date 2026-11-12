# Related Work: Motivation, Construction, Evaluation, and Project Choices

As of 2026-10-11, this survey collects primary papers and author materials for the current task-construction questions. **M** = read relevant method/evaluation sections; **A** = reviewed original abstract or author overview; **S** = additionally inspected an official interface or implementation. None implies a full reproduction. Some 2026 papers are preprints; their reported effects are authors’ claims and have not been verified locally.

The “project use and limits” column is our analysis. See [EXAMPLES.md](EXAMPLES.md) for concrete flows and Interaction Lab for the implementation. Entries are brief summaries, avoiding extended quotation.

## 1. Automatically Generated, Verifiable Training Tasks

| Work / reading | Motivation and construction | Evaluation | Project use and limits |
| --- | --- | --- | --- |
| [TMax (2026)](https://arxiv.org/abs/2606.23321) M/S | Terminal training lacks diverse tasks at useful difficulty; sample structural axes to build self-contained execution worlds and program verifiers | Repeated fixed-policy pass@k, composition balance, external terminal tests; filter zero reward-variance groups during training | Use factor sampling and empirical calibration; soft training filters do not replace generator-correctness checks |
| [Endless Terminals (2026)](https://arxiv.org/abs/2601.16443) A | Scale limited terminal benchmarks with generated tasks, containers, completion tests, and solvability filters | Executable tests and external terminal tasks | Construct task, environment, and verifier together; spatial worlds also require reachability checks |
| [RLVE (2026)](https://proceedings.mlr.press/v306/zeng26j.html) A | Generate problems procedurally with algorithmic rewards and adaptive difficulty | Reasoning training and generalization evaluation | Co-design parameterized difficulty and verification; many problems lack persistent spatial interaction |
| [Reasoning Gym (2025)](https://arxiv.org/abs/2505.24760) A | Create unlimited procedural questions across domains with controlled complexity | Verifiable answers and training/external reasoning tasks | Use reproducible generator/verifier modules; single-turn QA is not an interactive world |
| [CompoWorld (2026)](https://arxiv.org/abs/2609.33665) A | Train multi-service workflows with shared interfaces, typed state, and dependency graphs | Completion-oriented reward and external agent benchmarks | Cross-service dependencies suggest cross-ability dependencies; not every world-model tool is an exact simulator |
| [Agent-World (2026)](https://arxiv.org/abs/2604.18292) A | Discover environment themes, generate tasks, and train from capability gaps | Multi-environment RL and external tasks | Let generation respond to learning needs while defining verifier trust boundaries first |
| [SWE-smith (2025)](https://arxiv.org/abs/2504.21798) A | Scale repair tasks from software repositories | Executable tests and SWE evaluation | Real assets can supply tasks; one domain does not provide spatial coverage |
| [AutoForge (2025)](https://arxiv.org/abs/2512.22857) M | Generate database/Python operations from tool descriptions and compose complex tasks with tool-dependency DAGs | Execute reference workflows for terminal states; multi-turn RL and external tool benchmarks | Use dependencies and executable state; simulated-user error detection also relies on an LLM, so the system is not fully deterministic |
| [Environment Scaling / AgentScaler (2025)](https://arxiv.org/abs/2509.13311) A | Scale function-calling coverage with heterogeneous simulated environments and staged training | External tool tasks including τ, τ², and ACEBench | Expand mechanism diversity and task depth; test simulated-to-real interface transfer separately |

## 2. Multi-Turn Agent RL Systems

| Work / reading | Motivation and construction | Evaluation | Project use and limits |
| --- | --- | --- | --- |
| [AgentGym (2024)](https://arxiv.org/abs/2406.04151) A | Unify multi-environment interaction, trajectories, and self-evolution | Success and generalization across environments | Unify the environment interface; mixing suite tasks alone does not establish compositional generalization |
| [AgentGym-RL (2025)](https://arxiv.org/abs/2509.08755) A | Train long-horizon multi-turn RL with gradually extended interaction horizons | Multi-environment tasks and training stability | Include horizon in training design and hold budget fixed in comparisons |
| [RAGEN (2025)](https://arxiv.org/abs/2504.20073) A | Study multi-turn policy optimization and degeneration modes | Stylized worlds, reward variance, and trajectory behavior | Vary initial states and feedback granularity; high reward does not prove use of target ability |
| [VAGEN (2025)](https://arxiv.org/abs/2510.16907) M | Estimate state and predict transitions under partial visual observation in multi-turn RL | Games, navigation, and manipulation tasks | Keep state/next-state diagnostics; model-written reasoning is not automatically an accurate world model |
| [GEM (2025)](https://arxiv.org/abs/2510.01051) A/S | Standardize an experience interface and asynchronous/vectorized execution for agentic LMs | Multi-domain environments and RL integration | Decouple environment and trainer; a common interface cannot repair inconsistent task semantics |
| [SWE-Gym (2024)](https://arxiv.org/abs/2412.21139) A | Use real software tasks, execution runtime, and unit tests | Repair success and training/inference verification | Test independently against real execution; passing tests does not guarantee all user requirements |
| [OpenEnv](https://github.com/huggingface/OpenEnv) S | Standardize RL environment services and action/observation/reset interfaces | Example environments and trainer connections | A replaceable interface library does not automatically establish task validity or composability |

## 3. Tool, Web, Desktop, and Event Worlds

These works mainly provide evaluation environments. Executable scoring may support RL data collection, but does not establish that each original paper trained with RL.

| Work / reading | Motivation and construction | Evaluation | Project use and limits |
| --- | --- | --- | --- |
| [AppWorld (2024)](https://arxiv.org/abs/2407.18901) A | Use multi-app state and complex API workflows | State-based unit tests including collateral damage | Accept multiple valid solutions and verify protected unrelated objects |
| [ToolSandbox (2024)](https://arxiv.org/abs/2408.04682) A | Model real tool preconditions and conversation state | Trajectory milestones and dependency checks | Verify cause and order as well as final state; do not grade by call strings |
| [τ-bench (2024)](https://arxiv.org/abs/2406.12045) A | Constrain interaction jointly through users, tools, and policies | Data state and repeated-run reliability | Make constraints executable world rules; keep pass^k separate from pass@k |
| [τ²-Bench (2025)](https://arxiv.org/abs/2506.07982) A | Let user and agent both change state in a dual-control environment | Tool/conversation collaboration and environment-task success | Shared control could extend to spatial worlds; current prototype has one deterministic agent |
| [ARE (2025)](https://arxiv.org/abs/2509.17158) A | Provide extensible applications and agent orchestration | Dynamic tasks across environments | Keep events, tools, and verifiers separate from a particular Harness |
| [Gaia2 (2026)](https://arxiv.org/abs/2602.11964) A | Model time, asynchronous events, ambiguity, and collaboration | Write-action verification | Represent temporal dynamics explicitly; v1 action-triggered events are not full asynchronous simulation |
| [WebArena (2023)](https://arxiv.org/abs/2307.13854) A | Use self-hosted websites for realistic workflows | Functional state and programmatic evaluation | Reset/replay and terminal-state checks matter; site assets are not an automatic task generator |
| [BrowserGym (2024)](https://arxiv.org/abs/2412.05467) A/S | Unify web-task execution, observation, action, and experiment configuration | Consistent multi-benchmark execution and analysis | Adapter and Harness settings are evaluation conditions; platform tasks are not all RL-trained outcomes |
| [WebShop (2022)](https://arxiv.org/abs/2207.01206) A | Build shopping interaction from real products and instructions | Task reward for attributes and selections | Align goal semantics with actions and prevent answer leakage from retrieval metadata |
| [OSWorld (2024)](https://arxiv.org/abs/2404.07972) A | Operate real desktops and multiple apps | Execution-based evaluation | Check environment state and artifacts; measure GUI cost and fragility |
| [OSWorld 2.0 (2026)](https://arxiv.org/abs/2606.29537) A | Extend realistic workflows to longer end-to-end tasks | Long-horizon completion and tool cost | Treat length, context, and timeout as distinct difficulty axes |
| [OSWorld-MCP (2025)](https://arxiv.org/abs/2510.24563) A | Compare GUI and hybrid tool capabilities | Tool-operation evaluation in the same environment | Fix action granularity and Harness when comparing scores |
| [OSWorld-Science (2026)](https://arxiv.org/abs/2609.39903) A | Operate scientific software to produce deliverables | Artifact-based evaluation | Separate final deliverable verification from process diagnostics |
| [AgentDojo (2024)](https://arxiv.org/abs/2406.13352) A | Evaluate injection defense in dynamic tool tasks | User-task utility and attack success | Environment-returned data must not change scoring rules; this is a robustness evaluation framework |

## 4. Spatial, Embodied, and Generative Worlds

| Work / reading | Motivation and construction | Evaluation | Project use and limits |
| --- | --- | --- | --- |
| [BabyAI (2018/ICLR 2019)](https://arxiv.org/abs/1810.08272) A/S | Compose language instructions and graded interaction grids | Instruction completion, sample efficiency, and experts | Unify grammar, state, and verifier rather than merely randomizing object placement |
| [MiniGrid / MiniWorld (2023)](https://arxiv.org/abs/2306.13831) A | Offer rapidly customizable lightweight 2D/3D RL worlds | Goal-oriented reward and transfer examples | Build a small controlled core, then verify abstract skill benefits across backends |
| [gSCAN (2020)](https://arxiv.org/abs/2003.05161) A | Test systematic compositional generalization in world states | Explicit held-out rule/attribute combinations | Hold out compositions, not only random seeds; it is mainly a language-understanding benchmark |
| [XLand (2021)](https://arxiv.org/abs/2107.12808) A | Generate many worlds and games procedurally | Held-out tasks, normalized ability, and iterative progress | Vary world, rules, and goals; scale and multi-agent cost are substantial |
| [XLand-MiniGrid (2023)](https://arxiv.org/abs/2312.12044) A | Use minimal worlds with diverse rules and goals for meta-RL | New-task adaptation and high-throughput execution | Rule changes matter more than appearance randomization |
| [XLand-100B (2024)](https://arxiv.org/abs/2406.08973) A | Study in-context RL from complete learning histories | Learning across task histories | Retain feedback and failures in trajectories; do not copy its data scale at this stage |
| [Procgen (2019/ICML 2020)](https://arxiv.org/abs/1912.01588) A | Test sample efficiency and generalization in procedural worlds | Fixed training levels versus unseen levels | Separate fitting seen instances from generalization; many seeds are necessary but insufficient |
| [ProcTHOR (2022)](https://arxiv.org/abs/2206.06994) A | Procedurally generate large sets of interactive indoor scenes | Embodied training and unseen scenes | Construct geometry, reachability, and object properties together |
| [Habitat 2.0 (2021)](https://arxiv.org/abs/2106.14405) A | Support rearrangeable 3D scenes and long household tasks | RL, sense-plan-act, and layout/object generalization | Evaluate skill handoff and long chains separately; scene assets are not an installed runtime |
| [ALFRED (2019/CVPR 2020)](https://arxiv.org/abs/1912.01734) A | Execute irreversible compound household tasks under visual instructions | Whole-task and subgoal completion | Check preservation and terminal facts rather than exact expert trajectories |
| [DialFRED (2022)](https://arxiv.org/abs/2202.13330) A | Allow active clarification during embodied execution | Task completion after question answering | Information acquisition should be an action; v1 has no language clarification |
| [BEHAVIOR-1K (2024)](https://arxiv.org/abs/2403.09227) A | Model human-need household activities with real physics | Task/object state and long-horizon execution | Closer to real needs, but physics and asset validation are costly |
| [DeliveryBench (2025)](https://arxiv.org/abs/2512.19234) A/S | Model long-horizon city delivery with multiple constraints | Native profit/orders and user comparison | Real maps and API are a transfer target; local ten-map coverage must not be confused with paper evaluation cities |
| [SimWorld Studio (2026)](https://arxiv.org/abs/2605.09423) M | Co-evolve generated scenes and agents using code, compilation, physics, and visual feedback | Scene validity and navigation/curriculum experiments | Worlds need interfaces, tasks, and feedback; the official local package is not installed |
| [Theory of Space (2026)](https://arxiv.org/abs/2602.07055) M | Construct, revise, and use spatial beliefs under active exploration | Belief probes, route/survey tasks, and false-belief diagnostics | Direct motivation for memory and replanning; exploration accuracy is not downstream task completion |
| [RoboSpatial (2024)](https://arxiv.org/abs/2411.16537) A | Probe robot-relevant reference frames, relations, and spatial constraints | 2D/3D spatial tasks | Use terminal geometry as an external probe; QA gains do not substitute for manipulation success |
| [AlphaMaze (2025)](https://arxiv.org/abs/2502.14669) A | Improve maze decisions using GRPO | Path tasks and training comparisons | An exactly verifiable initial domain; navigation alone may still permit shortcuts |
| [From Building Blocks to Planning (2025)](https://arxiv.org/abs/2512.24532) A | Train atomic spatial transformations and multistep compositions | ASCII closed-loop and internal-state tasks | Measure primitive and composed abilities separately |
| [Beyond Spatial Benchmarks (2026)](https://arxiv.org/abs/2609.29934) M | Study mismatch between spatial specialty training and navigation | Paired base models, navigation frameworks, and staged spatial factors | Align transfer with tasks; spatial scores alone do not imply interaction gains |

## 5. Curricula and Open-Ended Interaction

| Work / reading | Motivation and construction | Evaluation | Project use and limits |
| --- | --- | --- | --- |
| [PAIRED (2020)](https://arxiv.org/abs/2012.02096) A | Design environments through antagonist/protagonist regret | Unseen environments and zero-shot transfer | Seek hard yet learnable tasks; current-policy failure does not prove unsolvability |
| [Prioritized Level Replay (2021)](https://proceedings.mlr.press/v139/jiang21b.html) A/S | Prioritize replay of training levels with learning potential | Procedural-world generalization and efficiency | Use learning feedback for priority; v1 is not full PLR |
| [ACCEL (2022)](https://arxiv.org/abs/2203.01302) A | Locally mutate levels with high regret | Mazes, continuous control, and curricula | Reverify feasibility after every mutation; growing maps alone is insufficient |
| [ALP-GMM (2020)](https://proceedings.mlr.press/v100/portelas20a.html) A | Use a learning-progress teacher over continuous parameters | Parameterized RL environments and transfer | Empirical difficulty changes with policy and is not a fixed hand label |
| [TeachMyAgent (2021)](https://arxiv.org/abs/2103.09815) A | Compare curricula under unsolvable regions and reward challenges | Specialized challenges and Parkour | Curricula need their own test environments; one random-generator baseline is inadequate |
| [POET (2019)](https://arxiv.org/abs/1901.01753) A | Co-evolve environments and solvers and transfer across them | Diversity and increasing task complexity | Track diversity and feasible-solution history; evolution does not guarantee valid tasks |
| [EnvGen (2024)](https://arxiv.org/abs/2403.12014) A/S | Generate/adapt simulator configurations with an LLM to address a smaller RL agent's weaknesses | Crafter/Heist efficiency, achievement success, and configuration ablations | Keep generator calls few and runtime cheap; evaluate on original distribution for overfitting |
| [OMNI-EPIC (2024/ICLR 2025)](https://arxiv.org/abs/2405.15568) A/S | Generate novel environment code and rewards with foundation models | Open-ended generation, learning progress, example worlds | Verify execution, reward, and task alignment after creative generation; official online game was unavailable |
| [TextWorld (2018)](https://arxiv.org/abs/1806.11532) A | Control text-game difficulty, scope, language, and state | Text-game learning and transfer | Logical preconditions and plans can inform the spatial DSL |
| [ScienceWorld (2022)](https://arxiv.org/abs/2203.07540) A | Move from memorized answers to active experiments | Completion of environment experiments | Observations and tests must affect decisions; hidden-rule discovery is an extension |
| [Crafter (2021)](https://arxiv.org/abs/2109.06780) A | Use survival, gathering, and crafting to measure general ability | Achievement coverage and sample efficiency | Shared resources form real dependencies; coverage does not equal whole-task completion |
| [Craftax (2024)](https://arxiv.org/abs/2402.16801) A | Increase open-ended RL throughput and challenge | PPO, interaction throughput, difficult goals | Treat simulator cost as a design metric; current Python core does not claim JAX throughput |
| [MineDojo (2022)](https://arxiv.org/abs/2206.08853) A | Provide rich open-ended tasks, a knowledge base, and general agents | Many embodied goals | Distinguish executable goals from open-ended ones; do not force binary rules onto the latter |
| [BALROG (2024)](https://arxiv.org/abs/2411.13543) A | Test agent reasoning in games of varying difficulty | Behavior and scores across RL games | Report cross-environment ability matrices; raw scores on different scales cannot be averaged directly |
| [Potential-based shaping (1999)](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) A | Derive conditions under which reward shaping preserves optimal policies | Theory and examples | Keep discount and terminal potential consistent; shaping is not additional correctness verification |

## Synthesis

1. **Benchmarks and training environments are different outputs.** A small annotated test set may assess ability but not offer diverse learning signal. Training needs an episode generator, reset, cheap transitions, and a stable verifier.
2. **Generate environment and scoring semantics with each task.** Separately generated instructions, world states, and success tests can disagree. Define an executable program first, then express it in language.
3. **Interaction changes state and obtains evidence.** Tool calls are only an interface. The ability requirement is whether feedback, memory, and replanning are necessary.
4. **Entity and rule composition matters more than prompt paraphrase.** Generated dependencies should change legal actions, reachability, resources, or final state.
5. **Record two kinds of difficulty.** Structural controls aid reproduction; policy behavior determines curriculum. Harness, action granularity, and budget change difficulty.
6. **Evaluate verifiers too.** Test known feasible solutions, illegal-action rejection, alternative-solution acceptance, collateral damage, and reward loops.
7. **High-throughput abstractions and real backends complement each other.** Graph/symbolic layers support mechanisms and efficient training; real simulators test vision, physics, and interface transfer. Check action and state semantics across them.
8. **Task variety alone is not generalization evidence.** Prespecify causal controls for held-out compositions, changed rules, topology scale, backend, and external benchmarks.
