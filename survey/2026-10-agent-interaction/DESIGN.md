# Shared Research Protocol

## Core direction: interaction must resolve a necessary information or control gap

The research unit is an **ability-demanding interaction**, not a named task, an attractive scene, or a successful privileged trajectory. State the expected behavior first, construct a setting that needs it, then inspect actual VLM execution. Use [the eight-task expectation registry](../../04_AgentTask/CustomAgentTasks/expectations.json) as the operational checklist. Preserve user-authored `00_Scope/`; implementation reviews belong here and in dated records.

For every case, specify `expected ability → initial information gap → useful available interaction → phase transition → feedback → verification`. A single initial image must not already resolve a task advertised as active exploration. Making the agent turn or walk a prescribed number of times does not prove interaction necessity. Use matched shortcut controls and counterfactual instances, such as two worlds with identical initial observations but different correct answers, distinguished by an executable action sequence.

A memory task needs an explicit bounded survey scope, a reason to inspect beyond the initial view, clear commitment semantics, and enough time to discover and record object-location evidence. The changed target is selected independently of the agent's inspection; never secretly select only what the policy happened to see. Keep the target identity hidden before detection. Put the target outside initial sight, and, for navigation levels, verify that stationary camera inspection cannot recover it. Generate and replay a physical exploration/revisit witness under the same budget. A pose turned away from a visible target is only a viewpoint task, not evidence of travel planning or global spatial memory.

Use stage-specific public instructions: explain the complete task at reset, the current phase and remaining actions at each step, how to commit memory, what the environment changes, and what must be submitted. Invalid actions consume budget and return nonprivileged diagnostic feedback. State whether notebook updates replace or append, which images are retained, and whether memory survives the transition. A timeout must not impersonate an agent decision to finish memorization. Do not supply evaluator annotations such as the missing object's identity to the policy.

### Case inspection and expectation feedback

1. Freeze the case, model, Harness, observation contract and budget; retain complete public trajectories and separate private evaluator evidence.
2. Read the actual action sequence and phase timing, not just the final reward or animation. Check new views, translation, revisits, observations supporting the answer, notebook retention, commitment and submission.
3. Compare expected behavior with the trace. Treat repeated actions, pose cycles and unchanged images as diagnostic flags, not automatic failure labels: repeated inspection can be useful if it reveals evidence.
4. Attribute a discrepancy provisionally to scene/observability, prompt/protocol, Harness memory/context, action feedback, verifier, policy, or infrastructure. Separate directly observed facts from causal hypotheses.
5. Make a targeted change and rerun a matched case. Compare scene-only, prompt-only and Harness-only changes when isolating a cause; then regenerate the broader case suite. Retain negative examples and report admission failures.
6. Promote new demo examples only with fresh unnumbered observations, exact source hashes, verified public-reference replay, shortcut checks and explicit expectation review. Keep reference witnesses separate from uncorrected VLM runs. Task correctness, useful policy behavior, learning and transfer are separate claims.

The first inspected missing-object smoke case committed at step 1 without exploration, then alternated pitch for 13 steps, with zero translation/yaw changes. Its target was initially visible; object identity was correct but the original receptacle was wrong. These facts identify a weak exploration setting and a notebook-retention concern; they do not establish that prompt wording alone caused the failure. See [the case review](../../05_Experiment/results/active-memory-review-20261025/CASE_REVIEW.md).

## Task-generation reference: TMAX

The user designated [TMAX](../../archive/tmax/rl_data/README.md) as the methodological and organizational reference for subsequent SpatialAgent task generation. See the [source review and adaptation](../../archive/agent-interaction/2026-10-24-tmax-task-generation.md) for the checked revision and implementation details. Apply its pipeline within the existing task families and shared backends:

1. **Sample a structured signature:** family, scene, compatible interaction primitives, object roles, observation conditions, event/relation program, complexity, and budget. Keep sampled factors in case metadata.
2. **Construct the instance:** produce the public instruction and reproducible scene/reset recipe, with private evaluator state. Ground required interactions in the simulator's available APIs and affordances.
3. **Separate initial and final checks:** verify the reset environment independently of the goal predicate. Retain replay and negative-trace checks for feasibility and verifier behavior; an executable environment alone does not establish task correctness.
4. **Run policies separately:** reuse the existing Harness to collect repeated attempts against frozen cases. Keep generation, policy execution, verification, and analysis separate.
5. **Calibrate and balance the corpus:** distinguish sampled complexity from measured difficulty; analyze success/pass@k, coverage, composition, and infrastructure failures. Record filtering settings explicitly, including any training-time zero-variance group filtering.

Task implementations remain in `04_AgentTask/`, simulator dependencies in `02_Simulator/`, and generated corpora/trajectories/reports in `05_Experiment/results/<run-id>/`. Reuse family-level verifiers and existing runtimes; a new case should primarily add data rather than another copy of the engine or evaluator. This is the reference for future generation work; it does not imply that an LLM-driven corpus generator or TMAX training has already been implemented.

## Environment, task, and agent

The environment defines state S, actions A, transition T, and observation O; a task defines initial state I, goal program G, constraints C, and budget B. The Harness receives only observation history and invokes A. Hidden maps, future events, certificates, and verifier state stay private. Generate versioned, content-hashed TaskSpecs with `generate(family, seed, level, split, assets)`.

## Composition and admission

V1 goal language is `SEQ(ALL(predicates), ...)`, followed by final predicates. A transition advances at most one stage. Predicates cover location, inventory, doors, placement/relations, delivery/time, battery, events, and untouched protected objects. Composition means a real dependency through preconditions, reachability, observations, shared resources, or simultaneous final constraints; a longer prompt alone does not qualify. OR/UNTIL/COUNT need separate semantics and tests before use.

Generate an ability motif, world, and goal program together. Admit an instance only after entity/precondition/connectivity checks and replay of a feasible trajectory under its final budget and deadlines. Store certificates apart from agent inputs; audit exact duplicates and split membership. Search timeout is `unknown`, not `unsolvable`. A witness upper-bounds optimal step count, and a privileged witness does not establish observation-limited feasibility or a calibrated budget.

## Verification and reward

Score true state and history, not model claims, reference-action strings, or reward values. Check temporal order, final goals, protected-object touch history, resources, and deadlines. Reject early/remote delivery, crossing locked doors, remote placement, unlimited-battery exploits, forged success flags, and actions after termination. Replay uses the same transition code and optional hashes; behavioral tests and cross-backend checks remain necessary. Distinguish success, failed `stop`, and budget truncation.

Default sparse reward is 1 only for verified success. Optional potential shaping is `r' = r + γΦ(next) − Φ(current)` with Φ=0 at every episode end, including truncation; compare it against sparse reward under the same discount. The public fixed action catalogue/mask does not disclose hidden-state legality. Invalid actions consume budget. Trainers must encode text observations; the optional Gymnasium wrapper alone is not an RL pipeline.

## Difficulty and evaluation

Structural difficulty varies topology, goals, dependency depth, relations, visibility/memory, events, resource slack, and budget slack. L1–L5 parameter profiles are reproducible but need empirical calibration. With fixed model revision, Harness, observation, actions, sampling, and budget, report success by family/level, failure modes, steps, tokens, wall time, pass@1 intervals, and `pass@k = 1-C(n-c,k)/C(n,k)` for k≤n. Curriculum sampling may favor a measured learning frontier only after generator validity and a fixed uniform baseline are established; the current sampler is not PAIRED/PLR/ACCEL.

Keep instance, layout, predicate-composition, size, intervention, and backend holdouts distinct. Current exact task/world hashes miss graph isomorphisms and near duplicates. Freeze final tests and external benchmarks during generator tuning and checkpoint selection. Group RL rollouts should share TaskSpec while policy randomness is separately recorded. Only a single-task tabular learning smoke has run so far; model RL and transfer remain hypotheses. See [TRANSFER_PROTOCOL.md](TRANSFER_PROTOCOL.md) for experiments.
