# Transfer from Controlled Interaction Training to External Benchmarks

## Research questions and hypotheses

RQ1: Is the trained policy more reliable on unseen layouts, compositions, and difficulty levels? RQ2: Does it transfer to interactive tasks in a real simulator? RQ3: Does it improve static or video spatial understanding? Evaluate each separately.

External work suggests spatial benchmark training need not improve navigation, so transfer tests must match task goals and required actions: [Beyond Spatial Benchmarks](https://arxiv.org/abs/2609.29934). The reverse direction, whether interaction training helps spatial QA, is also unproven here.

## Training comparisons

Fix Qwen3.8-27B base checkpoint/revision, precision, Harness, context, sampling, training tokens, environment steps, and GPU budget. Smaller models can support initial mechanism studies, but their gains must not be conflated with improvement of the intended base model.

Compare no-training baseline; navigation only; uniform five-family mixture; empirical-difficulty sampling; removal of partial observation; removal of dynamic events; primitive-only tasks; removal of relational composition; and sparse versus potential shaping. Use at least three training seeds. If budget is insufficient, reduce the matrix in advance and label the study exploratory.

Select checkpoints on validation. Neither final tests nor external benchmarks may tune the generator or select checkpoints. Stratify by family, level, and motif; report macro averages alongside sample counts.

## Internal generalization

- `test`: new instances in a separate seed namespace, with exact layout/task hash auditing.
- `composition`: hold out `left_of+above` for relations; for other families this currently means only new layouts and must be reported as such.
- `size`: larger layouts within currently supported node limits.
- `counterfactual`: a specified intervention paired with the same underlying test seed. Report paired changes in success, not as independent samples.

The next version should add graph-isomorphism and near-duplicate audits, true dependency-motif holdouts, and action/event-timing holdouts. A new seed alone is not all these forms of out-of-distribution testing.

## External interaction

**DeliveryBench:** Use the separate `gym_citynav` UE package. First map graph actions to MOVE, ACCEPT_ORDER, PICKUP, DROP_OFF, CHARGE, and WAIT. Graph nodes must map to physically reachable coordinates; projected POI connectors do not guarantee collision-free UE access. Read true orders, battery, and position; inspect action completion and native errors; then run bounded episodes. Do not score training with compact fallback zeros or the constant-zero Gym reward.

Validate actions, observations, units, and task state on small-city-11 first. Then compare under equal budgets across prespecified maps, seeds, and repeats. Preserve distinctions among failure, native lifecycle completion, truncation, and scheduler exit. Name and aggregate official benchmark protocol results separately from custom microtasks. Only graph import has been validated so far; this stage has no UE execution result.

**Indoor tasks:** Habitat scene assets are not prepared. AI2-THOR runtime lacks GPU launch validation. First acquire and verify scenes, actions, and poses, then map key-door preconditions and rearrangement predicates to real metadata. This is a transfer plan, not a completed adapter.

## External spatial evaluation

Start with Theory of Space (active spatial beliefs), MindCube (reference frames and rotation), and RoboSpatial (spatial relations), then extend to the 12 runnable SpatialBench tasks. Preserve each official task's inputs, parser, and scoring protocol. Do not mix the interaction environment's scalar reward into Overall Avg.

Run evaluations using existing `eval_configs` and runbook. A full Qwen baseline exists, but check whether the current baseline model and Harness settings match before drawing training-effect conclusions from scores with different inference budgets. OpenEQA's replacement judge protocol remains unresolved.

## Metrics and statistics

Report whole-task success rate (SR), stratified SR, repeated-run reliability on identical instances, pass@k, failure reasons, action/token/time cost, recovery rate on dynamic tasks, and delivery timeliness, completed orders, and native net earnings. Report path efficiency only when the optimal path is solved exactly under matching definitions; a construction certificate is not an optimal SPL denominator.

Pass@k, pass^k (all k runs succeed), and single-run SR are distinct metrics. Wilson intervals describe binomial success proportions. Correlated rollouts across instances and seeds need hierarchical bootstrap or paired tests, not an assumption that all rollouts are independent.

Every run follows the workspace experiment-record contract. `run_metadata.json` records the question, input hashes, task IDs, generator version, source hashes or commit, resources, actual settings, reward mode, termination conditions, and outputs. Check budget and account before submitting GPU jobs.
