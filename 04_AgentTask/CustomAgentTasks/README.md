# Custom interaction tasks

Eight independent task directories use the shared `TaskFrameworks/spatial_interaction` backend. Each owns its task definition, generation entrypoint, evaluator binding, documentation and benchmark recipes.

| Task | Simulator | Interaction |
| --- | --- | --- |
| `missing_object` | AI2-THOR | Active memorization, scene change and missing-object localization |
| `moved_object` | AI2-THOR | Active memorization and old/new location comparison |
| `shape_identification` | Shape3D | Change the camera view to distinguish 3D geometry |
| `pose_alignment` | Shape3D | Rotate the object to match a reference pose |
| `memory_maze` | AI2-THOR | Explore and revisit physical landmarks in order |
| `dependency_doors` | AI2-THOR | Open, retrieve, carry, place and restore container closure |
| `relational_rearrangement` | AI2-THOR | Place a target under spatial and preservation constraints |
| `dynamic_replanning` | AI2-THOR | Respond to a real obstruction and navigate a detour |

`delivery_cases/` constructs courier tasks over the dedicated SimWorld DeliveryBench simulator. See [tasks.md](tasks.md) for the Chinese task overview and each task README for native evidence and limitations.

Generation follows the local [TMAX reference](../../archive/tmax/rl_data/README.md): compose settings, construct tasks, validate initial/final states, retain failed attempts and calibrate difficulty through repeated solving. Follow the [interaction-first design protocol](../../survey/2026-10-agent-interaction/DESIGN.md).

The graph prototype and unfinished repair/delivery prototypes were retired to `archive/agent-interaction/legacy-demos/symbolic/interaction_lab/`. They are excluded from active APIs and the demo. Historical results remain preserved.

## Interaction expectations and case review

[DeliveryBench case construction](delivery_cases/README.md) adds fixed courier cases over the existing SimWorld DeliveryBench simulator and native APIs. The initial batch contains 36 recipes across four families and three structural levels. One pickup/delivery case passed native admission in job `389653`; 26 ordinary candidates and 9 visual-handoff drafts remain pending. Qwen3.8-27B completed the same admitted case in seven actual model decisions (job `389762`); this is a one-case development result. See the [batch examples](../../05_Experiment/results/delivery-cases-20261026/EXAMPLES.md) and [generation record](../../archive/agent-interaction/2026-10-26-delivery-case-generation.md).

New cases follow the [eight-task expectation registry](expectations.json) and the [shared interaction-first design protocol](../../survey/2026-10-agent-interaction/DESIGN.md#core-direction-interaction-must-resolve-a-necessary-information-or-control-gap). Generated recipes, simulator-admitted cases, reference witnesses and model outcomes are separate states. Inspect complete phase/action/observation traces before presenting an example as evidence of the intended ability.

## Local state, global memory and generation audit

[task_variables.json](task_variables.json) splits each task's environment variables into **local state** (public per-step observation fields) and **global memory** (latent variables the policy must build, update or keep frozen), with their lifecycle, how memory is scored, and the declared effect of every generation control. `scripts/experiments/task_generation_audit.py` extracts exactly these variables from generated cases and checks determinism, control effects and expectations (`fail` = generator contract broken, `warn` = admissible but weak for memory training):

```bash
PY=/gpfs/scrubbed/xuebiz/conda/envs/spatialagent/bin/python
$PY scripts/experiments/task_generation_audit.py --source live --output 05_Experiment/results/<new-run-id>      # Shape3D, CPU
$PY scripts/experiments/task_generation_audit.py --source recorded --output 05_Experiment/results/<new-run-id>  # demo recordings
sbatch scripts/experiments/task_generation_audit.slurm   # fresh native AI2-THOR generation on one MIG slice
```

Audits listed under `generation_audits` in the demo's `demo.json` appear in its *Task Generation & Memory Variables* section.
