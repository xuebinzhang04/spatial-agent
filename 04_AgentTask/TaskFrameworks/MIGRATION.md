# Current custom-task grouping — 2026-10-19

The requested layout keeps the original Interaction Lab together and leaves the four new Stage 1 tasks directly under CustomAgentTasks. See the [current catalog](../CustomAgentTasks/README.md).

| Previous location | Current location |
| --- | --- |
| TaskFrameworks/interaction_lab | CustomAgentTasks/interaction_lab |
| CustomAgentTasks/memory_maze | CustomAgentTasks/interaction_lab/tasks/memory_maze |
| CustomAgentTasks/dependency_doors | CustomAgentTasks/interaction_lab/tasks/dependency_doors |
| CustomAgentTasks/relational_rearrangement | CustomAgentTasks/interaction_lab/tasks/relational_rearrangement |
| CustomAgentTasks/dynamic_replanning | CustomAgentTasks/interaction_lab/tasks/dynamic_replanning |
| CustomAgentTasks/signal_repair | CustomAgentTasks/interaction_lab/tasks/signal_repair |
| CustomAgentTasks/resource_delivery | CustomAgentTasks/interaction_lab/tasks/resource_delivery |

`CustomAgentTasks/{missing_object,moved_object,shape_identification,pose_alignment}` are unchanged. Their shared implementation remains in `TaskFrameworks/spatial_interaction/`. The original Interaction Lab aggregate run.py and Python package are restored under its group directory; its individual task entrypoints are nested in tasks/ rather than removed. The meeting viewer has subsequently been consolidated into Demos/spatial_agent/; historical source is archived.

Active imports, commands and documentation use the current paths. Historical run metadata and source snapshots are not rewritten. The previous flat-layout migration evidence remains under `05_Experiment/results/custom-tasks-reorganization-20261019/`; this follow-up grouping has its own evidence under `05_Experiment/results/interaction-lab-regroup-20261019/`.
