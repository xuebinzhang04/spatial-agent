# Archive and workspace records

Use this existing directory for dated workspace notes and superseded material, grouped by topic. [workspace.md](../workspace.md) contains stable organization, coding, and execution principles; keep progress, software inventories, job IDs, scores, and temporary commands here. Do not create a separate records directory.

## Index

| Record | Contents |
| --- | --- |
| [Research and agent handoff workflow](agent-interaction/2026-11-12-research-and-agent-workflow.md) | One-case research cycle, demo acceptance check, and minimal Claude Code/Codex handoff; includes the transition to source-only Git tracking. |
| [tmux and demo recovery](workspace/2026-11-12-tmux-and-demo-recovery.md) | Current agent pane snapshot, durable recovery checks, and a future dedicated research demo session. |
| [Unified research demo](agent-interaction/2026-10-25-unified-demo.md) | Consolidated recordings, online environment/model APIs, simulator/task taxonomy, selector-driven definitions and exact-case media; legacy entrypoints archived. |
| [Demo coverage audit](agent-interaction/2026-10-25-demo-coverage.md) | Current player versus prior research, symbolic, and DeliveryBench demos; restored transition inspection and explicit unported capabilities. |
| [Interaction research presentation](agent-interaction/2026-10-25-interaction-presentation.md) | Corrected presentation scope, DeliveryBench/VLM-3R/PHAI reference review, eight task examples, model diagnostic, PDF/MP4/PNG export, and evidence checks. |
| [Project demo interface](agent-interaction/2026-10-24-research-demo-interface.md) | Shared project homepage and task/model episode explorer, one-command startup, initial weekly delivery, source evidence, and validation boundary. |
| [TMAX task generation](agent-interaction/2026-10-24-tmax-task-generation.md) | User-supplied full clone in `tmax/`, verified revision, task-generation source map, and the reference method for subsequent SpatialAgent task generation. |
| [2026-10-19-workspace-history.md](workspace/2026-10-19-workspace-history.md) | Imported environment setup, simulator reorganization, benchmark history, Interaction Lab, Queue Inspector, Stage 1 validation, and task-layout changes. The final grouping entry supersedes the earlier flat task layout. |
| [Original workspace snapshot](workspace/workspace-before-records-20261019.md) | Unmodified workspace document before separating policy from history. |
| [Previous simulator registry](simulator/REGISTRY-before-shared-layout-20261018.md) | Earlier simulator inventory before the shared-dependency layout. |

## Recording rule

Reuse a related topic directory and an active note when possible. For a distinct completed change, use `<topic>/YYYY-MM-DD-description.md`. Record purpose, affected paths, commands/configuration, evidence, outcome, limitations, and next steps. Link superseding decisions explicitly. Update this index for substantial records; avoid creating an index or duplicate note for every small change. Keep paths relative to the workspace root in prose and make Markdown links relative to the note.

Use English for canonical records and code identifiers. Keep scene/runtime versions and operational commands in the relevant project README or registry, and link them here. Experiment artifacts and `run_metadata.json` remain in `05_Experiment/results/<run-id>/`; records link to them rather than copying their contents. Scheduler state and resource availability must be queried live.

## Operational references

- [Simulator registry](../02_Simulator/REGISTRY.md): installed bundles, dependencies, entrypoints, and validation status.
- [Benchmark configuration](../03_SpatialBench/eval_configs/README.md): evaluation inventory and usage.
- [Custom task index](../04_AgentTask/CustomAgentTasks/README.md): task families and current grouping.
- [Experiment record contract](../05_Experiment/RECORD_CONTRACT.md): required run metadata.
- [Job runbook](../05_Experiment/JOB_RUNBOOK.md): scheduling and diagnostic history.
