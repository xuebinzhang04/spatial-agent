# SpatialAgent

SpatialAgent is a research workspace for interactive spatial learning environments
and VLM agents. The current work builds verifiable tasks and a research demo;
model learning and transfer are research questions, not established results.

This repository tracks first-party task, harness, demo, experiment, and research
source. Model weights, simulator installations, external benchmark checkouts,
generated media, and experiment results remain in the Tillicum workspace and are
not included in Git. A fresh clone needs those registered runtimes and assets
before native simulator jobs or the full recorded demo can run. See
[workspace.md](workspace.md), the [simulator registry](02_Simulator/REGISTRY.md),
and the [demo README](04_AgentTask/Demos/spatial_agent/README.md).

## Agent workflow

[AGENTS.md](AGENTS.md) contains the shared working agreement. Codex leads the
research queue, writes testable briefs, and reviews evidence. Claude Code is the
primary implementation owner. Each handoff includes a scoped question, frozen
case and control, diff, verification commands, and limitations. The
[current workflow note](archive/agent-interaction/2026-11-12-research-and-agent-workflow.md)
records the first case-oriented pass.

Read [00_Scope/](00_Scope/) for the user-authored research intent and the
[shared research protocol](survey/2026-10-agent-interaction/DESIGN.md) for case
admission and evidence levels. Run artifacts follow
[RECORD_CONTRACT.md](05_Experiment/RECORD_CONTRACT.md).
