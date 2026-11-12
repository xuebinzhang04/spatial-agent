# Workspace organization and execution principles

This file defines the stable organization, ownership, and execution rules for the numbered workspace folders. Keep it unchanged during routine research and implementation. Change it only when the user changes these principles or the permanent directory structure. Record project state, installed versions, commands, experiments, and progress under the existing [archive/](archive/README.md), using a dated Markdown file in the relevant topic directory. Do not create a separate records directory.

## Folder ownership

| Folder | Responsibility | Placement rule |
| --- | --- | --- |
| `00_Scope/` | User's core research ideas, motivation, insights, direction, and expected outcomes | User-authored and maintained. Agents read it to understand research intent and do not modify its contents. |
| `01_Models/` | Model weights, checkpoints, and model-specific assets | Keep model identity, revision, environment, and loading instructions with the model. |
| `02_Simulator/` | Simulator programs, source, scene assets, and runtime dependencies | Use simulator-named directories and `shared/`. Each simulator owns its version-specific dependencies. Record inventory, entrypoints, and verified runtime state in [REGISTRY.md](02_Simulator/REGISTRY.md). |
| `03_SpatialBench/` | Spatial benchmark datasets and evaluation infrastructure | Keep datasets and shared dataset caches in `benchmarks/`, evaluation definitions/configuration in `eval_configs/`, and evaluator source in its own checkout. |
| `04_AgentTask/` | Interactive tasks, agents, shared task frameworks, and demonstrations | Use `SpatialAgentBench/` for external task benchmarks, `CustomAgentTasks/` for original tasks, `TaskFrameworks/` for shared backends, `Harness/` for Agent integration, and `Demos/` for demonstrations. Keep runnable task definitions distinct from their reusable backend. |
| `05_Experiment/` | Experiment execution, monitoring, logs, and results | Put each run and its manifest in `results/<run-id>/`, execution logs in `logs/`, and monitoring code with its own tools. |

`00_Scope/` is the project's central intellectual reference: the user's ideas developed through paper reading and research direction-setting, including motivation, insights, hypotheses, and expectations. It is an important source for later paper writing and reasoning about the research contribution. Read it when interpreting research tasks and planning surveys, implementations, and experiments. Preserve its authorship and intended meaning; general workspace practices belong in this file, and Agent-generated findings or progress belong in the relevant project documentation or `archive/`.

## Dependency and file management

Keep one canonical copy of each asset or implementation. Reuse shared code through imports and dependencies rather than copying it into tasks. Place genuinely shared simulator dependencies and workspace-level simulator tools in `02_Simulator/shared/`; record versions and consumers in the simulator registry. Preserve independently packaged UE/Unity libraries and incompatible environments separately.

Task rules, goals, evaluators, and Agent policies belong under `04_AgentTask/`; rendering engines and simulator runtimes belong under `02_Simulator/`. Research surveys belong under `survey/`. Dated notes and superseded material belong under `archive/`, grouped by topic. Keep downloaded assets, model weights, and results out of `scripts/`.

Keep file management simple and easy to manually track. Reuse existing directories, keep nesting shallow, and give each file a clear responsibility. Use consistent, descriptive names that state the task or purpose; avoid ambiguous names such as `misc`, `temp`, `new`, or `final_v2` for maintained files. Use dates for historical records and run IDs for experiment outputs. Add a new directory, wrapper, index, or configuration only when it serves a concrete need; avoid parallel copies and redundant documentation.

When moving files, update active callers, configuration, editable installs, links, and documentation; exercise the affected entrypoints. Preserve historical run manifests, logs, and source snapshots as recorded. Explain their old paths in a dated record instead of rewriting the evidence. Remove redundant files only after verifying the canonical copy and the user's authorized scope.

## Code and comment standards

Use the simplest correct code with clear logic: short, focused functions, explicit data flow, and descriptive names. Follow the surrounding language and formatting conventions. Reuse existing helpers where appropriate; introduce abstractions, frameworks, compatibility layers, or fallback branches only for an actual requirement. Avoid duplicate implementations, unnecessary indirection, dead code, and speculative features. Prefer a small, understandable change that a person can inspect and trace manually.

Use English consistently for maintained code identifiers, comments, and docstrings. Comments explain intent, non-obvious constraints, units, assumptions, or tradeoffs; do not narrate obvious statements. Keep comments concise and accurate as code changes. Document public interfaces and surprising behavior where they are defined, and use concrete error messages that identify the failed operation and cause.

## Execution and experiment records

Read the applicable registry, task README, and runbook before execution. Use the runtime declared for the selected component. Check supported cluster modules before changing packages to solve compatibility. Run GPU and other compute-intensive work in scheduler allocations; use login nodes for light editing, transfers, scheduler queries, and small checks. Query current account/resource availability before submitting paid work.

Before starting an experiment, create its result directory and `run_metadata.json` using the [experiment record contract](05_Experiment/RECORD_CONTRACT.md). Record the actual research question, input tasks/assets, model and simulator revisions, configuration, seed/budget, source changes, scheduler resources, timestamps, outputs, and limitations. Use a unique project ID for each independent launch session; child jobs in the same session share it. Include the Slurm job ID in a scheduled run's name. Use explicit local run IDs for unscheduled checks.

Retain failed, canceled, and partial runs with diagnostics unless the user asks to remove them. Separate import checks, executable checks, scene launches, task completion, model evaluation, and transfer claims. Report results only with their exact inputs, completion status, and evidence paths. Keep live state and progress out of this file; update the relevant dated record and component registry/runbook.

## Script organization

| Category | Directory |
| --- | --- |
| Simulator acquisition, installation, and runtime setup | `scripts/simulator/` |
| Model acquisition and setup | `scripts/models/` |
| Dataset acquisition and preparation | `scripts/datasets/` |
| Experiment launch, analysis, and reporting | `scripts/experiments/` |
| Utilities shared across categories | `scripts/common/` |

Create a category when its first script is added. Keep helpers/configuration beside their caller, name scripts by action and target, and update callers/documentation when relocating them. Existing root scripts can migrate to their category when next changed.
