# Simulator registry

Updated: 2026-10-18. Paths below are relative to the SpatialAgent root. This is the current layout and status; the previous detailed registry is preserved at `archive/simulator/REGISTRY-before-shared-layout-20261018.md`.

## Placement rule

`02_Simulator/` contains four simulator-named directories and `shared/`. Each simulator owns its source, packaged executable, version-specific libraries, and Python environment. `shared/` holds workspace-level simulator tools and dependencies genuinely reused by multiple consumers at the **same version and ABI**. Do not merge libraries embedded in independently packaged UE or Unity programs. Task and benchmark code belongs under `04_AgentTask/`, and experiment artifacts under `05_Experiment/results/`.

The only item in `02_Simulator/shared/` today is `ue5-toolchain/`: a v19 Clang archive for possible UE development. **No current simulator requires it to run.** It is not a UE editor, UE source tree, or common installed UE runtime. There is no standalone UE editor/source checkout in this workspace.

## Installed simulator bundles

| Bundle | Contents and code role | Agent entrypoint | Verification |
| --- | --- | --- | --- |
| `02_Simulator/SimWorld/` | `source/` (Python city generation, map, traffic, weather, communication; commit `1f588ec946994614b04abe42faf1d5c3ad9bf3f5`); `Linux/` (packaged UE program and assets). | Python SimWorld API communicates with its own UE process. | Executable extracted; no independent scene-launch smoke result. |
| `02_Simulator/DeliveryBench/` | `Linux/gym_citynav/` (dedicated packaged UE program and maps). Its task rules and Agent code are in `04_AgentTask/SpatialAgentBench/DeliveryBench/`. Package asset revision `64a4889d72a2c7b792d242083e53a926b43ceaa9`. | Launch `Linux/gym_citynav/Binaries/Linux/gym_citynav` with short map name, then use the DeliveryBench Python wrapper over UnrealCV port 9000. | Real scene launch and interactions recorded in `05_Experiment/results/interaction-demo-345795/`; launcher paths updated after the move. A new GPU launch at the new path has not run. |
| `02_Simulator/Habitat/` | `sim/` (Habitat-Sim C++/Python rendering, sensors, navigation; commit `57ee4941dc4765240f0f91f70b2c97a919bf9038`); `lab/` (Habitat-Lab 0.3.2 tasks, datasets, metrics, baselines; commit `2c164c3a29d1d4eabcb3d183479e9696ed002515`); `runtime/` (installed Python 3.9, Sim/Lab 0.3.2). | `02_Simulator/Habitat/runtime/bin/python`; `habitat_sim` for simulation, `habitat.Env` for tasks. | Python activation, imports, and pip work after relocation. No benchmark scene collection or GPU scene test. |
| `02_Simulator/AI2THOR/` | `unity/` (scene/interaction source); `ai2thor/` (Python controller source, commit `24f79883b4889e3f0e6f4ae301808b9025872dfc`); `runtime/releases/` (matching CloudRendering Unity executable); `runtime/python/` (installed Python 3.10 and package 5.0.0). | `02_Simulator/AI2THOR/runtime/python/bin/python` and `ai2thor.controller.Controller`. | Python activation, imports, pip, and executable presence work after relocation. No GPU Unity scene test. |

The two UE packages are separate executable builds: DeliveryBench does **not** load its engine libraries from `SimWorld/Linux/`. Habitat-Lab is a task layer over Habitat-Sim, not another renderer. AI2-THOR contains both Unity simulation and its Python control API.

## Task layers outside this directory

| Task layer | Location | Depends on |
| --- | --- | --- |
| DeliveryBench | `04_AgentTask/SpatialAgentBench/DeliveryBench/` | Its dedicated `02_Simulator/DeliveryBench/Linux/gym_citynav/` package |
| Embodied Web Agents | `04_AgentTask/SpatialAgentBench/EmbodiedWebAgents/` | AI2-THOR plus separate outdoor/web components; combined stack is not installed or validated |
| Interaction Lab | `04_AgentTask/CustomAgentTasks/interaction_lab/` | Its own symbolic graph simulator; default five tasks require none of the 3D bundles. Optional `resource_delivery` imports DeliveryBench road/POI graphs, not UE physics. |

SimWorld Studio is not installed. It is not counted as a runnable simulator. The UE compiler archive is documented in `shared/ue5-toolchain/README.md`.

## Use and status rule

Read this registry before selecting a backend. A successful import, an executable file, and a successful 3D scene launch are different evidence levels. Do not claim end-to-end 3D readiness for base SimWorld, Habitat, or AI2-THOR until a scene is launched and actions/observations are checked on a compatible compute node. GPU jobs follow `workspace.md`, `05_Experiment/RECORD_CONTRACT.md`, and `05_Experiment/JOB_RUNBOOK.md`. Record subsequent setup and validation changes under `archive/`, following `archive/README.md`.


## 2026-10-19 Stage 1 software-rendering verification

The cached AI2-THOR 5.0.0 CloudRendering build now has real scene/action verification through [Spatial Interaction Stage 1](../04_AgentTask/TaskFrameworks/spatial_interaction/README.md). CPU lavapipe requires explicit Unity `-force-device-index 0`; FIFO avoids network sockets. No display server, simulator reinstall, or GPU allocation was used. Native deactivation and collision-checked relocation passed oracle episodes on FloorPlan1/2/3; six kitchen/living-room canonical snapshots passed native pose restoration. Missing means deactivation of renderers/colliders/interactions; destructive `RemoveFromScene` triggers a native metadata exception in this build. This does not establish GPU readiness. See [final validation](../05_Experiment/results/stage1-final-20261019/validation.json), with 28 passing tests, 12 oracle/replay checks, and 12 crash-free random rollouts. Original bundles remain unchanged.


## 2026-10-19 task layout update

The four Stage 1 entrypoints live directly under [CustomAgentTasks](../04_AgentTask/CustomAgentTasks/README.md). The original symbolic Interaction Lab implementation and its six task families are grouped under `04_AgentTask/CustomAgentTasks/interaction_lab/`; shared Stage 1 code lives in `04_AgentTask/TaskFrameworks/spatial_interaction/`. The original meeting viewer/manual bridge has since been consolidated into `04_AgentTask/Demos/spatial_agent/`; earlier source is archived. Simulator bundles and Python runtimes are unchanged. The post-migration regression passed 79 tests, ten task CLI/oracle/generator checks and four Stage 1 replays; this does not establish a new GPU/UE session or live model result.


Current task grouping: the original Interaction Lab is restored under `04_AgentTask/CustomAgentTasks/interaction_lab/` (individual family entrypoints in tasks/). The four new Stage 1 task directories remain directly under CustomAgentTasks. This follow-up grouping passed 79 tests and all entrypoint/replay checks; see `05_Experiment/results/interaction-lab-regroup-20261019/validation.json`. Simulator bundles and runtime configuration are unchanged.

## Demo simulator identity clarification, 2026-10-25

The demo environment for Delivery is **SimWorld, dedicated DeliveryBench `gym_citynav` build**. DeliveryBench supplies the delivery task layer, not a new simulator family. Its executable remains `02_Simulator/DeliveryBench/Linux/gym_citynav/Binaries/Linux/gym_citynav`, package asset revision `64a4889d72a2c7b792d242083e53a926b43ceaa9`. The original recorded run's `05_Experiment/results/interaction-demo-345795/simulator.log` reports UE `5.3.2-29314046`. The Python task checkout revision is `af7146c9e09edc06137c2c9e330190a0a177fdc8`. Standalone `SimWorld/Linux/` is a different package; the source checkout revision is not asserted to be this binary's source revision. Current selectors use simulator ID `simworld_delivery` plus task ID `deliverybench`. The unified online adapter is in `04_AgentTask/Demos/spatial_agent/delivery.py`; its server and the existing GPU launcher use the shared `run.py`. This consolidation has not run a fresh UE/GPU or model-weight session.
