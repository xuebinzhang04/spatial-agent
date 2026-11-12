# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Read [AGENTS.md](AGENTS.md) for the shared research loop and Claude Code/Codex handoff agreement.

## What this is

SpatialAgent is a research workspace (not a single package) for **interactive spatial learning environments for VLM agents**, aimed at later RL training. It combines a local Qwen3.8-27B model, several 3D simulators, a static spatial-benchmark panel (lmms-eval), and custom interactive tasks. It runs on UW Tillicum (Slurm, H200 GPUs, billed by GPU wall-clock time) from `/gpfs/scrubbed/xuebiz/SpatialAgent`. The workspace root tracks first-party source in Git; external checkouts, simulator runtimes, model weights, and experiment results are separate or ignored.

## Required reading before acting

Each of these files is canonical for its topic. Read the relevant one rather than relying on this summary.

- `workspace.md`: folder ownership, file placement, code/comment standards, execution rules. **Do not edit it** unless the user changes these principles.
- `00_Scope/`: the user's research intent. Read it for direction. **Never modify it.**
- `02_Simulator/REGISTRY.md`: simulator bundles, entrypoints, and *verified evidence level* for each.
- `05_Experiment/RECORD_CONTRACT.md`: every run needs `05_Experiment/results/<run-id>/run_metadata.json` (run ID includes the Slurm job ID; a shared `project_id` for each launch session) created **before** submission.
- `05_Experiment/JOB_RUNBOOK.md`: module setup, Slurm shape, diagnostic job history.
- `03_SpatialBench/eval_configs/README.md`: benchmark panel, scoring rules, Slurm eval usage.
- `archive/README.md`: where to put dated progress records (`archive/<topic>/YYYY-MM-DD-description.md`). Progress, job IDs, and scores go here, not in `workspace.md` or new top-level files.

## Architecture (cross-folder)

- **Model**: `01_Models/Qwen3.8-27B` (BF16) is served by vLLM as an OpenAI-compatible endpoint on `127.0.0.1:8000` (`scripts/serve_qwen.sh`, or inline in the `.slurm` scripts). Benchmarks, tasks, and the demo are all *clients* of this endpoint. None of them load weights themselves.
- **Simulators** (`02_Simulator/`): each bundle owns its executable and its Python runtime. Use the bundle's interpreter, not the main env:
  - AI2-THOR: `02_Simulator/AI2THOR/runtime/python/bin/python`
  - Habitat: `02_Simulator/Habitat/runtime/bin/python` (installed only; no task adapters)
  - DeliveryBench UE build `02_Simulator/DeliveryBench/Linux/gym_citynav/` (a dedicated SimWorld build, driven over UnrealCV port 9000). It is separate from standalone `02_Simulator/SimWorld/Linux/`.
- **Tasks** (`04_AgentTask/`):
  - `TaskFrameworks/spatial_interaction/`: the shared Gymnasium-style backend (`reset/step/render/close`) for AI2-THOR native tasks and the CPU-only Shape3D renderer. Also holds the episode schema, model adapter, and replay/validation tools.
  - `CustomAgentTasks/<task>/`: eight thin task entrypoints (`task.json`, `run.py`, `benchmarks/*.jsonl`). `TaskFrameworks/task_support.py` binds them to the shared backend. Put task logic in the framework and import it; don't copy it into tasks. `delivery_cases/` builds courier cases over DeliveryBench.
  - `SpatialAgentBench/`: external benchmark checkouts (DeliveryBench, EmbodiedWebAgents).
  - `Demos/spatial_agent/`: the single maintained web demo (FastAPI-style server, fixed port **8768**). It shows recorded episodes from `05_Experiment/results/` and has online reset/step/inference APIs.
- **Static benchmarks** (`03_SpatialBench/`): `lmms-eval/` (pinned commit plus local VSI-Bench patch), `benchmarks/<name>/` data plus a shared HF cache at `benchmarks/.hf_cache`, and `eval_configs/eval_matrix.json`, which maps the 13 score labels to lmms-eval tasks. Theory-of-Space and OpenEQA have special handling (see that README).
- **Experiments** (`05_Experiment/`): `results/<run-id>/` (manifest plus artifacts), `logs/` (Slurm output), and `queue_inspector/` (a web app that reads `run_metadata.json` for job cards).
- `scripts/`: launchers and helpers grouped as `simulator/`, `models/`, `datasets/`, `experiments/`, `common/`. Older root scripts migrate when changed.
- Retired implementations (symbolic Interaction Lab graph backend, legacy demos) live in `archive/agent-interaction/legacy-demos/`. Active APIs reject them.

## Common commands

Run from the `SpatialAgent` root unless a different directory is noted. The main env is `/gpfs/scrubbed/xuebiz/conda/envs/spatialagent`.

```bash
PY=/gpfs/scrubbed/xuebiz/conda/envs/spatialagent/bin/python
THOR_PY=02_Simulator/AI2THOR/runtime/python/bin/python

# Task-layout / entrypoint regression (main env)
$PY -m unittest discover -s 04_AgentTask/TaskFrameworks/tests -v
$PY 04_AgentTask/TaskFrameworks/tests/test_task_layout.py

# Framework unit tests (run in TaskFrameworks/spatial_interaction)
$PY -m unittest discover -s tests -v                          # shape backend; Unity tests skipped
SPATIAL_TEST_THOR=1 $THOR_PY -m unittest discover -s tests -v  # includes real Unity tests
# Single test: tests/ is not a package, so put it on the path and name module.Class.method
(cd tests && $PY -m unittest test_stage1.ShapeEnvironmentTests -v)

# Per-task / demo tests
$PY -m unittest discover -s 04_AgentTask/CustomAgentTasks/<task>/tests -v
$PY 04_AgentTask/Demos/spatial_agent/test_demo.py      # also test_session.py, test_server.py

# Full validator (writes its own run_metadata.json; use a fresh output dir)
$THOR_PY 04_AgentTask/TaskFrameworks/spatial_interaction/validate.py --thor --splits --output 05_Experiment/results/<new-run-id>

# Demo page (port 8768; restarts its own previous server)
python -u 04_AgentTask/Demos/spatial_agent/run.py
```

AI2-THOR on CPU (verified path; no GPU or display needed) needs `SPATIAL_THOR_SOFTWARE=1`, `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`, `LP_NUM_THREADS=2`, and a per-run `XDG_CONFIG_HOME`. See the spatial_interaction README. Shape3D tasks run on CPU in the main env.

GPU jobs (vLLM, Unreal/DeliveryBench, GPU THOR):

```bash
module load gcc/13.4.0 cuda/13.0.0   # CUDA modules are hidden until GCC is loaded; 13.0.0 matches Torch cu130
BENCH_TASKS=where2place BENCH_LIMIT=1 sbatch --export=ALL,BENCH_TASKS,BENCH_LIMIT scripts/qwen_spatialbench.slurm   # 1-sample smoke
sbatch scripts/qwen_delivery_smoke.slurm
python scripts/collect_job.py JOBID [--kind full]   # archive scheduler state + regenerate demo.html
```

## Cluster and run rules specific to this project

- Before any paid submission: check `hyakusage` (account `video`), `sinfo -s`, `squeue --me`, and live QoS/partition limits. Existing scripts use `--account=video --partition=gpu-h200 --qos=normal`, 1 GPU, 8 CPUs, 50G, and short time limits. Qwen3.8-27B BF16 (~51 GiB weights) does not fit a MIG slice, but Shape3D/CPU-THOR checks need no GPU at all.
- Before changing packages to fix compatibility, check `module spider` / `module avail` for a supported version. Do not alter Torch/vLLM in `spatialagent`. Incompatible stacks (e.g. Theory-of-Space) get their own environment.
- Simulator and model processes need separate module environments inside one job. The model service keeps GCC/CUDA loaded; the UE simulator unloads them (this worked around an earlier `VK_ERROR_DEVICE_LOST`). Clear `CUDA_VISIBLE_DEVICES` only for a CPU-rendered simulator process, never for the model server.
- Don't run concurrent DeliveryBench jobs against this checkout, because config and simulator paths are shared. Jobs must exit when the episode ends (an earlier hardcoded `_is_terminated()` kept a GPU billing). Cancel stragglers promptly.
- Evidence levels are distinct: import check < executable present < scene launch < task completion < model evaluation. Never present a smoke/limited run (`BENCH_LIMIT`, step caps) as a benchmark score. Withhold panel Overall Avg until all 13 labels are complete under one protocol. Keep failed, canceled, and partial runs.
- When moving files, update callers, configs, links, and docs. Leave historical manifests and logs unchanged, and explain old paths in a dated archive note.
- `/gpfs/scrubbed` is purged after 60 days of inactivity and has no backup. Remind the user to copy important results and records elsewhere (Kopah S3 / Lolo).
- Maintained code, identifiers, and canonical records are in English. Some task READMEs and `tasks.md` are in Chinese.
