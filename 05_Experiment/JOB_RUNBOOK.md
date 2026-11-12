# DeliveryBench scheduling and results

Artifact retention (2026-10-05): all earlier DeliveryBench, Harness, and SpatialBench smoke result directories and experiment logs were deleted at the user's request. The current SpatialBench full evaluation remains. The job history below is diagnostic context; old artifact paths are historical and no longer exist.

## Module compatibility comes first

User preference (2026-09-28): before changing packages to fix compatibility, inspect `module spider`, `module list`, and `module avail` for server-supported versions and solutions.

```bash
module list
module spider cuda
module spider cuda/13.0.0
module avail
module load gcc/13.4.0 cuda/13.0.0
module avail cuda
command -v nvcc
nvcc --version
```

Observed: initial shell loaded only conda. CUDA is hidden in the hierarchical module tree until GCC is loaded. CUDA 13.0.0 and 13.3.1 require gcc/13.4.0. Use CUDA 13.0.0 to match installed Torch 2.13.0+cu130. Successful `nvcc`, sqlite3 and Torch import checks do not replace GPU inference validation.

## Scheduling rules

Login node: edit, download, inspect scheduler, summarize small outputs. GPU inference and Unreal simulation: Slurm allocation only.
Check `sinfo -s`, `squeue --me`, `hyakusage` before submission. Explicit account `video`, partition `gpu-h200`, QoS `normal`, one GPU, eight CPU cores, 50 GB host RAM, 30-minute limit. The current cluster maximum is eight cores per GPU. Nominal upper cost is $0.45 at $0.90/hour; actual usage follows Slurm elapsed time and current billing rules.

```bash
cd /gpfs/scrubbed/xuebiz/SpatialAgent
sbatch scripts/qwen_delivery_smoke.slurm
squeue --me
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,NodeList,MaxRSS
tail -f 05_Experiment/logs/smoke-JOBID.log
```

Do not submit concurrent runs against this checkout: configuration and simulator paths are shared. Cancel a failed allocation promptly with `scancel JOBID`. Service PID checks now end the job early when a service dies.

## Records

Each `05_Experiment/results/qwen3.8-smoke-JOBID/` contains configurations, model-service and simulator logs, GPU details, package lock, loaded modules, nvcc version, Slurm job details and submitted scripts. Episode results appear in `reports/`, observation images and raw outputs in `trajectories/`, and CSV aggregates in `summary/`. `episode_status.json` distinguishes terminated, truncated and failed episodes.

```bash
conda activate /gpfs/scrubbed/xuebiz/conda/envs/spatialagent
python scripts/render_demo.py 05_Experiment/results/qwen3.8-smoke-JOBID
```

Open `demo.html` with its containing run folder accessible. For a portable copy, copy the whole run folder; image and artifact links are relative. The demo displays only real artifacts, with missing reports explicitly marked.

## Test history

- 335788: ran on g002; weights loaded (51.1 GiB), Unreal rendering initialized; vLLM JIT failed because nvcc/CUDA_HOME were unavailable. Cancelled after 10m29s; nominal GPU cost approximately $0.16, not an invoice. No agent episode or benchmark score.
- 335792: failed after 35 seconds on g022; Unreal reported VK_ERROR_DEVICE_LOST. Fail-fast process monitoring worked.
- 335794: model service retains GCC/CUDA; simulator unloads those modules. Same g022 node rendered successfully and model passed the previous missing-nvcc point. This supports environment isolation as a workaround but does not prove the precise cause of the prior Vulkan failure. During startup, switched map via UnrealCV from fallback demo_1 to `/Game/DeliveryBench/Maps/small-city-11`; the full map path is now in the script, with a log-based guard against fallback scenes. Consult run artifacts for final status.
- 335794 completed the three-step smoke successfully as a Slurm job (3m57s, max RSS 52.4 GB). vLLM loaded Qwen3.8-27B BF16 and served 3 vision requests; UE loaded the requested map but logged missing `CityDatabase/blueprints/BP_Building_85`, and the courier's MOVE_TO was stagnant. Episode status is `truncated`, completed orders 0; it is an integration diagnostic, not a score. Images, prompts, outputs, report, summary CSV and offline HTML are under `05_Experiment/results/qwen3.8-smoke-335794/`.
- Downloaded the full official DeliveryBench Linux.zip (39.8 GB). It includes the dedicated `gym_citynav` UE build and the referenced CityDatabase blueprints. Job 335796 failed quickly because this standalone UE package could not resolve the full `/Game/DeliveryBench/Maps/small-city-11` URI. Next attempt uses the benchmark's documented short map-name argument and retains the fail-before-inference map guard.
- 335797: dedicated `gym_citynav` loaded `/Game/small-city-11` without missing-blueprint warnings. Qwen3.8-27B completed three vision decisions; VIEW_ORDERS, ACCEPT_ORDER and MOVE_TO all succeeded. The three-step limit truncated the episode with one open order and zero deliveries. Slurm completed in 2m39s on g022. See `05_Experiment/results/qwen3.8-smoke-335797/demo.html`. The script now checks for the exact `LoadMap(/Game/small-city-11)` success message.
- 335834: single-map, seed-0 full lifecycle evaluation submitted with `scripts/qwen_delivery_full.slurm`. One H200, 8 CPUs, 50 GB host RAM, 2h10m Slurm limit; estimated maximum $1.95. Native lifecycle limits are two simulated active hours, two wall-clock hours, or 200 VLM calls; the runner's step cap is 2000 so the previous three-step cutoff does not apply. Collect via `python scripts/collect_job.py 335834 --kind full` after completion. Result status and metrics must be checked before being called a benchmark result.
- 335834 failed after 25m41s at step 56: the model chose `WAIT(until='charge_done')` for a 10.8-minute simulated charging action, but the gym interface's 120-second wall-time action wait returned `timeout_waiting_action_done`. The partial report recorded five completed orders; it is not a valid full-life score. `run_deliverybench.py` now passes a 900-second action timeout, explicitly recorded in the full-run Slurm command. The original report, episode error and accounting remain under its run directory.
- 335856: retry of the same small-city-11, seed-0 full lifecycle with a 900-second action timeout, submitted after rechecking `hyakusage`, `squeue --me` and `sinfo -s`. Its 2h10m Slurm cap estimates at most $1.95; collect with `python scripts/collect_job.py 335856 --kind full` and inspect `episode_status.json` and the lifecycle stop reason.
- 335856 reached the native two-hour simulation limit and exported a complete one-map report after 84 VLM calls. It recorded five completed orders, six timed-out orders, 0% on-time rate, 3.4 average stars, and +14.74 in-game net earnings (+7.37 per lifecycle hour). The report is a valid native lifecycle result for small-city-11/seed 0, not a ten-map score. The outer gym interface had `_is_terminated()` hardcoded false, so the GPU job kept running after export; it was cancelled at 44m49s to avoid further charges. `result_validation.json` records native completion separately from Slurm cancellation, and `demo.html` links all artifacts. The interface now detects `_lifecycle_done` while waiting and in `_is_terminated()`, so later runs can exit normally.
- 335911: `small-city-13`, seed 0, submitted with `DELIVERY_MAP=small-city-13` after the lifecycle-done fix. The generic full-run Slurm script validates `LoadMap(/Game/$DELIVERY_MAP)` and saves map identity in `experiment_config.json`; run outcomes remain isolated by job ID. One H200, 2h10m cap, estimated maximum $1.95. Confirm its natural exit and collect with `python scripts/collect_job.py 335911 --kind full`.
- 335911 failed on g001 after 36 seconds: Unreal loaded `/Game/small-city-13` but crashed with Vulkan `VK_ERROR_DEVICE_LOST` before the agent episode started. No score. Rechecked `module list`, `module spider cuda/13.0.0`, and `module avail cuda` after GCC load; the expected GCC 13.4.0/CUDA 13.0.0 hierarchy is present. g001 and successful g010 both report NVIDIA driver 610.57.04, so there is no evidence for a global CUDA module mismatch. Its logs and empty-result demo are archived.
- 335914: node-pinned g010 retry for `small-city-13`, seed 0, submitted after live account check. Same one-H200/2h10m cap; this tests whether the g001 Vulkan crash is node-specific or transient.
- 335914 entered the small-city-13 episode on g010 and was later cancelled at 10m08s without a native lifecycle report. It has no comparable score. `collect_job.py` archived scheduler status and a demo listing available diagnostic artifacts. No GPU jobs remain queued as of that check.
- 335914 loaded `small-city-13` on g010 and completed early agent actions, confirming the g001 crash was not a universal map/module failure. The user then chose to stop at the completed single-map result, so 335914 was cancelled at 10m08s; it has no native lifecycle report and is excluded from scores. `squeue --me` was empty afterward. Total current-cycle user charge shown by `hyakusage` was $1.54 across nine ended jobs. No further maps are scheduled.

Collect status and regenerate the demo with `python scripts/collect_job.py JOBID`. This records scheduler state at query time, including failed runs.

Three-step smoke validates integration only. A full evaluation must use declared maps/seeds/repeats, native lifecycle limits, preserve failed runs and avoid interpreting partial-episode hourly metrics as comparable benchmark scores.

Official scheduling reference: https://hyak.uw.edu/docs/systems/tillicum/scheduling-jobs/
