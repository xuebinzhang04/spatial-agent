# SpatialBench evaluation workspace

`03_SpatialBench/` has exactly three direct directories: [`../lmms-eval/`](../lmms-eval/) for the unified framework, [`../benchmarks/`](../benchmarks/) for benchmark data and shared caches, and this directory for evaluation configuration and documentation. Dataset files are kept once; Hugging Face snapshots live in `../benchmarks/.hf_cache/hub`, and each benchmark's `dataset` link and manifest point to its pinned revision.

## MolmoAct2 panel

The 13 MolmoAct2 score labels configured here are **Point-Bench, RefSpatial, RoboSpatial-Poi, Where2Place, BLINK, CV-Bench, ERQA, EmbSpatial, MindCube, RoboSpatial-VQ, SAT, OpenEQA, and VSI-Bench**. `RoboSpatial-Home` is shared source data for Poi and VQ, not a separate score. Data are stored under `03_SpatialBench/benchmarks/<benchmark>/`; “MolmoAct2” is only the evaluation grouping documented here, not a directory layer. [`eval_matrix.json`](eval_matrix.json) maps score labels to lmms-eval tasks and metrics. Theory-of-Space is another SpatialBench dataset under `../benchmarks/Theory-of-Space/` and is outside this panel. DeliveryBench is an interactive AgentTask benchmark; its source/runtime assets live under `04_AgentTask/SpatialAgentBench/DeliveryBench` and `02_Simulator/DeliveryBench`.

## Theory-of-Space (separate interactive evaluation)

Theory-of-Space measures spatial belief construction through active exploration, then evaluates route and survey questions and cognitive-map properties. It is a valid SpatialBench benchmark, but it is not one of the 13 static MolmoAct2 panel scores and does not run through lmms-eval. Use the upstream runner at [`benchmarks/Theory-of-Space/official_code`](../benchmarks/Theory-of-Space/official_code), currently pinned to `d48f52f`, with its own model config [`theory_of_space_model_config.yaml`](theory_of_space_model_config.yaml).

The model config targets the local Qwen3.8 OpenAI-compatible endpoint at `http://127.0.0.1:8000/v1`; the model server must be running and advertise `Qwen3.8-27B`. The dataset ZIP is already downloaded, but has not been expanded. Extract its contents so that `run00`–`run99` are directly inside `03_SpatialBench/benchmarks/Theory-of-Space/room_data/3-room/` (flatten the archive's enclosing `tos_dataset_0127_3room_100runs` directory). The official setup script would instead create a separate `tos` conda environment and download another copy; do not pass `--run-exp` because its built-in example runs GPT-5.2.

The upstream environment is not currently ready in `spatialagent`: the runner import fails on missing `gymnasium`, and upstream pins a separate dependency stack including Torch 2.9.1/vLLM 0.14.0 that conflicts with this project's Qwen serving environment. Install/test ToS dependencies in an isolated environment (or audit a minimal API-client dependency set) before running; do not alter `spatialagent`'s Torch/vLLM packages. A one-seed vision/active smoke is the sensible first run:

```bash
cd /gpfs/scrubbed/xuebiz/SpatialAgent/03_SpatialBench/benchmarks/Theory-of-Space/official_code
python scripts/SpatialGym/spatial_run.py \
  --phase all --model-name qwen3.8-local --num 1 \
  --data-dir ../room_data/3-room \
  --output-root /gpfs/scrubbed/xuebiz/SpatialAgent/05_Experiment/results/theory-of-space-qwen3.8-smoke \
  --render-mode vision --exp-type active --inference-mode direct --enable-think 1 \
  --base-model /gpfs/scrubbed/xuebiz/SpatialAgent/03_SpatialBench/eval_configs/theory_of_space_model_config.yaml
```

After the smoke validates all phases and output files, a paper-style core run uses `--num 100 --render-mode text,vision --exp-type active,passive` and a new output directory. That covers 100 seeds across the four active/passive × text/vision settings; false-belief/revision experiments are separate (`explore_fb` and `cogmap_fb`) and must be run and reported separately. These are GPU/model-server runs; follow the Tillicum usage, queue, and module checks above before scheduling them. Do not combine ToS scores into the 13-score MolmoAct2 Overall Avg.

The unified framework is based on lmms-eval commit `dcbfd21a9fc59dcd5187a36a5b15099e27dd0817` (version 0.7.3). It has two local VSI-Bench changes: `token: false` for the public Hugging Face dataset, and an aggregation guard for smoke runs that contain fewer than all direction question types. Complete-run aggregation is unchanged. Each run records the framework commit and patch. GPU integration checks are indexed in [`SMOKE_STATUS.md`](SMOKE_STATUS.md); they do not constitute full benchmark results.

VSI-Bench's three source ZIP archives were removed after verifying all 512 required MP4 files under `../benchmarks/.hf_cache/vsibench/{arkitscenes,scannet,scannetpp}`. They can be reacquired from the pinned snapshot with `python scripts/download_assets.py`. MindCube's original `data.zip` remains as a separate source asset; its `MindCube_lmmseval` dataset is also downloaded for this framework.

## Environment and downloads

Use the existing environment:

```bash
conda activate /gpfs/scrubbed/xuebiz/conda/envs/spatialagent
python -m pip install -e /gpfs/scrubbed/xuebiz/SpatialAgent/03_SpatialBench/lmms-eval
python -m pip install 'decord>=0.6.0,<0.7'
python -m pip check  # decord 0.6.0 emits a platform-metadata warning on this host
```

Resume lmms-eval dataset downloads from the project root:

```bash
HF_HOME=/gpfs/scrubbed/xuebiz/SpatialAgent/03_SpatialBench/benchmarks/.hf_cache python scripts/download_spatialbench.py
```

The script records pinned revisions in each benchmark manifest. `scripts/run_spatialbench_eval.sh` expects a local Qwen3.8-27B OpenAI-compatible endpoint on port 8000, writes per-sample predictions to `05_Experiment/results/<run-id>/`, and accepts comma-separated lmms-eval task names in `BENCH_TASKS`. Use `03_SpatialBench/eval_configs/eval_matrix.json`; run tasks without `BENCH_LIMIT` for comparable results. Report RoboSpatial-VQ as the arithmetic mean of Configuration and Compatibility. Do not report panel Overall Avg until all 13 score labels have complete results under the same protocol.

## Slurm evaluation

On Tillicum, check `hyakusage`, `sinfo -s`, and `squeue --me` before submitting. `scripts/qwen_spatialbench.slurm` requests one H200, eight CPUs, 50 GB host RAM, and defaults to 30 minutes. A one-sample integration check is:

```bash
BENCH_TASKS=where2place BENCH_LIMIT=1 sbatch --export=ALL,BENCH_TASKS,BENCH_LIMIT scripts/qwen_spatialbench.slurm
```

For a full task, set `BENCH_TASKS`, omit `BENCH_LIMIT`, and choose a task-appropriate time limit, e.g. `sbatch --time=02:00:00 --export=ALL,BENCH_TASKS scripts/qwen_spatialbench.slurm`. The job serves Qwen BF16 with context 32768, one GPU, up to 32 images per prompt, and thinking disabled. `--process_with_media` is required because Where2Place and RefSpatial evaluate predicted points against image masks. Qwen emits coordinates on a 0–1000 grid; `eval_configs/custom_tasks/refspatial_json.yaml` groups three RefSpatial JSON splits using size-weighted accuracy. Identify these prompt/format variants when comparing with published results. VSI-Bench samples 32 video frames, so validate a small sample before a full run.

The full non-OpenEQA panel can be scheduled with `scripts/qwen_spatialbench_full_array.slurm`. It runs one benchmark per H200 array task, caps concurrency at four, and uses distinct local vLLM ports. Small and medium arrays are dependency-ordered with 4- and 8-hour task caps. MindCube can use a dedicated multi-GPU tensor-parallel job. Each output directory is named `05_Experiment/results/qwen3.8-<benchmark>/`; the active run and Slurm IDs are tracked in [`spatialbench_qwen3.8_full_run.json`](../../05_Experiment/results/spatialbench_qwen3.8_full_run.json). These jobs do not set a sample limit.

After full runs, pass `*_results.json` files to `scripts/summarize_spatialbench.py`. It rejects limited runs, combines RoboSpatial-VQ's two tasks with equal weight, and withholds Overall Avg until all 13 scores exist. Supply the official OpenEQA `*-metrics.json` via `--openeqa-metrics`; its 1–5 LLM-Match values are normalized to 0–1. The matrix in this directory uses 0–1 scores; multiply by 100 for percentage presentation.

OpenEQA's official 1,636 questions and judge source are under `../benchmarks/OpenEQA/source`. Its official Dropbox media link did not provide an archive, so `scripts/download_openeqa_media.py` fetched a pinned third-party mirror of HM3D and ScanNet RGB episode histories (~87 GB) into `../benchmarks/.hf_cache/hub`, linked at `../benchmarks/OpenEQA/media`; all 152 referenced histories are present. The custom `openeqa_predictions` task samples 32 resized frames per episode. `openeqa_response_rate` is a pipeline diagnostic, not a benchmark score. Export complete samples with `scripts/export_openeqa_predictions.py`, then use `scripts/judge_openeqa.sh` with `OPENAI_API_KEY`. The official evaluator is pinned to `gpt-4-1106-preview`, which OpenAI lists as shut down on 2026-03-26; a replacement judge/model would change the protocol and needs explicit approval before it is treated as comparable. OpenEQA must stay out of the 13-score Overall Avg until an accepted judging protocol is available.
