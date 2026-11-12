#!/bin/bash
set -euo pipefail
ROOT=/gpfs/scrubbed/xuebiz/SpatialAgent
export PATH=/gpfs/scrubbed/xuebiz/conda/envs/spatialagent/bin:$PATH
export HF_HOME="$ROOT/03_SpatialBench/benchmarks/.hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$ROOT/.cache"
export OPENAI_API_KEY=local-unused
# Match the Qwen-native task configured for the Point-Bench score label.
TASKS=${BENCH_TASKS:-pointbench_json}
RUN_ID=${BENCH_RUN_ID:-qwen38-spatial-${SLURM_JOB_ID:-manual}}
QWEN_SERVER_PORT=${QWEN_SERVER_PORT:-8000}
OUT=${BENCH_OUTPUT_DIR:-"$ROOT/05_Experiment/results/$RUN_ID"}
mkdir -p "$OUT"
cp "$ROOT/03_SpatialBench/eval_configs/eval_matrix.json" "$OUT/eval_matrix.json"
git -C "$ROOT/03_SpatialBench/lmms-eval" rev-parse HEAD > "$OUT/lmms_eval_commit.txt"
git -C "$ROOT/03_SpatialBench/lmms-eval" diff > "$OUT/lmms_eval.patch"
python -m pip freeze > "$OUT/requirements.txt"
printf '%s\n' "$TASKS" > "$OUT/tasks.txt"
args=(--model openai
      --model_args "model=Qwen3.8-27B,base_url=http://127.0.0.1:${QWEN_SERVER_PORT}/v1,api_key=local-unused,timeout=900,max_retries=2,num_concurrent=1,max_frames_num=32,enable_thinking_kwarg=false"
      --tasks "$TASKS" --batch_size 1 --seed 0 --log_samples --process_with_media
      --include_path "$ROOT/03_SpatialBench/eval_configs/custom_tasks"
      --output_path "$OUT")
if [[ -n "${BENCH_LIMIT:-}" ]]; then
  args+=(--limit "$BENCH_LIMIT")
fi
python -m lmms_eval eval "${args[@]}" 2>&1 | tee "$OUT/eval.log"
if grep -q 'Error during evaluation:' "$OUT/eval.log"; then
  echo 'lmms-eval reported an evaluation error' >&2
  exit 1
fi
