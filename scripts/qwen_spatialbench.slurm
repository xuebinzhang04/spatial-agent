#!/bin/bash
#SBATCH --job-name=qwen38-spatial
#SBATCH --account=video
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/scrubbed/xuebiz/SpatialAgent/05_Experiment/logs/spatial-%j.log
set -euo pipefail
ulimit -c 0
module load gcc/13.4.0 cuda/13.0.0
ROOT=/gpfs/scrubbed/xuebiz/SpatialAgent
export PATH=/gpfs/scrubbed/xuebiz/conda/envs/spatialagent/bin:$PATH
export HF_HOME="$ROOT/03_SpatialBench/benchmarks/.hf_cache"
export XDG_CACHE_HOME="$ROOT/.cache"
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH=/gpfs/scrubbed/xuebiz/conda/envs/spatialagent/lib:${LD_LIBRARY_PATH:-}
RUN_ID=${BENCH_RUN_ID:-qwen38-spatial-${SLURM_JOB_ID}}
QWEN_SERVER_PORT=${QWEN_SERVER_PORT:-8000}
QWEN_TENSOR_PARALLEL_SIZE=${QWEN_TENSOR_PARALLEL_SIZE:-1}
export BENCH_RUN_ID="$RUN_ID"
export QWEN_SERVER_PORT
export QWEN_TENSOR_PARALLEL_SIZE
OUT=${BENCH_OUTPUT_DIR:-"$ROOT/05_Experiment/results/$RUN_ID"}
mkdir -p "$OUT"
module list > "$OUT/modules.txt" 2>&1
scontrol show job "$SLURM_JOB_ID" > "$OUT/slurm-job.txt"
nvidia-smi > "$OUT/gpu.txt"
vllm serve "$ROOT/01_Models/Qwen3.8-27B" \
  --served-model-name Qwen3.8-27B --host 127.0.0.1 --port "$QWEN_SERVER_PORT" \
  --dtype bfloat16 --tensor-parallel-size "$QWEN_TENSOR_PARALLEL_SIZE" --max-model-len 32768 \
  --max-num-seqs 1 --gpu-memory-utilization 0.65 --enforce-eager \
  --limit-mm-per-prompt '{"image":32,"video":0}' --reasoning-parser qwen3 \
  > "$OUT/vllm.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo 'vLLM exited early; inspect vllm.log' >&2
    exit 1
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:${QWEN_SERVER_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done
python -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2)' \
  "http://127.0.0.1:${QWEN_SERVER_PORT}/health"
bash "$ROOT/scripts/run_spatialbench_eval.sh"
