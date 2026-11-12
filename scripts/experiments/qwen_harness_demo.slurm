#!/bin/bash
#SBATCH --job-name=qwen-harness-demo
#SBATCH --account=video
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/scrubbed/xuebiz/SpatialAgent/05_Experiment/logs/qwen-harness-%j.log
set -euo pipefail
ulimit -c 0

ROOT=/gpfs/scrubbed/xuebiz/SpatialAgent
export PATH=/gpfs/scrubbed/xuebiz/conda/envs/spatialagent/bin:$PATH
export CONDA_PREFIX=/gpfs/scrubbed/xuebiz/conda/envs/spatialagent
export QWEN_BASE_URL=http://127.0.0.1:8000/v1
export QWEN_MODEL=Qwen3.8-27B
OUT="$ROOT/05_Experiment/results/qwen-harness-${SLURM_JOB_ID}"
mkdir -p "$OUT"

bash "$ROOT/scripts/serve_qwen.sh" > "$OUT/vllm.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo 'vLLM exited; inspect vllm.log' >&2
    exit 1
  fi
  if curl -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    break
  fi
  sleep 5
done
curl -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null
python "$ROOT/04_AgentTask/Harness/demo.py" --mode qwen \
  --output "$OUT/grid_trace.json" | tee "$OUT/demo.log"
