#!/bin/bash
set -euo pipefail
module load gcc/13.4.0 cuda/13.0.0
ROOT=/gpfs/scrubbed/xuebiz/SpatialAgent
export PATH=/gpfs/scrubbed/xuebiz/conda/envs/spatialagent/bin:$PATH
export HF_HOME="$ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$ROOT/.cache"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH=/gpfs/scrubbed/xuebiz/conda/envs/spatialagent/lib:${LD_LIBRARY_PATH:-}
exec vllm serve "$ROOT/01_Models/Qwen3.8-27B" \
  --served-model-name Qwen3.8-27B --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 32768 \
  --max-num-seqs 1 --gpu-memory-utilization 0.65 --enforce-eager \
  --limit-mm-per-prompt '{"image":3,"video":0}' --reasoning-parser qwen3
