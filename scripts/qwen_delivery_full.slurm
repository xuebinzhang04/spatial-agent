#!/bin/bash
#SBATCH --job-name=qwen38-delivery-full
#SBATCH --account=video
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=02:10:00
#SBATCH --output=/gpfs/scrubbed/xuebiz/SpatialAgent/05_Experiment/logs/full-%j.log
set -euo pipefail
ulimit -c 0
module load gcc/13.4.0 cuda/13.0.0
ROOT=/gpfs/scrubbed/xuebiz/SpatialAgent
export PATH=/gpfs/scrubbed/xuebiz/conda/envs/spatialagent/bin:$PATH
export CONDA_PREFIX=/gpfs/scrubbed/xuebiz/conda/envs/spatialagent
export QT_QPA_PLATFORM=offscreen
export XDG_CACHE_HOME="$ROOT/.cache"
export OPENAI_API_KEY=local-unused
export OMP_NUM_THREADS=4
RUN_ID="qwen3.8-full-${SLURM_JOB_ID}"
DELIVERY_MAP="${DELIVERY_MAP:-small-city-11}"
python "$ROOT/scripts/configure_deliverybench.py" --map "$DELIVERY_MAP" --run-id "$RUN_ID"
RUN_DIR="$ROOT/05_Experiment/results/$RUN_ID"
UE_ROOT="$ROOT/02_Simulator/DeliveryBench/Linux/gym_citynav"
test -x "$UE_ROOT/Binaries/Linux/gym_citynav"
nvidia-smi > "$RUN_DIR/gpu.txt"
python -m pip freeze > "$RUN_DIR/requirements.txt"
module list > "$RUN_DIR/modules.txt" 2>&1
nvcc --version > "$RUN_DIR/nvcc.txt"
scontrol show job "$SLURM_JOB_ID" > "$RUN_DIR/slurm-job.txt"
cp "$ROOT/scripts/serve_qwen.sh" "$RUN_DIR/serve_qwen.sh"
cp "$0" "$RUN_DIR/submitted-job.sh"
ue_pid=""
model_pid=""
cleanup() {
  if [[ -n "$ue_pid" ]]; then kill "$ue_pid" 2>/dev/null || true; fi
  if [[ -n "$model_pid" ]]; then kill "$model_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
bash "$ROOT/scripts/serve_qwen.sh" > "$RUN_DIR/vllm.log" 2>&1 &
model_pid=$!
cd "$UE_ROOT"
(
  module unload cuda/13.0.0 gcc/13.4.0
  exec ./Binaries/Linux/gym_citynav "$DELIVERY_MAP" -RenderOffscreen -nosound
) > "$RUN_DIR/simulator.log" 2>&1 &
ue_pid=$!
python "$ROOT/scripts/wait_services.py" --timeout 900 --pids "$model_pid" "$ue_pid"
if ! grep -Fq "LoadMap(/Game/$DELIVERY_MAP)" "$RUN_DIR/simulator.log"; then
  echo "Expected DeliveryBench map did not load" >&2
  exit 1
fi
if grep -Fq 'Cannot construct DefaultSceneRoot' "$RUN_DIR/simulator.log"; then
  echo "DeliveryBench map has missing blueprint dependencies" >&2
  exit 1
fi
cd "$ROOT/04_AgentTask/SpatialAgentBench/DeliveryBench"
timeout 7500 python -u -m vlm_delivery.scripts.run_deliverybench --max-steps 2000 --seed 0 --action-timeout-s 900 \
  > "$RUN_DIR/episode.log" 2>&1
python evaluation/agent_performance_analysis.py "$RUN_DIR/reports" -o "$RUN_DIR/summary"
python "$ROOT/scripts/render_demo.py" "$RUN_DIR"
