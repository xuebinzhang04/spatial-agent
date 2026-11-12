#!/bin/bash
# Candidate GPU rendering job. Check live account/budget/partition access first.
#SBATCH --job-name=stage1-action-tours
#SBATCH --account=video
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/scrubbed/xuebiz/SpatialAgent/05_Experiment/logs/stage1-action-tours-%j.log
set -euo pipefail
ulimit -c 0
ROOT=/gpfs/scrubbed/xuebiz/SpatialAgent
PY="$ROOT/02_Simulator/AI2THOR/runtime/python/bin/python"
RUN="$ROOT/05_Experiment/results/stage1-action-tours-$SLURM_JOB_ID"
CONFIG="$ROOT/05_Experiment/results/stage1-action-tours-config-$SLURM_JOB_ID"
mkdir -p "$CONFIG"
export XDG_CONFIG_HOME="$CONFIG"
export XDG_CACHE_HOME="$ROOT/.cache"
export OMP_NUM_THREADS=2
export SPATIAL_THOR_SOFTWARE=0
unset VK_ICD_FILENAMES
cd "$ROOT"
nvidia-smi > "$CONFIG/gpu.txt"
cp "$0" "$CONFIG/submitted-job.sh"
exec "$PY" -u scripts/experiments/prepare_stage1_action_tours.py \
    --output "$RUN" --native-size 448 --verify-replay
