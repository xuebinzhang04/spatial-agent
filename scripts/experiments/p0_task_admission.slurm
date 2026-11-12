#!/bin/bash
# P0 admission and shortcut controls for frozen interaction cases. No model service:
# AI2-THOR renders on CPU (lavapipe), so the cheapest allocation (one MIG slice) is used.
#SBATCH --job-name=p0-admission
#SBATCH --account=video
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2  # MIG slices allow at most 2 CPUs per GPU
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scrubbed/xuebiz/SpatialAgent/05_Experiment/logs/p0-admission-%j.log
set -euo pipefail
ROOT=/gpfs/scrubbed/xuebiz/SpatialAgent
OUT="$ROOT/05_Experiment/results/p0-admission-$SLURM_JOB_ID"
cd "$ROOT"
# CPU-rendered simulator only: hide the MIG device so Unity does not map CUDA to Vulkan.
unset CUDA_VISIBLE_DEVICES
export SPATIAL_THOR_SOFTWARE=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
export LP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OMP_NUM_THREADS=1
export XDG_CONFIG_HOME="/tmp/p0-admission-$SLURM_JOB_ID"
export XDG_CACHE_HOME="$ROOT/.cache"
mkdir -p "$XDG_CONFIG_HOME"
"$ROOT/02_Simulator/AI2THOR/runtime/python/bin/python" -u scripts/experiments/p0_task_admission.py --output "$OUT" "$@"
