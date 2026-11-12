#!/bin/bash
# Record the four AI2-THOR demo showcase cases with the current generator at 512 px
# (privileged task oracle). CPU lavapipe rendering: one MIG slice, 2 CPUs.
#SBATCH --job-name=showcase-native
#SBATCH --account=video
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=debug
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/scrubbed/xuebiz/SpatialAgent/05_Experiment/logs/showcase-native-%j.log
set -euo pipefail
ROOT=/gpfs/scrubbed/xuebiz/SpatialAgent
cd "$ROOT"
unset CUDA_VISIBLE_DEVICES
export SPATIAL_THOR_SOFTWARE=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
export LP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OMP_NUM_THREADS=1
export XDG_CONFIG_HOME="/tmp/showcase-native-$SLURM_JOB_ID"
export XDG_CACHE_HOME="$ROOT/.cache"
mkdir -p "$XDG_CONFIG_HOME"
trap 'rm -rf "$XDG_CONFIG_HOME"' EXIT
"$ROOT/02_Simulator/AI2THOR/runtime/python/bin/python" -u scripts/experiments/record_showcase_cases.py \
  --native --output "$ROOT/05_Experiment/results/showcase-cases-native-$SLURM_JOB_ID"
