#!/bin/bash
# Fresh native generation audit (controls -> global-memory variables -> expectations) for the
# six AI2-THOR tasks. No model service: AI2-THOR renders on CPU (lavapipe), so one MIG slice
# (1/7 of a full H200 rate) is the cheapest allocation. Partial results stream to cases.jsonl.
#SBATCH --job-name=task-gen-audit
#SBATCH --account=video
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/gpfs/scrubbed/xuebiz/SpatialAgent/05_Experiment/logs/task-gen-audit-%j.log
set -euo pipefail
ROOT=/gpfs/scrubbed/xuebiz/SpatialAgent
OUT="$ROOT/05_Experiment/results/task-generation-audit-native-$SLURM_JOB_ID"
cd "$ROOT"
# CPU-rendered simulator only: hide the MIG device so Unity does not map CUDA to Vulkan.
unset CUDA_VISIBLE_DEVICES
export SPATIAL_THOR_SOFTWARE=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
export LP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OMP_NUM_THREADS=1
export XDG_CONFIG_HOME="/tmp/task-gen-audit-$SLURM_JOB_ID"
export XDG_CACHE_HOME="$ROOT/.cache"
mkdir -p "$XDG_CONFIG_HOME"
trap 'rm -rf "$XDG_CONFIG_HOME" "$OUT/thor-runtime"' EXIT
"$ROOT/02_Simulator/AI2THOR/runtime/python/bin/python" -u scripts/experiments/task_generation_audit.py \
  --source live --seeds 3000 3001 --output "$OUT" \
  --tasks missing_object moved_object memory_maze dependency_doors relational_rearrangement dynamic_replanning "$@"
