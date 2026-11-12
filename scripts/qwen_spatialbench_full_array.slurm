#!/usr/bin/env bash
# Run the 12 score labels that have a usable local evaluation protocol.
# Each array element requests one H200; Slurm caps concurrency at four.
# OpenEQA is omitted because its official judge model is no longer available.
#SBATCH --job-name=qwen38-spatial-full
#SBATCH --account=video
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --array=0-6%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scrubbed/xuebiz/SpatialAgent/05_Experiment/logs/qwen3.8-array-%A_%a.log
set -euo pipefail

ROOT=/gpfs/scrubbed/xuebiz/SpatialAgent
case "${BENCH_BATCH:-small}" in
  small)
    BENCHMARKS=(pointbench refspatial robospatial-poi where2place erqa robospatial-vq sat)
    TASKS=(pointbench_json refspatial_json robo_spatial_context where2place_json erqa 'robo_spatial_configuration,robo_spatial_compatibility' sat)
    ;;
  medium)
    BENCHMARKS=(blink cv-bench embspatial vsi-bench)
    TASKS=(blink cv_bench embspatial vsibench_multiimage)
    ;;
  large)
    BENCHMARKS=(mindcube)
    TASKS=(mindcube_full)
    ;;
  *)
    echo "Unknown BENCH_BATCH: ${BENCH_BATCH}" >&2
    exit 2
    ;;
esac

index=${SLURM_ARRAY_TASK_ID:?Slurm array index is required}
if (( index < 0 || index >= ${#BENCHMARKS[@]} )); then
  echo "Invalid array index: $index" >&2
  exit 2
fi

export BENCH_RUN_ID="qwen3.8-${BENCHMARKS[$index]}"
export BENCH_OUTPUT_DIR="$ROOT/05_Experiment/results/spatialbench_qwen3.8/final-$(date +%Y%m%d)/benchmarks/${BENCHMARKS[$index]}"
export BENCH_TASKS="${TASKS[$index]}"
export QWEN_SERVER_PORT=$((18300 + index))
unset BENCH_LIMIT || true

exec bash "$ROOT/scripts/qwen_spatialbench.slurm"
