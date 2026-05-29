#!/bin/bash
#SBATCH --job-name=aniso-array
#SBATCH --output=slurm-array-%A_%a.out
#SBATCH --error=slurm-array-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-0
# One task per subdirectory of BASE (set max index after listing dirs).

set -euo pipefail

REPO="${REPO:-$HOME/toys/anisotropy}"
CONDA_ENV="${CONDA_ENV:-anisotropy-hpc}"
BASE="${BASE:-$SCRATCH/toys_runs}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$REPO"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
CPUS="${SLURM_CPUS_PER_TASK:-16}"

# Map SLURM_ARRAY_TASK_ID to protein folder (sorted names)
mapfile -t DIRS < <(find "$BASE" -mindepth 1 -maxdepth 1 -type d | sort)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
RUN_DIR="${DIRS[$IDX]}"
NAME="$(basename "$RUN_DIR")"

PDB="$RUN_DIR/${NAME}.pdb"
MESH="$RUN_DIR/${NAME}.ply"
OUT_DIR="$RUN_DIR/runs/orient_${SLURM_ARRAY_JOB_ID}_${IDX}"

mkdir -p "$OUT_DIR"
echo "Array task $IDX -> $RUN_DIR"

python orientation_sample.py \
  "$PDB" "$MESH" \
  --ising-params "$REPO/hpc/ising_params.hpc.yaml" \
  --outdir "$OUT_DIR" \
  --no-render \
  --parallel-workers "$CPUS" \
  --pka-source table
