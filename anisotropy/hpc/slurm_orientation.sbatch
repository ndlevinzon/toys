#!/bin/bash
#SBATCH --job-name=aniso-orient
#SBATCH --output=slurm-orient-%j.out
#SBATCH --error=slurm-orient-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --partition=YOUR_PARTITION

set -euo pipefail

# --- paths (edit for your site) ---
REPO="${REPO:-$HOME/toys/anisotropy}"
CONDA_ENV="${CONDA_ENV:-anisotropy-hpc}"
RUN_DIR="${RUN_DIR:-$SCRATCH/toys_runs/9yp6}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/runs/orient_${SLURM_JOB_ID}}"

PDB="${PDB:-$RUN_DIR/9yp6.pdb}"
MESH="${MESH:-$RUN_DIR/9yp6.ply}"
ISING_PARAMS="${ISING_PARAMS:-$REPO/hpc/ising_params.hpc.yaml}"

# --- modules (edit: module avail conda miniforge anaconda) ---
# module load miniforge
# conda activate must run after conda is on PATH (often via module)

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$REPO"

# One thread per process; multiprocessing uses SLURM_CPUS_PER_TASK workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CPUS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "$OUT_DIR"

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "REPO=$REPO OUT_DIR=$OUT_DIR CPUS=$CPUS"

python orientation_sample.py \
  "$PDB" "$MESH" \
  --ising-params "$ISING_PARAMS" \
  --outdir "$OUT_DIR" \
  --no-render \
  --parallel-workers "$CPUS" \
  --pka-source table \
  --verbose-console

echo "Done at $(date)"
echo "Outputs in $OUT_DIR"
