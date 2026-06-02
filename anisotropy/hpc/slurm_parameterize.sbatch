#!/bin/bash
#SBATCH --job-name=aniso-param
#SBATCH --output=slurm-param-%j.out
#SBATCH --error=slurm-param-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

REPO="${REPO:-$HOME/toys/anisotropy}"
CONDA_ENV="${CONDA_ENV:-anisotropy-hpc}"
RUN_DIR="${RUN_DIR:-$SCRATCH/toys_runs/9yp6}"

PDB="${PDB:-$RUN_DIR/9yp6.pdb}"
MESH="${MESH:-$RUN_DIR/9yp6.ply}"
NPZ="${NPZ:-$RUN_DIR/patch_features.npz}"

module load miniforge3/25.11.0
# shellcheck source=hpc/resolve_conda_env.sh
source "$REPO/hpc/resolve_conda_env.sh"
anisotropy_setup_conda "${CONDA_ENV:-anisotropy-hpc}"
anisotropy_verify_imports "$ANISOTROPY_PYTHON"
cd "$REPO"

export OMP_NUM_THREADS=1

"$ANISOTROPY_PYTHON" parameterize_mesh.py \
  "$PDB" "$MESH" \
  -o "$NPZ" \
  --charge-model ff19sb \
  --pka-source propka

echo "Wrote $NPZ"
