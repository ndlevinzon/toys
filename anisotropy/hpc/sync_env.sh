#!/usr/bin/env bash
# Re-sync the conda env after ``git pull`` without removing extra packages.
#
# On CHPC, load the same Miniforge module as your Slurm script first:
#   module load miniforge3/25.11.0
#
# Usage (from repo .../toys/anisotropy):
#   bash hpc/sync_env.sh
#
# Headless only (no PyVista):
#   ANISOTROPY_ENV_FILE=environment-hpc.yml ANISOTROPY_CONDA_ENV=anisotropy-hpc bash hpc/sync_env.sh
#
# HPC + PyVista:
#   ANISOTROPY_ENV_FILE=environment-hpc-viz.yml ANISOTROPY_CONDA_ENV=anisotropy-hpc-viz bash hpc/sync_env.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_FILE="${ANISOTROPY_ENV_FILE:-environment-hpc.yml}"
ENV_NAME="${ANISOTROPY_CONDA_ENV:-}"

if [[ -z "$ENV_NAME" ]]; then
  case "$ENV_FILE" in
    environment-hpc.yml) ENV_NAME="anisotropy-hpc" ;;
    environment-hpc-viz.yml) ENV_NAME="anisotropy-hpc-viz" ;;
    environment.yml) ENV_NAME="anisotropy" ;;
    *) echo "Set ANISOTROPY_CONDA_ENV when using ENV_FILE=$ENV_FILE" >&2; exit 1 ;;
  esac
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ROOT/$ENV_FILE" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not on PATH. On CHPC run first: module load miniforge3/25.11.0" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Updating conda env '$ENV_NAME' from $ENV_FILE (no --prune)"
  conda env update -n "$ENV_NAME" -f "$ENV_FILE"
else
  echo "Creating conda env '$ENV_NAME' from $ENV_FILE"
  conda env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

conda activate "$ENV_NAME"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "conda activate '$ENV_NAME' did not set CONDA_PREFIX — aborting." >&2
  exit 1
fi

ENV_PYTHON="${CONDA_PREFIX}/bin/python"
if [[ ! -x "$ENV_PYTHON" ]]; then
  echo "Missing $ENV_PYTHON" >&2
  exit 1
fi

echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "PYTHON=$ENV_PYTHON ($("$ENV_PYTHON" --version))"

# Install into the named env (works even if activate is flaky).
conda install -n "$ENV_NAME" -y -c conda-forge \
  "scikit-image>=0.22" \
  "numpy>=1.26" \
  "scipy>=1.11" \
  "propka>=3.5"

if [[ "$ENV_FILE" == *viz* ]] || grep -q '^[[:space:]]*- pyvista' "$ENV_FILE" 2>/dev/null; then
  conda install -n "$ENV_NAME" -y -c conda-forge "pyvista>=0.43" vtk
fi

# Never install into ~/.local when the env site-packages exists.
export PIP_USER=0
export PYTHONNOUSERSITE=1

PIP_LOG="$(mktemp)"
if ! "$ENV_PYTHON" -m pip install -e . --no-deps --no-user 2>&1 | tee "$PIP_LOG"; then
  echo "pip install failed" >&2
  exit 1
fi
if grep -q "Defaulting to user installation" "$PIP_LOG"; then
  echo "ERROR: pip fell back to ~/.local — conda env is not writable or not active." >&2
  echo "  CONDA_PREFIX=$CONDA_PREFIX" >&2
  echo "  Fix: module load miniforge3; conda activate $ENV_NAME; rerun sync_env.sh" >&2
  exit 1
fi
rm -f "$PIP_LOG"

"$ENV_PYTHON" - <<'PY'
import importlib
import sys

print("sys.executable:", sys.executable)
print("sys.prefix:", sys.prefix)

checks = ["anisotropy", "numpy", "scipy", "skimage", "propka"]
for name in checks:
    importlib.import_module(name)
    print(f"  OK  {name}")

try:
    import pyvista as pv
    print(f"  OK  pyvista {pv.__version__}")
except ImportError:
    print("  --  pyvista not installed (expected for environment-hpc.yml)")
PY

echo "Done. Slurm should use: conda activate ${ENV_NAME}"
echo "Verify: ${ENV_PYTHON} -c \"import skimage; print(skimage.__version__)\""
