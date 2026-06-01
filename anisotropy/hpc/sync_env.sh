#!/usr/bin/env bash
# Re-sync the conda env after ``git pull`` without removing extra packages.
#
# Usage (from repo .../toys/anisotropy):
#   bash hpc/sync_env.sh
#
# Headless only (no PyVista in YAML — use discrete curvatures / --no-render):
#   ANISOTROPY_ENV_FILE=environment-hpc.yml ANISOTROPY_CONDA_ENV=anisotropy-hpc bash hpc/sync_env.sh
#
# HPC + PyVista (recommended if you conda-installed pyvista by hand before):
#   ANISOTROPY_ENV_FILE=environment-hpc-viz.yml ANISOTROPY_CONDA_ENV=anisotropy-hpc-viz bash hpc/sync_env.sh
#
# Laptop / full pipeline:
#   ANISOTROPY_ENV_FILE=environment.yml ANISOTROPY_CONDA_ENV=anisotropy bash hpc/sync_env.sh

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

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Updating conda env '$ENV_NAME' from $ENV_FILE (no --prune)"
  conda env update -n "$ENV_NAME" -f "$ENV_FILE"
else
  echo "Creating conda env '$ENV_NAME' from $ENV_FILE"
  conda env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

conda activate "$ENV_NAME"

# env update sometimes skips newly listed packages on old envs — install explicitly.
conda install -n "$ENV_NAME" -y -c conda-forge \
  "scikit-image>=0.22" \
  "numpy>=1.26" \
  "scipy>=1.11" \
  "propka>=3.5"

if [[ "$ENV_FILE" == *viz* ]] || grep -q '^[[:space:]]*- pyvista' "$ENV_FILE" 2>/dev/null; then
  conda install -n "$ENV_NAME" -y -c conda-forge "pyvista>=0.43" vtk
fi

pip install -e . --no-deps

python - <<'PY'
import importlib
import sys

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

echo "Active env: $CONDA_DEFAULT_ENV ($(which python))"
