#!/usr/bin/env bash
# Re-sync the conda env after ``git pull`` without removing extra packages.
#
# CHPC: the Miniforge *module* base is read-only. You need a personal env in
# ~/.conda/envs/anisotropy-hpc (created by this script). Always:
#
#   module load miniforge3/25.11.0
#   bash hpc/sync_env.sh
#
# Usage (from repo .../toys/anisotropy):
#   bash hpc/sync_env.sh

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
CONDA_BASE="$(conda info --base)"

_resolve_env_prefix() {
  conda env list | awk -v e="$ENV_NAME" '$1==e {print $NF; exit}'
}

ENV_PREFIX="$(_resolve_env_prefix)"
if [[ -z "$ENV_PREFIX" ]]; then
  echo "Creating personal env '$ENV_NAME' (writable under \$HOME/.conda/envs/) ..."
  conda env create -n "$ENV_NAME" -f "$ENV_FILE"
  ENV_PREFIX="$(_resolve_env_prefix)"
fi

if [[ -z "$ENV_PREFIX" || ! -d "$ENV_PREFIX" ]]; then
  echo "Could not find conda env '$ENV_NAME' after create." >&2
  exit 1
fi

if [[ "$ENV_PREFIX" == "$CONDA_BASE" ]] || [[ "$ENV_PREFIX" == *"/sys/installdir/"* ]]; then
  echo "ERROR: env '$ENV_NAME' resolves to the read-only module base:" >&2
  echo "  $ENV_PREFIX" >&2
  echo "Remove it and recreate in your home directory:" >&2
  echo "  conda env remove -n $ENV_NAME" >&2
  echo "  conda env create -n $ENV_NAME -f $ENV_FILE" >&2
  exit 1
fi

ENV_PYTHON="${ENV_PREFIX}/bin/python"
if [[ ! -x "$ENV_PYTHON" ]]; then
  echo "Missing $ENV_PYTHON" >&2
  exit 1
fi

echo "ENV_PREFIX=$ENV_PREFIX"
echo "CONDA_BASE=$CONDA_BASE (read-only module — do not pip install here)"

conda env update -n "$ENV_NAME" -f "$ENV_FILE"

conda install -n "$ENV_NAME" -y -c conda-forge \
  "scikit-image>=0.22" \
  "numpy>=1.26" \
  "scipy>=1.11" \
  "propka>=3.5"

if [[ "$ENV_FILE" == *viz* ]] || grep -q '^[[:space:]]*- pyvista' "$ENV_FILE" 2>/dev/null; then
  conda install -n "$ENV_NAME" -y -c conda-forge "pyvista>=0.43" vtk
fi

export PIP_USER=0
export PYTHONNOUSERSITE=1

# Install with the env's Python by full path (never the module base python).
"$ENV_PYTHON" -m pip install -e . --no-deps --no-user

"$ENV_PYTHON" - <<'PY'
import importlib
import sys

print("sys.executable:", sys.executable)
if "/sys/installdir/" in sys.executable:
    raise SystemExit("Still using module base Python — wrong env")

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

echo ""
echo "Done. In Slurm and interactive shells:"
echo "  module load miniforge3/25.11.0"
echo "  source \"\$(conda info --base)/etc/profile.d/conda.sh\""
echo "  conda activate ${ENV_NAME}"
echo "  # CONDA_PREFIX should be: ${ENV_PREFIX}"
