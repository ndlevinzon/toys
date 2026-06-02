#!/usr/bin/env bash
# Resolve personal conda env and set ANISOTROPY_PYTHON (never rely on bare ``python``).
#
# Usage (after ``module load miniforge3/...``):
#   source "$(dirname "$0")/resolve_conda_env.sh"
#   anisotropy_setup_conda anisotropy-hpc
#   "$ANISOTROPY_PYTHON" -c "import skimage"

anisotropy_resolve_env_prefix() {
  local env_name="$1"
  local prefix=""

  if [[ -z "$env_name" ]]; then
    echo "anisotropy_resolve_env_prefix: env name required" >&2
    return 1
  fi

  if command -v conda >/dev/null 2>&1; then
    prefix="$(conda env list | awk -v e="$env_name" '$1==e {print $NF; exit}')"
  fi

  if [[ -z "$prefix" && -d "${HOME}/.conda/envs/${env_name}" ]]; then
    prefix="${HOME}/.conda/envs/${env_name}"
  fi

  if [[ -z "$prefix" || ! -d "$prefix" ]]; then
    echo "ERROR: conda env '${env_name}' not found." >&2
    echo "  module load miniforge3/25.11.0" >&2
    echo "  cd .../anisotropy && bash hpc/sync_env.sh" >&2
    echo "  conda env list" >&2
    return 1
  fi

  if [[ "$prefix" == *"/sys/installdir/"* ]]; then
    echo "ERROR: env '${env_name}' points at read-only module path:" >&2
    echo "  $prefix" >&2
    echo "  conda env remove -n ${env_name}" >&2
    echo "  bash hpc/sync_env.sh" >&2
    return 1
  fi

  if [[ ! -x "${prefix}/bin/python" ]]; then
    echo "ERROR: missing ${prefix}/bin/python" >&2
    return 1
  fi

  printf '%s' "$prefix"
}

# shellcheck disable=SC2120
anisotropy_setup_conda() {
  local env_name="${1:-${CONDA_ENV:-anisotropy-hpc}}"
  local prefix

  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not on PATH. Run: module load miniforge3/25.11.0" >&2
    return 1
  fi

  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"

  prefix="$(anisotropy_resolve_env_prefix "$env_name")" || return 1

  export CONDA_ENV="$env_name"
  export ANISOTROPY_ENV_PREFIX="$prefix"
  export ANISOTROPY_PYTHON="${prefix}/bin/python"

  # conda activate for CONDA_DEFAULT_ENV / some packages; PATH may still be wrong on CHPC.
  conda activate "$env_name" 2>/dev/null || true
  export PATH="${prefix}/bin:${PATH}"
  unset PYTHONUSERBASE
  export PIP_USER=0
  export PYTHONNOUSERSITE=1

  if [[ "${CONDA_PREFIX:-}" == *"/sys/installdir/"* ]]; then
    echo "WARNING: CONDA_PREFIX still on module base; using ANISOTROPY_PYTHON only." >&2
  fi

  if [[ "$(command -v python)" != "$ANISOTROPY_PYTHON" ]]; then
    echo "NOTE: bare 'python' is $(command -v python); jobs use ANISOTROPY_PYTHON=$ANISOTROPY_PYTHON" >&2
  fi
}

anisotropy_verify_imports() {
  local py="${1:-${ANISOTROPY_PYTHON:-}}"
  if [[ -z "$py" || ! -x "$py" ]]; then
    echo "anisotropy_verify_imports: set ANISOTROPY_PYTHON first" >&2
    return 1
  fi
  "$py" - <<'PY'
import importlib
import sys

print("executable:", sys.executable)
if "/sys/installdir/" in sys.executable:
    raise SystemExit("FATAL: still using module Miniforge Python")

for name in ("numpy", "scipy", "skimage", "propka", "anisotropy"):
    importlib.import_module(name)
    print("  OK", name)
PY
}
