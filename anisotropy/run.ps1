# Prefer conda env "anisotropy", else local .venv, else toys/.venv
$CondaPy = Join-Path $env:CONDA_PREFIX "python.exe"
$LocalPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$ToysRoot = Split-Path -Parent $PSScriptRoot
$ToysPy = Join-Path $ToysRoot "toys\.venv\Scripts\python.exe"
if ($env:CONDA_DEFAULT_ENV -eq "anisotropy" -and (Test-Path $CondaPy)) {
    $Python = $CondaPy
} elseif (Test-Path $LocalPy) {
    $Python = $LocalPy
} elseif (Test-Path $ToysPy) {
    $Python = $ToysPy
} else {
    Write-Error "No environment found. Run: conda activate anisotropy  (see CONDA.md)"
    exit 1
}
$ScriptDir = $PSScriptRoot
if ($args.Count -gt 0 -and $args[0] -match '\.py$') {
    & $Python @args
} else {
    & $Python (Join-Path $ScriptDir "fit_protein_mesh.py") @args
}
