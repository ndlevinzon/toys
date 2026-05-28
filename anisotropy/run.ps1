# Prefer local anisotropy/.venv, else cryo-EM toys/.venv (avoids MGLTools python on PATH).
$LocalPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$ToysRoot = Split-Path -Parent $PSScriptRoot
$ToysPy = Join-Path $ToysRoot "toys\.venv\Scripts\python.exe"
if (Test-Path $LocalPy) {
    $Python = $LocalPy
} elseif (Test-Path $ToysPy) {
    $Python = $ToysPy
} else {
    Write-Error "No venv found.`n  Local: $LocalPy`n  Toys:  $ToysPy`nCreate: py -3 -m venv .venv; .\.venv\Scripts\pip install -e `".[view]`""
    exit 1
}
$ScriptDir = $PSScriptRoot
if ($args.Count -gt 0 -and $args[0] -match '\.py$') {
    & $Python @args
} else {
    & $Python (Join-Path $ScriptDir "fit_protein_mesh.py") @args
}
