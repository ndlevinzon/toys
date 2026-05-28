@echo off
REM Use toys\.venv Python 3 — avoids MGLTools Python 2.7 on PATH.
set "TOYS_PY=%~dp0..\toys\.venv\Scripts\python.exe"
if not exist "%TOYS_PY%" (
    echo Toys venv not found: %TOYS_PY%
    exit /b 1
)
if "%~1"=="" (
    "%TOYS_PY%" "%~dp0fit_protein_mesh.py"
) else if /i "%~x1"==".py" (
    "%TOYS_PY%" %*
) else (
    "%TOYS_PY%" "%~dp0fit_protein_mesh.py" %*
)
