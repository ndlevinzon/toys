# Conda setup for anisotropy

All runtime dependencies are installed from **conda-forge**, then the `anisotropy` package itself is installed in editable mode with `pip install -e .` (declared under `pip:` in the YAML files).

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Miniforge](https://github.com/conda-forge/miniforge) (recommended on Linux/HPC)
- Or [Mambaforge](https://github.com/conda-forge/miniforge) for faster solves (`mamba` instead of `conda`)

## Full environment (laptop + visualization)

From the **`anisotropy/`** directory (where `environment.yml` lives):

```bash
cd path/to/toys/anisotropy
conda env create -f environment.yml
conda activate anisotropy
```

Windows (Anaconda Prompt or PowerShell with `conda` on PATH):

```powershell
cd C:\Users\ndlev\OneDrive\Documents\Research\thesis\toys\anisotropy
conda env create -f environment.yml
conda activate anisotropy
```

Verify:

```bash
python -c "import anisotropy, pyvista, propka; print('ok')"
python -m pytest tests/ -q
```

## Headless HPC (no PyVista)

```bash
conda env create -f environment-hpc.yml
conda activate anisotropy-hpc
```

Use `--no-render` on `orientation_sample.py`. See [hpc/README.md](hpc/README.md).

## Update after `git pull`

```bash
conda activate anisotropy
conda env update -f environment.yml --prune
pip install -e . --no-deps
```

## Optional: documentation tools

```bash
conda activate anisotropy
conda env update -f environment-docs.yml
sphinx-build -b html docs docs/_build/html
```

## Remove environment

```bash
conda deactivate
conda env remove -n anisotropy
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `conda: command not found` | Install Miniconda/Miniforge; restart shell |
| Slow solve | `conda install -n base conda-libmamba-solver` then `conda config --set solver libmamba` |
| `pip install -e .` fails | Run `conda activate` first; `cd` to `anisotropy/` |
| VTK / PyVista errors on HPC | Use `environment-hpc.yml` and `--no-render` |
| PROPKA not found | `conda install -c conda-forge propka` or recreate env |

## Relation to `requirements.txt`

`requirements.txt` mirrors pip package names for non-conda installs. **Conda is the supported path** for reproducible science environments; keep `environment.yml` in sync when you add dependencies.
