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

Use `--no-render` on `orientation_sample.py`. **`parameterize_mesh.py` runs without PyVista** (discrete cotangent / angle-deficit curvatures). For VTK-identical curvatures on a laptop, use `environment.yml`. See [hpc/README.md](hpc/README.md).

## Update after `git pull`

**`git pull` does not uninstall packages.** What usually removes `pyvista` / `scikit-image` is:

```bash
conda env update -f environment-hpc.yml --prune   # --prune deletes anything not in the YAML
```

`environment-hpc.yml` intentionally omits PyVista. Manual `conda install pyvista` is dropped the next time someone runs `--prune`.

Use the sync script (no `--prune`) and pick the YAML that matches what you need:

```bash
cd path/to/toys/anisotropy
source "$(conda info --base)/etc/profile.d/conda.sh"

# Laptop / full viz (pyvista + scikit-image in YAML)
ANISOTROPY_ENV_FILE=environment.yml ANISOTROPY_CONDA_ENV=anisotropy bash hpc/sync_env.sh

# HPC with PyVista (renders, optional VTK paths) — use this if you kept conda-installing pyvista
ANISOTROPY_ENV_FILE=environment-hpc-viz.yml ANISOTROPY_CONDA_ENV=anisotropy-hpc-viz bash hpc/sync_env.sh

# HPC headless only (--no-render; parameterize works without PyVista)
ANISOTROPY_ENV_FILE=environment-hpc.yml ANISOTROPY_CONDA_ENV=anisotropy-hpc bash hpc/sync_env.sh
```

Conda package names: **`scikit-image`** (import `skimage`), **`pyvista`**. There is no conda package named `skimage`.

### CHPC: read-only Miniforge / `Permission denied` on pip

The module `miniforge3` installs under `/uufs/.../sys/installdir/...` — **you cannot `pip install` there**. You need a **personal env** in `~/.conda/envs/anisotropy-hpc`.

```bash
module load miniforge3/25.11.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep anisotropy

cd /scratch/rai/vast1/u1116818/anisotropy/toys/anisotropy
bash hpc/sync_env.sh

conda activate anisotropy-hpc
echo "$CONDA_PREFIX"
# MUST be: /uufs/chpc.utah.edu/common/home/u1116818/.conda/envs/anisotropy-hpc
# NOT:     .../sys/installdir/.../miniforge3

~/.conda/envs/anisotropy-hpc/bin/python -c "import skimage, anisotropy; print('ok')"
```

If `conda env list` shows `anisotropy-hpc` pointing at `sys/installdir`, remove and recreate:

```bash
conda env remove -n anisotropy-hpc
conda env create -f environment-hpc.yml
```

### CHPC: `Defaulting to user installation` / `No module named skimage`

Symptom in Slurm (`run_ising.cpu` with `set -x`):

```text
PYTHON=/uufs/.../sys/installdir/.../miniforge3/.../bin/python
ModuleNotFoundError: No module named 'skimage'
```

`conda activate anisotropy-hpc` did **not** put your personal env first on `PATH`; bare `python` is still the **read-only module** interpreter (no scikit-image there).

**Fix (login node):**

```bash
module load miniforge3/25.11.0
cd .../anisotropy
bash hpc/sync_env.sh          # must end with OK skimage / anisotropy
conda env list | grep anisotropy-hpc
# path MUST be under $HOME/.conda/envs/..., NOT sys/installdir

~/.conda/envs/anisotropy-hpc/bin/python -c "import skimage; print('ok')"
```

**Slurm / batch:** after `git pull`, use updated `run_ising.cpu` / `hpc/slurm_*.sbatch` — they call `hpc/resolve_conda_env.sh` and run `"$ANISOTROPY_PYTHON"` instead of `python`.

If `sync_env.sh` says env not found, create it once:

```bash
conda env create -f environment-hpc.yml   # only if missing
bash hpc/sync_env.sh
```

If `conda env list` shows `anisotropy-hpc` under `sys/installdir`, remove and recreate (see above).

Optional: run sync automatically after every pull (once per clone):

```bash
cp hpc/git-hooks/post-merge .git/hooks/post-merge
chmod +x .git/hooks/post-merge
```

Edit `.git/hooks/post-merge` if you use `anisotropy-hpc` instead of `anisotropy-hpc-viz`.

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
| VTK / PyVista errors on HPC | Use `environment-hpc.yml` and `--no-render`; parameterize needs no PyVista |
| `No module named pyvista` on parameterize | `git pull` + `pip install -e .` (discrete curvature fallback) or use full `environment.yml` |
| PROPKA not found | `conda install -c conda-forge propka` or recreate env |

## Relation to `requirements.txt`

`requirements.txt` mirrors pip package names for non-conda installs. **Conda is the supported path** for reproducible science environments; keep `environment.yml` in sync when you add dependencies.
