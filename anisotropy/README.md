# Anisotropy — cryo-EM protein orientation at the air–water interface

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-Sphinx-4B8BBE.svg)](docs/index.rst)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Educational research code for **solvent-accessible surface (SAS) meshing**, **patch-wise interfacial chemistry** (ff19SB / PROPKA), and **Boltzmann orientation sampling** on SO(3) against a hybrid lattice-gas + electrostatic Hamiltonian at a vitrified water slab.

**LaTeX methods:** compile [`README.tex`](README.tex) for full equations.

**Pushing to GitHub?** See **[GITHUB_SETUP.md](GITHUB_SETUP.md)** (include list, `.gitignore`, Read the Docs, smoke tests).

---

## Features

| Capability | Highlights |
|------------|------------|
| **Mesh & shape** | Iterative SAS marching cubes; asphericity, principal axes |
| **Patches** | Curvature patches; hydropathy, charge, dipole, PROPKA pKa |
| **AWI slab** | Depth-dependent ε, κ, φ₀, E₀; asymmetric top/bottom interfaces |
| **Hamiltonian** | H_solv + hydration + screened Coulomb + film (+ optional flex) |
| **Sampling** | Uniform / hybrid / MCMC; **replica exchange** & simulated annealing |
| **Performance** | Fast evaluator, rotation-invariant H_el pair, parallel CPU chains |
| **CLI** | Append-only `anisotropy_run.log`; tqdm progress on console |

---

## Quick install (Conda — recommended)

Requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Miniforge](https://github.com/conda-forge/miniforge). Full details: **[CONDA.md](CONDA.md)**.

```powershell
git clone https://github.com/YOUR_USER/YOUR_REPO.git
cd YOUR_REPO\anisotropy
conda env create -f environment.yml
conda activate anisotropy
```

Headless clusters: `conda env create -f environment-hpc.yml` → `conda activate anisotropy-hpc`.

Configuration defaults: [`ising_params.yaml`](ising_params.yaml).

### Pip-only alternative

```powershell
py -3 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\pip install -e .
```

---

## Quick start (three commands)

Use your PDB and a fitted PLY mesh (or `--fit-mesh` on step 2):

```powershell
conda activate anisotropy
python fit_protein_mesh.py 1CRN.pdb -o 1crn_sas.ply
python parameterize_mesh.py 1CRN.pdb 1crn_sas.ply -o patch_features.npz --pka-source propka
python orientation_sample.py 1CRN.pdb 1crn_sas.ply --outdir orientation_diagnostics --no-render
```

Console shows a **progress bar only**; details append to **`anisotropy_run.log`**.

---

## Documentation

| Resource | Description |
|----------|-------------|
| [docs/](docs/) | Sphinx sources (Read the Docs) |
| [docs/HYBRID_HAMILTONIAN_PARAMETERS.md](docs/HYBRID_HAMILTONIAN_PARAMETERS.md) | Every YAML knob explained |
| [docs/user_guide/orientation_sampling.rst](docs/user_guide/orientation_sampling.rst) | β auto, hybrid, replica exchange |
| [.readthedocs.yaml](.readthedocs.yaml) | Hosted docs config |

Build locally:

```powershell
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

---

## Project layout

```
anisotropy/                 # import package
  awi_field.py
  fast_orientation_eval.py
  orientation_mcmc.py
  orientation_multimodal.py
  …
fit_protein_mesh.py         # CLI: PDB → SAS mesh
parameterize_mesh.py        # CLI: patches → .npz
orientation_sample.py       # CLI: SO(3) sampling + diagnostics
visualize_patches.py        # CLI: PyVista viewer
ising_params.yaml           # defaults
tests/
docs/
utils/.../forcefield_files/   # ff19SB amino19.lib (required)
```

---

## Cryo-EM toys monorepo (optional)

If this folder lives inside the larger **toys** thesis repo, you can use the shared venv:

```powershell
cd ..\toys
py -3 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\pip install -e "..\anisotropy[view]"
cd ..\anisotropy
..\toys\.venv\Scripts\python.exe orientation_sample.py ...
```

Windows helpers: `run.ps1` / `run.bat` (local `.venv` or toys venv).

---

## Orientation sampling (summary)

**Inverse temperature:** `beta: auto` in YAML calibrates β from energy spread (target ESS ≈ 20).

**Strategies** (`ising_params.yaml` → `sampling.strategy`):

| Strategy | Behavior |
|----------|----------|
| `uniform` | Random SO(3) only |
| `hybrid` | Uniform pool → β auto → refinement |
| `mcmc` | Refinement only |

**MCMC modes** (`sampling.mcmc.mode`):

| Mode | Use when |
|------|----------|
| `fixed_beta` | Fastest; standard Metropolis |
| `replica_exchange` | Multimodal landscapes (parallel tempering) |
| `simulated_annealing` | Explore → cool with optional reheat cycles |

**Performance flags** (`performance` in YAML): fast evaluator, invariant H_el pair, electrostatic cutoff, parallel MCMC chains — see [docs/user_guide/performance.rst](docs/user_guide/performance.rst).

**Outputs** (`--outdir`): energy traces, viewing-direction heatmaps & probability sphere, `top_poses.json`, optional `system_view_*.png`.

---

## Patch parameterization

Each patch \(f\) carries area, normal, curvatures, charge, dipole, hydropathy, etc. Default charges: **AMBER ff19SB** (`amino19.lib` under `utils/`). pKa: **PROPKA 3** when `pka_source` is `propka` or `auto`.

```powershell
.\.venv\Scripts\python parameterize_mesh.py protein.pdb protein.ply --charge-model ff19sb --pka-source propka
.\.venv\Scripts\python visualize_patches.py protein.ply patch_features.npz
```

---

## API (minimal)

```python
from anisotropy import load_pdb, fit_iterative_mesh, parameterize_mesh, load_ising_params

structure = load_pdb("protein.pdb")
mesh = fit_iterative_mesh(structure)
param = parameterize_mesh(mesh, structure, ph=7.0)
ising = load_ising_params()
```

---

## Tests

```powershell
.\.venv\Scripts\python -m pytest tests/ -q
```

---

## License

MIT — see [LICENSE](LICENSE). ff19SB force-field files retain AMBER upstream terms; see [GITHUB_SETUP.md](GITHUB_SETUP.md).
