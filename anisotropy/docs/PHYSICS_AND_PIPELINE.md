# Physics model and end-to-end pipeline

This document is the **integration map** for cryo-EM **preferential orientation at the
air–water interface (AWI)**. It links the staged workflow, Hamiltonian terms, YAML
configuration, HPC deployment, and what is rigorous vs approximate.

**Related deep dives:**

| Topic | Document |
|-------|----------|
| Every YAML knob | [HYBRID_HAMILTONIAN_PARAMETERS.md](HYBRID_HAMILTONIAN_PARAMETERS.md) |
| First-shell RISM + outer Ising | [RISM_LAYERED_SOLVATION.md](RISM_LAYERED_SOLVATION.md) |
| Slab Poisson–Boltzmann | [PB_SLAB_ELECTROSTATICS.md](PB_SLAB_ELECTROSTATICS.md) |
| Sampling & β | [user_guide/orientation_sampling.rst](user_guide/orientation_sampling.rst) (Sphinx) |
| HPC / Slurm | [../hpc/README.md](../hpc/README.md), [../hpc/DEPLOY.md](../hpc/DEPLOY.md) |
| Conda envs | [../CONDA.md](../CONDA.md) |

---

## 1. Scientific question

Cryo-EM particles at the AWI are not uniformly distributed over SO(3). We model

\[
\pi(\Omega) \propto \exp\bigl(-\beta\, H(\Omega)\bigr),
\qquad
H = H_{\mathrm{solv}} + H_{\mathrm{patch}} + H_{\mathrm{el}} + H_{\mathrm{film}} + H_{\mathrm{flex}},
\]

where \(\Omega\) is a rigid rotation of the protein in a **vitrified water slab** with
**asymmetric** top/bottom interfaces (sacrificial film vs more pristine AWI).

**What we sample today:** orientations \(\Omega\) only. **Lattice solvent occupancy**
\(n_i\) is fixed from a slab template (not summed over in a partition function \(Z\)).

**What is becoming more physical (defaults in `ising_params.yaml`):**

| Layer | Model | Module |
|-------|--------|--------|
| First solvation shell | 3D-RISM-inspired excess potential on voxels within 5 Å of surface | `rism_solvation.py` |
| Outer solvent | Lattice-gas / Ising on remaining solvent voxels | `lattice_solvent_hamiltonian.py` |
| Electrostatics | Linearized PB with \(\varepsilon(z)\), \(\kappa(z)\) from AWI profile | `pb_slab_solver.py` |
| AWI structure | Depth-dependent \(\varepsilon_\parallel,\varepsilon_\perp,\phi_0,\kappa\) | `awi_field.py` |
| Protein chemistry | Curvature patches, ff19SB charges, PROPKA | `patches.py` |

---

## 2. End-to-end pipeline (plugged in)

```text
  PDB
   │
   ▼
fit_protein_mesh.py          SAS mesh (marching cubes), shape anisotropy
   │
   ▼
parameterize_mesh.py         Patches on dual face graph; charge, pKa, hydropathy, dipole
   │                          (optional: save patch_features.npz for reuse / viz)
   ▼
orientation_sample.py        SO(3) sampling vs H(Ω); diagnostics + renders
   │
   ├── ising_params.yaml      All couplings, sampling, rism, electrostatics, output
   ├── CanonicalShellCache    RISM shell masks + distance field (rotation-invariant)
   ├── SlabPBSolver            FFT×z PB on lattice (per pose if method=pb_slab)
   └── FastOrientationEvaluator  Hot path when performance.use_fast_evaluator
```

### 2.1 Commands (laptop)

```bash
conda activate anisotropy
cd anisotropy

python fit_protein_mesh.py protein.pdb -o protein.ply
python parameterize_mesh.py protein.pdb protein.ply -o patch_features.npz --pka-source propka
python orientation_sample.py protein.pdb protein.ply \
  --ising-params ising_params.yaml \
  --outdir runs/orient_001
```

`orientation_sample.py` **re-runs** `parameterize_mesh` internally (it does not yet
load `patch_features.npz` automatically). Saving `.npz` is still useful for
`visualize_patches.py` and to avoid repeating PROPKA on the login node.

### 2.2 Commands (HPC)

```bash
module load miniforge3/25.11.0
cd $HOME/toys/anisotropy
bash hpc/sync_env.sh
conda activate anisotropy-hpc

sbatch hpc/slurm_orientation.sbatch   # uses hpc/ising_params.hpc.yaml, --no-render
```

After `git pull`: `bash hpc/sync_env.sh` (see [CONDA.md](../CONDA.md)).

---

## 3. Hamiltonian stack (current defaults)

### 3.1 Solvation — layered (`rism` + `solv`)

\[
H_{\mathrm{solv}} = H_{\mathrm{RISM}}^{(\mathrm{1st\ shell})} + H_{\mathrm{Ising}}^{(\mathrm{outer})}.
\]

- **First shell** (\(d \le\) `rism.first_shell_angstrom`, default 5 Å): site potentials from
  patch charges + KH-style \(c(r)\) + cavity term (`rism_solvation.py`). **Recomputed each pose**
  (patch centroids rotate).
- **Outer shell**: 6-neighbor Ising cohesion \(J\), chemical potential \(\mu\), slab
  confinement \(U_{\mathrm{film}}(z)\) on voxels with \(d >\) first-shell radius.

Set `rism.enabled: false` to use legacy single-region Ising on the full grid.

### 3.2 Patch–interface couplings

From outward rays on the dual mesh (patch = cluster of faces with similar normals):

- \(H_{\mathrm{hp}}\): hydropathy × air exposure \(\mathcal I_f^{\mathrm{air}}\)
- \(H_{\mathrm{pol}}\): polarity penalty at air
- \(H_{\mathrm{HB}}\): H-bond reward at interfacial water (ternary occupancy)

### 3.3 Electrostatics — `electrostatics.method: pb_slab`

- **Default:** linearized PB on the orientation lattice with \(\varepsilon(z)\), \(\kappa(z)\)
  from the slab (`pb_slab_solver.py`). Energy \(\sum_\alpha q_\alpha \phi(\mathbf r_\alpha)\)
  plus intrinsic \(\sum q\phi_0\) and \(\sum \boldsymbol\mu\cdot\mathbf E_0\).
- **Legacy:** `method: screened_pair` — midpoint Yukawa between patch centroids
  (`electrostatic_energy_pb_like` / `spectral_electrostatics.py`).

### 3.4 Film and flexibility

- \(H_{\mathrm{film}}\): patch softness × proximity to aged / sacrificial interfaces
- \(H_{\mathrm{flex}}\): optional penalty on patch softness spread (default off)

---

## 4. Sampling and inverse temperature

| YAML block | Role |
|------------|------|
| `sampling.strategy` | `hybrid` (uniform pool → MCMC), `mcmc`, or `uniform` |
| `sampling.beta` | `auto` calibrates \(\beta\) from energy spread (target ESS) |
| `sampling.mcmc.mode` | `fixed_beta`, `replica_exchange`, `simulated_annealing` |
| `performance.*` | Fast evaluator, parallel chains, PB/RISM compatibility flags |

**MAP pose** in outputs = **maximum Boltzmann weight**, not minimum energy.

---

## 5. Outputs (default)

Written to `--outdir` by `orientation_sample.py`:

| Artifact | Description |
|----------|-------------|
| `top_poses.json` | MAP pose, `most_probable_poses`, `least_probable_poses`, energies, YAML snapshot |
| `diagnostics_*.png` | Energy trace, histogram, tilt–energy |
| `diagnostics_orientation_sampling.*` | ESS, spread, interpretation notes |
| `diagnostics_viewing_*.png` | Azimuth–elevation maps (native \([-\pi,\pi]\), \([-\pi/2,\pi/2]\)) |
| `reference_view_az0_el0.png` | Reference pose render |
| `views_z_down/` | **10 highest** + **10 lowest** weight PNGs (camera lab +Z) |
| `system_view_map.png` | MAP orientation (+Z) |
| `anisotropy_run.log` | Full receipt log |

HPC jobs use `--no-render`; all JSON/PNG diagnostics except PyVista snapshots are still produced.

Configure counts in YAML:

```yaml
output:
  n_orientation_renders: 10
  render_snapshots: true
```

---

## 6. Rigor ladder (honest scope)

Use this when writing methods / thesis text.

| Level | What we claim | Status |
|-------|----------------|--------|
| **A** | Anisotropic AWI slab with distinct top/bottom state variables | Implemented (`awi_field`) |
| **B** | Patch-wise chemistry from structure + PROPKA | Implemented (`patches`) |
| **C** | Orientations drawn from \(\exp(-\beta H)\) with fixed solvent template | Implemented (`orientation_sample`) |
| **D** | First-shell solvation from integral-equation-inspired fields | Approximate RISM (`rism_solvation`) |
| **E** | Linearized PB with lateral homogeneity \(\varepsilon(z)\) | Fast slab solver (`pb_slab_solver`) |
| **F** | Full 3D RISM / APBS on atomic grid | Not implemented |
| **G** | Sum over solvent spins in \(Z\) | Not implemented |
| **H** | Ice trapping, beam damage, preferred orientation from optics alone | Out of scope |

**Near-term path to more rigor:** calibrate \(J,\mu,\lambda_h\) against MD or surface tension;
compare PB slab maps to APBS on a test system; validate RISM shell width with water density
from simulation; joint refinement of \(\beta\) against experimental 2D class distributions.

---

## 7. Configuration checklist

Before a production run, confirm:

- [ ] `ising_params.yaml` (or `hpc/ising_params.hpc.yaml`) matches the physics story you want
- [ ] `rism.enabled` and `electrostatics.method` set intentionally
- [ ] `slab.top` / `slab.bottom` reflect your interface chemistry narrative
- [ ] Lattice `pad_xy`, `slab.thickness` large enough for PB FFT images
- [ ] `performance.parallel_mcmc_chains: false` if using `pb_slab` on all chains (HPC profile sets this)
- [ ] HPC: `hpc/ising_params.hpc.yaml` uses `extends: ../ising_params.yaml` for full physics defaults
- [ ] Conda env synced on HPC (`bash hpc/sync_env.sh`)
- [ ] Slurm passes `--ising-params` and `--parallel-workers`

---

## 8. Code index (new physics modules)

| Module | Function |
|--------|----------|
| `anisotropy.rism_solvation` | `CanonicalShellCache`, `rism_first_shell_energy`, `precompute_outer_solvation_only` |
| `anisotropy.pb_slab_solver` | `SlabPBSolver`, `electrostatic_energy_dispatch` |
| `anisotropy.lattice_solvent_hamiltonian` | `evaluate_hybrid_hamiltonian`, layered `precompute_solvation_energy` |
| `anisotropy.fast_orientation_eval` | Pose loop: RISM shell + PB per orientation |
| `anisotropy.orientation_diagnostics` | Viewing-direction plots, sampling report |

Tests: `tests/test_rism_solvation.py`, `tests/test_pb_slab_solver.py`.
