# Hybrid lattice-gas (Ising) Hamiltonian — parameter reference

This document lists **every numerical knob** used in the current cryo-EM orientation
model (`anisotropy/lattice_solvent_hamiltonian.py`, `anisotropy/awi_field.py`,
`orientation_sample.py`).

**Source of truth:** [`../ising_params.yaml`](../ising_params.yaml) — all defaults below
match that file unless noted. Override at runtime with `orientation_sample.py` CLI flags
or `load_ising_params(path)`.

**Energy units.** Couplings produce energies in a single arbitrary scale (historically
“kcal·mol⁻¹-like” when electrostatics use `COULOMB_SCALE`). The inverse temperature
`β` in `orientation_sample.py` is **dimensionless** and rescales all terms together:
weights \(\propto \exp(-\beta H)\). To connect to experiment or MD, calibrate **one**
reference energy (e.g. hydrogen-bond or surface tension) and rescale the other
couplings relative to it.

**v0 scope.** `orientation_sample.py` uses a **fixed binary solvent occupancy**
(water slab in \(z\in[0,L]\), air outside). It does **not** Monte Carlo over lattice
spins \(n_i\) yet; only rigid orientations \(\Omega\) are sampled. Parameters below
still define \(H[n;\Omega]\) for when spin sampling is added.

---

## 1. Total Hamiltonian

\[
H = H_{\mathrm{solv}} + H_{\mathrm{hp}} + H_{\mathrm{pol}} + H_{\mathrm{HB}}
  + H_{\mathrm{el}} + H_{\mathrm{film}} + H_{\mathrm{flex}}
\]

| Symbol | Code / term | Default scale (`HybridHamiltonianCouplings` or CLI) |
|--------|-------------|-----------------------------------------------------|
| \(H_{\mathrm{solv}}\) | `H_solv` | See §2 |
| \(H_{\mathrm{hp}}\) | `H_hp` in `hydration_terms` | \(\lambda_h = 0.8\) (`--lambda-hp`; class default `1.0`) |
| \(H_{\mathrm{pol}}\) | `H_pol` | \(\lambda_p = 0.25\) (`--lambda-p`) |
| \(H_{\mathrm{HB}}\) | `H_hbond_channel` | \(\lambda_{\mathrm{HB}} = 0.25\) (`--lambda-hb`) |
| \(H_{\mathrm{el}}\) | `H_el` | From slab \(\varepsilon,\kappa,\phi_0,E_0\) + patch \(q,\boldsymbol\mu\) |
| \(H_{\mathrm{film}}\) | `H_film` | \(\lambda_{\mathrm{film}} = 0.3\) (`--lambda-film`; class default `1.0`) |
| \(H_{\mathrm{flex}}\) | `H_flex` | \(\eta_{\mathrm{flex}} = 0\) (`--eta-flex`) |

**Sampler:** \(\beta = 1.0\) (`--beta`).

---

## 2. Lattice gas / Ising solvent — \(H_{\mathrm{solv}}\)

### 2.1 Occupancy variables

| Quantity | Definition | Default in sampler |
|----------|------------|-------------------|
| \(n_i\) | Binary: `0` = air, `1` = water-like | Fixed template (not sampled) |
| \(\eta_i\) | Effective water density for bonds: \(\eta_i = n_i\) (binary) or \(\eta_i=\mathbb{1}[n_i\ge 1]\) (ternary) | Binary |
| Ternary \(\sigma_i\) | `0` air, `1` interfacial water, `2` bulk-like water | Implemented but **not** used in `orientation_sample` |

**Template construction** (`occupancy_binary_template`):

- Water band: \(z_i \in [0, L_{\mathrm{slab}}]\) with \(L_{\mathrm{slab}} = 300\) Å (`--slab-thickness`).
- Air elsewhere on the lattice.
- Protein interior voxels are excluded from ray indicators via `interior` mask (not flipped to air in the template).

### 2.2 Ising cohesion and chemical potential

\[
H_{\mathrm{solv}} \supset
  -J \sum_{\langle i,j\rangle} \eta_i \eta_j
  - \mu \sum_i \eta_i
  + u_{\mathrm{film,scale}} \sum_i U_{\mathrm{film}}(z_i)\,\eta_i
\]

| Parameter | Meaning | Default | CLI | Tunable from |
|-----------|---------|---------|-----|----------------|
| \(J\) | Nearest-neighbor water–water cohesion | `1.0` | `--J` | MD contact energy, surface tension, wetting |
| \(\mu\) | Bulk chemical potential (favors water if \(\mu>0\)) | `0.0` | `--mu` | Activity of water, vapour pressure, osmotic bias |
| Connectivity | 6-neighbor (±x, ±y, ±z) on cubic lattice | fixed | — | Lattice geometry |

**Code:** `solvation_energy_lattice_gas`, `HybridHamiltonianCouplings.J_solv`, `.mu_chemical`.

### 2.3 Slab confining field \(U_{\mathrm{film}}(z)\)

Penalizes water **outside** the vitrified slab \([z_{\mathrm{lo}}, z_{\mathrm{hi}}]\):

\[
U_{\mathrm{film}}(z) = \texttt{penalty\_outside} \cdot
  \frac{\max(z_{\mathrm{lo}}-z,\,0,\,z-z_{\mathrm{hi}})^2}
       {\texttt{interface\_softness}^2 + \cdots}
\]

| Parameter | Default | In code | Tunable from |
|-----------|---------|---------|--------------|
| `penalty_outside` | `1.0` | `film_potential_per_cell` | Strength of confinement vs air |
| `interface_softness` | `4.0` Å | `film_potential_per_cell` | Sharpness of slab edge |
| \(u_{\mathrm{film,scale}}\) | `0.25` | `HybridHamiltonianCouplings.u_film_scale` / `--u-film` | Overall weight vs \(J\) |
| \(z_{\mathrm{lo}}, z_{\mathrm{hi}}\) | `0`, `L_{\mathrm{slab}}` | `slab_z_bounds` from slab thickness | Experimental ice thickness |

**Note:** With the fixed template, \(\eta_i=0\) outside the slab so this term is often **zero** in v0; it matters when occupancy is allowed to vary.

---

## 3. Lattice geometry (orientation sampler)

| Parameter | Default | CLI | Role |
|-----------|---------|-----|------|
| Grid spacing \(h\) | `3.5` Å | `--grid-spacing` | Voxel size; coarser ⇒ fewer sites, cruder interface |
| Slab thickness \(L_{\mathrm{slab}}\) | `300` Å | `--slab-thickness` | Vitrified water extent in \(z\) |
| XY padding | `60` Å | `--pad-xy` | Lattice extent around protein in \(x,y\) |
| Z padding | `25` Å | `--pad-z` | Extra vacuum above/below slab in lattice box |
| Protein \(z_{\mathrm{COM}}\) | \(L_{\mathrm{slab}} - 0.8\times 7\) Å ≈ `294.4` Å | `--z-center` | Places protein near **top** AWI |
| Canonical interior pad | `16` Å | `--canonical-pad` | Bbox padding for fast voxel cache |

Lattice origin: \((x_{\min}-pad_{xy},\, y_{\min}-pad_{xy},\, -pad_z)\).

---

## 4. Patch–solvent indicators (couple protein to lattice)

Outward rays from patch centroid \(\mathbf r_f\) along \(\hat{\mathbf n}_f\):

| Parameter | Default | Function | Role |
|-----------|---------|----------|------|
| `ray_max_steps` | `48` | `patch_air_indicator`, `patch_interfacial_water_indicator` | Max solvent voxels along ray |
| `ray_step` | `0.55` × \(h\) | same | Step length along ray |
| `require_first_hit_solvent_steps` | `2` | `patch_air_indicator` | Minimum solvent voxels before “air” counts |
| `fractional_step` (generic ray) | `0.5` × \(h\) | `outward_solvent_ray_sample` default | Lower-level march |

| Indicator | Definition | Binary mode |
|-----------|------------|-------------|
| \(\mathcal I_f^{\mathrm{air}}\) | `1` if any air-like site on ray | Active |
| \(\mathcal I_f^{\mathrm{int}}\) | `1` if interfacial-water site on ray | Always `0` (needs ternary + finer grid) |

---

## 5. Hydration / polarity / H-bond — \(H_{\mathrm{hp}}, H_{\mathrm{pol}}, H_{\mathrm{HB}}\)

\[
\begin{aligned}
H_{\mathrm{hp}} &= -\lambda_h \sum_f a_f\, h_f\, \mathcal I_f^{\mathrm{air}} \\
H_{\mathrm{pol}} &= +\lambda_p \sum_f a_f\, p_f\, \mathcal I_f^{\mathrm{air}} \\
H_{\mathrm{HB}} &= -\lambda_{\mathrm{HB}} \sum_f a_f\, b_f\, \mathcal I_f^{\mathrm{int}}
\end{aligned}
\]

| Coupling | `HybridHamiltonianCouplings` | `orientation_sample` CLI |
|----------|------------------------------|---------------------------|
| \(\lambda_h\) | `1.0` | `0.8` (`--lambda-hp`) |
| \(\lambda_p\) | `0.25` | `0.25` (`--lambda-p`) |
| \(\lambda_{\mathrm{HB}}\) | `0.25` | `0.25` (`--lambda-hb`) |

### 5.1 Patch features \(h_f, p_f, b_f\) (inputs from mesh parameterization)

These are **not** Ising parameters but set the coupling strengths per patch:

| Feature | Symbol | Source | Typical scale |
|---------|--------|--------|---------------|
| Area | \(a_f\) | Sum of face areas | Å² |
| Hydropathy | \(h_f\) | Kyte–Doolittle × \(\|\hat{\mathbf n}_f\|\) | ≈ −4…+4 |
| Polar density | \(p_f\) | Fraction polar residues / atom | 0…1 |
| H-bond score | \(b_f\) | (donors+acceptors)/atoms in patch | 0…3+ |
| Charge | \(q_f\) | Sum of partial charges (e) | ff19SB default |
| Dipole | \(\boldsymbol\mu_f\) | \(\sum_i q_i(\mathbf r_i-\mathbf r_f)\) | Å·e |
| Softness | \(u_f\) | B-factor/100 or curvature proxy | 0…1 |

**Parameterization defaults** (`parameterize_mesh.py`):

| Parameter | Default |
|-----------|---------|
| `ph` | `7.0` |
| `charge_model` | `ff19sb` (AMBER `amino19.lib` partial charges) |
| `pka_source` | `auto` (PROPKA if available) |
| `patch_angle` | `25°` |
| `min_patch_area` | `50` Å² |

**Electrostatic prefactor** (pairwise term): `COULOMB_SCALE = 332.0636` (kcal·mol⁻¹·Å·e⁻²) in `patches.py`.

**Legacy charge tables** (`residue_chemistry.py`, if `--charge-model table`):

| Residue | pKa (approx.) | Charge at pH 7 |
|---------|---------------|----------------|
| ASP | 3.9 | −1 |
| GLU | 4.3 | −1 |
| HIS | 6.0 | +0.1 |
| CYS | 8.3 | (titratable) |
| TYR | 10.1 | (titratable) |
| LYS | 10.5 | +1 |
| ARG | 12.5 | +1 |

---

## 6. Electrostatics — \(H_{\mathrm{el}}\) (PB-like placeholder)

\[
H_{\mathrm{el}} \approx
  \sum_{\alpha<\beta} q_\alpha\, G_{\varepsilon,\kappa}(\mathbf r_{\alpha\beta})\, q_\beta
  + \sum_\alpha q_\alpha\,\phi_0(z_\alpha)
  + \sum_\alpha \boldsymbol\mu_\alpha\cdot\mathbf E_0(z_\alpha)
\]

**Screened Coulomb kernel** (midpoint dielectric, pairwise):

\[
G \approx \frac{k}{\varepsilon\, r}\, e^{-\kappa r}, \quad k = \texttt{COULOMB\_SCALE}
\]

| Parameter | Default | Override CLI |
|-----------|---------|--------------|
| \(\varepsilon\) | From slab profile \((\varepsilon_\parallel+\varepsilon_\perp)/2\) at patch \(z\) | `--homogeneous-epsilon` |
| \(\kappa\) (1/Å) | From slab profile (often `0`) | `--homogeneous-kappa` |
| `r_smooth` | `1e-2` Å | (code only) |
| Use \(\phi_0\) term | on | — |
| Use \(\boldsymbol\mu\cdot\mathbf E_0\) | on (\(E_0\) along lab \(+\hat z\)) | — |

**Not yet implemented:** heterogeneous Green’s function from full Poisson–Boltzmann; replace `green_yukawa_coulomb` when MD/FEM profiles are available.

---

## 7. AWI depth fields (slab dielectric & potential)

Built by `build_cryo_slab_preset` → two `AWIInterface`s (bottom pristine, top sacrificial film).

### 7.1 Slab-level

| Parameter | Default |
|-----------|---------|
| `thickness_angstrom` | `300` (CLI `--slab-thickness`) |
| `bulk_epsilon` | `78` |
| `bulk_rho` | `1.0` |
| `bulk_kappa` | `0` (1/Å) |

### 7.2 Per-interface scalars (cryo preset)

| Interface | `coverage` \(c\) | `mechanical_state` | `mechanical_age` | \(\gamma\) (N/m) | \(\phi_0(z{=}0)\) scale |
|-----------|----------------|--------------------|------------------|------------------|-------------------------|
| Bottom (z=0) | `PRISTINE` | `FRESH` | `0.0` | `0.0728` | `0.03` V |
| Top (z=L) | `PROTEIN_SACRIFICIAL` | `VISCOELASTIC_FILM` | `0.6` | `0.065` | `0.02` V |

Top interface also uses `surface_rho_enhancement = 2.2` (vs default `1.8` on profile builder).

### 7.3 Depth profile \(y(z) = (\rho, \varepsilon_\parallel, \varepsilon_\perp, \phi_0, E_0, \kappa)\)

`build_depth_profile` defaults (each interface unless overridden):

| Parameter | Default | Physical role |
|-----------|---------|---------------|
| `extent` | `40` Å | Profile grid length from interface |
| `dz` | `0.25` Å | Profile grid spacing |
| `decay_length` | `7.0` Å | Anisotropy / blending σ (`DEFAULT_ANISOTROPY_DECAY_ANGSTROM`) |
| `bulk_epsilon` | `78` | Bulk water |
| `surface_epsilon_parallel` | `2.5` | Surface ε∥ |
| `surface_epsilon_perpendicular` | `1.5` | Surface ε⊥ |
| `use_tensor_dielectric` | `True` | Tensor vs scalar ε |
| `bulk_rho` | `1.0` | Relative density scale |
| `surface_rho_enhancement` | `1.8` | Top layer density bump |
| `phi0_surface_volts` | `0.05` V (per-call override in preset) | Intrinsic potential at interface |
| `phi0_decay_length` | `7` Å (defaults to `decay_length`) | φ₀ decay into bulk |
| `debye_length_angstrom` | `None` → \(\kappa=0\) | Set to λ_D for ionic strength |
| `transition_width` | `2.0` Å | Dielectric tanh width |
| `transition_center` | `1.5` Å | Dielectric tanh center |
| Layer scale (ρ) | `3.0` Å | Near-interface density enhancement |

**Interface blending:** Gaussian weights vs distance to bottom/top with σ ≈ profile decay length (`sample_fields(..., blend_interfaces=True)`).

---

## 8. Film / age coupling — \(H_{\mathrm{film}}\)

\[
H_{\mathrm{film}} = \lambda_{\mathrm{film}} \sum_f a_f\, u_f\, I_f^{\mathrm{film}}
\]

\(I_f^{\mathrm{film}}\): Gaussian weight near interfaces with non-pristine `coverage` × `mechanical_age` (see `film_coupling_energy`, `patch_film_indicator`).

| Parameter | Default |
|-----------|---------|
| \(\lambda_{\mathrm{film}}\) | `1.0` (class) / `0.3` (sampler CLI) |
| `margin_angstrom` | `7.0` Å (same as anisotropy decay) |
| Film coverage set | `PROTEIN_SACRIFICIAL`, `SURFACTANT`, `CONTAMINATED` |

---

## 9. Flexibility penalty — \(H_{\mathrm{flex}}\)

\[
H_{\mathrm{flex}} = \eta_{\mathrm{flex}} \sum_f (u_f - \bar u)^2
\]

| Parameter | Default |
|-----------|---------|
| \(\eta_{\mathrm{flex}}\) | `0` (off) |

---

## 10. Orientation sampling (non-Hamiltonian knobs)

| Parameter | Default | CLI / YAML |
|-----------|---------|------------|
| Strategy | `hybrid` | `sampling.strategy`, `--sampling-strategy` |
| Uniform pool size | `400` | `sampling.n_uniform`, `--n-uniform` |
| MCMC mode | `replica_exchange` | `sampling.mcmc.mode` |
| MCMC chains | `4` | `sampling.mcmc.n_chains`, `--mcmc-chains` |
| Steps per chain | `2000` | `sampling.mcmc.steps_per_chain`, `--mcmc-steps` |
| \(\beta\) | `auto` | `sampling.beta`, `--beta` |
| \(\beta\) auto target ESS | `20` | `sampling.beta_target_ess` |
| Number of orientations (uniform only) | `600` | `sampling.n_samples`, `-n` |
| RNG seed | `0` | `sampling.seed`, `--seed` |
| Replica count | `8` | `sampling.mcmc.replica_exchange.n_replicas` |
| Annealing reheat cycles | `2` | `sampling.mcmc.annealing.n_reheat_cycles` |
| Run receipt log | `anisotropy_run.log` (append) | `--log-file`, `--log-overwrite` |
| Mesh fit resolutions | `3.5, 2.5, 1.8` Å | `--resolutions` (with `--fit-mesh`) |
| SAS probe radius | `1.4` Å | `fit_protein_mesh.py --probe` |

**Multimodal modes** (`anisotropy/orientation_multimodal.py`): `simulated_annealing` cools
\(\beta_k\) with optional reheating; `replica_exchange` uses parallel tempering with
detailed-balance swaps. Both reweight kept samples to \(\beta_{\mathrm{target}}\) via
`importance_weights_to_beta`. No artificial “escape potential” is added to \(H\).

---

## 11. Quick tuning map (experiment / simulation)

| Goal | Primary knobs |
|------|----------------|
| Wetting vs drying at AWI | \(J\), \(\mu\), \(\lambda_h\), \(\lambda_p\), placement \(z_{\mathrm{COM}}\) |
| Ice thickness / confinement | \(L_{\mathrm{slab}}\), \(u_{\mathrm{film,scale}}\), lattice \(h\) |
| Electrostatic orientation bias | Patch \(q_f,\boldsymbol\mu_f\) (ff19SB), \(\phi_0\), \(\varepsilon\), \(\kappa\), `COULOMB_SCALE` |
| Top vs bottom interface asymmetry | `build_cryo_slab_preset` coverage, `mechanical_age`, \(\gamma\), \(\phi_0\) |
| Sacrificial protein / film | Top `coverage=PROTEIN_SACRIFICIAL`, \(\lambda_{\mathrm{film}}\), `mechanical_age` |
| Finer interfacial water | Ternary occupancy, smaller \(h\), \(\mathcal I_f^{\mathrm{int}}\) |
| Absolute energy scale | Calibrate one term + \(\beta\); rescale remaining couplings |
| MD-derived parameters | \(J,\mu\) from water model; \(q,\boldsymbol\mu\) from AMBER/CHARMM; \(\varepsilon,\kappa\) from PB or profile; \(h_f\) from surface free energy |

---

## 12. Code entry points

| Task | Module / script |
|------|-----------------|
| Define couplings | `HybridHamiltonianCouplings` |
| Evaluate \(H\) | `evaluate_hybrid_hamiltonian` |
| Fixed-\(n\) solvation only | `precompute_solvation_energy` |
| AWI fields | `build_cryo_slab_preset`, `build_depth_profile` |
| Patch features | `parameterize_mesh` |
| Sample orientations | `orientation_sample.py` |

CLI help: `python orientation_sample.py --help`, `python parameterize_mesh.py --help`.
