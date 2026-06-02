# Slab Poisson–Boltzmann electrostatics

See also: [PHYSICS_AND_PIPELINE.md](PHYSICS_AND_PIPELINE.md) (workflow integration).

## Model

On the orientation lattice, solve **linearized Poisson–Boltzmann** with AWI profile
ε(z), κ(z) from `VitrifiedWaterSlab`:

\[
-\nabla\cdot(\varepsilon(z)\nabla\phi) + \kappa(z)^2\varepsilon(z)\,\phi = 4\pi k_{\mathrm{coul}}\,\rho
\]

- **Lateral:** FFT (periodic box) — O(N_x N_y log N) per z-slice stack
- **Normal:** tridiagonal solve along z for each (k_x, k_y) mode — O(N_z) per mode
- **Boundaries:** φ = 0 at grid z-faces (Dirichlet far-field box)

Charge density ρ is deposited from patch charges (Gaussian stencil, width
`pb_charge_sigma_voxels` × grid spacing).

Energy:

\[
H_{\mathrm{el}}^{\mathrm{PB}} = k_{\mathrm{coul}} \sum_\alpha q_\alpha \phi(\mathbf r_\alpha)
\]

plus optional slab intrinsic terms ∑ q φ₀(z) and ∑ μ·E₀(z) from `SlabZFieldLUT`.

## YAML

```yaml
electrostatics:
  method: pb_slab          # or screened_pair (legacy Yukawa)
  pb_coarse_factor: 2      # solve on a 2× coarser grid, then sample φ
  pb_charge_sigma_voxels: 1.25
```

## Performance

Typical orientation grids (≈40³–60³) with `pb_coarse_factor: 2` add **one FFT+tridiagonal
solve per pose** — usually cheaper than full 3D FEM PB and more faithful at the AWI than
mid-point Yukawa pairs.

## Limitations

- Linearized PB (no sinh nonlinearity at high potential)
- Laterally homogeneous ε(z) — no protein-induced dielectric cavity in ε yet
- Periodic (x, y) images from FFT; box should be large enough (`pad_xy` in lattice)
- Not a replacement for APBS on atomic grids; a **fast interface-aware** field model
