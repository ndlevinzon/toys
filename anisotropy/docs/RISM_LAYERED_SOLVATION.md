# Layered solvation: RISM first shell + Ising outer shell

See also: [PHYSICS_AND_PIPELINE.md](PHYSICS_AND_PIPELINE.md) (workflow integration).

## Model

| Region | Distance to protein surface (canonical frame) | Energy model |
|--------|-----------------------------------------------|--------------|
| **First shell** | \(0 < d \le\) `rism.first_shell_angstrom` (default 5 Å) | Simplified 3D-RISM-inspired site potential |
| **Outer shell** | \(d >\) `first_shell_angstrom` | Lattice-gas / Ising (`solv.J`, `solv.mu`, film field) |

Total solvent energy (fixed occupancy template):

\[
H_{\mathrm{solv}} = H_{\mathrm{RISM}}^{\mathrm{(1st\ shell)}} + H_{\mathrm{Ising}}^{\mathrm{(outer)}}
\]

The first-shell term is **recomputed for each orientation** (patch centroids rotate).
The outer Ising term is **precomputed once** when occupancy is fixed.

## First-shell physics (approximation)

Per water oxygen site \(i\) in the first shell:

\[
\mu_{\mathrm{ex}}(i) = \sum_{\alpha \in \mathrm{patches}}
  \left( \frac{k q_\alpha}{\epsilon r_{i\alpha}} + \rho_{\mathrm{bulk}}\, c(r_{i\alpha}) \right)
  + \gamma_{\mathrm{cav}} \sqrt{A_{\alpha^\*}/\pi}\, e^{-r_{i\alpha^\*}/\lambda}
\]

with \(c(r) \approx -\beta\, u_{\mathrm{LJ}}^{\mathrm{OO}}(r)\) (Kovalenko–Hirata-style direct correlation proxy)
and \(u_{\mathrm{LJ}}\) from TIP3P-like \(\sigma,\epsilon\) for oxygen–oxygen.

This is **not** a full GMMT / AmberTools 3D-RISM solve; it is a fast integral-equation-inspired
field for orientation sampling.

## YAML (`ising_params.yaml`)

```yaml
rism:
  enabled: true
  first_shell_angstrom: 5.0
  bulk_number_density: 0.0334
  temperature_k: 300.0
  dielectric: 78.0
  kh_strength: 1.0
  cavity_gamma_kcal_per_ang2: 0.12
  site_cutoff_angstrom: 8.0
```

Set `enabled: false` to use the legacy single-region lattice gas on the full grid.

## Code entry points

- `anisotropy.rism_solvation.CanonicalShellCache` — distance field + shell masks (rotation-invariant)
- `precompute_outer_solvation_only` — outer Ising for fast orientation scans
- `rism_first_shell_energy` — pose-dependent first shell
