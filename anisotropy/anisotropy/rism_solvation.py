"""
3D-RISM-inspired first solvation shell + combinatorial outer shell.

The first shell (voxels within ``first_shell_angstrom`` of the protein surface,
in the canonical mesh frame) uses a simplified **Kovalenko–Hirata**-style excess
chemical potential: screened Coulomb + LJ solute–oxygen + cavity term from patch
area, with a bulk direct-correlation proxy ``c(r) ≈ -β u_LJ(r)``.

Voxels beyond the first shell use the fast lattice-gas / Ising model
(:func:`~anisotropy.lattice_solvent_hamiltonian.solvation_energy_lattice_gas_masked`).

This is a **lightweight** integral-equation-inspired model for orientation sampling,
not a full GMMT / Ambertools 3D-RISM solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from anisotropy.lattice_solvent_hamiltonian import (
    CartesianLattice,
    CanonicalInteriorCache,
    ProteinMesh,
    effective_binary_water,
    film_potential_per_cell,
)
from anisotropy.patches import COULOMB_SCALE, PatchFeatures, PatchParameterization

try:
    from scipy import ndimage
except ImportError as exc:  # pragma: no cover
    raise ImportError("rism_solvation requires scipy (conda install scipy)") from exc


@dataclass(frozen=True)
class RismSolvationParams:
    """Parameters for the first-shell RISM-inspired term."""

    enabled: bool = True
    first_shell_angstrom: float = 5.0
    bulk_number_density: float = 0.0334
    temperature_k: float = 300.0
    dielectric: float = 78.0
    lj_sigma_oo_angstrom: float = 3.166
    lj_epsilon_oo_kcal: float = 0.1554
    coulomb_scale: float = COULOMB_SCALE
    kh_strength: float = 1.0
    cavity_gamma_kcal_per_ang2: float = 0.12
    site_cutoff_angstrom: float = 8.0
    energy_scale: float = 1.0


def distance_to_surface_canonical(inside_can: np.ndarray, spacing: float) -> np.ndarray:
    """Voxel distance (Å) to the protein surface from exterior voxels."""
    h = float(spacing)
    dist = ndimage.distance_transform_edt(~inside_can).astype(np.float64) * h
    dist[inside_can] = 0.0
    return dist


def _map_canonical_field_to_lab(
    field_can: np.ndarray,
    can_lattice: CartesianLattice,
    lab_lattice: CartesianLattice,
    R: np.ndarray,
    t: np.ndarray,
    *,
    lab_xyz_flat: np.ndarray | None = None,
) -> np.ndarray:
    """Sample a canonical boolean/float field onto the lab lattice for pose ``(R, t)``."""
    Rm = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tv = np.asarray(t, dtype=np.float64).reshape(3)
    if lab_xyz_flat is None:
        lab_xyz_flat = lab_lattice.grid_centers_xyz().reshape(-1, 3)
    x_can = (np.asarray(lab_xyz_flat, dtype=np.float64).reshape(-1, 3) - tv) @ Rm

    o = can_lattice.origin.reshape(3)
    h = float(can_lattice.spacing)
    nx_c, ny_c, nz_c = can_lattice.shape
    rel = (x_can - o) / h - 0.5
    ijk = np.floor(rel).astype(np.int64)
    in_b = (
        (ijk[:, 0] >= 0)
        & (ijk[:, 0] < nx_c)
        & (ijk[:, 1] >= 0)
        & (ijk[:, 1] < ny_c)
        & (ijk[:, 2] >= 0)
        & (ijk[:, 2] < nz_c)
    )
    out = np.zeros(x_can.shape[0], dtype=bool)
    ii = ijk[in_b, 0]
    jj = ijk[in_b, 1]
    kk = ijk[in_b, 2]
    out[in_b] = field_can[ii, jj, kk]
    return out.reshape(lab_lattice.shape)


@dataclass(frozen=True)
class CanonicalShellCache:
    """
  Canonical interior + distance field + first/outer shell masks (mesh frame).

  Distance to the surface is rotation-invariant: for lab pose ``(R, t)``,
  sample the canonical distance field at ``R.T (x_lab - t)``.
  """

    interior: CanonicalInteriorCache
    dist_can: np.ndarray
    first_shell_can: np.ndarray
    outer_solvent_can: np.ndarray

    @property
    def inside_can(self) -> np.ndarray:
        return self.interior.inside_can

    @property
    def can_lattice(self) -> CartesianLattice:
        return self.interior.can_lattice

    @classmethod
    def build(
        cls,
        mesh: ProteinMesh,
        *,
        spacing: float,
        pad_angstrom: float,
        first_shell_angstrom: float,
    ) -> CanonicalShellCache:
        interior = CanonicalInteriorCache.build(
            mesh, spacing=spacing, pad_angstrom=pad_angstrom
        )
        dist = distance_to_surface_canonical(interior.inside_can, spacing)
        r1 = float(first_shell_angstrom)
        solvent = ~interior.inside_can
        first = solvent & (dist > 0.0) & (dist <= r1)
        outer = solvent & (dist > r1)
        return cls(
            interior=interior,
            dist_can=dist,
            first_shell_can=first,
            outer_solvent_can=outer,
        )

    def lab_interior_mask(
        self,
        lab_lattice: CartesianLattice,
        R: np.ndarray,
        t: np.ndarray,
        *,
        lab_xyz_flat: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.interior.lab_interior_mask(
            lab_lattice, R, t, lab_xyz_flat=lab_xyz_flat
        )

    def lab_shell_masks(
        self,
        lab_lattice: CartesianLattice,
        R: np.ndarray,
        t: np.ndarray,
        *,
        lab_xyz_flat: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return ``(first_shell, outer_shell, interior)`` boolean masks on the lab grid.

        Shell assignment uses canonical distance; ``interior`` excludes protein voxels.
        """
        interior = self.lab_interior_mask(lab_lattice, R, t, lab_xyz_flat=lab_xyz_flat)
        first_c = _map_canonical_field_to_lab(
            self.first_shell_can,
            self.can_lattice,
            lab_lattice,
            R,
            t,
            lab_xyz_flat=lab_xyz_flat,
        )
        outer_c = _map_canonical_field_to_lab(
            self.outer_solvent_can,
            self.can_lattice,
            lab_lattice,
            R,
            t,
            lab_xyz_flat=lab_xyz_flat,
        )
        solvent = ~interior
        return solvent & first_c, solvent & outer_c, interior


def _lj_kcal(r: np.ndarray, sigma: float, epsilon: float) -> np.ndarray:
    x = np.maximum(sigma / np.maximum(r, 1e-6), 1e-6)
    x6 = x**6
    x12 = x6 * x6
    return 4.0 * epsilon * (x12 - x6)


def rism_excess_chemical_potential(
    xyz: np.ndarray,
    patch_centroids: np.ndarray,
    patch_charges: np.ndarray,
    patch_areas: np.ndarray,
    params: RismSolvationParams,
) -> np.ndarray:
    """
    Per-voxel excess chemical potential proxy (kcal/mol scale) for water oxygen sites.

    ``xyz`` shape ``(N, 3)``; patch arrays shape ``(P,)`` or ``(P, 3)``.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    cents = np.asarray(patch_centroids, dtype=np.float64).reshape(-1, 3)
    q = np.asarray(patch_charges, dtype=np.float64).reshape(-1)
    areas = np.asarray(patch_areas, dtype=np.float64).reshape(-1)
    if xyz.shape[0] == 0 or cents.shape[0] == 0:
        return np.zeros(xyz.shape[0], dtype=np.float64)

    diff = xyz[:, None, :] - cents[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    rc = float(params.site_cutoff_angstrom)
    valid = r < rc

    eps = max(float(params.dielectric), 1.0)
    k_c = float(params.coulomb_scale)
    u_coul = k_c * q[None, :] / (eps * np.maximum(r, 0.5))

    sig = float(params.lj_sigma_oo_angstrom)
    eps_lj = float(params.lj_epsilon_oo_kcal)
    u_lj = _lj_kcal(r, sig, eps_lj)

    beta = 1.0 / (0.0019872041 * max(float(params.temperature_k), 1.0))
    rho = float(params.bulk_number_density)
    kh = float(params.kh_strength)
    c_ij = -beta * kh * u_lj
    mu = np.sum((u_coul + rho * c_ij) * valid, axis=1)

    # Cavity / hydrophobic packing near nearest patch
    r_nearest = np.where(valid.any(axis=1), np.min(np.where(valid, r, np.inf), axis=1), rc)
    j_near = np.argmin(np.where(valid, r, np.inf), axis=1)
    mu += float(params.cavity_gamma_kcal_per_ang2) * np.sqrt(
        np.maximum(areas[j_near], 0.0) / np.pi
    ) * np.exp(-r_nearest / max(rc * 0.5, 1.0))

    return mu * float(params.energy_scale)


def rism_first_shell_energy(
    occupancy: np.ndarray,
    first_shell_mask: np.ndarray,
    lab_xyz_flat: np.ndarray,
    param_pose: PatchParameterization,
    params: RismSolvationParams,
    *,
    occupancy_mode: Literal["binary", "ternary"] = "binary",
) -> tuple[float, dict]:
    """
    ``H_RISM = sum_{i in first shell} eta_i * mu_ex(i)`` with pose-dependent patch geometry.
    """
    eta = effective_binary_water(occupancy, occupancy_mode)
    mask = np.asarray(first_shell_mask, dtype=bool) & (eta > 0.5)
    flat = mask.ravel()
    n = int(np.count_nonzero(flat))
    if n == 0:
        return 0.0, {"n_sites": 0, "mu_ex_mean": 0.0}

    xyz = np.asarray(lab_xyz_flat, dtype=np.float64).reshape(-1, 3)[flat]
    patches = param_pose.patches
    cents = np.stack([np.asarray(p.centroid, dtype=np.float64) for p in patches], axis=0)
    q = np.array([p.charge for p in patches], dtype=np.float64)
    areas = np.array([p.area for p in patches], dtype=np.float64)

    mu_ex = rism_excess_chemical_potential(xyz, cents, q, areas, params)
    eta_s = eta.ravel()[flat]
    h = float(np.dot(eta_s, mu_ex))
    return h, {
        "H_rism_first_shell": h,
        "n_first_shell_sites": n,
        "mu_ex_mean": float(np.mean(mu_ex)),
        "mu_ex_max": float(np.max(mu_ex)),
        "mu_ex_min": float(np.min(mu_ex)),
    }


def _six_neighbor_bond_sum_masked(eta: np.ndarray, mask: np.ndarray) -> float:
    """Nearest-neighbor ``sum eta_i eta_j`` only when both sites lie in ``mask``."""
    m = np.asarray(mask, dtype=bool)
    n = np.asarray(eta, dtype=np.float64)
    nx, ny, nz = n.shape
    acc = 0.0
    if nx > 1:
        both = m[:-1, :, :] & m[1:, :, :]
        acc += float(np.sum(n[:-1, :, :] * n[1:, :, :] * both))
    if ny > 1:
        both = m[:, :-1, :] & m[:, 1:, :]
        acc += float(np.sum(n[:, :-1, :] * n[:, 1:, :] * both))
    if nz > 1:
        both = m[:, :, :-1] & m[:, :, 1:]
        acc += float(np.sum(n[:, :, :-1] * n[:, :, 1:] * both))
    return acc


def solvation_energy_lattice_gas_masked(
    occ: np.ndarray,
    lattice: CartesianLattice,
    site_mask: np.ndarray,
    *,
    mode: Literal["binary", "ternary"] = "binary",
    J: float = 1.0,
    mu_chemical: float = 0.0,
    u_film: np.ndarray | None = None,
    u_film_scale: float = 0.2,
    slab_z_bounds: tuple[float, float] | None = None,
) -> tuple[float, dict]:
    """Lattice-gas cohesion on ``site_mask`` only (outer solvation shell)."""
    eta = effective_binary_water(occ, mode)
    mask = np.asarray(site_mask, dtype=bool)
    bond = _six_neighbor_bond_sum_masked(eta, mask)
    h_nn = -J * bond
    eta_m = np.where(mask, eta, 0.0)
    h_mu = -mu_chemical * float(np.sum(eta_m))

    if u_film is None:
        if slab_z_bounds is not None:
            z_lab = lattice.grid_centers_xyz()[..., 2].ravel()
            u_vec = film_potential_per_cell(
                z_lab, slab_z_bounds[0], slab_z_bounds[1]
            ).reshape(lattice.shape)
            h_fil = float(u_film_scale * np.sum(u_vec[mask] * eta[mask]))
        else:
            h_fil = 0.0
            u_vec = None
    else:
        u_arr = np.asarray(u_film, dtype=np.float64)
        h_fil = float(u_film_scale * np.sum(u_arr[mask] * eta[mask]))
        u_vec = u_arr

    h = h_nn + h_mu + h_fil
    return h, {
        "H_nn": h_nn,
        "H_mu": h_mu,
        "H_film_field": h_fil,
        "bond_sum_eta_eta": bond,
        "eta_sum_outer": float(np.sum(eta[mask])),
        "n_outer_sites": int(np.count_nonzero(mask)),
        "layer": "outer_ising",
    }


def precompute_layered_solvation_energy(
    occupancy: np.ndarray,
    lattice: CartesianLattice,
    slab: Any,
    coeffs: Any,
    shell_cache: CanonicalShellCache,
    rism_params: RismSolvationParams,
    param: PatchParameterization,
    pose_R: np.ndarray,
    pose_t: np.ndarray,
    *,
    occupancy_mode: Literal["binary", "ternary"] = "binary",
    confinement_penalty_outside: float = 1.0,
    confinement_interface_softness: float = 4.0,
    lab_xyz_flat: np.ndarray | None = None,
) -> tuple[float, dict]:
    """
    ``H_solv = H_RISM(first shell) + H_Ising(outer shell)``.

    Outer Ising part is pose-independent when occupancy is fixed; RISM first shell
    depends on rotated patch centroids and is recomputed when pose changes.
    """
    if lab_xyz_flat is None:
        lab_xyz_flat = lattice.grid_centers_xyz().reshape(-1, 3).astype(np.float64)

    first_mask, outer_mask, _interior = shell_cache.lab_shell_masks(
        lattice, pose_R, pose_t, lab_xyz_flat=lab_xyz_flat
    )

    slab_z_bounds = (0.0, float(slab.thickness_angstrom))
    z_lab = lattice.grid_centers_xyz()[..., 2].ravel()
    u_vec = film_potential_per_cell(
        z_lab,
        slab_z_bounds[0],
        slab_z_bounds[1],
        penalty_outside=confinement_penalty_outside,
        interface_softness=confinement_interface_softness,
    ).reshape(lattice.shape)

    h_outer, outer_terms = solvation_energy_lattice_gas_masked(
        occupancy,
        lattice,
        outer_mask,
        mode=occupancy_mode,
        J=coeffs.J_solv,
        mu_chemical=coeffs.mu_chemical,
        u_film_scale=coeffs.u_film_scale,
        u_film=u_vec,
    )

    from anisotropy.lattice_solvent_hamiltonian import rigid_patch_parameterization

    param_pose = rigid_patch_parameterization(param, pose_R, pose_t)
    h_rism, rism_terms = rism_first_shell_energy(
        occupancy,
        first_mask,
        lab_xyz_flat,
        param_pose,
        rism_params,
        occupancy_mode=occupancy_mode,
    )

    h_total = h_outer + h_rism
    terms = {
        **outer_terms,
        **rism_terms,
        "H_solv_outer": h_outer,
        "H_solv_total": h_total,
        "n_first_shell_voxels": int(np.count_nonzero(first_mask)),
        "n_outer_shell_voxels": int(np.count_nonzero(outer_mask)),
        "first_shell_angstrom": rism_params.first_shell_angstrom,
        "layered_solvation": True,
    }
    return h_total, terms


def precompute_outer_solvation_only(
    occupancy: np.ndarray,
    lattice: CartesianLattice,
    slab: Any,
    coeffs: Any,
    shell_cache: CanonicalShellCache,
    pose_R: np.ndarray,
    pose_t: np.ndarray,
    *,
    occupancy_mode: Literal["binary", "ternary"] = "binary",
    confinement_penalty_outside: float = 1.0,
    confinement_interface_softness: float = 4.0,
    lab_xyz_flat: np.ndarray | None = None,
) -> tuple[float, dict]:
    """Outer-shell Ising energy only (for fast eval: add RISM per pose)."""
    if lab_xyz_flat is None:
        lab_xyz_flat = lattice.grid_centers_xyz().reshape(-1, 3)
    _first, outer_mask, _ = shell_cache.lab_shell_masks(
        lattice, pose_R, pose_t, lab_xyz_flat=lab_xyz_flat
    )
    slab_z_bounds = (0.0, float(slab.thickness_angstrom))
    z_lab = lattice.grid_centers_xyz()[..., 2].ravel()
    u_vec = film_potential_per_cell(
        z_lab,
        slab_z_bounds[0],
        slab_z_bounds[1],
        penalty_outside=confinement_penalty_outside,
        interface_softness=confinement_interface_softness,
    ).reshape(lattice.shape)
    h_outer, terms = solvation_energy_lattice_gas_masked(
        occupancy,
        lattice,
        outer_mask,
        mode=occupancy_mode,
        J=coeffs.J_solv,
        mu_chemical=coeffs.mu_chemical,
        u_film_scale=coeffs.u_film_scale,
        u_film=u_vec,
    )
    terms["layered_solvation"] = True
    terms["H_solv_outer"] = h_outer
    return h_outer, terms
