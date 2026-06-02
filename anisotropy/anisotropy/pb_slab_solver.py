"""
Fast linearized Poisson–Boltzmann for a laterally homogeneous AWI slab.

The cryo-EM vitrified-water slab has ε(z), κ(z) from :class:`~anisotropy.awi_field.VitrifiedWaterSlab`.
We solve on the same Cartesian lattice as the lattice gas:

    -∇·(ε(z) ∇φ) + κ(z)² ε(z) φ = 4π k_coul ρ

with **FFT in (x, y)** and a **tridiagonal solve along z** for each lateral mode.
Boundaries: Dirichlet φ = 0 at the top/bottom of the grid (far-field box).

This is O(N log N) in the lateral plane and O(N_z) per mode — much faster than
3D finite-element PB, while resolving the dielectric jump at the air–water interface.

**Units:** patch charges in e, positions in Å; ``COULOMB_SCALE`` matches :mod:`patches`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from anisotropy.awi_field import VitrifiedWaterSlab
from anisotropy.lattice_solvent_hamiltonian import CartesianLattice
from anisotropy.patches import COULOMB_SCALE, PatchParameterization
from anisotropy.spectral_electrostatics import SlabZFieldLUT, intrinsic_electrostatic_energy


@dataclass(frozen=True)
class PBSolverParams:
    """Knobs for the slab FFT–PB solver."""

    enabled: bool = True
    method: Literal["screened_pair", "pb_slab"] = "pb_slab"
    coarse_factor: int = 2
    charge_sigma_voxels: float = 1.25
    coulomb_scale: float | None = None
    use_intrinsic_potential: bool = True
    use_dipole_E0_component: bool = True
    include_slab_phi0_in_pb: bool = False


def _k_coulomb(scale: float | None) -> float:
    return float(scale) if scale is not None else COULOMB_SCALE


def deposit_charges_trilinear(
    lattice: CartesianLattice,
    xyz_flat: np.ndarray,
    charges: np.ndarray,
    *,
    sigma_voxels: float = 1.25,
) -> np.ndarray:
    """
    Spread point charges onto a grid (e / Å³) with a compact Gaussian stencil.
    """
    nx, ny, nz = lattice.shape
    rho = np.zeros((nx, ny, nz), dtype=np.float64)
    h = float(lattice.spacing)
    o = lattice.origin.reshape(3)
    sig = max(float(sigma_voxels), 0.5) * h
    inv_2s2 = 1.0 / (2.0 * sig * sig)

    for c, q in zip(np.asarray(xyz_flat, dtype=np.float64).reshape(-1, 3), charges):
        rel = (c - o) / h - 0.5
        i0 = int(np.floor(rel[0]))
        j0 = int(np.floor(rel[1]))
        k0 = int(np.floor(rel[2]))
        stencil: list[tuple[int, int, int, float]] = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    i, j, k = i0 + di, j0 + dj, k0 + dk
                    if i < 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
                        continue
                    center = o + (np.array([i, j, k], dtype=np.float64) + 0.5) * h
                    d2 = float(np.sum((center - c) ** 2))
                    w = float(np.exp(-d2 * inv_2s2))
                    stencil.append((i, j, k, w))
        wsum = sum(w for _, _, _, w in stencil)
        if wsum < 1e-30:
            continue
        for i, j, k, w in stencil:
            rho[i, j, k] += float(q) * w / (wsum * h**3)
    return rho


def _thomas_solve_batch(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve tridiagonal systems for all rows of shape ``(n_mode, nz)``."""
    n_mode, nz = diag.shape
    cp = np.zeros((n_mode, nz), dtype=np.float64)
    dp = np.zeros((n_mode, nz), dtype=np.float64)
    x = np.zeros((n_mode, nz), dtype=np.float64)

    d0 = diag[:, 0].copy()
    d0[np.abs(d0) < 1e-30] = 1e-30
    cp[:, 0] = upper[:, 0] / d0
    dp[:, 0] = rhs[:, 0] / d0
    for i in range(1, nz):
        denom = diag[:, i] - lower[:, i] * cp[:, i - 1]
        denom[np.abs(denom) < 1e-30] = 1e-30
        if i < nz - 1:
            cp[:, i] = upper[:, i] / denom
        dp[:, i] = (rhs[:, i] - lower[:, i] * dp[:, i - 1]) / denom
    x[:, -1] = dp[:, -1]
    for i in range(nz - 2, -1, -1):
        x[:, i] = dp[:, i] - cp[:, i] * x[:, i + 1]
    return x


def solve_linearized_pb_slab(
    rho: np.ndarray,
    lattice: CartesianLattice,
    eps_z: np.ndarray,
    kappa_z: np.ndarray,
    *,
    coulomb_scale: float | None = None,
) -> np.ndarray:
    """
    Return φ on the grid solving linearized PB with ε(z), κ(z).

    ``rho`` shape matches ``lattice.shape`` (charge density, e/Å³).
    """
    nx, ny, nz = rho.shape
    h = float(lattice.spacing)
    dz = h
    k_c = _k_coulomb(coulomb_scale)

    eps = np.maximum(np.asarray(eps_z, dtype=np.float64).reshape(nz), 1.0)
    kap = np.maximum(np.asarray(kappa_z, dtype=np.float64).reshape(nz), 0.0)
    eps_half = 0.5 * (eps[:-1] + eps[1:])

    fx = np.fft.fftfreq(nx, d=h) * 2.0 * np.pi
    fy = np.fft.fftfreq(ny, d=h) * 2.0 * np.pi
    kx, ky = np.meshgrid(fx, fy, indexing="ij")
    k2 = (kx**2 + ky**2).reshape(-1)

    rho_k = np.fft.fftn(rho, axes=(0, 1))
    rho_modes = rho_k.reshape(nx * ny, nz)
    n_mode = nx * ny

    lower = np.zeros((n_mode, nz), dtype=np.float64)
    diag = np.zeros((n_mode, nz), dtype=np.float64)
    upper = np.zeros((n_mode, nz), dtype=np.float64)
    rhs = np.zeros((n_mode, nz), dtype=np.float64)

    for i in range(1, nz - 1):
        lower[:, i] = eps_half[i - 1] / (dz * dz)
        upper[:, i] = eps_half[i] / (dz * dz)

    for i in range(nz):
        if i == 0 or i == nz - 1:
            diag[:, i] = 1.0
            rhs[:, i] = 0.0
        else:
            a = lower[:, i]
            c = upper[:, i]
            diag[:, i] = -(a + c) - k2 * eps[i] - (kap[i] ** 2) * eps[i]
            rhs[:, i] = -4.0 * np.pi * k_c * np.real(rho_modes[:, i])

    phi_modes = _thomas_solve_batch(lower, diag, upper, rhs)
    phi_k = phi_modes.reshape(nx, ny, nz)
    phi = np.real(np.fft.ifftn(phi_k, axes=(0, 1)))
    return phi


def sample_trilinear(
    field: np.ndarray,
    lattice: CartesianLattice,
    points: np.ndarray,
) -> np.ndarray:
    """Trilinear sample ``field`` at lab-frame points ``(N, 3)``."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    nx, ny, nz = field.shape
    h = float(lattice.spacing)
    o = lattice.origin.reshape(3)
    rel = (pts - o) / h - 0.5
    out = np.zeros(len(pts), dtype=np.float64)
    for p, r in enumerate(rel):
        i0 = int(np.floor(r[0]))
        j0 = int(np.floor(r[1]))
        k0 = int(np.floor(r[2]))
        if i0 < 0 or i0 >= nx - 1 or j0 < 0 or j0 >= ny - 1 or k0 < 0 or k0 >= nz - 1:
            continue
        tx, ty, tz = r[0] - i0, r[1] - j0, r[2] - k0
        for di, wx in ((0, 1.0 - tx), (1, tx)):
            for dj, wy in ((0, 1.0 - ty), (1, ty)):
                for dk, wz in ((0, 1.0 - tz), (1, tz)):
                    out[p] += (
                        wx
                        * wy
                        * wz
                        * field[i0 + di, j0 + dj, k0 + dk]
                    )
    return out


@dataclass
class SlabPBSolver:
    """
    Reusable PB solver on a (possibly coarse) sub-grid of the orientation lattice.
    """

    lattice: CartesianLattice
    lab_xyz_flat: np.ndarray
    eps_z: np.ndarray
    kappa_z: np.ndarray
    params: PBSolverParams
    slab_lut: SlabZFieldLUT | None = None
    _coarse_factor: int = 1

    @classmethod
    def from_lattice(
        cls,
        lattice: CartesianLattice,
        slab: VitrifiedWaterSlab,
        params: PBSolverParams,
        *,
        lab_xyz_flat: np.ndarray | None = None,
        homogeneous_epsilon: float | None = None,
        homogeneous_kappa: float | None = None,
    ) -> SlabPBSolver:
        if lab_xyz_flat is None:
            lab_xyz_flat = lattice.grid_centers_xyz().reshape(-1, 3)
        cf = max(1, int(params.coarse_factor))
        nx, ny, nz = lattice.shape
        h = float(lattice.spacing)
        z_lab = lattice.grid_centers_xyz()[0, 0, :, 2]
        samp = slab.sample_fields(z_lab, blend_interfaces=True)
        eps = 0.5 * (
            np.asarray(samp["epsilon_parallel"], dtype=np.float64)
            + np.asarray(samp["epsilon_perpendicular"], dtype=np.float64)
        )
        kap = np.asarray(samp["kappa"], dtype=np.float64)
        if homogeneous_epsilon is not None:
            eps = np.full_like(eps, float(homogeneous_epsilon))
        if homogeneous_kappa is not None:
            kap = np.full_like(kap, float(homogeneous_kappa))

        if cf > 1:
            nx_c = max(4, nx // cf)
            ny_c = max(4, ny // cf)
            nz_c = max(4, nz // cf)
            h_c = h * cf
            o = lattice.origin.reshape(3)
            lat_c = CartesianLattice(
                origin=o,
                spacing=h_c,
                shape=(nx_c, ny_c, nz_c),
            )
            z_c = lat_c.grid_centers_xyz()[0, 0, :, 2]
            samp_c = slab.sample_fields(z_c, blend_interfaces=True)
            eps_c = 0.5 * (
                np.asarray(samp_c["epsilon_parallel"], dtype=np.float64)
                + np.asarray(samp_c["epsilon_perpendicular"], dtype=np.float64)
            )
            kap_c = np.asarray(samp_c["kappa"], dtype=np.float64)
            if homogeneous_epsilon is not None:
                eps_c = np.full_like(eps_c, float(homogeneous_epsilon))
            if homogeneous_kappa is not None:
                kap_c = np.full_like(kap_c, float(homogeneous_kappa))
            lut = SlabZFieldLUT.from_slab(slab, z_max=float(slab.thickness_angstrom))
            inst = cls(
                lattice=lat_c,
                lab_xyz_flat=lat_c.grid_centers_xyz().reshape(-1, 3),
                eps_z=eps_c,
                kappa_z=kap_c,
                params=params,
                slab_lut=lut,
            )
            inst._coarse_factor = cf
            return inst

        lut = SlabZFieldLUT.from_slab(slab, z_max=float(slab.thickness_angstrom))
        return cls(
            lattice=lattice,
            lab_xyz_flat=np.asarray(lab_xyz_flat, dtype=np.float64),
            eps_z=eps,
            kappa_z=kap,
            params=params,
            slab_lut=lut,
            _coarse_factor=1,
        )

    def _map_points_to_solver_grid(self, points: np.ndarray) -> np.ndarray:
        """Map lab points to coarse solver lattice if needed."""
        return np.asarray(points, dtype=np.float64)

    def energy_from_charges(
        self,
        centroids: np.ndarray,
        q_arr: np.ndarray,
        mu_mat: np.ndarray,
        *,
        slab: VitrifiedWaterSlab | None = None,
    ) -> tuple[float, dict]:
        """
        Grid PB interaction energy + optional slab intrinsic φ₀, E₀ terms.
        """
        cents = np.asarray(centroids, dtype=np.float64).reshape(-1, 3)
        q = np.asarray(q_arr, dtype=np.float64).reshape(-1)
        mu = np.asarray(mu_mat, dtype=np.float64).reshape(-1, 3)
        n_p = len(q)
        if n_p == 0:
            return 0.0, {"method": "pb_slab", "H_pb_grid": 0.0}

        rho = deposit_charges_trilinear(
            self.lattice,
            cents,
            q,
            sigma_voxels=self.params.charge_sigma_voxels,
        )
        phi = solve_linearized_pb_slab(
            rho,
            self.lattice,
            self.eps_z,
            self.kappa_z,
            coulomb_scale=self.params.coulomb_scale,
        )
        k_c = _k_coulomb(self.params.coulomb_scale)
        phi_at = sample_trilinear(phi, self.lattice, cents)
        h_pb = k_c * float(np.dot(q, phi_at))

        h_int = 0.0
        int_terms: dict[str, float] = {}
        if self.slab_lut is not None and (
            self.params.use_intrinsic_potential or self.params.use_dipole_E0_component
        ):
            h_int, int_terms = intrinsic_electrostatic_energy(
                cents[:, 2],
                q,
                mu,
                self.slab_lut,
                use_phi0=self.params.use_intrinsic_potential
                and not self.params.include_slab_phi0_in_pb,
                use_dipole=self.params.use_dipole_E0_component,
            )

        return float(h_pb + h_int), {
            "method": "pb_slab",
            "H_pb_grid": float(h_pb),
            "H_pb_intrinsic": float(h_int),
            "H_pair_screened": 0.0,
            "phi_at_charges_mean": float(np.mean(phi_at)),
            **int_terms,
        }


def electrostatic_energy_dispatch(
    param: PatchParameterization,
    slab: VitrifiedWaterSlab,
    *,
    pb_solver: SlabPBSolver | None,
    method: str,
    homogeneous_epsilon: float | None = None,
    homogeneous_kappa: float | None = None,
    use_intrinsic_potential: bool = True,
    use_dipole_E0_component: bool = True,
    r_smooth: float = 1e-2,
    coulomb_scale: float | None = None,
    pair_cutoff_angstrom: float | None = None,
) -> tuple[float, dict]:
    """Route to slab PB or legacy screened pairwise electrostatics."""
    mode = str(method).strip().lower()
    if mode == "pb_slab" and pb_solver is not None:
        patches = param.patches
        cents = np.stack([np.asarray(p.centroid, dtype=np.float64) for p in patches], axis=0)
        q = np.array([float(p.charge) for p in patches], dtype=np.float64)
        mu = np.stack([np.asarray(p.dipole, dtype=np.float64) for p in patches], axis=0)
        return pb_solver.energy_from_charges(cents, q, mu, slab=slab)

    from anisotropy.lattice_solvent_hamiltonian import electrostatic_energy_pb_like

    return electrostatic_energy_pb_like(
        param,
        slab,
        homogeneous_epsilon=homogeneous_epsilon,
        homogeneous_kappa=homogeneous_kappa,
        use_intrinsic_potential=use_intrinsic_potential,
        use_dipole_E0_component=use_dipole_E0_component,
        r_smooth=r_smooth,
        coulomb_scale=coulomb_scale,
    )
