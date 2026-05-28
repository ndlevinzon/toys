"""
Spectral / reciprocal-space view of orientation electrostatics.

Rigid rotations (fixed translation) preserve all pairwise distances, so the
screened Coulomb sum

    H_pair = sum_{i<j} q_i q_j G_epsilon,kappa(|r_i - r_j|)

is **constant on SO(3)** and is precomputed once in the body frame.

What still varies with orientation are slab **intrinsic** terms that depend on
lab-frame z and rotated dipoles:

    H_int = sum_i q_i phi_0(z_i) + sum_i mu_i · E_0(z_i)

For a laterally homogeneous slab, phi_0 and E_0 are functions of z only; we
tabulate them on a 1D grid (Green's function along the film normal) and
interpolate — the analog of a 1D spectral (Fourier in z) representation without
a full 3D PB FFT.

A full heterogeneous Poisson–Boltzmann solve in reciprocal space (FFT in x,y
and mode expansion in z) is not implemented here; pairwise precomputation
captures the dominant cost for rotating point charges.
"""

from __future__ import annotations

import numpy as np

from anisotropy.awi_field import VitrifiedWaterSlab
from anisotropy.patches import COULOMB_SCALE


def precompute_screened_pair_energy(
    cents_body: np.ndarray,
    q_arr: np.ndarray,
    slab: VitrifiedWaterSlab,
    *,
    homogeneous_epsilon: float | None,
    homogeneous_kappa: float | None,
    r_smooth: float,
    coulomb_scale: float | None,
    pair_cutoff_angstrom: float | None,
) -> float:
    """
    Pairwise screened Coulomb energy in the body frame (rotation-invariant).

    Uses mid-point (epsilon, kappa) at pair centroid z in the **body** frame,
    identical to the lab-frame result for H_pair when only R changes.
    """
    n_p = int(cents_body.shape[0])
    if n_p <= 1:
        return 0.0

    z_arr = cents_body[:, 2]
    samp = slab.sample_fields(z_arr, blend_interfaces=True)
    eps = 0.5 * (
        np.asarray(samp["epsilon_parallel"], dtype=np.float64)
        + np.asarray(samp["epsilon_perpendicular"], dtype=np.float64)
    )
    kap = np.asarray(samp["kappa"], dtype=np.float64)
    if homogeneous_epsilon is not None:
        eps = np.full_like(eps, float(homogeneous_epsilon))
    if homogeneous_kappa is not None:
        kap = np.full_like(kap, float(homogeneous_kappa))

    k_coul = float(coulomb_scale) if coulomb_scale is not None else COULOMB_SCALE
    diff = cents_body[:, np.newaxis, :] - cents_body[np.newaxis, :, :]
    r_mag = np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))
    ii, jj = np.triu_indices(n_p, k=1)
    rm = r_mag[ii, jj]
    if pair_cutoff_angstrom is not None and float(pair_cutoff_angstrom) > 0:
        keep = rm <= float(pair_cutoff_angstrom)
        ii, jj, rm = ii[keep], jj[keep], rm[keep]
    if rm.size == 0:
        return 0.0
    rm = np.maximum(rm, float(r_smooth))
    eps_mid = 0.5 * (eps[ii] + eps[jj])
    kap_mid = 0.5 * (kap[ii] + kap[jj])
    g = k_coul * np.exp(-np.maximum(kap_mid, 0.0) * rm) / (
        np.maximum(eps_mid, 1.0) * rm
    )
    return float(np.sum(q_arr[ii] * q_arr[jj] * g))


class SlabZFieldLUT:
    """1D lookup tables for phi_0(z) and E_0,z(z) along the slab normal."""

    __slots__ = ("z_grid", "phi0", "ez")

    def __init__(self, z_grid: np.ndarray, phi0: np.ndarray, ez: np.ndarray) -> None:
        self.z_grid = np.asarray(z_grid, dtype=np.float64)
        self.phi0 = np.asarray(phi0, dtype=np.float64)
        self.ez = np.asarray(ez, dtype=np.float64)

    @classmethod
    def from_slab(
        cls,
        slab: VitrifiedWaterSlab,
        *,
        z_min: float | None = None,
        z_max: float | None = None,
        dz: float = 0.25,
    ) -> SlabZFieldLUT:
        lo = 0.0 if z_min is None else float(z_min)
        hi = float(slab.thickness_angstrom) if z_max is None else float(z_max)
        pad = 20.0
        z = np.arange(lo - pad, hi + pad + dz, dz, dtype=np.float64)
        samp = slab.sample_fields(z, blend_interfaces=True)
        phi0 = np.asarray(samp["phi_0"], dtype=np.float64)
        ez = np.asarray(samp["E_0"], dtype=np.float64)
        return cls(z, phi0, ez)

    def sample(self, z_lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = np.asarray(z_lab, dtype=np.float64)
        phi = np.interp(z, self.z_grid, self.phi0)
        ez = np.interp(z, self.z_grid, self.ez)
        return phi, ez


def intrinsic_electrostatic_energy(
    z_lab: np.ndarray,
    q_arr: np.ndarray,
    mu_lab: np.ndarray,
    lut: SlabZFieldLUT,
    *,
    use_phi0: bool,
    use_dipole: bool,
) -> tuple[float, dict]:
    """Orientation-dependent part of H_el (after constant H_pair)."""
    phi0, ez = lut.sample(z_lab)
    sqp = float(np.dot(q_arr, phi0)) if use_phi0 else 0.0
    dip = float(np.sum(mu_lab[:, 2] * ez)) if use_dipole else 0.0
    return sqp + dip, {
        "sum_q_phi0": sqp,
        "dipole_dot_E0": dip,
    }


def yukawa_kspace_energy_homogeneous(
    cents: np.ndarray,
    q_arr: np.ndarray,
    *,
    epsilon: float = 78.0,
    kappa: float = 0.0,
    box_length: float = 200.0,
    grid_n: int = 64,
    coulomb_scale: float | None = None,
) -> float:
    """
    Reference energy via FFT convolution on a periodic box (homogeneous medium).

    Deposits charges on a grid, applies

        G_hat(k) = 4 pi / (epsilon |k|^2 + kappa^2)

    and returns 0.5 * integral rho * (G * rho). Useful for cross-checking the
    direct pairwise sum in ``precompute_screened_pair_energy`` (not used in the
    hot path).
    """
    k_coul = float(coulomb_scale) if coulomb_scale is not None else COULOMB_SCALE
    n = int(grid_n)
    L = float(box_length)
    rho = np.zeros((n, n, n), dtype=np.float64)
    origin = -0.5 * L
    h = L / n
    for c, q in zip(cents, q_arr):
        ix = int(np.clip((c[0] - origin) / h, 0, n - 1))
        iy = int(np.clip((c[1] - origin) / h, 0, n - 1))
        iz = int(np.clip((c[2] - origin) / h, 0, n - 1))
        rho[ix, iy, iz] += q / (h**3)

    rho_hat = np.fft.fftn(rho)
    fx = np.fft.fftfreq(n, d=h) * 2.0 * np.pi
    fy, fz = fx, fx
    kx, ky, kz = np.meshgrid(fx, fy, fz, indexing="ij")
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0
    g_hat = 4.0 * np.pi / (float(epsilon) * k2 + float(kappa) ** 2)
    g_hat[0, 0, 0] = 0.0
    phi_hat = rho_hat * g_hat * k_coul
    energy = 0.5 * float(np.real(np.vdot(rho_hat, phi_hat)) / rho.size)
    return energy
