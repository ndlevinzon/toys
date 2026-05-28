"""
Fast per-pose Hamiltonian evaluation for orientation / MCMC sweeps.

Avoids rebuilding :class:`~anisotropy.patches.PatchParameterization` and Python
loops over patches for ray indicators. Optional electrostatic pair cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from anisotropy.awi_field import VitrifiedWaterSlab
from anisotropy.lattice_solvent_hamiltonian import (
    CanonicalInteriorCache,
    CartesianLattice,
    HybridHamiltonianCouplings,
    HybridHamiltonianResult,
    TernaryOccupancy,
    coupling_flex_penalty,
)
from anisotropy.patches import PatchParameterization
from anisotropy.spectral_electrostatics import (
    SlabZFieldLUT,
    intrinsic_electrostatic_energy,
    precompute_screened_pair_energy,
)

def compute_patch_air_indicators_batched(
    centroids: np.ndarray,
    normals: np.ndarray,
    occupancy: np.ndarray,
    lattice: CartesianLattice,
    interior: np.ndarray,
    *,
    max_steps: int = 48,
    ray_step: float = 2.0,
    require_first_hit_solvent_steps: int = 2,
    mode: Literal["binary", "ternary"] = "binary",
) -> np.ndarray:
    """
    Vectorized \\mathcal{I}^{\\mathrm{air}}_f for all patches (n_patches,).

    Equivalent logic to per-patch ``patch_air_indicator`` (outward rays, skip
    protein interior voxels).
    """
    cents = np.asarray(centroids, dtype=np.float64).reshape(-1, 3)
    norms = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    n_p = cents.shape[0]
    if n_p == 0:
        return np.zeros(0, dtype=np.float64)

    h = float(lattice.spacing)
    o = lattice.origin.reshape(3)
    nx, ny, nz = lattice.shape
    s_max = int(max_steps)
    req = max(int(require_first_hit_solvent_steps), 1)

    n_hat = norms / (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-15)
    steps = float(ray_step) * np.arange(1, s_max + 1, dtype=np.float64)
    pts = cents[:, np.newaxis, :] + steps[np.newaxis, :, np.newaxis] * n_hat[:, np.newaxis, :]

    rel = (pts - o) / h - 0.5
    ij = np.floor(rel).astype(np.int64)
    valid = (
        (ij[..., 0] >= 0)
        & (ij[..., 0] < nx)
        & (ij[..., 1] >= 0)
        & (ij[..., 1] < ny)
        & (ij[..., 2] >= 0)
        & (ij[..., 2] < nz)
    )
    ii = np.clip(ij[..., 0], 0, nx - 1)
    jj = np.clip(ij[..., 1], 0, ny - 1)
    kk = np.clip(ij[..., 2], 0, nz - 1)

    inside_ray = interior[ii, jj, kk]
    solvent_step = valid & (~inside_ray)

    if mode == "ternary":
        occ_at = occupancy[ii, jj, kk]
        is_air = occ_at == int(TernaryOccupancy.AIR)
    else:
        is_air = occupancy[ii, jj, kk] == 0

    air_on_path = is_air & solvent_step
    cum_sol = np.cumsum(solvent_step.astype(np.int32), axis=1)
    shifted = np.concatenate(
        [np.zeros((n_p, 1), dtype=np.int32), cum_sol[:, :-1]], axis=1
    )
    qualified = air_on_path & (shifted >= req)
    enough_solvent = cum_sol[:, -1] >= req
    return (qualified.any(axis=1) & enough_solvent).astype(np.float64)


def lab_interior_masks_batch(
    canon_cache: CanonicalInteriorCache,
    lab_lattice: CartesianLattice,
    R_batch: np.ndarray,
    t: np.ndarray,
    lab_xyz_flat: np.ndarray,
) -> np.ndarray:
    """(B, nx, ny, nz) interior masks for B rotation matrices."""
    R_batch = np.asarray(R_batch, dtype=np.float64).reshape(-1, 3, 3)
    B = R_batch.shape[0]
    X = np.asarray(lab_xyz_flat, dtype=np.float64).reshape(-1, 3)
    tv = np.asarray(t, dtype=np.float64).reshape(3)
    x_can = np.einsum("nj,bjk->bnj", X - tv, R_batch)

    o = canon_cache.can_lattice.origin.reshape(3)
    h = float(canon_cache.can_lattice.spacing)
    nx_c, ny_c, nz_c = canon_cache.can_lattice.shape
    nx, ny, nz = lab_lattice.shape

    rel = (x_can - o) / h - 0.5
    ij = np.floor(rel).astype(np.int64)
    in_b = (
        (ij[..., 0] >= 0)
        & (ij[..., 0] < nx_c)
        & (ij[..., 1] >= 0)
        & (ij[..., 1] < ny_c)
        & (ij[..., 2] >= 0)
        & (ij[..., 2] < nz_c)
    )
    out = np.zeros((B, X.shape[0]), dtype=bool)
    ii = np.clip(ij[..., 0], 0, nx_c - 1)
    jj = np.clip(ij[..., 1], 0, ny_c - 1)
    kk = np.clip(ij[..., 2], 0, nz_c - 1)
    for b in range(B):
        mask_b = in_b[b]
        out[b, mask_b] = canon_cache.inside_can[ii[b, mask_b], jj[b, mask_b], kk[b, mask_b]]
    return out.reshape(B, *lab_lattice.shape)


def electrostatic_energy_from_arrays(
    cents: np.ndarray,
    q_arr: np.ndarray,
    mu_mat: np.ndarray,
    slab: VitrifiedWaterSlab,
    *,
    homogeneous_epsilon: float | None,
    homogeneous_kappa: float | None,
    use_intrinsic_potential: bool,
    use_dipole_E0_component: bool,
    r_smooth: float,
    coulomb_scale: float | None,
    pair_cutoff_angstrom: float | None,
) -> tuple[float, dict]:
    """Screened pairwise + intrinsic terms; optional distance cutoff on pairs."""
    n_p = cents.shape[0]
    if n_p == 0:
        return 0.0, {
            "H_pair_screened": 0.0,
            "sum_q_phi0": 0.0,
            "dipole_dot_E0": 0.0,
        }

    z_arr = cents[:, 2]
    samp = slab.sample_fields(z_arr, blend_interfaces=True)
    eps = 0.5 * (
        np.asarray(samp["epsilon_parallel"], dtype=np.float64)
        + np.asarray(samp["epsilon_perpendicular"], dtype=np.float64)
    )
    kap = np.asarray(samp["kappa"], dtype=np.float64)
    phi0 = np.asarray(samp["phi_0"], dtype=np.float64)
    ez_arr = np.asarray(samp["E_0"], dtype=np.float64)

    if homogeneous_epsilon is not None:
        eps = np.full_like(eps, float(homogeneous_epsilon))
    if homogeneous_kappa is not None:
        kap = np.full_like(kap, float(homogeneous_kappa))

    k_coul = float(coulomb_scale) if coulomb_scale is not None else COULOMB_SCALE
    h_pair = 0.0
    if n_p > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        r_mag = np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))
        ii, jj = np.triu_indices(n_p, k=1)
        rm = r_mag[ii, jj]
        if pair_cutoff_angstrom is not None and float(pair_cutoff_angstrom) > 0:
            keep = rm <= float(pair_cutoff_angstrom)
            ii, jj = ii[keep], jj[keep]
            rm = rm[keep]
        if rm.size > 0:
            eps_mid = 0.5 * (eps[ii] + eps[jj])
            kap_mid = 0.5 * (kap[ii] + kap[jj])
            rm = np.maximum(rm, float(r_smooth))
            eps_eff = np.maximum(eps_mid, 1.0)
            kap_eff = np.maximum(kap_mid, 0.0)
            g = k_coul * np.exp(-kap_eff * rm) / (eps_eff * rm)
            h_pair = float(np.sum(q_arr[ii] * q_arr[jj] * g))

    sqp = float(np.dot(q_arr, phi0)) if use_intrinsic_potential else 0.0
    dipole_term = (
        float(np.sum(mu_mat[:, 2] * ez_arr)) if use_dipole_E0_component else 0.0
    )
    diag = {
        "H_pair_screened": float(h_pair),
        "sum_q_phi0": float(sqp),
        "dipole_dot_E0": float(dipole_term),
    }
    return float(h_pair + sqp + dipole_term), diag


def film_energy_from_arrays(
    z_lab: np.ndarray,
    areas: np.ndarray,
    softness: np.ndarray,
    slab: VitrifiedWaterSlab,
    *,
    lambda_film: float,
    margin_angstrom: float,
) -> float:
    """Vectorized film coupling (same as ``film_coupling_energy``)."""
    z_arr = np.asarray(z_lab, dtype=np.float64).reshape(-1)
    dz_bot = np.clip(z_arr, 0.0, float(slab.thickness_angstrom))
    dz_top = np.clip(float(slab.thickness_angstrom) - z_arr, 0.0, float(slab.thickness_angstrom))
    from anisotropy.lattice_solvent_hamiltonian import _FILM_SURFACE_COVERS

    ww = np.exp(-(dz_bot**2) / (margin_angstrom**2)) * float(
        slab.bottom.coverage in _FILM_SURFACE_COVERS
    )
    wt = np.exp(-(dz_top**2) / (margin_angstrom**2)) * float(
        slab.top.coverage in _FILM_SURFACE_COVERS
    )
    mw = ww * float(slab.bottom.mechanical_age)
    mt = wt * float(slab.top.mechanical_age)
    weights_arr = np.clip((ww + wt + mw + mt) / 4.0, 0.0, 1.0)
    return float(lambda_film * np.sum(areas * softness * weights_arr))


@dataclass
class FastOrientationEvaluator:
    """
    Reusable evaluator: fixed occupancy / solvation; only pose (R, t) changes.

    Typical speedup vs ``evaluate_hybrid_hamiltonian``: 5–20× (fewer Python
    loops, no per-patch object construction).
    """

    occupancy: np.ndarray
    lattice: CartesianLattice
    slab: VitrifiedWaterSlab
    coeffs: HybridHamiltonianCouplings
    canon_cache: CanonicalInteriorCache
    lab_xyz_flat: np.ndarray
    t_base: np.ndarray
    h_solv: float
    solv_terms: dict
  # patch arrays in canonical (body) frame — rotate with R.T
    cents_can: np.ndarray
    norms_can: np.ndarray
    mu_can: np.ndarray
    areas: np.ndarray
    hydr: np.ndarray
    pola: np.ndarray
    hbon: np.ndarray
    q_arr: np.ndarray
    softness: np.ndarray
    h_flex: float
    eta_flex: float
    occupancy_mode: Literal["binary", "ternary"]
    ray_max_steps: int
    ray_step: float
    require_first_solvent: int
    el_r_smooth: float
    el_use_phi0: bool
    el_use_dipole: bool
    el_coulomb_scale: float | None
    el_pair_cutoff: float | None
    film_margin: float
    h_el_pair_const: float
    slab_lut: SlabZFieldLUT
    use_invariant_pair: bool
    _last_result: HybridHamiltonianResult | None = None

    @classmethod
    def build(
        cls,
        *,
        param: PatchParameterization,
        occupancy: np.ndarray,
        lattice: CartesianLattice,
        slab: VitrifiedWaterSlab,
        coeffs: HybridHamiltonianCouplings,
        canon_cache: CanonicalInteriorCache,
        lab_xyz_flat: np.ndarray,
        t_base: np.ndarray,
        h_solv_and_terms: tuple[float, dict],
        ising_params: Any | None = None,
    ) -> FastOrientationEvaluator:
        patches = param.patches
        n_p = len(patches)
        cents_can = np.stack(
            [np.asarray(p.centroid, dtype=np.float64) for p in patches], axis=0
        )
        norms_can = np.stack(
            [np.asarray(p.normal, dtype=np.float64) for p in patches], axis=0
        )
        mu_can = np.stack([np.asarray(p.dipole, dtype=np.float64) for p in patches], axis=0)

        ray_max_steps = 48
        ray_step_fraction = 0.55
        require_first = 2
        film_margin = 7.0
        el_r_smooth = 1e-2
        el_use_phi0 = True
        el_use_dipole = True
        el_coulomb_scale = None
        el_pair_cutoff = 80.0
        occupancy_mode: Literal["binary", "ternary"] = "binary"
        eta_flex = coeffs.eta_flex

        if ising_params is not None:
            ray_max_steps = ising_params.ray.ray_max_steps
            ray_step_fraction = ising_params.ray.ray_step_fraction
            require_first = ising_params.ray.require_first_hit_solvent_steps
            film_margin = ising_params.film.margin_angstrom
            el_r_smooth = ising_params.electrostatics.r_smooth_angstrom
            el_use_phi0 = ising_params.electrostatics.use_intrinsic_potential
            el_use_dipole = ising_params.electrostatics.use_dipole_E0_component
            el_coulomb_scale = ising_params.electrostatics.coulomb_scale
            perf = ising_params.performance
            co = perf.electrostatic_pair_cutoff_angstrom
            if co is not None:
                el_pair_cutoff = co
            occupancy_mode = ising_params.solv.occupancy_mode  # type: ignore[assignment]
            eta_flex = ising_params.flex.eta_flex

        u_vals = np.array([p.softness for p in patches], dtype=np.float64)
        h_flex, _ = coupling_flex_penalty(param, eta_flex=eta_flex)
        h_solv, solv_terms = h_solv_and_terms

        use_inv_pair = True
        if ising_params is not None:
            use_inv_pair = bool(
                ising_params.performance.precompute_rotation_invariant_pair
            )

        h_pair = 0.0
        if use_inv_pair and n_p > 1:
            h_pair = precompute_screened_pair_energy(
                cents_can,
                np.array([float(p.charge) for p in patches], dtype=np.float64),
                slab,
                homogeneous_epsilon=coeffs.homogeneous_epsilon,
                homogeneous_kappa=coeffs.homogeneous_kappa,
                r_smooth=float(el_r_smooth),
                coulomb_scale=el_coulomb_scale,
                pair_cutoff_angstrom=el_pair_cutoff,
            )

        slab_lut = SlabZFieldLUT.from_slab(
            slab,
            z_min=0.0,
            z_max=float(slab.thickness_angstrom),
            dz=max(0.25, float(lattice.spacing) * 0.1),
        )

        return cls(
            occupancy=occupancy,
            lattice=lattice,
            slab=slab,
            coeffs=coeffs,
            canon_cache=canon_cache,
            lab_xyz_flat=lab_xyz_flat,
            t_base=np.asarray(t_base, dtype=np.float64).reshape(3),
            h_solv=float(h_solv),
            solv_terms=dict(solv_terms),
            cents_can=cents_can,
            norms_can=norms_can,
            mu_can=mu_can,
            areas=np.array([float(p.area) for p in patches], dtype=np.float64),
            hydr=np.array([float(p.hydropathy) for p in patches], dtype=np.float64),
            pola=np.array([float(p.polar_density) for p in patches], dtype=np.float64),
            hbon=np.array([float(p.hbond_score) for p in patches], dtype=np.float64),
            q_arr=np.array([float(p.charge) for p in patches], dtype=np.float64),
            softness=u_vals,
            h_flex=float(h_flex),
            eta_flex=float(eta_flex),
            occupancy_mode=occupancy_mode,
            ray_max_steps=int(ray_max_steps),
            ray_step=float(ray_step_fraction) * float(lattice.spacing),
            require_first_solvent=int(require_first),
            el_r_smooth=float(el_r_smooth),
            el_use_phi0=bool(el_use_phi0),
            el_use_dipole=bool(el_use_dipole),
            el_coulomb_scale=el_coulomb_scale,
            el_pair_cutoff=el_pair_cutoff,
            film_margin=float(film_margin),
            h_el_pair_const=float(h_pair),
            slab_lut=slab_lut,
            use_invariant_pair=bool(use_inv_pair),
        )

    def state_dict(self) -> dict:
        """Pickle-friendly snapshot for process-pool workers."""
        return {
            "occupancy": self.occupancy,
            "lattice_origin": self.lattice.origin,
            "lattice_spacing": self.lattice.spacing,
            "lattice_shape": self.lattice.shape,
            "slab": self.slab,
            "coeffs": self.coeffs,
            "inside_can": self.canon_cache.inside_can,
            "can_origin": self.canon_cache.can_lattice.origin,
            "can_spacing": self.canon_cache.can_lattice.spacing,
            "can_shape": self.canon_cache.can_lattice.shape,
            "lab_xyz_flat": self.lab_xyz_flat,
            "t_base": self.t_base,
            "h_solv": self.h_solv,
            "solv_terms": self.solv_terms,
            "cents_can": self.cents_can,
            "norms_can": self.norms_can,
            "mu_can": self.mu_can,
            "areas": self.areas,
            "hydr": self.hydr,
            "pola": self.pola,
            "hbon": self.hbon,
            "q_arr": self.q_arr,
            "softness": self.softness,
            "h_flex": self.h_flex,
            "eta_flex": self.eta_flex,
            "occupancy_mode": self.occupancy_mode,
            "ray_max_steps": self.ray_max_steps,
            "ray_step": self.ray_step,
            "require_first_solvent": self.require_first_solvent,
            "el_r_smooth": self.el_r_smooth,
            "el_use_phi0": self.el_use_phi0,
            "el_use_dipole": self.el_use_dipole,
            "el_coulomb_scale": self.el_coulomb_scale,
            "el_pair_cutoff": self.el_pair_cutoff,
            "film_margin": self.film_margin,
            "h_el_pair_const": self.h_el_pair_const,
            "slab_lut_z": self.slab_lut.z_grid,
            "slab_lut_phi0": self.slab_lut.phi0,
            "slab_lut_ez": self.slab_lut.ez,
            "use_invariant_pair": self.use_invariant_pair,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> FastOrientationEvaluator:
        lattice = CartesianLattice(
            origin=np.asarray(d["lattice_origin"], dtype=np.float64),
            spacing=float(d["lattice_spacing"]),
            shape=tuple(d["lattice_shape"]),
        )
        canon = CanonicalInteriorCache(
            inside_can=np.asarray(d["inside_can"], dtype=bool),
            can_lattice=CartesianLattice(
                origin=np.asarray(d["can_origin"], dtype=np.float64),
                spacing=float(d["can_spacing"]),
                shape=tuple(d["can_shape"]),
            ),
        )
        lut = SlabZFieldLUT(d["slab_lut_z"], d["slab_lut_phi0"], d["slab_lut_ez"])
        return cls(
            occupancy=d["occupancy"],
            lattice=lattice,
            slab=d["slab"],
            coeffs=d["coeffs"],
            canon_cache=canon,
            lab_xyz_flat=d["lab_xyz_flat"],
            t_base=d["t_base"],
            h_solv=float(d["h_solv"]),
            solv_terms=d["solv_terms"],
            cents_can=d["cents_can"],
            norms_can=d["norms_can"],
            mu_can=d["mu_can"],
            areas=d["areas"],
            hydr=d["hydr"],
            pola=d["pola"],
            hbon=d["hbon"],
            q_arr=d["q_arr"],
            softness=d["softness"],
            h_flex=float(d["h_flex"]),
            eta_flex=float(d["eta_flex"]),
            occupancy_mode=d["occupancy_mode"],
            ray_max_steps=int(d["ray_max_steps"]),
            ray_step=float(d["ray_step"]),
            require_first_solvent=int(d["require_first_solvent"]),
            el_r_smooth=float(d["el_r_smooth"]),
            el_use_phi0=bool(d["el_use_phi0"]),
            el_use_dipole=bool(d["el_use_dipole"]),
            el_coulomb_scale=d["el_coulomb_scale"],
            el_pair_cutoff=d["el_pair_cutoff"],
            film_margin=float(d["film_margin"]),
            h_el_pair_const=float(d["h_el_pair_const"]),
            slab_lut=lut,
            use_invariant_pair=bool(d.get("use_invariant_pair", True)),
        )

    def _pose_geometry(self, R: np.ndarray, t: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Rm = np.asarray(R, dtype=np.float64).reshape(3, 3)
        tv = self.t_base if t is None else np.asarray(t, dtype=np.float64).reshape(3)
        cents = self.cents_can @ Rm.T + tv.reshape(1, 3)
        norms = self.norms_can @ Rm.T
        norms /= np.linalg.norm(norms, axis=1, keepdims=True) + 1e-15
        mu = self.mu_can @ Rm.T
        return cents, norms, mu

    def evaluate(
        self,
        R: np.ndarray,
        t: np.ndarray | None = None,
        *,
        energy_only: bool = False,
    ) -> HybridHamiltonianResult | float:
        """Evaluate H for one pose. ``energy_only=True`` skips heavy diagnostics arrays."""
        Rm = np.asarray(R, dtype=np.float64).reshape(3, 3)
        interior = self.canon_cache.lab_interior_mask(
            self.lattice, Rm, self.t_base, lab_xyz_flat=self.lab_xyz_flat
        )
        cents, norms, mu = self._pose_geometry(Rm, t)

        i_air = compute_patch_air_indicators_batched(
            cents,
            norms,
            self.occupancy,
            self.lattice,
            interior,
            max_steps=self.ray_max_steps,
            ray_step=self.ray_step,
            require_first_hit_solvent_steps=self.require_first_solvent,
            mode=self.occupancy_mode,
        )
        i_int = np.zeros_like(i_air)

        h_hp = -float(self.coeffs.lambda_hp * np.dot(self.areas * self.hydr, i_air))
        h_pol = float(self.coeffs.lambda_p * np.dot(self.areas * self.pola, i_air))
        h_hb = -float(self.coeffs.lambda_hb * np.dot(self.areas * self.hbon, i_int))
        hp_pol = h_hp + h_pol + h_hb
        hydration_terms = {"H_hp": h_hp, "H_pol": h_pol, "H_hbond_channel": h_hb}

        if self.use_invariant_pair:
            h_int, el_int_terms = intrinsic_electrostatic_energy(
                cents[:, 2],
                self.q_arr,
                mu,
                self.slab_lut,
                use_phi0=self.el_use_phi0,
                use_dipole=self.el_use_dipole,
            )
            hel = self.h_el_pair_const + h_int
            electrostatic_terms = {
                "H_pair_screened": float(self.h_el_pair_const),
                **el_int_terms,
            }
        else:
            hel, electrostatic_terms = electrostatic_energy_from_arrays(
                cents,
                self.q_arr,
                mu,
                self.slab,
                homogeneous_epsilon=self.coeffs.homogeneous_epsilon,
                homogeneous_kappa=self.coeffs.homogeneous_kappa,
                use_intrinsic_potential=self.el_use_phi0,
                use_dipole_E0_component=self.el_use_dipole,
                r_smooth=self.el_r_smooth,
                coulomb_scale=self.el_coulomb_scale,
                pair_cutoff_angstrom=self.el_pair_cutoff,
            )

        h_film = film_energy_from_arrays(
            cents[:, 2],
            self.areas,
            self.softness,
            self.slab,
            lambda_film=self.coeffs.lambda_film,
            margin_angstrom=self.film_margin,
        )

        h_flex = float(self.h_flex)
        h_tot = self.h_solv + hp_pol + hel + h_film + h_flex

        if energy_only:
            return float(h_tot)

        res = HybridHamiltonianResult(
            H_total=float(h_tot),
            H_solv=float(self.h_solv),
            H_hp_pol_hbond=float(hp_pol),
            H_el=float(hel),
            H_film=float(h_film),
            H_flex=float(h_flex),
            solv_terms=self.solv_terms,
            hydration_terms=hydration_terms,
            electrostatic_terms=electrostatic_terms,
            i_air=i_air,
            i_int=i_int,
        )
        self._last_result = res
        return res

    def energy(self, R: np.ndarray, t: np.ndarray | None = None) -> float:
        out = self.evaluate(R, t, energy_only=True)
        assert isinstance(out, float)
        return out

    def energies_batch(self, R_batch: np.ndarray) -> np.ndarray:
        """
        Vectorized energies for B rotations (B, 3, 3) — amortizes setup overhead.

        Interior masks are computed in a batch; hydration rays run per pose in the
        batch (still vectorized over patches).
        """
        R_batch = np.asarray(R_batch, dtype=np.float64).reshape(-1, 3, 3)
        B = R_batch.shape[0]
        if B == 0:
            return np.zeros(0, dtype=np.float64)
        if B == 1:
            return np.array([self.energy(R_batch[0])], dtype=np.float64)

        interiors = lab_interior_masks_batch(
            self.canon_cache,
            self.lattice,
            R_batch,
            self.t_base,
            self.lab_xyz_flat,
        )
        tv = self.t_base.reshape(1, 3)
        cents = np.einsum("ij,bjk->bik", self.cents_can, R_batch) + tv
        norms = np.einsum("ij,bjk->bik", self.norms_can, R_batch)
        norms /= np.linalg.norm(norms, axis=2, keepdims=True) + 1e-15

        hp = self.coeffs.lambda_hp
        lp = self.coeffs.lambda_p
        lb = self.coeffs.lambda_hb
        lf = self.coeffs.lambda_film
        base = self.h_solv + self.h_flex + self.h_el_pair_const

        out = np.empty(B, dtype=np.float64)
        for b in range(B):
            i_air = compute_patch_air_indicators_batched(
                cents[b],
                norms[b],
                self.occupancy,
                self.lattice,
                interiors[b],
                max_steps=self.ray_max_steps,
                ray_step=self.ray_step,
                require_first_hit_solvent_steps=self.require_first_solvent,
                mode=self.occupancy_mode,
            )
            hp_pol = (
                -hp * np.dot(self.areas * self.hydr, i_air)
                + lp * np.dot(self.areas * self.pola, i_air)
            )
            z = cents[b, :, 2]
            mu_b = self.mu_can @ R_batch[b].T
            h_int, _ = intrinsic_electrostatic_energy(
                z,
                self.q_arr,
                mu_b,
                self.slab_lut,
                use_phi0=self.el_use_phi0,
                use_dipole=self.el_use_dipole,
            )
            h_film = film_energy_from_arrays(
                z, self.areas, self.softness, self.slab, lambda_film=lf, margin_angstrom=self.film_margin
            )
            out[b] = base + hp_pol + h_int + h_film
        return out
