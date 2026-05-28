r"""
Hybrid lattice-gas / electrostatics Hamiltonian for protein at the AWI (cryo-EM).

Combines:

* Lattice-gas / **Ising** solvent :math:`n_i\in\{0,1\}` (binary) or explicit
  three-state occupancy (air / interfacial water / bulk-like water) when grid
  spacing ≳ 3 Å.
* Optional **PB-like** pairwise electrostatics from patch charges and dipoles
  in a screened Coulomb kernel (placeholder for full heterogeneous Green's
  function / PB).
* **Hydrophobic / polar / H-bond** couplings from :class:`~anisotropy.patches.PatchFeatures`
  and outward-ray indicators.
* **Film / interface ageing** via :mod:`awi_field` coverage and mechanical state.
* Optional **flex** penalty on patch softness.

Uses :math:`P(\Omega)\propto \int d\mathbf R\, Z(\Omega,\mathbf{R})` with
:math:`Z=\sum_{\{n\}}\exp(-\beta H[n;\Omega,\mathbf{R}])`; MC / kinetics are downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal

import numpy as np

from anisotropy.awi_field import (
    DEFAULT_ANISOTROPY_DECAY_ANGSTROM,
    SurfaceCoverage,
    VitrifiedWaterSlab,
)
from anisotropy.mesh import ProteinMesh
from anisotropy.patches import COULOMB_SCALE, PatchFeatures, PatchParameterization

# -----------------------------------------------------------------------------
# Lattice and occupancy conventions
# -----------------------------------------------------------------------------


class TernaryOccupancy(IntEnum):
    AIR = 0
    INTERFACIAL_WATER = 1
    BULK_LIKE_WATER = 2


@dataclass
class CartesianLattice:
    """
    Uniform grid covering a cuboid in lab Å.

    Sites are voxel **centers**:

        xyz(i,j,k) = origin + (i+0.5, j+0.5, k+0.5) * spacing
    """

    origin: np.ndarray  # (3,)
    spacing: float
    shape: tuple[int, int, int]  # (nx, ny, nz)

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(3)
        if self.spacing <= 0:
            raise ValueError("spacing must be positive")
        if any(s < 1 for s in self.shape):
            raise ValueError("shape must be at least 1 in each axis")

    @property
    def n_sites(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz

    def grid_centers_xyz(self) -> np.ndarray:
        """(nx, ny, nz, 3) lattice site positions Å."""
        nx, ny, nz = self.shape
        h = float(self.spacing)
        o = self.origin.reshape(3)
        ix = (np.arange(nx) + 0.5) * h + o[0]
        iy = (np.arange(ny) + 0.5) * h + o[1]
        iz = (np.arange(nz) + 0.5) * h + o[2]
        gx, gy, gz = np.meshgrid(ix, iy, iz, indexing="ij")
        return np.stack([gx, gy, gz], axis=-1)


def lattice_indices_from_xyz(
    xyz: np.ndarray,
    origin: np.ndarray,
    spacing: float,
    shape: tuple[int, int, int],
    *,
    clip: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Continuous indices (i_float, ...) for CartesianLattice.

    Returns
    -------
    idx : ndarray (..., 3) int floor indices onto grid axes
    inside : ndarray (...,) bool (True if point lies strictly inside voxel range)
    """
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    rel = (np.asarray(xyz, dtype=np.float64) - o) / float(spacing) - 0.5
    idx = np.floor(rel).astype(int)
    inside = np.ones(idx.shape[:-1], dtype=bool)
    nx, ny, nz = shape
    for axis, ni in enumerate((nx, ny, nz)):
        inside &= idx[..., axis] >= 0
        inside &= idx[..., axis] <= ni - 1
        if clip:
            idx[..., axis] = np.clip(idx[..., axis], 0, ni - 1)
    return idx, inside


# -----------------------------------------------------------------------------
# Voxelizing the mesh (PyVista enclosed points)
# -----------------------------------------------------------------------------

try:
    import pyvista as pv

    _HAS_PYVISTA = True
except ImportError:  # pragma: no cover
    _HAS_PYVISTA = False


def rigid_transform_points(xyz: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Rotate then translate columns of ``xyz`` (N,3)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    return (xyz @ R.T) + t.reshape(1, 3)


def mesh_to_pv_surface(mesh: ProteinMesh) -> "pv.PolyData":
    if not _HAS_PYVISTA:
        raise ImportError("mesh voxelization requires pyvista (pip install pyvista)")
    v = mesh.vertices.astype(np.float64)
    f = mesh.faces.astype(np.int64)
    nf = int(f.shape[0])
    cells = np.hstack([np.full((nf, 1), 3, dtype=np.int64), f])
    surf = pv.PolyData(v, cells)
    surf = surf.triangulate()
    surf.clean(inplace=True)
    return surf


def rigid_patch_parameterization(
    param: PatchParameterization,
    R: np.ndarray,
    t: np.ndarray,
) -> PatchParameterization:
    """Apply rigid body \\(R \\mathbf{r} + \\mathbf{t}\\) to patch geometry (lab frame pose)."""
    Rm = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tv = np.asarray(t, dtype=np.float64).reshape(3)
    base_c = np.stack([np.asarray(p.centroid, dtype=np.float64) for p in param.patches], axis=0)
    base_n = np.stack([np.asarray(p.normal, dtype=np.float64) for p in param.patches], axis=0)
    base_mu = np.stack([np.asarray(p.dipole, dtype=np.float64) for p in param.patches], axis=0)

    cen_m = base_c @ Rm.T + tv.reshape(1, 3)
    nh_m = base_n @ Rm.T
    nrm = np.linalg.norm(nh_m, axis=1, keepdims=True)
    nh_m = nh_m / (nrm + 1e-15)
    mu_m = base_mu @ Rm.T

    new_list: list[PatchFeatures] = []
    for i, p in enumerate(param.patches):
        new_list.append(
            PatchFeatures(
                patch_id=p.patch_id,
                area=p.area,
                centroid=cen_m[i],
                normal=nh_m[i],
                mean_curvature=p.mean_curvature,
                gaussian_curvature=p.gaussian_curvature,
                charge=p.charge,
                potential=p.potential,
                pka_acid=p.pka_acid,
                hydropathy=p.hydropathy,
                polar_density=p.polar_density,
                hbond_score=p.hbond_score,
                dipole=mu_m[i],
                softness=p.softness,
                face_indices=p.face_indices,
                n_atoms=p.n_atoms,
            )
        )
    return PatchParameterization(
        patches=new_list,
        face_patch_ids=param.face_patch_ids.copy(),
        ph=param.ph,
        metadata=dict(param.metadata),
    )


def voxelize_protein_interior(
    mesh: ProteinMesh,
    lattice: CartesianLattice,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Boolean mask interior\\{protein}\\ (excluded volume) on lattice sites.

    Uses ``surface.select_enclosed_points`` — surface must enclose cleanly;
    leaky SAS meshes may need repair.
    """
    surf_orig = mesh_to_pv_surface(mesh)
    v_tf = rigid_transform_points(surf_orig.points, R, t)
    surf_tf = surf_orig.copy()
    surf_tf.points = v_tf
    grid_xyz = lattice.grid_centers_xyz().reshape(-1, 3)

    pts = pv.PolyData(grid_xyz)
    if hasattr(pts, "select_interior_points"):
        sel = pts.select_interior_points(surf_tf)
        pdata = sel.point_data.get("selected_points", sel.point_data.get("SelectedPoints"))
        if pdata is None:
            raise RuntimeError("pyvista interior selection missing expected arrays")
        inside = np.asarray(pdata, dtype=bool)
    else:
        sel = pts.select_enclosed_points(
            surf_tf,
            tolerance=1e-5,
            check_surface=False,
        )
        inside = sel["SelectedPoints"].astype(bool)

    return inside.reshape(lattice.shape)


@dataclass(frozen=True)
class CanonicalInteriorCache:
    r"""
    Boolean interior mask **in the canonical mesh frame**
    (identity \(R=\mathbb I,\;t=\mathbf 0\)).

    For laboratory pose \(\mathbf{x}_{lab} = \mathbf{R}\mathbf{x}_{can}+\mathbf{t}\),

    \(\mathbf{x}_{can}=\mathbf{R}^{\mathsf T}(\mathbf{x}_{lab}-\mathbf{t})\),

    equivalently rows:

    \(\mathbf{r}_{can}=(\mathbf{r}_{lab}-\mathbf{t}^{\mathsf T})\mathbf{R}\).
    """

    inside_can: np.ndarray
    can_lattice: CartesianLattice

    @classmethod
    def build(cls, mesh: ProteinMesh, *, spacing: float, pad_angstrom: float) -> CanonicalInteriorCache:
        v = mesh.vertices.astype(np.float64)
        bb_min = v.min(axis=0)
        bb_max = v.max(axis=0)
        pad = float(pad_angstrom)
        origin = bb_min - pad
        extent = bb_max - bb_min + 2.0 * pad
        h = float(spacing)
        shape = tuple(np.maximum(np.ceil(extent / h), 1.0).astype(int).tolist())
        can_lat = CartesianLattice(origin=origin, spacing=h, shape=shape)  # type: ignore[arg-type]
        eye = np.eye(3, dtype=np.float64)
        zero = np.zeros(3, dtype=np.float64)
        inside = voxelize_protein_interior(mesh, can_lat, eye, zero)
        return cls(inside_can=inside, can_lattice=can_lat)

    def lab_interior_mask(
        self,
        lab_lattice: CartesianLattice,
        R: np.ndarray,
        t: np.ndarray,
        *,
        lab_xyz_flat: np.ndarray | None = None,
    ) -> np.ndarray:
        """Interior mask matching ``lab_lattice.shape`` for pose ``(R, t)`` (no PyVista)."""
        Rm = np.asarray(R, dtype=np.float64).reshape(3, 3)
        tv = np.asarray(t, dtype=np.float64).reshape(3)
        if lab_xyz_flat is None:
            lab_xyz_flat = lab_lattice.grid_centers_xyz().reshape(-1, 3)
        X = np.asarray(lab_xyz_flat, dtype=np.float64).reshape(-1, 3)
        x_can = (X - tv) @ Rm

        o = self.can_lattice.origin.reshape(3)
        h = float(self.can_lattice.spacing)
        nx_c, ny_c, nz_c = self.can_lattice.shape
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
        out = np.zeros(X.shape[0], dtype=bool)
        ii = ijk[in_b, 0]
        jj = ijk[in_b, 1]
        kk = ijk[in_b, 2]
        out[in_b] = self.inside_can[ii, jj, kk]
        return out.reshape(lab_lattice.shape)


# -----------------------------------------------------------------------------
# Solvent Hamiltonian pieces
# -----------------------------------------------------------------------------


def _six_neighbor_bond_sum_binary(n: np.ndarray) -> float:
    """Sum over unique nearest-neighbor bonds of n_i n_j (6-connectivity)."""
    nx, ny, nz = n.shape
    acc = 0.0
    if nx > 1:
        acc += float(np.sum(n[:-1, :, :] * n[1:, :, :]))
    if ny > 1:
        acc += float(np.sum(n[:, :-1, :] * n[:, 1:, :]))
    if nz > 1:
        acc += float(np.sum(n[:, :, :-1] * n[:, :, 1:]))
    return acc


def effective_binary_water(occ: np.ndarray, mode: Literal["binary", "ternary"]) -> np.ndarray:
    """
    Map occupancy to water-like variable \\(\\eta_i\\) for cohesion.

    Binary: occ is {0,1}; ternary: air=0, interfacial and bulk = 1.
    """
    if mode == "binary":
        return occ.astype(np.float64)
    return (occ >= TernaryOccupancy.INTERFACIAL_WATER).astype(np.float64)


def film_potential_per_cell(
    z_lab: np.ndarray,
    z_slab_lo: float,
    z_slab_hi: float,
    *,
    penalty_outside: float = 1.0,
    interface_softness: float = 4.0,
) -> np.ndarray:
    """
    Confining field \\(U_{\\mathrm{film}}(z)\\) *per site* (dimensionless scale).

    Penalizes water **outside** the finite slab ``[z_slab_lo, z_slab_hi]`` quadratically.

    Multiply by occupancy and a user energy scale before adding to \\(H\\).
    """
    z = np.asarray(z_lab, dtype=np.float64)
    below = np.maximum(z_slab_lo - z, 0.0)
    above = np.maximum(z - z_slab_hi, 0.0)
    band = np.maximum(below, above)
    return penalty_outside * (band ** 2 / (interface_softness ** 2 + band ** 2 + 1e-12))


def solvation_energy_lattice_gas(
    occ: np.ndarray,
    lattice: CartesianLattice,
    *,
    mode: Literal["binary", "ternary"] = "binary",
    J: float = 1.0,
    mu_chemical: float = 0.0,
    u_film: np.ndarray | None = None,
    u_film_scale: float = 0.2,
    slab_z_bounds: tuple[float, float] | None = None,
) -> tuple[float, dict]:
    """
    Lattice-gas analogue of \\(H_{\\mathrm{solv}}\\).

    \\(H \\supseteq - J \\sum_{\\langle i,j\\rangle} \\eta_i \\eta_j
         - \\mu \\sum_i \\eta_i + \\cdots\\).

    Uses ``effective_binary_water`` for ``\\eta`` under ternary mode.
    Chemical potential subtracts favours water vs air when \\( \\mu \\) is positive.

    Optionally adds ``sum_i U_{film}(z_i) \\times \\eta_i`` from ``slab_z_bounds``
    if ``u_film`` omitted.
    """
    eta = effective_binary_water(occ, mode)
    bond = _six_neighbor_bond_sum_binary(eta)
    h_nn = -J * bond

    eta_flat = eta.ravel()
    h_mu = -mu_chemical * float(np.sum(eta_flat))

    if u_film is None:
        if slab_z_bounds is not None:
            z_lab = lattice.grid_centers_xyz()[..., 2].ravel()
            u_vec = film_potential_per_cell(
                z_lab,
                slab_z_bounds[0],
                slab_z_bounds[1],
            )
            u_vec = u_vec.reshape(lattice.shape)
            h_fil = float(u_film_scale * np.sum(u_vec.ravel() * eta.ravel()))
        else:
            h_fil = 0.0
            u_vec = None
    else:
        u_arr = np.asarray(u_film, dtype=np.float64)
        if u_arr.shape != lattice.shape:
            raise ValueError("u_film shape must match lattice.shape")
        h_fil = float(u_film_scale * np.sum(u_arr.ravel() * eta.ravel()))
        u_vec = u_arr

    h = h_nn + h_mu + h_fil
    return h, {
        "H_nn": h_nn,
        "H_mu": h_mu,
        "H_film_field": h_fil,
        "bond_sum_eta_eta": bond,
        "eta_sum": float(np.sum(eta)),
    }


# -----------------------------------------------------------------------------
# Patch–solvent indicators (air vs interfacial water)
# -----------------------------------------------------------------------------


def _occ_is_air(
    occupancy: np.ndarray,
    lattice_idx: tuple[int, int, int],
    mode: Literal["binary", "ternary"],
) -> bool:
    i, j, k = lattice_idx
    v = int(occupancy[i, j, k])
    if mode == "binary":
        return v == 0
    return v == int(TernaryOccupancy.AIR)


def _occ_is_interfacial_water(
    occupancy: np.ndarray,
    lattice_idx: tuple[int, int, int],
    mode: Literal["binary", "ternary"],
) -> bool:
    if mode != "ternary":
        return False
    i, j, k = lattice_idx
    return int(occupancy[i, j, k]) == int(TernaryOccupancy.INTERFACIAL_WATER)


def outward_solvent_ray_sample(
    centroid: np.ndarray,
    outward_normal: np.ndarray,
    lattice: CartesianLattice,
    protein_interior: np.ndarray,
    *,
    max_steps: int = 32,
    fractional_step: float = 0.5,
) -> list[tuple[int, int, int]]:
    """
    March from patch centroid along ``outward_normal`` returning solvent voxel indices.

    Skips lattice sites flagged as protein interior.
    """
    h = float(lattice.spacing)
    n_hat = np.asarray(outward_normal, dtype=np.float64).reshape(3)
    rn = np.linalg.norm(n_hat)
    if rn < 1e-12:
        return []
    n_hat = n_hat / rn
    c = np.asarray(centroid, dtype=np.float64).reshape(3)
    coords: list[tuple[int, int, int]] = []
    nx, ny, nz = lattice.shape
    o = lattice.origin.reshape(3)
    steps = fractional_step * h * np.arange(1, max_steps + 1, dtype=np.float64)
    pts = c.reshape(1, 3) + steps.reshape(-1, 1) * n_hat.reshape(1, 3)
    rel = (pts - o) / h - 0.5
    ijf = np.floor(rel).astype(np.int64)

    valid = (
        (ijf[:, 0] >= 0)
        & (ijf[:, 0] < nx)
        & (ijf[:, 1] >= 0)
        & (ijf[:, 1] < ny)
        & (ijf[:, 2] >= 0)
        & (ijf[:, 2] < nz)
    )
    for s in range(len(steps)):
        if not valid[s]:
            break
        i, j, k = int(ijf[s, 0]), int(ijf[s, 1]), int(ijf[s, 2])
        if protein_interior[i, j, k]:
            continue
        coords.append((i, j, k))
        if len(coords) >= max_steps:
            break
    return coords


def patch_air_indicator(
    p: PatchFeatures,
    occupancy: np.ndarray,
    lattice: CartesianLattice,
    protein_interior: np.ndarray,
    *,
    mode: Literal["binary", "ternary"] = "binary",
    ray_max_steps: int = 48,
    ray_step: float = 0.55,
    require_first_hit_solvent_steps: int = 2,
) -> float:
    """
    \\mathcal{I}_f^{\\mathrm{air}}: 1 if outward ray encounters an air-like site.
    """
    solvox = outward_solvent_ray_sample(
        p.centroid,
        p.normal,
        lattice,
        protein_interior,
        max_steps=ray_max_steps,
        fractional_step=ray_step,
    )
    if len(solvox) < max(1, require_first_hit_solvent_steps):
        return 0.0
    for idx in solvox:
        if _occ_is_air(occupancy, idx, mode):
            return 1.0
    return 0.0


def patch_interfacial_water_indicator(
    p: PatchFeatures,
    occupancy: np.ndarray,
    lattice: CartesianLattice,
    protein_interior: np.ndarray,
    *,
    ray_max_steps: int = 48,
    ray_step: float = 0.55,
) -> float:
    """
    \\mathcal{I}_f^{\\mathrm{int}} — requires ``ternary`` occupancy.

    Binary mode always returns 0 (use finer grid or three-state solvent).
    """
    solvox = outward_solvent_ray_sample(
        p.centroid,
        p.normal,
        lattice,
        protein_interior,
        max_steps=ray_max_steps,
        fractional_step=ray_step,
    )
    for idx in solvox:
        if _occ_is_interfacial_water(occupancy, idx, "ternary"):
            return 1.0
    return 0.0


def compute_patch_indicators_for_parameterization(
    param: PatchParameterization,
    occupancy: np.ndarray,
    lattice: CartesianLattice,
    protein_interior: np.ndarray,
    *,
    mode: Literal["binary", "ternary"] = "binary",
    ray_max_steps: int = 48,
    ray_step: float | None = None,
    ray_step_fraction: float = 0.55,
    require_first_hit_solvent_steps: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return vectors (I_air_f, I_int_f) shaped (n_patches,)."""
    if ray_step is None:
        ray_step = ray_step_fraction * float(lattice.spacing)
    i_air = np.zeros(param.n_patches, dtype=np.float64)
    i_int = np.zeros(param.n_patches, dtype=np.float64)
    for p in param.patches:
        pid = p.patch_id
        i_air[pid] = patch_air_indicator(
            p,
            occupancy,
            lattice,
            protein_interior,
            mode=mode,
            ray_max_steps=ray_max_steps,
            ray_step=ray_step,
            require_first_hit_solvent_steps=require_first_hit_solvent_steps,
        )
        i_int[pid] = (
            patch_interfacial_water_indicator(
                p,
                occupancy,
                lattice,
                protein_interior,
                ray_max_steps=ray_max_steps,
                ray_step=ray_step,
            )
            if mode == "ternary"
            else 0.0
        )
    return i_air, i_int


# -----------------------------------------------------------------------------
# Hydrophobic, polar/H-bond, film, flex
# -----------------------------------------------------------------------------


def hydration_coupling_hp_pol_hbond(
    param: PatchParameterization,
    i_air: np.ndarray,
    i_int: np.ndarray,
    *,
    lambda_hp: float = 1.0,
    lambda_p: float = 0.25,
    lambda_hb: float = 0.25,
) -> tuple[float, dict]:
    """
    \\(H_{\\mathrm{hp}}\\) + \\(H_{\\mathrm{pol}}\\) with signs from the prompt.

        -\\lambda_h \\sum_f a_f h_f I^{\\mathrm{air}}_f
        +\\lambda_p \\sum_f a_f p_f I^{\\mathrm{air}}_f
        -\\lambda_{HB} \\sum_f a_f b_f I^{\\mathrm{int}}_f.
    """
    ids = np.array([p.patch_id for p in param.patches], dtype=np.int64)
    areas = np.array([float(p.area) for p in param.patches])
    hydr = np.array([float(p.hydropathy) for p in param.patches])
    pola = np.array([float(p.polar_density) for p in param.patches])
    hbon = np.array([float(p.hbond_score) for p in param.patches])
    ia = i_air[ids]
    ii = i_int[ids]

    h_hp = -float(lambda_hp * np.dot(areas * hydr, ia))
    h_pol = float(lambda_p * np.dot(areas * pola, ia))
    h_hb = -float(lambda_hb * np.dot(areas * hbon, ii))

    h_tot = h_hp + h_pol + h_hb
    return h_tot, {"H_hp": h_hp, "H_pol": h_pol, "H_hbond_channel": h_hb}


_FILM_SURFACE_COVERS: frozenset[SurfaceCoverage] = frozenset(
    {
        SurfaceCoverage.PROTEIN_SACRIFICIAL,
        SurfaceCoverage.SURFACTANT,
        SurfaceCoverage.CONTAMINATED,
    }
)


def patch_film_indicator(
    p: PatchFeatures,
    slab: VitrifiedWaterSlab,
    *,
    margin_angstrom: float = DEFAULT_ANISOTROPY_DECAY_ANGSTROM,
) -> float:
    """
    \\(I^{\\mathrm{film}}(\\Omega,\\mathbf R;c,m)\\): patch lies near a non-pristine
    slab interface (coverage + mechanical age).
    """
    z = float(np.asarray(p.centroid).reshape(3)[2])
    dz_bot = float(slab.depth_from_bottom(z)[0])
    dz_top = float(slab.depth_from_top(z)[0])

    iw = np.exp(-(dz_bot**2) / (margin_angstrom**2))
    ww = iw if slab.bottom.coverage in _FILM_SURFACE_COVERS else 0.0

    it = np.exp(-(dz_top**2) / (margin_angstrom**2))
    wt = it if slab.top.coverage in _FILM_SURFACE_COVERS else 0.0

    mw = ww * float(slab.bottom.mechanical_age)
    mt = wt * float(slab.top.mechanical_age)

    factor = ww + wt + mw + mt
    return float(np.clip(factor / 4.0, 0.0, 1.0))


def film_coupling_energy(
    param: PatchParameterization,
    slab: VitrifiedWaterSlab,
    *,
    lambda_film: float = 1.0,
    margin_angstrom: float = DEFAULT_ANISOTROPY_DECAY_ANGSTROM,
) -> tuple[float, np.ndarray]:
    """
    \\(\\sum_f a_f \\eta_f I^{\\mathrm{film}}_f\\) with \\(\\eta_f = u_f\\) (softness).
    """
    z_arr = np.array([float(np.asarray(p.centroid).reshape(3)[2]) for p in param.patches], dtype=np.float64)
    dz_bot = np.clip(z_arr, 0.0, float(slab.thickness_angstrom))
    dz_top = np.clip(float(slab.thickness_angstrom) - z_arr, 0.0, float(slab.thickness_angstrom))

    ww = np.exp(-(dz_bot**2) / (margin_angstrom**2)) * float(slab.bottom.coverage in _FILM_SURFACE_COVERS)
    wt = np.exp(-(dz_top**2) / (margin_angstrom**2)) * float(slab.top.coverage in _FILM_SURFACE_COVERS)
    mw = ww * float(slab.bottom.mechanical_age)
    mt = wt * float(slab.top.mechanical_age)
    weights_arr = np.clip((ww + wt + mw + mt) / 4.0, 0.0, 1.0)

    areas = np.array([float(p.area) for p in param.patches])
    softness = np.array([float(p.softness) for p in param.patches])
    contrib = float(lambda_film) * areas * softness * weights_arr
    return float(np.sum(contrib)), weights_arr


def coupling_flex_penalty(
    param: PatchParameterization,
    *,
    eta_flex: float = 0.05,
    reference_softness_mean: float | None = None,
) -> tuple[float, np.ndarray]:
    """Optional \\(H_{\\mathrm{flex}}\\) vs mean patch softness."""
    u_vals = np.array([p.softness for p in param.patches], dtype=np.float64)
    ref = float(np.mean(u_vals)) if reference_softness_mean is None else reference_softness_mean
    dif = u_vals - ref
    energies = eta_flex * (dif**2)
    return float(np.sum(energies)), energies


# -----------------------------------------------------------------------------
# Nonlocal electrostatics (medium + screened Coulomb — PB kernel placeholder)
# -----------------------------------------------------------------------------


def _slab_electrostatics_at_xyz(
    r: np.ndarray,
    slab: VitrifiedWaterSlab,
) -> tuple[float, float, float, np.ndarray]:
    """Return (ε_iso, κ (1/Å), φ₀ Volts, E₀ vector with slab normal along z)."""
    z = float(np.asarray(r).reshape(3)[2])
    samp = slab.sample_fields(z, blend_interfaces=True)
    eps = 0.5 * (
        float(samp["epsilon_parallel"][0]) + float(samp["epsilon_perpendicular"][0])
    )
    kap = float(samp["kappa"][0])
    phi0 = float(samp["phi_0"][0])
    ez = float(samp["E_0"][0])
    evec = np.array([0.0, 0.0, ez], dtype=np.float64)
    return eps, kap, phi0, evec


def green_yukawa_coulomb(
    r_ij: np.ndarray,
    *,
    epsilon: float,
    kappa: float,
    r_smooth: float = 1e-2,
) -> float:
    """
    Heuristic \\(G_{\\epsilon,\\kappa}(\\mathbf r)\\): screened Coulomb.

        G ≈ (k / ε r) exp(-κ r)

    with ``k = COULOMB_SCALE`` (same coarse units as :mod:`patches`).

    **Not** a heterogeneous Green's function — supply better kernels from FEM/PB later.
    """
    r_mag = float(max(np.linalg.norm(np.asarray(r_ij, dtype=np.float64)), r_smooth))
    eps = float(max(epsilon, 1.0))
    kappa_eff = float(max(kappa, 0.0))
    yukawa = float(np.exp(-kappa_eff * r_mag))
    pref = COULOMB_SCALE / eps
    return float(pref * yukawa / r_mag)


def electrostatic_energy_pb_like(
    param: PatchParameterization,
    slab: VitrifiedWaterSlab,
    *,
    homogeneous_epsilon: float | None = None,
    homogeneous_kappa: float | None = None,
    use_intrinsic_potential: bool = True,
    use_dipole_E0_component: bool = True,
    r_smooth: float = 1e-2,
    coulomb_scale: float | None = None,
) -> tuple[float, dict]:
    """
    \\(H_{\\mathrm{el}} \\approx \\sum_{\\alpha<\\beta} q_\\alpha G q_\\beta
       + \\sum q_\\alpha \\phi_0 + \\sum \\boldsymbol{\\mu}_\\alpha\\cdot\\mathbf{E}_0\\).

    Pairwise \\(G\\) uses mid-point \\((\\varepsilon, \\kappa)\\) from the slab unless
    ``homogeneous_*`` overrides are set. Vectorised over patches.
    """
    patches = param.patches
    n_p = len(patches)
    if n_p == 0:
        return 0.0, {
            "H_pair_screened": 0.0,
            "sum_q_phi0": 0.0,
            "dipole_dot_E0": 0.0,
        }

    cents = np.stack([np.asarray(p.centroid, dtype=np.float64) for p in patches], axis=0)
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

    q_arr = np.array([float(p.charge) for p in patches], dtype=np.float64)
    mu_mat = np.stack([np.asarray(p.dipole, dtype=np.float64) for p in patches], axis=0)
    k_coul = float(coulomb_scale) if coulomb_scale is not None else COULOMB_SCALE

    h_pair = 0.0
    if n_p > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        r_mag = np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))
        eps_mid = 0.5 * (eps[:, np.newaxis] + eps[np.newaxis, :])
        kap_mid = 0.5 * (kap[:, np.newaxis] + kap[np.newaxis, :])
        rm = np.maximum(r_mag, float(r_smooth))
        eps_eff = np.maximum(eps_mid, 1.0)
        kap_eff = np.maximum(kap_mid, 0.0)
        g_mat = k_coul * np.exp(-kap_eff * rm) / (eps_eff * rm)
        qa = q_arr[:, np.newaxis]
        qb = q_arr[np.newaxis, :]
        up = qa * qb * g_mat
        ii, jj = np.triu_indices(n_p, k=1)
        h_pair = float(np.sum(up[ii, jj]))

    sqp = float(np.dot(q_arr, phi0)) if use_intrinsic_potential else 0.0
    dipole_term = (
        float(np.sum(mu_mat[:, 2] * ez_arr)) if use_dipole_E0_component else 0.0
    )
    h_intrinsic = sqp + dipole_term

    diag = {
        "H_pair_screened": float(h_pair),
        "sum_q_phi0": float(sqp),
        "dipole_dot_E0": float(dipole_term),
    }
    return float(h_pair + h_intrinsic), diag


# -----------------------------------------------------------------------------
# Full hybrid Hamiltonian aggregator
# -----------------------------------------------------------------------------


@dataclass
class HybridHamiltonianCouplings:
    """Dimensionless-ish scales tying literature terms to lattice energy."""

    J_solv: float = 1.0
    mu_chemical: float = 0.0
    u_film_scale: float = 0.25
    lambda_hp: float = 1.0
    lambda_p: float = 0.25
    lambda_hb: float = 0.25
    lambda_film: float = 1.0
    eta_flex: float = 0.0
    homogeneous_epsilon: float | None = None
    homogeneous_kappa: float | None = None

    @classmethod
    def from_ising_params(cls, path: str | None = None) -> HybridHamiltonianCouplings:
        """Load couplings from ``ising_params.yaml`` (see :mod:`anisotropy.ising_params`)."""
        from anisotropy.ising_params import load_ising_params

        return load_ising_params(path).to_hybrid_couplings()


def precompute_solvation_energy(
    occupancy: np.ndarray,
    lattice: CartesianLattice,
    slab: VitrifiedWaterSlab,
    coeffs: HybridHamiltonianCouplings,
    *,
    occupancy_mode: Literal["binary", "ternary"] = "binary",
    confinement_penalty_outside: float = 1.0,
    confinement_interface_softness: float = 4.0,
) -> tuple[float, dict]:
    r"""Compute \(H_{\mathrm{solv}}\) once when occupancy does not vary with pose."""
    slab_z_bounds = (0.0, float(slab.thickness_angstrom))
    z_lab = lattice.grid_centers_xyz()[..., 2].ravel()
    u_vec = film_potential_per_cell(
        z_lab,
        slab_z_bounds[0],
        slab_z_bounds[1],
        penalty_outside=confinement_penalty_outside,
        interface_softness=confinement_interface_softness,
    ).reshape(lattice.shape)
    return solvation_energy_lattice_gas(
        occupancy,
        lattice,
        mode=occupancy_mode,
        J=coeffs.J_solv,
        mu_chemical=coeffs.mu_chemical,
        u_film_scale=coeffs.u_film_scale,
        u_film=u_vec,
    )


@dataclass
class HybridHamiltonianResult:
    """Decomposed \\(H[n;\\Omega,\\mathbf{R}]\\) for one occupancy state."""

    H_total: float
    H_solv: float
    H_hp_pol_hbond: float
    H_el: float
    H_film: float
    H_flex: float
    solv_terms: dict
    hydration_terms: dict
    electrostatic_terms: dict
    i_air: np.ndarray
    i_int: np.ndarray


def evaluate_hybrid_hamiltonian(
    occupancy: np.ndarray,
    lattice: CartesianLattice,
    mesh: ProteinMesh,
    pose_R: np.ndarray,
    pose_t: np.ndarray,
    param: PatchParameterization,
    slab: VitrifiedWaterSlab,
    coeffs: HybridHamiltonianCouplings,
    *,
    occupancy_mode: Literal["binary", "ternary"] = "binary",
    canonical_interior: CanonicalInteriorCache | None = None,
    lab_xyz_flat: np.ndarray | None = None,
    h_solv_and_terms: tuple[float, dict] | None = None,
    use_slow_pyvista_voxel: bool = False,
    ising_params: Any | None = None,
) -> HybridHamiltonianResult:
    """
    Evaluate \\(H = H_{\\mathrm{solv}} + H_{\\mathrm{hp/pol/HB}}
    + H_{\\mathrm{el}} + H_{\\mathrm{film}} (+ H_{\\mathrm{flex}})\\).

    Optimisations (cryo‑EM orientation scans):
    - ``h_solv_and_terms``: reuse fixed‑occupancy solvation energy across poses.
    - ``canonical_interior``: avoid PyVista per pose (map canonical mask to lab grid).
    - ``lab_xyz_flat``: optional cached voxel centers ``(nx*ny*nz, 3)``.

    Legacy slow path (debug only): ``use_slow_pyvista_voxel=True`` runs PyVista every pose.
    """
    param_pose = rigid_patch_parameterization(param, pose_R, pose_t)
    if occupancy.shape != lattice.shape:
        raise ValueError("occupancy shape mismatch")

    if lab_xyz_flat is None:
        lab_xyz_flat_arr = lattice.grid_centers_xyz().reshape(-1, 3).astype(np.float64, copy=False)
    else:
        lab_xyz_flat_arr = lab_xyz_flat

    if use_slow_pyvista_voxel or canonical_interior is None:
        interior = voxelize_protein_interior(mesh, lattice, pose_R, pose_t)
    else:
        interior = canonical_interior.lab_interior_mask(
            lattice, pose_R, pose_t, lab_xyz_flat=lab_xyz_flat_arr
        )

    ray_max_steps = 48
    ray_step_fraction = 0.55
    require_first_solvent = 2
    film_margin = DEFAULT_ANISOTROPY_DECAY_ANGSTROM
    el_r_smooth = 1e-2
    el_use_phi0 = True
    el_use_dipole = True
    el_coulomb_scale: float | None = None
    if ising_params is not None:
        ray_max_steps = ising_params.ray.ray_max_steps
        ray_step_fraction = ising_params.ray.ray_step_fraction
        require_first_solvent = ising_params.ray.require_first_hit_solvent_steps
        film_margin = ising_params.film.margin_angstrom
        el_r_smooth = ising_params.electrostatics.r_smooth_angstrom
        el_use_phi0 = ising_params.electrostatics.use_intrinsic_potential
        el_use_dipole = ising_params.electrostatics.use_dipole_E0_component
        el_coulomb_scale = ising_params.electrostatics.coulomb_scale
        occupancy_mode = ising_params.solv.occupancy_mode  # type: ignore[assignment]

    if h_solv_and_terms is not None:
        h_sol, solv_terms = float(h_solv_and_terms[0]), dict(h_solv_and_terms[1])
    else:
        conf_pen = 1.0
        conf_soft = 4.0
        if ising_params is not None:
            conf_pen = ising_params.confinement.penalty_outside
            conf_soft = ising_params.confinement.interface_softness_angstrom
        h_sol, solv_terms = precompute_solvation_energy(
            occupancy,
            lattice,
            slab,
            coeffs,
            occupancy_mode=occupancy_mode,
            confinement_penalty_outside=conf_pen,
            confinement_interface_softness=conf_soft,
        )

    i_air, i_int = compute_patch_indicators_for_parameterization(
        param_pose,
        occupancy,
        lattice,
        interior,
        mode=occupancy_mode,
        ray_max_steps=ray_max_steps,
        ray_step_fraction=ray_step_fraction,
        require_first_hit_solvent_steps=require_first_solvent,
    )
    hp_pol, hydration_terms = hydration_coupling_hp_pol_hbond(
        param_pose,
        i_air,
        i_int,
        lambda_hp=coeffs.lambda_hp,
        lambda_p=coeffs.lambda_p,
        lambda_hb=coeffs.lambda_hb,
    )

    hel, electrostatic_terms = electrostatic_energy_pb_like(
        param_pose,
        slab,
        homogeneous_epsilon=coeffs.homogeneous_epsilon,
        homogeneous_kappa=coeffs.homogeneous_kappa,
        use_intrinsic_potential=el_use_phi0,
        use_dipole_E0_component=el_use_dipole,
        r_smooth=el_r_smooth,
        coulomb_scale=el_coulomb_scale,
    )

    h_film_, _film_w = film_coupling_energy(
        param_pose,
        slab,
        lambda_film=coeffs.lambda_film,
        margin_angstrom=film_margin,
    )

    flex_, _flex_e = coupling_flex_penalty(param_pose, eta_flex=coeffs.eta_flex)

    h_tot = h_sol + hp_pol + hel + h_film_ + flex_
    return HybridHamiltonianResult(
        H_total=float(h_tot),
        H_solv=float(h_sol),
        H_hp_pol_hbond=float(hp_pol),
        H_el=float(hel),
        H_film=float(h_film_),
        H_flex=float(flex_),
        solv_terms=solv_terms,
        hydration_terms=hydration_terms,
        electrostatic_terms=electrostatic_terms,
        i_air=i_air,
        i_int=i_int,
    )


def occupancy_binary_template(
    lattice: CartesianLattice,
    *,
    solvent_z_within: tuple[float, float],
    water_value: int = 1,
    air_value: int = 0,
    fill_water_in_band: bool = True,
) -> np.ndarray:
    """
    Helper: initialise a binary lattice with bulk water slab in ``solvent_z_within``.

    Does **not** carve protein — ``logical_and`` against ``~interior`` afterward.
    """
    lo, hi = solvent_z_within
    z = lattice.grid_centers_xyz()[..., 2]
    base = np.full(lattice.shape, air_value, dtype=np.int8)
    mask = (z >= lo) & (z <= hi)
    if fill_water_in_band:
        base[mask] = np.int8(water_value)
    else:
        base[~mask] = np.int8(water_value)
    return base