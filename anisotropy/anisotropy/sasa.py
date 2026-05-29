"""
Solvent-accessible surface (SAS) distance field and surface extraction.

The SAS boundary is the locus of points where a probe sphere of radius
``probe_radius`` (default 1.4 Å water) is tangent to the van der Waals envelope
of the protein. We represent it as the zero isosurface of

    Φ(x) = min_i ( ||x − c_i|| − (r_vdw,i + r_probe) ).

This is a standard algebraic SAS model; it is fast and grid-friendly for
iterative mesh refinement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from anisotropy.pdb import ProteinStructure

DEFAULT_PROBE_RADIUS = 1.4  # Å, water
DEFAULT_MAX_GRID_DIM = 256


@dataclass
class DistanceFieldGrid:
    """Sampled Φ on a regular 3D grid (negative inside the SAS envelope)."""

    values: np.ndarray  # (nx, ny, nz), float32
    origin: np.ndarray  # (3,) corner of voxel (0,0,0) in Å
    spacing: float  # voxel edge length Å

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.values.shape)

    def world_coordinates(self) -> np.ndarray:
        """(nx, ny, nz, 3) physical coordinates for each voxel center."""
        xs, ys, zs = _axis_coordinates(self.origin, self.spacing, self.values.shape)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.stack([xx, yy, zz], axis=-1)


def _axis_coordinates(
    origin: np.ndarray, spacing: float, shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1D voxel-center coordinates along each axis."""
    nx, ny, nz = (int(s) for s in shape)
    xs = origin[0] + np.arange(nx, dtype=np.float64) * spacing
    ys = origin[1] + np.arange(ny, dtype=np.float64) * spacing
    zs = origin[2] + np.arange(nz, dtype=np.float64) * spacing
    return xs, ys, zs


def _grid_from_bounds(
    bb_min: np.ndarray,
    bb_max: np.ndarray,
    *,
    resolution: float,
    max_grid_dim: int,
) -> tuple[np.ndarray, float, tuple[int, int, int]]:
    extent = bb_max - bb_min
    spacing = float(resolution)
    dims = np.ceil(extent / spacing).astype(int) + 1
    while dims.max() > max_grid_dim:
        spacing *= 1.15
        dims = np.ceil(extent / spacing).astype(int) + 1
    origin = bb_min.copy()
    return origin, spacing, (int(dims[0]), int(dims[1]), int(dims[2]))


def _atom_voxel_slices(
    center: np.ndarray,
    radius: float,
    origin: np.ndarray,
    spacing: float,
    dims: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    """Voxel index slices covering the sphere bounding box (inclusive)."""
    nx, ny, nz = dims
    lo = origin
    sp = spacing
    i0 = max(0, int(np.floor((center[0] - radius - lo[0]) / sp)))
    i1 = min(nx - 1, int(np.ceil((center[0] + radius - lo[0]) / sp)))
    j0 = max(0, int(np.floor((center[1] - radius - lo[1]) / sp)))
    j1 = min(ny - 1, int(np.ceil((center[1] + radius - lo[1]) / sp)))
    k0 = max(0, int(np.floor((center[2] - radius - lo[2]) / sp)))
    k1 = min(nz - 1, int(np.ceil((center[2] + radius - lo[2]) / sp)))
    return slice(i0, i1 + 1), slice(j0, j1 + 1), slice(k0, k1 + 1)


def _phi_sphere_on_grid(
    center: np.ndarray,
    radius: float,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    sl_x: slice,
    sl_y: slice,
    sl_z: slice,
) -> np.ndarray:
    """
    ||x - c|| - r on a sub-grid via separable broadcasting (no (nx,ny,nz,3) buffer).
    """
    dx = xs[sl_x] - center[0]
    dy = ys[sl_y] - center[1]
    dz = zs[sl_z] - center[2]
    return np.sqrt(
        dx[:, None, None] ** 2 + dy[None, :, None] ** 2 + dz[None, None, :] ** 2
    ) - radius


def _accumulate_sas_phi(
    phi: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    origin: np.ndarray,
    spacing: float,
) -> None:
    """In-place Φ = min_i (||x-c_i|| - r_i) using per-atom local sub-grids."""
    xs, ys, zs = _axis_coordinates(origin, spacing, phi.shape)
    dims = phi.shape
    for center, radius in zip(centers, radii):
        sl_x, sl_y, sl_z = _atom_voxel_slices(center, float(radius), origin, spacing, dims)
        if sl_x.start >= sl_x.stop or sl_y.start >= sl_y.stop or sl_z.start >= sl_z.stop:
            continue
        local = _phi_sphere_on_grid(center, float(radius), xs, ys, zs, sl_x, sl_y, sl_z)
        np.minimum(phi[sl_x, sl_y, sl_z], local, out=phi[sl_x, sl_y, sl_z])


def build_sasa_distance_field(
    structure: ProteinStructure,
    *,
    resolution: float = 1.5,
    probe_radius: float = DEFAULT_PROBE_RADIUS,
    padding: float = 6.0,
    max_grid_dim: int = DEFAULT_MAX_GRID_DIM,
) -> DistanceFieldGrid:
    """
    Build Φ on a cubic grid covering the structure.

    ``resolution`` is voxel size in Å. If the implied grid exceeds
    ``max_grid_dim`` along any axis, resolution is increased automatically.
    """
    if structure.n_atoms < 1:
        raise ValueError("Structure has no atoms")

    centers = structure.centers
    radii = structure.vdw_radii + float(probe_radius)
    bb_min, bb_max = structure.bounding_box(padding=padding + float(radii.max()))

    origin, spacing, dims = _grid_from_bounds(
        bb_min, bb_max, resolution=resolution, max_grid_dim=max_grid_dim
    )
    phi = np.full(dims, np.inf, dtype=np.float64)
    _accumulate_sas_phi(phi, centers, radii, origin, spacing)

    return DistanceFieldGrid(values=phi.astype(np.float32), origin=origin, spacing=spacing)


_SPHERE_OFFSETS_CACHE: dict[int, np.ndarray] = {}


def _sphere_offsets(radius_vox: int) -> np.ndarray:
    """
    Integer voxel offsets within a sphere of radius ``radius_vox`` (inclusive).

    Cached because many atoms share the same radius after quantization.
    """
    radius_vox = int(max(radius_vox, 0))
    if radius_vox in _SPHERE_OFFSETS_CACHE:
        return _SPHERE_OFFSETS_CACHE[radius_vox]
    r = radius_vox
    if r == 0:
        offsets = np.array([[0, 0, 0]], dtype=np.int32)
    else:
        grid = np.arange(-r, r + 1, dtype=np.int32)
        dx, dy, dz = np.meshgrid(grid, grid, grid, indexing="ij")
        mask = (dx * dx + dy * dy + dz * dz) <= (r * r)
        offsets = np.stack([dx[mask], dy[mask], dz[mask]], axis=1).astype(np.int32)
    _SPHERE_OFFSETS_CACHE[radius_vox] = offsets
    return offsets


def _paint_voxel_spheres(
    inside: np.ndarray,
    centers_vox: np.ndarray,
    radii_vox: np.ndarray,
) -> None:
    """Rasterize union of spheres onto ``inside`` (batched by quantized radius)."""
    nx, ny, nz = inside.shape
    bounds = np.array([nx, ny, nz], dtype=np.int32)

    for r in np.unique(radii_vox):
        r_int = int(max(r, 0))
        atom_mask = radii_vox == r
        if not np.any(atom_mask):
            continue
        cents = centers_vox[atom_mask]
        offsets = _sphere_offsets(r_int)
        # (n_atoms, n_offsets, 3)
        ijk = cents[:, None, :] + offsets[None, :, :]
        ijk = ijk.reshape(-1, 3)
        valid = (ijk >= 0).all(axis=1) & (ijk < bounds).all(axis=1)
        ijk = ijk[valid]
        inside[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True


def build_sasa_signed_distance_field_voxel(
    structure: ProteinStructure,
    *,
    resolution: float = 2.0,
    probe_radius: float = DEFAULT_PROBE_RADIUS,
    padding: float = 6.0,
    max_grid_dim: int = DEFAULT_MAX_GRID_DIM,
    smooth_sigma_voxels: float = 0.75,
) -> DistanceFieldGrid:
    """
    Fast SAS signed distance field via voxelized union-of-spheres + distance transform.

    This approximates the SAS envelope by rasterizing spheres of radius
    ``r_vdw + probe_radius`` onto a boolean grid, then computing a signed distance
    field using Euclidean distance transforms:

        d_out = EDT(outside),  d_in = EDT(inside)
        sdf = d_out; sdf[inside] = -d_in[inside]

    Performance: scales roughly with grid size and the total painted voxel volume,
    and runs mostly in compiled code (NumPy + SciPy). For large atom counts this is
    far faster than the exact per-atom Φ(x) minimum.
    """
    if structure.n_atoms < 1:
        raise ValueError("Structure has no atoms")

    centers = structure.centers
    radii = structure.vdw_radii + float(probe_radius)
    bb_min, bb_max = structure.bounding_box(padding=padding + float(radii.max()))
    origin, spacing, dims = _grid_from_bounds(
        bb_min,
        bb_max,
        resolution=resolution,
        max_grid_dim=max_grid_dim,
    )

    inside = np.zeros(dims, dtype=bool)
    radii_vox = np.ceil(radii / spacing).astype(np.int32)
    centers_vox = np.rint((centers - origin.reshape(1, 3)) / spacing).astype(np.int32)
    _paint_voxel_spheres(inside, centers_vox, radii_vox)

    # Signed distance transform in Å.
    dist_out = ndimage.distance_transform_edt(~inside).astype(np.float64) * spacing
    dist_in = ndimage.distance_transform_edt(inside).astype(np.float64) * spacing
    sdf = dist_out
    sdf[inside] = -dist_in[inside]

    grid = DistanceFieldGrid(values=sdf.astype(np.float32), origin=origin, spacing=spacing)
    if smooth_sigma_voxels and smooth_sigma_voxels > 0:
        grid = refine_distance_field(grid, smooth_sigma_voxels=float(smooth_sigma_voxels))
    return grid


def build_sasa_distance_field_auto(
    structure: ProteinStructure,
    *,
    resolution: float = 1.5,
    probe_radius: float = DEFAULT_PROBE_RADIUS,
    padding: float = 6.0,
    max_grid_dim: int = DEFAULT_MAX_GRID_DIM,
) -> DistanceFieldGrid:
    """
    Choose a distance-field backend based on structure size.

    - Small structures: exact Φ(x) minimum (cleanest)
    - Large structures: voxel SDF (much faster)
    """
    if structure.n_atoms >= 12000:
        return build_sasa_signed_distance_field_voxel(
            structure,
            resolution=max(1.8, float(resolution)),
            probe_radius=probe_radius,
            padding=padding,
            max_grid_dim=max_grid_dim,
        )
    return build_sasa_distance_field(
        structure,
        resolution=resolution,
        probe_radius=probe_radius,
        padding=padding,
        max_grid_dim=max_grid_dim,
    )


def estimate_sasa_area(
    structure: ProteinStructure,
    *,
    resolution: float = 1.0,
    probe_radius: float = DEFAULT_PROBE_RADIUS,
) -> float:
    """
    Approximate total SASA (Å²) by counting zero-level voxels.

    Educational estimate; not calibrated to analytical Lee–Richards.
    """
    grid = build_sasa_distance_field_auto(
        structure,
        resolution=resolution,
        probe_radius=probe_radius,
    )
    phi = grid.values
    # Voxels near Φ = 0
    band = np.abs(phi) <= 0.6 * grid.spacing
    n_surface = int(band.sum())
    return float(n_surface * (grid.spacing**2))


def mask_inside_vdw(
    structure: ProteinStructure,
    grid: DistanceFieldGrid,
) -> np.ndarray:
    """Boolean mask: voxel centers inside the union of vdW spheres."""
    xs, ys, zs = _axis_coordinates(grid.origin, grid.spacing, grid.shape)
    phi_vdw = np.full(grid.shape, np.inf, dtype=np.float64)
    _accumulate_sas_phi(phi_vdw, structure.centers, structure.vdw_radii, grid.origin, grid.spacing)
    return phi_vdw < 0.0


def refine_distance_field(
    grid: DistanceFieldGrid,
    *,
    smooth_sigma_voxels: float = 0.75,
) -> DistanceFieldGrid:
    """Light Gaussian smoothing of Φ for stabler marching-cubes surfaces."""
    smoothed = ndimage.gaussian_filter(
        grid.values.astype(np.float64),
        sigma=smooth_sigma_voxels,
        mode="nearest",
    )
    return DistanceFieldGrid(
        values=smoothed.astype(np.float32),
        origin=grid.origin.copy(),
        spacing=grid.spacing,
    )
