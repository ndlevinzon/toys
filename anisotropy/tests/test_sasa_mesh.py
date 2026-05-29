"""Numerical regression tests for SAS distance fields and mesh smoothing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anisotropy.mesh import laplacian_smooth, mesh_from_distance_field
from anisotropy.pdb import load_pdb
from anisotropy.sasa import (
    DistanceFieldGrid,
    _accumulate_sas_phi,
    _axis_coordinates,
    build_sasa_distance_field,
    build_sasa_signed_distance_field_voxel,
)


def _brute_force_phi(
    centers: np.ndarray,
    radii: np.ndarray,
    origin: np.ndarray,
    spacing: float,
    dims: tuple[int, int, int],
) -> np.ndarray:
    """Reference Φ on full grid (legacy all-voxel algorithm)."""
    nx, ny, nz = dims
    coords = np.empty((nx, ny, nz, 3), dtype=np.float64)
    xs, ys, zs = _axis_coordinates(origin, spacing, dims)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    coords[..., 0] = xx
    coords[..., 1] = yy
    coords[..., 2] = zz
    phi = np.full(dims, np.inf, dtype=np.float64)
    for center, radius in zip(centers, radii):
        dist = np.linalg.norm(coords - center, axis=-1)
        phi = np.minimum(phi, dist - radius)
    return phi


@pytest.fixture(scope="module")
def small_structure():
    pdb = Path(__file__).resolve().parents[1] / "1CRN.pdb"
    if not pdb.is_file():
        pytest.skip("1CRN.pdb not available")
    return load_pdb(pdb)


def test_exact_phi_matches_brute_force(small_structure) -> None:
    origin = small_structure.bounding_box(padding=4.0)[0]
    spacing = 1.2
    extent = small_structure.bounding_box(padding=4.0)[1] - origin
    dims = tuple(int(np.ceil(extent / spacing).astype(int) + 1))

    centers = small_structure.centers
    radii = small_structure.vdw_radii + 1.4
    phi_fast = np.full(dims, np.inf, dtype=np.float64)
    _accumulate_sas_phi(phi_fast, centers, radii, origin, spacing)
    phi_ref = _brute_force_phi(centers, radii, origin, spacing, dims)

    np.testing.assert_allclose(phi_fast, phi_ref, rtol=0, atol=1e-9)


def test_laplacian_smooth_unchanged(small_structure) -> None:
    grid = build_sasa_distance_field(small_structure, resolution=2.0)
    verts, faces = mesh_from_distance_field(grid, smooth_iterations=2)
    n = verts.shape[0]
    nbrs_sets: list[set[int]] = [set() for _ in range(n)]
    for a, b, c in faces:
        nbrs_sets[a].update((b, c))
        nbrs_sets[b].update((a, c))
        nbrs_sets[c].update((a, b))
    neighbors = [sorted(s) for s in nbrs_sets]

    v = verts.astype(np.float64)
    lam = 0.45
    for _ in range(2):
        new_v = v.copy()
        for i, idx in enumerate(neighbors):
            if not idx:
                continue
            new_v[i] = v[i] + lam * (v[idx].mean(axis=0) - v[i])
        v = new_v

    smoothed = laplacian_smooth(verts, faces, iterations=2, lam=lam)
    np.testing.assert_allclose(smoothed, v, rtol=0, atol=1e-12)


def test_voxel_and_exact_grids_finite(small_structure) -> None:
    exact = build_sasa_distance_field(small_structure, resolution=2.5)
    voxel = build_sasa_signed_distance_field_voxel(small_structure, resolution=2.5)
    assert np.isfinite(exact.values).all()
    assert np.isfinite(voxel.values).all()
    assert isinstance(exact, DistanceFieldGrid)
