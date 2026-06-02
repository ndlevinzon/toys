"""Tests for layered RISM first shell + outer Ising solvation."""

from __future__ import annotations

import numpy as np

from anisotropy.lattice_solvent_hamiltonian import (
    CartesianLattice,
    occupancy_binary_template,
)
from anisotropy.mesh import ProteinMesh
from anisotropy.rism_solvation import (
    CanonicalShellCache,
    RismSolvationParams,
    precompute_outer_solvation_only,
    rism_excess_chemical_potential,
    rism_first_shell_energy,
)


def _tiny_mesh() -> ProteinMesh:
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
    return ProteinMesh(vertices=verts, faces=faces)


def test_distance_field_positive_outside() -> None:
    mesh = _tiny_mesh()
    cache = CanonicalShellCache.build(
        mesh, spacing=2.0, pad_angstrom=4.0, first_shell_angstrom=5.0
    )
    dist = cache.dist_can
    assert dist[cache.inside_can].max() == 0.0
    assert np.any(dist[~cache.inside_can] > 0.0)


def test_shell_partition_disjoint() -> None:
    mesh = _tiny_mesh()
    cache = CanonicalShellCache.build(
        mesh, spacing=2.0, pad_angstrom=4.0, first_shell_angstrom=4.0
    )
    assert not np.any(cache.first_shell_can & cache.outer_solvent_can)
    assert np.all(cache.first_shell_can | cache.outer_solvent_can | cache.inside_can)


def test_rism_mu_ex_finite() -> None:
    xyz = np.array([[5.0, 5.0, 12.0]], dtype=np.float64)
    cents = np.array([[5.0, 5.0, 5.0]], dtype=np.float64)
    mu = rism_excess_chemical_potential(
        xyz, cents, np.array([-1.0]), np.array([20.0]), RismSolvationParams()
    )
    assert np.all(np.isfinite(mu))
    assert mu.shape == (1,)


def test_layered_energy_split() -> None:
    mesh = _tiny_mesh()
    h = 2.5
    origin = np.array([-5.0, -5.0, -5.0])
    shape = (12, 12, 12)
    lattice = CartesianLattice(origin=origin, spacing=h, shape=shape)  # type: ignore[arg-type]
    occ = occupancy_binary_template(lattice, solvent_z_within=(0.0, 30.0))

    cache = CanonicalShellCache.build(
        mesh, spacing=h, pad_angstrom=8.0, first_shell_angstrom=5.0
    )

    class _Coeffs:
        J_solv = 1.0
        mu_chemical = 0.0
        u_film_scale = 0.0

    class _Slab:
        thickness_angstrom = 30.0

    eye = np.eye(3)
    t0 = np.zeros(3)
    h_outer, _ = precompute_outer_solvation_only(
        occ, lattice, _Slab(), _Coeffs(), cache, eye, t0, u_film_scale=0.0
    )

    from anisotropy.patches import PatchFeatures, PatchParameterization

    param = PatchParameterization(
        patches=[
            PatchFeatures(
                patch_id=0,
                area=50.0,
                centroid=np.array([5.0, 5.0, 5.0]),
                normal=np.array([0.0, 0.0, 1.0]),
                mean_curvature=0.0,
                gaussian_curvature=0.0,
                charge=-0.5,
                potential=0.0,
                pka_acid=7.0,
                hydropathy=1.0,
                polar_density=0.1,
                hbond_score=0.2,
                dipole=np.zeros(3),
                softness=0.5,
                face_indices=np.array([0], dtype=np.int64),
                n_atoms=10,
            )
        ],
        face_patch_ids=np.zeros(4, dtype=np.int32),
        ph=7.0,
    )

    first, _, _ = cache.lab_shell_masks(lattice, eye, t0, lab_xyz_flat=lattice.grid_centers_xyz().reshape(-1, 3))
    h_rism, terms = rism_first_shell_energy(
        occ,
        first,
        lattice.grid_centers_xyz().reshape(-1, 3),
        param,
        RismSolvationParams(),
    )
    assert np.isfinite(h_outer)
    assert np.isfinite(h_rism)
    assert terms["n_first_shell_sites"] >= 0
