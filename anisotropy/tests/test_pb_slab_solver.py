"""Tests for slab FFT–PB electrostatics."""

from __future__ import annotations

import numpy as np

from anisotropy.awi_field import build_cryo_slab_preset
from anisotropy.lattice_solvent_hamiltonian import CartesianLattice
from anisotropy.pb_slab_solver import (
    PBSolverParams,
    SlabPBSolver,
    deposit_charges_trilinear,
    solve_linearized_pb_slab,
)


def test_pb_homogeneous_vs_yukawa_order() -> None:
    """Single charge in bulk water: PB potential should be positive and finite."""
    slab = build_cryo_slab_preset(thickness_angstrom=80.0)
    h = 4.0
    shape = (16, 16, 20)
    lattice = CartesianLattice(
        origin=np.array([-32.0, -32.0, 10.0]),
        spacing=h,
        shape=shape,
    )
    z = lattice.grid_centers_xyz()[0, 0, :, 2]
    samp = slab.sample_fields(z, blend_interfaces=True)
    eps = np.full(shape[2], 78.0)
    kap = np.zeros(shape[2])

    center = lattice.origin + 0.5 * np.array(shape) * h
    rho = deposit_charges_trilinear(lattice, center.reshape(1, 3), np.array([1.0]))
    phi = solve_linearized_pb_slab(rho, lattice, eps, kap)
    val = phi[shape[0] // 2, shape[1] // 2, shape[2] // 2]
    assert np.isfinite(val)
    assert val > 0.0


def test_slab_pb_solver_energy_finite() -> None:
    slab = build_cryo_slab_preset(thickness_angstrom=60.0)
    lattice = CartesianLattice(
        origin=np.array([-20.0, -20.0, 5.0]),
        spacing=3.5,
        shape=(12, 12, 14),
    )
    params = PBSolverParams(method="pb_slab", coarse_factor=2)
    solver = SlabPBSolver.from_lattice(lattice, slab, params)
    cents = np.array([[0.0, 0.0, 25.0], [5.0, 0.0, 28.0]], dtype=np.float64)
    q = np.array([1.0, -0.5])
    mu = np.zeros((2, 3))
    e, terms = solver.energy_from_charges(cents, q, mu)
    assert np.isfinite(e)
    assert terms["method"] == "pb_slab"
