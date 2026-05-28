"""Tests for ising_params.yaml loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from anisotropy.ising_params import (
    DEFAULT_ISING_PARAMS_PATH,
    load_ising_params,
)
from anisotropy.lattice_solvent_hamiltonian import HybridHamiltonianCouplings


@pytest.fixture(scope="module")
def params():
    if not DEFAULT_ISING_PARAMS_PATH.is_file():
        pytest.skip("ising_params.yaml not found")
    return load_ising_params()


def test_yaml_exists() -> None:
    assert DEFAULT_ISING_PARAMS_PATH.is_file()


def test_solv_defaults(params) -> None:
    assert params.solv.J == 1.0
    assert params.solv.mu == 0.0
    assert params.solv.u_film_scale == 0.25
    assert params.solv.occupancy_mode == "binary"


def test_hybrid_couplings(params) -> None:
    c = params.to_hybrid_couplings()
    assert isinstance(c, HybridHamiltonianCouplings)
    assert c.lambda_hp == 0.8
    assert c.lambda_film == 0.3


def test_slab_build(params) -> None:
    slab = params.build_slab()
    assert slab.thickness_angstrom == 300.0


def test_from_ising_params_classmethod() -> None:
    c = HybridHamiltonianCouplings.from_ising_params()
    assert c.J_solv == 1.0
