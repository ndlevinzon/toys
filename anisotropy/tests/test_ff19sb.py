"""Tests for AMBER ff19SB library parsing and charge assignment."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from anisotropy.ff19sb import (
    DEFAULT_AMINO19_LIB,
    Ff19sbLibrary,
    assign_ff19sb_charges,
    default_ff19sb_library,
    select_amber_residue_name,
)
from anisotropy.pdb import Atom, ProteinStructure, load_pdb


@pytest.fixture(scope="module")
def library() -> Ff19sbLibrary:
    if not DEFAULT_AMINO19_LIB.is_file():
        pytest.skip(f"Bundled ff19SB library not found: {DEFAULT_AMINO19_LIB}")
    return default_ff19sb_library()


def test_library_parses_standard_residues(library: Ff19sbLibrary) -> None:
    assert "ALA" in library.templates
    assert "ASP" in library.templates
    assert "ASH" in library.templates
    assert "HIE" in library.templates
    ala = library.get("ALA")
    assert "CA" in ala.charges
    assert abs(ala.net_charge) < 1e-3


def test_asp_deprotonated_neutral(library: Ff19sbLibrary) -> None:
    asp = library.get("ASP")
    assert abs(asp.net_charge + 1.0) < 1e-2


def test_ash_protonated_neutral(library: Ff19sbLibrary) -> None:
    ash = library.get("ASH")
    assert abs(ash.net_charge) < 1e-2


def test_protonation_hydrogen_detection() -> None:
    asp_atoms = [
        Atom(1, "CG", "ASP", "A", 1, np.zeros(3), "C"),
        Atom(2, "HD2", "ASP", "A", 1, np.zeros(3), "H"),
    ]
    assert select_amber_residue_name("ASP", asp_atoms) == "ASH"
    assert select_amber_residue_name("ASP", [asp_atoms[0]]) == "ASP"


def test_assign_crn_net_charge_near_zero(library: Ff19sbLibrary) -> None:
    pdb = Path(__file__).resolve().parents[1] / "1CRN.pdb"
    if not pdb.is_file():
        pytest.skip("1CRN.pdb not available")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        structure = load_pdb(pdb, include_hydrogen=True)
    assignment = assign_ff19sb_charges(structure, library, ph=7.0)
    assert assignment.n_unknown_residues == 0
    assert abs(assignment.total_charge) < 0.5
