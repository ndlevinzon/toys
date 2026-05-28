"""
Residue-level chemistry proxies for patch parameterization.

Not a full force field: tables are chosen for interpretable patch features
(hydropathy, titration, H-bond capacity) aligned with interfacial modeling.
"""

from __future__ import annotations

import math

# Kyte–Doolittle hydropathy (positive = hydrophobic).
KYTE_DOOLITTLE: dict[str, float] = {
    "ILE": 4.5,
    "VAL": 4.2,
    "LEU": 3.8,
    "PHE": 2.8,
    "CYS": 2.5,
    "MET": 1.9,
    "ALA": 1.8,
    "GLY": -0.4,
    "THR": -0.7,
    "SER": -0.8,
    "TRP": -0.9,
    "TYR": -1.3,
    "PRO": -1.6,
    "HIS": -3.2,
    "GLU": -3.5,
    "GLN": -3.5,
    "ASP": -3.5,
    "ASN": -3.5,
    "LYS": -3.9,
    "ARG": -4.5,
}
DEFAULT_HYDROPATHY = 0.0

# Effective pKa for Henderson–Hasselbalch fractional charge (toy model).
RESIDUE_PKA: dict[str, float] = {
    "ASP": 3.9,
    "GLU": 4.3,
    "HIS": 6.0,
    "CYS": 8.3,
    "TYR": 10.1,
    "LYS": 10.5,
    "ARG": 12.5,
    "NTERM": 8.0,
    "CTERM": 3.2,
}

# Formal charge at pH 7 (fallback when not titratable).
RESIDUE_CHARGE_PH7: dict[str, float] = {
    "ARG": 1.0,
    "LYS": 1.0,
    "HIS": 0.1,
    "ASP": -1.0,
    "GLU": -1.0,
}

# H-bond donor / acceptor counts per residue (heavy-atom proxy).
HBOND_DONORS: dict[str, int] = {
    "ARG": 5,
    "LYS": 3,
    "TRP": 2,
    "ASN": 2,
    "GLN": 2,
    "HIS": 2,
    "SER": 1,
    "THR": 1,
    "TYR": 1,
    "GLY": 0,
}
HBOND_ACCEPTORS: dict[str, int] = {
    "ASP": 2,
    "GLU": 2,
    "ASN": 2,
    "GLN": 2,
    "HIS": 2,
    "SER": 1,
    "THR": 1,
    "TYR": 1,
    "MET": 1,
    "CYS": 1,
    "PHE": 0,
    "VAL": 0,
}

POLAR_RESIDUES = frozenset(
    {"SER", "THR", "ASN", "GLN", "ASP", "GLU", "LYS", "ARG", "HIS", "TYR", "CYS"}
)
CHARGED_RESIDUES = frozenset({"ASP", "GLU", "LYS", "ARG", "HIS"})

# PROPKA group types (see ``propka_pka.PROPKA_ACID_TYPES``).
ACIDIC_GROUP_TYPES = frozenset({"COO", "CYS", "TYR", "C-", "OCO", "SH", "OP", "SER"})
BASIC_GROUP_TYPES = frozenset(
    {"ARG", "LYS", "HIS", "N+", "CG", "C2N", "N30", "N31", "N32", "N33", "NAR"}
)


def is_acidic_group_type(group_type: str) -> bool:
    return group_type.upper() in ACIDIC_GROUP_TYPES


def fractional_charge_from_pka(pka: float, *, acidic: bool, ph: float) -> float:
    """Henderson–Hasselbalch fractional charge from a single pKa."""
    if acidic:
        return -1.0 / (1.0 + 10.0 ** (ph - pka))
    return 1.0 / (1.0 + 10.0 ** (pka - ph))


def residue_hydropathy(resname: str) -> float:
    return KYTE_DOOLITTLE.get(resname.upper(), DEFAULT_HYDROPATHY)


def residue_charge_at_ph(resname: str, ph: float) -> float:
    """Fractional charge from tabulated residue pKa (fallback when PROPKA is off)."""
    key = resname.upper()
    if key not in RESIDUE_PKA:
        return RESIDUE_CHARGE_PH7.get(key, 0.0)
    pka = RESIDUE_PKA[key]
    acidic = key in {"ASP", "GLU", "CYS", "TYR", "CTERM"}
    return fractional_charge_from_pka(pka, acidic=acidic, ph=ph)


def residue_pka_mean(resname: str) -> float | None:
    return RESIDUE_PKA.get(resname.upper())


def atom_partial_charge_proxy(
    resname: str,
    atom_name: str,
    ph: float,
    *,
    n_atoms_in_residue: int = 4,
) -> float:
    """Per-atom charge: residue charge spread over heavy atoms in the residue."""
    res_charge = residue_charge_at_ph(resname, ph)
    return res_charge / max(n_atoms_in_residue, 1)


def hbond_donors_acceptors(resname: str) -> tuple[int, int]:
    key = resname.upper()
    return HBOND_DONORS.get(key, 0), HBOND_ACCEPTORS.get(key, 0)
