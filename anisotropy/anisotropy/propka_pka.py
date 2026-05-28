"""
Structure-based pKa via PROPKA 3 (optional dependency).

One PROPKA run per PDB; results are mapped to residues and used for patch
charges and mean patch pKa features.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from anisotropy.pdb import Atom, ProteinStructure
from anisotropy.residue_chemistry import (
    fractional_charge_from_pka,
    residue_charge_at_ph,
    residue_pka_mean,
)

# PROPKA group ``type`` strings treated as acids / bases for Henderson–Hasselbalch.
PROPKA_ACID_TYPES = frozenset({"COO", "CYS", "TYR", "C-", "OCO", "SH", "OP", "SER"})
PROPKA_BASE_TYPES = frozenset(
    {"ARG", "LYS", "HIS", "N+", "CG", "C2N", "N30", "N31", "N32", "N33", "NAR"}
)


@dataclass(frozen=True)
class IonizableSite:
    """One titratable group from a PROPKA run."""

    chain: str
    resseq: int
    resname: str
    group_type: str
    pka: float
    label: str
    acidic: bool


@dataclass
class PropkaPkaLookup:
    """
    Residue-keyed PROPKA pKa values for a single structure.

    Keys are ``(chain, resseq, resname)`` matching :class:`~anisotropy.pdb.Atom`.
    """

    sites: dict[tuple[str, int, str], IonizableSite]
    source_path: str

    def pka_for_atom(self, atom: Atom) -> float | None:
        site = self.sites.get(_residue_key(atom))
        return site.pka if site is not None else None

    def residue_charge(self, atom: Atom, ph: float) -> float | None:
        """Fractional charge from PROPKA pKa; ``None`` if residue is not titratable."""
        site = self.sites.get(_residue_key(atom))
        if site is None:
            return None
        return fractional_charge_from_pka(site.pka, acidic=site.acidic, ph=ph)

    @classmethod
    def from_pdb(cls, pdb_path: str | Path) -> PropkaPkaLookup:
        try:
            from propka.run import single
        except ImportError as exc:
            raise ImportError(
                "PROPKA is not installed. Run: pip install propka"
            ) from exc

        pdb_path = Path(pdb_path)
        if not pdb_path.is_file():
            raise FileNotFoundError(pdb_path)

        mol = single(str(pdb_path), write_pka=False)
        conf_name = next(iter(mol.conformations))
        conformation = mol.conformations[conf_name]

        sites: dict[tuple[str, int, str], IonizableSite] = {}
        for group in conformation.groups:
            if not getattr(group, "titratable", False):
                continue
            atom = group.atom
            resname = str(atom.res_name).strip().upper()
            chain = str(atom.chain_id).strip() or " "
            resseq = int(atom.res_num)
            gtype = str(group.type).strip().upper()
            acidic = gtype in PROPKA_ACID_TYPES or (
                gtype not in PROPKA_BASE_TYPES and float(getattr(group, "charge", 0)) < 0
            )
            pka = float(group.pka_value)
            key = (chain, resseq, resname)
            sites[key] = IonizableSite(
                chain=chain,
                resseq=resseq,
                resname=resname,
                group_type=gtype,
                pka=pka,
                label=str(group.label).strip(),
                acidic=acidic,
            )
        return cls(sites=sites, source_path=str(pdb_path.resolve()))

    @classmethod
    def try_from_pdb(cls, pdb_path: str | Path) -> PropkaPkaLookup | None:
        try:
            return cls.from_pdb(pdb_path)
        except ImportError:
            return None
        except Exception as exc:
            warnings.warn(f"PROPKA failed ({exc}); using tabulated pKa.", stacklevel=2)
            return None


def residue_key(atom: Atom) -> tuple[str, int, str]:
    """Hashable key ``(chain, resseq, resname)`` for residue-level lookups."""
    chain = atom.chain.strip() if atom.chain else " "
    return (chain, int(atom.resseq), atom.resname.upper())


def _residue_key(atom: Atom) -> tuple[str, int, str]:
    return residue_key(atom)


def residue_atom_counts(structure: ProteinStructure) -> dict[tuple[str, int, str], int]:
    counts: dict[tuple[str, int, str], int] = defaultdict(int)
    for atom in structure.atoms:
        counts[_residue_key(atom)] += 1
    return dict(counts)


def atom_charge_at_ph(
    atom: Atom,
    ph: float,
    *,
    pka_lookup: PropkaPkaLookup | None = None,
    residue_atom_counts: dict[tuple[str, int, str], int] | None = None,
) -> float:
    """
    Per-atom charge proxy: residue fractional charge divided by heavy-atom count.

    Uses PROPKA when ``pka_lookup`` is provided; otherwise tabulated residue pKa.
    """
    key = _residue_key(atom)
    n_in_res = 1
    if residue_atom_counts is not None:
        n_in_res = max(residue_atom_counts.get(key, 1), 1)

    if pka_lookup is not None:
        q_res = pka_lookup.residue_charge(atom, ph)
        if q_res is not None:
            return q_res / n_in_res

    q_res = residue_charge_at_ph(atom.resname, ph)
    return q_res / n_in_res


def pka_for_patch_atom(
    atom: Atom,
    *,
    pka_lookup: PropkaPkaLookup | None = None,
) -> float | None:
    if pka_lookup is not None:
        return pka_lookup.pka_for_atom(atom)
    return residue_pka_mean(atom.resname)
