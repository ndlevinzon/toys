"""
Minimal PDB reader for protein heavy atoms.

Parses ATOM records (optionally HETATM excluding solvent). Van der Waals radii
follow Bondi values used in standard SASA models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Bondi vdW radii (Å); unknown elements fall back to carbon.
VDW_RADIUS_ANGSTROM: dict[str, float] = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "SE": 1.90,
    "ZN": 1.39,
    "FE": 1.42,
    "MG": 1.73,
    "CA": 2.00,
}
DEFAULT_VDW = VDW_RADIUS_ANGSTROM["C"]

# Residue names treated as solvent (skipped unless include_solvent=True).
SOLVENT_RESIDUES = frozenset({"HOH", "WAT", "H2O", "DOD", "TIP", "SOL"})


@dataclass(frozen=True)
class Atom:
    """One heavy atom from a PDB coordinate file."""

    serial: int
    name: str
    resname: str
    chain: str
    resseq: int
    xyz: np.ndarray  # (3,) Å
    element: str
    bfactor: float | None = None  # crystallographic B-factor if present in PDB

    @property
    def vdw_radius(self) -> float:
        return VDW_RADIUS_ANGSTROM.get(self.element, DEFAULT_VDW)


@dataclass
class ProteinStructure:
    """Parsed atom list with metadata."""

    atoms: list[Atom]
    pdb_id: str | None = None
    source_path: str | None = None
    _centers_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _vdw_cache: np.ndarray | None = field(default=None, init=False, repr=False)

    @property
    def centers(self) -> np.ndarray:
        """(N, 3) coordinates."""
        if self._centers_cache is None:
            self._centers_cache = np.stack([a.xyz for a in self.atoms], axis=0)
        return self._centers_cache

    @property
    def vdw_radii(self) -> np.ndarray:
        if self._vdw_cache is None:
            self._vdw_cache = np.array([a.vdw_radius for a in self.atoms], dtype=np.float64)
        return self._vdw_cache

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    def bounding_box(self, padding: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounds (min, max) with optional padding (Å)."""
        c = self.centers
        return c.min(axis=0) - padding, c.max(axis=0) + padding


def _guess_element(atom_name: str, resname: str) -> str:
    """Infer element symbol from PDB atom and residue names."""
    name = atom_name.strip().upper()
    if len(name) >= 2 and name[0].isalpha() and name[1].isalpha():
        two = name[:2]
        if two in VDW_RADIUS_ANGSTROM:
            return two
    if name and name[0].isalpha():
        return name[0]
    if resname and resname[0].isalpha():
        return resname[0]
    return "C"


def load_pdb(
    path: str | Path,
    *,
    include_hetatm: bool = False,
    include_solvent: bool = False,
    include_hydrogen: bool = False,
) -> ProteinStructure:
    """
    Load ATOM records from a PDB file.

    Parameters
    ----------
    path
        Path to ``.pdb`` or ``.ent`` file.
    include_hetatm
        If True, parse HETATM as well as ATOM (ligands, cofactors).
    include_solvent
        If False, skip water and common solvent residue names.
    include_hydrogen
        If False, skip hydrogen atoms.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    atoms: list[Atom] = []
    pdb_id: str | None = None

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("HEADER") and pdb_id is None:
                parts = line.split()
                if len(parts) >= 4:
                    pdb_id = parts[3].lower()
            record = line[0:6].strip()
            if record == "ATOM" or (include_hetatm and record == "HETATM"):
                if len(line) < 54:
                    continue
                resname = line[17:20].strip().upper()
                if not include_solvent and resname in SOLVENT_RESIDUES:
                    continue
                name = line[12:16].strip().upper()
                element = (line[76:78].strip().upper() if len(line) >= 78 else "") or _guess_element(
                    name, resname
                )
                if not include_hydrogen and element == "H":
                    continue
                try:
                    serial = int(line[6:11])
                    resseq = int(line[22:26])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    bfactor = (
                        float(line[60:66])
                        if len(line) >= 66 and line[60:66].strip()
                        else None
                    )
                except ValueError:
                    continue
                chain = line[21] if len(line) > 21 else "A"
                atoms.append(
                    Atom(
                        serial=serial,
                        name=name,
                        resname=resname,
                        chain=chain,
                        resseq=resseq,
                        xyz=np.array([x, y, z], dtype=np.float64),
                        element=element,
                        bfactor=bfactor,
                    )
                )

    if not atoms:
        raise ValueError(f"No atoms parsed from {path}")

    return ProteinStructure(atoms=atoms, pdb_id=pdb_id, source_path=str(path.resolve()))
