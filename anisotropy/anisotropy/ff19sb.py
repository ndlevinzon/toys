"""
AMBER ff19SB partial charges from LEAP ``amino19.lib`` (OFF format).

Follows standard AMBER protein conventions (see ``leaprc.protein.ff19SB``):
partial charges are in units of *e* (elementary charge); dipoles from
:math:`\\boldsymbol{\\mu} = \\sum_i q_i \\mathbf{r}_i` use the same convention
(Å·e; multiply by 4.80320425 to convert to Debye).

Terminal caps (``NALA`` / ``CALA`` from ``aminont12.lib`` / ``aminoct12.lib``) are
not bundled here; chain termini use the interior residue templates with a warning.
"""

from __future__ import annotations

import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from anisotropy.pdb import Atom, ProteinStructure
from anisotropy.propka_pka import IonizableSite, PropkaPkaLookup, residue_key

# Bundled force field (ff19SB release 2019-07).
_PACKAGE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FF19SB_FORCEFIELD_DIR = (
    _PACKAGE_DIR
    / "utils"
    / "ff19SB_201907-master"
    / "ff19SB_201907-master"
    / "forcefield_files"
)
DEFAULT_AMINO19_LIB = DEFAULT_FF19SB_FORCEFIELD_DIR / "amino19.lib"

_ATOM_TABLE_HEADER = re.compile(
    r"^!entry\.([A-Za-z0-9]+)\.unit\.atoms table\b"
)
# OFF atom row: name, type, typex, resx, flags, seq, element-code, charge (e).
_ATOM_ROW = re.compile(
    r'^\s*"([^"]+)"\s+"([^"]+)"\s+'
    r"(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"
)
_POS_TABLE_HEADER = re.compile(
    r"^!entry\.([A-Za-z0-9]+)\.unit\.positions table\b"
)
_POS_ROW = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*$"
)

# PDB three-letter names → default ff19SB library residue (interior).
_PDB_TO_AMBER: dict[str, str] = {
    "ALA": "ALA",
    "ARG": "ARG",
    "ASN": "ASN",
    "ASP": "ASP",
    "CYS": "CYS",
    "CYX": "CYX",
    "GLN": "GLN",
    "GLU": "GLU",
    "GLY": "GLY",
    "HIS": "HIE",  # leaprc default; overridden by H detection / pKa
    "HID": "HID",
    "HIE": "HIE",
    "HIP": "HIP",
    "HYP": "HYP",
    "ILE": "ILE",
    "LEU": "LEU",
    "LYS": "LYS",
    "MET": "MET",
    "PHE": "PHE",
    "PRO": "PRO",
    "SER": "SER",
    "THR": "THR",
    "TRP": "TRP",
    "TYR": "TYR",
    "VAL": "VAL",
    "ASH": "ASH",
    "GLH": "GLH",
    "LYN": "LYN",
    "CYM": "CYM",
}


@dataclass(frozen=True)
class ResidueTemplate:
    """One residue unit from ``amino19.lib``."""

    name: str
    charges: dict[str, float]
    positions: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def net_charge(self) -> float:
        return float(sum(self.charges.values()))


@dataclass
class Ff19sbLibrary:
    """Parsed ff19SB amino acid OFF library."""

    templates: dict[str, ResidueTemplate]
    source_path: str

    def get(self, amber_resname: str) -> ResidueTemplate:
        key = amber_resname.upper()
        if key not in self.templates:
            raise KeyError(f"Residue '{key}' not in ff19SB library ({self.source_path})")
        return self.templates[key]

    @classmethod
    def from_amino_lib(cls, path: str | Path) -> Ff19sbLibrary:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)

        templates: dict[str, ResidueTemplate] = {}
        current: str | None = None
        mode: str | None = None
        charge_rows: list[tuple[str, float]] = []
        pos_rows: list[np.ndarray] = []
        atom_names_order: list[str] = []

        def _flush_residue() -> None:
            nonlocal current, charge_rows, pos_rows, atom_names_order, mode
            if current is None:
                return
            charges = {name: chg for name, chg in charge_rows}
            positions: dict[str, np.ndarray] = {}
            if pos_rows and atom_names_order and len(pos_rows) == len(atom_names_order):
                positions = {
                    name: pos_rows[i] for i, name in enumerate(atom_names_order)
                }
            templates[current] = ResidueTemplate(
                name=current,
                charges=charges,
                positions=positions,
            )
            current = None
            mode = None
            charge_rows = []
            pos_rows = []
            atom_names_order = []

        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.strip()
                m_head = _ATOM_TABLE_HEADER.match(line)
                if m_head:
                    _flush_residue()
                    current = m_head.group(1).upper()
                    mode = "atoms"
                    charge_rows = []
                    atom_names_order = []
                    continue

                m_pos = _POS_TABLE_HEADER.match(line)
                if m_pos:
                    res = m_pos.group(1).upper()
                    if current is not None and current != res:
                        _flush_residue()
                    current = res
                    mode = "positions"
                    pos_rows = []
                    continue

                if mode == "atoms" and line.startswith('"'):
                    m_row = _ATOM_ROW.match(line)
                    if m_row and current is not None:
                        aname = m_row.group(1).strip()
                        chg = float(m_row.group(8))
                        charge_rows.append((aname, chg))
                        atom_names_order.append(aname)
                    continue

                if mode == "positions":
                    m_p = _POS_ROW.match(line)
                    if m_p and current is not None:
                        pos_rows.append(
                            np.array(
                                [float(m_p.group(1)), float(m_p.group(2)), float(m_p.group(3))],
                                dtype=np.float64,
                            )
                        )
                        continue
                    if line.startswith("!entry."):
                        mode = None
                    continue

        _flush_residue()
        if not templates:
            raise ValueError(f"No residues parsed from {path}")

        return cls(templates=templates, source_path=str(path.resolve()))


@dataclass
class Ff19sbChargeAssignment:
    """Per-atom partial charges from ff19SB for one structure."""

    charges: np.ndarray  # (n_atoms,) aligned with ``structure.atoms``
    amber_residue: dict[tuple[str, int, str], str]  # residue key → template name
    structure: ProteinStructure
    library_path: str
    n_missing_atoms: int = 0
    n_unknown_residues: int = 0
    terminal_residues: list[tuple[str, int, str]] = field(default_factory=list)
    hydrogens_in_structure: bool = False

    @property
    def total_charge(self) -> float:
        return float(self.charges.sum())


def default_ff19sb_library(lib_path: str | Path | None = None) -> Ff19sbLibrary:
    path = Path(lib_path) if lib_path is not None else DEFAULT_AMINO19_LIB
    return Ff19sbLibrary.from_amino_lib(path)


def _residue_atoms(
    structure: ProteinStructure,
) -> dict[tuple[str, int, str], list[Atom]]:
    grouped: dict[tuple[str, int, str], list[Atom]] = defaultdict(list)
    for atom in structure.atoms:
        grouped[residue_key(atom)].append(atom)
    return dict(grouped)


def _chain_terminal_keys(
    structure: ProteinStructure,
) -> set[tuple[str, int, str]]:
    """First and last residue index per chain (by resseq)."""
    by_chain: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    seen: set[tuple[str, int, str]] = set()
    for atom in structure.atoms:
        key = residue_key(atom)
        if key in seen:
            continue
        seen.add(key)
        by_chain[key[0]].append(key)
    terminals: set[tuple[str, int, str]] = set()
    for keys in by_chain.values():
        keys.sort(key=lambda k: k[1])
        terminals.add(keys[0])
        terminals.add(keys[-1])
    return terminals


def _normalize_atom_name(name: str) -> str:
    """PDB atom name to AMBER library key (four-char PDB field, stripped)."""
    return name.strip().upper()


def _atom_names(res_atoms: list[Atom]) -> set[str]:
    return {_normalize_atom_name(a.name) for a in res_atoms}


def _protonation_from_hydrogens(pdb_resname: str, names: set[str]) -> str | None:
    """Infer AMBER variant from titration hydrogens present in the PDB."""
    res = pdb_resname.upper()

    if res in {"ASP", "ASH"}:
        return "ASH" if "HD2" in names else "ASP"
    if res in {"GLU", "GLH"}:
        return "GLH" if "HE2" in names else "GLU"
    if res in {"LYS", "LYN"}:
        # LYN lacks HZ3 (and typically HZ2 in some builds); protonated LYS has HZ3.
        return "LYS" if "HZ3" in names else "LYN"
    if res in {"CYS", "CYM", "CYX"}:
        if res == "CYX":
            return "CYX"
        return "CYS" if "HG" in names else "CYM"
    if res in {"HIS", "HID", "HIE", "HIP"}:
        has_hd1 = "HD1" in names
        has_he2 = "HE2" in names
        if has_hd1 and has_he2:
            return "HIP"
        if has_hd1:
            return "HID"
        if has_he2:
            return "HIE"
        return None
    return None


def _protonation_from_pka(
    pdb_resname: str,
    *,
    ph: float,
    site: IonizableSite | None,
) -> str | None:
    """Pick AMBER variant using PROPKA pKa and Henderson–Hasselbalch."""
    if site is None:
        return None
    res = pdb_resname.upper()
    pka = site.pka
    acidic = site.acidic

    if res in {"ASP", "ASH"}:
        # Protonated (ASH) when pH < pKa for acid.
        return "ASH" if ph < pka else "ASP"
    if res in {"GLU", "GLH"}:
        return "GLH" if ph < pka else "GLU"
    if res in {"LYS", "LYN"}:
        return "LYS" if ph > pka else "LYN"
    if res in {"CYS", "CYM"}:
        return "CYS" if ph < pka else "CYM"
    if res in {"HIS", "HID", "HIE", "HIP"}:
        # Single PROPKA pKa: approximate doubly protonated when well below pKa.
        if ph < pka - 1.0:
            return "HIP"
        if ph > pka + 1.0:
            return "HIE"
        return "HIE"
    if res == "TYR":
        # Neutral TYR in library; no separate protonated entry in amino19.lib index.
        return "TYR"
    return None


def select_amber_residue_name(
    pdb_resname: str,
    res_atoms: list[Atom],
    *,
    ph: float = 7.0,
    pka_site: IonizableSite | None = None,
) -> str:
    """
    Map a PDB residue to an ``amino19.lib`` template name.

    Priority: explicit AMBER names in PDB → hydrogen pattern → PROPKA/pH →
    leaprc default (HIS→HIE).
    """
    res = pdb_resname.upper()
    if res in _PDB_TO_AMBER and res not in {"HIS", "ASP", "GLU", "LYS", "CYS"}:
        return _PDB_TO_AMBER[res]

    names = _atom_names(res_atoms)
    from_h = _protonation_from_hydrogens(res, names)
    if from_h is not None:
        return from_h

    from_pka = _protonation_from_pka(res, ph=ph, site=pka_site)
    if from_pka is not None:
        return from_pka

    if res == "HIS":
        return "HIE"
    if res == "ASP":
        return "ASP"
    if res == "GLU":
        return "GLU"
    if res == "LYS":
        return "LYS"
    if res == "CYS":
        return "CYS"

    return _PDB_TO_AMBER.get(res, res)


def assign_ff19sb_charges(
    structure: ProteinStructure,
    library: Ff19sbLibrary,
    *,
    ph: float = 7.0,
    pka_lookup: PropkaPkaLookup | None = None,
) -> Ff19sbChargeAssignment:
    """
  Assign AMBER ff19SB partial charges to every atom in ``structure``.

  For titratable residues, load hydrogens (``load_pdb(..., include_hydrogen=True)``)
  when possible so protonation matches the PDB; otherwise PROPKA/pH selects the
  library variant.
    """
    grouped = _residue_atoms(structure)
    terminals = _chain_terminal_keys(structure)
    charges = np.zeros(structure.n_atoms, dtype=np.float64)
    amber_residue: dict[tuple[str, int, str], str] = {}
    n_missing = 0
    n_unknown = 0
    terminal_list: list[tuple[str, int, str]] = []
    has_h = any(a.element == "H" for a in structure.atoms)

    atom_to_index = {id(a): i for i, a in enumerate(structure.atoms)}

    for key, res_atoms in grouped.items():
        pdb_resname = key[2]
        site = pka_lookup.sites.get(key) if pka_lookup is not None else None
        try:
            amber_name = select_amber_residue_name(
                pdb_resname,
                res_atoms,
                ph=ph,
                pka_site=site,
            )
            template = library.get(amber_name)
        except KeyError:
            n_unknown += 1
            amber_name = pdb_resname
            amber_residue[key] = amber_name
            continue

        amber_residue[key] = amber_name
        if key in terminals:
            terminal_list.append(key)

        for atom in res_atoms:
            aname = _normalize_atom_name(atom.name)
            q = template.charges.get(aname)
            if q is None:
                n_missing += 1
                continue
            charges[atom_to_index[id(atom)]] = q

    if terminal_list and not has_h:
        warnings.warn(
            "Chain termini detected but aminont12/aminoct12 libraries are not loaded; "
            "using interior ff19SB templates for N/C-terminal residues. "
            "For publication-quality termini, add AmberTools cap libraries.",
            stacklevel=2,
        )

    if not has_h:
        warnings.warn(
            "No hydrogen atoms in structure: titration state uses PROPKA/pH only. "
            "Pass include_hydrogen=True when loading the PDB for best accuracy.",
            stacklevel=2,
        )

    return Ff19sbChargeAssignment(
        charges=charges,
        amber_residue=amber_residue,
        structure=structure,
        library_path=library.source_path,
        n_missing_atoms=n_missing,
        n_unknown_residues=n_unknown,
        terminal_residues=terminal_list,
        hydrogens_in_structure=has_h,
    )
