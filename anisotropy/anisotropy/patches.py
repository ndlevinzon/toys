"""
Patch-wise parameterization of a protein SAS mesh.

Each patch f carries a feature vector inspired by interfacial determinants:

    x_f = (a_f, r_f, n_hat_f, H_f, K_f, q_f, phi_f, pKa_a,f, h_f, p_f,
           b_f, mu_f, u_f)

Geometry is local (area, normal, curvature). Chemistry is assigned by
projecting PDB atoms onto the nearest surface patch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from anisotropy.ff19sb import Ff19sbChargeAssignment, assign_ff19sb_charges, default_ff19sb_library
from anisotropy.curvature import vertex_curvatures
from anisotropy.mesh import ProteinMesh
from anisotropy.pdb import Atom, ProteinStructure, load_pdb
from anisotropy.propka_pka import (
    PropkaPkaLookup,
    atom_charge_at_ph,
    pka_for_patch_atom,
    residue_atom_counts,
    residue_key,
)
from anisotropy.residue_chemistry import (
    POLAR_RESIDUES,
    hbond_donors_acceptors,
    residue_charge_at_ph,
    residue_hydropathy,
)

# Coulomb constant in vacuum (kJ·mol⁻¹·nm·e⁻²); distances in Å → scale.
COULOMB_SCALE = 332.0636  # kcal·mol⁻¹·Å·e⁻² (educational magnitude)

# Columns of :meth:`PatchParameterization.feature_matrix` / ``feature_matrix`` in .npz.
FEATURE_MATRIX_NAMES: tuple[str, ...] = (
    "area",
    "mean_curvature",
    "gaussian_curvature",
    "charge",
    "potential",
    "pka_acid",
    "hydropathy",
    "polar_density",
    "hbond_score",
    "dipole_magnitude",
    "softness",
)


@dataclass
class PatchFeatures:
    """
    Local patch descriptor x_f for interface / orientation modeling.

    All vectors are in the same lab frame as the mesh (Å).
    """

    patch_id: int
    area: float  # a_f (Å²)
    centroid: np.ndarray  # r_f (3,)
    normal: np.ndarray  # n_hat_f (3,) unit, outward
    mean_curvature: float  # H_f
    gaussian_curvature: float  # K_f
    charge: float  # q_f (e)
    potential: float  # phi_f (arbitrary units, bulk-referenced)
    pka_acid: float  # pKa_a,f — mean effective pKa of titratable groups
    hydropathy: float  # h_f — directional hydrophobic presentation
    polar_density: float  # p_f
    hbond_score: float  # b_f
    dipole: np.ndarray  # mu_f (3,)
    softness: float  # u_f
    face_indices: np.ndarray = field(repr=False)  # indices into mesh.faces
    n_atoms: int = 0

    def as_vector_dict(self) -> dict[str, float | list[float]]:
        """Serialize scalars + dipole for JSON/npz."""
        return {
            "patch_id": self.patch_id,
            "area": self.area,
            "centroid": self.centroid.tolist(),
            "normal": self.normal.tolist(),
            "mean_curvature": self.mean_curvature,
            "gaussian_curvature": self.gaussian_curvature,
            "charge": self.charge,
            "potential": self.potential,
            "pka_acid": self.pka_acid,
            "hydropathy": self.hydropathy,
            "polar_density": self.polar_density,
            "hbond_score": self.hbond_score,
            "dipole": self.dipole.tolist(),
            "softness": self.softness,
            "n_atoms": self.n_atoms,
        }


@dataclass
class PatchParameterization:
    """Full patch decomposition of one mesh + structure."""

    patches: list[PatchFeatures]
    face_patch_ids: np.ndarray  # (n_faces,) int
    ph: float
    metadata: dict = field(default_factory=dict)

    @property
    def n_patches(self) -> int:
        return len(self.patches)

    def feature_matrix(self) -> np.ndarray:
        """
        Compact table (n_patches, 14): area, H, K, q, phi, pKa, h, p, b, |mu|, u
        plus normal (3) and centroid (3) stored separately in npz export.
        """
        n = self.n_patches
        if n == 0:
            return np.zeros((0, 11), dtype=np.float64)
        out = np.empty((n, 11), dtype=np.float64)
        for i, p in enumerate(self.patches):
            out[i, 0] = p.area
            out[i, 1] = p.mean_curvature
            out[i, 2] = p.gaussian_curvature
            out[i, 3] = p.charge
            out[i, 4] = p.potential
            out[i, 5] = p.pka_acid
            out[i, 6] = p.hydropathy
            out[i, 7] = p.polar_density
            out[i, 8] = p.hbond_score
            out[i, 9] = float(np.linalg.norm(p.dipole))
            out[i, 10] = p.softness
        return out


def _face_geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-face area, unit normal, centroid."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = np.zeros_like(cross)
    valid = areas > 1e-12
    normals[valid] = cross[valid] / (2.0 * areas[valid, None])
    return areas, normals, centroids


def _build_face_adjacency(faces: np.ndarray) -> list[list[int]]:
    """Two faces adjacent if they share an edge."""
    n_faces = int(faces.shape[0])
    if n_faces == 0:
        return []
    f = np.asarray(faces, dtype=np.int64)
    v0, v1, v2 = f[:, 0], f[:, 1], f[:, 2]
    face_id = np.arange(n_faces, dtype=np.int64)
    e_a = np.concatenate([v0, v1, v0])
    e_b = np.concatenate([v1, v2, v2])
    fid = np.concatenate([face_id, face_id, face_id])
    lo = np.minimum(e_a, e_b)
    hi = np.maximum(e_a, e_b)
    max_v = int(hi.max()) + 1
    keys = lo.astype(np.int64) * max_v + hi.astype(np.int64)
    order = np.argsort(keys, kind="mergesort")
    keys_s = keys[order]
    fid_s = fid[order]
    adj: list[list[int]] = [[] for _ in range(n_faces)]
    breaks = np.concatenate(
        [[0], np.flatnonzero(keys_s[1:] != keys_s[:-1]) + 1, [keys_s.size]]
    )
    for b in range(int(breaks.size) - 1):
        s, e = int(breaks[b]), int(breaks[b + 1])
        group = fid_s[s:e]
        if group.size < 2:
            continue
        for i in group:
            ii = int(i)
            for j in group:
                jj = int(j)
                if jj != ii:
                    adj[ii].append(jj)
    for i in range(n_faces):
        if adj[i]:
            adj[i] = sorted(set(adj[i]))
    return adj


def segment_mesh_patches(
    mesh: ProteinMesh,
    *,
    normal_angle_deg: float = 25.0,
    min_patch_area: float = 50.0,
) -> np.ndarray:
    """
    Region-growing on the triangle mesh: merge adjacent faces with similar normals.

    Returns ``face_patch_ids`` (n_faces,) int.
    """
    vertices = mesh.vertices
    faces = mesh.faces
    areas, normals, _ = _face_geometry(vertices, faces)
    adj = _build_face_adjacency(faces)
    cos_thresh = float(np.cos(np.deg2rad(normal_angle_deg)))

    face_patch = np.full(faces.shape[0], -1, dtype=np.int32)
    patch_id = 0
    order = np.argsort(-areas)

    for seed in order:
        if face_patch[seed] >= 0:
            continue
        stack = [int(seed)]
        patch_faces: list[int] = [int(seed)]
        face_patch[seed] = patch_id
        patch_area = float(areas[seed])
        seed_n = normals[seed]

        while stack:
            f = stack.pop()
            for nb in adj[f]:
                if face_patch[nb] >= 0:
                    continue
                if float(np.dot(normals[nb], seed_n)) < cos_thresh:
                    continue
                face_patch[nb] = patch_id
                patch_area += float(areas[nb])
                patch_faces.append(int(nb))
                stack.append(nb)

        if patch_area < min_patch_area and patch_id > 0:
            # Merge small patch into largest neighbor patch.
            neighbors: set[int] = set()
            for f in patch_faces:
                for nb in adj[f]:
                    nb_pid = int(face_patch[nb])
                    if nb_pid != patch_id and nb_pid >= 0:
                        neighbors.add(nb_pid)
            if neighbors:
                target = max(neighbors, key=lambda pid: int(np.sum(face_patch == pid)))
                face_patch[face_patch == patch_id] = target
            else:
                patch_id += 1
                continue
        patch_id += 1

    return np.unique(face_patch, return_inverse=True)[1].astype(np.int32)


def _vertex_curvatures(mesh: ProteinMesh) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex mean and Gaussian curvature (PyVista if available, else discrete)."""
    return vertex_curvatures(mesh.vertices, mesh.faces)


def _assign_atoms_to_patches(
    structure: ProteinStructure,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_patch_ids: np.ndarray,
    face_centroids: np.ndarray,
) -> list[np.ndarray]:
    """For each patch, atom indices whose closest face centroid belongs to that patch."""
    n_patches = int(face_patch_ids.max()) + 1
    tree = cKDTree(face_centroids)
    _, nearest_face = tree.query(structure.centers, k=1)
    atom_patch = face_patch_ids[np.asarray(nearest_face, dtype=np.int64)]
    patch_atoms: list[np.ndarray] = []
    atom_ids = np.arange(structure.n_atoms, dtype=np.int64)
    for pid in range(n_patches):
        patch_atoms.append(atom_ids[atom_patch == pid])
    return patch_atoms


def _group_faces_by_patch(
    face_patch_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sorted face indices and slice boundaries per contiguous patch id."""
    order = np.argsort(face_patch_ids, kind="mergesort")
    sorted_pids = face_patch_ids[order]
    unique_pids, starts, counts = np.unique(
        sorted_pids, return_index=True, return_counts=True
    )
    ends = starts + counts
    return unique_pids.astype(np.int32), order, ends


def _atom_property_arrays(
    structure: ProteinStructure,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-atom hydropathy, polar mask, H-bond donor/acceptor counts (one pass)."""
    n = structure.n_atoms
    hydro = np.empty(n, dtype=np.float64)
    polar = np.zeros(n, dtype=bool)
    donors = np.zeros(n, dtype=np.int32)
    acceptors = np.zeros(n, dtype=np.int32)
    for i, atom in enumerate(structure.atoms):
        res = atom.resname.upper()
        hydro[i] = residue_hydropathy(res)
        polar[i] = res in POLAR_RESIDUES
        d, a = hbond_donors_acceptors(res)
        donors[i] = d
        acceptors[i] = a
    return hydro, polar, donors, acceptors


def _precompute_atom_charges(
    structure: ProteinStructure,
    ph: float,
    *,
    pka_lookup: PropkaPkaLookup | None,
    res_atom_counts: dict[tuple[str, int, str], int],
) -> np.ndarray:
    """Residue-aware per-atom charges (same rules as :func:`atom_charge_at_ph`)."""
    charges = np.zeros(structure.n_atoms, dtype=np.float64)
    res_q: dict[tuple[str, int, str], float] = {}
    for i, atom in enumerate(structure.atoms):
        key = residue_key(atom)
        if key not in res_q:
            if pka_lookup is not None:
                q_res = pka_lookup.residue_charge(atom, ph)
                if q_res is None:
                    q_res = residue_charge_at_ph(atom.resname, ph)
            else:
                q_res = residue_charge_at_ph(atom.resname, ph)
            res_q[key] = float(q_res)
        n_in_res = max(res_atom_counts.get(key, 1), 1)
        charges[i] = res_q[key] / n_in_res
    return charges


def _aggregate_patch(
    patch_id: int,
    face_idx: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_areas: np.ndarray,
    face_normals: np.ndarray,
    face_centroids: np.ndarray,
    mean_h_vert: np.ndarray,
    gauss_k_vert: np.ndarray,
    atom_indices: np.ndarray,
    structure: ProteinStructure,
    *,
    ph: float,
    atom_bfactor: np.ndarray | None,
    pka_lookup: PropkaPkaLookup | None = None,
    res_atom_counts: dict[tuple[str, int, str], int] | None = None,
    atom_charges: np.ndarray | None = None,
    atom_hydro: np.ndarray | None = None,
    atom_polar: np.ndarray | None = None,
    atom_donors: np.ndarray | None = None,
    atom_acceptors: np.ndarray | None = None,
    centers: np.ndarray | None = None,
) -> PatchFeatures:
    tri = faces[face_idx]
    areas = face_areas[face_idx]
    a_f = float(areas.sum())
    if a_f < 1e-12:
        a_f = 1e-12

    # Area-weighted centroid and normal.
    w = areas / areas.sum()
    r_f = (w[:, None] * face_centroids[face_idx]).sum(axis=0)
    n_raw = (w[:, None] * face_normals[face_idx]).sum(axis=0)
    n_norm = np.linalg.norm(n_raw)
    n_hat = n_raw / (n_norm + 1e-12)

    # Curvature: average over patch vertices.
    vert_ids = np.unique(tri.ravel())
    H_f = float(np.mean(mean_h_vert[vert_ids]))
    K_f = float(np.mean(gauss_k_vert[vert_ids]))

    ai = np.asarray(atom_indices, dtype=np.int64)
    n_atoms = int(ai.size)
    if centers is None:
        centers = structure.centers

    if n_atoms == 0:
        q_f = 0.0
        phi_f = 0.0
        pka_a = 7.0
        h_f = 0.0
        p_f = 0.0
        b_f = 0.0
        mu = np.zeros(3, dtype=np.float64)
        u_f = float(min(1.0, abs(H_f) * 2.0 + 0.3))
    else:
        if atom_charges is not None:
            q_atoms = atom_charges[ai]
        else:
            assert res_atom_counts is not None
            q_atoms = np.array(
                [
                    atom_charge_at_ph(
                        structure.atoms[int(i)],
                        ph,
                        pka_lookup=pka_lookup,
                        residue_atom_counts=res_atom_counts,
                    )
                    for i in ai
                ],
                dtype=np.float64,
            )
        q_f = float(q_atoms.sum())
        xyz = centers[ai]
        dist = np.linalg.norm(xyz - r_f, axis=1)
        phi_f = float((COULOMB_SCALE * q_atoms / np.maximum(dist, 1.0)).sum())
        mu = (q_atoms[:, None] * (xyz - r_f)).sum(axis=0)

        if atom_hydro is not None:
            hydro_vals = atom_hydro[ai]
        else:
            hydro_vals = np.array(
                [residue_hydropathy(structure.atoms[int(i)].resname) for i in ai],
                dtype=np.float64,
            )
        if atom_polar is not None:
            polar_count = int(atom_polar[ai].sum())
        else:
            polar_count = sum(
                1 for i in ai if structure.atoms[int(i)].resname in POLAR_RESIDUES
            )
        if atom_donors is not None and atom_acceptors is not None:
            donors = int(atom_donors[ai].sum())
            acceptors = int(atom_acceptors[ai].sum())
        else:
            donors = 0
            acceptors = 0
            for i in ai:
                d, a = hbond_donors_acceptors(structure.atoms[int(i)].resname)
                donors += d
                acceptors += a

        pka_vals: list[float] = []
        seen: set[tuple[str, int, str]] = set()
        for i in ai:
            atom = structure.atoms[int(i)]
            rkey = residue_key(atom)
            if rkey in seen:
                continue
            seen.add(rkey)
            pka = pka_for_patch_atom(atom, pka_lookup=pka_lookup)
            if pka is not None:
                pka_vals.append(pka)

        pka_a = float(np.mean(pka_vals)) if pka_vals else 7.0
        h_scalar = float(np.mean(hydro_vals)) if hydro_vals.size else 0.0
        h_f = h_scalar * float(np.linalg.norm(n_hat))
        p_f = float(polar_count) / n_atoms
        b_f = float(donors + acceptors) / n_atoms

        if atom_bfactor is not None:
            u_f = float(atom_bfactor[ai].sum() / n_atoms) / 100.0
        else:
            u_f = float(min(1.0, abs(H_f) * 2.0 + (1.0 - min(p_f, 1.0)) * 0.3))

    return PatchFeatures(
        patch_id=patch_id,
        area=a_f,
        centroid=r_f,
        normal=n_hat,
        mean_curvature=H_f,
        gaussian_curvature=K_f,
        charge=q_f,
        potential=phi_f,
        pka_acid=pka_a,
        hydropathy=h_f,
        polar_density=p_f,
        hbond_score=b_f,
        dipole=mu,
        softness=u_f,
        face_indices=face_idx,
        n_atoms=n_atoms,
    )


def parameterize_mesh(
    mesh: ProteinMesh,
    structure: ProteinStructure,
    *,
    ph: float = 7.0,
    normal_angle_deg: float = 25.0,
    min_patch_area: float = 50.0,
    atom_bfactor: np.ndarray | None = None,
    pka_source: str = "auto",
    pdb_path: str | None = None,
    charge_model: str = "ff19sb",
    ff19sb_lib: str | None = None,
) -> PatchParameterization:
    """
    Build patch features x_f for every surface patch on ``mesh``.

    Parameters
    ----------
    mesh
        SAS triangle mesh.
    structure
        Parsed PDB (same frame as mesh).
    ph
        pH for protonation-aware charge features.
    normal_angle_deg
        Max angle between face normals in one patch.
    min_patch_area
        Merge patches smaller than this (Å²).
    atom_bfactor
        Optional per-atom B-factors from PDB (same order as structure.atoms).
    pka_source
        ``"auto"`` (PROPKA if available, else tables), ``"propka"``, or ``"table"``.
    pdb_path
        PDB path for PROPKA; defaults to ``structure.source_path``.
    charge_model
        ``"ff19sb"`` (AMBER partial charges from ``amino19.lib``), ``"propka"``,
        or ``"table"`` (legacy uniform charge per heavy atom).
    ff19sb_lib
        Path to ``amino19.lib`` (default: bundled ff19SB force field files).
    """
    if charge_model not in ("ff19sb", "propka", "table"):
        raise ValueError('charge_model must be "ff19sb", "propka", or "table"')

    pka_lookup: PropkaPkaLookup | None = None
    if pka_source not in ("auto", "propka", "table"):
        raise ValueError('pka_source must be "auto", "propka", or "table"')

    if pka_source in ("auto", "propka"):
        path = pdb_path or structure.source_path
        if path is None:
            if pka_source == "propka":
                raise ValueError("pka_source='propka' requires a PDB path on the structure")
        elif pka_source == "propka":
            pka_lookup = PropkaPkaLookup.from_pdb(path)
        else:
            pka_lookup = PropkaPkaLookup.try_from_pdb(path)

    chem_structure = structure
    atom_charges: np.ndarray | None = None
    ff19_assignment: Ff19sbChargeAssignment | None = None

    patch_bfactor = atom_bfactor
    if charge_model == "ff19sb":
        lib = default_ff19sb_library(ff19sb_lib)
        path = pdb_path or structure.source_path
        has_h = any(a.element == "H" for a in structure.atoms)
        if path is not None and not has_h:
            chem_structure = load_pdb(path, include_hydrogen=True)
        elif has_h:
            chem_structure = structure
        if (
            atom_bfactor is not None
            and chem_structure is not structure
            and len(atom_bfactor) == structure.n_atoms
        ):
            serial_to_b = {
                a.serial: float(atom_bfactor[i]) for i, a in enumerate(structure.atoms)
            }
            patch_bfactor = np.array(
                [serial_to_b.get(a.serial, np.nan) for a in chem_structure.atoms],
                dtype=np.float64,
            )
        ff19_assignment = assign_ff19sb_charges(
            chem_structure,
            lib,
            ph=ph,
            pka_lookup=pka_lookup,
        )
        atom_charges = ff19_assignment.charges
    elif charge_model == "propka":
        pass  # uses pka_lookup in _aggregate_patch
    # "table" uses tabulated residue pKa only

    res_atom_counts = residue_atom_counts(chem_structure)
    if atom_charges is None and charge_model != "ff19sb":
        atom_charges = _precompute_atom_charges(
            chem_structure,
            ph,
            pka_lookup=pka_lookup,
            res_atom_counts=res_atom_counts,
        )

    atom_hydro, atom_polar, atom_donors, atom_acceptors = _atom_property_arrays(
        chem_structure
    )
    atom_centers = chem_structure.centers

    face_patch_ids = segment_mesh_patches(
        mesh,
        normal_angle_deg=normal_angle_deg,
        min_patch_area=min_patch_area,
    )
    n_patches = int(face_patch_ids.max()) + 1
    face_areas, face_normals, face_centroids = _face_geometry(mesh.vertices, mesh.faces)
    mean_h, gauss_k = _vertex_curvatures(mesh)
    patch_atoms = _assign_atoms_to_patches(
        chem_structure, mesh.vertices, mesh.faces, face_patch_ids, face_centroids
    )

    unique_pids, face_order, patch_ends = _group_faces_by_patch(face_patch_ids)
    patches: list[PatchFeatures] = []
    start = 0
    for out_i, pid in enumerate(unique_pids):
        end = int(patch_ends[out_i])
        face_idx = face_order[start:end]
        start = end
        patches.append(
            _aggregate_patch(
                int(pid),
                face_idx,
                mesh.vertices,
                mesh.faces,
                face_areas,
                face_normals,
                face_centroids,
                mean_h,
                gauss_k,
                patch_atoms[int(pid)],
                chem_structure,
                ph=ph,
                atom_bfactor=patch_bfactor,
                pka_lookup=pka_lookup,
                res_atom_counts=res_atom_counts,
                atom_charges=atom_charges,
                atom_hydro=atom_hydro,
                atom_polar=atom_polar,
                atom_donors=atom_donors,
                atom_acceptors=atom_acceptors,
                centers=atom_centers,
            )
        )

    meta: dict = {
        "n_faces": mesh.n_faces,
        "n_vertices": mesh.n_vertices,
        "normal_angle_deg": normal_angle_deg,
        "min_patch_area": min_patch_area,
        "pka_source": pka_source,
        "pka_backend": "propka" if pka_lookup is not None else "table",
        "n_propka_sites": len(pka_lookup.sites) if pka_lookup else 0,
        "charge_model": charge_model,
    }
    if ff19_assignment is not None:
        meta.update(
            {
                "ff19sb_library": ff19_assignment.library_path,
                "ff19sb_total_charge_e": ff19_assignment.total_charge,
                "ff19sb_hydrogens": ff19_assignment.hydrogens_in_structure,
                "ff19sb_missing_atom_mappings": ff19_assignment.n_missing_atoms,
                "ff19sb_unknown_residues": ff19_assignment.n_unknown_residues,
                "ff19sb_n_terminal_residues": len(ff19_assignment.terminal_residues),
            }
        )

    return PatchParameterization(
        patches=patches,
        face_patch_ids=face_patch_ids,
        ph=ph,
        metadata=meta,
    )


def load_mesh_ply(path: str) -> ProteinMesh:
    """Minimal PLY loader for meshes written by ``ProteinMesh.save_ply``."""
    path = str(path)
    with open(path, encoding="utf-8") as handle:
        lines = [ln.strip() for ln in handle]

    n_verts = 0
    n_faces = 0
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            n_verts = int(line.split()[-1])
        elif line.startswith("element face"):
            n_faces = int(line.split()[-1])
        elif line == "end_header":
            data_start = i + 1
            break

    body = lines[data_start : data_start + n_verts + n_faces]
    vert_lines = body[:n_verts]
    face_lines = body[n_verts : n_verts + n_faces]

    if n_verts:
        verts = np.fromstring(
            " ".join(vert_lines), sep=" ", dtype=np.float64
        ).reshape(n_verts, 3)
    else:
        verts = np.zeros((0, 3), dtype=np.float64)

    if n_faces:
        face_data = np.fromstring(
            " ".join(face_lines), sep=" ", dtype=np.int64
        ).reshape(n_faces, 4)
        if not np.all(face_data[:, 0] == 3):
            raise ValueError(f"Expected triangular faces in {path}")
        faces = face_data[:, 1:4]
    else:
        faces = np.zeros((0, 3), dtype=np.int64)

    return ProteinMesh(
        vertices=verts,
        faces=faces,
        resolution_angstrom=0.0,
        probe_radius=1.4,
    )


def save_parameterization(path: str, param: PatchParameterization) -> None:
    """Write patch table to compressed .npz."""
    n = param.n_patches
    centroids = np.stack([p.centroid for p in param.patches], axis=0)
    normals = np.stack([p.normal for p in param.patches], axis=0)
    dipoles = np.stack([p.dipole for p in param.patches], axis=0)
    np.savez_compressed(
        path,
        feature_matrix=param.feature_matrix(),
        centroids=centroids,
        normals=normals,
        dipoles=dipoles,
        face_patch_ids=param.face_patch_ids,
        ph=np.float32(param.ph),
        patch_ids=np.arange(n, dtype=np.int32),
        feature_names=np.array(FEATURE_MATRIX_NAMES, dtype="U32"),
    )


def load_parameterization(path: str) -> PatchParameterization:
    """Load patch table written by :func:`save_parameterization`."""
    data = np.load(path, allow_pickle=False)
    fm = np.asarray(data["feature_matrix"], dtype=np.float64)
    n = int(fm.shape[0])
    centroids = np.asarray(data["centroids"], dtype=np.float64)
    normals = np.asarray(data["normals"], dtype=np.float64)
    dipoles = np.asarray(data["dipoles"], dtype=np.float64)
    face_patch_ids = np.asarray(data["face_patch_ids"], dtype=np.int64)
    ph = float(np.asarray(data["ph"]).reshape(()))

    patches: list[PatchFeatures] = []
    for i in range(n):
        row = fm[i]
        mu = dipoles[i]
        patches.append(
            PatchFeatures(
                patch_id=i,
                area=float(row[0]),
                centroid=centroids[i],
                normal=normals[i],
                mean_curvature=float(row[1]),
                gaussian_curvature=float(row[2]),
                charge=float(row[3]),
                potential=float(row[4]),
                pka_acid=float(row[5]),
                hydropathy=float(row[6]),
                polar_density=float(row[7]),
                hbond_score=float(row[8]),
                dipole=mu,
                softness=float(row[10]),
                face_indices=np.array([], dtype=np.int64),
            )
        )

    meta: dict = {"source_npz": str(path)}
    if "feature_names" in data:
        meta["feature_names"] = [str(x) for x in data["feature_names"]]

    return PatchParameterization(
        patches=patches,
        face_patch_ids=face_patch_ids,
        ph=ph,
        metadata=meta,
    )
