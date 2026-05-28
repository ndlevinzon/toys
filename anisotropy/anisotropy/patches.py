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
        rows = []
        for p in self.patches:
            rows.append(
                [
                    p.area,
                    p.mean_curvature,
                    p.gaussian_curvature,
                    p.charge,
                    p.potential,
                    p.pka_acid,
                    p.hydropathy,
                    p.polar_density,
                    p.hbond_score,
                    float(np.linalg.norm(p.dipole)),
                    p.softness,
                ]
            )
        return np.asarray(rows, dtype=np.float64)


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
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    n_faces = faces.shape[0]
    for fi in range(n_faces):
        tri = faces[fi]
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])):
            edge = (min(a, b), max(a, b))
            edge_to_faces.setdefault(edge, []).append(fi)
    adj: list[set[int]] = [set() for _ in range(n_faces)]
    for face_list in edge_to_faces.values():
        if len(face_list) < 2:
            continue
        for i in face_list:
            for j in face_list:
                if i != j:
                    adj[i].add(j)
    return [sorted(s) for s in adj]


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
                stack.append(nb)

        if patch_area < min_patch_area and patch_id > 0:
            # Merge small patch into largest neighbor patch.
            neighbors: set[int] = set()
            for f in np.where(face_patch == patch_id)[0]:
                for nb in adj[f]:
                    if face_patch[nb] != patch_id and face_patch[nb] >= 0:
                        neighbors.add(int(face_patch[nb]))
            if neighbors:
                target = max(neighbors, key=lambda pid: (face_patch == pid).sum())
                face_patch[face_patch == patch_id] = target
            else:
                patch_id += 1
                continue
        patch_id += 1

    # Renumber contiguously.
    unique = np.unique(face_patch[face_patch >= 0])
    remap = {old: new for new, old in enumerate(unique)}
    for i in range(face_patch.shape[0]):
        if face_patch[i] >= 0:
            face_patch[i] = remap[int(face_patch[i])]
    return face_patch


def _vertex_curvatures(mesh: ProteinMesh) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex mean and Gaussian curvature via PyVista/VTK."""
    surface = mesh.to_pyvista()
    surface = surface.compute_normals(inplace=False)
    try:
        mean_h = np.asarray(surface.curvature(curv_type="mean"))
        gauss_k = np.asarray(surface.curvature(curv_type="gaussian"))
    except Exception:
        mean_h = np.zeros(surface.n_points)
        gauss_k = np.zeros(surface.n_points)
    return mean_h, gauss_k


def _assign_atoms_to_patches(
    structure: ProteinStructure,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_patch_ids: np.ndarray,
    face_centroids: np.ndarray,
) -> list[list[int]]:
    """For each patch, list indices of atoms whose closest face centroid belongs to that patch."""
    n_patches = int(face_patch_ids.max()) + 1
    tree = cKDTree(face_centroids)
    _, nearest_face = tree.query(structure.centers, k=1)
    patch_atoms: list[list[int]] = [[] for _ in range(n_patches)]
    for ai, face_i in enumerate(nearest_face):
        pid = int(face_patch_ids[int(face_i)])
        patch_atoms[pid].append(ai)
    return patch_atoms


def _aggregate_patch(
    patch_id: int,
    face_mask: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_areas: np.ndarray,
    face_normals: np.ndarray,
    face_centroids: np.ndarray,
    mean_h_vert: np.ndarray,
    gauss_k_vert: np.ndarray,
    atom_indices: list[int],
    structure: ProteinStructure,
    *,
    ph: float,
    atom_bfactor: np.ndarray | None,
    pka_lookup: PropkaPkaLookup | None = None,
    res_atom_counts: dict[tuple[str, int, str], int] | None = None,
    atom_charges: np.ndarray | None = None,
) -> PatchFeatures:
    tri = faces[face_mask]
    areas = face_areas[face_mask]
    a_f = float(areas.sum())
    if a_f < 1e-12:
        a_f = 1e-12

    # Area-weighted centroid and normal.
    w = areas / areas.sum()
    r_f = (w[:, None] * face_centroids[face_mask]).sum(axis=0)
    n_raw = (w[:, None] * face_normals[face_mask]).sum(axis=0)
    n_norm = np.linalg.norm(n_raw)
    n_hat = n_raw / (n_norm + 1e-12)

    # Curvature: average over patch vertices.
    vert_ids = np.unique(tri.ravel())
    H_f = float(np.mean(mean_h_vert[vert_ids]))
    K_f = float(np.mean(gauss_k_vert[vert_ids]))

    # Chemistry from assigned atoms.
    q_f = 0.0
    phi_f = 0.0
    pka_vals: list[float] = []
    hydro_vals: list[float] = []
    polar_count = 0
    donors = 0
    acceptors = 0
    mu = np.zeros(3, dtype=np.float64)
    b_sum = 0.0
    n_atoms = len(atom_indices)
    seen_residue: set[tuple[str, int, str]] = set()

    for ai in atom_indices:
        atom = structure.atoms[ai]
        if atom_charges is not None:
            q_atom = float(atom_charges[ai])
        else:
            q_atom = atom_charge_at_ph(
                atom,
                ph,
                pka_lookup=pka_lookup,
                residue_atom_counts=res_atom_counts,
            )
        q_f += q_atom
        dist = max(float(np.linalg.norm(atom.xyz - r_f)), 1.0)
        phi_f += COULOMB_SCALE * q_atom / dist
        rkey = residue_key(atom)
        if rkey not in seen_residue:
            seen_residue.add(rkey)
            pka = pka_for_patch_atom(atom, pka_lookup=pka_lookup)
            if pka is not None:
                pka_vals.append(pka)
        hp = residue_hydropathy(atom.resname)
        hydro_vals.append(hp)
        if atom.resname in POLAR_RESIDUES:
            polar_count += 1
        d, a = hbond_donors_acceptors(atom.resname)
        donors += d
        acceptors += a
        mu += q_atom * (atom.xyz - r_f)
        if atom_bfactor is not None:
            b_sum += float(atom_bfactor[ai])

    pka_a = float(np.mean(pka_vals)) if pka_vals else 7.0
    h_scalar = float(np.mean(hydro_vals)) if hydro_vals else 0.0
    # Directional hydropathy: hydrophobic character projected along outward normal.
    # Positive => hydrophobic face pointing outward (favorable for AWI in simple models).
    h_f = h_scalar * float(np.linalg.norm(n_hat))
    p_f = float(polar_count) / max(n_atoms, 1)
    b_f = float(donors + acceptors) / max(n_atoms, 1)

    if atom_bfactor is not None and n_atoms > 0:
        u_f = float(b_sum / n_atoms) / 100.0
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
        face_indices=np.where(face_mask)[0],
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
        if path is not None:
            chem_structure = load_pdb(path, include_hydrogen=True)
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

    patches: list[PatchFeatures] = []
    for pid in range(n_patches):
        mask = face_patch_ids == pid
        patches.append(
            _aggregate_patch(
                pid,
                mask,
                mesh.vertices,
                mesh.faces,
                face_areas,
                face_normals,
                face_centroids,
                mean_h,
                gauss_k,
                patch_atoms[pid],
                chem_structure,
                ph=ph,
                atom_bfactor=patch_bfactor,
                pka_lookup=pka_lookup,
                res_atom_counts=res_atom_counts,
                atom_charges=atom_charges,
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
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    mode = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("element vertex"):
                mode = "v"
                continue
            if line.startswith("element face"):
                mode = "f"
                continue
            if line == "end_header":
                mode = "data"
                continue
            if mode == "data":
                parts = line.split()
                if len(parts) == 3:
                    verts.append([float(x) for x in parts])
                elif len(parts) == 4 and parts[0] == "3":
                    faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
    return ProteinMesh(
        vertices=np.asarray(verts, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
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
