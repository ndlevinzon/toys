"""
Iterative SAS mesh fitting and surface smoothing.

Each refinement pass rebuilds Φ at a finer grid resolution, extracts the
Φ = 0 isosurface, then applies Laplacian smoothing so later force-based
models can use a clean triangle mesh.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from skimage import measure

from anisotropy.pdb import ProteinStructure
from anisotropy.sasa import (
    DEFAULT_PROBE_RADIUS,
    DistanceFieldGrid,
    build_sasa_distance_field,
    build_sasa_distance_field_auto,
    build_sasa_signed_distance_field_voxel,
    refine_distance_field,
)


@dataclass
class ProteinMesh:
    """Triangle mesh approximating the protein SAS envelope."""

    vertices: np.ndarray  # (V, 3)
    faces: np.ndarray  # (F, 3) int
    resolution_angstrom: float
    probe_radius: float
    refinement_level: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    def center_of_mass(self) -> np.ndarray:
        return self.vertices.mean(axis=0)

    def to_pyvista(self):
        """Optional PyVista PolyData for visualization."""
        import pyvista as pv

        faces_pv = np.hstack(
            [np.full((self.n_faces, 1), 3, dtype=np.int64), self.faces.astype(np.int64)]
        ).ravel()
        return pv.PolyData(self.vertices, faces_pv)

    def save_ply(self, path: str) -> None:
        """Write ASCII PLY (vertices + triangular faces)."""
        path = str(path)
        v = self.vertices
        f = self.faces.astype(np.int64)
        with open(path, "w", encoding="utf-8") as out:
            out.write("ply\nformat ascii 1.0\n")
            out.write(f"element vertex {v.shape[0]}\n")
            out.write("property float x\nproperty float y\nproperty float z\n")
            out.write(f"element face {f.shape[0]}\n")
            out.write("property list uchar int vertex_indices\n")
            out.write("end_header\n")
            for row in v:
                out.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
            for tri in f:
                out.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def _vertex_neighbors(n_vertices: int, faces: np.ndarray) -> list[list[int]]:
    nbrs: list[set[int]] = [set() for _ in range(n_vertices)]
    for a, b, c in faces:
        nbrs[a].update((b, c))
        nbrs[b].update((a, c))
        nbrs[c].update((a, b))
    return [sorted(s) for s in nbrs]


def _neighbor_index_table(
    neighbors: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Pad neighbor lists to (n_vertices, max_degree) for vectorized gathers."""
    n_vertices = len(neighbors)
    max_deg = max((len(n) for n in neighbors), default=0)
    if max_deg == 0:
        return np.zeros((n_vertices, 0), dtype=np.int64), np.zeros(n_vertices, dtype=bool)
    nbr_idx = np.zeros((n_vertices, max_deg), dtype=np.int64)
    mask = np.zeros((n_vertices, max_deg), dtype=bool)
    for i, idx in enumerate(neighbors):
        if not idx:
            continue
        nbr_idx[i, : len(idx)] = idx
        mask[i, : len(idx)] = True
    return nbr_idx, mask


def laplacian_smooth(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int = 4,
    lam: float = 0.45,
) -> np.ndarray:
    """Uniform Laplacian smoothing (does not change connectivity)."""
    if iterations < 1:
        return vertices.copy()
    neighbors = _vertex_neighbors(vertices.shape[0], faces)
    nbr_idx, mask = _neighbor_index_table(neighbors)
    v = vertices.astype(np.float64)
    if nbr_idx.shape[1] == 0:
        return v
    for _ in range(iterations):
        gathered = v[nbr_idx]  # (n, max_deg, 3)
        gathered = np.where(mask[..., None], gathered, 0.0)
        counts = mask.sum(axis=1, dtype=np.float64)
        mean_nbr = gathered.sum(axis=1) / np.maximum(counts, 1.0)[:, None]
        isolated = counts < 1.0
        mean_nbr[isolated] = v[isolated]
        v = v + lam * (mean_nbr - v)
    return v.astype(np.float64)


def extract_isosurface_mesh(
    grid: DistanceFieldGrid,
    *,
    level: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Marching cubes on Φ; returns (vertices, faces) in world coordinates."""
    try:
        verts, faces, _normals, _vals = measure.marching_cubes(
            grid.values,
            level=level,
            spacing=(grid.spacing, grid.spacing, grid.spacing),
        )
    except (ValueError, RuntimeError) as exc:
        raise ValueError(
            "Isosurface extraction failed; try coarser resolution or a larger structure."
        ) from exc

    verts = verts + grid.origin.reshape(1, 3)
    # skimage returns faces as (F, 3)
    return verts.astype(np.float64), faces.astype(np.int64)


def mesh_from_distance_field(
    grid: DistanceFieldGrid,
    *,
    smooth_iterations: int = 3,
    smooth_lambda: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    """Φ = 0 surface plus optional Laplacian smooth."""
    grid = refine_distance_field(grid)
    verts, faces = extract_isosurface_mesh(grid, level=0.0)
    if smooth_iterations > 0 and verts.shape[0] > 0:
        verts = laplacian_smooth(verts, faces, iterations=smooth_iterations, lam=smooth_lambda)
    return verts, faces


def fit_iterative_mesh(
    structure: ProteinStructure,
    *,
    resolutions: tuple[float, ...] = (2.5, 1.5, 1.0),
    probe_radius: float = DEFAULT_PROBE_RADIUS,
    smooth_iterations: tuple[int, ...] | None = None,
    padding: float = 6.0,
    method: str = "auto",
) -> ProteinMesh:
    """
    Iteratively fit a SAS mesh by refining grid resolution.

    Coarse passes capture global shape; fine passes add detail. Smoothing
    increases slightly on later passes to keep the mesh watertight enough
    for downstream mechanics.
    """
    if not resolutions:
        raise ValueError("resolutions must be non-empty")

    if smooth_iterations is None:
        smooth_iterations = tuple(2 + i for i in range(len(resolutions)))
    if len(smooth_iterations) != len(resolutions):
        raise ValueError("smooth_iterations must match resolutions length")

    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None

    for level, (res, n_smooth) in enumerate(zip(resolutions, smooth_iterations)):
        if method == "exact":
            grid = build_sasa_distance_field(
                structure,
                resolution=res,
                probe_radius=probe_radius,
                padding=padding,
            )
        elif method == "voxel":
            grid = build_sasa_signed_distance_field_voxel(
                structure,
                resolution=res,
                probe_radius=probe_radius,
                padding=padding,
            )
        elif method == "auto":
            grid = build_sasa_distance_field_auto(
                structure,
                resolution=res,
                probe_radius=probe_radius,
                padding=padding,
            )
        else:
            raise ValueError("method must be one of: auto, exact, voxel")
        vertices, faces = mesh_from_distance_field(
            grid,
            smooth_iterations=n_smooth,
        )
        if vertices.shape[0] < 4 or faces.shape[0] < 2:
            raise ValueError(
                f"Mesh extraction produced a degenerate mesh at resolution {res} Å"
            )

    assert vertices is not None and faces is not None
    return ProteinMesh(
        vertices=vertices,
        faces=faces,
        resolution_angstrom=float(resolutions[-1]),
        probe_radius=float(probe_radius),
        refinement_level=len(resolutions) - 1,
        metadata={
            "resolutions": list(resolutions),
            "n_atoms": structure.n_atoms,
            "source": structure.source_path,
            "method": method,
        },
    )
