"""
Discrete per-vertex curvatures on triangle meshes (no VTK / PyVista).

Uses standard cotangent Laplacian mean curvature and angle-deficit Gaussian
curvature (Meyer et al., *Discrete Differential-Geometry Operators*). Suitable
for headless HPC when PyVista is not installed.
"""

from __future__ import annotations

import numpy as np


def _cotangent_weight(
    opp: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """Cotangent of the angle at ``opp`` in triangle (opp, a, b)."""
    u = a - opp
    v = b - opp
    cross_norm = np.linalg.norm(np.cross(u, v))
    if cross_norm < 1e-14:
        return 0.0
    return float(np.dot(u, v) / cross_norm)


def _vertex_incidence(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[list[list[int]], np.ndarray]:
    """Neighbor lists (unique) and per-vertex mixed Voronoi area (1/3 triangle sum)."""
    n_verts = int(vertices.shape[0])
    f = np.asarray(faces, dtype=np.int64)
    nbrs: list[set[int]] = [set() for _ in range(n_verts)]
    area_mix = np.zeros(n_verts, dtype=np.float64)

    v0 = vertices[f[:, 0]]
    v1 = vertices[f[:, 1]]
    v2 = vertices[f[:, 2]]
    tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    for fi in range(f.shape[0]):
        tri = f[fi]
        a = float(tri_areas[fi])
        third = a / 3.0
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        area_mix[i0] += third
        area_mix[i1] += third
        area_mix[i2] += third
        nbrs[i0].update((i1, i2))
        nbrs[i1].update((i0, i2))
        nbrs[i2].update((i0, i1))

    return [sorted(s) for s in nbrs], area_mix


def _edge_cotangent_weights(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> dict[tuple[int, int], float]:
    """Undirected edge -> sum of cotangents of opposite angles."""
    f = np.asarray(faces, dtype=np.int64)
    weights: dict[tuple[int, int], float] = {}

    for tri in f:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        p0, p1, p2 = vertices[i0], vertices[i1], vertices[i2]
        for a, b, opp in ((i1, i2, i0), (i0, i2, i1), (i0, i1, i2)):
            lo, hi = (min(a, b), max(a, b))
            key = (lo, hi)
            w = _cotangent_weight(
                vertices[opp],
                vertices[lo],
                vertices[hi],
            )
            weights[key] = weights.get(key, 0.0) + w
    return weights


def discrete_vertex_curvatures(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-vertex mean and Gaussian curvature (Å⁻¹ scale).

    Mean: ``0.5 * ||L|| / A`` with cotangent Laplacian ``L``.
    Gaussian: angle deficit divided by mixed area.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    n_verts = int(verts.shape[0])
    if n_verts == 0 or faces.shape[0] == 0:
        return np.zeros(n_verts), np.zeros(n_verts)

    neighbors, area_mix = _vertex_incidence(verts, faces)
    edge_w = _edge_cotangent_weights(verts, faces)

    lap = np.zeros((n_verts, 3), dtype=np.float64)
    for i in range(n_verts):
        for j in neighbors[i]:
            key = (min(i, j), max(i, j))
            w = edge_w.get(key, 0.0)
            lap[i] += w * (verts[j] - verts[i])

    mean_h = np.zeros(n_verts, dtype=np.float64)
    safe_a = np.maximum(area_mix, 1e-12)
    mean_h = 0.5 * np.linalg.norm(lap, axis=1) / safe_a
    mean_h[area_mix < 1e-12] = 0.0

    # Gaussian: angle deficit at each vertex
    gauss_k = np.zeros(n_verts, dtype=np.float64)
    angle_sum = np.zeros(n_verts, dtype=np.float64)
    for tri in faces:
        for k in range(3):
            vi = int(tri[k])
            vj = int(tri[(k + 1) % 3])
            vk = int(tri[(k + 2) % 3])
            e1 = verts[vj] - verts[vi]
            e2 = verts[vk] - verts[vi]
            n1 = np.linalg.norm(e1)
            n2 = np.linalg.norm(e2)
            if n1 < 1e-14 or n2 < 1e-14:
                continue
            c = float(np.clip(np.dot(e1, e2) / (n1 * n2), -1.0, 1.0))
            angle_sum[vi] += float(np.arccos(c))

    gauss_k = (2.0 * np.pi - angle_sum) / safe_a
    gauss_k[area_mix < 1e-12] = 0.0

    return mean_h, gauss_k


def vertex_curvatures(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    prefer_pyvista: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean and Gaussian curvature per vertex.

    Uses PyVista/VTK when importable and ``prefer_pyvista`` is True; otherwise
    :func:`discrete_vertex_curvatures`.
    """
    if prefer_pyvista:
        try:
            import pyvista as pv

            faces_pv = np.hstack(
                [
                    np.full((faces.shape[0], 1), 3, dtype=np.int64),
                    np.asarray(faces, dtype=np.int64),
                ]
            ).ravel()
            surface = pv.PolyData(np.asarray(vertices, dtype=np.float64), faces_pv)
            surface = surface.compute_normals(inplace=False)
            try:
                mean_h = np.asarray(surface.curvature(curv_type="mean"), dtype=np.float64)
                gauss_k = np.asarray(surface.curvature(curv_type="gaussian"), dtype=np.float64)
                if mean_h.shape[0] == vertices.shape[0]:
                    return mean_h, gauss_k
            except Exception:
                pass
        except ImportError:
            pass

    return discrete_vertex_curvatures(vertices, faces)
