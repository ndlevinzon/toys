"""Curvature backends (discrete vs optional PyVista)."""

from __future__ import annotations

import numpy as np

from anisotropy.curvature import discrete_vertex_curvatures, vertex_curvatures


def test_discrete_curvature_on_tetrahedron() -> None:
    # Regular tetrahedron — finite curvatures, no NaNs
    verts = np.array(
        [
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
    h, k = discrete_vertex_curvatures(verts, faces)
    assert h.shape == (4,)
    assert k.shape == (4,)
    assert np.isfinite(h).all()
    assert np.isfinite(k).all()


def test_vertex_curvatures_without_pyvista() -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    h, k = vertex_curvatures(verts, faces, prefer_pyvista=False)
    assert h.shape == (3,)
