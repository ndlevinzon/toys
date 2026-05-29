"""Regression tests for patch segmentation and parameterization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anisotropy.mesh import ProteinMesh
from anisotropy.patches import (
    _build_face_adjacency,
    _face_geometry,
    parameterize_mesh,
    segment_mesh_patches,
)
from anisotropy.pdb import load_pdb


def _legacy_face_adjacency(faces: np.ndarray) -> list[list[int]]:
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


def test_face_adjacency_matches_legacy() -> None:
    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [1, 4, 2], [2, 4, 5], [3, 2, 5]],
        dtype=np.int64,
    )
    assert _build_face_adjacency(faces) == _legacy_face_adjacency(faces)


@pytest.fixture(scope="module")
def crn():
    pdb = Path(__file__).resolve().parents[1] / "1CRN.pdb"
    if not pdb.is_file():
        pytest.skip("1CRN.pdb not available")
    return load_pdb(pdb)


def test_parameterize_table_charges(crn) -> None:
    """Smoke test: table charges, no PROPKA subprocess."""
    verts = crn.centers + np.random.default_rng(0).normal(scale=0.5, size=crn.centers.shape)
    faces = np.array([[0, 1, 2], [0, 2, 3], [1, 2, 4]], dtype=np.int64)
    mesh = ProteinMesh(vertices=verts, faces=faces, resolution_angstrom=2.0, probe_radius=1.4)
    param = parameterize_mesh(
        mesh,
        crn,
        ph=7.0,
        pka_source="table",
        charge_model="table",
        min_patch_area=1.0,
    )
    assert param.n_patches >= 1
    fm = param.feature_matrix()
    assert fm.shape[0] == param.n_patches
    assert np.isfinite(fm).all()
