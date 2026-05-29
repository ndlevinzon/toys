"""
Shape anisotropy from a protein surface mesh.

Uses the vertex cloud as a uniform surface sample of the SAS envelope. Metrics
follow standard gyration-tensor definitions (asphericity, prolateness, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anisotropy.mesh import ProteinMesh


@dataclass
class ShapeAnisotropy:
    """Principal shape descriptors of a mesh (lengths in Å)."""

    center: np.ndarray  # (3,)
    gyration_tensor: np.ndarray  # (3, 3) symmetric
    eigenvalues: np.ndarray  # λ1 ≥ λ2 ≥ λ3 (variances along principal axes)
    principal_axes: np.ndarray  # (3, 3) rows = unit eigenvectors
    radius_gyration: float
    asphericity: float  # 0 sphere, 1 highly aspherical
    prolateness: float  # >0 prolate, <0 oblate (relative to sphere)
    axis_lengths: np.ndarray  # 2 * sqrt(λ_i) — effective semi-axes

    def as_dict(self) -> dict[str, float]:
        """Flat summary for printing or JSON export."""
        return {
            "radius_gyration": self.radius_gyration,
            "asphericity": self.asphericity,
            "prolateness": self.prolateness,
            "lambda_1": float(self.eigenvalues[0]),
            "lambda_2": float(self.eigenvalues[1]),
            "lambda_3": float(self.eigenvalues[2]),
            "axis_length_1": float(self.axis_lengths[0]),
            "axis_length_2": float(self.axis_lengths[1]),
            "axis_length_3": float(self.axis_lengths[2]),
        }


def shape_anisotropy_from_mesh(
    mesh: ProteinMesh,
    *,
    use_faces: bool = True,
) -> ShapeAnisotropy:
    """
    Compute shape anisotropy from mesh vertices (optionally area-weighted).

    Parameters
    ----------
    mesh
        SAS triangle mesh.
    use_faces
        If True, weight each vertex by summed incident triangle areas
        (better surface sampling). If False, uniform vertex weights.
    """
    verts = mesh.vertices.astype(np.float64)
    if verts.shape[0] < 4:
        raise ValueError("Need at least four mesh vertices for shape tensor")

    if use_faces:
        weights = _vertex_area_weights(verts, mesh.faces)
    else:
        weights = np.ones(verts.shape[0], dtype=np.float64)

    weights /= weights.sum() + 1e-12
    center = (weights[:, None] * verts).sum(axis=0)
    centered = verts - center

    # Gyration tensor S = <r r^T>
    wc = centered * weights[:, None]
    gyration = wc.T @ centered

    evals, evecs = np.linalg.eigh(gyration)
    order = np.argsort(evals)[::-1]
    eigenvalues = evals[order]
    principal_axes = evecs[:, order].T

    rg_sq = float(eigenvalues.sum())
    radius_gyration = float(np.sqrt(max(rg_sq, 0.0)))

    # Asphericity (Δ): 1 − 3 λ3 / (λ1+λ2+λ3)  — zero for sphere
    lam_sum = float(eigenvalues.sum())
    if lam_sum > 1e-12:
        asphericity = float(1.0 - 3.0 * eigenvalues[2] / lam_sum)
    else:
        asphericity = 0.0

    # Relative shape anisotropy (κ²) prolateness sign from λ ordering
    if lam_sum > 1e-12:
        prolateness = float(
            (3.0 * eigenvalues[0] - lam_sum) / (2.0 * lam_sum)
            - (3.0 * eigenvalues[2] - lam_sum) / (2.0 * lam_sum)
        )
    else:
        prolateness = 0.0

    axis_lengths = 2.0 * np.sqrt(np.maximum(eigenvalues, 0.0))

    return ShapeAnisotropy(
        center=center,
        gyration_tensor=gyration,
        eigenvalues=eigenvalues,
        principal_axes=principal_axes,
        radius_gyration=radius_gyration,
        asphericity=asphericity,
        prolateness=prolateness,
        axis_lengths=axis_lengths,
    )


def _vertex_area_weights(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-vertex weight ∝ sum of incident triangle areas / 3."""
    weights = np.zeros(vertices.shape[0], dtype=np.float64)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    for k in range(3):
        np.add.at(weights, faces[:, k], areas / 3.0)
    floor = float(np.percentile(weights[weights > 0], 5)) if (weights > 0).any() else 1.0
    weights = np.maximum(weights, floor * 0.01)
    return weights
