"""
Sample particle orientations for synthetic cryo-EM datasets.

Uniform on SO(3) (Haar) or biased via a von Mises–Fisher distribution on the
viewing axis (same κ mapping as ``orientation_distribution.py``).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_PREFERRED_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)
DEFAULT_KAPPA_MAX = 85.0


def bias_to_kappa(bias: float, kappa_max: float) -> float:
    """Map orientation-bias slider ∈ [0, 1] to vMF concentration κ (0 = uniform on S²)."""
    bias = float(np.clip(bias, 0.0, 1.0))
    if bias < 0.02:
        return 0.0
    return float((bias**1.35) * kappa_max)


def sample_vmf_unit_vector(
    mu: np.ndarray,
    kappa: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw one unit vector from vMF(μ, κ) on S² ⊂ R³.

    Uses uniform-on-sphere proposal with acceptance exp(κ(μ·v − 1)).
    """
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    mu /= np.linalg.norm(mu) + 1e-12
    if kappa < 1e-8:
        v = rng.standard_normal(3)
        return (v / np.linalg.norm(v)).astype(np.float64)

    while True:
        v = rng.standard_normal(3)
        v /= np.linalg.norm(v) + 1e-12
        if np.log(rng.random()) <= float(kappa * (np.dot(v, mu) - 1.0)):
            return v.astype(np.float64)


def rotation_from_view_axis_and_spin(view_axis_lab: np.ndarray, inplane_deg: float) -> Rotation:
    """
    VTK body→lab rotation: body +Z maps to ``view_axis_lab``, then spin about that axis.

    Uniform in-plane spin with uniform viewing axis gives uniform SO(3).
    """
    view_axis_lab = np.asarray(view_axis_lab, dtype=np.float64).reshape(3)
    view_axis_lab /= np.linalg.norm(view_axis_lab) + 1e-12
    z = np.array([0.0, 0.0, 1.0])
    align = Rotation.align_vectors(z.reshape(1, 3), view_axis_lab.reshape(1, 3))[0]
    spin = Rotation.from_rotvec(np.deg2rad(inplane_deg) * view_axis_lab)
    return spin * align


def sample_particle_rotation(
    rng: np.random.Generator,
    *,
    preferred_axis_xyz: np.ndarray = DEFAULT_PREFERRED_AXIS,
    orientation_bias: float = 0.0,
    kappa_max: float = DEFAULT_KAPPA_MAX,
) -> Rotation:
    """
    Sample one particle orientation R_i (VTK x, y, z).

    ``orientation_bias = 0``: uniform on SO(3).
    ``orientation_bias → 1``: views cluster near ``preferred_axis_xyz`` (pyramid apex +Z).
    """
    mu = np.asarray(preferred_axis_xyz, dtype=np.float64).reshape(3)
    mu /= np.linalg.norm(mu) + 1e-12
    kappa = bias_to_kappa(orientation_bias, kappa_max)
    if kappa < 1e-8:
        return Rotation.random(random_state=rng)
    view_axis = sample_vmf_unit_vector(mu, kappa, rng)
    inplane = float(rng.uniform(0.0, 360.0))
    return rotation_from_view_axis_and_spin(view_axis, inplane)


def viewing_axis_from_rotation(rotation: Rotation) -> np.ndarray:
    """Lab-frame direction of body +Z after applying ``rotation``."""
    return rotation.apply(np.array([0.0, 0.0, 1.0]))


def summarize_orientation_bias(
    rotations: list[Rotation] | Rotation,
    preferred_axis_xyz: np.ndarray,
) -> dict[str, float]:
    """Mean alignment with μ (1 = all views on preferred axis)."""
    if isinstance(rotations, Rotation):
        rots = list(rotations)
    else:
        rots = rotations
    mu = np.asarray(preferred_axis_xyz, dtype=np.float64).reshape(3)
    mu /= np.linalg.norm(mu) + 1e-12
    axes = np.stack([viewing_axis_from_rotation(r) for r in rots], axis=0)
    dots = np.clip(axes @ mu, -1.0, 1.0)
    return {
        "mean_dot_mu": float(np.mean(dots)),
        "min_dot_mu": float(np.min(dots)),
        "max_dot_mu": float(np.max(dots)),
    }
