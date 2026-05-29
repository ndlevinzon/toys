#!/usr/bin/env python3
"""
Orientation sampling density on the particle outer surface (default: pyramid).

Visual story (cryo-EM intuition)
-------------------------------
* **No orientation bias (slider = 0):** thick blue arrows **evenly spaced** on the
  whole pyramid — every viewing direction equally likely.

* **Orientation bias (slider → 1):** same arrow count, but arrows **cluster** on
  the gold preferred view (apex); most of the surface stays **bare** (missing
  orientations in the dataset).

Arrow **direction** = outward normal (viewing axis). **Thickness/length** are fixed;
**local spacing / density** on the surface shows sampling probability.

Examples
--------
::

    python orientation_distribution.py
    python orientation_distribution.py --kappa-max 80 --arrows 700
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from beam_projection_toy import make_pyramid_mesh
from orientation_sampling import bias_to_kappa

BEAM_LAB_XYZ = np.array([0.0, 0.0, -1.0], dtype=np.float64)


@dataclass
class OrientationSamplingField:
    """Arrow sites on the outer surface (all arrows share one length)."""

    points: np.ndarray  # (M, 3)
    normals: np.ndarray  # (M, 3) unit outward
    weights: np.ndarray  # (M,) sampling weight at each shown site
    n_candidates: int
    n_hidden: int  # biased: candidates not drawn


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float64)
    if v.ndim == 1:
        v = v.reshape(1, 3)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def outer_surface_mesh(mesh: pv.PolyData) -> pv.PolyData:
    """Exterior shell only (largest connected component, outward normals)."""
    surface = mesh.extract_surface(algorithm="dataset_surface").triangulate().clean()
    if surface.n_cells < 1:
        return mesh.triangulate().clean()
    exterior = surface.connectivity(extraction_mode="largest")
    return exterior.compute_normals(
        cell_normals=True,
        point_normals=False,
        consistent_normals=True,
        auto_orient_normals=True,
        inplace=False,
    )


def sample_surface_candidates(
    mesh: pv.PolyData,
    n_candidates: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Random points on the outer surface with face normals (area-weighted)."""
    tri = outer_surface_mesh(mesh)
    sizes = tri.compute_cell_sizes(length=False, area=True, volume=False)
    areas = np.asarray(sizes.cell_data["Area"], dtype=np.float64)
    probs = areas / areas.sum()
    cell_normals = np.asarray(tri.cell_normals, dtype=np.float64)
    centroid = np.asarray(tri.center, dtype=np.float64)

    points = np.empty((n_candidates, 3), dtype=np.float64)
    normals = np.empty((n_candidates, 3), dtype=np.float64)
    for i in range(n_candidates):
        cell_id = int(rng.choice(tri.n_cells, p=probs))
        cell = tri.get_cell(cell_id)
        tri_pts = tri.points[cell.point_ids]
        u, v = rng.random(2)
        if u + v > 1.0:
            u, v = 1.0 - u, 1.0 - v
        w = 1.0 - u - v
        point = u * tri_pts[0] + v * tri_pts[1] + w * tri_pts[2]
        normal = cell_normals[cell_id].copy()
        if np.dot(normal, point - centroid) < 0.0:
            normal = -normal
        points[i] = point
        normals[i] = normal

    return points, _unit_rows(normals)


def sampling_weight(
    points: np.ndarray,
    normals: np.ndarray,
    centroid: np.ndarray,
    mu: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """
    Relative probability of imaging from direction μ (vMF on S²).

    Uses the better of outward **normal** and radial ``(p − c)`` alignment with μ.
    On a pyramid, facet normals never point exactly at the +Z apex, but the radial
    direction does — so clusters appear on the preferred-view cap.
    """
    if kappa < 1e-8:
        return np.ones(len(points), dtype=np.float64)

    mu = _unit_rows(mu)[0]
    radial = _unit_rows(points - centroid)
    align = np.maximum(
        np.clip(normals @ mu, -1.0, 1.0),
        np.clip(radial @ mu, -1.0, 1.0),
    )
    log_w = float(kappa) * (align - 1.0)
    return np.exp(np.clip(log_w, -80.0, 0.0))


def bias_to_uniformity_blend(bias: float) -> float:
    """0 = enforce even spacing; 1 = spacing follows sampling probability only."""
    return float(np.clip(bias, 0.0, 1.0))


def _per_site_min_separation(
    weight: float,
    w_max: float,
    base_sep: float,
    uniformity_blend: float,
) -> float:
    """
    Minimum distance to accept another arrow near this site.

    Uniform (blend=0): constant ``base_sep`` → even density.
    Biased (blend=1): ``base_sep * sqrt(w_max / w)`` → density ∝ weight.
    """
    if uniformity_blend < 1e-8:
        return base_sep
    w = max(float(weight), 1e-12 * w_max)
    inhomogeneous = base_sep * np.sqrt(w_max / w)
    return (1.0 - uniformity_blend) * base_sep + uniformity_blend * inhomogeneous


def _greedy_density_pack(
    points: np.ndarray,
    weights: np.ndarray,
    base_sep: float,
    uniformity_blend: float,
    n_arrows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Place up to ``n_arrows`` sites; local spacing encodes probability density.

    High weight → small exclusion radius → arrows can pack tightly.
    Low weight → large radius → few or no arrows (bare surface).
    """
    n = len(points)
    w_max = float(np.max(weights))
    if w_max <= 0:
        return np.array([], dtype=np.int64)

    if uniformity_blend < 1e-8:
        order = rng.permutation(n)
    else:
        jitter = rng.random(n) * 1e-9
        order = np.argsort(-(weights + jitter))

    kept: list[int] = []
    tree: cKDTree | None = None
    for idx in order:
        if len(kept) >= n_arrows:
            break
        sep = _per_site_min_separation(
            weights[idx],
            w_max,
            base_sep,
            uniformity_blend,
        )
        if tree is None:
            kept.append(int(idx))
            tree = cKDTree(points[kept])
            continue
        dist, _ = tree.query(points[idx], k=1)
        if float(dist) >= sep:
            kept.append(int(idx))
            tree = cKDTree(points[kept])

    return np.asarray(kept, dtype=np.int64)


def _farthest_point_fill(
    points: np.ndarray,
    n_arrows: int,
    rng: np.random.Generator,
    *,
    forbidden: set[int] | None = None,
) -> np.ndarray:
    """Even coverage fallback: maximize minimum distance between chosen sites."""
    n = len(points)
    forbidden = forbidden or set()
    start = int(rng.integers(0, n))
    while start in forbidden and len(forbidden) < n:
        start = int(rng.integers(0, n))
    chosen = [start]
    tree = cKDTree(points[chosen])
    while len(chosen) < n_arrows:
        dists, _ = tree.query(points, k=1)
        for idx in np.argsort(-dists):
            if int(idx) in forbidden or int(idx) in chosen:
                continue
            chosen.append(int(idx))
            tree = cKDTree(points[chosen])
            break
        else:
            break
    return np.asarray(chosen[:n_arrows], dtype=np.int64)


def select_sampling_sites(
    points: np.ndarray,
    normals: np.ndarray,
    mu: np.ndarray,
    *,
    centroid: np.ndarray,
    bias: float,
    kappa_max: float,
    n_arrows: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Arrow sites whose **spatial density** mirrors orientation sampling probability.

    * bias ≈ 0: Poisson-style even spacing (uniform on S²).
    * bias → 1: tight clusters on high-probability normals; large empty regions.
    """
    kappa = bias_to_kappa(bias, kappa_max)
    blend = bias_to_uniformity_blend(bias)
    weights = (
        sampling_weight(points, normals, centroid, mu, kappa)
        if blend > 1e-8
        else np.ones(len(points), dtype=np.float64)
    )

    n = len(points)
    if n <= n_arrows:
        return np.arange(n), weights

    span = float(np.ptp(points, axis=0).max())

    if blend >= 0.98:
        # Strong bias: pack by weight with tight spacing near the preferred cap.
        base_sep = max(0.008 * span * (1.0 - 0.7 * blend), 1e-6)
        kept = _greedy_density_pack(
            points, weights, base_sep, 1.0, n_arrows, rng
        )
        return kept, weights

    lo, hi = 0.012 * span, 0.42 * span
    best = np.array([], dtype=np.int64)

    for _ in range(24):
        base_sep = 0.5 * (lo + hi)
        kept = _greedy_density_pack(
            points, weights, base_sep, blend, n_arrows, rng
        )
        n_kept = len(kept)
        if n_kept >= n_arrows:
            best = kept[:n_arrows]
            lo = base_sep
            if blend < 0.08 and n_kept == n_arrows:
                break
        else:
            if n_kept > len(best):
                best = kept
            hi = base_sep

    if len(best) < n_arrows:
        extra = _farthest_point_fill(
            points,
            n_arrows,
            rng,
            forbidden=set(best.tolist()),
        )
        if len(best) > 0:
            merged = list(best.tolist())
            for idx in extra:
                if idx not in merged:
                    merged.append(int(idx))
                if len(merged) >= n_arrows:
                    break
            best = np.asarray(merged[:n_arrows], dtype=np.int64)
        else:
            best = extra

    return best[:n_arrows], weights


def _fraction_bare_surface(
    points: np.ndarray,
    arrow_points: np.ndarray,
    cover_radius: float,
) -> float:
    """Share of candidate sites with no arrow within ``cover_radius`` (unsampled)."""
    if len(arrow_points) == 0:
        return 1.0
    tree = cKDTree(arrow_points)
    dists, _ = tree.query(points, k=1)
    return float(np.mean(dists > cover_radius))


def build_sampling_field(
    mesh: pv.PolyData,
    *,
    mu: np.ndarray,
    bias: float,
    kappa_max: float,
    n_arrows: int = 700,
    n_candidates: int = 12000,
    seed: int = 0,
    candidate_points: np.ndarray | None = None,
    candidate_normals: np.ndarray | None = None,
) -> OrientationSamplingField:
    """Place arrow sites on the outer surface; density encodes sampling probability."""
    rng = np.random.default_rng(seed)
    if candidate_points is None or candidate_normals is None:
        points, normals = sample_surface_candidates(mesh, n_candidates, rng)
    else:
        points, normals = candidate_points, candidate_normals

    centroid = np.asarray(points, dtype=np.float64).mean(axis=0)
    indices, all_weights = select_sampling_sites(
        points,
        normals,
        mu,
        centroid=centroid,
        bias=bias,
        kappa_max=kappa_max,
        n_arrows=n_arrows,
        rng=rng,
    )
    span = float(np.ptp(points, axis=0).max())
    cover = 0.06 * span
    bare_frac = _fraction_bare_surface(points, points[indices], cover)

    return OrientationSamplingField(
        points=points[indices],
        normals=normals[indices],
        weights=all_weights[indices],
        n_candidates=n_candidates,
        n_hidden=int(round(bare_frac * n_candidates)),
    )


def build_sampling_field_from_bias(
    candidate_points: np.ndarray,
    candidate_normals: np.ndarray,
    mu: np.ndarray,
    bias: float,
    *,
    kappa_max: float,
    n_arrows: int,
    n_candidates: int,
    seed: int,
) -> OrientationSamplingField:
    """Fixed arrow count; bias 0 = even spacing, bias 1 = tight clusters + empty regions."""
    seed_bias = int(seed) + int(round(bias * 10000))
    return build_sampling_field(
        pv.PolyData(candidate_points),
        mu=mu,
        bias=bias,
        kappa_max=kappa_max,
        n_arrows=n_arrows,
        n_candidates=n_candidates,
        seed=seed_bias,
        candidate_points=candidate_points,
        candidate_normals=candidate_normals,
    )


def _glyph_mesh_from_field(
    field: OrientationSamplingField,
    arrow_length: float,
    arrow_shaft: float,
) -> pv.PolyData:
    """Build thick glyph arrows (fixed length; direction = outward normal)."""
    cloud = pv.PolyData(field.points)
    cloud["orient"] = field.normals * arrow_length
    arrow_geom = pv.Arrow(
        start=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 1.0),
        scale=1.0,
        shaft_radius=arrow_shaft,
        tip_radius=arrow_shaft * 2.6,
        tip_length=0.32,
    )
    return cloud.glyph(orient="orient", scale=False, geom=arrow_geom)


def _surface_point_along_mu(mesh: pv.PolyData, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Point on the outer surface farthest along ``mu`` (for the preferred-view marker)."""
    mu = _unit_rows(mu)[0]
    exterior = outer_surface_mesh(mesh)
    pts = np.asarray(exterior.points, dtype=np.float64)
    centroid = np.asarray(exterior.center, dtype=np.float64)
    idx = int(np.argmax((pts - centroid) @ mu))
    normal = mu.copy()
    return pts[idx], normal


def _add_preferred_view_marker(
    plotter: pv.Plotter,
    mesh: pv.PolyData,
    mu: np.ndarray,
    *,
    marker_length: float,
    marker_shaft: float,
) -> None:
    """Gold arrow: direction cryo-EM favors when the particle is orientation-biased."""
    origin, direction = _surface_point_along_mu(mesh, mu)
    tip = origin + direction * marker_length
    shaft = pv.Line(origin, tip - direction * (0.28 * marker_length))
    tube = shaft.tube(radius=marker_shaft * 1.15, n_sides=24)
    plotter.add_mesh(tube, color="#f1c40f", opacity=0.95, name="preferred_view_shaft")
    cone = pv.Cone(
        center=tip - direction * (0.18 * marker_length),
        direction=direction,
        height=0.36 * marker_length,
        radius=marker_shaft * 3.2,
        resolution=24,
    )
    plotter.add_mesh(cone, color="#f39c12", opacity=0.95, name="preferred_view_tip")


def _arrow_color_for_bias(bias: float) -> str:
    """Unbiased = cool blue; increasing bias → orange → deep red clusters."""
    if bias < 0.05:
        return "#1a5276"
    if bias < 0.35:
        return "#d35400"
    return "#922b21"


def caption_for_bias(
    bias: float,
    field: OrientationSamplingField,
    *,
    kappa_max: float,
    n_arrows: int,
) -> tuple[str, str]:
    """
    Title + body for the on-screen legend (pedagogical, two-line block).

    Returns ``(title, body)``.
    """
    kappa = bias_to_kappa(bias, kappa_max)
    if kappa < 1e-8:
        return (
            "NO orientation bias (uniform on SO(3))",
            (
                "Arrows are evenly spaced over the whole surface — every outward "
                "normal (viewing direction) is equally likely. Same arrow count "
                f"({n_arrows}); spacing encodes probability, not length."
            ),
        )
    pct_bare = 100.0 * field.n_hidden / max(field.n_candidates, 1)
    return (
        f"ORIENTATION BIAS (slider = {bias:.2f}, kappa ~ {kappa:.1f})",
        (
            "Arrow density = sampling probability: tight red clusters on the gold "
            "preferred view (often imaged in 2D). Large empty patches = orientations "
            f"rarely collected ({n_arrows} arrows; ~{pct_bare:.0f}% of surface "
            "has no arrow nearby)."
        ),
    )


def show_interactive_orientation_bias(
    mesh: pv.PolyData,
    mu: np.ndarray,
    *,
    n_arrows: int = 700,
    n_candidates: int = 12000,
    kappa_max: float = 85.0,
    initial_bias: float = 0.0,
    seed: int = 0,
    arrow_length: float = 5.8,
    arrow_shaft: float = 0.082,
) -> None:
    """
    Interactive viewer: pyramid mesh + slider for orientation bias.

    Same ``n_arrows`` always; bias 0 = even Poisson spacing (no bias), bias 1 =
    thick red clusters on μ plus large bare regions (missing orientations).
    """
    mu = _unit_rows(mu)[0]
    rng = np.random.default_rng(seed)
    cand_points, cand_normals = sample_surface_candidates(mesh, n_candidates, rng)

    plotter = pv.Plotter(
        title="Orientation bias demo (pyramid)",
        window_size=(1280, 900),
    )
    plotter.set_background("white")

    exterior = outer_surface_mesh(mesh)
    plotter.add_mesh(
        exterior,
        color="#b8d4e8",
        opacity=0.9,
        smooth_shading=True,
        show_edges=False,
        name="particle",
    )
    # marker_len = 0.22 * float(np.ptp(np.asarray(exterior.points), axis=0).max())
    # _add_preferred_view_marker(
    #     plotter,
    #     mesh,
    #     mu,
    #     marker_length=marker_len,
    #     marker_shaft=arrow_shaft * 1.35,
    # )

    state: dict = {
        "glyph_actor": None,
        "title_actor": None,
        "body_actor": None,
    }

    def set_text_actor(key: str, text: str, position: str, font_size: int, color: str) -> None:
        if state[key] is not None:
            try:
                state[key].SetText(0, text)
                return
            except AttributeError:
                plotter.remove_actor(state[key])
        state[key] = plotter.add_text(
            text,
            position=position,
            font_size=font_size,
            color=color,
            shadow=False,
        )

    def apply_bias(bias: float) -> None:
        field = build_sampling_field_from_bias(
            cand_points,
            cand_normals,
            mu,
            bias,
            kappa_max=kappa_max,
            n_arrows=n_arrows,
            n_candidates=n_candidates,
            seed=seed,
        )
        glyphs = _glyph_mesh_from_field(field, arrow_length, arrow_shaft)

        if state["glyph_actor"] is not None:
            plotter.remove_actor(state["glyph_actor"])
        state["glyph_actor"] = plotter.add_mesh(
            glyphs,
            color=_arrow_color_for_bias(bias),
            opacity=0.94,
            name="sampling_arrows",
        )

        title, body = caption_for_bias(
            bias,
            field,
            kappa_max=kappa_max,
            n_arrows=n_arrows,
        )
        title_color = "#1a5276" if bias < 0.05 else "#922b21"
        set_text_actor("title_actor", title, "upper_edge", 13, title_color)
        set_text_actor("body_actor", body, (0.02, 0.88), 10, "#333333")

    def on_slider(bias_value: float) -> None:
        apply_bias(float(bias_value))

    plotter.add_slider_widget(
        on_slider,
        rng=[0.0, 1.0],
        value=float(initial_bias),
        title="Orientation bias",
        pointa=(0.05, 0.11),
        pointb=(0.55, 0.11),
        style="modern",
        interaction_event="always",
        title_height=0.022,
        title_color="black",
        color="#2c5f8a",
    )

    plotter.add_text(
        "0 — no bias\n(even spacing)",
        position=(0.05, 0.155),
        font_size=9,
        color="#1a5276",
        shadow=False,
    )
    plotter.add_text(
        "1 — strong bias\n(clusters + gaps)",
        position=(0.38, 0.155),
        font_size=9,
        color="#922b21",
        shadow=False,
    )
    plotter.add_text(
        "Gold arrow = preferred view μ (+Z apex)  |  "
        "Blue arrows = which orientations were collected  |  "
        "Arrow direction = viewing axis  |  local spacing/density = probability",
        position=(0.05, 0.03),
        font_size=9,
        color="#444444",
        shadow=False,
    )

    apply_bias(initial_bias)
    plotter.add_axes()
    plotter.view_yz()
    plotter.reset_camera()
    plotter.camera.zoom(1.12)
    plotter.show()


def parse_mu(text: str) -> np.ndarray:
    parts = [float(x) for x in text.replace(",", " ").split()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three components for --mu")
    return np.array(parts, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive orientation-bias field on the pyramid surface.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Use the slider: 0 = unbiased (even arrow spacing), 1 = strong bias\n"
            "(same arrow count, clustered on favored views, bare elsewhere)."
        ),
    )
    parser.add_argument(
        "--mu",
        type=parse_mu,
        default=np.array([0.0, 0.0, 1.0]),
        help="Preferred viewing direction (default: +Z, toward apex)",
    )
    parser.add_argument(
        "--kappa-max",
        type=float,
        default=85.0,
        help="kappa at slider=1 (default: 85)",
    )
    parser.add_argument(
        "--initial-bias",
        type=float,
        default=0.0,
        help="Starting slider value in [0, 1] (default: 0)",
    )
    parser.add_argument(
        "--arrows",
        type=int,
        default=700,
        help="Number of arrows (fixed count; spacing shows density; default: 700)",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=12000,
        help="Surface candidate pool for placement (default: 12000)",
    )
    parser.add_argument(
        "--arrow-length",
        type=float,
        default=10,
        help="Arrow length on surface (default: 5.8)",
    )
    parser.add_argument(
        "--arrow-shaft",
        type=float,
        default=0.5,
        help="Arrow shaft radius — thicker = clearer direction (default: 0.082)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mesh-size", type=int, default=96)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh = make_pyramid_mesh(size=args.mesh_size)
    show_interactive_orientation_bias(
        mesh,
        args.mu,
        n_arrows=args.arrows,
        n_candidates=args.candidates,
        kappa_max=args.kappa_max,
        initial_bias=args.initial_bias,
        seed=args.seed,
        arrow_length=args.arrow_length,
        arrow_shaft=args.arrow_shaft,
    )


if __name__ == "__main__":
    main()
