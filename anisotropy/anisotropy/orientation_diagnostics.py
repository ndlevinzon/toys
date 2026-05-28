"""
Cryo-EM orientation distribution plots for ``orientation_sample.py``.

Viewing direction convention (lab frame)
--------------------------------------
Beam propagates along **-Z** (electrons toward the detector at +Z).
For each particle rotation ``R`` (maps particle → lab), the viewing axis in the
particle frame is ``d = R^T @ [0, 0, 1]`` (unit vector). Spherical angles:

* **Azimuth** — ``atan2(d_y, d_x)`` in degrees, range ``(-180, 180]``
* **Elevation** — ``arcsin(d_z)`` in degrees above the lab XY plane, range ``[-90, 90]``

**In-plane rotation** (spin about the viewing axis) is a third Euler degree of freedom;
it changes the 3D render but not ``(azimuth, elevation)``.

Boltzmann weights ``w_i ∝ exp(-β (E_i - E_min))`` can collapse onto one sample when
``β · ΔE`` is large — the heatmaps then show one bright bin even though many poses
look different in rendered snapshots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Fixed cryo-EM beam / micrograph axis (matches orientation_sample slab +Z).
VIEW_DIRECTION_LAB = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def unit_vectors(rows: np.ndarray) -> np.ndarray:
    v = np.asarray(rows, dtype=np.float64)
    if v.ndim == 1:
        v = v.reshape(1, 3)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def viewing_direction_particle_frame(rotation: np.ndarray) -> np.ndarray:
    """Unit viewing direction in the particle frame for lab viewing axis +Z."""
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return unit_vectors((R.T @ VIEW_DIRECTION_LAB).reshape(1, 3))[0]


def viewing_angles_degrees(rotation: np.ndarray) -> tuple[float, float]:
    """
    Return (azimuth, elevation) in degrees for one rotation matrix.

    Azimuth is measured in the lab XY plane; elevation is angle above that plane.
    """
    d = viewing_direction_particle_frame(rotation)
    azimuth = float(np.degrees(np.arctan2(d[1], d[0])))
    elevation = float(np.degrees(np.arcsin(np.clip(d[2], -1.0, 1.0))))
    return azimuth, elevation


def inplane_rotation_degrees(rotation: np.ndarray) -> float:
    """
    In-plane spin (deg) about the viewing axis — visible in 3D renders, not in az/el.

    Measures rotation of the lab +X axis projected onto the plane ⊥ viewing axis.
    """
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    d = viewing_direction_particle_frame(R)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(ref, d))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(d, ref)
    u = unit_vectors(u.reshape(1, 3))[0]
    v = np.cross(d, u)
    e_x_particle = R[:, 0]
    e_x_particle = e_x_particle - np.dot(e_x_particle, d) * d
    norm = float(np.linalg.norm(e_x_particle))
    if norm < 1e-12:
        return 0.0
    e_x_particle /= norm
    return float(np.degrees(np.arctan2(np.dot(e_x_particle, v), np.dot(e_x_particle, u))))


def viewing_angles_from_rotations(rotations: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Batch (azimuth, elevation) in degrees, shape (N,)."""
    az = np.empty(len(rotations), dtype=np.float64)
    el = np.empty(len(rotations), dtype=np.float64)
    for i, R in enumerate(rotations):
        az[i], el[i] = viewing_angles_degrees(R)
    return az, el


def inplane_angles_from_rotations(rotations: list[np.ndarray]) -> np.ndarray:
    return np.array([inplane_rotation_degrees(R) for R in rotations], dtype=np.float64)


def directions_from_rotations(rotations: list[np.ndarray]) -> np.ndarray:
    """(N, 3) unit viewing directions in particle frame."""
    return np.stack([viewing_direction_particle_frame(R) for R in rotations], axis=0)


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = np.maximum(w, 0.0)
    s = float(w.sum())
    if s <= 0:
        return np.ones_like(w) / max(len(w), 1)
    return w / s


def effective_sample_size(weights: np.ndarray) -> float:
    """ESS = 1 / sum(w_i^2) for normalized weights."""
    w = _normalize_weights(weights)
    return float(1.0 / (np.sum(w * w) + 1e-30))


def boltzmann_weights(energies: np.ndarray, beta: float) -> np.ndarray:
    """Normalized weights w_i ∝ exp(-beta * (E_i - E_min))."""
    E = np.asarray(energies, dtype=np.float64).reshape(-1)
    if len(E) == 0:
        return np.array([], dtype=np.float64)
    if float(beta) <= 0.0:
        return np.ones(len(E), dtype=np.float64) / len(E)
    Emin = float(E.min())
    logw = -float(beta) * (E - Emin)
    logw -= float(np.max(logw))
    w = np.exp(logw)
    return w / (float(w.sum()) + 1e-30)


def effective_sample_size_for_beta(energies: np.ndarray, beta: float) -> float:
    return effective_sample_size(boltzmann_weights(energies, beta))


def calibrate_beta_auto(
    energies: np.ndarray,
    *,
    target_ess: float = 20.0,
    method: str = "ess",
    weight_ratio_at_reference: float = 0.1,
    ess_local_pool_fraction: float = 0.125,
) -> tuple[float, dict[str, Any]]:
    """
    Choose beta after uniform SO(3) energy samples are collected.

    Methods
    -------
    ``ess`` (default)
        Binary-search beta so ESS ≈ ``target_ess``.
    ``top10``
        beta = -ln(ratio) / (E_rank10 - E_min) using lowest-energy decile edge.
    ``rank2``
        beta = -ln(ratio) / (E_rank2 - E_min).

    Returns ``(beta, details)`` for logging / JSON export.
    """
    E = np.asarray(energies, dtype=np.float64).reshape(-1)
    n = len(E)
    if n == 0:
        return 1.0, {"method": method, "error": "no samples"}

    Emin = float(E.min())
    gaps = np.sort(E) - Emin
    spread = float(np.max(E) - Emin)
    if spread < 1e-12:
        return 1.0, {
            "method": method,
            "note": "degenerate energies",
            "achieved_ess": float(n),
        }

    ratio = float(np.clip(weight_ratio_at_reference, 1e-6, 1.0))
    log_ratio = float(-np.log(ratio))
    method = str(method).lower()

    if method == "rank2":
        d_ref = float(gaps[1]) if len(gaps) > 1 else spread
        beta = log_ratio / max(d_ref, 1e-30)
        details: dict[str, Any] = {
            "method": "rank2",
            "reference_delta_E": d_ref,
            "weight_ratio_at_reference": ratio,
        }
    elif method == "top10":
        k = min(10, n)
        d_ref = float(np.sort(E)[:k].max() - Emin) if k > 1 else spread
        beta = log_ratio / max(d_ref, 1e-30)
        details = {
            "method": "top10",
            "reference_delta_E": d_ref,
            "weight_ratio_at_reference": ratio,
        }
    elif method != "ess":
        raise ValueError(f"Unknown beta auto method: {method!r} (use ess, top10, rank2)")

    if method == "ess":
        target = float(np.clip(target_ess, 1.5, n - 1e-9))
        pool_k = int(
            np.clip(
                max(target * 2, 10, round(n * ess_local_pool_fraction)),
                10,
                n,
            )
        )
        E_pool = np.sort(E)[:pool_k]

        def ess_at(b: float) -> float:
            return effective_sample_size_for_beta(E_pool, b)

        if ess_at(0.0) <= target:
            beta = 0.0
        else:
            lo, hi = 0.0, 1.0
            while ess_at(hi) > target and hi < 1e12:
                hi *= 2.0
            if ess_at(hi) > target:
                span = float(np.max(E_pool) - Emin)
                hi = max(log_ratio / max(span, 1e-30), hi)
            for _ in range(64):
                mid = 0.5 * (lo + hi)
                if ess_at(mid) > target:
                    lo = mid
                else:
                    hi = mid
            beta = 0.5 * (lo + hi)

        details = {
            "method": "ess",
            "target_ess": target,
            "ess_local_pool_size": pool_k,
            "achieved_ess_local_pool": ess_at(beta),
            "achieved_ess_all_samples": effective_sample_size_for_beta(E, beta),
        }
    else:
        details["achieved_ess"] = effective_sample_size_for_beta(E, beta)

    return float(beta), details


def angular_spread_degrees(directions: np.ndarray) -> float:
    """RMS angular deviation (deg) of unit directions from their mean direction."""
    d = unit_vectors(directions)
    mean = unit_vectors(d.mean(axis=0, keepdims=True))[0]
    cos = np.clip(d @ mean, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos))
    return float(np.sqrt(np.mean(angles * angles)))


def summarize_orientation_sampling(
    energies: np.ndarray,
    weights: np.ndarray,
    rotations: list[np.ndarray],
    *,
    beta: float,
    n_azimuth_bins: int = 72,
    n_elevation_bins: int = 36,
) -> dict[str, Any]:
    """Diagnostics explaining sparse heatmaps vs varied 3D renders."""
    w = _normalize_weights(weights)
    E = np.asarray(energies, dtype=np.float64)
    az, el = viewing_angles_from_rotations(rotations)
    inplane = inplane_angles_from_rotations(rotations)
    dirs = directions_from_rotations(rotations)

    Emin = float(E.min())
    gaps = np.sort(E) - Emin
    w_sorted = np.sort(w)[::-1]

    az_edges = np.linspace(-180.0, 180.0, n_azimuth_bins + 1)
    el_edges = np.linspace(-90.0, 90.0, n_elevation_bins + 1)
    H, _, _ = np.histogram2d(az, el, bins=[az_edges, el_edges], weights=w)
    H_unw, _, _ = np.histogram2d(az, el, bins=[az_edges, el_edges])
    occupied_weighted = int(np.count_nonzero(H > 0))
    occupied_uniform = int(np.count_nonzero(H_unw > 0))

    top10_idx = np.argsort(E)[: min(10, len(E))]
    top10_spread_view = angular_spread_degrees(dirs[top10_idx])
    top10_spread_inplane = float(np.std(inplane[top10_idx]))

    notes = []
    ess = effective_sample_size(weights)
    if ess < 5:
        notes.append(
            f"Boltzmann weights are highly concentrated (ESS~{ess:.1f}): "
            "weighted heatmaps will show one peak even if many poses differ in 3D."
        )
    if beta * float(gaps[min(10, len(gaps) - 1)]) > 5:
        notes.append(
            f"beta*dE to 10th-best pose ~ {beta * float(gaps[min(10, len(gaps) - 1)]):.1f} — "
            "top-10 energies can look distinct while weights are negligible."
        )
    if top10_spread_view < 15 and top10_spread_inplane > 20:
        notes.append(
            f"Top-10 lowest-energy poses span only ~{top10_spread_view:.1f}° on S² "
            f"but ~{top10_spread_inplane:.1f}° std in-plane — renders vary, az/el map does not."
        )
    if occupied_uniform < n_azimuth_bins * n_elevation_bins * 0.05:
        notes.append(
            f"Only {occupied_uniform} az/el bins hit by {len(E)} uniform SO(3) draws "
            f"(of {n_azimuth_bins * n_elevation_bins}) — sparse bins are expected, not a bug."
        )

    d2 = float(gaps[1]) if len(gaps) > 1 else 0.0
    d10 = float(gaps[min(10, len(gaps) - 1)])
    top_k = min(10, len(E))
    d_top10 = float(np.sort(E)[:top_k].max() - Emin) if top_k > 1 else 0.0
    iqr = float(np.percentile(E, 75) - np.percentile(E, 25))

    def beta_for_ratio(delta_e: float, ratio: float = 0.1) -> float | None:
        if delta_e <= 0:
            return None
        return float(-np.log(ratio) / delta_e)

    beta_rec_r2 = beta_for_ratio(d2)
    beta_rec_r10 = beta_for_ratio(d10)
    beta_rec_top10 = beta_for_ratio(d_top10)

    if beta > 0 and d2 > 0 and beta * d2 > 20:
        notes.append(
            f"Current beta={beta:g} gives exp(-beta*dE_rank2) ~ "
            f"{np.exp(-beta * d2):.2e}. Try beta ~ {beta_rec_r2:.3e} "
            "so rank-2 has ~10% of rank-1 weight, or reduce H_el scale in ising_params.yaml."
        )

    return {
        "n_samples": int(len(E)),
        "beta": float(beta),
        "E_min": Emin,
        "E_max": float(E.max()),
        "E_mean": float(E.mean()),
        "E_std": float(E.std()),
        "energy_iqr": iqr,
        "delta_E_top10_spread": d_top10,
        "delta_E_to_rank2": d2,
        "delta_E_to_rank10": d10,
        "beta_times_deltaE_rank2": float(beta * d2),
        "beta_recommended_for_rank2_weight_10pct": beta_rec_r2,
        "beta_recommended_for_rank10_weight_10pct": beta_rec_r10,
        "beta_recommended_for_top10_spread_weight_10pct": beta_rec_top10,
        "weight_rank1": float(w_sorted[0]),
        "weight_rank2": float(w_sorted[1]) if len(w_sorted) > 1 else 0.0,
        "weight_rank10": float(w_sorted[min(9, len(w_sorted) - 1)]),
        "effective_sample_size": ess,
        "viewing_direction_spread_deg_all": angular_spread_degrees(dirs),
        "viewing_direction_spread_deg_top10": top10_spread_view,
        "inplane_spread_deg_top10_std": top10_spread_inplane,
        "occupied_az_el_bins_weighted": occupied_weighted,
        "occupied_az_el_bins_uniform": occupied_uniform,
        "interpretation_notes": notes,
    }


def write_orientation_sampling_report(
    outdir: str | Path,
    summary: dict[str, Any],
    *,
    prefix: str = "diagnostics",
) -> Path:
    outdir = Path(outdir)
    path_json = outdir / f"{prefix}_orientation_sampling.json"
    path_txt = outdir / f"{prefix}_orientation_sampling.txt"
    path_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "Orientation sampling diagnostics",
        "================================",
        f"n_samples: {summary['n_samples']}",
        f"beta: {summary['beta']}",
        f"E_min: {summary['E_min']:.4g}  E_max: {summary['E_max']:.4g}  "
        f"E_std: {summary['E_std']:.4g}",
        f"ΔE (rank 2): {summary['delta_E_to_rank2']:.4g}  "
        f"ΔE (rank 10): {summary['delta_E_to_rank10']:.4g}",
        f"weight rank 1: {summary['weight_rank1']:.4g}  "
        f"rank 2: {summary['weight_rank2']:.4g}  "
        f"rank 10: {summary['weight_rank10']:.4g}",
        f"effective sample size (ESS): {summary['effective_sample_size']:.2f}",
        f"viewing-dir spread (all samples): "
        f"{summary['viewing_direction_spread_deg_all']:.1f}°",
        f"viewing-dir spread (top-10 by energy): "
        f"{summary['viewing_direction_spread_deg_top10']:.1f}°",
        f"in-plane std (top-10 by energy): "
        f"{summary['inplane_spread_deg_top10_std']:.1f}°",
        f"occupied az/el bins (weighted / uniform): "
        f"{summary['occupied_az_el_bins_weighted']} / "
        f"{summary['occupied_az_el_bins_uniform']}",
        "",
        "Beta calibration (for w_i = exp(-beta * (E_i - E_min)) / Z):",
        f"  beta * dE_rank2 = {summary.get('beta_times_deltaE_rank2', 0):.4g}",
    ]
    for key, label in (
        ("beta_recommended_for_rank2_weight_10pct", "rank-2 at 10% of rank-1"),
        ("beta_recommended_for_rank10_weight_10pct", "rank-10 at 10% of rank-1"),
        ("beta_recommended_for_top10_spread_weight_10pct", "worst of top-10 at 10% of rank-1"),
    ):
        val = summary.get(key)
        if val is not None:
            lines.append(f"  beta ~ {val:.6g}  ({label})")
    lines.extend(
        [
            f"  energy IQR (all samples): {summary.get('energy_iqr', 0):.4g}",
            f"  dE across top-10 energies: {summary.get('delta_E_top10_spread', 0):.4g}",
            "",
            "Notes:",
        ]
    )
    for note in summary.get("interpretation_notes", []):
        lines.append(f"  - {note}")
    if not summary.get("interpretation_notes"):
        lines.append("  (none)")
    path_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path_txt


def _histogram_az_el(
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    weights: np.ndarray | None,
    *,
    n_azimuth_bins: int,
    n_elevation_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    az_edges = np.linspace(-180.0, 180.0, n_azimuth_bins + 1)
    el_edges = np.linspace(-90.0, 90.0, n_elevation_bins + 1)
    if weights is None:
        w = np.ones(len(azimuth_deg), dtype=np.float64) / max(len(azimuth_deg), 1)
    else:
        w = _normalize_weights(weights)
    H, _, _ = np.histogram2d(
        np.asarray(azimuth_deg, dtype=np.float64),
        np.asarray(elevation_deg, dtype=np.float64),
        bins=[az_edges, el_edges],
        weights=w,
    )
    return H.T, az_edges, el_edges


def plot_viewing_direction_distribution(
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    weights: np.ndarray,
    outpath: str | Path,
    *,
    n_azimuth_bins: int = 36,
    n_elevation_bins: int = 18,
    cmap: str = "viridis",
    log_floor: float = 1e-4,
) -> None:
    """
    2D map: x = azimuth, y = elevation.

    Left: log10 weighted density (shows low-probability tails).
    Right: scatter of all samples (size ∝ weight) — raw SO(3) exploration on S².
    """
    import matplotlib.pyplot as plt

    w = _normalize_weights(weights)
    az = np.asarray(azimuth_deg, dtype=np.float64)
    el = np.asarray(elevation_deg, dtype=np.float64)

    H, az_edges, el_edges = _histogram_az_el(
        az, el, weights, n_azimuth_bins=n_azimuth_bins, n_elevation_bins=n_elevation_bins
    )
    peak = float(H.max())
    H_log = np.log10(H + peak * log_floor) if peak > 0 else H

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pcm = axes[0].pcolormesh(az_edges, el_edges, H_log, cmap=cmap, shading="flat")
    fig.colorbar(pcm, ax=axes[0], label="log10 probability (+ floor)")
    axes[0].set_xlabel("azimuth (deg)")
    axes[0].set_ylabel("elevation (deg)")
    axes[0].set_title("Weighted distribution (log scale)")
    axes[0].set_xlim(-180, 180)
    axes[0].set_ylim(-90, 90)

    sizes = 8.0 + 120.0 * (w / (w.max() + 1e-30))
    sc = axes[1].scatter(az, el, c=np.log10(w + 1e-30), s=sizes, cmap=cmap, alpha=0.65, edgecolors="none")
    fig.colorbar(sc, ax=axes[1], label="log10 weight")
    axes[1].set_xlabel("azimuth (deg)")
    axes[1].set_ylabel("elevation (deg)")
    axes[1].set_title("All samples (marker size ∝ weight)")
    axes[1].set_xlim(-180, 180)
    axes[1].set_ylim(-90, 90)
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("Viewing direction distribution (cryo-EM axis in particle frame)", y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_viewing_direction_uniform(
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    outpath: str | Path,
    *,
    n_azimuth_bins: int = 36,
    n_elevation_bins: int = 18,
    cmap: str = "cividis",
) -> None:
    """Unweighted histogram — density of random SO(3) draws on S² (should be ~flat)."""
    import matplotlib.pyplot as plt

    az = np.asarray(azimuth_deg, dtype=np.float64)
    el = np.asarray(elevation_deg, dtype=np.float64)
    H, az_edges, el_edges = _histogram_az_el(
        az, el, None, n_azimuth_bins=n_azimuth_bins, n_elevation_bins=n_elevation_bins
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    pcm = ax.pcolormesh(az_edges, el_edges, H, cmap=cmap, shading="flat")
    fig.colorbar(pcm, ax=ax, label="fraction of samples")
    ax.scatter(az, el, s=6, c="white", alpha=0.35, edgecolors="none")
    ax.set_xlabel("azimuth (deg)")
    ax.set_ylabel("elevation (deg)")
    ax.set_title("Uniform SO(3) sample density (unweighted; not Boltzmann)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_inplane_rotation_distribution(
    inplane_deg: np.ndarray,
    weights: np.ndarray,
    outpath: str | Path,
) -> None:
    """In-plane spin about viewing axis — explains 3D render diversity at fixed az/el."""
    import matplotlib.pyplot as plt

    w = _normalize_weights(weights)
    x = np.asarray(inplane_deg, dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(x, bins=min(36, max(12, len(x) // 20)), weights=w, color="steelblue", alpha=0.85)
    axes[0].set_xlabel("in-plane angle (deg)")
    axes[0].set_ylabel("weighted count")
    axes[0].set_title("Weighted in-plane rotation")

    axes[1].hist(x, bins=min(36, max(12, len(x) // 20)), color="gray", alpha=0.85)
    axes[1].set_xlabel("in-plane angle (deg)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Unweighted in-plane rotation (all SO(3) samples)")
    fig.suptitle("In-plane rotation (not shown on azimuth–elevation maps)", y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _spherical_probability_grid(
    rotations: list[np.ndarray],
    weights: np.ndarray,
    *,
    n_phi: int,
    n_theta: int,
    smooth_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (PHI, THETA, P_linear, P_display, peak) on colatitude/latitude grid."""
    w_n = _normalize_weights(weights)
    dirs = directions_from_rotations(rotations)
    phi = np.arctan2(dirs[:, 1], dirs[:, 0])
    theta = np.arccos(np.clip(dirs[:, 2], -1.0, 1.0))
    phi_edges = np.linspace(-np.pi, np.pi, n_phi + 1)
    theta_edges = np.linspace(0.0, np.pi, n_theta + 1)
    H, _, _ = np.histogram2d(phi, theta, bins=[phi_edges, theta_edges], weights=w_n)
    H = H.T
    if smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter

            H = gaussian_filter(H, sigma=smooth_sigma, mode="wrap")
        except ImportError:
            pass
    total = float(H.sum())
    if total > 0:
        H = H / total
    peak = float(H.max())
    phi_c = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    theta_c = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    PHI, THETA = np.meshgrid(phi_c, theta_c)
    return PHI, THETA, H, H, peak


def plot_orientation_probability_sphere(
    rotations: list[np.ndarray],
    weights: np.ndarray,
    outpath: str | Path,
    *,
    n_phi: int = 64,
    n_theta: int = 32,
    cmap: str = "inferno",
    radial_scale: float = 0.55,
    smooth_sigma: float = 1.8,
    color_by_height: bool = True,
) -> None:
    """
    Deformed sphere: radius and face color both encode probability on S².

    High-probability caps bulge outward and appear bright/hot in the colormap.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    PHI, THETA, P, _, peak = _spherical_probability_grid(
        rotations,
        weights,
        n_phi=n_phi,
        n_theta=n_theta,
        smooth_sigma=smooth_sigma,
    )
    if peak <= 0:
        P = np.ones_like(P) / P.size

    Rrad = 1.0 + radial_scale * (P / max(peak, 1e-30))
    X = Rrad * np.sin(THETA) * np.cos(PHI)
    Y = Rrad * np.sin(THETA) * np.sin(PHI)
    Z = Rrad * np.cos(THETA)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    norm = plt.Normalize(vmin=0.0, vmax=peak)
    height_for_color = P if color_by_height else (P / max(peak, 1e-30))
    facecolors = plt.colormaps[cmap](norm(height_for_color))
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    surf.set_zorder(2)

    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 24)
    U, V = np.meshgrid(u, v)
    ax.plot_wireframe(
        np.sin(V) * np.cos(U),
        np.sin(V) * np.sin(U),
        np.cos(V),
        color="0.35",
        alpha=0.15,
        linewidth=0.35,
    )

    m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array(P.ravel())
    cbar = fig.colorbar(m, ax=ax, shrink=0.55, pad=0.1)
    cbar.set_label("probability density (height & color)")

    ax.set_title("Orientation probability sphere (viewing axis)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=24, azim=-58)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_viewing_direction_smoothed(
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    weights: np.ndarray,
    outpath: str | Path,
    *,
    n_azimuth_bins: int = 72,
    n_elevation_bins: int = 36,
    smooth_sigma: float = 2.0,
    cmap: str = "inferno",
) -> None:
    """Smoothed azimuth–elevation map (Gaussian-filtered histogram)."""
    import matplotlib.pyplot as plt

    w = _normalize_weights(weights)
    H, az_edges, el_edges = _histogram_az_el(
        azimuth_deg,
        elevation_deg,
        weights,
        n_azimuth_bins=n_azimuth_bins,
        n_elevation_bins=n_elevation_bins,
    )
    if smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter

            H = gaussian_filter(H, sigma=smooth_sigma, mode="nearest")
        except ImportError:
            pass
    s = float(H.sum())
    if s > 0:
        H = H / s

    fig, ax = plt.subplots(figsize=(10, 5))
    pcm = ax.pcolormesh(az_edges, el_edges, H, cmap=cmap, shading="flat")
    fig.colorbar(pcm, ax=ax, label="probability density")
    ax.set_xlabel("azimuth (deg)")
    ax.set_ylabel("elevation (deg)")
    ax.set_title("Viewing direction distribution (smoothed)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def save_orientation_distribution_plots(
    outdir: str | Path,
    rotations: list[np.ndarray],
    weights: np.ndarray,
    *,
    prefix: str = "diagnostics",
    mcmc_mask: np.ndarray | None = None,
) -> dict[str, Path]:
    """Write orientation distribution figures; return output paths."""
    outdir = Path(outdir)
    az, el = viewing_angles_from_rotations(rotations)
    inplane = inplane_angles_from_rotations(rotations)

    paths = {
        "viewing_weighted": outdir / f"{prefix}_viewing_direction_distribution.png",
        "viewing_smoothed": outdir / f"{prefix}_viewing_direction_smoothed.png",
        "viewing_uniform": outdir / f"{prefix}_viewing_direction_uniform.png",
        "inplane": outdir / f"{prefix}_inplane_rotation.png",
        "sphere": outdir / f"{prefix}_orientation_probability_sphere.png",
    }
    plot_viewing_direction_distribution(az, el, weights, paths["viewing_weighted"])
    plot_viewing_direction_smoothed(az, el, weights, paths["viewing_smoothed"])
    plot_viewing_direction_uniform(az, el, paths["viewing_uniform"])
    plot_inplane_rotation_distribution(inplane, weights, paths["inplane"])
    plot_orientation_probability_sphere(rotations, weights, paths["sphere"])

    if mcmc_mask is not None and np.any(mcmc_mask):
        mcmc_paths = {
            "viewing_mcmc": outdir / f"{prefix}_viewing_direction_mcmc.png",
            "sphere_mcmc": outdir / f"{prefix}_orientation_sphere_mcmc.png",
        }
        w_mcmc = np.asarray(weights, dtype=np.float64)[mcmc_mask]
        Rs_mcmc = [R for R, keep in zip(rotations, mcmc_mask) if keep]
        az_m = az[mcmc_mask]
        el_m = el[mcmc_mask]
        plot_viewing_direction_smoothed(az_m, el_m, w_mcmc, mcmc_paths["viewing_mcmc"])
        plot_orientation_probability_sphere(Rs_mcmc, w_mcmc, mcmc_paths["sphere_mcmc"])
        paths.update(mcmc_paths)

    return paths
