#!/usr/bin/env python3
"""Interactive 3D viewer for ab initio reconstruction volumes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

from beam_projection_toy import make_demo_volume, volume_to_mesh


def load_volume(path: Path, key: str = "volume") -> np.ndarray:
    """Load a 3D volume from .npz (key) or .npy."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        vol = np.load(path)
    else:
        with np.load(path) as data:
            if key not in data:
                raise KeyError(f"'{key}' not in {path.name}; keys: {list(data.files)}")
            vol = np.asarray(data[key])
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}")
    return np.asarray(vol, dtype=np.float32)


def auto_iso_level(volume: np.ndarray, percentile: float = 88.0) -> float:
    """Pick an isosurface level from the volume intensity histogram."""
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return 0.3
    return float(np.percentile(finite, percentile))


def iso_slider_range(
    volume: np.ndarray,
    *,
    lo_percentile: float = 2.0,
    hi_percentile: float = 99.5,
) -> tuple[float, float]:
    """Min/max isosurface levels for the interactive slider."""
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo = float(np.percentile(finite, lo_percentile))
    hi = float(np.percentile(finite, hi_percentile))
    if hi <= lo:
        hi = lo + max(float(finite.max() - finite.min()) * 0.01, 1e-6)
    return lo, hi


def show_reconstruction(
    volume: np.ndarray,
    *,
    iso_level: float | None = None,
    title: str = "Ab initio reconstruction",
    compare_volume: np.ndarray | None = None,
    window_size: tuple[int, int] = (1000, 760),
) -> None:
    """
    PyVista viewer with a slider to change the reconstruction isosurface on the fly.
    """
    iso_lo, iso_hi = iso_slider_range(volume)
    if iso_level is None:
        iso_level = float(np.clip(auto_iso_level(volume), iso_lo, iso_hi))
    else:
        iso_level = float(np.clip(iso_level, iso_lo, iso_hi))

    plotter = pv.Plotter(title=title, window_size=window_size)
    plotter.set_background("white")

    state: dict = {"mesh_actor": None, "iso_text": None}

    if compare_volume is not None:
        iso_ref = auto_iso_level(compare_volume)
        ref_mesh = volume_to_mesh(compare_volume, level=iso_ref)
        plotter.add_mesh(
            ref_mesh,
            color="#cccccc",
            opacity=0.25,
            style="wireframe",
            line_width=1,
            label="reference",
        )

    def set_iso_text(level: float) -> None:
        text = f"Isosurface level = {level:.5f}"
        if state["iso_text"] is not None:
            try:
                state["iso_text"].SetText(0, text)
                return
            except AttributeError:
                plotter.remove_actor(state["iso_text"])
        state["iso_text"] = plotter.add_text(
            text,
            position="upper_left",
            font_size=11,
            color="black",
            shadow=False,
        )

    def apply_iso(level: float) -> None:
        level = float(np.clip(level, iso_lo, iso_hi))
        mesh = volume_to_mesh(volume, level=level)
        if state["mesh_actor"] is not None:
            plotter.remove_actor(state["mesh_actor"])
        state["mesh_actor"] = plotter.add_mesh(
            mesh,
            color="#5ba3d9",
            opacity=0.92,
            smooth_shading=True,
            specular=0.35,
            specular_power=18,
            ambient=0.35,
            diffuse=0.85,
            label="reconstruction",
        )
        set_iso_text(level)

    def on_iso_slider(value: float) -> None:
        apply_iso(float(value))

    plotter.add_slider_widget(
        on_iso_slider,
        rng=[iso_lo, iso_hi],
        value=iso_level,
        title="Isosurface level",
        pointa=(0.05, 0.10),
        pointb=(0.55, 0.10),
        style="modern",
        interaction_event="always",
        title_height=0.022,
        title_color="black",
        color="#2c5f8a",
    )

    plotter.add_text(
        "Drag slider to tune density threshold  |  "
        "wireframe = optional reference (--with-phantom)",
        position=(0.05, 0.03),
        font_size=9,
        color="#444444",
        shadow=False,
    )

    apply_iso(iso_level)
    plotter.add_axes()
    if compare_volume is not None:
        plotter.add_legend()
    plotter.view_yz()
    plotter.reset_camera()
    plotter.camera.zoom(1.08)
    plotter.show()


def parse_args() -> argparse.Namespace:
    """CLI for the 3D reconstruction viewer."""
    parser = argparse.ArgumentParser(description="View a 3D reconstruction volume.")
    parser.add_argument(
        "reconstruction",
        nargs="?",
        type=Path,
        default=Path("reconstruction_ab_initio.npz"),
        help="Output from ab_initio_reconstruct.py",
    )
    parser.add_argument(
        "--iso",
        type=float,
        default=None,
        help="Initial isosurface level (default: auto; adjust with slider)",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Optional reference volume (.npz / .npy) shown as wireframe",
    )
    parser.add_argument(
        "--compare-key",
        default="volume",
        help="Array key in compare file (default: volume)",
    )
    parser.add_argument(
        "--with-phantom",
        action="store_true",
        help="Overlay the built-in pyramid phantom for comparison",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: load volume and open the PyVista isosurface window."""
    args = parse_args()
    path = Path(args.reconstruction)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run:\n  python ab_initio_reconstruct.py"
        )

    volume = load_volume(path)
    compare = None

    if args.with_phantom:
        compare = make_demo_volume(size=volume.shape[0])
    elif args.compare is not None:
        compare = load_volume(args.compare, key=args.compare_key)

    title = f"Ab initio reconstruction — {path.name}"
    show_reconstruction(
        volume,
        iso_level=args.iso,
        title=title,
        compare_volume=compare,
    )


if __name__ == "__main__":
    main()
