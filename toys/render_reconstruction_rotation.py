#!/usr/bin/env python3
"""
Export a presentation MP4 of an ab-initio reconstruction rotating in place.

Matches ``render_bulldog_rotation.py``: one revolution about +Y in 8 s at 30 fps,
top-down orthographic camera, white background. Loads the volume from
``ab_initio_reconstruct.py`` output and contours it as an isosurface.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

from beam_projection_toy import volume_to_mesh
from view_reconstruction import auto_iso_level, load_volume

DEFAULT_COLOR = "#5ba3d9"
DEFAULT_OUTPUT = Path("reconstruction_rotation.mp4")
DEFAULT_INPUT = Path("recon_gt.np.npz")
DEFAULT_DURATION = 8.0
DEFAULT_FPS = 30
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1088


def parse_args() -> argparse.Namespace:
    """CLI for ab-initio reconstruction rotation video export."""
    parser = argparse.ArgumentParser(
        description="Export a rotating ab-initio reconstruction MP4 for presentations.",
    )
    parser.add_argument(
        "reconstruction",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Reconstruction .npz or .npy (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output video path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--volume-key",
        default="volume",
        help="Array key in .npz (default: volume)",
    )
    parser.add_argument(
        "--iso",
        type=float,
        default=None,
        help="Isosurface level (default: auto from histogram)",
    )
    parser.add_argument(
        "--iso-percentile",
        type=float,
        default=88.0,
        help="Auto isosurface percentile when --iso is omitted (default: 88)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help=f"Length of one revolution in seconds (default: {DEFAULT_DURATION}).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Frames per second (default: {DEFAULT_FPS}).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Video width in pixels (default: {DEFAULT_WIDTH}).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Video height in pixels (default: {DEFAULT_HEIGHT}).",
    )
    parser.add_argument(
        "--color",
        type=str,
        default=DEFAULT_COLOR,
        help="Mesh color (default: soft blue on white).",
    )
    return parser.parse_args()


def rotation_matrix_y(degrees: float) -> np.ndarray:
    """4×4 VTK user matrix: rotation about +Y (same axis as the pyramid movie)."""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("y", degrees, degrees=True).as_matrix()
    return matrix


def _set_presentation_camera(plotter: pv.Plotter) -> None:
    """Match render_bulldog_rotation / beam_projection_toy: top view from +Z."""
    plotter.enable_parallel_projection()
    plotter.view_xy(negative=False)
    plotter.reset_camera()
    plotter.camera.zoom(1.15)


def render_reconstruction_rotation_video(
    volume: np.ndarray,
    output: Path,
    *,
    iso_level: float | None = None,
    iso_percentile: float = 88.0,
    duration: float = DEFAULT_DURATION,
    fps: int = DEFAULT_FPS,
    window_size: tuple[int, int] = (DEFAULT_WIDTH, DEFAULT_HEIGHT),
    color: str = DEFAULT_COLOR,
) -> Path:
    """
    Render one full revolution about +Y to an MP4 file.

    Spin rate and duration match ``render_bulldog_rotation.py`` (360° per
    ``duration`` seconds at ``fps``).
    """
    try:
        import imageio  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Install imageio for MP4 export: pip install 'imageio[ffmpeg]'"
        ) from exc

    if iso_level is None:
        iso_level = auto_iso_level(volume, percentile=iso_percentile)

    mesh = volume_to_mesh(volume, level=iso_level)
    if mesh.n_cells < 1:
        raise ValueError(
            f"Isosurface at level {iso_level} is empty; try a lower --iso or "
            f"--iso-percentile."
        )

    n_frames = max(2, int(round(duration * fps)))
    angles = np.linspace(0.0, 360.0, n_frames, endpoint=False)

    plotter = pv.Plotter(
        off_screen=True,
        window_size=list(window_size),
        lighting="three lights",
    )
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")

    actor = plotter.add_mesh(
        mesh,
        color=color,
        opacity=0.92,
        smooth_shading=True,
        specular=0.4,
        specular_power=20,
        ambient=0.35,
        diffuse=0.85,
    )
    _set_presentation_camera(plotter)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    plotter.open_movie(str(output), framerate=fps, quality=9)
    for angle in angles:
        actor.user_matrix = rotation_matrix_y(float(angle))
        plotter.render()
        plotter.write_frame()
    plotter.close()

    return output


def main() -> None:
    """Entry point: load reconstruction and render MP4."""
    args = parse_args()
    path = Path(args.reconstruction)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run:\n  python ab_initio_reconstruct.py"
        )

    volume = load_volume(path, key=args.volume_key)
    iso = args.iso
    if iso is None:
        iso = auto_iso_level(volume, percentile=args.iso_percentile)

    print(
        f"Rendering {args.duration}s @ {args.fps} fps "
        f"({path.name}, iso={iso:.5f})..."
    )
    out = render_reconstruction_rotation_video(
        volume,
        args.output,
        iso_level=iso,
        duration=args.duration,
        fps=args.fps,
        window_size=(args.width, args.height),
        color=args.color,
    )
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
