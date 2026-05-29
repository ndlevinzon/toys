#!/usr/bin/env python3
"""
Export a presentation MP4 of the default square-pyramid phantom rotating in place.

Uses the same mesh, top-down orthographic camera, and styling as
``beam_projection_toy.py`` (fixed view from +Z; particle spins about +Y).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

from beam_projection_toy import FIXED_BEAM_DIRECTION_XYZ, make_pyramid_mesh

DEFAULT_COLOR = "#5ba3d9"
DEFAULT_OUTPUT = Path("pyramid_rotation.mp4")


def parse_args() -> argparse.Namespace:
    """CLI for pyramid rotation video export."""
    parser = argparse.ArgumentParser(
        description="Export a rotating square-pyramid phantom MP4 for presentations.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output video path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Length of one revolution in seconds (default: 8).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=96,
        help="Phantom grid size (default: 96, same as the interactive toy).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Video width in pixels (default: 1920).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1088,
        help="Video height in pixels (default: 1088, divisible by 16 for H.264).",
    )
    parser.add_argument(
        "--color",
        type=str,
        default=DEFAULT_COLOR,
        help="Mesh color (default: soft blue on white).",
    )
    parser.add_argument(
        "--show-beam",
        action="store_true",
        help="Draw a faint fixed −Z electron beam through the particle.",
    )
    return parser.parse_args()


def rotation_matrix_y(degrees: float) -> np.ndarray:
    """4×4 VTK user matrix: rotation about +Y (same axis as trackball drag in the toy)."""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("y", degrees, degrees=True).as_matrix()
    return matrix


def _set_presentation_camera(plotter: pv.Plotter) -> None:
    """Match the interactive toy: orthographic top view from +Z."""
    plotter.enable_parallel_projection()
    plotter.view_xy(negative=False)
    plotter.reset_camera()
    plotter.camera.zoom(1.15)


def _add_beam_hint(plotter: pv.Plotter, mesh: pv.PolyData) -> None:
    """Optional fixed beam column along −Z (source above, into the specimen)."""
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    center = np.array(
        [
            0.5 * (xmin + xmax),
            0.5 * (ymin + ymax),
            0.5 * (zmin + zmax),
        ]
    )
    span = max(xmax - xmin, ymax - ymin, zmax - zmin, 1.0)
    prop = FIXED_BEAM_DIRECTION_XYZ / np.linalg.norm(FIXED_BEAM_DIRECTION_XYZ)
    half = 0.9 * span
    line = pv.Line(center - prop * half, center + prop * half)
    tube = line.tube(radius=0.03 * span, n_sides=18)
    plotter.add_mesh(tube, color="#ffd966", opacity=0.28)


def render_rotation_video(
    output: Path,
    duration: float = 8.0,
    fps: int = 30,
    volume_size: int = 96,
    window_size: tuple[int, int] = (1920, 1080),
    color: str = DEFAULT_COLOR,
    show_beam: bool = False,
) -> Path:
    """
    Render one full revolution about +Y to an MP4 file.

    The camera stays fixed (top view); the pyramid rotates like dragging the
    particle in ``beam_projection_toy.py``.
    """
    try:
        import imageio  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Install imageio for MP4 export: pip install 'imageio[ffmpeg]'"
        ) from exc

    n_frames = max(2, int(round(duration * fps)))
    angles = np.linspace(0.0, 360.0, n_frames, endpoint=False)

    mesh = make_pyramid_mesh(volume_size)

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
    if show_beam:
        _add_beam_hint(plotter, mesh)

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
    """Entry point: render MP4 to ``args.output``."""
    args = parse_args()
    path = render_rotation_video(
        output=args.output,
        duration=args.duration,
        fps=args.fps,
        volume_size=args.size,
        window_size=(args.width, args.height),
        color=args.color,
        show_beam=args.show_beam,
    )
    print(f"Wrote {path.resolve()}")


if __name__ == "__main__":
    main()
