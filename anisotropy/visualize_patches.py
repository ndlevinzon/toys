#!/usr/bin/env python3
"""
Interactive viewer for patch-wise mesh parameterization.

Example::

    ..\\toys\\.venv\\Scripts\\python.exe parameterize_mesh.py 1CRN.pdb 1crn_sas.ply -o patch_features.npz
    ..\\toys\\.venv\\Scripts\\python.exe visualize_patches.py 1crn_sas.ply patch_features.npz
"""
from __future__ import annotations

import sys

if sys.version_info < (3, 10):
    sys.exit(
        "Python 3.10+ required.\n"
        f"You ran: {sys.executable} ({sys.version.split()[0]})\n"
        "Use: ..\\toys\\.venv\\Scripts\\python.exe visualize_patches.py ..."
    )

import argparse
from pathlib import Path

from anisotropy.cli_runtime import RunSession, add_logging_arguments, run_main, task_step
from anisotropy.patch_visualizer import show_patch_parameterization
from anisotropy.patches import load_mesh_ply, load_parameterization


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive patch-parameter viewer (PyVista checkboxes).",
    )
    p.add_argument("mesh", type=Path, help="PLY mesh (same frame as parameterization)")
    p.add_argument(
        "patches",
        type=Path,
        help="patch_features.npz from parameterize_mesh.py",
    )
    p.add_argument(
        "--arrow-scale",
        type=float,
        default=None,
        help="Arrow length in Angstrom (default: ~12%% of mesh extent)",
    )
    p.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="Save a static PNG instead of opening an interactive window",
    )
    add_logging_arguments(p)
    return p.parse_args()


def main_impl(args: argparse.Namespace, run: RunSession) -> None:
    with task_step(run, "Load mesh"):
        mesh = load_mesh_ply(str(args.mesh))
    with task_step(run, "Load patch NPZ"):
        param = load_parameterization(str(args.patches))
    run.log(
        f"Loaded mesh {args.mesh} ({mesh.n_vertices} verts, {mesh.n_faces} faces)\n"
        f"Loaded {param.n_patches} patches from {args.patches} (pH {param.ph})"
    )

    if args.screenshot is not None:
        with task_step(run, "Render screenshot"):
            show_patch_parameterization(
                mesh,
                param,
                arrow_scale=args.arrow_scale,
                screenshot=str(args.screenshot),
            )
        run.log(f"Wrote {args.screenshot.resolve()}")
    else:
        with task_step(run, "Open interactive viewer"):
            show_patch_parameterization(
                mesh,
                param,
                arrow_scale=args.arrow_scale,
            )


def main() -> None:
    args = parse_args()
    run_main("visualize_patches", args, main_impl)


if __name__ == "__main__":
    main()
