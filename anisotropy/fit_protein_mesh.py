#!/usr/bin/env python3
"""
..\toys\.venv\Scripts\python.exe fit_protein_mesh.py 9yp6.pdb --method voxel --resolutions 3.5 2.5 1.8 --view -o 9yp6.ply
Fit an iterative SAS mesh to a PDB structure and report shape anisotropy.

Example::

    py -3 fit_protein_mesh.py path/to/protein.pdb
    py -3 fit_protein_mesh.py 1crn.pdb --view -o crn_mesh.ply
"""
from __future__ import annotations

import sys

if sys.version_info < (3, 10):
    sys.exit(
        "This script requires Python 3.10 or newer.\n"
        f"You ran: {sys.executable} ({sys.version.split()[0]})\n\n"
        "On Windows, use:\n"
        "  py -3 fit_protein_mesh.py 1CRN.pdb --view\n"
        "Or activate a venv:\n"
        "  .\\.venv\\Scripts\\python.exe fit_protein_mesh.py 1CRN.pdb --view"
    )

import argparse
import json
from pathlib import Path

from anisotropy.cli_runtime import RunSession, add_logging_arguments, run_main, task_step
from anisotropy.mesh import fit_iterative_mesh
from anisotropy.pdb import load_pdb
from anisotropy.sasa import estimate_sasa_area
from anisotropy.shape import shape_anisotropy_from_mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterative SAS mesh fit and shape anisotropy from a PDB file.",
    )
    parser.add_argument("pdb", type=Path, help="Input PDB path")
    parser.add_argument(
        "-o",
        "--output-mesh",
        type=Path,
        default=None,
        help="Optional output mesh (.ply)",
    )
    parser.add_argument(
        "--resolutions",
        type=float,
        nargs="+",
        default=[2.5, 1.5, 1.0],
        help="Grid resolutions for iterative refinement (Å)",
    )
    parser.add_argument(
        "--probe",
        type=float,
        default=1.4,
        help="Probe radius for SAS (Å, default 1.4)",
    )
    parser.add_argument(
        "--method",
        choices=("auto", "voxel", "exact"),
        default="auto",
        help="Meshing backend (default: auto)",
    )
    parser.add_argument(
        "--hetatm",
        action="store_true",
        help="Include HETATM records (ligands, etc.)",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open PyVista viewer (mesh + principal axes)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write shape metrics to JSON",
    )
    add_logging_arguments(parser)
    return parser.parse_args()


def main_impl(args: argparse.Namespace, run: RunSession) -> None:
    with task_step(run, "Load PDB"):
        structure = load_pdb(args.pdb, include_hetatm=args.hetatm)
    run.log(f"Loaded {structure.n_atoms} heavy atoms from {args.pdb}")

    with task_step(run, f"Fit SAS mesh ({args.method})"):
        mesh = fit_iterative_mesh(
            structure,
            resolutions=tuple(args.resolutions),
            probe_radius=args.probe,
            method=args.method,
        )
    run.log(
        f"Mesh: {mesh.n_vertices} vertices, {mesh.n_faces} faces "
        f"(final grid {mesh.resolution_angstrom:.2f} Å, probe {mesh.probe_radius:.2f} Å)"
    )

    with task_step(run, "Estimate SASA"):
        sasa_est = estimate_sasa_area(structure, resolution=1.2, probe_radius=args.probe)
    run.log(f"Estimated SASA (grid): {sasa_est:.0f} Å²")

    with task_step(run, "Shape anisotropy"):
        shape = shape_anisotropy_from_mesh(mesh)
    run.log("Shape anisotropy (SAS mesh):")
    for key, val in shape.as_dict().items():
        run.log(f"  {key}: {val:.4f}")

    if args.output_mesh is not None:
        with task_step(run, "Write mesh PLY"):
            mesh.save_ply(args.output_mesh)
        run.log(f"Wrote mesh {args.output_mesh.resolve()}")

    if args.json is not None:
        payload = {
            "pdb": str(Path(args.pdb).resolve()),
            "n_atoms": structure.n_atoms,
            "mesh_vertices": mesh.n_vertices,
            "mesh_faces": mesh.n_faces,
            **shape.as_dict(),
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        run.log(f"Wrote metrics {args.json.resolve()}")

    if args.view:
        with task_step(run, "Open PyVista viewer"):
            _show_mesh(mesh, shape)


def _show_mesh(mesh, shape) -> None:
    import pyvista as pv

    plotter = pv.Plotter()
    plotter.set_background("white")
    surface = mesh.to_pyvista()
    plotter.add_mesh(surface, color="#6fa8dc", opacity=0.85, smooth_shading=True)

    center = shape.center
    scale = float(shape.axis_lengths.max()) * 0.55
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for i, (axis, color) in enumerate(zip(shape.principal_axes, colors)):
        end = center + scale * axis
        line = pv.Line(center, end)
        plotter.add_mesh(line.tube(radius=scale * 0.02), color=color, label=f"axis {i + 1}")

    plotter.add_legend()
    plotter.add_text(
        f"asphericity={shape.asphericity:.3f}  Rg={shape.radius_gyration:.1f} Å",
        position="upper_left",
    )
    plotter.show()


def main() -> None:
    args = parse_args()
    run_main("fit_protein_mesh", args, main_impl)


if __name__ == "__main__":
    main()
