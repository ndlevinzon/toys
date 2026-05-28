#!/usr/bin/env python3
"""
Assign patch-wise interfacial descriptors to a protein SAS mesh.

Each patch f receives a feature vector x_f (area, centroid, normal, curvatures,
charge, potential, pKa proxy, hydropathy, polarity, H-bond score, dipole, softness).

Example::

..\toys\.venv\Scripts\python.exe parameterize_mesh.py 9yp6.pdb 9yp6.ply --charge-model ff19sb --pka-source propka
"""
from __future__ import annotations

import sys

if sys.version_info < (3, 10):
    sys.exit(
        "Python 3.10+ required.\n"
        f"You ran: {sys.executable} ({sys.version.split()[0]})\n"
        "Use: ..\\toys\\.venv\\Scripts\\python.exe parameterize_mesh.py ..."
    )

import argparse
import json
from pathlib import Path

import numpy as np

from anisotropy.cli_runtime import RunSession, add_logging_arguments, run_main, task_step
from anisotropy.mesh import fit_iterative_mesh
from anisotropy.patches import (
    load_mesh_ply,
    parameterize_mesh,
    save_parameterization,
)
from anisotropy.pdb import load_pdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch-wise mesh parameterization for interfacial modeling.",
    )
    parser.add_argument("pdb", type=Path, help="Input PDB (same frame as mesh)")
    parser.add_argument(
        "mesh",
        nargs="?",
        type=Path,
        default=None,
        help="PLY mesh from fit_protein_mesh.py (omit with --fit-mesh)",
    )
    parser.add_argument(
        "--fit-mesh",
        action="store_true",
        help="Fit SAS mesh from PDB before parameterizing",
    )
    parser.add_argument(
        "--method",
        choices=("auto", "voxel", "exact"),
        default="auto",
        help="Mesh backend if --fit-mesh (default: auto)",
    )
    parser.add_argument(
        "--resolutions",
        type=float,
        nargs="+",
        default=[3.5, 2.5, 1.8],
        help="Grid resolutions if --fit-mesh",
    )
    parser.add_argument(
        "--ph",
        type=float,
        default=7.0,
        help="pH for protonation-aware charge (default: 7.0)",
    )
    parser.add_argument(
        "--pka-source",
        choices=("auto", "propka", "table"),
        default="auto",
        help="pKa for charge/patch features: PROPKA, tables, or auto (default: auto)",
    )
    parser.add_argument(
        "--patch-angle",
        type=float,
        default=25.0,
        help="Max normal angle (deg) for patch growth (default: 25)",
    )
    parser.add_argument(
        "--min-patch-area",
        type=float,
        default=80.0,
        help="Minimum patch area in Å² (default: 80)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("patch_features.npz"),
        help="Output NPZ path",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional JSON summary per patch",
    )
    parser.add_argument(
        "--charge-model",
        choices=("ff19sb", "propka", "table"),
        default="ff19sb",
        help="Charge/dipole source: AMBER ff19SB (default), PROPKA spread, or tables",
    )
    parser.add_argument(
        "--ff19sb-lib",
        type=Path,
        default=None,
        help="Path to amino19.lib (default: bundled utils/ff19SB force field)",
    )
    add_logging_arguments(parser)
    return parser.parse_args()


def main_impl(args: argparse.Namespace, run: RunSession) -> None:
    with task_step(run, "Load PDB"):
        structure = load_pdb(args.pdb)
    run.log(f"Loaded {structure.n_atoms} atoms from {args.pdb}")

    if args.fit_mesh:
        run.log(f"Fitting mesh ({args.method}, resolutions={args.resolutions})...")
        with task_step(run, "Fit SAS mesh"):
            mesh = fit_iterative_mesh(
                structure,
                resolutions=tuple(args.resolutions),
                method=args.method,
            )
        run.log(f"  mesh: {mesh.n_vertices} verts, {mesh.n_faces} faces")
    elif args.mesh is not None:
        with task_step(run, "Load PLY mesh"):
            mesh = load_mesh_ply(args.mesh)
        run.log(f"Loaded mesh {args.mesh} ({mesh.n_vertices} verts)")
    else:
        raise SystemExit("Provide mesh.ply or use --fit-mesh")

    bfactors = np.array(
        [a.bfactor if a.bfactor is not None else np.nan for a in structure.atoms],
        dtype=np.float64,
    )
    atom_bfactor = bfactors if np.isfinite(bfactors).any() else None

    with task_step(run, "Patch parameterization"):
        param = parameterize_mesh(
            mesh,
            structure,
            ph=args.ph,
            normal_angle_deg=args.patch_angle,
            min_patch_area=args.min_patch_area,
            atom_bfactor=atom_bfactor,
            pka_source=args.pka_source,
            pdb_path=str(args.pdb.resolve()),
            charge_model=args.charge_model,
            ff19sb_lib=str(args.ff19sb_lib.resolve()) if args.ff19sb_lib else None,
        )
    backend = param.metadata.get("pka_backend", "?")
    n_pk = param.metadata.get("n_propka_sites", 0)
    charge_model = param.metadata.get("charge_model", args.charge_model)
    run.log(
        f"Parameterized {param.n_patches} patches at pH {args.ph} "
        f"(charges: {charge_model}; pKa: {backend}"
        + (f", {n_pk} ionizable sites)" if backend == "propka" else ")")
    )
    if charge_model == "ff19sb":
        q_tot = param.metadata.get("ff19sb_total_charge_e")
        if q_tot is not None:
            run.log(f"  ff19SB net charge: {q_tot:+.4f} e")

    with task_step(run, "Write NPZ"):
        save_parameterization(args.output, param)
    run.log(f"Wrote {args.output.resolve()}")

    if args.json is not None:
        payload = {
            "pdb": str(args.pdb.resolve()),
            "ph": args.ph,
            "n_patches": param.n_patches,
            "patches": [p.as_vector_dict() for p in param.patches],
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        run.log(f"Wrote {args.json.resolve()}")

    run.log(" patch   area    |q|     h      p      b      u")
    for p in param.patches[:12]:
        run.log(
            f"{p.patch_id:5d} {p.area:7.0f} {abs(p.charge):6.2f} "
            f"{p.hydropathy:6.2f} {p.polar_density:6.2f} {p.hbond_score:6.2f} {p.softness:6.2f}"
        )
    if param.n_patches > 12:
        run.log(f"  ... ({param.n_patches - 12} more patches)")


def main() -> None:
    args = parse_args()
    run_main("parameterize_mesh", args, main_impl)


if __name__ == "__main__":
    main()
