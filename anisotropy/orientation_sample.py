#!/usr/bin/env python3
r"""
..\toys\.venv\Scripts\python.exe orientation_sample.py 9yp6.pdb 9yp6.ply -n 800 --beta 1.0 --grid-spacing 3.5 --slab-thickness 300 --pka-source propka --outdir orientation_diagnostics --beta auto

Sample rigid-body orientations of a protein mesh at a cryo-EM air–water interface.

Hamiltonian couplings and geometry default from ``ising_params.yaml`` in this
directory (override any key via CLI; see --help).

    H = H_solv + H_hp/pol/HB + H_el + H_film (+ H_flex)
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np

from anisotropy.ising_params import (
    DEFAULT_ISING_PARAMS_PATH,
    apply_cli_overrides,
    load_ising_params,
)
from anisotropy.mesh import ProteinMesh, fit_iterative_mesh
from anisotropy.patches import load_mesh_ply, parameterize_mesh
from anisotropy.pdb import load_pdb
from anisotropy.shape import shape_anisotropy_from_mesh
from anisotropy.lattice_solvent_hamiltonian import (
    CanonicalInteriorCache,
    CartesianLattice,
    evaluate_hybrid_hamiltonian,
    occupancy_binary_template,
    precompute_solvation_energy,
)
from anisotropy.orientation_diagnostics import (
    calibrate_beta_auto,
    inplane_rotation_degrees,
    rotation_for_viewing_angles,
    save_orientation_distribution_plots,
    summarize_orientation_sampling,
    viewing_angles_degrees,
    viewing_angles_radians,
    write_orientation_sampling_report,
)
from anisotropy.fast_orientation_eval import FastOrientationEvaluator
from anisotropy.orientation_parallel import (
    default_worker_count,
    parallel_mcmc_chains,
    parallel_uniform_samples,
)
from anisotropy.orientation_mcmc import (
    OrientationSample,
    run_hybrid_sampling,
    run_mcmc_from_seed_pool,
    run_mcmc_only,
    run_uniform_batch,
    samples_to_arrays,
)
from anisotropy.orientation_multimodal import (
    importance_weights_to_beta,
    run_multimodal_from_seeds,
)
from anisotropy.cli_runtime import RunSession, add_logging_arguments, run_main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Orientation sampling for protein at cryo AWI (Hamiltonian from ising_params.yaml).",
    )
    p.add_argument("pdb", type=Path, help="Input PDB (same frame as mesh)")
    p.add_argument(
        "mesh",
        nargs="?",
        type=Path,
        default=None,
        help="PLY mesh (omit with --fit-mesh)",
    )
    p.add_argument("--fit-mesh", action="store_true", help="Fit SAS mesh from PDB first")
    p.add_argument(
        "--method",
        choices=("auto", "voxel", "exact"),
        default="auto",
        help="Mesh backend if --fit-mesh",
    )
    p.add_argument(
        "--resolutions",
        type=float,
        nargs="+",
        default=[3.5, 2.5, 1.8],
        help="Grid resolutions if --fit-mesh",
    )
    p.add_argument("--ph", type=float, default=7.0, help="pH for PROPKA/table charges")
    p.add_argument(
        "--pka-source",
        choices=("auto", "propka", "table"),
        default="auto",
        help="pKa backend for patch chemistry",
    )
    p.add_argument(
        "--ising-params",
        type=Path,
        default=None,
        help=f"YAML parameter file (default: {DEFAULT_ISING_PARAMS_PATH.name})",
    )

    # Overrides (None = use value from ising_params.yaml)
    p.add_argument("--slab-thickness", type=float, default=None, help="Override slab.thickness_angstrom")
    p.add_argument("--grid-spacing", type=float, default=None, help="Override lattice.grid_spacing_angstrom")
    p.add_argument("--pad-xy", type=float, default=None, help="Override lattice.pad_xy_angstrom")
    p.add_argument("--pad-z", type=float, default=None, help="Override lattice.pad_z_angstrom")
    p.add_argument(
        "--z-center",
        type=float,
        default=None,
        help="Override lattice.protein_z_center_angstrom",
    )
    p.add_argument("-n", "--n-samples", type=int, default=None, help="Override sampling.n_samples")
    p.add_argument("--seed", type=int, default=None, help="Override sampling.seed")
    p.add_argument(
        "--beta",
        default=None,
        help="Inverse temperature, or 'auto' to calibrate from energy spread (see --beta-target-ess)",
    )
    p.add_argument(
        "--beta-target-ess",
        type=float,
        default=None,
        help="Target effective sample size when --beta auto (default from YAML: 20)",
    )
    p.add_argument(
        "--beta-auto-method",
        choices=("ess", "top10", "rank2"),
        default=None,
        help="Beta auto rule: ess (default), top10, or rank2",
    )
    p.add_argument(
        "--sampling-strategy",
        choices=("uniform", "mcmc", "hybrid"),
        default=None,
        help="uniform | mcmc | hybrid (default from YAML: hybrid)",
    )
    p.add_argument(
        "--n-uniform",
        type=int,
        default=None,
        help="Uniform SO(3) draws (hybrid / beta calibration pool)",
    )
    p.add_argument("--mcmc-chains", type=int, default=None, help="Number of parallel MCMC chains")
    p.add_argument("--mcmc-steps", type=int, default=None, help="Metropolis steps per chain")
    p.add_argument("--mcmc-burn-in", type=int, default=None, help="Burn-in steps per chain")
    p.add_argument("--mcmc-thin", type=int, default=None, help="Thinning interval after burn-in")
    p.add_argument(
        "--mcmc-step-deg",
        type=float,
        default=None,
        help="Initial random-walk step size on SO(3) (degrees)",
    )
    p.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help="CPU workers for parallel energy eval (default: all cores)",
    )
    p.add_argument("--J", type=float, default=None, help="Override solv.J")
    p.add_argument("--mu", type=float, default=None, help="Override solv.mu")
    p.add_argument("--u-film", type=float, default=None, help="Override solv.u_film_scale")
    p.add_argument("--lambda-hp", type=float, default=None, help="Override hydration.lambda_hp")
    p.add_argument("--lambda-p", type=float, default=None, help="Override hydration.lambda_p")
    p.add_argument("--lambda-hb", type=float, default=None, help="Override hydration.lambda_hb")
    p.add_argument("--lambda-film", type=float, default=None, help="Override film.lambda_film")
    p.add_argument("--eta-flex", type=float, default=None, help="Override flex.eta_flex")
    p.add_argument(
        "--homogeneous-epsilon",
        type=float,
        default=None,
        help="Override electrostatics.homogeneous_epsilon",
    )
    p.add_argument(
        "--homogeneous-kappa",
        type=float,
        default=None,
        help="Override electrostatics.homogeneous_kappa",
    )

    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("orientation_diagnostics"),
        help="Output directory",
    )
    p.add_argument(
        "--no-render",
        action="store_true",
        help="Skip PyVista snapshots (sets output.render_snapshots false)",
    )
    p.add_argument(
        "--n-render-poses",
        type=int,
        default=None,
        help="Number of most/least probable orientations to render (default from YAML: 10)",
    )
    p.add_argument(
        "--canonical-pad",
        type=float,
        default=None,
        help="Override lattice.canonical_interior_pad_angstrom",
    )
    p.add_argument(
        "--slow-voxelization",
        action="store_true",
        help="Legacy: run PyVista interior test on every pose (much slower)",
    )
    add_logging_arguments(p)
    return p.parse_args()


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation in SO(3) via random unit quaternion."""
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    x, y, z, w = q1, q2, q3, q4
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def euler_zyx_from_R(R: np.ndarray) -> tuple[float, float, float]:
    """Intrinsic Z-Y-X Euler angles (yaw, pitch, roll) in radians."""
    R = np.asarray(R, dtype=np.float64)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-12:
        yaw = math.atan2(R[1, 0], R[0, 0])
        pitch = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[2, 1], R[2, 2])
    else:
        yaw = math.atan2(-R[0, 1], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        roll = 0.0
    return yaw, pitch, roll


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _run_mcmc_from_seeds(
    pre: list[OrientationSample],
    *,
    fast_eval: FastOrientationEvaluator | None,
    evaluate_pose_mcmc,
    rng: np.random.Generator,
    beta: float,
    mcmc_cfg,
    ising,
    n_workers: int | None,
    use_parallel: bool,
) -> tuple[list[OrientationSample], object | None, dict | None]:
    """Launch MCMC / annealing / replica-exchange from seed pool."""
    mode = str(getattr(mcmc_cfg, "mode", "fixed_beta")).lower()
    multimodal_meta: dict | None = None

    if mode in ("simulated_annealing", "replica_exchange", "annealing", "replica"):
        if mode == "annealing":
            mode = "simulated_annealing"
        elif mode == "replica":
            mode = "replica_exchange"
        kept, multimodal_meta = run_multimodal_from_seeds(
            pre,
            evaluate_pose_mcmc,
            rng,
            mode=mode,
            beta_target=beta,
            mcmc_cfg=mcmc_cfg,
            anneal_cfg=mcmc_cfg.annealing,
            replica_cfg=mcmc_cfg.replica_exchange,
        )
        return kept, None, multimodal_meta

    step_rad = float(np.deg2rad(mcmc_cfg.step_deg))
    n_pool = min(mcmc_cfg.seed_pool, len(pre))
    order = np.argsort([s.energy for s in pre])[:n_pool]
    pick = np.linspace(0, n_pool - 1, mcmc_cfg.n_chains).astype(int)
    R0s = [pre[int(order[int(i)])].rotation for i in pick]
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(mcmc_cfg.n_chains)]

    if (
        use_parallel
        and fast_eval is not None
        and ising.performance.parallel_mcmc_chains
        and mcmc_cfg.n_chains > 1
    ):
        kept, summary = parallel_mcmc_chains(
            fast_eval,
            R0s,
            beta=beta,
            n_steps=mcmc_cfg.steps_per_chain,
            burn_in=mcmc_cfg.burn_in,
            thin=mcmc_cfg.thin,
            step_rad=step_rad,
            seeds=seeds,
            target_acceptance=mcmc_cfg.target_acceptance,
            n_workers=n_workers,
        )
        for s in kept:
            if s.beta_at_sample is None:
                s.beta_at_sample = float(beta)
        return kept, summary, None

    kept, summary = run_mcmc_from_seed_pool(
        pre,
        evaluate_pose_mcmc,
        rng,
        beta=beta,
        n_chains=mcmc_cfg.n_chains,
        mcmc_steps_per_chain=mcmc_cfg.steps_per_chain,
        mcmc_burn_in=mcmc_cfg.burn_in,
        mcmc_thin=mcmc_cfg.thin,
        mcmc_step_deg=mcmc_cfg.step_deg,
        mcmc_target_acceptance=mcmc_cfg.target_acceptance,
    )
    for s in kept:
        if s.beta_at_sample is None:
            s.beta_at_sample = float(beta)
    return kept, summary, None


def select_pose_indices_by_weight(
    weights: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (most_probable_indices, least_probable_indices), each length <= n."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    k = int(min(max(n, 0), len(w)))
    if k == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    most = np.argsort(-w, kind="stable")[:k]
    least = np.argsort(w, kind="stable")[:k]
    return most, least


def build_pose_entry(
    rank: int,
    idx: int,
    Rk: np.ndarray,
    t_base: np.ndarray,
    energy: float,
    weight: float,
    resk,
    *,
    category: str = "",
) -> dict:
    """One pose record for JSON export."""
    yaw, pitch, roll = euler_zyx_from_R(Rk)
    az_deg, el_deg = viewing_angles_degrees(Rk)
    return {
        "rank": rank,
        "category": category,
        "index": int(idx),
        "H_total": float(energy),
        "weight": float(weight),
        "viewing_azimuth_deg": az_deg,
        "viewing_elevation_deg": el_deg,
        "inplane_rotation_deg": float(inplane_rotation_degrees(Rk)),
        "pose": {
            "R": Rk.tolist(),
            "t": t_base.tolist(),
            "euler_zyx_rad": [yaw, pitch, roll],
            "euler_zyx_deg": [
                math.degrees(yaw),
                math.degrees(pitch),
                math.degrees(roll),
            ],
        },
        "energy_terms": {
            "H_solv": float(resk.H_solv),
            "H_hp_pol_hbond": float(resk.H_hp_pol_hbond),
            "H_el": float(resk.H_el),
            "H_film": float(resk.H_film),
            "H_flex": float(resk.H_flex),
            "solv_terms": resk.solv_terms,
            "hydration_terms": resk.hydration_terms,
            "electrostatic_terms": resk.electrostatic_terms,
        },
    }


def save_plots(
    outdir: Path,
    energies: np.ndarray,
    tilts: np.ndarray,
    weights: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(energies, lw=0.8)
    ax.set_xlabel("sample index")
    ax.set_ylabel("H")
    ax.set_title("Energy trace")
    fig.tight_layout()
    fig.savefig(outdir / "diagnostics_energy_trace.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(energies, bins=min(40, max(10, len(energies) // 15)), color="steelblue", alpha=0.85)
    ax.set_xlabel("H")
    ax.set_ylabel("count")
    ax.set_title("Energy histogram")
    fig.tight_layout()
    fig.savefig(outdir / "diagnostics_energy_hist.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(tilts, energies, c=weights, cmap="viridis", s=12, alpha=0.75)
    ax.set_xlabel("tilt angle (deg)")
    ax.set_ylabel("H")
    ax.set_title("Tilt–energy landscape (color ~ relative weight)")
    fig.colorbar(sc, ax=ax, label="weight")
    fig.tight_layout()
    fig.savefig(outdir / "diagnostics_tilt_energy.png", dpi=200)
    plt.close(fig)


def render_system_snapshot(
    outdir: Path,
    mesh: ProteinMesh,
    R: np.ndarray,
    t: np.ndarray,
    slab_thickness: float,
    *,
    filename: str = "system_view.png",
    subdirectory: str | None = None,
) -> Path | None:
    """
    PyVista snapshot with camera along **lab +Z** (looking down the cryo-EM axis).

    Uses ``view_xy()`` — same convention as the reference ``reference_view_az0_el0`` pose.
    Returns the path written, or None if PyVista is unavailable.
    """
    try:
        import pyvista as pv
    except Exception:
        return None

    v = mesh.vertices.astype(np.float64)
    f = mesh.faces.astype(np.int64)
    nf = int(f.shape[0])
    cells = np.hstack([np.full((nf, 1), 3, dtype=np.int64), f])
    surf = pv.PolyData(v, cells).triangulate().clean()
    surf.points = (surf.points @ R.T) + t.reshape(1, 3)

    bounds = surf.bounds
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    box = pv.Box(bounds=(xmin, xmax, ymin, ymax, 0.0, slab_thickness))

    pl = pv.Plotter(off_screen=True, window_size=(900, 650))
    pl.set_background("white")
    pl.add_mesh(box, color="lightblue", opacity=0.15, show_edges=True)
    pl.add_mesh(surf, color="tomato", opacity=0.9)
    scene_bounds = [
        min(box.bounds[0], xmin),
        max(box.bounds[1], xmax),
        min(box.bounds[2], ymin),
        max(box.bounds[3], ymax),
        min(box.bounds[4], zmin),
        max(box.bounds[5], zmax),
    ]
    pl.view_xy()
    pl.reset_camera(bounds=scene_bounds)
    dest = outdir / filename if subdirectory is None else outdir / subdirectory / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    pl.show(screenshot=str(dest))
    return dest


def write_weighted_orientation_renders(
    outdir: Path,
    mesh: ProteinMesh,
    poses: list[dict],
    t_base: np.ndarray,
    slab_thickness: float,
    *,
    subdirectory: str = "views_z_down",
    name_prefix: str,
) -> list[Path]:
    """Render each pose dict (with ``pose.R``) looking down lab +Z."""
    written: list[Path] = []
    for entry in poses:
        Rk = np.asarray(entry["pose"]["R"], dtype=np.float64)
        w = float(entry["weight"])
        rank = int(entry["rank"])
        fname = f"{name_prefix}_rank{rank:02d}_w{w:.4e}.png"
        path = render_system_snapshot(
            outdir,
            mesh,
            Rk,
            t_base,
            slab_thickness,
            filename=fname,
            subdirectory=subdirectory,
        )
        if path is not None:
            written.append(path)
    return written


def main() -> None:
    args = parse_args()
    run_main("orientation_sample", args, main_impl)


def main_impl(args: argparse.Namespace, run: RunSession) -> None:
    ensure_dir(args.outdir)

    yaml_path = args.ising_params or DEFAULT_ISING_PARAMS_PATH
    ising = apply_cli_overrides(load_ising_params(yaml_path), args)
    run.log(f"Ising parameters: {ising.source_path}")

    structure = load_pdb(args.pdb)

    if args.fit_mesh:
        mesh = fit_iterative_mesh(
            structure, resolutions=tuple(args.resolutions), method=args.method
        )
    elif args.mesh is not None:
        mesh = load_mesh_ply(str(args.mesh))
    else:
        raise SystemExit("Provide mesh.ply or use --fit-mesh")

    param = parameterize_mesh(
        mesh,
        structure,
        ph=float(args.ph),
        pka_source=str(args.pka_source),
        pdb_path=str(args.pdb.resolve()),
    )

    slab = ising.build_slab()
    coeffs = ising.to_hybrid_couplings()

    v = mesh.vertices
    bb_min = v.min(axis=0)
    bb_max = v.max(axis=0)
    pad_xy = ising.lattice.pad_xy_angstrom
    pad_z = ising.lattice.pad_z_angstrom
    slab_thickness = ising.slab.thickness_angstrom

    origin = np.array(
        [bb_min[0] - pad_xy, bb_min[1] - pad_xy, -pad_z],
        dtype=np.float64,
    )
    extent = np.array(
        [
            (bb_max[0] - bb_min[0]) + 2 * pad_xy,
            (bb_max[1] - bb_min[1]) + 2 * pad_xy,
            slab_thickness + 2 * pad_z,
        ],
        dtype=np.float64,
    )
    h = ising.lattice.grid_spacing_angstrom
    shape = tuple((np.ceil(extent / h)).astype(int).tolist())
    lattice = CartesianLattice(origin=origin, spacing=h, shape=shape)  # type: ignore[arg-type]

    com = mesh.center_of_mass()
    xy_center = origin[:2] + 0.5 * extent[:2]
    z_center = ising.protein_z_center()
    if z_center is None:
        z_center = ising.default_protein_z_center()
    t_base = np.array([xy_center[0], xy_center[1], z_center], dtype=np.float64) - com

    occ = occupancy_binary_template(
        lattice, solvent_z_within=(0.0, slab_thickness)
    ).astype(np.float64)

    lab_xyz_flat = lattice.grid_centers_xyz().reshape(-1, 3).astype(np.float64)
    rism_cfg = ising.rism.to_rism_module_params()
    shell_cache = None
    canon_cache = None
    canon_pad = ising.lattice.canonical_interior_pad_angstrom
    eye3 = np.eye(3, dtype=np.float64)

    if rism_cfg.enabled:
        from anisotropy.rism_solvation import (
            CanonicalShellCache,
            precompute_outer_solvation_only,
        )

        shell_cache = CanonicalShellCache.build(
            mesh,
            spacing=h,
            pad_angstrom=canon_pad,
            first_shell_angstrom=rism_cfg.first_shell_angstrom,
        )
        canon_cache = shell_cache.interior
        h_solv_and_terms = precompute_outer_solvation_only(
            occ,
            lattice,
            slab,
            coeffs,
            shell_cache,
            eye3,
            t_base,
            occupancy_mode=ising.solv.occupancy_mode,
            confinement_penalty_outside=ising.confinement.penalty_outside,
            confinement_interface_softness=ising.confinement.interface_softness_angstrom,
            lab_xyz_flat=lab_xyz_flat,
        )
        run.log(
            f"Layered H_solv: 3D-RISM-inspired first shell (<= {rism_cfg.first_shell_angstrom:.1f} A) "
            f"+ Ising outer shell (H_outer={h_solv_and_terms[0]:.4g}); RISM recomputed per pose"
        )
    else:
        h_solv_and_terms = precompute_solvation_energy(
            occ,
            lattice,
            slab,
            coeffs,
            occupancy_mode=ising.solv.occupancy_mode,
            confinement_penalty_outside=ising.confinement.penalty_outside,
            confinement_interface_softness=ising.confinement.interface_softness_angstrom,
        )

    if not args.slow_voxelization and canon_cache is None:
        run.log(
            "Fast path: canonical interior cache + reused H_solv "
            f"(canonical pad={canon_pad:.1f} Ang; use --slow-voxelization for legacy PyVista per pose)"
        )
        canon_cache = CanonicalInteriorCache.build(
            mesh,
            spacing=h,
            pad_angstrom=canon_pad,
        )
    elif args.slow_voxelization:
        run.log("Slow path: PyVista voxelization every pose (still reuses H_solv).")

    pb_solver = None
    if str(ising.electrostatics.method).lower() == "pb_slab":
        from anisotropy.pb_slab_solver import SlabPBSolver

        pb_solver = SlabPBSolver.from_lattice(
            lattice,
            slab,
            ising.electrostatics.to_pb_solver_params(),
            lab_xyz_flat=lab_xyz_flat,
            homogeneous_epsilon=ising.electrostatics.homogeneous_epsilon,
            homogeneous_kappa=ising.electrostatics.homogeneous_kappa,
        )
        run.log(
            f"Electrostatics: linearized PB on AWI slab (coarse_factor="
            f"{ising.electrostatics.pb_coarse_factor}, "
            f"grid {pb_solver.lattice.shape}, h={pb_solver.lattice.spacing:.2f} A)"
        )
        if ising.performance.parallel_mcmc_chains:
            run.log(
                "  NOTE: parallel MCMC worker processes do not yet rebuild the PB solver; "
                "use parallel_mcmc_chains: false for full PB electrostatics on all chains."
            )

    rng = np.random.default_rng(ising.sampling.seed)
    shape0 = shape_anisotropy_from_mesh(mesh)
    e1 = np.asarray(shape0.principal_axes[0], dtype=np.float64).reshape(3)
    slab_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    strategy = str(ising.sampling.strategy).lower()
    mcmc_cfg = ising.sampling.mcmc
    n_uniform = int(ising.sampling.n_uniform)
    if strategy == "uniform":
        n_uniform = int(ising.sampling.n_samples)

    fast_eval: FastOrientationEvaluator | None = None
    if (
        not args.slow_voxelization
        and canon_cache is not None
        and ising.performance.use_fast_evaluator
    ):
        fast_eval = FastOrientationEvaluator.build(
            param=param,
            occupancy=occ,
            lattice=lattice,
            slab=slab,
            coeffs=coeffs,
            canon_cache=canon_cache,
            lab_xyz_flat=lab_xyz_flat,
            t_base=t_base,
            h_solv_and_terms=h_solv_and_terms,
            ising_params=ising,
            shell_cache=shell_cache,
            pb_solver=pb_solver,
        )
        if pb_solver is not None:
            run.log("Fast evaluator: slab PB electrostatics (per-pose charge deposition)")
        else:
            run.log(
                "Fast evaluator: vectorized rays + electrostatic cutoff "
                f"({fast_eval.el_pair_cutoff} Ang)"
            )
        if fast_eval.use_invariant_pair:
            run.log(
                f"  rotation-invariant H_el pair precomputed: {fast_eval.h_el_pair_const:.4g} "
                "(intrinsic slab terms only per pose)"
            )
        nw = ising.performance.parallel_workers
        if args.parallel_workers is not None:
            nw = args.parallel_workers
        run.log(
            f"  parallel workers: {default_worker_count(1000, nw)} "
            f"(chains parallel={ising.performance.parallel_mcmc_chains})"
        )

    mcmc_energy_only = ising.performance.mcmc_energy_only
    _eval_serial = 0

    def evaluate_pose(R: np.ndarray, *, energy_only: bool = False):
        nonlocal _eval_serial
        _eval_serial += 1
        tag = "energy-only" if energy_only else "Hamiltonian"
        run.progress.begin(f"{tag} eval #{_eval_serial}")
        try:
            if fast_eval is not None:
                if energy_only:
                    return float(fast_eval.energy(R)), None
                res = fast_eval.evaluate(R)
                assert not isinstance(res, float)
                return float(res.H_total), res
            res = evaluate_hybrid_hamiltonian(
                occ,
                lattice,
                mesh,
                R,
                t_base,
                param,
                slab,
                coeffs,
                occupancy_mode=ising.solv.occupancy_mode,
                canonical_interior=None if args.slow_voxelization else canon_cache,
                shell_cache=shell_cache,
                lab_xyz_flat=lab_xyz_flat,
                h_solv_and_terms=h_solv_and_terms,
                use_slow_pyvista_voxel=bool(args.slow_voxelization),
                ising_params=ising,
                pb_solver=pb_solver,
            )
            return float(res.H_total), res
        finally:
            run.progress.complete()

    def evaluate_pose_mcmc(R: np.ndarray):
        return evaluate_pose(R, energy_only=bool(mcmc_energy_only and fast_eval is not None))

    mcmc_summary = None
    multimodal_meta: dict | None = None
    run.log(f"Sampling strategy: {strategy}  (mcmc.mode={mcmc_cfg.mode})")

    n_workers = ising.performance.parallel_workers
    if args.parallel_workers is not None:
        n_workers = args.parallel_workers
    use_parallel = fast_eval is not None and (n_workers is None or n_workers != 0)

    if strategy == "uniform":
        run.progress.set_total(n_uniform)
        if use_parallel and n_uniform > 32:
            run.log(f"Parallel uniform sampling: {n_uniform} rotations")

            def _on_uniform_done(i: int, n_tot: int) -> None:
                run.progress.begin(f"uniform SO(3) energy {i + 1}/{n_tot}")
                run.progress.complete()

            samples = parallel_uniform_samples(
                fast_eval,
                n_uniform,
                rng,
                n_workers=n_workers,
                on_energy_complete=_on_uniform_done,
            )
            for s in samples:
                if s.result is None:
                    _, s.result = evaluate_pose(s.rotation)
        else:
            samples = []
            for idx in range(n_uniform):
                R = random_rotation_matrix(rng)
                e, res = evaluate_pose(R)
                samples.append(OrientationSample(R, e, res, "uniform"))
    elif strategy == "mcmc":
        beta_mcmc = float(ising.sampling.beta)
        if ising.sampling.beta_auto:
            run.log("MCMC-only: short uniform preflight for beta calibration...")
            pre = run_uniform_batch(
                min(200, max(80, n_uniform // 2)), evaluate_pose, rng
            )
            e_pre = np.array([s.energy for s in pre], dtype=np.float64)
            beta_mcmc, _ = calibrate_beta_auto(e_pre, target_ess=ising.sampling.beta_target_ess)
            run.log(f"  preflight beta: {beta_mcmc:.6g}")
        run.log(
            f"MCMC: {mcmc_cfg.n_chains} chains x {mcmc_cfg.steps_per_chain} steps "
            f"(burn={mcmc_cfg.burn_in}, thin={mcmc_cfg.thin})"
        )
        mode = str(mcmc_cfg.mode).lower()
        if mode in ("simulated_annealing", "replica_exchange", "annealing", "replica"):
            pre = run_uniform_batch(
                min(200, max(80, n_uniform // 2)), evaluate_pose, rng
            )
            mcmc_kept, mcmc_summary, multimodal_meta = _run_mcmc_from_seeds(
                pre,
                fast_eval=fast_eval,
                evaluate_pose_mcmc=evaluate_pose_mcmc,
                rng=rng,
                beta=beta_mcmc,
                mcmc_cfg=mcmc_cfg,
                ising=ising,
                n_workers=n_workers,
                use_parallel=use_parallel,
            )
            samples = list(pre) + mcmc_kept
        else:
            samples, mcmc_summary = run_mcmc_only(
                evaluate_pose_mcmc,
                rng,
                beta=beta_mcmc,
                n_chains=mcmc_cfg.n_chains,
                mcmc_steps_per_chain=mcmc_cfg.steps_per_chain,
                mcmc_burn_in=mcmc_cfg.burn_in,
                mcmc_thin=mcmc_cfg.thin,
                mcmc_step_deg=mcmc_cfg.step_deg,
                mcmc_target_acceptance=mcmc_cfg.target_acceptance,
            )
        if multimodal_meta is not None:
            run.log(
                f"  Multimodal mode: {multimodal_meta.get('mode')}  "
                f"(beta_target={multimodal_meta.get('beta_target'):.6g})"
            )
        elif mcmc_summary is not None:
            run.log(
                f"  MCMC acceptance: {mcmc_summary.mean_acceptance_rate:.3f} "
                f"({mcmc_summary.total_accepted}/{mcmc_summary.total_proposed})"
            )
    else:
        beta_mcmc = float(ising.sampling.beta)
        if ising.sampling.beta_auto:
            run.log(f"Hybrid: {n_uniform} uniform draws, beta auto, then MCMC refinement...")
        else:
            run.log(
                f"Hybrid: {n_uniform} uniform + MCMC "
                f"({mcmc_cfg.n_chains} chains x {mcmc_cfg.steps_per_chain} steps)"
            )
        if ising.sampling.beta_auto:
            pre = run_uniform_batch(n_uniform, evaluate_pose, rng)
            e_pre = np.array([s.energy for s in pre], dtype=np.float64)
            beta_mcmc, _ = calibrate_beta_auto(
                e_pre,
                target_ess=ising.sampling.beta_target_ess,
                method=ising.sampling.beta_auto_method,
            )
            run.log(f"  beta for MCMC: {beta_mcmc:.6g}")
            mcmc_kept, mcmc_summary, multimodal_meta = _run_mcmc_from_seeds(
                pre,
                fast_eval=fast_eval,
                evaluate_pose_mcmc=evaluate_pose_mcmc,
                rng=rng,
                beta=beta_mcmc,
                mcmc_cfg=mcmc_cfg,
                ising=ising,
                n_workers=n_workers,
                use_parallel=use_parallel,
            )
            samples = list(pre) + mcmc_kept
        else:
            pre = run_uniform_batch(n_uniform, evaluate_pose, rng)
            mcmc_kept, mcmc_summary, multimodal_meta = _run_mcmc_from_seeds(
                pre,
                fast_eval=fast_eval,
                evaluate_pose_mcmc=evaluate_pose_mcmc,
                rng=rng,
                beta=beta_mcmc,
                mcmc_cfg=mcmc_cfg,
                ising=ising,
                n_workers=n_workers,
                use_parallel=use_parallel,
            )
            samples = list(pre) + mcmc_kept
        if multimodal_meta is not None:
            mode = multimodal_meta.get("mode", "")
            run.log(f"  Multimodal mode: {mode}  (beta_target={multimodal_meta.get('beta_target'):.6g})")
            for ch in multimodal_meta.get("chains", []):
                st = ch.get("stats")
                if ch.get("type") == "replica" and st is not None:
                    run.log(
                        f"    replica swap rate: {st.mean_swap_rate:.3f} "
                        f"({st.n_swap_accepted}/{st.n_swap_attempts})"
                    )
        elif mcmc_summary is not None:
            run.log(
                f"  MCMC acceptance: {mcmc_summary.mean_acceptance_rate:.3f} "
                f"({mcmc_summary.total_accepted}/{mcmc_summary.total_proposed})"
            )

    Rs, energies, results, sample_sources, betas_at = samples_to_arrays(samples)
    n_samples = len(samples)
    sources_arr = np.array(sample_sources)
    refined_mask = np.isin(sources_arr, ("mcmc", "anneal", "replica"))
    run.log(
        f"Total samples: {n_samples}  "
        f"(uniform={int(np.sum(sources_arr == 'uniform'))}, "
        f"refined={int(np.sum(refined_mask))})"
    )

    tilts = np.zeros(n_samples, dtype=np.float64)
    for i, R in enumerate(Rs):
        ax = R @ e1
        cos_th = float(np.clip(np.dot(ax, slab_normal) / (np.linalg.norm(ax) + 1e-12), -1.0, 1.0))
        tilts[i] = float(np.degrees(np.arccos(abs(cos_th))))

    Emin = float(energies.min())
    beta_calibration: dict | None = None
    if ising.sampling.beta_auto and strategy == "uniform":
        target_ess = ising.sampling.beta_target_ess
        if args.beta_target_ess is not None:
            target_ess = float(args.beta_target_ess)
        method = ising.sampling.beta_auto_method
        if args.beta_auto_method is not None:
            method = str(args.beta_auto_method)
        beta, beta_calibration = calibrate_beta_auto(
            energies,
            target_ess=target_ess,
            method=method,
        )
        ess_all = beta_calibration.get("achieved_ess_all_samples")
        ess_loc = beta_calibration.get(
            "achieved_ess_local_pool", beta_calibration.get("achieved_ess")
        )
        run.log(
            f"beta auto: {beta:.6g}  (method={method}, target_ESS={target_ess:g}, "
            f"ESS_pool={ess_loc:.2f}, ESS_all={ess_all:.2f})"
        )
    elif ising.sampling.beta_auto and strategy in ("hybrid", "mcmc"):
        beta = float(beta_mcmc)
        beta_calibration = {"method": "preflight_for_mcmc", "beta": beta}
    else:
        beta = float(ising.sampling.beta)

    beta_draw = betas_at.copy()
    uniform_mask = sources_arr == "uniform"
    beta_draw[np.isnan(beta_draw) & uniform_mask] = 0.0
    beta_draw[np.isnan(beta_draw)] = float(beta)
    multimodal_mode = str(mcmc_cfg.mode).lower() not in ("fixed_beta", "mcmc", "")
    if multimodal_mode or np.any(np.abs(beta_draw - float(beta)) > 1e-30 * max(abs(beta), 1.0)):
        weights = importance_weights_to_beta(energies, beta_draw, float(beta))
        run.log("  Boltzmann weights: importance reweighting to beta_target")
    else:
        logw = -beta * (energies - Emin)
        logw -= float(np.max(logw))
        weights = np.exp(logw)
        weights /= float(np.sum(weights) + 1e-30)

    save_plots(args.outdir, energies, tilts, weights)
    orient_paths = save_orientation_distribution_plots(
        args.outdir, Rs, weights, mcmc_mask=refined_mask
    )
    r_ref = rotation_for_viewing_angles(0.0, 0.0, 0.0)
    orient_paths["reference_view"] = args.outdir / "reference_view_az0_el0.png"
    az_ref, el_ref = viewing_angles_radians(r_ref)
    ref_path = args.outdir / "reference_view_az0_el0.png"
    render_system_snapshot(
        args.outdir,
        mesh,
        r_ref,
        t_base,
        slab_thickness,
        filename=ref_path.name,
    )
    run.log(
        f"Reference pose (az={az_ref:.4f}, el={el_ref:.4f}) rad at plot origin; "
        "lab +Z viewing ≈ particle +X"
    )
    if ref_path.is_file():
        run.log(f"  wrote: {ref_path.resolve()}")
    else:
        run.log("  reference render skipped (PyVista not available)")
    sampling_summary = summarize_orientation_sampling(
        energies, weights, Rs, beta=beta
    )
    report_path = write_orientation_sampling_report(args.outdir, sampling_summary)

    n_render = int(ising.output.n_orientation_renders)
    idx_most, idx_least = select_pose_indices_by_weight(weights, n_render)
    map_idx = int(np.argmax(weights))

    for idx in np.unique(np.concatenate([idx_most, idx_least, np.array([map_idx])])):
        if results[int(idx)] is None:
            _, results[int(idx)] = evaluate_pose(Rs[int(idx)])

    def _entries(indices: np.ndarray, category: str) -> list[dict]:
        out: list[dict] = []
        for rank, idx in enumerate(indices, start=1):
            out.append(
                build_pose_entry(
                    rank,
                    int(idx),
                    Rs[int(idx)],
                    t_base,
                    float(energies[int(idx)]),
                    float(weights[int(idx)]),
                    results[int(idx)],
                    category=category,
                )
            )
        return out

    most_probable = _entries(idx_most, "most_probable")
    least_probable = _entries(idx_least, "least_probable")
    map_entry = build_pose_entry(
        1,
        map_idx,
        Rs[map_idx],
        t_base,
        float(energies[map_idx]),
        float(weights[map_idx]),
        results[map_idx],
        category="map",
    )

    R_map = np.asarray(map_entry["pose"]["R"], dtype=np.float64)
    t_map = np.asarray(map_entry["pose"]["t"], dtype=np.float64)

    payload = {
        "pdb": str(args.pdb.resolve()),
        "mesh": (str(args.mesh.resolve()) if args.mesh is not None else None),
        "ising_params": ising.source_path,
        "ising_params_values": ising.as_dict(),
        "n_samples": n_samples,
        "sampling_strategy": strategy,
        "n_uniform": n_uniform,
        "n_mcmc_kept": int(np.sum(refined_mask)),
        "mcmc_summary": (
            {
                "mean_acceptance_rate": mcmc_summary.mean_acceptance_rate,
                "total_proposed": mcmc_summary.total_proposed,
                "total_accepted": mcmc_summary.total_accepted,
                "chains": [
                    {
                        "chain_id": c.chain_id,
                        "acceptance_rate": c.acceptance_rate,
                        "final_step_deg": c.final_step_deg,
                        "energy_start": c.energy_start,
                        "energy_end": c.energy_end,
                    }
                    for c in mcmc_summary.chains
                ],
            }
            if mcmc_summary is not None
            else None
        ),
        "seed": ising.sampling.seed,
        "beta": beta,
        "beta_auto": bool(ising.sampling.beta_auto),
        "beta_calibration": beta_calibration,
        "E_min": float(np.min(energies)),
        "E_mean": float(np.mean(energies)),
        "E_std": float(np.std(energies)),
        "most_probable_poses": most_probable,
        "least_probable_poses": least_probable,
        "top_poses": most_probable,
        "bottom_poses": least_probable,
        "orientation_sampling_diagnostics": sampling_summary,
        "couplings": coeffs.__dict__,
        "map_pose": map_entry,
        "output": {
            "n_orientation_renders": n_render,
            "render_snapshots": bool(ising.output.render_snapshots),
            "view_camera": "lab_plus_z_down",
        },
    }
    (args.outdir / "top_poses.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    render_paths: list[Path] = []
    do_render = bool(ising.output.render_snapshots) and not args.no_render
    if do_render:
        ref = render_system_snapshot(
            args.outdir, mesh, R_map, t_map, slab_thickness, filename="system_view_map.png"
        )
        if ref is not None:
            render_paths.append(ref)
        render_paths.extend(
            write_weighted_orientation_renders(
                args.outdir,
                mesh,
                most_probable,
                t_base,
                slab_thickness,
                name_prefix="most_probable",
            )
        )
        render_paths.extend(
            write_weighted_orientation_renders(
                args.outdir,
                mesh,
                least_probable,
                t_base,
                slab_thickness,
                name_prefix="least_probable",
            )
        legacy = args.outdir / "system_view.png"
        if ref is not None and not legacy.exists():
            shutil.copy2(ref, legacy)

    run.log("MAP orientation = highest Boltzmann weight")
    run.log(f"  weight_max = {float(weights[map_idx]):.6g}")
    run.log(f"  H at MAP = {float(energies[map_idx]):.0f}")
    run.log(f"  index = {map_idx}/{n_samples - 1}")
    euler = map_entry["pose"]["euler_zyx_deg"]
    run.log(f"  Euler Z-Y-X (deg) = [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]")
    if do_render:
        run.log(
            f"  rendered {len(most_probable)} most + {len(least_probable)} least probable "
            f"(+Z down) under views_z_down/"
        )
    run.log(f"  wrote: {str((args.outdir / 'top_poses.json').resolve())}")
    run.log(f"  orientation diagnostics: {str(report_path.resolve())}")
    for note in sampling_summary.get("interpretation_notes", []):
        run.log(f"    ! {note}")
    for path in [
        args.outdir / "diagnostics_energy_trace.png",
        args.outdir / "diagnostics_energy_hist.png",
        args.outdir / "diagnostics_tilt_energy.png",
        report_path,
        *orient_paths.values(),
    ]:
        if path is not None:
            run.log(f"  wrote: {str(path.resolve())}")
    if do_render:
        for rp in render_paths[:5]:
            run.log(f"  wrote: {rp.resolve()}")
        if len(render_paths) > 5:
            run.log(f"  ... and {len(render_paths) - 5} more PNGs under views_z_down/")
    run.log(f"OUTDIR: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
