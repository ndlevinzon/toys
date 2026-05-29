#!/usr/bin/env python3
"""
Ab initio 3D reconstruction from 2D particle images (cryoSPARC-style toy).

Alternates orientation assignment (3D rotation + in-plane angle + shift) with
weighted, ramp-filtered back-projection. See module docstring notes on avoiding
the common "donut" failure mode.

Important
---------
* **Do not** apply 2D *class* in-plane alignments before ab initio — they rotate
  each particle to a different reference frame and destroy shared 3D geometry.
  Use ``--center-only`` (default) to recenter particles only.
* Match particles to **geometric** line integrals when possible
  (``--match-geometric``, uses ``geometric_projections`` from the dataset).
  Wave-optics ``images`` have CTF fringes that a geometric forward model cannot
  explain, which drives wrong orientations and ring-like artifacts.
* More particles help, but orientation **search density** and **iterations**
  matter as much as count. Use ``--quality`` for large stacks.

Output: ``reconstruction_ab_initio.npz`` — view with ``view_reconstruction.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

from beam_projection_toy import (
    normalize_display,
    project_along_beam_vtk,
    rotate_volume,
    xyz_to_zyx_vector,
)
from classify_2d import cross_correlate_shift, load_particle_stack, preprocess_particle, rotate_image

DEFAULT_BEAM = np.array([0.0, 0.0, -1.0], dtype=np.float64)


@dataclass
class ParticleMetadata:
    """Per-particle orientation assignment and alignment from one refinement round."""

    index: int
    rotation: Rotation
    inplane_deg: float
    shift_yx: tuple[float, float]
    score: float


@dataclass
class ReconstructionResult:
    """Final 3D map and per-particle orientation metadata from ab initio."""

    volume: np.ndarray
    orientations: Rotation
    scores: np.ndarray
    n_iterations: int
    source_particles: str
    metadata: list[ParticleMetadata]


def beam_align_rotation(beam_direction_xyz: np.ndarray) -> Rotation:
    """Rotation that maps the lab beam direction to volume axis 0 (Z, Y, X)."""
    direction = np.asarray(beam_direction_xyz, dtype=float).reshape(3)
    direction /= np.linalg.norm(direction) + 1e-12
    d_zyx = xyz_to_zyx_vector(direction)
    d_zyx /= np.linalg.norm(d_zyx) + 1e-12
    rot, _ = Rotation.align_vectors([d_zyx], [np.array([1.0, 0.0, 0.0])])
    return rot


def downsample_volume(volume: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a 3D volume by integer factor for fast orientation matching."""
    if factor <= 1:
        return volume
    zoom = 1.0 / factor
    return ndimage.zoom(volume, zoom=(zoom, zoom, zoom), order=1).astype(np.float32)


def downsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a 2D particle image by integer factor."""
    if factor <= 1:
        return image
    zoom = 1.0 / factor
    return ndimage.zoom(image, zoom=(zoom, zoom), order=1).astype(np.float32)


def backproject_image(
    image: np.ndarray,
    particle_rotation: Rotation,
    beam_direction_xyz: np.ndarray,
    out_shape: tuple[int, int, int],
) -> np.ndarray:
    """Smear a 2D image along the beam, then rotate into the lab frame."""
    nz, ny, nx = out_shape
    align_beam = beam_align_rotation(beam_direction_xyz)

    slab = np.zeros((nz, ny, nx), dtype=np.float32)
    slab[:] = image.astype(np.float32)[np.newaxis, :, :]

    slab = rotate_volume(slab, align_beam.inv())
    slab = rotate_volume(slab, particle_rotation.inv())
    return slab


def ramp_filter_2d(image: np.ndarray) -> np.ndarray:
    """Simple ramp pre-filter (reduces low-frequency buildup / donut artifacts)."""
    img = image.astype(np.float64)
    ny, nx = img.shape
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing="ij")
    ramp = np.sqrt(fx_grid**2 + fy_grid**2)
    ramp[0, 0] = 0.0
    filtered = np.fft.ifft2(np.fft.fft2(img) * ramp)
    return np.real(filtered).astype(np.float32)


def normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def prepare_volume_for_projection(volume: np.ndarray) -> np.ndarray:
    vol = volume.astype(np.float64)
    vol -= vol.min()
    peak = vol.max()
    if peak > 0:
        vol /= peak
    return vol.astype(np.float32)


def spherical_support_mask(shape: tuple[int, int, int], radius_frac: float = 0.42) -> np.ndarray:
    nz, ny, nx = shape
    cz, cy, cx = (nz - 1) / 2.0, (ny - 1) / 2.0, (nx - 1) / 2.0
    zz, yy, xx = np.mgrid[0:nz, 0:ny, 0:nx]
    r2 = ((zz - cz) / nz) ** 2 + ((yy - cy) / ny) ** 2 + ((xx - cx) / nx) ** 2
    return (r2 <= radius_frac**2).astype(np.float32)


def center_particle(image: np.ndarray) -> np.ndarray:
    """Recenter by intensity centroid (preserves 3D viewing geometry)."""
    img = image.astype(np.float64)
    weights = np.clip(img - img.min(), 0.0, None)
    total = weights.sum()
    if total < 1e-8:
        return image.astype(np.float32)
    cy, cx = ndimage.center_of_mass(weights)
    h, w = img.shape
    shift_y = (h - 1) / 2.0 - cy
    shift_x = (w - 1) / 2.0 - cx
    return ndimage.shift(
        img,
        shift=(shift_y, shift_x),
        order=1,
        mode="constant",
        cval=0.0,
    ).astype(np.float32)


def center_particle_stack(stack: np.ndarray) -> np.ndarray:
    return np.stack([center_particle(p) for p in stack], axis=0)


def fibonacci_orientations(n: int) -> list[Rotation]:
    """Roughly uniform directions on SO(3) via Fibonacci sphere + in-plane spin."""
    if n < 1:
        return [Rotation.identity()]
    golden = np.pi * (3.0 - np.sqrt(5.0))
    rotations: list[Rotation] = []
    b_hat = DEFAULT_BEAM / np.linalg.norm(DEFAULT_BEAM)
    spins = np.linspace(0.0, 360.0, num=max(6, int(np.sqrt(n))), endpoint=False)
    count = 0
    for i in range(n * 2):
        if count >= n:
            break
        t = (i + 0.5) / max(n * 2, 1)
        z = 1.0 - 2.0 * t
        r = np.sqrt(max(0.0, 1.0 - z * z))
        phi = golden * i
        viewing = np.array([r * np.cos(phi), r * np.sin(phi), z])
        viewing /= np.linalg.norm(viewing) + 1e-12
        align, _ = Rotation.align_vectors([viewing], [b_hat])
        for psi in spins:
            if count >= n:
                break
            spin = Rotation.from_rotvec(np.deg2rad(psi) * b_hat)
            rotations.append(spin * align)
            count += 1
    return rotations[:n]


def compose_orientation(
    base: Rotation,
    inplane_deg: float,
    beam_axis: np.ndarray,
) -> Rotation:
    """Apply in-plane spin (degrees) about the lab beam axis to a 3D rotation."""
    spin = Rotation.from_rotvec(np.deg2rad(inplane_deg) * beam_axis)
    return spin * base


def make_initial_volume(
    shape: tuple[int, int, int],
    rng: np.random.Generator,
    particles: np.ndarray | None = None,
) -> np.ndarray:
    nz, ny, nx = shape
    cz, cy, cx = (nz - 1) / 2.0, (ny - 1) / 2.0, (nx - 1) / 2.0
    zz, yy, xx = np.mgrid[0:nz, 0:ny, 0:nx]
    r2 = ((zz - cz) / (0.20 * nz)) ** 2 + ((yy - cy) / (0.22 * ny)) ** 2 + ((xx - cx) / (0.22 * nx)) ** 2
    blob = np.exp(-r2).astype(np.float32)

    if particles is not None and len(particles) > 0:
        mean2d = normalize_display(np.mean(particles, axis=0).astype(np.float32))
        z0, z1 = max(0, nz // 2 - 3), min(nz, nz // 2 + 4)
        blob[z0:z1] += 0.25 * mean2d

    blob *= spherical_support_mask(shape, radius_frac=0.44)
    blob += 0.01 * rng.standard_normal(blob.shape).astype(np.float32)
    blob = np.clip(blob, 0.0, None)
    return prepare_volume_for_projection(blob)


def perturb_orientation(
    base: Rotation,
    rng: np.random.Generator,
    sigma_deg: float,
) -> Rotation:
    if sigma_deg <= 0:
        return base
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis) + 1e-12
    delta = np.deg2rad(rng.normal(0.0, sigma_deg))
    return Rotation.from_rotvec(axis * delta) * base


def build_candidate_orientations(
    n_global: int,
    previous: Rotation | None,
    rng: np.random.Generator,
    n_local: int,
    local_sigma_deg: float,
) -> list[Rotation]:
    """Global Fibonacci samples plus local perturbations around the previous pose."""
    candidates = fibonacci_orientations(n_global)
    candidates.extend(sample_random_orientations(max(8, n_global // 4), rng))
    if previous is not None:
        for _ in range(n_local):
            candidates.append(perturb_orientation(previous, rng, local_sigma_deg))
        candidates.append(previous)
    return candidates


def sample_random_orientations(n: int, rng: np.random.Generator) -> list[Rotation]:
    """Extra uniform SO(3) samples to supplement the Fibonacci grid."""
    return list(Rotation.random(n, random_state=rng))


def orientation_search(
    particle: np.ndarray,
    volume: np.ndarray,
    beam_direction_xyz: np.ndarray,
    candidates: list[Rotation],
    *,
    inplane_steps: int = 72,
    max_shift_px: int = 6,
    previous_inplane: float = 0.0,
) -> ParticleMetadata:
    """3D candidates + in-plane rotation + sub-pixel shift (like 2D class alignment)."""
    best_score = -np.inf
    best_rot = candidates[0]
    best_inplane = previous_inplane
    best_shift = (0.0, 0.0)

    vol = prepare_volume_for_projection(volume)
    beam_axis = beam_direction_xyz / (np.linalg.norm(beam_direction_xyz) + 1e-12)

    if inplane_steps > 1:
        half = max(18, inplane_steps // 2)
        inplane_angles = np.linspace(
            previous_inplane - 45.0,
            previous_inplane + 45.0,
            num=half,
            endpoint=False,
        )
    else:
        inplane_angles = np.array([previous_inplane])

    for base_rot in candidates:
        for psi in inplane_angles:
            rot = compose_orientation(base_rot, float(psi), beam_axis)
            proj = project_along_beam_vtk(vol, rot, beam_direction_xyz)
            score, shift = cross_correlate_shift(proj, particle, max_shift_px)
            if score > best_score:
                best_score = score
                best_rot = rot
                best_inplane = float(psi)
                best_shift = shift

    return ParticleMetadata(
        index=-1,
        rotation=best_rot,
        inplane_deg=best_inplane,
        shift_yx=best_shift,
        score=best_score,
    )


def align_particle_for_backproject(
    particle: np.ndarray,
    meta: ParticleMetadata,
) -> np.ndarray:
    """Apply the matched in-plane shift/rotation before back-projection."""
    img = rotate_image(particle, -meta.inplane_deg)
    return ndimage.shift(
        img,
        shift=meta.shift_yx,
        order=1,
        mode="constant",
        cval=0.0,
    ).astype(np.float32)


def postprocess_volume(
    volume: np.ndarray,
    mask: np.ndarray,
    *,
    sigma_vox: float = 0.6,
) -> np.ndarray:
    vol = np.clip(volume.astype(np.float32), 0.0, None) * mask
    if sigma_vox > 0:
        vol = ndimage.gaussian_filter(vol, sigma=sigma_vox)
    vol *= mask
    return prepare_volume_for_projection(vol)


def load_particles_for_reconstruction(
    classification_path: Path | None,
    dataset_path: Path | None,
    *,
    images_key: str = "auto",
    apply_class_alignments: bool = False,
    center_particles: bool = True,
) -> tuple[np.ndarray, str]:
    """
    Load particles for ab initio.

    ``images_key='auto'`` prefers ``geometric_projections`` when present (matches
    the geometric forward model). Class-based in-plane alignments are **off** by
    default because they break the shared 3D frame.
    """
    if dataset_path is not None and dataset_path.is_file():
        with np.load(dataset_path) as data:
            key = images_key
            if key == "auto":
                key = "geometric_projections" if "geometric_projections" in data.files else "images"
            if key not in data.files:
                raise KeyError(f"'{key}' not in {dataset_path.name}; keys: {data.files}")
            raw = np.asarray(data[key], dtype=np.float32)
            source = f"{dataset_path.resolve()} [{key}]"
        particles = np.stack([preprocess_particle(p) for p in raw], axis=0)
        if center_particles:
            particles = center_particle_stack(particles)
        return particles, source

    if classification_path is not None and classification_path.is_file():
        with np.load(classification_path) as data:
            particles = np.asarray(data["particles"], dtype=np.float32)
            source = str(classification_path.resolve())
            if apply_class_alignments and "angles_deg" in data and "shifts_yx" in data:
                from classify_2d import apply_alignment

                angles = np.asarray(data["angles_deg"])
                shifts = np.asarray(data["shifts_yx"])
                aligned = [
                    apply_alignment(
                        particles[i],
                        float(angles[i]),
                        (float(shifts[i, 0]), float(shifts[i, 1])),
                    )
                    for i in range(len(particles))
                ]
                particles = np.stack(aligned, axis=0)
        if center_particles:
            particles = center_particle_stack(particles)
        return particles, source

    raise FileNotFoundError("Provide projection_dataset.npz and/or classification_2d.npz")


def load_ground_truth_orientations(dataset_path: Path) -> Rotation | None:
    """Load ``rotations`` from a projection dataset for diagnostic reconstruction."""
    with np.load(dataset_path) as data:
        if "rotations" not in data.files:
            return None
        matrices = np.asarray(data["rotations"], dtype=np.float64)
    return Rotation.from_matrix(matrices)


def ab_initio_reconstruct(
    particles: np.ndarray,
    *,
    volume_shape: tuple[int, int, int] | None = None,
    beam_direction_xyz: np.ndarray = DEFAULT_BEAM,
    n_iterations: int = 15,
    n_global_orientations: int = 60,
    n_local_orientations: int = 24,
    local_sigma_deg: float = 10.0,
    inplane_steps: int = 72,
    max_shift_px: int = 6,
    match_downsample: int = 2,
    sirt_steps: int = 2,
    use_ramp_filter: bool = True,
    seed: int = 0,
    gt_orientations: Rotation | None = None,
    progress: bool = True,
) -> ReconstructionResult:
    n_particles, h, w = particles.shape
    if volume_shape is None:
        volume_shape = (h, w, w) if h == w else (h, h, w)

    rng = np.random.default_rng(seed)
    mask = spherical_support_mask(volume_shape, radius_frac=0.44)
    volume = make_initial_volume(volume_shape, rng, particles) * mask

    particles_match = np.stack(
        [downsample_image(p, match_downsample) for p in particles],
        axis=0,
    )
    metadata: list[ParticleMetadata] = [
        ParticleMetadata(i, Rotation.identity(), 0.0, (0.0, 0.0), 0.0) for i in range(n_particles)
    ]

    if gt_orientations is not None:
        if len(gt_orientations) != n_particles:
            raise ValueError("Ground-truth orientation count does not match particle count.")
        metadata = [
            ParticleMetadata(i, gt_orientations[i], 0.0, (0.0, 0.0), 1.0)
            for i in range(n_particles)
        ]

    for it in range(n_iterations):
        sigma = local_sigma_deg * (0.7 ** it)
        volume_match = downsample_volume(volume, match_downsample)

        if gt_orientations is None:
            for i in range(n_particles):
                prev = metadata[i]
                candidates = build_candidate_orientations(
                    n_global_orientations,
                    prev.rotation if it > 0 else None,
                    rng,
                    n_local_orientations,
                    sigma,
                )
                metadata[i] = orientation_search(
                    particles_match[i],
                    volume_match,
                    beam_direction_xyz,
                    candidates,
                    inplane_steps=inplane_steps,
                    max_shift_px=max_shift_px,
                    previous_inplane=prev.inplane_deg,
                )
                metadata[i].index = i

        acc = np.zeros(volume_shape, dtype=np.float64)
        weight_sum = 0.0
        for meta in metadata:
            aligned = align_particle_for_backproject(particles[meta.index], meta)
            if use_ramp_filter:
                aligned = ramp_filter_2d(aligned)
            w = max(meta.score, 1e-6) ** 2
            acc += w * backproject_image(
                aligned,
                meta.rotation,
                beam_direction_xyz,
                volume_shape,
            )
            weight_sum += w

        volume = (acc / max(weight_sum, 1e-8)).astype(np.float32)
        volume = postprocess_volume(volume, mask)

        for _ in range(sirt_steps):
            residual_acc = np.zeros(volume_shape, dtype=np.float64)
            vol_prep = prepare_volume_for_projection(volume)
            for meta in metadata:
                proj = project_along_beam_vtk(vol_prep, meta.rotation, beam_direction_xyz)
                aligned = align_particle_for_backproject(particles[meta.index], meta)
                residual = aligned - proj
                w = max(meta.score, 1e-6)
                if use_ramp_filter:
                    residual = ramp_filter_2d(residual)
                residual_acc += w * backproject_image(
                    residual,
                    meta.rotation,
                    beam_direction_xyz,
                    volume_shape,
                )
            volume = postprocess_volume(
                volume + 0.35 * residual_acc / max(n_particles, 1),
                mask,
                sigma_vox=0.4,
            )

        scores = np.array([m.score for m in metadata], dtype=np.float32)
        if progress:
            print(
                f"  iteration {it + 1}/{n_iterations}: "
                f"mean NCC={scores.mean():.3f}, max={scores.max():.3f}"
            )

    orientations = Rotation.concatenate([m.rotation for m in metadata])
    return ReconstructionResult(
        volume=volume,
        orientations=orientations,
        scores=scores,
        n_iterations=n_iterations,
        source_particles="",
        metadata=metadata,
    )


def save_reconstruction(result: ReconstructionResult, path: Path) -> None:
    """Write volume, orientation matrices, and scores to .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        volume=result.volume.astype(np.float32),
        orientation_matrices=result.orientations.as_matrix().astype(np.float32),
        scores=result.scores.astype(np.float32),
        n_iterations=np.int32(result.n_iterations),
        source_particles=np.array(result.source_particles),
        beam_direction_xyz=DEFAULT_BEAM.astype(np.float32),
    )
    print(f"Wrote {path}  (volume shape {result.volume.shape})")


def parse_args() -> argparse.Namespace:
    """CLI for ab initio reconstruction."""
    parser = argparse.ArgumentParser(
        description="Ab initio 3D reconstruction (toy).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Tips for a pyramid-shaped map instead of a donut:\n"
            "  1. Rebuild with geometric_projections: generate_projection_dataset.py\n"
            "  2. Run: ab_initio_reconstruct.py --dataset projection_dataset.npz --quality\n"
            "  3. Do NOT use 2D class alignments (default is --center-only)\n"
            "  4. Diagnostic upper bound: --use-gt-orientations"
        ),
    )
    parser.add_argument("--classification", type=Path, default=Path("classification_2d.npz"))
    parser.add_argument("--dataset", type=Path, default=Path("projection_dataset.npz"))
    parser.add_argument("-o", "--output", type=Path, default=Path("reconstruction_ab_initio.npz"))
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--global-orientations", type=int, default=60)
    parser.add_argument("--local-orientations", type=int, default=24)
    parser.add_argument("--inplane-steps", type=int, default=72)
    parser.add_argument("--max-shift", type=int, default=6)
    parser.add_argument("--match-downsample", type=int, default=2)
    parser.add_argument("--sirt-steps", type=int, default=2)
    parser.add_argument("--local-sigma-deg", type=float, default=10.0)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--images-key", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--apply-class-align",
        action="store_true",
        help="Apply 2D class in-plane alignments (usually harmful for ab initio)",
    )
    parser.add_argument(
        "--no-center",
        action="store_true",
        help="Skip centroid recentering of particles",
    )
    parser.add_argument(
        "--no-ramp",
        action="store_true",
        help="Disable ramp filtering before back-projection",
    )
    parser.add_argument(
        "--use-gt-orientations",
        action="store_true",
        help="Diagnostic: use known rotations from projection_dataset.npz",
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Higher search density: more iterations, full-res matching",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick low-resolution run (not for final maps)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Entry point: load particles, reconstruct, and save reconstruction_ab_initio.npz."""
    args = parse_args()
    class_path = args.classification if args.classification.is_file() else None
    dataset_path = args.dataset if args.dataset.is_file() else None

    print("Loading particles ...")
    particles, source = load_particles_for_reconstruction(
        class_path,
        dataset_path,
        images_key=args.images_key,
        apply_class_alignments=args.apply_class_align,
        center_particles=not args.no_center,
    )

    shape = (args.size, args.size, args.size) if args.size else None
    if shape is None:
        shape = (particles.shape[1], particles.shape[2], particles.shape[2])

    iterations = args.iterations
    n_global = args.global_orientations
    n_local = args.local_orientations
    inplane = args.inplane_steps
    match_ds = args.match_downsample

    if args.quality:
        iterations = max(iterations, 20)
        n_global = max(n_global, 90)
        n_local = max(n_local, 36)
        inplane = max(inplane, 90)
        match_ds = 1

    if args.fast:
        iterations = 6
        n_global = 24
        n_local = 10
        inplane = 36
        match_ds = max(match_ds, 2)
        if args.size is None and max(particles.shape[1:]) > 72:
            particles = np.stack([downsample_image(p, 2) for p in particles], axis=0)
            shape = (particles.shape[1], particles.shape[2], particles.shape[2])
            print(f"  --fast: {shape[0]}^3 volume")

    gt = None
    if args.use_gt_orientations:
        if dataset_path is None:
            raise FileNotFoundError("--use-gt-orientations needs projection_dataset.npz")
        gt = load_ground_truth_orientations(dataset_path)
        if gt is None:
            raise KeyError("No 'rotations' array in dataset")
        print("  Using ground-truth orientations (diagnostic only)")

    print(
        f"Ab initio: {len(particles)} particles, volume {shape}, "
        f"{iterations} iterations, match downsample x{match_ds}"
    )
    print(f"  source: {source}")

    result = ab_initio_reconstruct(
        particles,
        volume_shape=shape,
        n_iterations=iterations,
        n_global_orientations=n_global,
        n_local_orientations=n_local,
        local_sigma_deg=args.local_sigma_deg,
        inplane_steps=inplane,
        max_shift_px=args.max_shift,
        match_downsample=match_ds,
        sirt_steps=args.sirt_steps,
        use_ramp_filter=not args.no_ramp,
        seed=args.seed,
        gt_orientations=gt,
        progress=not args.quiet,
    )
    result.source_particles = source
    save_reconstruction(result, args.output)
    print("Done. View with:  python view_reconstruction.py", args.output)
    print("       Movie:  python render_reconstruction_rotation.py", args.output)


if __name__ == "__main__":
    main()
