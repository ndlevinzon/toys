#!/usr/bin/env python3
"""
Generate a stack of synthetic single-particle images I_i for reconstruction experiments.

Uses the forward model from beam_projection_toy.py:

    I_i = h_i * T_{t_i} P_{R_i} z + η_i

Default phantom: square pyramid (apex +Z, base in XY), same voxel grid as the
interactive toy. Orientations are uniform on SO(3) by default, or vMF-biased
toward a preferred viewing axis (same κ mapping as ``orientation_distribution.py``).
The electron beam is fixed along −Z with detector up +Y.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from beam_projection_toy import (
    FIXED_BEAM_DIRECTION_XYZ,
    FIXED_DETECTOR_UP_XYZ,
    add_detector_noise,
    electron_wavelength_angstrom,
    load_volume,
    make_pyramid_volume,
    project_along_beam_vtk,
    psf_sigma_pixels,
    simulate_wave_detector_image,
    translate_image,
)
from orientation_sampling import (
    DEFAULT_KAPPA_MAX,
    DEFAULT_PREFERRED_AXIS,
    bias_to_kappa,
    sample_particle_rotation,
    summarize_orientation_bias,
)

DEFAULT_PHANTOM = "pyramid"
DEFAULT_PHANTOM_SIZE = 96
DEFAULT_OUTPUT_UNIFORM = Path("projection_dataset.npz")
DEFAULT_OUTPUT_BIASED = Path("projection_dataset_biased.npz")


def load_dataset_volume(
    volume_path: Path | None,
    *,
    phantom_size: int = DEFAULT_PHANTOM_SIZE,
) -> tuple[np.ndarray, str]:
    """
    Load or build the 3D reference map for batch projection.

    Returns ``(volume_zyx, phantom_id)``. With no path, builds the default pyramid.
    """
    if volume_path is None:
        return make_pyramid_volume(phantom_size), DEFAULT_PHANTOM
    return load_volume(volume_path), "custom"


def parse_preferred_axis(text: str) -> np.ndarray:
    """Parse ``--mu`` as three floats (default pyramid apex +Z)."""
    parts = [float(x) for x in text.replace(",", " ").split()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three components for --mu")
    axis = np.array(parts, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        raise argparse.ArgumentTypeError("--mu must be non-zero")
    return axis / norm


def random_translation_px(
    rng: np.random.Generator,
    half_range: float,
) -> tuple[float, float]:
    """Draw random in-plane shifts (t_x, t_y) uniform in [-half_range, half_range] pixels."""
    if half_range <= 0:
        return (0.0, 0.0)
    tx = float(rng.uniform(-half_range, half_range))
    ty = float(rng.uniform(-half_range, half_range))
    return (tx, ty)


def form_detector_image(
    volume: np.ndarray,
    rotation: Rotation,
    beam: np.ndarray,
    translation_px: tuple[float, float],
    *,
    use_wave_optics: bool,
    wavelength_angstrom: float,
    defocus_angstrom: float,
    pixel_size_angstrom: float,
    psf_sigma_px: float,
    phase_scale: float,
    absorption_strength: float,
    noise_sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (geometric_projection, detector_image I_i).

    Geometric projection uses :func:`project_along_beam_vtk` (same lab frame as the toy).
    Wave mode: Fresnel propagation with weak absorption + phase (diffraction fringes).
    PSF mode: geometric blur only (legacy, stored on a large intensity scale).
    """
    projection = project_along_beam_vtk(
        volume,
        rotation,
        beam,
        FIXED_DETECTOR_UP_XYZ,
    )
    translated = translate_image(projection, translation_px[1], translation_px[0])

    if use_wave_optics:
        detector = simulate_wave_detector_image(
            translated,
            wavelength_angstrom,
            defocus_angstrom,
            pixel_size_angstrom=pixel_size_angstrom,
            phase_scale=phase_scale,
            absorption_strength=absorption_strength,
        )
    else:
        from beam_projection_toy import convolve_psf

        detector, _ = convolve_psf(translated, psf_sigma_px)

    observed = add_detector_noise(detector, noise_sigma, rng)
    return projection, observed


def generate_projection_dataset(
    n_views: int,
    volume: np.ndarray | None = None,
    *,
    volume_path: Path | None = None,
    phantom_size: int = DEFAULT_PHANTOM_SIZE,
    seed: int = 0,
    beam_direction_xyz: np.ndarray = FIXED_BEAM_DIRECTION_XYZ,
    pixel_size_angstrom: float = 1.0,
    voltage_kv: float = 300.0,
    defocus_angstrom: float = 2.0e4,
    noise_sigma: float = 0.04,
    translation_half_range_px: float = 8.0,
    use_wave_optics: bool = True,
    phase_scale: float = np.pi,
    absorption_strength: float = 0.28,
    orientation_bias: float = 0.0,
    preferred_axis_xyz: np.ndarray | None = None,
    kappa_max: float = DEFAULT_KAPPA_MAX,
    progress: bool = True,
) -> dict[str, np.ndarray]:
    """
    Simulate ``n_views`` detector images with random orientations.

    ``orientation_bias = 0``: i.i.d. uniform on SO(3).
    ``orientation_bias > 0``: vMF cluster around ``preferred_axis_xyz`` (apex +Z).

    Returns a dict of arrays (also suitable for ``np.savez_compressed``).
    """
    if n_views < 1:
        raise ValueError("n_views must be at least 1")

    if volume is None:
        volume, phantom_id = load_dataset_volume(volume_path, phantom_size=phantom_size)
    else:
        phantom_id = "custom"

    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")

    rng = np.random.default_rng(seed)
    mu = (
        np.asarray(preferred_axis_xyz, dtype=np.float64).reshape(3)
        if preferred_axis_xyz is not None
        else DEFAULT_PREFERRED_AXIS.copy()
    )
    mu /= np.linalg.norm(mu) + 1e-12
    orientation_bias = float(np.clip(orientation_bias, 0.0, 1.0))
    kappa = bias_to_kappa(orientation_bias, kappa_max)
    sampling_mode = "uniform" if kappa < 1e-8 else "vmf"

    beam = np.asarray(beam_direction_xyz, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(beam)
    if norm < 1e-8:
        raise ValueError("beam_direction_xyz must be non-zero")
    beam /= norm

    wavelength_angstrom = electron_wavelength_angstrom(voltage_kv)
    detector_shape = (int(volume.shape[1]), int(volume.shape[2]))
    psf_sigma = psf_sigma_pixels(
        wavelength_angstrom,
        defocus_angstrom,
        pixel_size_angstrom,
        detector_shape,
    )

    images: list[np.ndarray] = []
    projections: list[np.ndarray] = []
    rotation_matrices = np.empty((n_views, 3, 3), dtype=np.float64)
    translations = np.empty((n_views, 2), dtype=np.float32)
    rotation_list: list[Rotation] = []

    for i in range(n_views):
        rotation = sample_particle_rotation(
            rng,
            preferred_axis_xyz=mu,
            orientation_bias=orientation_bias,
            kappa_max=kappa_max,
        )
        rotation_list.append(rotation)
        translation_px = random_translation_px(rng, translation_half_range_px)

        projection, observed = form_detector_image(
            volume,
            rotation,
            beam,
            translation_px,
            use_wave_optics=use_wave_optics,
            wavelength_angstrom=wavelength_angstrom,
            defocus_angstrom=defocus_angstrom,
            pixel_size_angstrom=pixel_size_angstrom,
            psf_sigma_px=psf_sigma,
            phase_scale=phase_scale,
            absorption_strength=absorption_strength,
            noise_sigma=noise_sigma,
            rng=rng,
        )

        images.append(observed)
        projections.append(projection)
        rotation_matrices[i] = rotation.as_matrix()
        translations[i] = translation_px

        if progress and (i == 0 or (i + 1) % max(1, n_views // 10) == 0 or i + 1 == n_views):
            print(f"  view {i + 1}/{n_views}")

    if progress and sampling_mode == "vmf":
        stats = summarize_orientation_bias(rotation_list, mu)
        print(
            f"  orientation bias: κ={kappa:.1f}, mean(μ·view)={stats['mean_dot_mu']:.3f} "
            f"(1.0 = all on preferred axis)"
        )

    stack = np.stack(images, axis=0)
    proj_stack = np.stack(projections, axis=0)

    return {
        "images": stack.astype(np.float32),
        "geometric_projections": proj_stack.astype(np.float32),
        "rotations": rotation_matrices,
        "translations_px": translations,
        "beam_direction_xyz": beam.astype(np.float32),
        "detector_up_xyz": FIXED_DETECTOR_UP_XYZ.astype(np.float32),
        "phantom": np.array(phantom_id),
        "psf_sigma_px": np.float32(psf_sigma),
        "noise_sigma": np.float32(noise_sigma),
        "wavelength_angstrom": np.float32(wavelength_angstrom),
        "defocus_angstrom": np.float32(defocus_angstrom),
        "pixel_size_angstrom": np.float32(pixel_size_angstrom),
        "voltage_kv": np.float32(voltage_kv),
        "volume_shape_zyx": np.array(volume.shape, dtype=np.int32),
        "seed": np.int64(seed),
        "n_views": np.int64(n_views),
        "use_wave_optics": np.bool_(use_wave_optics),
        "phase_scale": np.float32(phase_scale),
        "absorption_strength": np.float32(absorption_strength),
        "orientation_bias": np.float32(orientation_bias),
        "orientation_kappa": np.float32(kappa),
        "preferred_axis_xyz": mu.astype(np.float32),
        "orientation_sampling": np.array(sampling_mode),
    }


def save_dataset(dataset: dict[str, np.ndarray], path: Path) -> None:
    """Write a compressed .npz with images, projections, rotations, and metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)
    phantom = str(dataset.get("phantom", "?"))
    sampling = str(dataset.get("orientation_sampling", "?"))
    bias = float(dataset.get("orientation_bias", 0.0))
    print(
        f"Wrote {path}  ({dataset['n_views']} views, "
        f"phantom={phantom}, orientations={sampling}, bias={bias:.2f}, "
        f"shape {dataset['images'].shape})"
    )


def parse_args() -> argparse.Namespace:
    """CLI for synthetic projection dataset generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate N synthetic 2D pyramid projections with random orientations "
            "(fixed beam −Z, same forward model as beam_projection_toy.py)."
        ),
    )
    parser.add_argument(
        "-n",
        "--num-views",
        type=int,
        default=1000,
        help="Number of projections to simulate (default: 1000).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output .npz (default: projection_dataset.npz, or "
            "projection_dataset_biased.npz if --orientation-bias > 0)."
        ),
    )
    parser.add_argument(
        "--orientation-bias",
        type=float,
        default=0.0,
        metavar="B",
        help=(
            "Orientation bias in [0, 1], same as orientation_distribution slider "
            "(0 = uniform SO(3), 1 = strong vMF toward --mu)."
        ),
    )
    parser.add_argument(
        "--mu",
        type=parse_preferred_axis,
        default=None,
        help="Preferred viewing axis for biased sampling (default: 0 0 1, pyramid apex).",
    )
    parser.add_argument(
        "--kappa-max",
        type=float,
        default=DEFAULT_KAPPA_MAX,
        help=f"vMF κ at orientation-bias=1 (default: {DEFAULT_KAPPA_MAX}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for orientations, shifts, and noise (default: 0).",
    )
    parser.add_argument(
        "volume",
        nargs="?",
        type=Path,
        default=None,
        help="Optional 3D map (.mrc, .npy, .npz). Default: built-in square pyramid.",
    )
    parser.add_argument(
        "--phantom-size",
        type=int,
        default=DEFAULT_PHANTOM_SIZE,
        metavar="N",
        help=f"Voxel grid edge length for the default pyramid (default: {DEFAULT_PHANTOM_SIZE}).",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=1.0,
        help="Voxel size in Å (default: 1.0).",
    )
    parser.add_argument(
        "--voltage-kv",
        type=float,
        default=300.0,
        help="Beam energy in kV (default: 300).",
    )
    parser.add_argument(
        "--defocus",
        type=float,
        default=2.0e4,
        help="Defocus in Å (default: 20000).",
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=0.04,
        help="Additive detector noise std (default: 0.04).",
    )
    parser.add_argument(
        "--translation-range",
        type=float,
        default=8.0,
        help="Max |t_x|, |t_y| in pixels; 0 for centered particles (default: 8).",
    )
    parser.add_argument(
        "--psf-only",
        action="store_true",
        help="Use Gaussian PSF blur instead of wave propagation (legacy; large raw values).",
    )
    parser.add_argument(
        "--phase-scale",
        type=float,
        default=float(np.pi),
        help="Phase strength in wave model (default: pi).",
    )
    parser.add_argument(
        "--absorption",
        type=float,
        default=0.28,
        help="Weak absorption in wave model; lower = more transparent / phase-heavy (default: 0.28).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-view progress messages.",
    )
    return parser.parse_args()


def resolve_output_path(args: argparse.Namespace) -> Path:
    """Pick default .npz name from uniform vs biased orientations."""
    if args.output is not None:
        return Path(args.output)
    if args.orientation_bias > 0.02:
        return DEFAULT_OUTPUT_BIASED
    return DEFAULT_OUTPUT_UNIFORM


def main() -> None:
    """Entry point: simulate N pyramid views and save projection_dataset.npz."""
    args = parse_args()
    output = resolve_output_path(args)
    mu = args.mu if args.mu is not None else DEFAULT_PREFERRED_AXIS.copy()
    phantom_desc = (
        f"custom volume {args.volume}"
        if args.volume is not None
        else f"pyramid {args.phantom_size}^3"
    )
    orient_desc = (
        "uniform SO(3)"
        if args.orientation_bias < 0.02
        else f"biased (bias={args.orientation_bias:.2f}, mu={mu})"
    )
    print(
        f"Generating {args.num_views} projections "
        f"({phantom_desc}, {orient_desc}, seed={args.seed}, beam along -Z)..."
    )
    dataset = generate_projection_dataset(
        args.num_views,
        volume_path=args.volume,
        phantom_size=args.phantom_size,
        seed=args.seed,
        pixel_size_angstrom=args.pixel_size,
        voltage_kv=args.voltage_kv,
        defocus_angstrom=args.defocus,
        noise_sigma=args.noise_sigma,
        translation_half_range_px=args.translation_range,
        use_wave_optics=not args.psf_only,
        phase_scale=args.phase_scale,
        absorption_strength=args.absorption,
        orientation_bias=args.orientation_bias,
        preferred_axis_xyz=mu,
        kappa_max=args.kappa_max,
        progress=not args.quiet,
    )
    save_dataset(dataset, output)
    print("Done. Example downstream commands:")
    print(f"  python classify_2d.py {output} -o classification_2d.npz")
    print(
        f"  python ab_initio_reconstruct.py --dataset {output} "
        "--quality -o reconstruction_ab_initio.npz"
    )


if __name__ == "__main__":
    main()
