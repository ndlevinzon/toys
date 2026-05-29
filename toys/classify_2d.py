#!/usr/bin/env python3
"""
2D classification of synthetic particle images (cryoSPARC-style toy).

Pipeline (similar in spirit to cryoSPARC 2D classification):
  1. Preprocess particles (display normalize, zero-mean unit-variance).
  2. Initialize K class references from random particles.
  3. Iterate: assign each particle to the best-matching class using
     in-plane rotation + translation search (normalized cross-correlation),
     then update class averages from aligned members.
  4. Save class averages, assignments, and alignment parameters.

Run view_2d_classes.py to inspect the result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

from beam_projection_toy import normalize_display


@dataclass
class Alignment:
    """In-plane alignment of a particle onto its assigned class reference."""

    class_id: int  # k in {0, ..., K-1}
    angle_deg: float
    shift_y: float
    shift_x: float
    score: float


@dataclass
class ClassificationResult:
    """Output of 2D classification."""

    class_averages: np.ndarray  # (K, H, W)
    assignments: np.ndarray  # (N,) int
    alignments: list[Alignment]
    particles: np.ndarray  # (N, H, W) preprocessed
    class_sizes: np.ndarray  # (K,)
    n_iterations: int
    source_dataset: str


def load_particle_stack(path: Path, key: str = "images") -> np.ndarray:
    with np.load(path) as data:
        if key not in data:
            raise KeyError(f"'{key}' not in {path.name}; keys: {data.files}")
        stack = np.asarray(data[key], dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError(f"Expected (N, H, W), got {stack.shape}")
    return stack


def preprocess_particle(image: np.ndarray) -> np.ndarray:
    """Per-particle normalization (cryoSPARC-style contrast scaling)."""
    out = normalize_display(image.astype(np.float64))
    out -= out.mean()
    std = out.std()
    if std > 1e-8:
        out /= std
    return out.astype(np.float32)


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a 2D image in the detector plane by ``angle_deg`` (counterclockwise)."""
    return ndimage.rotate(
        image,
        angle_deg,
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
    ).astype(np.float32)


def cross_correlate_shift(
    reference: np.ndarray,
    particle: np.ndarray,
    max_shift_px: int,
) -> tuple[float, tuple[float, float]]:
    """
    Normalized cross-correlation peak and sub-pixel shift (dy, dx).

    Positive shift_y / shift_x moves the particle to align with the reference.
    """
    ref = reference.astype(np.float64)
    part = particle.astype(np.float64)
    ref -= ref.mean()
    part -= part.mean()
    ref_std = ref.std()
    part_std = part.std()
    if ref_std < 1e-8 or part_std < 1e-8:
        return 0.0, (0.0, 0.0)

    ref_n = ref / ref_std
    part_n = part / part_std
    cc = signal.fftconvolve(ref_n, part_n[::-1, ::-1], mode="same")

    h, w = cc.shape
    cy, cx = h // 2, w // 2
    y0, y1 = max(0, cy - max_shift_px), min(h, cy + max_shift_px + 1)
    x0, x1 = max(0, cx - max_shift_px), min(w, cx + max_shift_px + 1)
    window = cc[y0:y1, x0:x1]
    peak_rel = np.unravel_index(int(np.argmax(window)), window.shape)
    peak_y = y0 + peak_rel[0]
    peak_x = x0 + peak_rel[1]
    shift_y = float(peak_y - cy)
    shift_x = float(peak_x - cx)
    return float(cc[peak_y, peak_x]), (shift_y, shift_x)


def apply_alignment(
    particle: np.ndarray,
    angle_deg: float,
    shift_yx: tuple[float, float],
) -> np.ndarray:
    aligned = rotate_image(particle, angle_deg)
    return ndimage.shift(
        aligned,
        shift=shift_yx,
        order=1,
        mode="constant",
        cval=0.0,
    ).astype(np.float32)


def match_particle_to_reference(
    reference: np.ndarray,
    particle: np.ndarray,
    angles_deg: np.ndarray,
    max_shift_px: int,
) -> tuple[float, float, tuple[float, float]]:
    """Best NCC score over in-plane rotations and small translations."""
    best_score = -np.inf
    best_angle = 0.0
    best_shift = (0.0, 0.0)

    for angle in angles_deg:
        rotated = rotate_image(particle, float(angle))
        score, shift = cross_correlate_shift(reference, rotated, max_shift_px)
        if score > best_score:
            best_score = score
            best_angle = float(angle)
            best_shift = shift

    return best_score, best_angle, best_shift


def initialize_references(
    particles: np.ndarray,
    n_classes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick K random particles as initial class-average references."""
    n = particles.shape[0]
    if n_classes > n:
        raise ValueError(f"Need at least {n_classes} particles, got {n}")
    indices = rng.choice(n, size=n_classes, replace=False)
    return particles[indices].copy()


def update_class_averages(
    particles: np.ndarray,
    assignments: np.ndarray,
    alignments: list[Alignment],
    n_classes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Update each class average as the mean of aligned member particles."""
    _, h, w = particles.shape
    averages = np.zeros((n_classes, h, w), dtype=np.float32)
    counts = np.zeros(n_classes, dtype=np.int32)

    for i, align in enumerate(alignments):
        k = align.class_id
        aligned = apply_alignment(particles[i], align.angle_deg, (align.shift_y, align.shift_x))
        averages[k] += aligned
        counts[k] += 1

    for k in range(n_classes):
        if counts[k] > 0:
            averages[k] /= counts[k]
            averages[k] = preprocess_particle(averages[k])
        else:
            averages[k] = particles[int(rng.integers(0, len(particles)))]

    return averages


def classify_2d(
    particles_raw: np.ndarray,
    n_classes: int,
    *,
    n_iterations: int = 8,
    n_angle_steps: int = 36,
    max_shift_px: int = 10,
    seed: int = 0,
    progress: bool = True,
) -> ClassificationResult:
    """
    Run iterative 2D classification with alignment.

    Parameters
    ----------
    particles_raw
        (N, H, W) detector images (any intensity scale).
    n_classes
        Number of 2D classes K.
    n_iterations
        EM-style refinement rounds.
    n_angle_steps
        In-plane rotations tested per particle–class pair (0–360°).
    max_shift_px
        Maximum translational search radius in pixels.
    """
    n = particles_raw.shape[0]
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    if n_classes > n:
        raise ValueError(f"Need at least {n_classes} particles, got {n}")

    rng = np.random.default_rng(seed)
    particles = np.stack([preprocess_particle(p) for p in particles_raw], axis=0)
    angles_deg = np.linspace(0.0, 360.0, num=n_angle_steps, endpoint=False)

    class_averages = initialize_references(particles, n_classes, rng)
    assignments = np.zeros(n, dtype=np.int32)
    alignments: list[Alignment] = []

    for it in range(n_iterations):
        alignments = []
        changed = 0

        for i in range(n):
            best_k = 0
            best_align = Alignment(0, 0.0, 0.0, 0.0, -np.inf)

            for k in range(n_classes):
                score, angle, (sy, sx) = match_particle_to_reference(
                    class_averages[k],
                    particles[i],
                    angles_deg,
                    max_shift_px,
                )
                if score > best_align.score:
                    best_align = Alignment(k, angle, sy, sx, score)

            if assignments[i] != best_align.class_id:
                changed += 1
            assignments[i] = best_align.class_id
            alignments.append(best_align)

        class_averages = update_class_averages(
            particles, assignments, alignments, n_classes, rng
        )

        if progress:
            sizes = np.bincount(assignments, minlength=n_classes)
            print(
                f"  iteration {it + 1}/{n_iterations}: "
                f"{changed} reassignments, class sizes {sizes.tolist()}"
            )

    class_sizes = np.bincount(assignments, minlength=n_classes)
    return ClassificationResult(
        class_averages=class_averages,
        assignments=assignments,
        alignments=alignments,
        particles=particles,
        class_sizes=class_sizes,
        n_iterations=n_iterations,
        source_dataset="",
    )


def save_classification(result: ClassificationResult, path: Path) -> None:
    """Persist class averages, assignments, and alignment parameters to .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    angles = np.array([a.angle_deg for a in result.alignments], dtype=np.float32)
    shifts = np.array([(a.shift_y, a.shift_x) for a in result.alignments], dtype=np.float32)
    scores = np.array([a.score for a in result.alignments], dtype=np.float32)

    np.savez_compressed(
        path,
        class_averages=result.class_averages,
        assignments=result.assignments,
        particles=result.particles,
        class_sizes=result.class_sizes,
        angles_deg=angles,
        shifts_yx=shifts,
        scores=scores,
        n_iterations=np.int32(result.n_iterations),
        source_dataset=np.array(result.source_dataset),
    )
    print(f"Wrote {path}  ({result.class_averages.shape[0]} classes)")


def load_classification(path: Path) -> dict[str, np.ndarray]:
    """Load all arrays from a classification_2d.npz file."""
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def parse_args() -> argparse.Namespace:
    """CLI for 2D classification."""
    parser = argparse.ArgumentParser(
        description="2D classification of a projection dataset (cryoSPARC-style toy).",
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=Path("projection_dataset.npz"),
        help="Input .npz from generate_projection_dataset.py",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("classification_2d.npz"),
        help="Output .npz (default: classification_2d.npz)",
    )
    parser.add_argument(
        "-k",
        "--classes",
        type=int,
        default=3,
        help="Number of 2D classes (default: 8)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=8,
        help="Refinement iterations (default: 8)",
    )
    parser.add_argument(
        "--angles",
        type=int,
        default=36,
        help="Rotation samples per match, 360/angles deg steps (default: 36)",
    )
    parser.add_argument(
        "--max-shift",
        type=int,
        default=10,
        help="Max translational search in pixels (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for initialization (default: 0)",
    )
    parser.add_argument(
        "--key",
        default="images",
        help="Dataset array key (default: images)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress iteration logs",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: classify particles and write classification_2d.npz."""
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"{dataset_path} not found")

    print(f"Loading {dataset_path} ...")
    stack = load_particle_stack(dataset_path, key=args.key)
    print(
        f"Classifying {stack.shape[0]} particles into {args.classes} classes "
        f"({args.iterations} iterations)..."
    )

    result = classify_2d(
        stack,
        args.classes,
        n_iterations=args.iterations,
        n_angle_steps=args.angles,
        max_shift_px=args.max_shift,
        seed=args.seed,
        progress=not args.quiet,
    )
    result.source_dataset = str(dataset_path.resolve())
    save_classification(result, args.output)
    print("Done. Inspect with:  python view_2d_classes.py", args.output)


if __name__ == "__main__":
    main()
