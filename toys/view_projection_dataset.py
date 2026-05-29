#!/usr/bin/env python3
"""Display a subset of images from a projection_dataset.npz file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beam_projection_toy import normalize_display


def load_images(path: Path, key: str = "images") -> np.ndarray:
    """Load a particle stack (N, H, W) from a projection dataset .npz."""
    with np.load(path) as data:
        if key not in data:
            keys = ", ".join(sorted(data.files))
            raise KeyError(f"'{key}' not in {path.name}; available: {keys}")
        stack = np.asarray(data[key])
    if stack.ndim != 3:
        raise ValueError(f"Expected (N, H, W), got shape {stack.shape}")
    return stack


def pick_indices(n_total: int, indices: list[int] | None, count: int, seed: int) -> list[int]:
    """Return explicit indices or a reproducible random subset of size ``count``."""
    if indices is not None:
        for i in indices:
            if i < 0 or i >= n_total:
                raise IndexError(f"Index {i} out of range [0, {n_total})")
        return indices
    count = min(count, n_total)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n_total, size=count, replace=False)
    return sorted(int(i) for i in chosen)


def show_subset(
    stack: np.ndarray,
    indices: list[int],
    *,
    cols: int = 4,
    title_prefix: str = "I",
    save_path: Path | None = None,
) -> None:
    """Matplotlib grid of selected particles with per-panel contrast scaling."""
    n = len(indices)
    cols = max(1, min(cols, n))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.8 * rows), squeeze=False)
    fig.suptitle(f"Projection dataset ({n} of {stack.shape[0]} views)", fontsize=12)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, idx in zip(axes.ravel(), indices):
        # Raw stacks are often unnormalized (sums >> 1); match the interactive toy scaling.
        panel = normalize_display(stack[idx])
        ax.imshow(panel, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
        ax.set_title(f"{title_prefix}_{idx}")
        ax.axis("off")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    """CLI for the projection dataset viewer."""
    parser = argparse.ArgumentParser(description="View a subset of generated 2D projections.")
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=Path("projection_dataset.npz"),
        help="Path to .npz from generate_projection_dataset.py",
    )
    parser.add_argument(
        "--key",
        default="images",
        help="Array to display: images (I_i) or geometric_projections (default: images)",
    )
    parser.add_argument(
        "-k",
        "--count",
        type=int,
        default=12,
        help="How many views to show if --indices not set (default: 12)",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        metavar="I",
        help="Explicit view indices, e.g. --indices 0 3 7 15",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Grid columns (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed when sampling --count random views (default: 0)",
    )
    parser.add_argument(
        "-o",
        "--save",
        type=Path,
        default=None,
        help="Save montage to PNG instead of opening a window",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: load dataset and show or save a montage."""
    args = parse_args()
    path = Path(args.dataset)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Generate first:\n"
            f"  python generate_projection_dataset.py -n 50 -o {path}"
        )

    stack = load_images(path, key=args.key)
    indices = pick_indices(stack.shape[0], args.indices, args.count, args.seed)
    print(f"Showing indices: {indices}")
    show_subset(stack, indices, cols=args.cols, save_path=args.save)


if __name__ == "__main__":
    main()
