#!/usr/bin/env python3
"""Viewer for 2D classification results (class averages + exemplar particles)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beam_projection_toy import normalize_display
from classify_2d import apply_alignment, load_classification


def display_image(ax: plt.Axes, image: np.ndarray, title: str = "") -> None:
    """Show one particle or class average with robust percentile contrast."""
    panel = normalize_display(image) if image.max() > 1.5 or image.min() < -0.5 else image
    lo, hi = np.percentile(panel, (1.0, 99.0))
    if hi <= lo:
        lo, hi = float(panel.min()), float(panel.max() + 1e-6)
    ax.imshow(panel, cmap="gray", vmin=lo, vmax=hi, interpolation="bilinear")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def pick_exemplars(
    assignments: np.ndarray,
    scores: np.ndarray,
    class_id: int,
    n_exemplars: int,
) -> list[int]:
    """Highest-scoring members of a class (best aligned to the average)."""
    members = np.where(assignments == class_id)[0]
    if members.size == 0:
        return []
    order = members[np.argsort(-scores[members])]
    return order[:n_exemplars].tolist()


def show_2d_classes(
    class_averages: np.ndarray,
    particles: np.ndarray,
    assignments: np.ndarray,
    *,
    angles_deg: np.ndarray | None = None,
    shifts_yx: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    class_sizes: np.ndarray | None = None,
    n_exemplars: int = 5,
    cols: int | None = None,
    title: str = "2D classification",
    save_path: Path | None = None,
) -> None:
    """Rows of class averages with top-scoring aligned exemplar particles."""
    n_classes, _, _ = class_averages.shape
    if scores is None:
        scores = np.zeros(len(assignments), dtype=np.float32)
    if class_sizes is None:
        class_sizes = np.bincount(assignments, minlength=n_classes)

    cols = cols or min(4, n_classes)
    cols = max(1, min(cols, n_classes))
    rows_per_block = 1 + n_exemplars
    fig_h = 2.2 * n_classes / cols * rows_per_block
    fig, axes = plt.subplots(
        n_classes,
        1 + n_exemplars,
        figsize=(2.0 * (1 + n_exemplars), fig_h),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.4] + [1.0] * n_exemplars},
    )
    fig.suptitle(title, fontsize=12, y=0.995)

    for k in range(n_classes):
        row_axes = axes[k]
        count = int(class_sizes[k]) if k < len(class_sizes) else 0
        display_image(row_axes[0], class_averages[k]) # title=f"Class {k}  (n={count})"

        exemplar_indices = pick_exemplars(assignments, scores, k, n_exemplars)
        for j in range(1, 1 + n_exemplars):
            ax = row_axes[j]
            if j - 1 < len(exemplar_indices):
                idx = exemplar_indices[j - 1]
                particle = particles[idx]
                if angles_deg is not None and shifts_yx is not None:
                    particle = apply_alignment(
                        particle,
                        float(angles_deg[idx]),
                        (float(shifts_yx[idx, 0]), float(shifts_yx[idx, 1])),
                    )
                display_image(ax, particle, title=f"#{idx}")
            else:
                ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")
    else:
        plt.show()


def show_class_grid(
    class_averages: np.ndarray,
    class_sizes: np.ndarray | None = None,
    *,
    cols: int = 4,
    title: str = "2D class averages",
    save_path: Path | None = None,
) -> None:
    """Compact grid of class averages only (cryoSPARC-style class panel)."""
    n_classes = class_averages.shape[0]
    cols = max(1, min(cols, n_classes))
    rows = int(np.ceil(n_classes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.8 * rows), squeeze=False)
    fig.suptitle(title, fontsize=12)

    for ax in axes.ravel():
        ax.axis("off")

    for k, ax in enumerate(axes.ravel()):
        if k >= n_classes:
            break
        label = f"Class {k}"
        if class_sizes is not None and k < len(class_sizes):
            label += f"  (n={int(class_sizes[k])})"
        display_image(ax, class_averages[k], title=label)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    """CLI for the 2D classification viewer."""
    parser = argparse.ArgumentParser(description="View 2D classification results.")
    parser.add_argument(
        "classification",
        nargs="?",
        type=Path,
        default=Path("classification_2d.npz"),
        help="Output from classify_2d.py",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "averages"),
        default="full",
        help="full: averages + exemplars; averages: class grid only",
    )
    parser.add_argument(
        "--exemplars",
        type=int,
        default=5,
        help="Exemplar particles per class in full mode (default: 5)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Grid columns in averages mode (default: 4)",
    )
    parser.add_argument(
        "-o",
        "--save",
        type=Path,
        default=None,
        help="Save figure to PNG",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: load classification_2d.npz and open the viewer."""
    args = parse_args()
    path = Path(args.classification)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run:\n  python classify_2d.py projection_dataset.npz"
        )

    data = load_classification(path)
    class_averages = np.asarray(data["class_averages"])
    assignments = np.asarray(data["assignments"])
    particles = np.asarray(data["particles"])
    class_sizes = np.asarray(data.get("class_sizes", []))
    angles = np.asarray(data["angles_deg"]) if "angles_deg" in data else None
    shifts = np.asarray(data["shifts_yx"]) if "shifts_yx" in data else None
    scores = np.asarray(data["scores"]) if "scores" in data else None

    source = str(data.get("source_dataset", ""))
    title = f"2D classification — {path.name}"
    if source:
        title += f"\n(from {source})"

    if args.mode == "averages":
        show_class_grid(
            class_averages,
            class_sizes=class_sizes if class_sizes.size else None,
            cols=args.cols,
            title=title,
            save_path=args.save,
        )
    else:
        show_2d_classes(
            class_averages,
            particles,
            assignments,
            angles_deg=angles,
            shifts_yx=shifts,
            scores=scores,
            class_sizes=class_sizes if class_sizes.size else None,
            n_exemplars=args.exemplars,
            title=title,
            save_path=args.save,
        )


if __name__ == "__main__":
    main()
