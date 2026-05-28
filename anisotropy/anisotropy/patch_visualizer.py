"""
Interactive PyVista viewer for patch-wise mesh parameterization.

Loads a SAS mesh + ``patch_features.npz`` and exposes checkboxes for scalar
fields (hydropathy, charge, …) and vector overlays (dipoles, normals).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from anisotropy.mesh import ProteinMesh
from anisotropy.patches import FEATURE_MATRIX_NAMES, PatchParameterization

try:
    import pyvista as pv
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyVista is required for patch visualization. "
        'Install with: pip install -e ".[view]"'
    ) from exc


@dataclass(frozen=True)
class ScalarLayer:
    """One per-face scalar coloring layer."""

    key: str
    label: str
    cmap: str
    clim: tuple[float, float] | None = None  # None → data min/max


@dataclass(frozen=True)
class VectorLayer:
    """Arrow glyph overlay at patch centroids."""

    key: str
    label: str
    color: str
    scale_key: str  # "dipole" | "normal"


SCALAR_LAYERS: tuple[ScalarLayer, ...] = (
    ScalarLayer("hydropathy", "Hydropathy", "coolwarm"),
    ScalarLayer("polar_density", "Polar density", "viridis", (0.0, 1.0)),
    ScalarLayer("hbond_score", "H-bond score", "plasma"),
    ScalarLayer("charge", "Charge (e)", "coolwarm"),
    ScalarLayer("potential", "Potential", "coolwarm"),
    ScalarLayer("pka_acid", "pKa (acid)", "cividis"),
    ScalarLayer("mean_curvature", "Mean curvature H", "RdBu_r"),
    ScalarLayer("gaussian_curvature", "Gaussian curvature K", "RdBu_r"),
    ScalarLayer("softness", "Softness", "magma"),
    ScalarLayer("area", "Patch area", "viridis"),
    ScalarLayer("dipole_magnitude", "|Dipole|", "viridis"),
    ScalarLayer("patch_id", "Patch ID", "tab20"),
)

VECTOR_LAYERS: tuple[VectorLayer, ...] = (
    VectorLayer("dipoles", "Dipole vectors", "darkorange", "dipole"),
    VectorLayer("normals", "Outward normals", "seagreen", "normal"),
)


def _patch_values(param: PatchParameterization, key: str) -> np.ndarray:
    if key == "patch_id":
        return np.arange(param.n_patches, dtype=np.float64)
    col = FEATURE_MATRIX_NAMES.index(key)
    return param.feature_matrix()[:, col]


def _face_scalars(param: PatchParameterization, key: str) -> np.ndarray:
    vals = _patch_values(param, key)
    return vals[param.face_patch_ids]


def _mesh_with_face_scalars(mesh: ProteinMesh, param: PatchParameterization) -> pv.PolyData:
    surf = mesh.to_pyvista()
    for layer in SCALAR_LAYERS:
        surf.cell_data[layer.key] = _face_scalars(param, layer.key)
    return surf


def _arrow_glyph(
    centroids: np.ndarray,
    vectors: np.ndarray,
    *,
    color: str,
    length_scale: float,
) -> pv.PolyData:
    mag = np.linalg.norm(vectors, axis=1)
    nonzero = mag > 1e-12
    if not np.any(nonzero):
        return pv.PolyData()
    pts = centroids[nonzero]
    vecs = vectors[nonzero]
    mag_nz = mag[nonzero]
    vecs = vecs / mag_nz[:, None] * (mag_nz[:, None] * length_scale)
    cloud = pv.PolyData(pts)
    cloud["vectors"] = vecs
    return cloud.glyph(orient="vectors", scale=False, factor=1.0, color=color)


class PatchMeshVisualizer:
    """PyVista plotter with checkbox widgets for patch features."""

    def __init__(
        self,
        mesh: ProteinMesh,
        param: PatchParameterization,
        *,
        window_size: tuple[int, int] = (1280, 800),
        arrow_scale: float | None = None,
    ) -> None:
        self.mesh = mesh
        self.param = param
        self.surface = _mesh_with_face_scalars(mesh, param)
        self.centroids = np.stack([p.centroid for p in param.patches], axis=0)
        self.dipoles = np.stack([p.dipole for p in param.patches], axis=0)
        self.normals = np.stack([p.normal for p in param.patches], axis=0)

        bounds = self.surface.bounds
        extent = max(
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
            1.0,
        )
        self.arrow_length = (
            float(arrow_scale) if arrow_scale is not None else 0.12 * extent
        )

        self.plotter = pv.Plotter(window_size=window_size)
        self.plotter.set_background("white")
        self._active_scalar: str | None = None
        self._vector_actors: dict[str, Any] = {}
        self._checkbox_widgets: dict[str, Any] = {}
        self._suppress_callbacks = False

        self.mesh_actor = self.plotter.add_mesh(
            self.surface,
            color="#b0b0b0",
            opacity=0.92,
            smooth_shading=True,
            show_scalar_bar=False,
        )
        self.plotter.add_text(
            f"Patch parameterization (pH={param.ph:.1f}, {param.n_patches} patches)\n"
            "Scalars: one colormap at a time. Vectors stack with scalars.",
            position="upper_left",
            font_size=10,
            color="black",
        )

    def _apply_scalar(self, key: str | None) -> None:
        self._active_scalar = key
        self.plotter.remove_actor(self.mesh_actor)
        if key is None:
            self.mesh_actor = self.plotter.add_mesh(
                self.surface,
                color="#b0b0b0",
                opacity=0.92,
                smooth_shading=True,
                show_scalar_bar=False,
            )
            self.plotter.render()
            return

        layer = next(ly for ly in SCALAR_LAYERS if ly.key == key)
        scalars = np.asarray(self.surface.cell_data[key], dtype=np.float64)
        clim = layer.clim
        if clim is None:
            clim = (float(np.nanmin(scalars)), float(np.nanmax(scalars)))
            if clim[0] == clim[1]:
                clim = (clim[0] - 1.0, clim[1] + 1.0)

        self.mesh_actor = self.plotter.add_mesh(
            self.surface,
            scalars=key,
            cmap=layer.cmap,
            clim=clim,
            opacity=0.92,
            smooth_shading=True,
            scalar_bar_args={"title": layer.label, "vertical": True},
        )
        self.plotter.render()

    def _uncheck_other_scalars(self, active_key: str) -> None:
        self._suppress_callbacks = True
        try:
            for layer in SCALAR_LAYERS:
                if layer.key == active_key:
                    continue
                widget = self._checkbox_widgets.get(f"scalar:{layer.key}")
                if widget is not None:
                    rep = widget.GetRepresentation()
                    if rep.GetState():
                        rep.SetState(0)
        finally:
            self._suppress_callbacks = False

    def _toggle_scalar(self, key: str, state: bool) -> None:
        if self._suppress_callbacks:
            return
        if state:
            self._uncheck_other_scalars(key)
            self._apply_scalar(key)
        elif self._active_scalar == key:
            self._apply_scalar(None)

    def _toggle_vectors(self, layer: VectorLayer, state: bool) -> None:
        if not state:
            actor = self._vector_actors.pop(layer.key, None)
            if actor is not None:
                self.plotter.remove_actor(actor)
            self.plotter.render()
            return

        if layer.scale_key == "dipole":
            vecs = self.dipoles
            color = layer.color
        else:
            vecs = self.normals
            color = layer.color

        glyph = _arrow_glyph(
            self.centroids,
            vecs,
            color=color,
            length_scale=self.arrow_length,
        )
        if glyph.n_points == 0:
            return
        actor = self.plotter.add_mesh(glyph, color=color, opacity=0.95)
        self._vector_actors[layer.key] = actor
        self.plotter.render()

    def _add_labeled_checkbox(
        self,
        *,
        key: str,
        label: str,
        callback: Callable[[bool], None],
        position: tuple[float, float],
        value: bool = False,
    ) -> None:
        widget = self.plotter.add_checkbox_button_widget(
            callback,
            value=value,
            position=position,
            size=22,
            border_size=2,
        )
        self._checkbox_widgets[key] = widget
        self.plotter.add_text(
            label,
            position=(position[0] + 30, position[1] + 4),
            font_size=9,
            color="black",
        )

    def add_widgets(self) -> None:
        """Place checkbox column on the left margin."""
        x0, y0 = 12.0, 12.0
        dy = 34.0
        y = y0 + (len(SCALAR_LAYERS) + len(VECTOR_LAYERS) + 1) * dy

        self.plotter.add_text(
            "Surface scalars",
            position=(x0, y),
            font_size=10,
            color="dimgray",
        )
        y -= dy

        for layer in SCALAR_LAYERS:
            key = f"scalar:{layer.key}"

            def _cb(state: bool, *, _k: str = layer.key) -> None:
                self._toggle_scalar(_k, state)

            self._add_labeled_checkbox(
                key=key,
                label=layer.label,
                callback=_cb,
                position=(x0, y),
            )
            y -= dy

        y -= 10
        self.plotter.add_text(
            "Vector overlays",
            position=(x0, y),
            font_size=10,
            color="dimgray",
        )
        y -= dy

        for layer in VECTOR_LAYERS:
            key = f"vector:{layer.key}"

            def _vcb(state: bool, *, _ly: VectorLayer = layer) -> None:
                self._toggle_vectors(_ly, state)

            self._add_labeled_checkbox(
                key=key,
                label=layer.label,
                callback=_vcb,
                position=(x0, y),
            )
            y -= dy

    def show(self, *, interactive: bool = True) -> None:
        self.add_widgets()
        self.plotter.show(interactive=interactive, auto_close=False)


def show_patch_parameterization(
    mesh: ProteinMesh,
    param: PatchParameterization,
    **kwargs: Any,
) -> None:
    """Open the interactive patch visualizer."""
    viz = PatchMeshVisualizer(mesh, param, **kwargs)
    viz.show()
