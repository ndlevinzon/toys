# Cryo-EM single-particle toys

Educational Python tools that mirror pieces of a cryo-EM single-particle workflow: forward-model projections of a **square pyramid** phantom (apex +Z), 2D classification, ab initio reconstruction, and orientation-bias visualization. Built for thesis demos and experimentation—not a replacement for cryoSPARC/RELION.

## Setup

```powershell
cd toys
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

**Dependencies:** NumPy, SciPy, Matplotlib, PyVista, PyVistaQt, PyQt5, `mrcfile` (optional MRC I/O), `imageio[ffmpeg]` (video export).

## Recommended pipeline

### Uniform orientations (i.i.d. on SO(3))

```text
beam_projection_toy.py          Interactive forward model (optional exploration)
        ↓
generate_projection_dataset.py  → projection_dataset.npz
        ↓
view_projection_dataset.py        QC montage
        ↓
classify_2d.py                    → classification_2d.npz
        ↓
view_2d_classes.py                Inspect classes
        ↓
ab_initio_reconstruct.py          → reconstruction_ab_initio.npz
        ↓
view_reconstruction.py            3D isosurface viewer
```

### Orientation-biased dataset (same pipeline, new filenames)

Matches the vMF peak used in `orientation_distribution.py` (preferred view = pyramid apex +Z):

```powershell
python generate_projection_dataset.py -n 1000 --orientation-bias 0.85
# default output: projection_dataset_biased.npz

python view_projection_dataset.py projection_dataset_biased.npz -k 12
python classify_2d.py projection_dataset_biased.npz -o classification_2d_biased.npz
python view_2d_classes.py classification_2d_biased.npz
python ab_initio_reconstruct.py --dataset projection_dataset_biased.npz --quality -o reconstruction_ab_initio_biased.npz
python view_reconstruction.py reconstruction_ab_initio_biased.npz --with-phantom
```

**Orientation bias (interactive):** `orientation_distribution.py` previews the same bias parameter on the pyramid surface (arrow density).

**Presentation videos:** `render_bulldog_rotation.py` (ground-truth pyramid spin) and `render_reconstruction_rotation.py` (ab-initio isosurface spin, same 8 s / 30 fps / top view).

## Python files

### `beam_projection_toy.py`

**Role:** Core library + interactive GUI for the single-particle forward model  
`I_i = h_i * T_{t_i} P_{R_i} z + η_i`.

| Area | Contents |
|------|----------|
| Phantom | `make_pyramid_mesh`, `make_pyramid_volume`, `make_demo_volume` (also `make_french_bulldog_*` legacy) |
| I/O | `load_volume` (.npy, .npz, .mrc) |
| Geometry | `project_along_beam`, `rotate_volume`, `xyz_to_zyx_vector`, `actor_rotation` |
| Optics | `simulate_wave_detector_image`, `electron_wavelength_angstrom`, PSF helpers |
| Forward model | `run_image_formation`, `ImageFormationResult` |
| GUI | `ProjectionToy` — top-down PyVistaQt 3D view + matplotlib 4-panel pipeline (beam along view axis) |

**Run:** `python beam_projection_toy.py`  
Keys: **B** beam, **P** project, **C** CTF (Fresnel); checkbox **CTF (wave / Fresnel)** applies defocus fringes instead of Gaussian PSF. Drag particle for `R_i`, sliders for `t_i`, `λ`, `Δz`, `η_i`.

---

### `generate_projection_dataset.py`

**Role:** Batch-generate `projection_dataset.npz` from the **square pyramid** phantom with random SO(3) orientations.

- Uses `project_along_beam_vtk` (same lab frame and fixed −Z beam as the interactive toy).
- Default detector images use **Fresnel wave propagation** (diffraction fringes).
- Stores `images`, `geometric_projections`, `rotations`, `phantom`, beam/detector metadata.
- **`--orientation-bias B`** (0–1): uniform SO(3) at 0; vMF toward `--mu` at 1 (shared `orientation_sampling.py` with the visualizer).

**Run (uniform):** `python generate_projection_dataset.py -n 1000`

**Run (biased):** `python generate_projection_dataset.py -n 1000 --orientation-bias 0.85 -o projection_dataset_biased.npz`

Optional: `--phantom-size 96`, `--kappa-max 85`, `--mu 0 0 1`; pass a `.mrc`/`.npy` path for a custom map.

---

### `view_projection_dataset.py`

**Role:** Matplotlib montage of a subset of particles from a dataset.

**Run:** `python view_projection_dataset.py projection_dataset.npz -k 12`

---

### `classify_2d.py`

**Role:** cryoSPARC-style **2D classification** toy — iterative class assignment with in-plane rotation + translation (NCC), then class-average updates.

**Output:** `classification_2d.npz` (`class_averages`, `assignments`, `particles`, alignments).

**Run:** `python classify_2d.py projection_dataset.npz -k 8 --iterations 8`

---

### `view_2d_classes.py`

**Role:** Viewer for 2D class averages and top-scoring exemplars per class.

**Run:** `python view_2d_classes.py classification_2d.npz`  
`--mode averages` for a compact class grid only.

---

### `ab_initio_reconstruct.py`

**Role:** **Ab initio 3D reconstruction** — alternates orientation search (3D + in-plane + shift) with ramp-filtered weighted back-projection and light SIRT.

**Important:**
- Prefer `--dataset projection_dataset.npz` (auto-uses `geometric_projections`).
- Do **not** apply 2D class in-plane alignments (breaks 3D geometry).
- Avoid donut artifacts: use `--quality`, enough iterations; diagnostic: `--use-gt-orientations`.

**Run:** `python ab_initio_reconstruct.py --dataset projection_dataset.npz --quality`

---

### `view_reconstruction.py`

**Role:** PyVista 3D isosurface of `reconstruction_ab_initio.npz` with an **isosurface level** slider.

**Run:** `python view_reconstruction.py reconstruction_ab_initio.npz --with-phantom`

---

### `orientation_distribution.py`

**Role:** Intuitive **orientation sampling** on the pyramid outer surface. The mesh is drawn first; short equal-length normal arrows sit on the skin. **Spacing** (not length) encodes probability: **uniform** = evenly spaced arrows (all orientations equally likely); **biased** = clusters where views are sampled, bare patches where orientations are missing from the dataset.

**Run:** `python orientation_distribution.py` — interactive **orientation bias** slider (0 = evenly spaced arrows; 1 = dense clusters on the gold preferred view + large empty regions). Local arrow **density** encodes sampling probability; direction = viewing axis.

Provides `sample_rotations()` hooks for future biased projection generation.

---

### `render_bulldog_rotation.py`

**Role:** Off-screen render of a slow 360° rotation about +Y to `pyramid_rotation.mp4` (white background, top-down orthographic camera matching the toy).

**Run:** `python render_bulldog_rotation.py -o pyramid_rotation.mp4 --duration 8`

Optional: `--show-beam` draws the fixed −Z beam column.

### `render_reconstruction_rotation.py`

**Role:** Same spin settings as the pyramid movie (8 s, 30 fps, +Y, top view) for `reconstruction_ab_initio.npz`.

**Run:** `python render_reconstruction_rotation.py -o reconstruction_rotation.mp4`

Optional: `--iso` or `--iso-percentile` for the isosurface threshold.

---

## Data files (generated)

| File | Description |
|------|-------------|
| `projection_dataset.npz` | Uniform-orientation particle stack + metadata |
| `projection_dataset_biased.npz` | vMF-biased orientations (`orientation_bias`, `preferred_axis_xyz`) |
| `classification_2d.npz` | Class averages, assignments, preprocessed particles |
| `reconstruction_ab_initio.npz` | Reconstructed volume + estimated orientations |
| `pyramid_rotation.mp4` | Ground-truth pyramid spin (default output) |
| `reconstruction_rotation.mp4` | Ab-initio reconstruction spin (default output) |

## Coordinate conventions

- **Volume arrays:** `(Z, Y, X)` NumPy indexing.
- **VTK / PyVista meshes:** `(x, y, z)` lab frame; default pyramid apex toward **+Z**, base in **XY**.
- **Beam default:** propagation **−Z** (top → bottom), matching the interactive toy.

## See also

In-code API documentation: every public function and class method has a docstring. Browse `beam_projection_toy.py` for the shared forward-model primitives imported by other scripts.
