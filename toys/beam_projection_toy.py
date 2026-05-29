#!/usr/bin/env python3
"""
Cryo-EM single-particle toy: interactive 3D structure z and forward-model pipeline

    I_i = h_i * T_{t_i} P_{R_i} z + η_i

with visual stages for projection, translation, PSF convolution, and noise.

Default phantom: a square pyramid (apex +Z, square base in XY) so top, bottom,
and side views are easy to tell apart in projections and reconstruction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from scipy import ndimage
from scipy.spatial.transform import Rotation

try:
    import mrcfile
except ImportError:  # pragma: no cover
    mrcfile = None


def _mesh_to_volume(mesh: pv.PolyData, size: int) -> np.ndarray:
    """Voxelize a closed surface into a smooth density map (Z, Y, X), apex toward +Z."""
    mesh = mesh.triangulate().clean()
    half = (size - 1) / 2.0
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size]
    coords = np.c_[
        (xx - half).ravel(),
        (yy - half).ravel(),
        (zz - half).ravel(),
    ]
    cloud = pv.PolyData(coords)
    interior = cloud.select_interior_points(mesh, check_surface=False)
    mask = interior["selected_points"].reshape((size, size, size)).astype(np.float32)
    density = ndimage.gaussian_filter(mask, sigma=0.9)
    peak = float(density.max())
    if peak > 0:
        density /= peak

    # VTK interior test can flip Z; keep the density peak on the mesh apex (+Z).
    apex_z = float(mesh.points[:, 2].max())
    iz_peak = int(np.argmax(density))
    iz_apex = int(np.argmin(np.abs((zz - half)[:, 0, 0] - apex_z)))
    if abs(iz_peak - iz_apex) > size // 4:
        density = density[::-1, :, :]

    return density


def _fit_mesh_to_voxel_grid(mesh: pv.PolyData, size: int) -> pv.PolyData:
    """Center a mesh and scale it to fit inside a ``size``³ voxel grid."""
    mesh = mesh.triangulate().clean()
    mesh.points -= np.array(mesh.center)
    extent = np.ptp(mesh.points, axis=0).max()
    target = 0.82 * (size - 1)
    if extent > 0:
        mesh.points *= target / extent
    return mesh


def make_pyramid_mesh(size: int = 96) -> pv.PolyData:
    """
    Square pyramid surface mesh (VTK x, y, z).

    Apex at +Z, square base parallel to the XY plane. Side views show a triangle;
    top/bottom views show a square (base) or a point (apex).
    """
    height = 52.0
    base_radius = 24.0
    pyramid = pv.Cone(
        center=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 1.0),
        height=height,
        radius=base_radius,
        capping=True,
        resolution=4,
    )
    return _fit_mesh_to_voxel_grid(pyramid, size)


def make_pyramid_volume(size: int = 96) -> np.ndarray:
    """Procedural pyramid density (Z, Y, X); see :func:`make_pyramid_mesh`."""
    return _mesh_to_volume(make_pyramid_mesh(size), size)


def _bulldog_part(
    shape: pv.DataSet,
    center: tuple[float, float, float],
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    rotate: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> pv.PolyData:
    """Place and scale a primitive; coordinates are (x, y, z), nose toward +y."""
    part = shape.copy()
    part.scale(scale, inplace=True)
    if rotate[0]:
        part.rotate_x(rotate[0], inplace=True)
    if rotate[1]:
        part.rotate_y(rotate[1], inplace=True)
    if rotate[2]:
        part.rotate_z(rotate[2], inplace=True)
    part.translate(center, inplace=True)
    return part


def make_french_bulldog_mesh(size: int = 96) -> pv.PolyData:
    """
    Procedural French bulldog surface mesh (VTK x, y, z; nose toward +Y).

    Same geometry as :func:`make_french_bulldog_volume`, returned before voxelization.
    """
    parts: list[pv.PolyData] = []

    def add(shape: pv.DataSet, center, scale=(1.0, 1.0, 1.0), rotate=(0.0, 0.0, 0.0)) -> None:
        parts.append(_bulldog_part(shape, center, scale, rotate))

    # Chest and abdomen (compact, muscular)
    add(pv.Sphere(radius=13), (0, -7, -3), (1.2, 1.35, 1.05))
    add(pv.Sphere(radius=10), (0, -5, -11), (1.05, 1.15, 0.75))
    add(pv.Sphere(radius=8), (0, -12, -6), (0.95, 1.0, 0.9))

    # Neck
    add(pv.Sphere(radius=7), (0, 0, 0), (1.05, 0.85, 0.95))

    # Skull: wide, square, brachycephalic
    add(pv.Sphere(radius=12), (0, 7, 5), (1.45, 1.0, 1.2))
    add(pv.Sphere(radius=9), (0, 5, 11), (1.25, 0.85, 0.55))

    # Flat muzzle and nose (signature Frenchie face)
    add(pv.Sphere(radius=8), (0, 17, 0), (1.55, 0.62, 0.9))
    add(pv.Sphere(radius=6), (0, 21, -2), (1.35, 0.42, 0.95))
    add(pv.Sphere(radius=2.8), (0, 23.5, -1), (1.0, 0.75, 0.8))

    # Cheek jowls
    add(pv.Sphere(radius=5), (-8, 14, -3), (0.85, 0.9, 0.95))
    add(pv.Sphere(radius=5), (8, 14, -3), (0.85, 0.9, 0.95))

    # Bat ears, set wide on the skull
    add(pv.Sphere(radius=5.2), (-11.5, 9, 13), (0.5, 0.42, 1.45), rotate=(18, 0, 24))
    add(pv.Sphere(radius=5.2), (11.5, 9, 13), (0.5, 0.42, 1.45), rotate=(18, 0, -24))

    # Forehead skin folds
    add(pv.Sphere(radius=2.2), (-4, 12, 9), (1.6, 0.45, 0.55))
    add(pv.Sphere(radius=2.2), (4, 12, 9), (1.6, 0.45, 0.55))
    add(pv.Sphere(radius=1.8), (0, 13, 8), (2.0, 0.4, 0.5))

    # Eyes (slight indent bulges around sockets)
    add(pv.Sphere(radius=2.4), (-5.5, 15, 5), (1.0, 0.65, 0.85))
    add(pv.Sphere(radius=2.4), (5.5, 15, 5), (1.0, 0.65, 0.85))

    # Short, bowed legs and paws
    for x, y in ((-7.5, 2), (7.5, 2), (-7.5, -8), (7.5, -8)):
        add(pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=3.6, height=10), (x, y, -15))
        add(pv.Sphere(radius=3.8), (x, y, -20), (1.1, 1.15, 0.65))

    # Rump and nub tail
    add(pv.Sphere(radius=9), (0, -16, -2), (1.0, 0.95, 0.95))
    add(pv.Sphere(radius=4.2), (0, -21, 3), (0.75, 0.95, 0.85), rotate=(35, 0, 0))

    dog = parts[0]
    for part in parts[1:]:
        dog = dog.merge(part)
    dog = dog.clean()
    return _fit_mesh_to_voxel_grid(dog, size)


def make_french_bulldog_volume(size: int = 96) -> np.ndarray:
    """Procedural French bulldog density (Z, Y, X); see :func:`make_french_bulldog_mesh`."""
    return _mesh_to_volume(make_french_bulldog_mesh(size), size)


def make_demo_volume(size: int = 96) -> np.ndarray:
    """Default phantom: square pyramid."""
    return make_pyramid_volume(size)


def load_volume(path: Path | None) -> np.ndarray:
    if path is None:
        return make_demo_volume()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".npy":
        vol = np.load(path)
    elif suffix in {".npz"}:
        with np.load(path) as data:
            key = "vol" if "vol" in data else data.files[0]
            vol = data[key]
    elif suffix in {".mrc", ".rec"}:
        if mrcfile is None:
            raise ImportError("Install mrcfile to read MRC volumes: pip install mrcfile")
        with mrcfile.open(path, permissive=True) as handle:
            vol = np.asarray(handle.data, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported format: {path.suffix} (use .npy, .npz, or .mrc)")

    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {vol.shape}")
    return np.asarray(vol, dtype=np.float32)


def volume_to_mesh(volume: np.ndarray, level: float | None = None) -> pv.PolyData:
    """
    Isosurface mesh centered at the origin for trackball rotation.

    VTK ``ImageData`` uses (x, y, z) with x the fastest index; the cryo volume is
    stored (z, y, x), so we transpose before contouring so the mesh matches the
    same lab frame as :func:`project_along_beam_vtk`.
    """
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        raise ValueError("Volume has no finite values")

    if level is None:
        level = 0.35 * float(np.percentile(finite, 92))

    nz, ny, nx = volume.shape
    vol_xyz = np.transpose(volume, (2, 1, 0))
    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.spacing = (1.0, 1.0, 1.0)
    grid.origin = (-(nx - 1) / 2.0, -(ny - 1) / 2.0, -(nz - 1) / 2.0)
    grid.point_data["values"] = vol_xyz.ravel(order="C")
    return grid.contour([level], scalars="values").clean()


# VTK/display uses (x, y, z); volume array is indexed (z, y, x).
_XYZ_TO_ZYX = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)

# Fixed lab-frame beam and detector (top view: source at +Z, propagation −Z).
FIXED_BEAM_DIRECTION_XYZ = np.array([0.0, 0.0, -1.0], dtype=np.float64)
FIXED_DETECTOR_UP_XYZ = np.array([0.0, 1.0, 0.0], dtype=np.float64)


def xyz_to_zyx_vector(vector_xyz: np.ndarray) -> np.ndarray:
    """Map a direction from VTK lab (x, y, z) to NumPy volume axes (z, y, x)."""
    v = np.asarray(vector_xyz, dtype=float).reshape(3)
    return np.array([v[2], v[1], v[0]])


def _unit_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=float).reshape(3)
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        out = np.asarray(fallback, dtype=float).reshape(3)
        out /= np.linalg.norm(out) + 1e-12
        return out
    return v / norm


def detector_frame_xyz(
    beam_direction_xyz: np.ndarray,
    view_up_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Orthonormal detector frame in VTK (x, y, z).

    Returns ``(propagation, u, v)`` where propagation is source→detector (into the
    specimen), ``u`` is screen right, and ``v`` is screen up — matching PyVista's
    ``cross(view_direction, view_up)`` convention.
    """
    propagation = _unit_vector(beam_direction_xyz, np.array([0.0, 0.0, -1.0]))
    up = np.asarray(view_up_xyz, dtype=float).reshape(3)
    v = up - np.dot(up, propagation) * propagation
    v = _unit_vector(v, np.array([0.0, 1.0, 0.0]))
    u = np.cross(propagation, v)
    u = _unit_vector(u, np.array([1.0, 0.0, 0.0]))
    v = np.cross(u, propagation)
    v = _unit_vector(v, np.array([0.0, 1.0, 0.0]))
    return propagation, u, v


def _actor_matrix_xyz(actor: pv.Actor) -> np.ndarray:
    """Return a proper 3×3 rotation from the actor's VTK user/matrix transform."""
    matrix = np.eye(4)
    if hasattr(actor, "user_matrix") and actor.user_matrix is not None:
        matrix = np.array(actor.user_matrix, dtype=float)
    elif hasattr(actor, "matrix") and actor.matrix is not None:
        matrix = np.array(actor.matrix, dtype=float)
    rot = matrix[:3, :3].copy()
    u, _, vt = np.linalg.svd(rot)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    return rot


def actor_rotation_vtk(actor: pv.Actor) -> Rotation:
    """Particle orientation R_i in VTK (x, y, z) — same frame as the on-screen mesh."""
    return Rotation.from_matrix(_actor_matrix_xyz(actor))


def actor_rotation_for_projection(actor: pv.Actor) -> Rotation:
    """Same VTK rotation as the on-screen mesh (requires aligned :func:`volume_to_mesh`)."""
    return actor_rotation_vtk(actor)


def actor_rotation(actor: pv.Actor) -> Rotation:
    """
    Legacy (z, y, x) rotation from the actor matrix.

    Prefer :func:`actor_rotation_for_projection` for the interactive viewer.
    """
    rot_xyz = _actor_matrix_xyz(actor)
    return Rotation.from_matrix(_XYZ_TO_ZYX @ rot_xyz @ _XYZ_TO_ZYX.T)


def volume_zyx_to_xyz(volume_zyx: np.ndarray) -> np.ndarray:
    """Reorder (z, y, x) cryo volume to VTK (x, y, z) voxel indices."""
    return np.transpose(volume_zyx, (2, 1, 0))


def rotate_volume_xyz(volume_xyz: np.ndarray, rotation: Rotation) -> np.ndarray:
    """Resample an (x, y, z) volume after applying ``rotation`` in VTK lab coordinates."""
    nx, ny, nz = volume_xyz.shape
    center = np.array([(nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0])
    inv = rotation.inv().as_matrix()
    offset = center - inv @ center
    return ndimage.affine_transform(
        volume_xyz,
        matrix=inv,
        offset=offset,
        order=1,
        mode="constant",
        cval=0.0,
    )


def rotate_volume(volume: np.ndarray, rotation: Rotation) -> np.ndarray:
    """Resample volume in lab frame after applying particle rotation."""
    nz, ny, nx = volume.shape
    center = np.array([(nz - 1) / 2.0, (ny - 1) / 2.0, (nx - 1) / 2.0])
    inv = rotation.inv().as_matrix()
    offset = center - inv @ center
    return ndimage.affine_transform(
        volume,
        matrix=inv,
        offset=offset,
        order=1,
        mode="constant",
        cval=0.0,
    )


def project_along_beam_vtk(
    volume_zyx: np.ndarray,
    particle_rotation_xyz: Rotation,
    beam_direction_xyz: np.ndarray | None = None,
    detector_up_xyz: np.ndarray | None = None,
    *,
    actor_matrix_xyz: np.ndarray | None = None,
) -> np.ndarray:
    """
    Parallel-beam line integral in VTK (x, y, z), matched to the 3D viewer.

    Uses the same lab-frame particle pose as the PyVista actor: the reference volume
    is rotated by ``R`` before integrating along the fixed beam. Detector rows/columns
    follow ``detector_up_xyz`` and the beam (default −Z, up +Y).
    """
    if beam_direction_xyz is None:
        beam_direction_xyz = FIXED_BEAM_DIRECTION_XYZ
    if detector_up_xyz is None:
        detector_up_xyz = FIXED_DETECTOR_UP_XYZ

    if actor_matrix_xyz is not None:
        rotation = Rotation.from_matrix(np.asarray(actor_matrix_xyz, dtype=np.float64).reshape(3, 3))
    else:
        rotation = particle_rotation_xyz

    vol = rotate_volume_xyz(volume_zyx_to_xyz(volume_zyx), rotation)
    propagation, u, v = detector_frame_xyz(beam_direction_xyz, detector_up_xyz)

    frame_rot, _ = Rotation.align_vectors(
        np.stack([propagation, v, u]),
        np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    vol = rotate_volume_xyz(vol, frame_rot)
    return vol.sum(axis=2).astype(np.float32)


def project_along_beam(
    volume: np.ndarray,
    particle_rotation: Rotation,
    beam_direction_xyz: np.ndarray,
    detector_up_xyz: np.ndarray | None = None,
) -> np.ndarray:
    """
    Line integral along the beam (batch datasets, rotation in volume z, y, x axes).

    For the interactive viewer use :func:`project_along_beam_vtk` with
    :func:`actor_rotation_for_projection`.
    """
    aligned = rotate_volume(volume, particle_rotation)
    default_up = np.array([0.0, 1.0, 0.0])
    if detector_up_xyz is None:
        propagation, u, v = detector_frame_xyz(beam_direction_xyz, default_up)
    else:
        propagation, u, v = detector_frame_xyz(beam_direction_xyz, detector_up_xyz)

    frame_zyx = np.stack(
        [
            xyz_to_zyx_vector(propagation),
            xyz_to_zyx_vector(v),
            xyz_to_zyx_vector(u),
        ],
        axis=0,
    )
    target = np.eye(3, dtype=float)
    align_frame, _ = Rotation.align_vectors(frame_zyx, target)
    aligned = rotate_volume(aligned, align_frame)
    return aligned.sum(axis=0).astype(np.float32)


def beam_tilt_angles_degrees(beam_direction_xyz: np.ndarray) -> tuple[float, float]:
    """Return (tilt from +Z, azimuth in XY) in degrees for display."""
    d = np.asarray(beam_direction_xyz, dtype=float).reshape(3)
    d /= np.linalg.norm(d)
    tilt = float(np.degrees(np.arccos(np.clip(d[2], -1.0, 1.0))))
    azimuth = float(np.degrees(np.arctan2(d[1], d[0])))
    return tilt, azimuth


def electron_wavelength_angstrom(voltage_kv: float) -> float:
    """Relativistic de Broglie wavelength for electrons (Å)."""
    voltage = voltage_kv * 1000.0
    return 12.2643247 / np.sqrt(voltage * (1.0 + 0.978466e-6 * voltage))


def simulate_wave_detector_image(
    geometric_projection: np.ndarray,
    wavelength_angstrom: float,
    defocus_angstrom: float,
    pixel_size_angstrom: float = 1.0,
    phase_scale: float = np.pi,
    absorption_strength: float = 0.28,
) -> np.ndarray:
    """
    Propagate a weakly scattering exit wave to the detector (Fresnel / angular spectrum).

    The geometric projection is treated as projected potential V(u, v). The exit
    wave combines weak absorption and phase:

        ψ₀(u, v) = (1 - α·V) · exp(i·φ·V)

    Defocus phase in the back focal plane:

        χ(k) = π · λ · Δz · |k|²

    Detector intensity I(u, v) = |ψ_Δz(u, v)|². Larger λ spreads diffraction and
    shifts CTF zeros (more pronounced fringes / contrast variation).
    """
    wavelength_angstrom = max(float(wavelength_angstrom), 1e-6)
    pixel_size_angstrom = max(float(pixel_size_angstrom), 1e-6)

    potential = geometric_projection.astype(np.float64)
    potential -= potential.min()
    peak = potential.max()
    if peak > 0:
        potential /= peak

    psi0 = (1.0 - absorption_strength * potential) * np.exp(1j * phase_scale * potential)

    ny, nx = psi0.shape
    fx = np.fft.fftfreq(nx, d=pixel_size_angstrom)
    fy = np.fft.fftfreq(ny, d=pixel_size_angstrom)
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing="ij")
    chi = np.pi * wavelength_angstrom * defocus_angstrom * (fx_grid**2 + fy_grid**2)
    propagator = np.exp(1j * chi)

    psi_detector = np.fft.ifft2(np.fft.fft2(psi0) * propagator)
    intensity = np.abs(psi_detector) ** 2
    intensity -= intensity.min()
    peak = intensity.max()
    if peak > 0:
        intensity /= peak
    return intensity.astype(np.float32)


def ctf_transfer_preview(
    image_shape: tuple[int, int],
    wavelength_angstrom: float,
    defocus_angstrom: float,
    pixel_size_angstrom: float = 1.0,
) -> np.ndarray:
    """Weak-phase CTF envelope sin(χ) for the 2D pipeline inset (same χ as propagation)."""
    ny, nx = image_shape
    pixel_size_angstrom = max(float(pixel_size_angstrom), 1e-6)
    fx = np.fft.fftfreq(nx, d=pixel_size_angstrom)
    fy = np.fft.fftfreq(ny, d=pixel_size_angstrom)
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing="ij")
    chi = np.pi * wavelength_angstrom * defocus_angstrom * (fx_grid**2 + fy_grid**2)
    ctf = np.sin(chi)
    ctf -= ctf.min()
    peak = ctf.max()
    if peak > 0:
        ctf /= peak
    return ctf.astype(np.float32)


@dataclass
class ImageFormationResult:
    """Stages of the single-particle forward model for one view i."""

    projection: np.ndarray  # P_{R_i} z
    translated: np.ndarray  # T_{t_i} P_{R_i} z
    blurred: np.ndarray  # h_i * T_{t_i} P_{R_i} z
    observed: np.ndarray  # I_i
    psf_kernel: np.ndarray
    rotation: Rotation
    translation_px: tuple[float, float]
    psf_sigma_px: float
    noise_sigma: float
    use_ctf: bool = False


def translate_image(image: np.ndarray, shift_y: float, shift_x: float) -> np.ndarray:
    """T_t: shift in detector (row=y, col=x) pixels."""
    return ndimage.shift(
        image,
        shift=(shift_y, shift_x),
        order=1,
        mode="constant",
        cval=0.0,
    ).astype(np.float32)


def gaussian_psf_kernel(sigma_px: float, size: int | None = None) -> np.ndarray:
    """Circular Gaussian PSF h_i (normalized)."""
    sigma_px = max(float(sigma_px), 0.05)
    if size is None:
        size = int(max(7, 2 * round(3 * sigma_px) + 1))
    if size % 2 == 0:
        size += 1
    yy, xx = np.mgrid[0:size, 0:size]
    center = (size - 1) / 2.0
    kernel = np.exp(-((yy - center) ** 2 + (xx - center) ** 2) / (2 * sigma_px**2))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def convolve_psf(image: np.ndarray, sigma_px: float) -> tuple[np.ndarray, np.ndarray]:
    """h_i * image using a Gaussian microscope PSF."""
    kernel = gaussian_psf_kernel(sigma_px)
    blurred = ndimage.convolve(image, kernel, mode="constant", cval=0.0)
    return blurred.astype(np.float32), kernel


def psf_sigma_pixels(
    wavelength_angstrom: float,
    defocus_angstrom: float,
    pixel_size_angstrom: float,
    image_shape: tuple[int, int],
) -> float:
    """Heuristic PSF width (px) from defocus and wavelength (toy CTF scale)."""
    ny, nx = image_shape
    n = max(nx, ny)
    scale = wavelength_angstrom * abs(defocus_angstrom) / max(pixel_size_angstrom**2, 1e-6)
    sigma = 0.35 * np.sqrt(scale / n) * n / 12.0
    return float(np.clip(sigma, 0.4, 10.0))


def add_detector_noise(
    image: np.ndarray,
    noise_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """η_i: additive white noise on the detector image."""
    if noise_sigma <= 0:
        return image.astype(np.float32)
    noisy = image + rng.normal(0.0, noise_sigma, image.shape)
    return np.clip(noisy, 0.0, None).astype(np.float32)


def normalize_display(image: np.ndarray) -> np.ndarray:
    """Scale an image to [0, 1] for display (min–max per array)."""
    out = image.astype(np.float64)
    out -= out.min()
    peak = out.max()
    if peak > 0:
        out /= peak
    return out.astype(np.float32)


def run_image_formation(
    volume: np.ndarray,
    particle_rotation: Rotation,
    beam_direction_xyz: np.ndarray,
    translation_px: tuple[float, float],
    psf_sigma_px: float,
    noise_sigma: float,
    rng: np.random.Generator,
    detector_up_xyz: np.ndarray | None = None,
    *,
    particle_actor: pv.Actor | None = None,
    use_ctf: bool = False,
    wavelength_angstrom: float = 1.0,
    defocus_angstrom: float = 2.0e4,
    pixel_size_angstrom: float = 1.0,
    phase_scale: float = np.pi,
    absorption_strength: float = 0.28,
) -> ImageFormationResult:
    """Full forward model: I_i = h_i * T_{t_i} P_{R_i} z + η_i (PSF or Fresnel CTF)."""
    if particle_actor is not None:
        rotation = actor_rotation_for_projection(particle_actor)
        projection = project_along_beam_vtk(
            volume,
            rotation,
            beam_direction_xyz=beam_direction_xyz,
            detector_up_xyz=detector_up_xyz,
            actor_matrix_xyz=_actor_matrix_xyz(particle_actor),
        )
        particle_rotation = rotation
    else:
        projection = project_along_beam(
            volume,
            particle_rotation,
            beam_direction_xyz,
            detector_up_xyz=detector_up_xyz,
        )
    tx, ty = translation_px
    translated = translate_image(projection, ty, tx)
    if use_ctf:
        blurred = simulate_wave_detector_image(
            translated,
            wavelength_angstrom,
            defocus_angstrom,
            pixel_size_angstrom=pixel_size_angstrom,
            phase_scale=phase_scale,
            absorption_strength=absorption_strength,
        )
        kernel = ctf_transfer_preview(
            translated.shape,
            wavelength_angstrom,
            defocus_angstrom,
            pixel_size_angstrom,
        )
    else:
        blurred, kernel = convolve_psf(translated, psf_sigma_px)
    observed = add_detector_noise(blurred, noise_sigma, rng)
    return ImageFormationResult(
        projection=projection,
        translated=translated,
        blurred=blurred,
        observed=observed,
        psf_kernel=kernel,
        rotation=particle_rotation,
        translation_px=(tx, ty),
        psf_sigma_px=psf_sigma_px,
        noise_sigma=noise_sigma,
        use_ctf=use_ctf,
    )


# PyVista slider layout (normalized viewport coordinates).
_SLIDER_ROW_TRANSLATION_Y = 0.18
_SLIDER_ROW_OPTICS_Y = 0.10
_SLIDER_TRACKS = {
    "three_col": [(0.04, 0.32), (0.36, 0.64), (0.68, 0.96)],
    "two_col": [(0.06, 0.48), (0.52, 0.94)],
}
_VIEWPORT_W = 1100
_VIEWPORT_H = 720


def configure_matplotlib_for_pyvistaqt() -> None:
    """
    Matplotlib must use a Qt-compatible backend when sharing the PyVistaQt app.

    Call once in main() before creating figures. Do not call plt.ion() or app.exec()
    separately — use a single plt.show() at the end.
    """
    import matplotlib

    backend = matplotlib.get_backend().lower()
    if "inline" in backend or "qt" in backend or "tk" in backend:
        return
    for name in ("QtAgg", "qtagg", "TkAgg"):
        try:
            matplotlib.use(name)
            return
        except ImportError:
            continue


class ProjectionToy:
    """
    Interactive cryo-EM forward-model demo.

    Fixed top-down camera and fixed electron beam along −Z. Dragging the particle
    updates the 2D forward-model panels so they match the 3D view (same VTK frame).
    """

    def __init__(
        self,
        volume: np.ndarray,
        iso_level: float | None = None,
        pixel_size_angstrom: float = 1.0,
        voltage_kv: float = 300.0,
    ) -> None:
        """Build 3D + 2D viewers for ``volume`` (Z, Y, X) with default 300 kV optics."""
        self.volume = volume
        n = int(volume.shape[0])
        # Default phantom: share one pyramid surface for the viewer and the voxel grid.
        if volume.shape[0] == volume.shape[1] == volume.shape[2]:
            self.mesh = make_pyramid_mesh(n)
        else:
            self.mesh = volume_to_mesh(volume, level=iso_level)
        self.pixel_size_angstrom = pixel_size_angstrom
        self.voltage_kv = voltage_kv
        self.wavelength_angstrom = electron_wavelength_angstrom(voltage_kv)
        self.defocus_angstrom = 2.0e4
        self.translation_px = (0.0, 0.0)
        self.noise_sigma = 0.04
        self.apply_ctf = False
        self.phase_scale = float(np.pi)
        self.absorption_strength = 0.28
        self._rng = np.random.default_rng(0)
        self._arrow_patches: list = []
        self.beam_direction_xyz = FIXED_BEAM_DIRECTION_XYZ.copy()
        self.detector_up_xyz = FIXED_DETECTOR_UP_XYZ.copy()

        # --- 3D view (PyVistaQt) ---
        self.plotter = BackgroundPlotter(
            title="Cryo-EM projection toy (pyramid)",
            window_size=(_VIEWPORT_W, _VIEWPORT_H),
            show=True,
        )
        self.plotter.set_background("white")
        self._enable_trackball_actor()

        self.particle_actor = self.plotter.add_mesh(
            self.mesh,
            color="#5ba3d9",
            opacity=0.92,
            smooth_shading=True,
            specular=0.4,
            specular_power=20,
            ambient=0.35,
            diffuse=0.85,
            pickable=True,
            name="particle",
        )
        self._set_initial_camera()

        self._rotation_label = self.plotter.add_text(
            "R_i : drag particle",
            position="upper_right",
            font_size=10,
            color="black",
            shadow=False,
        )
        self.plotter.add_text(
            "z (3D)  |  fixed top view  |  beam −Z  |  drag particle  |  B / P / C",
            position="lower_left",
            font_size=9,
            color="black",
            shadow=False,
        )

        self.beam_on = False
        self.beam_actor_entries: list = []
        self.beam_point_a = np.array([0.0, 0.0, 1.0])
        self.beam_point_b = np.array([0.0, 0.0, -1.0])
        self._reset_beam_endpoints()

        self._add_control_sliders()

        self.plotter.add_key_event("b", lambda: self.toggle_beam())
        self.plotter.add_key_event("p", lambda: self.project())
        self.plotter.add_checkbox_button_widget(
            self.set_beam,
            value=False,
            position=(20, 20),
            size=28,
            border_size=2,
            color_on="#ffd966",
            color_off="#888888",
        )
        self.plotter.add_text(
            "Electron beam",
            position=(58, 24),
            font_size=10,
            color="black",
            shadow=False,
        )
        self.plotter.add_checkbox_button_widget(
            self.set_apply_ctf,
            value=self.apply_ctf,
            position=(20, 56),
            size=28,
            border_size=2,
            color_on="#5ba3d9",
            color_off="#888888",
        )
        self.plotter.add_text(
            "CTF (wave / Fresnel)",
            position=(58, 60),
            font_size=10,
            color="black",
            shadow=False,
        )
        self.plotter.add_key_event("c", lambda: self.toggle_ctf())
        self.plotter.iren.add_observer(
            "EndInteractionEvent",
            lambda *_: self._on_particle_interaction(),
        )

        # --- 2D pipeline (matplotlib), created after the Qt app exists ---
        self.fig = plt.figure(figsize=(13.0, 5.2), facecolor="white")
        manager = self.fig.canvas.manager
        if manager is not None:
            manager.set_window_title(
                r"Forward model  $I_i = h_i * T_{t_i} P_{R_i} z + \eta_i$"
            )
        self._pipeline_axes: list[plt.Axes] = []
        self._show_placeholder_projection()

    def _set_initial_camera(self) -> None:
        """Orthographic top view: camera at +Z, looking down at the XY plane."""
        self.plotter.enable_parallel_projection()
        self.plotter.view_xy(negative=False, render=False)
        self.plotter.reset_camera()
        self.plotter.camera.zoom(1.15)
        self._reference_camera = self._snapshot_camera()

    def _snapshot_camera(self) -> dict[str, np.ndarray]:
        """Store camera pose so we can restore the fixed top-down view."""
        cam = self.plotter.camera
        return {
            "position": np.asarray(cam.position, dtype=float).copy(),
            "focal_point": np.asarray(cam.focal_point, dtype=float).copy(),
            "up": np.asarray(cam.up, dtype=float).copy(),
        }

    def _restore_reference_camera(self) -> None:
        """Keep the 3D view locked; only the particle orientation changes."""
        ref = self._reference_camera
        cam = self.plotter.camera
        cam.position = tuple(ref["position"])
        cam.focal_point = tuple(ref["focal_point"])
        cam.up = tuple(ref["up"])
        self.plotter.render()

    def _enable_trackball_actor(self) -> None:
        """Use trackball rotation on the picked particle actor (not the camera)."""
        self.plotter.enable_trackball_actor_style()

    def _add_slider(
        self,
        callback,
        rng: tuple[float, float],
        value: float,
        title: str,
        pointa: tuple[float, float],
        pointb: tuple[float, float],
        fmt: str | None = None,
    ) -> None:
        """Add one labeled slider with consistent styling."""
        kwargs = dict(
            callback=callback,
            rng=rng,
            value=value,
            title=title,
            pointa=pointa,
            pointb=pointb,
            interaction_event="end",
            title_color="black",
            title_height=0.028,
            color="#2c5f8a",
        )
        if fmt is not None:
            kwargs["fmt"] = fmt
        self.plotter.add_slider_widget(**kwargs)

    def _add_control_sliders(self) -> None:
        """Create translation, noise, wavelength, and defocus sliders in the 3D viewport."""
        lambda_default = electron_wavelength_angstrom(self.voltage_kv)
        self.wavelength_angstrom = lambda_default
        row1 = _SLIDER_TRACKS["three_col"]
        row2 = _SLIDER_TRACKS["two_col"]
        y1 = _SLIDER_ROW_TRANSLATION_Y
        y2 = _SLIDER_ROW_OPTICS_Y

        self._add_slider(
            self._set_translation_x,
            (-18.0, 18.0),
            0.0,
            "t_x  (pixels)",
            (row1[0][0], y1),
            (row1[0][1], y1),
            fmt="%.1f",
        )
        self._add_slider(
            self._set_translation_y,
            (-18.0, 18.0),
            0.0,
            "t_y  (pixels)",
            (row1[1][0], y1),
            (row1[1][1], y1),
            fmt="%.1f",
        )
        self._add_slider(
            self._set_noise_sigma,
            (0.0, 0.2),
            self.noise_sigma,
            "noise  sigma",
            (row1[2][0], y1),
            (row1[2][1], y1),
            fmt="%.3f",
        )
        self._add_slider(
            self._set_wavelength_angstrom,
            (0.002, 0.25),
            lambda_default,
            "wavelength  (A)",
            (row2[0][0], y2),
            (row2[0][1], y2),
            fmt="%.3f",
        )
        self._add_slider(
            self._set_defocus_angstrom,
            (0.0, 8.0e4),
            self.defocus_angstrom,
            "defocus  dz (A)",
            (row2[1][0], y2),
            (row2[1][1], y2),
            fmt="%.0f",
        )

    def _set_translation_x(self, value: float) -> None:
        """Slider callback: update t_x and refresh projection if beam is on."""
        self.translation_px = (float(value), self.translation_px[1])
        self._maybe_auto_project()

    def _set_translation_y(self, value: float) -> None:
        """Slider callback: update t_y and refresh projection if beam is on."""
        self.translation_px = (self.translation_px[0], float(value))
        self._maybe_auto_project()

    def _set_noise_sigma(self, value: float) -> None:
        """Slider callback: update additive noise σ on the detector."""
        self.noise_sigma = float(value)
        self._maybe_auto_project()

    def _set_wavelength_angstrom(self, value: float) -> None:
        """Slider callback: update electron wavelength (Å) and redraw beam rings."""
        self.wavelength_angstrom = float(value)
        self._draw_beam()
        self._maybe_auto_project()

    def _set_defocus_angstrom(self, value: float) -> None:
        """Slider callback: update defocus Δz (Å) for the PSF model."""
        self.defocus_angstrom = float(value)
        self._maybe_auto_project()

    def _update_rotation_label(self, rotation: Rotation) -> None:
        """Refresh the on-screen Euler-angle label for the current particle pose."""
        euler = rotation.as_euler("zyx", degrees=True)
        self._rotation_label.SetText(
            0,
            f"R_i : ({euler[0]:.0f}°, {euler[1]:.0f}°, {euler[2]:.0f}°)  [drag particle]",
        )

    def _particle_bounds(self) -> tuple[float, ...]:
        """VTK axis-aligned bounds (xmin, xmax, ymin, ymax, zmin, zmax) of the particle."""
        return self.particle_actor.GetBounds()

    def _beam_direction_xyz(self) -> np.ndarray:
        """Fixed lab-frame beam propagation (source at +Z → detector at −Z)."""
        return self.beam_direction_xyz

    def _reset_beam_endpoints(self) -> None:
        """Fixed −Z column through the particle (independent of camera motion)."""
        xmin, xmax, ymin, ymax, zmin, zmax = self._particle_bounds()
        center = np.array(
            [
                0.5 * (xmin + xmax),
                0.5 * (ymin + ymax),
                0.5 * (zmin + zmax),
            ]
        )
        span = max(xmax - xmin, ymax - ymin, zmax - zmin, 1.0)
        half = 0.85 * span
        prop = self.beam_direction_xyz
        self.beam_point_a = center - prop * half
        self.beam_point_b = center + prop * half

    def _on_particle_interaction(self) -> None:
        """After dragging the particle, restore the camera and refresh the 2D panels."""
        self._restore_reference_camera()
        self._maybe_auto_project()

    def _clear_beam_graphics(self) -> None:
        """Remove tube, wavefront disks, arrow, and detector plane actors."""
        for entry in self.beam_actor_entries:
            self.plotter.remove_actor(entry)
        self.beam_actor_entries.clear()

    def _draw_beam(self) -> None:
        """Draw beam tube, λ-spaced wavefronts, source arrow, and detector plane."""
        self._clear_beam_graphics()
        if not self.beam_on:
            return

        xmin, xmax, ymin, ymax, zmin, zmax = self._particle_bounds()
        span = max(xmax - xmin, ymax - ymin, zmax - zmin, 1.0)
        tube_radius = 0.035 * span
        direction = self._beam_direction_xyz()

        # Coherent wave column: tube + λ-spaced wavefront disks along propagation
        line = pv.Line(self.beam_point_a, self.beam_point_b)
        tube = line.tube(radius=tube_radius, n_sides=18)
        actor = self.plotter.add_mesh(
            tube,
            color="#ffd966",
            opacity=0.35,
            name="beam",
        )
        self.beam_actor_entries.append(actor)

        wavelength_vox = self.wavelength_angstrom / self.pixel_size_angstrom
        path_length = float(np.linalg.norm(self.beam_point_b - self.beam_point_a))
        if wavelength_vox > 1e-4:
            n_rings = min(24, max(3, int(path_length / wavelength_vox)))
            ring_radius = 0.55 * max(xmax - xmin, ymax - ymin)
            for i in range(n_rings + 1):
                t = i / max(n_rings, 1)
                center = self.beam_point_a + t * (self.beam_point_b - self.beam_point_a)
                disc = pv.Disc(
                    center=center,
                    inner=0.0,
                    outer=ring_radius,
                    normal=direction,
                    c_res=1,
                    r_res=48,
                )
                actor = self.plotter.add_mesh(
                    disc,
                    color="#fff2b3" if i % 2 == 0 else "#ffd966",
                    opacity=0.12,
                    style="surface",
                    name="wavefront",
                )
                self.beam_actor_entries.append(actor)

        # Arrowhead at the camera (beam origin), pointing into the specimen
        arrow_len = 0.2 * span
        arrow = pv.Arrow(
            start=self.beam_point_a,
            direction=direction,
            scale=arrow_len,
            tip_length=0.35,
            tip_radius=0.12,
            shaft_radius=0.04,
        )
        actor = self.plotter.add_mesh(arrow, color="#ffeb99", opacity=0.85, name="beam_arrow")
        self.beam_actor_entries.append(actor)

        # Detector plane perpendicular to the beam at the downstream end
        pad = 0.18 * span
        plane = pv.Plane(
            center=self.beam_point_b,
            direction=direction,
            i_size=(xmax - xmin) + 2 * pad,
            j_size=(ymax - ymin) + 2 * pad,
        )
        actor = self.plotter.add_mesh(
            plane,
            color="#ffe08a",
            opacity=0.2,
            style="surface",
            name="detector",
        )
        self.beam_actor_entries.append(actor)

    def set_beam(self, on: bool) -> None:
        """Enable or disable the electron beam (checkbox / key B)."""
        self.beam_on = on
        if self.beam_on:
            self._reset_beam_endpoints()
            self._draw_beam()
            self.project()
        else:
            self._clear_beam_graphics()

    def toggle_beam(self) -> None:
        """Flip beam on/off (keyboard shortcut)."""
        self.set_beam(not self.beam_on)

    def set_apply_ctf(self, state: bool) -> None:
        """Checkbox: Fresnel / CTF corruption instead of Gaussian PSF blur."""
        self.apply_ctf = bool(state)
        self._maybe_auto_project()

    def toggle_ctf(self) -> None:
        """Toggle CTF wave model (keyboard shortcut C)."""
        self.set_apply_ctf(not self.apply_ctf)

    def _maybe_auto_project(self) -> None:
        """Re-run the forward model when parameters change and the beam is active."""
        if self.beam_on:
            self.project()

    def _clear_figure_arrows(self) -> None:
        """Remove ConnectionPatch arrows between matplotlib pipeline panels."""
        for patch in self._arrow_patches:
            try:
                patch.remove()
            except (ValueError, AttributeError):
                pass
        self._arrow_patches.clear()

    def _draw_pipeline_arrows(self) -> None:
        """Draw arrows between the four forward-model stage axes."""
        if len(self._pipeline_axes) < 4:
            return
        self._clear_figure_arrows()
        for left, right in zip(self._pipeline_axes[:-1], self._pipeline_axes[1:]):
            arrow = ConnectionPatch(
                xyA=(1.01, 0.5),
                coordsA=left.transAxes,
                xyB=(-0.01, 0.5),
                coordsB=right.transAxes,
                axesA=left,
                axesB=right,
                arrowstyle="-|>",
                shrinkA=0,
                shrinkB=0,
                mutation_scale=14,
                linewidth=1.2,
                color="#555555",
            )
            self.fig.add_artist(arrow)
            self._arrow_patches.append(arrow)

    def _show_formation_pipeline(self, result: ImageFormationResult, beam_tilt: float) -> None:
        """Render P, T, h*, and I_i panels for the current forward-model result."""
        self._clear_figure_arrows()
        self.fig.clf()
        self.fig.suptitle(
            r"$I_i = h_i * T_{t_i} P_{R_i} z + \eta_i$   "
            rf"(fixed beam −Z, tilt {beam_tilt:.0f}$^\circ$, "
            rf"$\lambda$={self.wavelength_angstrom * 100:.2f} pm, "
            rf"$\Delta z$={self.defocus_angstrom / 1e4:.2f} $\mu$m)",
            fontsize=10,
            y=0.98,
        )
        self._pipeline_axes = list(self.fig.subplots(1, 4))

        stages = [
            (result.projection, r"$P_{R_i} z$", "projection"),
            (
                result.translated,
                r"$T_{t_i}(P_{R_i} z)$",
                f"shift ({result.translation_px[0]:+.1f}, {result.translation_px[1]:+.1f}) px",
            ),
            (
                result.blurred,
                r"$h_i * (\cdot)$" if not result.use_ctf else r"CTF $\cdot$ (Fresnel)",
                (
                    f"PSF σ = {result.psf_sigma_px:.1f} px"
                    if not result.use_ctf
                    else (
                        rf"$\Delta z$ = {self.defocus_angstrom / 1e4:.2f} $\mu$m, "
                        rf"$\lambda$ = {self.wavelength_angstrom * 100:.1f} pm"
                    )
                ),
            ),
            (result.observed, r"$I_i$", rf"noise $\sigma$ = {result.noise_sigma:.3f}"),
        ]
        for ax, (image, title_main, title_sub) in zip(self._pipeline_axes, stages):
            ax.imshow(
                normalize_display(image),
                cmap="gray",
                origin="lower",
                interpolation="bilinear",
            )
            ax.set_title(f"{title_main}\n{title_sub}", fontsize=9, pad=6, linespacing=1.15)
            ax.set_xlabel("detector u  (lab X)", fontsize=8)
            if ax is self._pipeline_axes[0]:
                ax.set_ylabel("detector v  (lab Y)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_aspect("equal", adjustable="box")

        ax_psf = self._pipeline_axes[2].inset_axes([0.58, 0.58, 0.38, 0.38])
        ax_psf.imshow(result.psf_kernel, cmap="viridis", origin="lower")
        kernel_title = r"CTF $\sin\chi$" if result.use_ctf else r"$h_i$ kernel"
        ax_psf.set_title(kernel_title, fontsize=7, pad=2)
        ax_psf.set_xticks([])
        ax_psf.set_yticks([])

        self._draw_pipeline_arrows()
        self.fig.subplots_adjust(left=0.05, right=0.99, top=0.82, bottom=0.10, wspace=0.38)
        self.fig.canvas.draw_idle()

    def project(self) -> None:
        """Run the full forward model for the current pose and update the 2D window."""
        if not self.beam_on:
            return

        rotation = actor_rotation_for_projection(self.particle_actor)
        self._update_rotation_label(rotation)
        psf_sigma = psf_sigma_pixels(
            self.wavelength_angstrom,
            self.defocus_angstrom,
            self.pixel_size_angstrom,
            self.volume.shape[1:],
        )
        result = run_image_formation(
            self.volume,
            rotation,
            self.beam_direction_xyz,
            self.translation_px,
            psf_sigma,
            self.noise_sigma,
            self._rng,
            detector_up_xyz=self.detector_up_xyz,
            particle_actor=self.particle_actor,
            use_ctf=self.apply_ctf,
            wavelength_angstrom=self.wavelength_angstrom,
            defocus_angstrom=self.defocus_angstrom,
            pixel_size_angstrom=self.pixel_size_angstrom,
            phase_scale=self.phase_scale,
            absorption_strength=self.absorption_strength,
        )
        direction = self.beam_direction_xyz
        tilt, _ = beam_tilt_angles_degrees(direction)
        self._show_formation_pipeline(result, tilt)

    def _show_placeholder_projection(self) -> None:
        """Show instructions in the 2D window before the beam is enabled."""
        self._clear_figure_arrows()
        self.fig.clf()
        self._pipeline_axes = []
        self.fig.suptitle(
            r"$I_i = h_i * T_{t_i} P_{R_i} z + \eta_i$",
            fontsize=12,
            y=0.98,
        )
        ax = self.fig.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.5,
            0.55,
            "Turn on the electron beam (B)\nto run the forward-model pipeline\n"
            "Check CTF (C) for Fresnel / defocus fringes",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#555555",
            fontsize=12,
        )
        ax.text(
            0.5,
            0.30,
            "Fixed beam along −Z (top view)\n"
            "2D panels match the 3D viewer — drag particle to change R_i",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#777777",
            fontsize=10,
        )
        self.fig.subplots_adjust(top=0.88)
        self.fig.canvas.draw_idle()


def parse_args() -> argparse.Namespace:
    """CLI for the interactive projection toy."""
    parser = argparse.ArgumentParser(
        description="Interactive 3D volume viewer: top view, beam along view axis, forward-model pipeline.",
    )
    parser.add_argument(
        "volume",
        nargs="?",
        type=Path,
        default=None,
        help="3D map (.mrc, .npy, .npz). Default: built-in square pyramid phantom.",
    )
    parser.add_argument(
        "--iso",
        type=float,
        default=None,
        help="Isosurface level (default: auto from 92nd percentile).",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=1.0,
        help="Voxel size in Å for wave propagation (default: 1.0).",
    )
    parser.add_argument(
        "--voltage-kv",
        type=float,
        default=300.0,
        help="Microscope voltage in kV for default electron wavelength (default: 300).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: launch the interactive PyVistaQt + matplotlib application."""
    configure_matplotlib_for_pyvistaqt()
    args = parse_args()
    volume = load_volume(args.volume)
    ProjectionToy(
        volume,
        iso_level=args.iso,
        pixel_size_angstrom=args.pixel_size,
        voltage_kv=args.voltage_kv,
    )
    # Single Qt event loop for both PyVistaQt and matplotlib windows.
    plt.show()


if __name__ == "__main__":
    main()
