"""
Air–water interface (AWI) as a depth-resolved field for cryo-EM solvent modeling.

The interface is not a plane with a single dielectric constant or fixed surface
charge. Experimental constraints motivate:

* Structural anisotropy decay length ~6–8 Å (not a delta function at z=0).
* Sum-frequency generation at water–air interfaces dominated by the topmost
  one to two water layers.
* Depth-dependent dielectric response (parallel vs perpendicular split), with a
  sharp but finite transition across the interface (Chiang-type profiles).

Field vector along depth z (distance into the fluid from one interface, Å):

    y(z) = (ρ(z), ε∥(z), ε⊥(z), φ₀(z), E₀(z), κ(z), γ, c, m)

Scalars γ (capillary stiffness / surface tension), c (surface coverage state),
and m (interface age / viscoelastic film state) are attached per interface.

Cryo-EM uses a **finite unsupported vitrified water slab** with **two**
independent interfaces (top and bottom). Each side may have different c and m
(e.g. sacrificial protein film on top, nearer-pristine AWI below).

**Electrostatic caution:** electrostatic, electrochemical, and electrokinetic
surface potentials are distinct observables and may differ in sign. This module
keeps dielectric structure (ε∥, ε⊥) and intrinsic potential φ₀(z) as separate
fields. Do not fold them into a single fixed σ_surf. Optional calibration hooks
map φ₀ or κ to a named observable (e.g. zeta) when comparing to experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np

# Typical decay of interfacial structural anisotropy (Å).
DEFAULT_ANISOTROPY_DECAY_ANGSTROM = 7.0  # mid-range of ~6–8 Å literature band

# Grid extent beyond which bulk solvent values are reached (Å).
DEFAULT_PROFILE_EXTENT_ANGSTROM = 40.0


class SurfaceCoverage(str, Enum):
    """Surface coverage state c (per interface)."""

    PRISTINE = "pristine"
    SURFACTANT = "surfactant"
    PROTEIN_SACRIFICIAL = "protein_sacrificial"
    PASSIVATED = "passivated"
    CONTAMINATED = "contaminated"


class InterfaceMechanicalState(str, Enum):
    """
    Interface age / viscoelastic film state m (per interface).

    First-generation models may only use γ; m is reserved for later coupling to
    interfacial rheology and orientation bias.
    """

    FRESH = "fresh"
    EQUILIBRATED = "equilibrated"
    VISCOELASTIC_FILM = "viscoelastic_film"
    AGED_PROTEIN_LAYER = "aged_protein_layer"


class PotentialObservable(str, Enum):
    """Which experimental potential a calibration target refers to."""

    INTRINSIC_PHI0 = "intrinsic_phi0"
    ELECTROCHEMICAL = "electrochemical"
    ELECTROKINETIC_ZETA = "electrokinetic_zeta"


@dataclass
class DepthProfile:
    """
    Depth-resolved fields on a 1D grid z ≥ 0 (into the fluid from one interface).

    All array fields share the same shape as ``z``.
    """

    z: np.ndarray
    rho: np.ndarray  # density / occupancy (arbitrary units or relative to bulk)
    epsilon_parallel: np.ndarray
    epsilon_perpendicular: np.ndarray
    phi_0: np.ndarray  # intrinsic interfacial potential φ₀(z) (V)
    E_0: np.ndarray  # normal component of field tied to φ₀, E₀(z) = −dφ₀/dz (V/Å)
    kappa: np.ndarray  # ionic screening field κ(z) (1/Å), e.g. Debye 1/λ_D

    def __post_init__(self) -> None:
        n = self.z.shape[0]
        for name in (
            "rho",
            "epsilon_parallel",
            "epsilon_perpendicular",
            "phi_0",
            "E_0",
            "kappa",
        ):
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")

    @property
    def epsilon_scalar(self) -> np.ndarray:
        """Isotropic proxy ε = (ε∥ + ε⊥) / 2 when tensor detail is unused."""
        return 0.5 * (self.epsilon_parallel + self.epsilon_perpendicular)

    def y_vector(self, index: int) -> np.ndarray:
        """
        Stack y(z_i) = (ρ, ε∥, ε⊥, φ₀, E₀, κ) at grid index ``index``.

        γ, c, m are interface scalars and are not included here.
        """
        return np.array(
            [
                self.rho[index],
                self.epsilon_parallel[index],
                self.epsilon_perpendicular[index],
                self.phi_0[index],
                self.E_0[index],
                self.kappa[index],
            ],
            dtype=np.float64,
        )

    def sample(self, z_query: float | np.ndarray) -> dict[str, np.ndarray]:
        """Linear interpolation of all profile fields at depth(s) z_query (Å)."""
        zq = np.atleast_1d(np.asarray(z_query, dtype=np.float64))
        out: dict[str, np.ndarray] = {}
        for key in (
            "rho",
            "epsilon_parallel",
            "epsilon_perpendicular",
            "phi_0",
            "E_0",
            "kappa",
        ):
            out[key] = np.interp(zq, self.z, getattr(self, key))
        out["z"] = zq
        return out


@dataclass
class AWIInterface:
    """
    One air–water (or air–vitrified water) interface with its own (γ, c, m).

    ``normal`` is a unit vector pointing from the interface into the solvent slab.
    """

    profile: DepthProfile
    gamma: float  # capillary stiffness / surface tension (N/m)
    coverage: SurfaceCoverage = SurfaceCoverage.PRISTINE
    mechanical_state: InterfaceMechanicalState = InterfaceMechanicalState.FRESH
    mechanical_age: float = 0.0  # dimensionless auxiliary in [0, 1] for future rheology
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    potential_observable: PotentialObservable = PotentialObservable.INTRINSIC_PHI0
    phi0_calibration_scale: float = 1.0  # multiply φ₀ when matching an observable

    def __post_init__(self) -> None:
        n = np.linalg.norm(self.normal)
        if n < 1e-12:
            raise ValueError("interface normal must be non-zero")
        self.normal = np.asarray(self.normal, dtype=np.float64) / n

    @property
    def c(self) -> SurfaceCoverage:
        return self.coverage

    @property
    def m(self) -> InterfaceMechanicalState:
        return self.mechanical_state

    def phi0_observable(self, z: float = 0.0) -> float:
        """φ₀ at the interface, scaled for the chosen observable calibration."""
        val = float(self.profile.sample(z)["phi_0"][0])
        return val * self.phi0_calibration_scale


@dataclass
class VitrifiedWaterSlab:
    """
    Finite unsupported solvent slab for cryo-EM (two AWIs).

    Slab occupies z ∈ [0, thickness] in lab frame; bottom interface at z=0,
    top at z=thickness. Air / vacuum lies outside the slab along ±normal.
    """

    thickness_angstrom: float
    bottom: AWIInterface  # typically nearer-pristine
    top: AWIInterface  # may carry sacrificial protein / different c, m
    bulk_epsilon: float = 78.0
    bulk_rho: float = 1.0
    bulk_kappa: float = 0.0  # 1/Å; set from ionic strength if needed

    def depth_from_bottom(self, z_lab: float | np.ndarray) -> np.ndarray:
        z = np.atleast_1d(np.asarray(z_lab, dtype=np.float64))
        return np.clip(z, 0.0, self.thickness_angstrom)

    def depth_from_top(self, z_lab: float | np.ndarray) -> np.ndarray:
        z = np.atleast_1d(np.asarray(z_lab, dtype=np.float64))
        return np.clip(self.thickness_angstrom - z, 0.0, self.thickness_angstrom)

    def sample_fields(
        self,
        z_lab: float | np.ndarray,
        *,
        blend_interfaces: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Evaluate depth fields at lab-frame z inside the slab.

        When ``blend_interfaces`` is True, weights are Gaussian in distance to
        each interface (σ = decay length from bottom profile grid step or 7 Å).
        Otherwise the nearer interface dominates.
        """
        z = np.atleast_1d(np.asarray(z_lab, dtype=np.float64))
        d_bot = self.depth_from_bottom(z)
        d_top = self.depth_from_top(z)

        bot = self.bottom.profile
        top = self.top.profile

        sb = bot.sample(d_bot)
        st = top.sample(d_top)

        sigma = DEFAULT_ANISOTROPY_DECAY_ANGSTROM
        if not blend_interfaces:
            use_top = d_top < d_bot
            out = {k: np.where(use_top, st[k], sb[k]) for k in sb if k != "z"}
            out["z_lab"] = z
            out["depth_bottom"] = d_bot
            out["depth_top"] = d_top
            out["gamma"] = np.where(use_top, self.top.gamma, self.bottom.gamma)
            return out

        w_top = np.exp(-(d_top / sigma) ** 2)
        w_bot = np.exp(-(d_bot / sigma) ** 2)
        w_sum = w_bot + w_top + 1e-30

        def blend(key: str) -> np.ndarray:
            return (w_bot * sb[key] + w_top * st[key]) / w_sum

        out = {k: blend(k) for k in sb if k != "z"}
        out["z_lab"] = z
        out["depth_bottom"] = d_bot
        out["depth_top"] = d_top
        out["weight_bottom"] = w_bot / w_sum
        out["weight_top"] = w_top / w_sum
        return out

    def interface_state_at(self, z_lab: float) -> tuple[Literal["bottom", "top", "bulk"], AWIInterface]:
        """Which interface primarily governs chemistry at this z (heuristic)."""
        d_bot = float(self.depth_from_bottom(z_lab))
        d_top = float(self.depth_from_top(z_lab))
        margin = DEFAULT_ANISOTROPY_DECAY_ANGSTROM
        if d_bot < margin and d_bot <= d_top:
            return "bottom", self.bottom
        if d_top < margin:
            return "top", self.top
        return "bulk", self.bottom  # bulk: return bottom as placeholder

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        bot, top = self.bottom.profile, self.top.profile
        np.savez(
            path,
            thickness_angstrom=self.thickness_angstrom,
            bulk_epsilon=self.bulk_epsilon,
            z_grid=bot.z,
            bottom_rho=bot.rho,
            bottom_eps_par=bot.epsilon_parallel,
            bottom_eps_perp=bot.epsilon_perpendicular,
            bottom_phi0=bot.phi_0,
            bottom_E0=bot.E_0,
            bottom_kappa=bot.kappa,
            bottom_gamma=self.bottom.gamma,
            bottom_coverage=self.bottom.coverage.value,
            bottom_mechanical=self.bottom.mechanical_state.value,
            top_rho=top.rho,
            top_eps_par=top.epsilon_parallel,
            top_eps_perp=top.epsilon_perpendicular,
            top_phi0=top.phi_0,
            top_E0=top.E_0,
            top_kappa=top.kappa,
            top_gamma=self.top.gamma,
            top_coverage=self.top.coverage.value,
            top_mechanical=self.top.mechanical_state.value,
        )


# ---------------------------------------------------------------------------
# Profile builders
# ---------------------------------------------------------------------------


def _z_grid(
    extent: float = DEFAULT_PROFILE_EXTENT_ANGSTROM,
    dz: float = 0.25,
) -> np.ndarray:
    n = max(int(np.ceil(extent / dz)) + 1, 2)
    return np.linspace(0.0, extent, n, dtype=np.float64)


def _exp_decay(z: np.ndarray, length: float) -> np.ndarray:
    length = max(float(length), 0.5)
    return np.exp(-z / length)


def _tanh_transition(z: np.ndarray, z0: float, width: float) -> np.ndarray:
    width = max(float(width), 0.2)
    return 0.5 * (1.0 + np.tanh((z - z0) / width))


def _gradient(phi: np.ndarray, z: np.ndarray) -> np.ndarray:
    return -np.gradient(phi, z, edge_order=2)


def build_depth_profile(
    *,
    extent: float = DEFAULT_PROFILE_EXTENT_ANGSTROM,
    dz: float = 0.25,
    decay_length: float = DEFAULT_ANISOTROPY_DECAY_ANGSTROM,
    bulk_epsilon: float = 78.0,
    surface_epsilon_parallel: float = 2.5,
    surface_epsilon_perpendicular: float = 1.5,
    use_tensor_dielectric: bool = True,
    bulk_rho: float = 1.0,
    surface_rho_enhancement: float = 1.8,
    phi0_surface_volts: float = 0.05,
    phi0_decay_length: float | None = None,
    debye_length_angstrom: float | None = None,
    transition_width: float = 2.0,
    transition_center: float = 1.5,
) -> DepthProfile:
    """
  Build a schematic AWI depth profile.

  Dielectric: ε∥ and ε⊥ transition from surface values to ``bulk_epsilon`` over
  ``transition_width`` (Chiang-type sharp but finite step). If
  ``use_tensor_dielectric`` is False, both components follow the same scalar
  ε(z).

  ρ(z): enhanced near z=0 (top water layers), exponential decay into bulk.

  φ₀(z): small intrinsic potential at the surface, decaying separately from ε;
  **not** a fixed sheet charge. E₀ = −dφ₀/dz.

  κ(z): 1/λ_D in bulk if ``debye_length_angstrom`` is set, else 0.
  """
    z = _z_grid(extent, dz)
    decay = max(float(decay_length), 0.5)
    phi_decay = float(phi0_decay_length) if phi0_decay_length is not None else decay

    # Density: top 1–2 layers enhanced (~3 Å scale), then bulk.
    layer_scale = 3.0
    rho = bulk_rho * (
        1.0
        + (surface_rho_enhancement - 1.0)
        * np.exp(-z / layer_scale)
        * (1.0 + 0.3 * np.exp(-z / (layer_scale * 0.5)))
    )

    t = _tanh_transition(z, transition_center, transition_width)
    eps_par = surface_epsilon_parallel + (bulk_epsilon - surface_epsilon_parallel) * t
    eps_perp = surface_epsilon_perpendicular + (
        bulk_epsilon - surface_epsilon_perpendicular
    ) * t
    if not use_tensor_dielectric:
        eps_scalar = 0.5 * (eps_par + eps_perp)
        eps_par = eps_perp = eps_scalar

    # Intrinsic potential: separate from dielectric; small at interface.
    phi_0 = phi0_surface_volts * _exp_decay(z, phi_decay)
    E_0 = _gradient(phi_0, z)

    if debye_length_angstrom is not None and debye_length_angstrom > 0:
        kappa_bulk = 1.0 / float(debye_length_angstrom)
        kappa = kappa_bulk * (1.0 - _exp_decay(z, decay))
    else:
        kappa = np.zeros_like(z)

    return DepthProfile(
        z=z,
        rho=rho,
        epsilon_parallel=eps_par,
        epsilon_perpendicular=eps_perp,
        phi_0=phi_0,
        E_0=E_0,
        kappa=kappa,
    )


def build_interface(
    *,
    gamma: float = 0.0728,
    coverage: SurfaceCoverage = SurfaceCoverage.PRISTINE,
    mechanical_state: InterfaceMechanicalState = InterfaceMechanicalState.FRESH,
    mechanical_age: float = 0.0,
    normal: np.ndarray | None = None,
    profile: DepthProfile | None = None,
    **profile_kwargs,
) -> AWIInterface:
    """Assemble :class:`AWIInterface` with a generated or supplied profile."""
    if profile is None:
        profile = build_depth_profile(**profile_kwargs)
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return AWIInterface(
        profile=profile,
        gamma=gamma,
        coverage=coverage,
        mechanical_state=mechanical_state,
        mechanical_age=mechanical_age,
        normal=normal,
    )


def build_cryo_slab_preset(
    thickness_angstrom: float = 300.0,
    *,
    bottom_coverage: SurfaceCoverage = SurfaceCoverage.PRISTINE,
    top_coverage: SurfaceCoverage = SurfaceCoverage.PROTEIN_SACRIFICIAL,
    bottom_mechanical: InterfaceMechanicalState = InterfaceMechanicalState.FRESH,
    top_mechanical: InterfaceMechanicalState = InterfaceMechanicalState.VISCOELASTIC_FILM,
    decay_length: float = DEFAULT_ANISOTROPY_DECAY_ANGSTROM,
    use_tensor_dielectric: bool = True,
) -> VitrifiedWaterSlab:
    """
    Cryo-EM slab: asymmetric top/bottom AWI.

    Default narrative: bottom AWI nearer pristine water–air; top may carry a
    sacrificial protein film (different c, m) as in passivation experiments.
    """
    common = dict(
        decay_length=decay_length,
        use_tensor_dielectric=use_tensor_dielectric,
    )
    bottom = build_interface(
        coverage=bottom_coverage,
        mechanical_state=bottom_mechanical,
        mechanical_age=0.0,
        normal=np.array([0.0, 0.0, 1.0]),
        phi0_surface_volts=0.03,
        **common,
    )
    top = build_interface(
        coverage=top_coverage,
        mechanical_state=top_mechanical,
        mechanical_age=0.6,
        normal=np.array([0.0, 0.0, -1.0]),
        phi0_surface_volts=0.02,
        surface_rho_enhancement=2.2,
        gamma=0.065,
        **common,
    )
    return VitrifiedWaterSlab(
        thickness_angstrom=thickness_angstrom,
        bottom=bottom,
        top=top,
    )


def calibrate_phi0_to_observable(
    interface: AWIInterface,
    target_value_volts: float,
    observable: PotentialObservable = PotentialObservable.ELECTROKINETIC_ZETA,
) -> AWIInterface:
    """
    Scale φ₀ so the value at z=0 matches ``target_value_volts`` for the named
    observable—without merging dielectric and charge into one σ_surf.
    """
    raw = interface.profile.sample(0.0)["phi_0"][0]
    scale = target_value_volts / raw if abs(raw) > 1e-12 else 1.0
    return AWIInterface(
        profile=interface.profile,
        gamma=interface.gamma,
        coverage=interface.coverage,
        mechanical_state=interface.mechanical_state,
        mechanical_age=interface.mechanical_age,
        normal=interface.normal.copy(),
        potential_observable=observable,
        phi0_calibration_scale=scale,
    )
