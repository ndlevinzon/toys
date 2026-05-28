"""
Load and apply Ising / hybrid Hamiltonian parameters from ``ising_params.yaml``.

Default file location: ``<anisotropy_project>/ising_params.yaml`` (next to
``orientation_sample.py``). Override with ``--ising-params`` on the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from anisotropy.awi_field import (
    InterfaceMechanicalState,
    SurfaceCoverage,
    VitrifiedWaterSlab,
    build_interface,
)
from anisotropy.lattice_solvent_hamiltonian import HybridHamiltonianCouplings

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyYAML is required to load ising_params.yaml. Install with: pip install pyyaml"
    ) from exc

OccupancyMode = Literal["binary", "ternary"]

_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _PACKAGE_DIR.parent
DEFAULT_ISING_PARAMS_PATH = _PROJECT_DIR / "ising_params.yaml"

_COVERAGE_MAP: dict[str, SurfaceCoverage] = {
    "pristine": SurfaceCoverage.PRISTINE,
    "surfactant": SurfaceCoverage.SURFACTANT,
    "protein_sacrificial": SurfaceCoverage.PROTEIN_SACRIFICIAL,
    "passivated": SurfaceCoverage.PASSIVATED,
    "contaminated": SurfaceCoverage.CONTAMINATED,
}

_MECHANICAL_MAP: dict[str, InterfaceMechanicalState] = {
    "fresh": InterfaceMechanicalState.FRESH,
    "equilibrated": InterfaceMechanicalState.EQUILIBRATED,
    "viscoelastic_film": InterfaceMechanicalState.VISCOELASTIC_FILM,
    "aged_protein_layer": InterfaceMechanicalState.AGED_PROTEIN_LAYER,
}


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    block = data.get(key)
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise TypeError(f"ising_params.yaml: '{key}' must be a mapping")
    return block


def _f(block: dict[str, Any], key: str, default: float) -> float:
    if key not in block or block[key] is None:
        return float(default)
    return float(block[key])


def _i(block: dict[str, Any], key: str, default: int) -> int:
    if key not in block or block[key] is None:
        return int(default)
    return int(block[key])


def _b(block: dict[str, Any], key: str, default: bool) -> bool:
    if key not in block or block[key] is None:
        return bool(default)
    return bool(block[key])


def _opt_f(block: dict[str, Any], key: str) -> float | None:
    if key not in block or block[key] is None:
        return None
    return float(block[key])


def _enum_coverage(name: str) -> SurfaceCoverage:
    key = name.strip().lower()
    if key not in _COVERAGE_MAP:
        raise ValueError(f"Unknown surface coverage '{name}' in ising_params.yaml")
    return _COVERAGE_MAP[key]


def _enum_mechanical(name: str) -> InterfaceMechanicalState:
    key = name.strip().lower()
    if key not in _MECHANICAL_MAP:
        raise ValueError(f"Unknown mechanical_state '{name}' in ising_params.yaml")
    return _MECHANICAL_MAP[key]


@dataclass(frozen=True)
class SolvParams:
    """Ising cohesion J, chemical potential mu, slab confinement scale."""

    J: float = 1.0
    mu: float = 0.0
    u_film_scale: float = 0.25
    occupancy_mode: OccupancyMode = "binary"


@dataclass(frozen=True)
class ConfinementParams:
    """U_film(z) penalty outside the vitrified slab band."""

    penalty_outside: float = 1.0
    interface_softness_angstrom: float = 4.0


@dataclass(frozen=True)
class HydrationCouplingParams:
    lambda_hp: float = 0.8
    lambda_p: float = 0.25
    lambda_hb: float = 0.25


@dataclass(frozen=True)
class RayIndicatorParams:
    ray_max_steps: int = 48
    ray_step_fraction: float = 0.55
    require_first_hit_solvent_steps: int = 2
    outward_ray_step_fraction: float = 0.5

    def ray_step_angstrom(self, grid_spacing: float) -> float:
        return float(self.ray_step_fraction) * float(grid_spacing)


@dataclass(frozen=True)
class FilmCouplingParams:
    lambda_film: float = 0.3
    margin_angstrom: float = 7.0


@dataclass(frozen=True)
class FlexParams:
    eta_flex: float = 0.0


@dataclass(frozen=True)
class ElectrostaticParams:
    homogeneous_epsilon: float | None = None
    homogeneous_kappa: float | None = None
    r_smooth_angstrom: float = 0.01
    use_intrinsic_potential: bool = True
    use_dipole_E0_component: bool = True
    coulomb_scale: float | None = None


@dataclass(frozen=True)
class AwiInterfaceParams:
    coverage: str = "pristine"
    mechanical_state: str = "fresh"
    mechanical_age: float = 0.0
    gamma_N_per_m: float = 0.0728
    phi0_surface_volts: float = 0.05
    surface_rho_enhancement: float | None = None


@dataclass(frozen=True)
class AwiProfileParams:
    extent_angstrom: float = 40.0
    dz_angstrom: float = 0.25
    decay_length_angstrom: float = 7.0
    bulk_epsilon: float = 78.0
    surface_epsilon_parallel: float = 2.5
    surface_epsilon_perpendicular: float = 1.5
    use_tensor_dielectric: bool = True
    bulk_rho: float = 1.0
    surface_rho_enhancement: float = 1.8
    phi0_surface_volts: float = 0.05
    phi0_decay_length_angstrom: float | None = None
    debye_length_angstrom: float | None = None
    transition_width_angstrom: float = 2.0
    transition_center_angstrom: float = 1.5
    layer_scale_angstrom: float = 3.0


@dataclass(frozen=True)
class SlabParams:
    thickness_angstrom: float = 300.0
    bulk_epsilon: float = 78.0
    bulk_rho: float = 1.0
    bulk_kappa: float = 0.0
    bottom: AwiInterfaceParams = field(default_factory=AwiInterfaceParams)
    top: AwiInterfaceParams = field(
        default_factory=lambda: AwiInterfaceParams(
            coverage="protein_sacrificial",
            mechanical_state="viscoelastic_film",
            mechanical_age=0.6,
            gamma_N_per_m=0.065,
            phi0_surface_volts=0.02,
            surface_rho_enhancement=2.2,
        )
    )


@dataclass(frozen=True)
class LatticeGeometryParams:
    grid_spacing_angstrom: float = 3.5
    pad_xy_angstrom: float = 60.0
    pad_z_angstrom: float = 25.0
    protein_z_center_angstrom: float | None = None
    z_center_decay_fraction: float = 0.8
    canonical_interior_pad_angstrom: float = 16.0


@dataclass(frozen=True)
class PerformanceParams:
    """Numerical shortcuts for orientation / MCMC sweeps."""

    use_fast_evaluator: bool = True
    electrostatic_pair_cutoff_angstrom: float | None = 80.0
    mcmc_energy_only: bool = True
    precompute_rotation_invariant_pair: bool = True
    parallel_workers: int | None = None
    parallel_mcmc_chains: bool = True
    batch_size: int = 16


@dataclass(frozen=True)
class AnnealingParams:
    """Simulated annealing: β_k ramps from β_min to β_max (see orientation_multimodal)."""

    beta_min_fraction: float = 0.05
    schedule: str = "geometric"
    n_reheat_cycles: int = 1


@dataclass(frozen=True)
class ReplicaExchangeParams:
    """Parallel tempering: fixed β ladder + Metropolis swaps."""

    n_replicas: int = 8
    swap_interval: int = 1


@dataclass(frozen=True)
class McmcSamplingParams:
    # fixed_beta | simulated_annealing | replica_exchange
    mode: str = "fixed_beta"
    n_chains: int = 4
    steps_per_chain: int = 2000
    burn_in: int = 300
    thin: int = 2
    step_deg: float = 10.0
    target_acceptance: float = 0.28
    seed_pool: int = 12
    annealing: AnnealingParams = field(default_factory=AnnealingParams)
    replica_exchange: ReplicaExchangeParams = field(default_factory=ReplicaExchangeParams)


@dataclass(frozen=True)
class SamplingParams:
    n_samples: int = 600
    seed: int = 0
    beta: float = 1.0
    beta_auto: bool = False
    beta_target_ess: float = 20.0
    beta_auto_method: str = "ess"  # ess | top10 | rank2
    # uniform | mcmc | hybrid (uniform pool + MCMC refinement)
    strategy: str = "hybrid"
    n_uniform: int = 400
    mcmc: McmcSamplingParams = field(default_factory=McmcSamplingParams)


@dataclass(frozen=True)
class IsingParams:
    """Full parameter set for the hybrid lattice-gas Hamiltonian."""

    source_path: str
    solv: SolvParams = field(default_factory=SolvParams)
    confinement: ConfinementParams = field(default_factory=ConfinementParams)
    hydration: HydrationCouplingParams = field(default_factory=HydrationCouplingParams)
    ray: RayIndicatorParams = field(default_factory=RayIndicatorParams)
    film: FilmCouplingParams = field(default_factory=FilmCouplingParams)
    flex: FlexParams = field(default_factory=FlexParams)
    electrostatics: ElectrostaticParams = field(default_factory=ElectrostaticParams)
    slab: SlabParams = field(default_factory=SlabParams)
    awi_profile: AwiProfileParams = field(default_factory=AwiProfileParams)
    lattice: LatticeGeometryParams = field(default_factory=LatticeGeometryParams)
    sampling: SamplingParams = field(default_factory=SamplingParams)
    performance: PerformanceParams = field(default_factory=PerformanceParams)

    def to_hybrid_couplings(self) -> HybridHamiltonianCouplings:
        return HybridHamiltonianCouplings(
            J_solv=self.solv.J,
            mu_chemical=self.solv.mu,
            u_film_scale=self.solv.u_film_scale,
            lambda_hp=self.hydration.lambda_hp,
            lambda_p=self.hydration.lambda_p,
            lambda_hb=self.hydration.lambda_hb,
            lambda_film=self.film.lambda_film,
            eta_flex=self.flex.eta_flex,
            homogeneous_epsilon=self.electrostatics.homogeneous_epsilon,
            homogeneous_kappa=self.electrostatics.homogeneous_kappa,
        )

    def protein_z_center(self) -> float | None:
        return self.lattice.protein_z_center_angstrom

    def default_protein_z_center(self) -> float:
        """COM z when protein_z_center_angstrom is null (near top AWI)."""
        frac = self.lattice.z_center_decay_fraction
        return float(self.slab.thickness_angstrom) - frac * self.awi_profile.decay_length_angstrom

    def build_slab(self) -> VitrifiedWaterSlab:
        return build_slab_from_ising_params(self)

    def as_dict(self) -> dict[str, Any]:
        """Nested dict for JSON export (e.g. orientation_sample output)."""
        from dataclasses import asdict

        return asdict(self)


def _parse_interface(block: dict[str, Any], defaults: AwiInterfaceParams) -> AwiInterfaceParams:
    return AwiInterfaceParams(
        coverage=str(block.get("coverage", defaults.coverage)),
        mechanical_state=str(block.get("mechanical_state", defaults.mechanical_state)),
        mechanical_age=_f(block, "mechanical_age", defaults.mechanical_age),
        gamma_N_per_m=_f(block, "gamma_N_per_m", defaults.gamma_N_per_m),
        phi0_surface_volts=_f(block, "phi0_surface_volts", defaults.phi0_surface_volts),
        surface_rho_enhancement=(
            float(block["surface_rho_enhancement"])
            if block.get("surface_rho_enhancement") is not None
            else defaults.surface_rho_enhancement
        ),
    )


def _load_performance_params(block: dict[str, Any] | None) -> PerformanceParams:
    b = block or {}
    cutoff = b.get("electrostatic_pair_cutoff_angstrom", 80.0)
    if cutoff is None or (isinstance(cutoff, str) and cutoff.lower() in ("none", "null")):
        cutoff_val = None
    else:
        cutoff_val = float(cutoff)
    pw = b.get("parallel_workers", None)
    if pw is None or (isinstance(pw, str) and pw.lower() in ("null", "none", "auto")):
        parallel_workers = None
    else:
        parallel_workers = int(pw)
    return PerformanceParams(
        use_fast_evaluator=bool(b.get("use_fast_evaluator", True)),
        electrostatic_pair_cutoff_angstrom=cutoff_val,
        mcmc_energy_only=bool(b.get("mcmc_energy_only", True)),
        precompute_rotation_invariant_pair=bool(
            b.get("precompute_rotation_invariant_pair", True)
        ),
        parallel_workers=parallel_workers,
        parallel_mcmc_chains=bool(b.get("parallel_mcmc_chains", True)),
        batch_size=_i(b, "batch_size", 16),
    )


def _load_annealing_params(block: dict[str, Any] | None) -> AnnealingParams:
    b = block or {}
    return AnnealingParams(
        beta_min_fraction=_f(b, "beta_min_fraction", 0.05),
        schedule=str(b.get("schedule", "geometric")),
        n_reheat_cycles=_i(b, "n_reheat_cycles", 1),
    )


def _load_replica_exchange_params(block: dict[str, Any] | None) -> ReplicaExchangeParams:
    b = block or {}
    return ReplicaExchangeParams(
        n_replicas=_i(b, "n_replicas", 8),
        swap_interval=_i(b, "swap_interval", 1),
    )


def _load_mcmc_params(block: dict[str, Any] | None) -> McmcSamplingParams:
    b = block or {}
    return McmcSamplingParams(
        mode=str(b.get("mode", "fixed_beta")).lower(),
        n_chains=_i(b, "n_chains", 4),
        steps_per_chain=_i(b, "steps_per_chain", 2000),
        burn_in=_i(b, "burn_in", 300),
        thin=_i(b, "thin", 2),
        step_deg=_f(b, "step_deg", 10.0),
        target_acceptance=_f(b, "target_acceptance", 0.28),
        seed_pool=_i(b, "seed_pool", 12),
        annealing=_load_annealing_params(b.get("annealing")),
        replica_exchange=_load_replica_exchange_params(b.get("replica_exchange")),
    )


def _load_sampling_params(samp_b: dict[str, Any]) -> SamplingParams:
    """Parse ``sampling`` block; ``beta: auto`` enables post-hoc ESS calibration."""
    beta_raw = samp_b.get("beta", 1.0)
    beta_auto = bool(samp_b.get("beta_auto", False))
    if isinstance(beta_raw, str) and beta_raw.strip().lower() == "auto":
        beta_auto = True
        beta = 1.0
    else:
        beta = float(beta_raw)
    strategy = str(samp_b.get("strategy", "hybrid")).lower()
    return SamplingParams(
        n_samples=_i(samp_b, "n_samples", 600),
        seed=_i(samp_b, "seed", 0),
        beta=beta,
        beta_auto=beta_auto,
        beta_target_ess=_f(samp_b, "beta_target_ess", 20.0),
        beta_auto_method=str(samp_b.get("beta_auto_method", "ess")),
        strategy=strategy,
        n_uniform=_i(samp_b, "n_uniform", 400),
        mcmc=_load_mcmc_params(samp_b.get("mcmc")),
    )


def load_ising_params(path: str | Path | None = None) -> IsingParams:
    """
    Load ``ising_params.yaml``.

    Parameters
    ----------
    path
        YAML file path. Default: ``<project>/ising_params.yaml``.
    """
    yaml_path = Path(path) if path is not None else DEFAULT_ISING_PARAMS_PATH
    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"Ising parameter file not found: {yaml_path}\n"
            f"Expected default at {DEFAULT_ISING_PARAMS_PATH}"
        )

    with yaml_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{yaml_path}: root must be a YAML mapping")

    solv_b = _section(raw, "solv")
    occ = str(solv_b.get("occupancy_mode", "binary")).strip().lower()
    if occ not in ("binary", "ternary"):
        raise ValueError(f"occupancy_mode must be 'binary' or 'ternary', got {occ!r}")

    conf_b = _section(raw, "confinement")
    hyd_b = _section(raw, "hydration")
    ray_b = _section(raw, "ray_indicators")
    film_b = _section(raw, "film")
    flex_b = _section(raw, "flex")
    el_b = _section(raw, "electrostatics")
    slab_b = _section(raw, "slab")
    prof_b = _section(raw, "awi_profile")
    lat_b = _section(raw, "lattice")
    samp_b = _section(raw, "sampling")

    default_bottom = AwiInterfaceParams()
    default_top = SlabParams().top

    return IsingParams(
        source_path=str(yaml_path.resolve()),
        solv=SolvParams(
            J=_f(solv_b, "J", 1.0),
            mu=_f(solv_b, "mu", 0.0),
            u_film_scale=_f(solv_b, "u_film_scale", 0.25),
            occupancy_mode=occ,  # type: ignore[arg-type]
        ),
        confinement=ConfinementParams(
            penalty_outside=_f(conf_b, "penalty_outside", 1.0),
            interface_softness_angstrom=_f(conf_b, "interface_softness_angstrom", 4.0),
        ),
        hydration=HydrationCouplingParams(
            lambda_hp=_f(hyd_b, "lambda_hp", 0.8),
            lambda_p=_f(hyd_b, "lambda_p", 0.25),
            lambda_hb=_f(hyd_b, "lambda_hb", 0.25),
        ),
        ray=RayIndicatorParams(
            ray_max_steps=_i(ray_b, "ray_max_steps", 48),
            ray_step_fraction=_f(ray_b, "ray_step_fraction", 0.55),
            require_first_hit_solvent_steps=_i(ray_b, "require_first_hit_solvent_steps", 2),
            outward_ray_step_fraction=_f(ray_b, "outward_ray_step_fraction", 0.5),
        ),
        film=FilmCouplingParams(
            lambda_film=_f(film_b, "lambda_film", 0.3),
            margin_angstrom=_f(film_b, "margin_angstrom", 7.0),
        ),
        flex=FlexParams(eta_flex=_f(flex_b, "eta_flex", 0.0)),
        electrostatics=ElectrostaticParams(
            homogeneous_epsilon=_opt_f(el_b, "homogeneous_epsilon"),
            homogeneous_kappa=_opt_f(el_b, "homogeneous_kappa"),
            r_smooth_angstrom=_f(el_b, "r_smooth_angstrom", 0.01),
            use_intrinsic_potential=_b(el_b, "use_intrinsic_potential", True),
            use_dipole_E0_component=_b(el_b, "use_dipole_E0_component", True),
            coulomb_scale=_opt_f(el_b, "coulomb_scale"),
        ),
        slab=SlabParams(
            thickness_angstrom=_f(slab_b, "thickness_angstrom", 300.0),
            bulk_epsilon=_f(slab_b, "bulk_epsilon", 78.0),
            bulk_rho=_f(slab_b, "bulk_rho", 1.0),
            bulk_kappa=_f(slab_b, "bulk_kappa", 0.0),
            bottom=_parse_interface(_section(slab_b, "bottom"), default_bottom),
            top=_parse_interface(_section(slab_b, "top"), default_top),
        ),
        awi_profile=AwiProfileParams(
            extent_angstrom=_f(prof_b, "extent_angstrom", 40.0),
            dz_angstrom=_f(prof_b, "dz_angstrom", 0.25),
            decay_length_angstrom=_f(prof_b, "decay_length_angstrom", 7.0),
            bulk_epsilon=_f(prof_b, "bulk_epsilon", 78.0),
            surface_epsilon_parallel=_f(prof_b, "surface_epsilon_parallel", 2.5),
            surface_epsilon_perpendicular=_f(prof_b, "surface_epsilon_perpendicular", 1.5),
            use_tensor_dielectric=_b(prof_b, "use_tensor_dielectric", True),
            bulk_rho=_f(prof_b, "bulk_rho", 1.0),
            surface_rho_enhancement=_f(prof_b, "surface_rho_enhancement", 1.8),
            phi0_surface_volts=_f(prof_b, "phi0_surface_volts", 0.05),
            phi0_decay_length_angstrom=_opt_f(prof_b, "phi0_decay_length_angstrom"),
            debye_length_angstrom=_opt_f(prof_b, "debye_length_angstrom"),
            transition_width_angstrom=_f(prof_b, "transition_width_angstrom", 2.0),
            transition_center_angstrom=_f(prof_b, "transition_center_angstrom", 1.5),
            layer_scale_angstrom=_f(prof_b, "layer_scale_angstrom", 3.0),
        ),
        lattice=LatticeGeometryParams(
            grid_spacing_angstrom=_f(lat_b, "grid_spacing_angstrom", 3.5),
            pad_xy_angstrom=_f(lat_b, "pad_xy_angstrom", 60.0),
            pad_z_angstrom=_f(lat_b, "pad_z_angstrom", 25.0),
            protein_z_center_angstrom=_opt_f(lat_b, "protein_z_center_angstrom"),
            z_center_decay_fraction=_f(lat_b, "z_center_decay_fraction", 0.8),
            canonical_interior_pad_angstrom=_f(lat_b, "canonical_interior_pad_angstrom", 16.0),
        ),
        sampling=_load_sampling_params(samp_b),
        performance=_load_performance_params(_section(raw, "performance")),
    )


def build_slab_from_ising_params(params: IsingParams) -> VitrifiedWaterSlab:
    """Build :class:`~anisotropy.awi_field.VitrifiedWaterSlab` from YAML slab + profile sections."""
    p = params.awi_profile
    profile_kwargs = dict(
        extent=p.extent_angstrom,
        dz=p.dz_angstrom,
        decay_length=p.decay_length_angstrom,
        bulk_epsilon=p.bulk_epsilon,
        surface_epsilon_parallel=p.surface_epsilon_parallel,
        surface_epsilon_perpendicular=p.surface_epsilon_perpendicular,
        use_tensor_dielectric=p.use_tensor_dielectric,
        bulk_rho=p.bulk_rho,
        surface_rho_enhancement=p.surface_rho_enhancement,
        phi0_surface_volts=p.phi0_surface_volts,
        phi0_decay_length=p.phi0_decay_length_angstrom,
        debye_length_angstrom=p.debye_length_angstrom,
        transition_width=p.transition_width_angstrom,
        transition_center=p.transition_center_angstrom,
    )

    def _iface(side: AwiInterfaceParams, normal: np.ndarray) -> Any:
        kw = dict(profile_kwargs)
        kw["phi0_surface_volts"] = side.phi0_surface_volts
        if side.surface_rho_enhancement is not None:
            kw["surface_rho_enhancement"] = side.surface_rho_enhancement
        return build_interface(
            gamma=side.gamma_N_per_m,
            coverage=_enum_coverage(side.coverage),
            mechanical_state=_enum_mechanical(side.mechanical_state),
            mechanical_age=side.mechanical_age,
            normal=normal,
            **kw,
        )

    bottom = _iface(params.slab.bottom, np.array([0.0, 0.0, 1.0]))
    top = _iface(params.slab.top, np.array([0.0, 0.0, -1.0]))
    return VitrifiedWaterSlab(
        thickness_angstrom=params.slab.thickness_angstrom,
        bottom=bottom,
        top=top,
        bulk_epsilon=params.slab.bulk_epsilon,
        bulk_rho=params.slab.bulk_rho,
        bulk_kappa=params.slab.bulk_kappa,
    )


def apply_cli_overrides(params: IsingParams, args: Any) -> IsingParams:
    """
    Return a copy of ``params`` with non-None CLI overrides from ``orientation_sample`` args.
    """
    from dataclasses import replace

    solv = params.solv
    hydration = params.hydration
    film = params.film
    flex = params.flex
    el = params.electrostatics
    slab = params.slab
    lattice = params.lattice
    sampling = params.sampling

    if getattr(args, "J", None) is not None:
        solv = replace(solv, J=float(args.J))
    if getattr(args, "mu", None) is not None:
        solv = replace(solv, mu=float(args.mu))
    if getattr(args, "u_film", None) is not None:
        solv = replace(solv, u_film_scale=float(args.u_film))
    if getattr(args, "lambda_hp", None) is not None:
        hydration = replace(hydration, lambda_hp=float(args.lambda_hp))
    if getattr(args, "lambda_p", None) is not None:
        hydration = replace(hydration, lambda_p=float(args.lambda_p))
    if getattr(args, "lambda_hb", None) is not None:
        hydration = replace(hydration, lambda_hb=float(args.lambda_hb))
    if getattr(args, "lambda_film", None) is not None:
        film = replace(film, lambda_film=float(args.lambda_film))
    if getattr(args, "eta_flex", None) is not None:
        flex = replace(flex, eta_flex=float(args.eta_flex))
    if getattr(args, "homogeneous_epsilon", None) is not None:
        el = replace(el, homogeneous_epsilon=float(args.homogeneous_epsilon))
    if getattr(args, "homogeneous_kappa", None) is not None:
        el = replace(el, homogeneous_kappa=float(args.homogeneous_kappa))
    if getattr(args, "slab_thickness", None) is not None:
        slab = replace(slab, thickness_angstrom=float(args.slab_thickness))
    if getattr(args, "grid_spacing", None) is not None:
        lattice = replace(lattice, grid_spacing_angstrom=float(args.grid_spacing))
    if getattr(args, "pad_xy", None) is not None:
        lattice = replace(lattice, pad_xy_angstrom=float(args.pad_xy))
    if getattr(args, "pad_z", None) is not None:
        lattice = replace(lattice, pad_z_angstrom=float(args.pad_z))
    if getattr(args, "z_center", None) is not None:
        lattice = replace(lattice, protein_z_center_angstrom=float(args.z_center))
    if getattr(args, "canonical_pad", None) is not None:
        lattice = replace(
            lattice, canonical_interior_pad_angstrom=float(args.canonical_pad)
        )
    if getattr(args, "n_samples", None) is not None:
        sampling = replace(sampling, n_samples=int(args.n_samples))
    if getattr(args, "seed", None) is not None:
        sampling = replace(sampling, seed=int(args.seed))
    if getattr(args, "beta", None) is not None:
        beta_arg = args.beta
        if isinstance(beta_arg, str) and beta_arg.strip().lower() == "auto":
            sampling = replace(sampling, beta_auto=True)
        else:
            sampling = replace(
                sampling,
                beta=float(beta_arg),
                beta_auto=False,
            )
    if getattr(args, "beta_target_ess", None) is not None:
        sampling = replace(
            sampling, beta_target_ess=float(args.beta_target_ess)
        )
    if getattr(args, "beta_auto_method", None) is not None:
        sampling = replace(
            sampling, beta_auto_method=str(args.beta_auto_method)
        )
    if getattr(args, "sampling_strategy", None) is not None:
        sampling = replace(sampling, strategy=str(args.sampling_strategy).lower())
    if getattr(args, "n_uniform", None) is not None:
        sampling = replace(sampling, n_uniform=int(args.n_uniform))
    mcmc = sampling.mcmc
    if getattr(args, "mcmc_chains", None) is not None:
        mcmc = replace(mcmc, n_chains=int(args.mcmc_chains))
    if getattr(args, "mcmc_steps", None) is not None:
        mcmc = replace(mcmc, steps_per_chain=int(args.mcmc_steps))
    if getattr(args, "mcmc_burn_in", None) is not None:
        mcmc = replace(mcmc, burn_in=int(args.mcmc_burn_in))
    if getattr(args, "mcmc_thin", None) is not None:
        mcmc = replace(mcmc, thin=int(args.mcmc_thin))
    if getattr(args, "mcmc_step_deg", None) is not None:
        mcmc = replace(mcmc, step_deg=float(args.mcmc_step_deg))
    sampling = replace(sampling, mcmc=mcmc)

    perf = params.performance
    if getattr(args, "parallel_workers", None) is not None:
        perf = replace(perf, parallel_workers=int(args.parallel_workers))

    return replace(
        params,
        solv=solv,
        hydration=hydration,
        film=film,
        flex=flex,
        electrostatics=el,
        slab=slab,
        lattice=lattice,
        sampling=sampling,
        performance=perf,
    )
