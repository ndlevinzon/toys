"""
Metropolis–Hastings orientation sampling on SO(3) for the hybrid AWI Hamiltonian.

Used by ``orientation_sample.py`` in ``mcmc`` and ``hybrid`` strategies. Uniform
draws explore the sphere; MCMC chains started from low-energy seeds concentrate
on the Boltzmann distribution at the chosen ``beta``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np

SampleSource = Literal["uniform", "mcmc", "anneal", "replica"]


@dataclass
class OrientationSample:
    """One evaluated rigid-body orientation."""

    rotation: np.ndarray
    energy: float
    result: Any
    source: SampleSource
    chain_id: int = -1
    accepted: bool = True
    beta_at_sample: float | None = None


@dataclass
class McmcChainStats:
    chain_id: int
    n_proposed: int
    n_accepted: int
    acceptance_rate: float
    final_step_deg: float
    energy_start: float
    energy_end: float


@dataclass
class McmcRunSummary:
    chains: list[McmcChainStats] = field(default_factory=list)
    total_proposed: int = 0
    total_accepted: int = 0
    mean_acceptance_rate: float = 0.0


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation in SO(3) via random unit quaternion."""
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    q1 = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    q2 = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    x, y, z, w = q1, q2, q3, q4
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def perturb_rotation(
    R: np.ndarray,
    rng: np.random.Generator,
    step_rad: float,
) -> np.ndarray:
    """Small random rotation applied on the right: R' = R @ Exp([axis] * angle)."""
    step_rad = float(max(step_rad, 1e-12))
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-30
    angle = float(rng.uniform(-step_rad, step_rad))
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    dR = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    return R @ dR


def wrap_energy_evaluator(
    evaluate: Callable[[np.ndarray], float | tuple[float, Any]],
) -> Callable[[np.ndarray], tuple[float, Any]]:
    """Adapt ``evaluator.energy`` (scalar) or ``(E, result)`` callables for MCMC."""
    def wrapped(R: np.ndarray) -> tuple[float, Any]:
        out = evaluate(R)
        if isinstance(out, tuple):
            return float(out[0]), out[1] if len(out) > 1 else None
        return float(out), None

    return wrapped


def metropolis_accept(
    energy_cur: float,
    energy_prop: float,
    beta: float,
    rng: np.random.Generator,
) -> bool:
    dE = float(energy_prop - energy_cur)
    if dE <= 0.0:
        return True
    if beta <= 0.0:
        return True
    if dE > 500.0 / max(beta, 1e-30):
        return False
    return float(rng.random()) < np.exp(-beta * dE)


def run_mcmc_chain(
    R0: np.ndarray,
    evaluate: Callable[[np.ndarray], tuple[float, Any]],
    rng: np.random.Generator,
    *,
    beta: float,
    n_steps: int,
    burn_in: int,
    thin: int,
    step_rad: float,
    chain_id: int = 0,
    adapt_interval: int = 50,
    target_acceptance: float = 0.28,
    step_min_rad: float = np.deg2rad(0.5),
    step_max_rad: float = np.deg2rad(45.0),
) -> tuple[list[OrientationSample], McmcChainStats]:
    """Run one Metropolis chain; return kept samples after burn-in / thinning."""
    evaluate_fn = wrap_energy_evaluator(evaluate)
    R = np.asarray(R0, dtype=np.float64).copy()
    energy, result = evaluate_fn(R)
    energy_start = float(energy)
    kept: list[OrientationSample] = []
    n_accept = 0
    step = float(step_rad)
    thin = max(int(thin), 1)
    burn_in = max(int(burn_in), 0)

    for step_idx in range(int(n_steps)):
        R_prop = perturb_rotation(R, rng, step)
        e_prop, res_prop = evaluate_fn(R_prop)
        acc = metropolis_accept(energy, e_prop, beta, rng)
        if acc:
            R, energy, result = R_prop, e_prop, res_prop
            n_accept += 1

        if adapt_interval > 0 and (step_idx + 1) % adapt_interval == 0:
            acc_rate = n_accept / float(step_idx + 1)
            if acc_rate > target_acceptance * 1.15:
                step = min(step * 1.12, step_max_rad)
            elif acc_rate < target_acceptance * 0.85:
                step = max(step / 1.12, step_min_rad)

        past_burn = step_idx >= burn_in
        is_thin = ((step_idx - burn_in) % thin) == 0 if past_burn else False
        if past_burn and is_thin:
            kept.append(
                OrientationSample(
                    rotation=R.copy(),
                    energy=float(energy),
                    result=result,
                    source="mcmc",
                    chain_id=chain_id,
                    accepted=True,
                )
            )

    stats = McmcChainStats(
        chain_id=chain_id,
        n_proposed=int(n_steps),
        n_accepted=int(n_accept),
        acceptance_rate=float(n_accept / max(n_steps, 1)),
        final_step_deg=float(np.degrees(step)),
        energy_start=energy_start,
        energy_end=float(energy),
    )
    return kept, stats


def run_uniform_batch(
    n: int,
    evaluate: Callable[[np.ndarray], tuple[float, Any]],
    rng: np.random.Generator,
) -> list[OrientationSample]:
    out: list[OrientationSample] = []
    for _ in range(int(n)):
        R = random_rotation_matrix(rng)
        e, res = evaluate(R)
        out.append(
            OrientationSample(
                rotation=R,
                energy=float(e),
                result=res,
                source="uniform",
            )
        )
    return out


def run_mcmc_from_seed_pool(
    seed_samples: list[OrientationSample],
    evaluate: Callable[[np.ndarray], tuple[float, Any]],
    rng: np.random.Generator,
    *,
    beta: float,
    n_chains: int,
    mcmc_steps_per_chain: int,
    mcmc_burn_in: int,
    mcmc_thin: int,
    mcmc_step_deg: float,
    mcmc_target_acceptance: float = 0.28,
) -> tuple[list[OrientationSample], McmcRunSummary]:
    """MCMC chains started from low-energy poses in ``seed_samples``."""
    if len(seed_samples) < 1:
        return [], McmcRunSummary()

    n_pool = len(seed_samples)
    order = np.argsort([s.energy for s in seed_samples])
    pick = np.linspace(0, n_pool - 1, max(1, n_chains)).astype(int)

    kept_all: list[OrientationSample] = []
    chain_stats: list[McmcChainStats] = []
    step_rad = float(np.deg2rad(mcmc_step_deg))

    for cid, idx in enumerate(pick[:n_chains]):
        R0 = seed_samples[int(order[int(idx)])].rotation
        kept, st = run_mcmc_chain(
            R0,
            evaluate,
            rng,
            beta=beta,
            n_steps=mcmc_steps_per_chain,
            burn_in=mcmc_burn_in,
            thin=mcmc_thin,
            step_rad=step_rad,
            chain_id=cid,
            target_acceptance=mcmc_target_acceptance,
        )
        kept_all.extend(kept)
        chain_stats.append(st)

    total_prop = sum(s.n_proposed for s in chain_stats)
    total_acc = sum(s.n_accepted for s in chain_stats)
    summary = McmcRunSummary(
        chains=chain_stats,
        total_proposed=total_prop,
        total_accepted=total_acc,
        mean_acceptance_rate=float(total_acc / max(total_prop, 1)),
    )
    return kept_all, summary


def run_hybrid_sampling(
    evaluate: Callable[[np.ndarray], tuple[float, Any]],
    rng: np.random.Generator,
    *,
    beta: float,
    n_uniform: int,
    n_chains: int,
    mcmc_steps_per_chain: int,
    mcmc_burn_in: int,
    mcmc_thin: int,
    mcmc_step_deg: float,
    mcmc_target_acceptance: float = 0.28,
    seed_pool_for_chains: int = 8,
) -> tuple[list[OrientationSample], McmcRunSummary, list[OrientationSample]]:
    """
    Uniform exploration, then independent MCMC chains from diverse low-energy seeds.

    Returns ``(all_samples, mcmc_summary, uniform_only)``.
    """
    uniform = run_uniform_batch(n_uniform, evaluate, rng)
    if n_chains <= 0 or mcmc_steps_per_chain <= 0:
        return uniform, McmcRunSummary(), uniform

    n_pool = min(max(seed_pool_for_chains, n_chains), len(uniform))
    order = np.argsort([s.energy for s in uniform])[:n_pool]
    # Spread chain starts across the low-energy pool (not all identical minima).
    pick = np.linspace(0, n_pool - 1, n_chains).astype(int)

    all_samples = list(uniform)
    chain_stats: list[McmcChainStats] = []
    step_rad = float(np.deg2rad(mcmc_step_deg))

    for cid, idx in enumerate(pick):
        R0 = uniform[int(order[int(idx)])].rotation
        kept, st = run_mcmc_chain(
            R0,
            evaluate,
            rng,
            beta=beta,
            n_steps=mcmc_steps_per_chain,
            burn_in=mcmc_burn_in,
            thin=mcmc_thin,
            step_rad=step_rad,
            chain_id=cid,
            target_acceptance=mcmc_target_acceptance,
        )
        all_samples.extend(kept)
        chain_stats.append(st)

    total_prop = sum(s.n_proposed for s in chain_stats)
    total_acc = sum(s.n_accepted for s in chain_stats)
    summary = McmcRunSummary(
        chains=chain_stats,
        total_proposed=total_prop,
        total_accepted=total_acc,
        mean_acceptance_rate=float(total_acc / max(total_prop, 1)),
    )
    return all_samples, summary, uniform


def run_mcmc_only(
    evaluate: Callable[[np.ndarray], tuple[float, Any]],
    rng: np.random.Generator,
    *,
    beta: float,
    n_chains: int,
    mcmc_steps_per_chain: int,
    mcmc_burn_in: int,
    mcmc_thin: int,
    mcmc_step_deg: float,
    mcmc_target_acceptance: float = 0.28,
) -> tuple[list[OrientationSample], McmcRunSummary]:
    """MCMC without an initial uniform pool (random chain starts)."""
    all_samples: list[OrientationSample] = []
    chain_stats: list[McmcChainStats] = []
    step_rad = float(np.deg2rad(mcmc_step_deg))
    for cid in range(int(n_chains)):
        R0 = random_rotation_matrix(rng)
        kept, st = run_mcmc_chain(
            R0,
            evaluate,
            rng,
            beta=beta,
            n_steps=mcmc_steps_per_chain,
            burn_in=mcmc_burn_in,
            thin=mcmc_thin,
            step_rad=step_rad,
            chain_id=cid,
            target_acceptance=mcmc_target_acceptance,
        )
        all_samples.extend(kept)
        chain_stats.append(st)
    total_prop = sum(s.n_proposed for s in chain_stats)
    total_acc = sum(s.n_accepted for s in chain_stats)
    summary = McmcRunSummary(
        chains=chain_stats,
        total_proposed=total_prop,
        total_accepted=total_acc,
        mean_acceptance_rate=float(total_acc / max(total_prop, 1)),
    )
    return all_samples, summary


def samples_to_arrays(
    samples: list[OrientationSample],
) -> tuple[list[np.ndarray], np.ndarray, list, list[str], np.ndarray]:
    """Extract rotations, energies, results, sources, beta_at_sample (NaN if unset)."""
    Rs = [s.rotation for s in samples]
    energies = np.array([s.energy for s in samples], dtype=np.float64)
    results = [s.result for s in samples]
    sources = [s.source for s in samples]
    betas = np.array(
        [np.nan if s.beta_at_sample is None else float(s.beta_at_sample) for s in samples],
        dtype=np.float64,
    )
    return Rs, energies, results, sources, betas
