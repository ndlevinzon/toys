"""
Multimodal orientation sampling on SO(3) for the hybrid AWI Hamiltonian.

Two statistically principled alternatives to ad-hoc "escape potentials":

1. **Simulated annealing (SA)** — inhomogeneous Markov chain with inverse temperature
   β_k increasing from β_min to β_max. At step k the chain targets
   π_k(Ω) ∝ exp(-β_k H(Ω)). Slow cooling explores at high T then settles; kept
   states are reweighted to the target β using importance sampling.

2. **Replica exchange (parallel tempering)** — M replicas at fixed {β_m}, each a
   Metropolis chain; occasional swaps between adjacent temperatures satisfy detailed
   balance and allow trajectories to visit many basins while maintaining equilibrium
   in the extended ensemble. Samples at β_m are reweighted to β_target.

Both methods use the same Metropolis moves on SO(3) as ``orientation_mcmc``; only
the temperature protocol differs. No non-physical bias potentials are added to H.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from anisotropy.orientation_mcmc import (
    McmcChainStats,
    McmcRunSummary,
    OrientationSample,
    metropolis_accept,
    perturb_rotation,
    random_rotation_matrix,
    wrap_energy_evaluator,
)


def importance_weights_to_beta(
    energies: np.ndarray,
    beta_at_sample: np.ndarray,
    beta_target: float,
) -> np.ndarray:
    """
    Reweight samples drawn at β_i to expectations under β_target.

    w_i ∝ exp(-β_target H_i) / exp(-β_i H_i) = exp(-(β_target - β_i) H_i).
    """
    E = np.asarray(energies, dtype=np.float64).reshape(-1)
    bi = np.asarray(beta_at_sample, dtype=np.float64).reshape(-1)
    bt = float(beta_target)
    logw = -(bt - bi) * (E - float(E.min()))
    logw -= float(np.max(logw))
    w = np.exp(logw)
    return w / (float(w.sum()) + 1e-30)


def beta_schedule(
    n_steps: int,
    beta_min: float,
    beta_max: float,
    schedule: str = "geometric",
) -> np.ndarray:
    """
    Inverse-temperature schedule β_k for k = 0 … n_steps-1.

    ``geometric`` — β_k = β_min (β_max/β_min)^{k/(K-1)} (standard SA cooling).
    ``linear`` — uniform ramp in β.
    """
    n = max(int(n_steps), 1)
    b0 = float(max(beta_min, 1e-30))
    b1 = float(max(beta_max, b0))
    k = np.arange(n, dtype=np.float64)
    if n <= 1:
        return np.array([b1], dtype=np.float64)
    sched = str(schedule).lower()
    if sched == "linear":
        return b0 + (b1 - b0) * (k / (n - 1))
    # geometric default
    return b0 * (b1 / b0) ** (k / (n - 1))


def replica_beta_ladder(
    n_replicas: int,
    beta_min: float,
    beta_max: float,
) -> np.ndarray:
    """Geometric ladder β_0 < … < β_{M-1} for replica exchange."""
    m = max(int(n_replicas), 2)
    return beta_schedule(m, beta_min, beta_max, schedule="geometric")


@dataclass
class AnnealingStats:
    chain_id: int
    n_steps: int
    n_accepted: int
    beta_min: float
    beta_max: float
    energy_start: float
    energy_end: float
    n_reheat_cycles: int


@dataclass
class ReplicaExchangeStats:
    n_replicas: int
    n_steps: int
    n_swap_attempts: int
    n_swap_accepted: int
    betas: np.ndarray
    mean_swap_rate: float


def run_simulated_annealing_chain(
    R0: np.ndarray,
    evaluate: Callable[[np.ndarray], float | tuple[float, Any]],
    rng: np.random.Generator,
    *,
    beta_min: float,
    beta_max: float,
    n_steps: int,
    burn_in: int,
    thin: int,
    step_rad: float,
    schedule: str = "geometric",
    n_reheat_cycles: int = 1,
    chain_id: int = 0,
    adapt_interval: int = 50,
    target_acceptance: float = 0.28,
) -> tuple[list[OrientationSample], AnnealingStats]:
    """
    Simulated annealing Metropolis on SO(3).

    Each cycle cools β from β_min to β_max. Optional ``n_reheat_cycles > 1`` resets
    β to β_min between cycles (reheating) to escape successive minima while
    retaining the same Markov state (configuration memory).
    """
    evaluate_fn = wrap_energy_evaluator(evaluate)
    R = np.asarray(R0, dtype=np.float64).copy()
    energy, result = evaluate_fn(R)
    energy_start = float(energy)
    n_accept = 0
    step = float(step_rad)
    thin = max(int(thin), 1)
    burn_in = max(int(burn_in), 0)
    n_cycles = max(int(n_reheat_cycles), 1)
    steps_per_cycle = max(int(n_steps) // n_cycles, 1)
    kept: list[OrientationSample] = []
    global_step = 0

    for cycle in range(n_cycles):
        betas = beta_schedule(steps_per_cycle, beta_min, beta_max, schedule=schedule)
        for local_idx, beta_k in enumerate(betas):
            R_prop = perturb_rotation(R, rng, step)
            e_prop, res_prop = evaluate_fn(R_prop)
            if metropolis_accept(energy, e_prop, float(beta_k), rng):
                R, energy, result = R_prop, e_prop, res_prop
                n_accept += 1

            if adapt_interval > 0 and (local_idx + 1) % adapt_interval == 0:
                acc_rate = n_accept / float(global_step + local_idx + 1)
                if acc_rate > target_acceptance * 1.15:
                    step = min(step * 1.12, np.deg2rad(45.0))
                elif acc_rate < target_acceptance * 0.85:
                    step = max(step / 1.12, np.deg2rad(0.5))

            global_step += 1
            past_burn = global_step > burn_in
            is_thin = ((global_step - burn_in) % thin) == 0 if past_burn else False
            if past_burn and is_thin:
                kept.append(
                    OrientationSample(
                        rotation=R.copy(),
                        energy=float(energy),
                        result=result,
                        source="anneal",
                        chain_id=chain_id,
                        accepted=True,
                        beta_at_sample=float(beta_k),
                    )
                )

    stats = AnnealingStats(
        chain_id=chain_id,
        n_steps=int(n_steps),
        n_accepted=int(n_accept),
        beta_min=float(beta_min),
        beta_max=float(beta_max),
        energy_start=energy_start,
        energy_end=float(energy),
        n_reheat_cycles=n_cycles,
    )
    return kept, stats


def _replica_swap_accept(
    energy_i: float,
    energy_j: float,
    beta_i: float,
    beta_j: float,
    rng: np.random.Generator,
) -> bool:
    """
    Metropolis swap between replica i (β_i, E_i) and j (β_j, E_j).

    acc = min(1, exp((β_j - β_i) (E_i - E_j))).
    """
    d = (float(beta_j) - float(beta_i)) * (float(energy_i) - float(energy_j))
    if d >= 0.0:
        return True
    if d < -80.0:
        return False
    return float(rng.random()) < np.exp(d)


def run_replica_exchange(
    R0: np.ndarray,
    evaluate: Callable[[np.ndarray], float | tuple[float, Any]],
    rng: np.random.Generator,
    *,
    betas: np.ndarray,
    n_steps: int,
    burn_in: int,
    thin: int,
    step_rad: float,
    swap_interval: int = 1,
    chain_id: int = 0,
    adapt_interval: int = 50,
    target_acceptance: float = 0.28,
) -> tuple[list[OrientationSample], ReplicaExchangeStats]:
    """
    Parallel-tempering / replica-exchange MCMC on SO(3).

    Maintains M configurations (R_m, E_m) at fixed β_m. Each step: one Metropolis
    move per replica, then optional adjacent swaps.
    """
    evaluate_fn = wrap_energy_evaluator(evaluate)
    betas = np.asarray(betas, dtype=np.float64).reshape(-1)
    M = len(betas)
    if M < 2:
        raise ValueError("replica exchange requires at least 2 replicas")

    Rs = [np.asarray(R0, dtype=np.float64).copy() for _ in range(M)]
    for m in range(1, M):
        Rs[m] = perturb_rotation(Rs[m - 1], rng, step_rad * (1.0 + 0.25 * m))

    energies = np.empty(M, dtype=np.float64)
    results: list[Any] = [None] * M
    for m in range(M):
        energies[m], results[m] = evaluate_fn(Rs[m])

    steps = [float(step_rad)] * M
    n_acc_move = 0
    n_swap_try = 0
    n_swap_acc = 0
    thin = max(int(thin), 1)
    burn_in = max(int(burn_in), 0)
    swap_every = max(int(swap_interval), 1)
    kept: list[OrientationSample] = []

    for step_idx in range(int(n_steps)):
        for m in range(M):
            R_prop = perturb_rotation(Rs[m], rng, steps[m])
            e_prop, res_prop = evaluate_fn(R_prop)
            if metropolis_accept(float(energies[m]), e_prop, float(betas[m]), rng):
                Rs[m], energies[m], results[m] = R_prop, e_prop, res_prop
                n_acc_move += 1

        if (step_idx + 1) % swap_every == 0:
            for m in range(M - 1):
                n_swap_try += 1
                if _replica_swap_accept(
                    float(energies[m]),
                    float(energies[m + 1]),
                    float(betas[m]),
                    float(betas[m + 1]),
                    rng,
                ):
                    Rs[m], Rs[m + 1] = Rs[m + 1], Rs[m]
                    energies[m], energies[m + 1] = energies[m + 1], energies[m]
                    results[m], results[m + 1] = results[m + 1], results[m]
                    n_swap_acc += 1

        if adapt_interval > 0 and (step_idx + 1) % adapt_interval == 0:
            rate = n_acc_move / float((step_idx + 1) * M)
            if rate > target_acceptance * 1.15:
                steps = [min(s * 1.08, np.deg2rad(45.0)) for s in steps]
            elif rate < target_acceptance * 0.85:
                steps = [max(s / 1.08, np.deg2rad(0.5)) for s in steps]

        past_burn = step_idx >= burn_in
        is_thin = ((step_idx - burn_in) % thin) == 0 if past_burn else False
        if past_burn and is_thin:
            for m in range(M):
                kept.append(
                    OrientationSample(
                        rotation=Rs[m].copy(),
                        energy=float(energies[m]),
                        result=results[m],
                        source="replica",
                        chain_id=chain_id,
                        accepted=True,
                        beta_at_sample=float(betas[m]),
                    )
                )

    stats = ReplicaExchangeStats(
        n_replicas=M,
        n_steps=int(n_steps),
        n_swap_attempts=n_swap_try,
        n_swap_accepted=n_swap_acc,
        betas=betas.copy(),
        mean_swap_rate=float(n_swap_acc / max(n_swap_try, 1)),
    )
    return kept, stats


def run_multimodal_from_seeds(
    seed_samples: list[OrientationSample],
    evaluate: Callable[[np.ndarray], float | tuple[float, Any]],
    rng: np.random.Generator,
    *,
    mode: str,
    beta_target: float,
    mcmc_cfg: Any,
    anneal_cfg: Any,
    replica_cfg: Any,
) -> tuple[list[OrientationSample], dict]:
    """
    Launch multimodal sampling from low-energy seeds.

    ``mode``: ``fixed_beta`` | ``simulated_annealing`` | ``replica_exchange``.
    """
    mode = str(mode).lower()
    n_chains = int(mcmc_cfg.n_chains)
    n_pool = min(int(mcmc_cfg.seed_pool), len(seed_samples))
    order = np.argsort([s.energy for s in seed_samples])[:n_pool]
    pick = np.linspace(0, n_pool - 1, max(1, n_chains)).astype(int)

    kept_all: list[OrientationSample] = []
    meta: dict = {"mode": mode, "chains": []}
    step_rad = float(np.deg2rad(mcmc_cfg.step_deg))

    beta_min_frac = float(getattr(anneal_cfg, "beta_min_fraction", 0.05))
    beta_min = float(beta_target) * max(beta_min_frac, 1e-12)
    beta_max = float(beta_target)

    for cid, idx in enumerate(pick[:n_chains]):
        R0 = seed_samples[int(order[int(idx)])].rotation
        if mode == "simulated_annealing":
            kept, st = run_simulated_annealing_chain(
                R0,
                evaluate,
                rng,
                beta_min=beta_min,
                beta_max=beta_max,
                n_steps=mcmc_cfg.steps_per_chain,
                burn_in=mcmc_cfg.burn_in,
                thin=mcmc_cfg.thin,
                step_rad=step_rad,
                schedule=str(getattr(anneal_cfg, "schedule", "geometric")),
                n_reheat_cycles=int(getattr(anneal_cfg, "n_reheat_cycles", 1)),
                chain_id=cid,
                target_acceptance=mcmc_cfg.target_acceptance,
            )
            meta["chains"].append({"type": "anneal", "stats": st})
        elif mode == "replica_exchange":
            n_rep = int(getattr(replica_cfg, "n_replicas", 8))
            betas = replica_beta_ladder(n_rep, beta_min, beta_max)
            kept, st = run_replica_exchange(
                R0,
                evaluate,
                rng,
                betas=betas,
                n_steps=mcmc_cfg.steps_per_chain,
                burn_in=mcmc_cfg.burn_in,
                thin=mcmc_cfg.thin,
                step_rad=step_rad,
                swap_interval=int(getattr(replica_cfg, "swap_interval", 1)),
                chain_id=cid,
                target_acceptance=mcmc_cfg.target_acceptance,
            )
            meta["chains"].append({"type": "replica", "stats": st})
        else:
            from anisotropy.orientation_mcmc import run_mcmc_chain

            kept, st = run_mcmc_chain(
                R0,
                evaluate,
                rng,
                beta=beta_target,
                n_steps=mcmc_cfg.steps_per_chain,
                burn_in=mcmc_cfg.burn_in,
                thin=mcmc_cfg.thin,
                step_rad=step_rad,
                chain_id=cid,
                target_acceptance=mcmc_cfg.target_acceptance,
            )
            for s in kept:
                s.beta_at_sample = beta_target
            meta["chains"].append({"type": "mcmc", "stats": st})
        kept_all.extend(kept)

    meta["beta_target"] = beta_target
    meta["beta_min"] = beta_min
    return kept_all, meta
