"""
Parallel orientation energy evaluations (multi-core CPU).

Uses ``concurrent.futures`` with picklable :class:`~anisotropy.fast_orientation_eval.FastOrientationEvaluator`
state. MCMC chains can run one per worker process.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, Sequence

import numpy as np

from anisotropy.fast_orientation_eval import FastOrientationEvaluator
from anisotropy.orientation_mcmc import (
    McmcRunSummary,
    OrientationSample,
    random_rotation_matrix,
    run_mcmc_chain,
)


def default_worker_count(n_tasks: int, cap: int | None = None) -> int:
    """Reasonable worker count for CPU-bound NumPy work."""
    n_cpu = os.cpu_count() or 4
    cap = n_cpu if cap is None else min(int(cap), n_cpu)
    return max(1, min(cap, int(n_tasks)))


# Process-pool globals (initializer)
_G_EVAL: FastOrientationEvaluator | None = None


def _init_pool(eval_state: dict) -> None:
    global _G_EVAL
    _G_EVAL = FastOrientationEvaluator.from_state_dict(eval_state)


def _energy_worker(R_flat: np.ndarray) -> float:
    assert _G_EVAL is not None
    R = R_flat.reshape(3, 3)
    return float(_G_EVAL.energy(R))


def parallel_energies(
    evaluator: FastOrientationEvaluator,
    rotations: Sequence[np.ndarray],
    *,
    n_workers: int | None = None,
    use_threads: bool = False,
    chunk_size: int = 8,
    on_complete: Callable[[int], None] | None = None,
) -> np.ndarray:
    """
    Evaluate H for many rotations in parallel.

    ``use_threads=True`` avoids process spawn overhead (good for small batches);
    processes scale better for long runs on Windows.
    """
    n = len(rotations)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.array([evaluator.energy(rotations[0])], dtype=np.float64)

    flats = [np.asarray(R, dtype=np.float64).reshape(-1) for R in rotations]
    nw = default_worker_count(n, n_workers)

    if nw <= 1:
        return np.array([evaluator.energy(R) for R in rotations], dtype=np.float64)

    if use_threads:
        with ThreadPoolExecutor(max_workers=nw) as ex:
            return np.array(list(ex.map(evaluator.energy, rotations)), dtype=np.float64)

    state = evaluator.state_dict()
    out = np.empty(n, dtype=np.float64)
    with ProcessPoolExecutor(
        max_workers=nw,
        initializer=_init_pool,
        initargs=(state,),
    ) as ex:
        futs = {ex.submit(_energy_worker, flats[i]): i for i in range(n)}
        for fut in as_completed(futs):
            i = futs[fut]
            out[i] = fut.result()
            if on_complete is not None:
                on_complete(int(i))
    return out


def parallel_uniform_samples(
    evaluator: FastOrientationEvaluator,
    n: int,
    rng: np.random.Generator,
    *,
    n_workers: int | None = None,
    on_energy_complete: Callable[[int, int], None] | None = None,
) -> list[OrientationSample]:
    """Draw ``n`` uniform rotations and evaluate energies in parallel."""
    Rs = [random_rotation_matrix(rng) for _ in range(int(n))]
    n_eval = int(n)

    def _done(i: int) -> None:
        if on_energy_complete is not None:
            on_energy_complete(int(i), n_eval)

    energies = parallel_energies(
        evaluator, Rs, n_workers=n_workers, on_complete=_done
    )
    return [
        OrientationSample(R=Rs[i], energy=float(energies[i]), result=None, source="uniform")
        for i in range(n)
    ]


def _mcmc_chain_job(payload: tuple) -> tuple[list[OrientationSample], dict]:
    (
        eval_state,
        R0_flat,
        beta,
        n_steps,
        burn_in,
        thin,
        step_rad,
        chain_id,
        seed,
        target_acc,
    ) = payload
    ev = FastOrientationEvaluator.from_state_dict(eval_state)
    R0 = np.asarray(R0_flat, dtype=np.float64).reshape(3, 3)
    rng = np.random.default_rng(int(seed))

    kept, stats = run_mcmc_chain(
        R0,
        ev.energy,
        rng,
        beta=float(beta),
        n_steps=int(n_steps),
        burn_in=int(burn_in),
        thin=int(thin),
        step_rad=float(step_rad),
        chain_id=int(chain_id),
        target_acceptance=float(target_acc),
    )
    return kept, {
        "chain_id": stats.chain_id,
        "n_proposed": stats.n_proposed,
        "n_accepted": stats.n_accepted,
        "acceptance_rate": stats.acceptance_rate,
        "final_step_deg": stats.final_step_deg,
        "energy_start": stats.energy_start,
        "energy_end": stats.energy_end,
    }


def parallel_mcmc_chains(
    evaluator: FastOrientationEvaluator,
    R0_list: list[np.ndarray],
    *,
    beta: float,
    n_steps: int,
    burn_in: int,
    thin: int,
    step_rad: float,
    seeds: list[int],
    target_acceptance: float = 0.28,
    n_workers: int | None = None,
) -> tuple[list[OrientationSample], McmcRunSummary]:
    """Run independent MCMC chains (one seed pose per chain) in parallel."""
    n_chains = len(R0_list)
    state = evaluator.state_dict()
    jobs = [
        (
            state,
            np.asarray(R0, dtype=np.float64).reshape(-1),
            beta,
            n_steps,
            burn_in,
            thin,
            step_rad,
            cid,
            seeds[cid],
            target_acceptance,
        )
        for cid, R0 in enumerate(R0_list)
    ]
    nw = default_worker_count(n_chains, n_workers)
    all_kept: list[OrientationSample] = []
    chain_stats = []

    if nw <= 1 or n_chains == 1:
        for job in jobs:
            kept, st = _mcmc_chain_job(job)
            all_kept.extend(kept)
            chain_stats.append(st)
    else:
        with ProcessPoolExecutor(max_workers=nw) as ex:
            for kept, st in ex.map(_mcmc_chain_job, jobs):
                all_kept.extend(kept)
                chain_stats.append(st)

    total_prop = sum(int(s["n_proposed"]) for s in chain_stats)
    total_acc = sum(int(s["n_accepted"]) for s in chain_stats)
    from anisotropy.orientation_mcmc import McmcChainStats

    chains = [
        McmcChainStats(
            chain_id=int(s["chain_id"]),
            n_proposed=int(s["n_proposed"]),
            n_accepted=int(s["n_accepted"]),
            acceptance_rate=float(s["acceptance_rate"]),
            final_step_deg=float(s["final_step_deg"]),
            energy_start=float(s["energy_start"]),
            energy_end=float(s["energy_end"]),
        )
        for s in chain_stats
    ]
    summary = McmcRunSummary(
        chains=chains,
        total_proposed=total_prop,
        total_accepted=total_acc,
        mean_acceptance_rate=float(total_acc / max(total_prop, 1)),
    )
    return all_kept, summary
