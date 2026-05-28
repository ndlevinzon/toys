Performance
===========

Fast orientation evaluator
--------------------------

Controlled by ``performance`` in ``ising_params.yaml``:

* **Vectorized rays** — no per-pose patch object rebuild.
* **Rotation-invariant pair sum** — screened Coulomb pairwise term computed once.
* **Electrostatic cutoff** — skip distant charge pairs (default 80 Å).
* **``mcmc_energy_only``** — scalar \(H\) during MCMC; full term breakdown for top poses.
* **Parallel workers** — uniform batches and independent MCMC chains across CPU cores.

Spectral note
-------------

The dominant pairwise term depends only on body-frame distances; per-orientation
cost is dominated by slab intrinsic potential and dipole coupling tabulated on a 1D
\(z\) grid (``anisotropy.spectral_electrostatics``).

Debugging
---------

``--slow-voxelization`` forces legacy PyVista interior tests every pose.
