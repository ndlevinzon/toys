Orientation sampling
====================

Physical target
---------------

Rigid rotations \(\Omega\) are sampled from the Boltzmann law

.. math::

   \pi(\Omega) \propto \exp\bigl(-\beta\, H(\Omega)\bigr),

with **fixed** lattice-gas occupancy \(n_i\) (vitrified slab template) in the current
sampler. Only \(\Omega\) is explored; spin MCMC is reserved for future work.

Inverse temperature
-----------------

Set ``sampling.beta`` in ``ising_params.yaml`` or use ``beta: auto`` to calibrate
\(\beta\) from the energy spread so effective sample size (ESS) is usable (default
target ESS ≈ 20). A single dominant peak in azimuth–elevation maps usually means
\(\beta\) is too large, not that SO(3) was poorly explored.

Sampling strategies
-------------------

``sampling.strategy`` (CLI ``--sampling-strategy``):

``uniform``
  Random SO(3) draws only.

``hybrid`` (default)
  ``n_uniform`` uniform poses → \(\beta\) auto → refinement.

``mcmc``
  Refinement chains only (short uniform preflight when ``beta: auto``).

MCMC mode (``sampling.mcmc.mode``)
----------------------------------

``fixed_beta``
  Standard Metropolis at target \(\beta\).

``simulated_annealing``
  \(\beta_k\) ramps from ``beta_min_fraction × β_target`` to ``β_target``; optional
  ``n_reheat_cycles`` reheats between cycles. Kept states are **importance-reweighted**
  to \(\beta_{\mathrm{target}}\).

``replica_exchange`` (recommended for multimodal landscapes)
  Parallel tempering: fixed \(\beta\) ladder, Metropolis moves, and swap moves satisfying
  detailed balance. No non-physical bias is added to \(H\).

Outputs
-------

``orientation_sample.py`` writes energy traces, viewing-direction heatmaps and
probability spheres, ``top_poses.json``, and optional PyVista snapshots under
``--outdir``. See the README for the full file list.
