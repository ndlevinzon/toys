Orientation sampling
====================

Physical target
---------------

Rigid rotations \(\Omega\) are sampled from the Boltzmann law

.. math::

   \pi(\Omega) \propto \exp\bigl(-\beta\, H(\Omega)\bigr),

with **fixed** lattice-gas occupancy \(n_i\) (vitrified slab template) in the current
sampler. Only \(\Omega\) is explored; spin MCMC is reserved for future work.

**Default Hamiltonian** (``ising_params.yaml``):

* **Solvation:** RISM-inspired first shell (pose-dependent) + Ising outer shell (precomputed).
* **Electrostatics:** linearized Poisson–Boltzmann on the lattice (``electrostatics.method: pb_slab``).
* **AWI:** depth-dependent \(\varepsilon\), \(\phi_0\), \(\kappa\) from ``awi_field``.

See :doc:`../PHYSICS_AND_PIPELINE`, :doc:`../RISM_LAYERED_SOLVATION`, and
:doc:`../PB_SLAB_ELECTROSTATICS`.

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

``orientation_sample.py`` writes under ``--outdir``:

* Energy traces and orientation diagnostics (azimuth in \([-\pi,\pi]\), elevation in \([-\pi/2,\pi/2]\)).
* ``top_poses.json`` — MAP pose = **maximum Boltzmann weight**; ``most_probable_poses`` /
  ``least_probable_poses`` ranked by weight.
* ``views_z_down/`` — default **10** highest- and **10** lowest-weight PNGs (camera along lab +Z).
* ``system_view_map.png`` — MAP orientation render.

Use ``--no-render`` on HPC. Configure ``output.n_orientation_renders`` in YAML or
``--n-render-poses`` on the CLI.

Full list: :doc:`../PHYSICS_AND_PIPELINE` §5 and the repository README.
