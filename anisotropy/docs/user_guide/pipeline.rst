Pipeline overview
=================

End-to-end workflow
-------------------

1. **``fit_protein_mesh.py``** — PDB → iterative SAS mesh; shape anisotropy metrics.
2. **``parameterize_mesh.py``** — grow curvature patches; assign charge, pKa, hydropathy, dipoles (ff19SB / PROPKA).
3. **``visualize_patches.py``** — optional PyVista inspection of patch scalars.
4. **``orientation_sample.py``** — sample rigid-body rotations \(\Omega \in \mathrm{SO}(3)\) against
   \(H[n;\Omega]\); write diagnostics, MAP pose, and viewing-direction maps.

Core library modules
--------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Module
     - Role
   * - ``anisotropy.pdb``
     - ATOM parsing, Bondi radii
   * - ``anisotropy.mesh``
     - SAS marching cubes + refinement
   * - ``anisotropy.patches``
     - Patch features and NPZ I/O
   * - ``anisotropy.awi_field``
     - AWI slab profiles (\(\varepsilon\), \(\phi_0\), \(\kappa\))
   * - ``anisotropy.lattice_solvent_hamiltonian``
     - Hybrid \(H = H_{\mathrm{solv}} + \cdots\)
   * - ``anisotropy.ising_params``
     - YAML parameter loading
   * - ``anisotropy.fast_orientation_eval``
     - Vectorized orientation energy path
   * - ``anisotropy.orientation_mcmc``
     - Metropolis on SO(3)
   * - ``anisotropy.orientation_multimodal``
     - Simulated annealing & replica exchange
   * - ``anisotropy.cli_runtime``
     - Shared run receipt log & console progress
