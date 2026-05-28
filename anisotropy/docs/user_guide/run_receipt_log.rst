Run receipt log
===============

All command-line utilities share one **append-only** log file so successive runs form
a chronological receipt of what executed, with what arguments, and what was produced.

Default path
------------

``anisotropy/anisotropy_run.log`` (next to ``ising_params.yaml``).

Each run writes a block:

- UTC-local timestamp per line
- Utility name, PID, working directory, full ``sys.argv``
- All status messages (paths, energies, diagnostics)
- ``STATUS: OK`` or failure traceback
- Elapsed seconds

Console behaviour
-----------------

By default **only a tqdm progress bar** appears on stdout. The bar description updates
whenever a calculation **starts** (energy evaluation, mesh fit step, etc.).

Mirror log lines to the console with:

.. code-block:: powershell

   --verbose-console

CLI flags (all utilities)
-------------------------

``--log-file PATH``
  Custom log path (still append unless overwrite is set).

``--log-overwrite``
  Truncate the log at the start of this run instead of appending.

``--verbose-console``
  Echo every ``run.log()`` line to stdout in addition to the progress bar.

Example
-------

.. code-block:: powershell

   python orientation_sample.py 9yp6.pdb 9yp6.ply --outdir od1
   python orientation_sample.py 9yp6.pdb 9yp6.ply --outdir od2 --log-overwrite

The first command appends to ``anisotropy_run.log``. The second replaces the file.

Implementation
--------------

See :mod:`anisotropy.cli_runtime` — :class:`~anisotropy.cli_runtime.RunSession`,
:class:`~anisotropy.cli_runtime.CalculationProgress`, and :func:`~anisotropy.cli_runtime.run_main`.
