Getting started
===============

Python environment
------------------

Use **Python 3.10+**. The cryo-EM **toys** project venv is recommended:

.. code-block:: powershell

   cd ..\toys
   py -3 -m venv .venv
   .\.venv\Scripts\pip install -r requirements.txt
   .\.venv\Scripts\pip install -e "..\anisotropy[view,docs]"

Quick pipeline
--------------

.. code-block:: powershell

   cd anisotropy
   ..\toys\.venv\Scripts\python.exe fit_protein_mesh.py 1CRN.pdb -o 1crn_sas.ply
   ..\toys\.venv\Scripts\python.exe parameterize_mesh.py 1CRN.pdb 1crn_sas.ply -o patch_features.npz
   ..\toys\.venv\Scripts\python.exe orientation_sample.py 1CRN.pdb 1crn_sas.ply --outdir orientation_diagnostics

During execution the **console shows only a live progress bar**; full status text is
appended to :doc:`user_guide/run_receipt_log`.

Configuration
-------------

Hamiltonian and sampler defaults live in ``ising_params.yaml``. Override keys on the
CLI (``orientation_sample.py``) or load a custom file with ``--ising-params``.

Build documentation locally
---------------------------

.. code-block:: powershell

   cd anisotropy
   pip install -e ".[docs]"
   sphinx-build -b html docs docs/_build/html

Open ``_build/html/index.html``.

Read the Docs
-------------

Hosted builds use ``.readthedocs.yaml`` at the project root. Connect the repository
on https://readthedocs.org/ and set the documentation root to ``anisotropy/docs/``.
