# Running anisotropy on Slurm (HPC)

CPU-only pipeline. No GPU support is required. The heavy step is **`orientation_sample.py`** (hybrid / MCMC / replica exchange).

## Layout on the cluster

```text
$HOME/toys/                    # git clone of your repo
  anisotropy/
    hpc/                      # this folder — Slurm templates
    anisotropy/               # Python package
    orientation_sample.py
    ising_params.yaml

$SCRATCH/toys_runs/            # large / fast disk (site-specific)
  9yp6/
    9yp6.pdb
    9yp6.ply
    patch_features.npz        # from parameterize_mesh (recommended)
    runs/
      orient_001/
```

Replace `$SCRATCH` with your site’s variable (`$SLURM_TMPDIR`, `/scratch/$USER`, etc.).

---

## One-time setup (login node)

### 1. Clone code (code only — no PDBs in git)

```bash
cd $HOME
git clone https://github.com/ndlevinzon/toys.git
cd toys/anisotropy
```

### 2. Python environment (Conda)

```bash
module load miniforge    # or anaconda — site-specific
cd $HOME/toys/anisotropy

# Login / viz node: full env
conda env create -f environment.yml
conda activate anisotropy

# Compute nodes (--no-render): headless env
# conda env create -f environment-hpc.yml
# conda activate anisotropy-hpc
```

See [../CONDA.md](../CONDA.md). After `git pull`: `conda env update -f environment-hpc.yml --prune && pip install -e . --no-deps`

### 3. Test import (login node)

```bash
python -c "from anisotropy.ising_params import load_ising_params; print(load_ising_params().sampling.strategy)"
```

---

## Stage input data (not in GitHub)

From your **Windows** machine (PowerShell), copy structures to the cluster:

```powershell
scp anisotropy\9yp6.pdb anisotropy\9yp6.ply USER@cluster:$SCRATCH/toys_runs/9yp6/
scp anisotropy\patch_features.npz USER@cluster:$SCRATCH/toys_runs/9yp6/
```

Or `rsync`:

```bash
rsync -avz anisotropy/9yp6.{pdb,ply} USER@cluster:~/scratch/toys_runs/9yp6/
```

---

## Recommended workflow

| Step | Where | Command |
|------|--------|---------|
| 1. Mesh (if needed) | login / short job | `fit_protein_mesh.py` |
| 2. Parameterize | login or 1–2 h job | `parameterize_mesh.py` (PROPKA) |
| 3. Orientation sampling | **compute node** | `orientation_sample.py` (hours) |

For step 3, use **`--no-render`** and match **`--parallel-workers`** to Slurm CPUs.

Copy `hpc/ising_params.hpc.yaml` and pass `--ising-params` for cluster-tuned settings.

---

## Submit a job

```bash
cd $HOME/toys/anisotropy/hpc
export RUN_DIR=$SCRATCH/toys_runs/9yp6
export OUT_DIR=$RUN_DIR/runs/orient_${SLURM_JOB_ID:-local}

# Edit #SBATCH lines in slurm_orientation.sbatch first (account, partition, time).
sbatch slurm_orientation.sbatch
```

Monitor:

```bash
squeue -u $USER
tail -f slurm-orient-*.out   # job name from script
```

Results: `$OUT_DIR/top_poses.json`, diagnostics PNG/JSON, `anisotropy_run.log`.

---

## Slurm resources (starting point)

| Setting | Suggestion |
|---------|------------|
| `--cpus-per-task` | 16–32 for parallel uniform + `fixed_beta` MCMC |
| `--mem` | 16–32 GB (large lattice + mesh) |
| `--time` | 4–24 h for replica exchange; 1–4 h for `fixed_beta` |
| GPU | Not used |

**Replica exchange** (`mcmc.mode: replica_exchange`) uses 8 energy evals per step × many steps — prefer **`fixed_beta`** on HPC unless you need multimodal sampling.

Set in YAML or CLI:

```yaml
mcmc:
  mode: fixed_beta
```

---

## Environment variables in batch jobs

The provided `slurm_orientation.sbatch` sets:

- `OMP_NUM_THREADS=1` — avoid oversubscription with multiprocessing
- `MKL_NUM_THREADS=1` — same for NumPy/SciPy
- `--parallel-workers $SLURM_CPUS_PER_TASK` — **critical** (do not rely on `os.cpu_count()` alone)

---

## Array jobs (many structures)

See `slurm_orientation_array.sbatch`. One array task per protein directory under `$SCRATCH/toys_runs/*/`.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Job uses 1 core | Set `--cpus-per-task` and `--parallel-workers` |
| PROPKA fails | `module load propka` or run parameterize on laptop, `scp` the `.npz` |
| OOM | Increase `--mem` or coarsen `--grid-spacing` |
| No PNG needed | `--no-render` |
| Import error | `cd` to `anisotropy/` and `pip install -e .` in job script |

Site-specific module names: ask `module avail python` on your cluster.
