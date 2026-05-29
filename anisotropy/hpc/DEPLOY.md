# Exact steps: GitHub → HPC → Conda → Slurm

Replace placeholders once:

| Placeholder | Your value (example) |
|-------------|----------------------|
| `CLUSTER` | `login.myuniversity.edu` |
| `USER` | your HPC username |
| `SCRATCH` | `/scratch/USER` or `$SCRATCH` (run `echo $SCRATCH` on cluster) |
| `ACCOUNT` | Slurm account from `acctmgr` / docs |
| `PARTITION` | e.g. `standard`, `cpu`, `batch` |
| `GITHUB_REPO` | `https://github.com/ndlevinzon/toys.git` |

Repo layout on GitHub: **`toys`** monorepo with code in **`toys/anisotropy/`**.

---

## Part A — On your PC (once): put code on GitHub

Skip if `toys` is already on GitHub with `anisotropy/`, `environment*.yml`, and `hpc/`.

### A1. Backup (Windows PowerShell)

```powershell
cd C:\Users\ndlev\OneDrive\Documents\Research\thesis
Copy-Item -Recurse toys "toys-backup-$(Get-Date -Format yyyy-MM-dd)"
```

### A2. Commit and push (from `toys` git root)

```powershell
cd C:\Users\ndlev\OneDrive\Documents\Research\thesis\toys

git status
git add .gitignore GIT_WORKFLOW.md anisotropy/
git commit -m "Pipeline code, conda envs, HPC Slurm scripts"
git push origin main
```

If push is rejected, finish [GIT_WORKFLOW.md](../GIT_WORKFLOW.md) first (untrack PDBs, then push).

### A3. Confirm on GitHub in a browser

You should see:

- `anisotropy/environment.yml`
- `anisotropy/environment-hpc.yml`
- `anisotropy/hpc/slurm_orientation.sbatch`
- `anisotropy/utils/.../forcefield_files/amino19.lib`

---

## Part B — On the HPC login node: clone + Conda

SSH in:

```bash
ssh USER@CLUSTER
```

### B1. Find scratch and modules

```bash
echo "HOME=$HOME"
echo "SCRATCH=${SCRATCH:-not set — use /scratch/$USER}"
module avail 2>&1 | grep -iE 'conda|mamba|miniforge|anaconda|python' | head -20
```

Write down the module name you will use (examples below use `miniforge` or `anaconda`).

### B2. Clone the repository

```bash
cd $HOME
git clone https://github.com/ndlevinzon/toys.git
cd toys/anisotropy
pwd
# should end in .../toys/anisotropy
```

Private repo: use SSH `git@github.com:ndlevinzon/toys.git` and set up a deploy key on the cluster.

### B3. Load Conda and create the **headless** environment

```bash
# Pick ONE line that works on your cluster:
module load miniforge
# module load anaconda
# module load Miniforge3

source "$(conda info --base)/etc/profile.d/conda.sh"

cd $HOME/toys/anisotropy
conda env create -f environment-hpc.yml
```

Wait until it finishes (5–20 minutes). Then:

```bash
conda activate anisotropy-hpc
python -c "import anisotropy, numpy, scipy, propka; print('anisotropy OK')"
which python
# expect .../envs/anisotropy-hpc/bin/python
```

Optional: also create full env on login node (for `visualize_patches` / PROPKA-heavy parameterize):

```bash
conda env create -f environment.yml
conda activate anisotropy
python -c "import pyvista; print('pyvista OK')"
```

### B4. (Optional) Full env with PROPKA for parameterize on cluster

```bash
conda activate anisotropy
cd $HOME/toys/anisotropy
python parameterize_mesh.py --help
```

---

## Part C — Copy your structures to the cluster (not in Git)

On **Windows** (PowerShell), from a machine that can `scp`:

```powershell
# Set these:
$USER = "your_hpc_username"
$CLUSTER = "login.myuniversity.edu"
$SCRATCH = "/scratch/your_hpc_username"   # ask HPC docs if unsure

ssh ${USER}@${CLUSTER} "mkdir -p ${SCRATCH}/toys_runs/9yp6/runs"

scp C:\Users\ndlev\OneDrive\Documents\Research\thesis\toys\anisotropy\9yp6.pdb `
    ${USER}@${CLUSTER}:${SCRATCH}/toys_runs/9yp6/

scp C:\Users\ndlev\OneDrive\Documents\Research\thesis\toys\anisotropy\9yp6.ply `
    ${USER}@${CLUSTER}:${SCRATCH}/toys_runs/9yp6/
```

If you have `patch_features.npz` from your laptop:

```powershell
scp C:\Users\ndlev\OneDrive\Documents\Research\thesis\toys\anisotropy\patch_features.npz `
    ${USER}@${CLUSTER}:${SCRATCH}/toys_runs/9yp6/
```

On the **cluster**, verify:

```bash
ls -la $SCRATCH/toys_runs/9yp6/
```

---

## Part D — Edit the Slurm script (login node)

```bash
cd $HOME/toys/anisotropy/hpc
nano slurm_orientation.sbatch
```

Change these lines (uncomment and set real values):

```bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
```

Confirm paths at top of script (defaults are usually fine):

```bash
REPO=$HOME/toys/anisotropy
RUN_DIR=$SCRATCH/toys_runs/9yp6
CONDA_ENV=anisotropy-hpc
```

If your cluster does **not** provide `conda` until you `module load`, uncomment and set:

```bash
module load miniforge
```

before the `source "$(conda info --base)/..."` line.

---

## Part E — Submit the heavy job

```bash
cd $HOME/toys/anisotropy/hpc
export SCRATCH=/scratch/USER          # use your real scratch path
export RUN_DIR=$SCRATCH/toys_runs/9yp6

sbatch slurm_orientation.sbatch
```

You should see: `Submitted batch job 12345678`

### E1. Monitor

```bash
squeue -u $USER
tail -f slurm-orient-12345678.out    # replace with your job id
```

### E2. When finished

```bash
ls -la $SCRATCH/toys_runs/9yp6/runs/orient_*/
# top_poses.json, diagnostics_*.png, diagnostics_orientation_sampling.json
```

---

## Part F — Copy results back to your PC

```powershell
scp -r USER@CLUSTER:/scratch/USER/toys_runs/9yp6/runs/orient_JOBID `
    C:\Users\ndlev\OneDrive\Documents\Research\thesis\toys\anisotropy\from_hpc\
```

---

## Part G — After you change code on GitHub

On the **login node**:

```bash
cd $HOME/toys/anisotropy
git pull origin main

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate anisotropy-hpc
conda env update -f environment-hpc.yml --prune
pip install -e . --no-deps
```

Resubmit `sbatch` — no need to recreate the env unless dependencies changed.

---

## Quick reference: two conda envs

| Env | File | Use on |
|-----|------|--------|
| `anisotropy-hpc` | `environment-hpc.yml` | Compute nodes, `orientation_sample.py --no-render` |
| `anisotropy` | `environment.yml` | Login node, PROPKA parameterize, PyVista |

---

## If something fails

| Symptom | Command / fix |
|---------|----------------|
| `conda: command not found` | `module load miniforge` (or ask HPC support) |
| `conda activate` fails in batch job | Add `module load` + `source .../conda.sh` in `.sbatch` (already in template) |
| `import anisotropy` fails | `cd $HOME/toys/anisotropy && conda activate anisotropy-hpc && pip install -e .` |
| Job uses 1 CPU | Set `#SBATCH --cpus-per-task=16`; script passes `--parallel-workers $SLURM_CPUS_PER_TASK` |
| Out of memory | `#SBATCH --mem=64G` or coarser grid: add `--grid-spacing 4.0` |
| PROPKA slow | Use `--pka-source table` in batch (already in `slurm_orientation.sbatch`) or parameterize on laptop and `scp` the `.npz` |
| `git clone` asks password | Use SSH keys or HTTPS token |

---

## Order checklist

- [ ] A: Code on GitHub (`anisotropy/`, `environment-hpc.yml`, `hpc/`, ff19SB `forcefield_files/`)
- [ ] B: `git clone` on HPC
- [ ] B: `conda env create -f environment-hpc.yml`
- [ ] B: `python -c "import anisotropy"` works
- [ ] C: `9yp6.pdb` and `9yp6.ply` on `$SCRATCH/toys_runs/9yp6/`
- [ ] D: `#SBATCH --account` and `--partition` set
- [ ] E: `sbatch slurm_orientation.sbatch`
- [ ] F: Download `runs/orient_*` when done
