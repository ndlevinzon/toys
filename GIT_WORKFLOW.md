# Git workflow: publish code, keep work local

Your repository root is **`toys/`** (remote: `github.com/ndlevinzon/toys`).  
This protocol **resets what Git tracks** so GitHub gets only the runnable pipeline, while PDBs, meshes, diagnostics, and logs stay on your machine.

---

## What goes where

| On GitHub (tracked) | Local only (ignored) |
|---------------------|----------------------|
| `anisotropy/anisotropy/*.py` | `*.pdb`, `*.ply`, `*.npz`, `*.mrc` |
| `anisotropy/*.py` CLIs | `orientation_diagnostics/` |
| `anisotropy/ising_params.yaml` | `anisotropy_run.log`, `_test_*`, `_tmp*` |
| `anisotropy/tests/`, `docs/` | `.venv/` |
| `anisotropy/utils/.../forcefield_files/` | Run PNGs (except under `docs/`) |
| `pyproject.toml`, `requirements.txt`, README, LICENSE | Personal structure files (`9yp6.pdb`, etc.) |
| `toys/*.py` (beam toy code) | `toys/datasets/`, large outputs |

**.gitignore only affects untracked files.** Files already in Git history stay tracked until you remove them with `git rm --cached`.

---

## Before you start

1. **Do not run `git pull` yet** — your branch is behind `origin/main`; pulling can delete local copies of files GitHub removed.
2. Close anything that might lock files in `anisotropy/`.
3. Optional but recommended: copy your whole folder as a safety backup:

```powershell
cd C:\Users\ndlev\OneDrive\Documents\Research\thesis
Copy-Item -Recurse toys "toys-backup-$(Get-Date -Format yyyy-MM-dd)"
```

---

## Step 1 — Fix ignore rules (done in repo)

Root [`.gitignore`](.gitignore) now uses paths like `anisotropy/**/*.pdb` (correct for this repo).  
[`anisotropy/.gitignore`](anisotropy/.gitignore) adds extra rules inside that folder.

Verify a work file is ignored:

```powershell
cd C:\Users\ndlev\OneDrive\Documents\Research\thesis\toys
git check-ignore -v anisotropy\9yp6.pdb
```

You should see a matching rule from `.gitignore`. If not, fix the pattern before continuing.

---

## Step 2 — Stop tracking local work (files stay on disk)

From the **repo root** (`toys/`):

```powershell
cd C:\Users\ndlev\OneDrive\Documents\Research\thesis\toys

# Un-track data & outputs under anisotropy (does NOT delete from disk)
git rm -r --cached anisotropy/*.pdb 2>$null
git rm -r --cached anisotropy/*.ply 2>$null
git rm -r --cached anisotropy/*.npz 2>$null
git rm -r --cached anisotropy/orientation_diagnostics 2>$null
git rm -r --cached anisotropy/_tmp* 2>$null
git rm --cached anisotropy/anisotropy_run.log 2>$null

# Re-stage only what should be public (respects .gitignore)
git add .gitignore GIT_WORKFLOW.md
git add anisotropy/.gitignore anisotropy/GITHUB_SETUP.md
git add anisotropy/anisotropy/
git add anisotropy/*.py
git add anisotropy/ising_params.yaml
git add anisotropy/pyproject.toml anisotropy/requirements.txt
git add anisotropy/tests/
git add anisotropy/docs/
git add anisotropy/.readthedocs.yaml
git add anisotropy/README.md anisotropy/LICENSE
git add anisotropy/utils/ff19SB_201907-master/ff19SB_201907-master/forcefield_files/
git add toys/*.py toys/requirements.txt 2>$null
```

Check what Git will commit (no PDBs/PLYs should appear):

```powershell
git status
```

Your PDB/PLY/NPZ files should appear under **“Untracked files”** or not at all — not under “Changes to be committed” as additions.

---

## Step 3 — Commit the clean snapshot

```powershell
git commit -m "Track pipeline code only; ignore local structures and run outputs"
```

---

## Step 4 — Publish to GitHub (replace remote history)

You are **not** merging the old 12 remote commits that deleted files. You are publishing **your** clean tree.

**Warning:** This rewrites `main` on GitHub. Only do this if you are fine replacing remote history.

```powershell
git push --force-with-lease origin main
```

If GitHub rejects the lease, someone else pushed — stop and inspect. If you are the only user:

```powershell
git push --force origin main
```

After this, GitHub matches your local code-only tree. Local work files remain on disk; they are simply not in the repo.

---

## Step 5 — Daily protocol (ongoing)

### Run the pipeline (local work)

```powershell
cd anisotropy
..\.venv\Scripts\python.exe orientation_sample.py 9yp6.pdb 9yp6.ply --outdir orientation_diagnostics
```

Outputs land in ignored folders — Git will not ask you to commit them.

### Publish code changes

```powershell
cd C:\Users\ndlev\OneDrive\Documents\Research\thesis\toys
git status
git add anisotropy/anisotropy/ anisotropy/*.py anisotropy/ising_params.yaml anisotropy/tests/ anisotropy/docs/
git commit -m "Describe the code change"
git push
```

Never `git add .` blindly — review `git status` first.

### Sync from GitHub (safe habit)

```powershell
git pull
```

Safe **after** work files are untracked and ignored; pull will not delete ignored local PDBs.

---

## Optional: fresh orphan branch (hard reset)

Use only if Step 2–4 feels messy and you want a single clean commit on `main`.

```powershell
cd C:\Users\ndlev\OneDrive\Documents\Research\thesis\toys

git checkout --orphan clean-main
git reset --hard
git add .gitignore GIT_WORKFLOW.md
# … same selective git add as Step 2 …
git commit -m "Clean pipeline-only repository"
git branch -M main
git push --force-with-lease origin main
```

---

## Optional: separate repo for anisotropy only

If you prefer GitHub to show only the orientation package (not the whole `toys` monorepo):

1. Create `github.com/ndlevinzon/anisotropy` (empty).
2. Copy only the `anisotropy/` folder to a new directory, `git init` there, push to the new remote.
3. Keep this `toys` repo for beam-projection toys + local work.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `git pull` deleted my PDBs | Restore from `toys-backup-…` or OneDrive version history |
| PDB still listed in `git status` staged | `git rm --cached anisotropy/your.pdb` then commit |
| `check-ignore` shows nothing | Path wrong in `.gitignore`; use `anisotropy/**/*.pdb` from repo root |
| Force push scared | Use a new branch first: `git push origin main:code-only` and open a PR |

---

## One-line summary

**Ignore work in `.gitignore` → `git rm --cached` old data → commit code → `push --force-with-lease` once → thereafter `git add` only code paths and `git pull` freely.**
