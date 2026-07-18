# Reproducing the results on Windows (with GPU) — step by step

This is a complete, copy‑paste guide to set up the project **from scratch on a
Windows PC with an NVIDIA GPU** and reproduce the study's results. It assumes you
have **no environment yet**. Commands are written for **PowerShell** (the default
Windows terminal, and the integrated terminal in VS Code).

> The repository has **7 stacked branches** (`step-1-…` → `step-7-…`) plus `main`.
> Each step branch builds on the previous one. **You only need one branch to run
> everything: `step-7-harvest-contingencies`** — it is the top of the stack and
> contains *all* the code from steps 1–7 (data generation, the 6 models, the
> experiment driver, the validation gates, and the committed grid files). `main`
> holds documentation, figures and the saved results — not the runnable pipeline.

---

## 0. What you need before you start
- **Windows 10/11**, ~5 GB free disk, internet access.
- **An NVIDIA GPU** + a recent NVIDIA driver. Check it works by running `nvidia-smi`
  in PowerShell — you should see your GPU and a "CUDA Version: XX.X" in the top‑right.
  (No CUDA toolkit install is needed; the PyTorch wheel ships its own CUDA runtime.)
- We install everything else below (Python, Git, PyTorch, PyTorch‑Geometric).

Total hands‑on time is ~15 minutes; a full 6‑model run trains for a while (see §10).

---

## 1. Install Python 3.11 and Git
The simplest way is Windows' built‑in `winget`. Open **PowerShell** and run:

```powershell
winget install -e --id Python.Python.3.11
winget install -e --id Git.Git
```

**Close and reopen PowerShell** so the new `python` and `git` are on your PATH, then
verify:

```powershell
py -3.11 --version
git --version
```

You should see `Python 3.11.x` and a git version. (If `winget` is unavailable,
install Python from https://www.python.org/downloads/ — tick **"Add python.exe to
PATH"** — and Git from https://git-scm.com/download/win.)

---

## 2. Get the code and the demand data (two repos, side by side)
The pipeline needs **PowerGraph‑Node** for the raw per‑bus hourly demand profiles
(`hourlyDemandBusnew.mat`, ~45 MB total). Clone both repos into the **same parent
folder** so the default paths line up. We'll use `C:\gnn` as the workspace:

```powershell
mkdir C:\gnn; cd C:\gnn
git clone https://github.com/Davjes15/eval_gnn_generalization_pg.git
git clone https://github.com/PowerGraph-Datasets/PowerGraph-Node.git
```

Now switch the project to the branch that contains the full pipeline:

```powershell
cd C:\gnn\eval_gnn_generalization_pg
git fetch origin
git checkout step-7-harvest-contingencies
```

> The grid case files (`transmission\cases\IEEE24.mat`, `IEEE39`, `IEEE118`, `UK`)
> are **already committed**, so you do **not** need MATLAB/Octave or the Step‑1
> conversion. You also do **not** need any of the large (GB‑scale) figshare
> downloads — those are only for the optional "harvested contingencies" mode (§11).

---

## 3. Create and activate a virtual environment
From `C:\gnn\eval_gnn_generalization_pg`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked by an execution‑policy error, run this once in the current
window and retry the activate line:

```powershell
Set-ExecutionPolicy -Scope Process -Bypass -Force
.\.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`. Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

---

## 4. Install PyTorch **with CUDA (GPU)** — do this first
Install the CUDA build of PyTorch from PyTorch's own index **before** the other
packages, so you don't get the CPU‑only wheel by accident. The `cu121` build works
with essentially all current NVIDIA drivers:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> If you have a very new driver and prefer a newer CUDA, pick the matching command
> from https://pytorch.org/get-started/locally/ (e.g. swap `cu121` for `cu124`).

**Verify the GPU is visible** (this must print `True` and your GPU name):

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If it prints `False`, your driver is too old or the CPU wheel got installed — update
the NVIDIA driver and re‑run the `pip install torch …cu121` line.

---

## 5. Install the remaining dependencies
```powershell
pip install torch_geometric
pip install -r requirements.txt
```

`requirements.txt` covers `pandapower`, `scipy`, `numpy`, `pandas`, `networkx`,
`numba` (a big speed‑up for the AC solver) and `matplotlib` for figures. The six GNN
layers used here run on stock `torch_geometric` — no extra compiled extensions
needed.

---

## 6. Point the pipeline at the demand data
The code finds the demand files via the `POWERGRAPH_NODE_DIR` environment variable.
Set it **for the current PowerShell session** (do this each time you open a new
terminal):

```powershell
$env:POWERGRAPH_NODE_DIR = "C:\gnn\PowerGraph-Node\13_Power_system"
```

To set it **permanently** for your user (so you never have to repeat it), run once
and then reopen PowerShell:

```powershell
[Environment]::SetEnvironmentVariable("POWERGRAPH_NODE_DIR", "C:\gnn\PowerGraph-Node\13_Power_system", "User")
```

---

## 7. Sanity‑check the grids load and solve
```powershell
python transmission_grids.py
```

Expected — one line per grid, each ending `converged=True` and a demand shape:

```
IEEE24   buses=  24 ... converged=True demand=(24, 35040)
IEEE39   buses=  39 ... converged=True demand=(39, 35040)
IEEE118  buses= 118 ... converged=True demand=(118, 35040)
UK       buses=  29 ... converged=True demand=(29, 35040)
```

If you instead see `demand file not found`, `POWERGRAPH_NODE_DIR` is wrong — re‑check
the path in §6 points at the folder that contains the `IEEE24`, `IEEE39`, … subfolders.

---

## 8. Generate the datasets
This samples demand snapshots + random N‑1/N‑2 line outages, **re‑solves AC power
flow** for each, and writes `data\<GRID>\<split>\dataset.pt`.

**First, a quick smoke test** (tiny, ~1 minute) to confirm everything works:

```powershell
python transmission_graph_gen.py --grid IEEE24 --n_train 30 --n_val 6 --n_test 6 --max_k 2 --out_dir data
```

**Then the full generation.** You can do all four grids at once:

```powershell
python transmission_graph_gen.py --grid all --n_train 800 --n_val 100 --n_test 100 --max_k 2 --out_dir data
```

…**or one grid at a time** (identical result, easier to run in chunks). The seeding
is fixed per grid, so this reproduces the published data exactly:

```powershell
python transmission_graph_gen.py --grid IEEE24  --n_train 800 --n_val 100 --n_test 100 --max_k 2 --out_dir data
python transmission_graph_gen.py --grid IEEE39  --n_train 800 --n_val 100 --n_test 100 --max_k 2 --out_dir data
python transmission_graph_gen.py --grid IEEE118 --n_train 800 --n_val 100 --n_test 100 --max_k 2 --out_dir data
python transmission_graph_gen.py --grid UK      --n_train 800 --n_val 100 --n_test 100 --max_k 2 --out_dir data
```

That produces **4,000 graphs total** (1,000 per grid). Generation is CPU‑bound
(the solver), so `numba` matters here; the GPU is used later, in training.

---

## 9. Validate before trusting anything
```powershell
python validate.py --data_dir data
```

Expected final line: **`ALL GATES PASSED`** (it checks tensor shapes, bus‑type
masking, that topology actually varies across samples, and that within‑grid MMD is
smaller than cross‑grid MMD).

---

## 10. Run the experiments — one model / one experiment at a time
The driver is `experiments.py`. It **auto‑detects your GPU** (you'll see
`device=cuda:0` printed at the start — no flag needed). Key flags:

| flag | meaning |
|---|---|
| `--experiment` | `cross` (train‑on‑1, test‑on‑all), `ood` (leave‑one‑grid‑out), or `both` |
| `--models` | any subset of `gcn arma_gnn gat gin transformer nnconv` |
| `--grids` | subset of `IEEE24 IEEE39 IEEE118 UK` (default: all four) |
| `--epochs` | training epochs (200 for the full run; 20 for quick tests) |
| `--data_dir` / `--out` | input data folder / output results folder |
| `--save_models <dir>` | also save every trained model's weights |

**Quick smoke test** (2 models, 2 grids, few epochs — a couple of minutes):

```powershell
python experiments.py --experiment both --models gcn gat --grids IEEE24 IEEE39 --epochs 20 --data_dir data --out results_smoke
```

### Running ONE model at a time (recommended for you)
Because the driver **overwrites** the shared `cross_context.csv` / `ood.csv` on each
run, give **each model its own output folder** so runs don't clobber each other. Run
these one by one — each is independent and uses the GPU:

```powershell
python experiments.py --experiment both --models gcn         --epochs 200 --data_dir data --out results\gcn         --save_models models\gcn
python experiments.py --experiment both --models arma_gnn    --epochs 200 --data_dir data --out results\arma_gnn    --save_models models\arma_gnn
python experiments.py --experiment both --models gat         --epochs 200 --data_dir data --out results\gat         --save_models models\gat
python experiments.py --experiment both --models gin         --epochs 200 --data_dir data --out results\gin         --save_models models\gin
python experiments.py --experiment both --models transformer --epochs 200 --data_dir data --out results\transformer --save_models models\transformer
python experiments.py --experiment both --models nnconv      --epochs 200 --data_dir data --out results\nnconv      --save_models models\nnconv
```

Each `results\<model>\` folder is self‑contained: `transfer_matrix_<model>.csv`,
`cross_context.csv`, `ood.csv`, `gscore.csv`, `gscore_ood.csv`, the (model‑independent)
`mmd_degree.csv` / `mmd_laplacian.csv` / `dc_baseline.csv` / `ood_distance.csv`, and
`summary.json`.

> **Note on transfer:** cross‑grid transfer needs **≥ 2 grids**. Passing a single
> `--grids IEEE39` gives only the within‑grid (diagonal) number, not a transfer matrix.
> The one‑model‑at‑a‑time commands above keep all four grids, which is what you want.

### Merge the per‑model folders into one (for the combined figures)
After the six runs, combine them into a single `results\` folder so the figure
script sees every model. Copy‑paste this whole block:

```powershell
$dst = "results"; New-Item -ItemType Directory -Force -Path $dst | Out-Null
# per-model transfer matrices (one file each) + saved copies
Get-ChildItem results\*\transfer_matrix_*.csv | Copy-Item -Destination $dst -Force
# model-independent tables: copy once from any model folder (use gcn)
Copy-Item results\gcn\mmd_degree.csv, results\gcn\mmd_laplacian.csv, results\gcn\dc_baseline.csv, results\gcn\ood_distance.csv $dst -Force
# stack the per-model rows into single CSVs
python -c "import pandas as pd,glob,os; [pd.concat([pd.read_csv(f) for f in glob.glob(f'results/*/{n}.csv')],ignore_index=True).to_csv(f'results/{n}.csv',index=False) for n in ['cross_context','ood','gscore','gscore_ood']]"
```

Now `results\` holds the full multi‑model set, identical in structure to the
published `full_run\results\`.

---

## 11. (Optional) Reproduce the figures
The figure script resolves its results folder from its first argument. Point it at
your merged `results\` folder:

```powershell
python full_run\results\make_figures.py (Resolve-Path results)
```

Figures are written to `results\figures\` (`fig_transfer_matrix.png`,
`fig_per_quantity.png`, `fig_gnn_vs_dc.png`, `fig_mmd_range.png`,
`fig_mmd_heatmap.png`, `fig_performance.png`, `fig_generalizability_curve.png`,
`fig_gscore_ood.png`).

---

## 12. (Optional) Harvested contingencies instead of random
This drives generation with the **actual outage sets** PowerGraph‑Graph simulated
(re‑solved by us). It requires a large (~2.7 GB) one‑time download and `mat73`:

```powershell
pip install mat73
# download + extract PowerGraph-Graph raw data, then:
$env:PG_GRAPH_RAW_DIR = "C:\gnn\PowerGraph-Graph\raw_root"
python transmission_graph_gen.py --grid IEEE24 --contingency_source harvest --pg_graph_raw $env:PG_GRAPH_RAW_DIR --n_train 800 --n_val 100 --n_test 100 --out_dir data
```

The default (`--contingency_source random`) needs none of this and is what the
published results use.

---

## 13. Troubleshooting
- **`torch.cuda.is_available()` is `False`** — you likely got the CPU wheel or your
  driver is old. Update the NVIDIA driver, then `pip uninstall torch -y` and re‑run
  the `pip install torch --index-url …cu121` line from §4.
- **`cannot be loaded because running scripts is disabled`** — run
  `Set-ExecutionPolicy -Scope Process -Bypass -Force` then re‑activate the venv (§3).
- **`Demand file not found …`** — `POWERGRAPH_NODE_DIR` is wrong (§6); it must point
  at the `13_Power_system` folder that contains the `IEEE24` … subfolders.
- **`ModuleNotFoundError`** — make sure the venv is active (prompt shows `(.venv)`)
  and you ran §4–§5 inside it.
- **A power‑flow sample fails to converge** — expected occasionally; the generator
  filters non‑converged/disconnected cases and keeps sampling, so just let it run.
- **Long path errors on Windows** — keep the workspace shallow (e.g. `C:\gnn`) as in
  this guide rather than a deeply nested folder.

---

## 14. Minimal "just reproduce it" sequence (all in one place)
```powershell
winget install -e --id Python.Python.3.11; winget install -e --id Git.Git
# reopen PowerShell, then:
mkdir C:\gnn; cd C:\gnn
git clone https://github.com/Davjes15/eval_gnn_generalization_pg.git
git clone https://github.com/PowerGraph-Datasets/PowerGraph-Node.git
cd C:\gnn\eval_gnn_generalization_pg; git fetch origin; git checkout step-7-harvest-contingencies
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric; pip install -r requirements.txt
$env:POWERGRAPH_NODE_DIR = "C:\gnn\PowerGraph-Node\13_Power_system"
python transmission_grids.py
python transmission_graph_gen.py --grid all --n_train 800 --n_val 100 --n_test 100 --max_k 2 --out_dir data
python validate.py --data_dir data
python experiments.py --experiment both --epochs 200 --data_dir data --out results --save_models models
python full_run\results\make_figures.py (Resolve-Path results)
```
