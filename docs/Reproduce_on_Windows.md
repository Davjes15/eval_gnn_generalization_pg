# Reproducing the results on Windows (with GPU) — step by step

This is a complete, copy‑paste guide to set up the project **from scratch on a
Windows PC with an NVIDIA GPU** and reproduce the study's results. It assumes you
have **no environment yet**. Commands are written for **PowerShell** (the default
Windows terminal, and the integrated terminal in VS Code).

> Use the current release branch and the v3 datasets/results. The full protocol is in `docs/Reproducibility.md`.

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

`main` already contains the full pipeline, so there is nothing to check out:

```powershell
cd C:\gnn\eval_gnn_generalization_pg
git fetch origin
git checkout main
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

**Then the full generation — optional now.** Both published datasets are committed
(`data_regime_a_fixed_topology_v3/`, `data_regime_b_contingencies_v3/`, ~79 MB), so you can skip straight to training or to
the checkpoint replay and still be on the exact tensors behind the reported
numbers. Regenerate only if you are changing the protocol or checking that the
generator still produces an equivalent draw. The benchmark needs **two** datasets
— a fixed-topology control arm and a varying-topology transfer arm — and their
protocols differ:

```powershell
# Regime A: fixed topology, one demand snapshot per sample
python transmission_graph_gen.py --grid all --max_k 0 --unique_demand --n_train 800 --n_val 100 --n_test 100 --out_dir data_regime_a_fixed_topology_v3

# Regime B: random N-1/N-2 line outages, disjoint blocked demand windows (audit item A5)
python transmission_graph_gen.py --grid all --max_k 2 --time_split blocked --n_train 800 --n_val 100 --n_test 100 --out_dir data_regime_b_contingencies_v3
```

`--time_split blocked` is not optional for Regime B: without it, train/val/test
draw demand snapshots from the whole year, and neighbouring 15-minute snapshots
leak across splits.

…**or one grid at a time** (easier to run in chunks) — substitute `--grid IEEE24`,
`IEEE39`, `IEEE118`, `UK` for `--grid all` in either command above.

A regenerated dataset is a **fresh draw from the same protocol**, not a byte copy
of the committed one: rejected samples (non-convergent AC solve, voltages outside
[0.8, 1.2] p.u.) shift the sampling. Reproduce the *protocol* with the validation
gate below — or use the committed datasets if you want the published draw. The
exact published numbers come from the saved checkpoints, released as
`ckpt_v3.tar.gz` on tag `v1.0.0` (`docs/Reproducibility.md` §3).

Each dataset is **4,000 graphs** (1,000 per grid), so the pair is 8,000 and about
80 MB. Generation is CPU‑bound (the solver), so `numba` matters here; the GPU is
used later, in training.

---

## 9. Validate before trusting anything
```powershell
python validate.py --data_dir data_regime_a_fixed_topology_v3 --regime a           # Regime A is fixed-topology by design
python validate.py --data_dir data_regime_b_contingencies_v3 --expect_blocked
```

Expected final line: **`ALL GATES PASSED`** (it checks tensor shapes, bus‑type
masking, that topology actually varies across samples, and that within‑grid MMD is
smaller than cross‑grid MMD). `--expect_blocked` adds gate **H**, split hygiene:
no demand snapshot shared between splits, no repeated `(t_idx, out_lines)`
scenario within a split, and disjoint chronological windows. Do not train on a
Regime B dataset that fails gate H — the transfer numbers would be leaked.

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
| `--experiment within` | the fixed-topology control arm (use with `data_regime_a_fixed_topology_v3`) |
| `--normalize` | `none` (raw physical units, no scaling; not used for the published results) or `pu_zscore` (**the protocol**) |
| `--arch_config` | frozen widths/depths/learning rates; always pass `configs\arch_config.json` |
| `--seeds` | `0 100 300 700 1000` (`0 100 300` for `nnconv`) |
| `--regime_tag` | `A` or `B`, stamped into the results rows |
| `--skip_existing` | resume a campaign; requires `--save_models` |

**Quick smoke test** (2 models, 2 grids, few epochs — a couple of minutes):

```powershell
python experiments.py --experiment both --models gcn gat --grids IEEE24 IEEE39 --epochs 20 --data_dir data_regime_b_contingencies_v3 --out results_smoke --normalize pu_zscore
```

### Running ONE model at a time (recommended for you)
Because the driver **overwrites** the shared `cross_context.csv` / `ood.csv` on each
run, give **each model its own output folder** so runs don't clobber each other. Run
these one by one — each is independent and uses the GPU:

```powershell
$cfg = "configs\arch_config.json"
$seeds = "0","100","300","700","1000"

# control arm (Regime A, fixed topology)
foreach ($m in "gcn","arma_gnn","gat","gin","transformer") {
  python experiments.py --experiment within --models $m --epochs 200 `
    --data_dir data_regime_a_fixed_topology_v3 --out results_v3\within_$m --regime_tag A `
    --normalize pu_zscore --arch_config $cfg --seeds $seeds `
    --save_models ckpt_v3\within_$m --skip_existing
}

# transfer arms (Regime B, varying topology)
foreach ($m in "gcn","arma_gnn","gat","gin","transformer") {
  python experiments.py --experiment cross --models $m --epochs 200 `
    --data_dir data_regime_b_contingencies_v3 --out results_v3\cross_$m --regime_tag B `
    --normalize pu_zscore --arch_config $cfg --seeds $seeds `
    --save_models ckpt_v3\cross_$m --skip_existing
  python experiments.py --experiment ood --models $m --epochs 200 `
    --data_dir data_regime_b_contingencies_v3 --out results_v3\ood_$m --regime_tag B `
    --batch_size_ood 96 --normalize pu_zscore --arch_config $cfg --seeds $seeds `
    --save_models ckpt_v3\ood_$m --skip_existing
}
```

`nnconv` is run separately with `--seeds 0 100 300` (documented asymmetry; it is
the most expensive and the highest-variance model). On Linux/macOS the same
campaign is one command: `python launch_campaign.py --data_a data_regime_a_fixed_topology_v3 --data_b data_regime_b_contingencies_v3 --out results_v3 --ckpt ckpt_v3 --logs logs_v3 --pool 7`, which
runs the 18 jobs through a bounded process pool.

On a multi-core CPU set `$env:OMP_NUM_THREADS = "1"` before launching several jobs
in parallel; multi-threaded jobs run concurrently oversubscribe the cores and
measurably slow the campaign down.

Each per-model invocation writes to its own arm folder under `results_v3\`
(`within_$m`, `cross_$m`, or `ood_$m`): `transfer_matrix_<model>.csv`,
`cross_context.csv`, `ood.csv`, `gscore.csv`, `gscore_ood.csv`, the (model‑independent)
`mmd_degree.csv` / `mmd_laplacian.csv` / `dc_baseline.csv` / `ood_distance.csv`, and
`summary.json`.

> **Note on transfer:** cross‑grid transfer needs **≥ 2 grids**. Passing a single
> `--grids IEEE39` gives only the within‑grid (diagonal) number, not a transfer matrix.
> The one‑model‑at‑a‑time commands above keep all four grids, which is what you want.

### Merge the per-model folders
After the runs, gather the per-model shards into the three published arm folders.
The commands mirror §5 of [`Reproducibility.md`](Reproducibility.md):

```powershell
python gather_results.py --shards "results_v3/within_*" --file within_grid.csv `
  --out results_v3/all_within --models gcn arma_gnn gat gin transformer nnconv --seed_shards
python gather_results.py --shards "results_v3/cross_*" --file cross_context.csv `
  --out results_v3/all_cross --models gcn arma_gnn gat gin transformer nnconv --seed_shards
python gather_results.py --shards "results_v3/ood_*" --file ood.csv `
  --out results_v3/all_ood --models gcn arma_gnn gat gin transformer nnconv --seed_shards
```

The merged arm tables are written to `results_v3/all_within/`,
`results_v3/all_cross/`, and `results_v3/all_ood/`.

---

## 11. Published results
The published tables and their source files are documented in
[`Results.md`](Results.md).

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
cd C:\gnn\eval_gnn_generalization_pg; git fetch origin; git checkout main; git pull
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric; pip install -r requirements.txt
$env:POWERGRAPH_NODE_DIR = "C:\gnn\PowerGraph-Node\13_Power_system"
python transmission_grids.py
python transmission_graph_gen.py --grid all --max_k 0 --unique_demand --n_train 800 --n_val 100 --n_test 100 --out_dir data_regime_a_fixed_topology_v3
python transmission_graph_gen.py --grid all --max_k 2 --time_split blocked --n_train 800 --n_val 100 --n_test 100 --out_dir data_regime_b_contingencies_v3
python validate.py --data_dir data_regime_a_fixed_topology_v3 --regime a           # Regime A is fixed-topology by design
python validate.py --data_dir data_regime_b_contingencies_v3 --expect_blocked
$env:OMP_NUM_THREADS = "1"
python experiments.py --experiment within --epochs 200 --data_dir data_regime_a_fixed_topology_v3 --out results_v3 --regime_tag A --normalize pu_zscore --arch_config configs\arch_config.json --seeds 0 100 300 700 1000 --save_models ckpt_v3 --skip_existing
python experiments.py --experiment both   --epochs 200 --data_dir data_regime_b_contingencies_v3 --out results_v3 --regime_tag B --normalize pu_zscore --arch_config configs\arch_config.json --seeds 0 100 300 700 1000 --batch_size_ood 96 --save_models ckpt_v3 --skip_existing
python eval_checkpoints.py --ckpt_root ckpt_v3 --data_a data_regime_a_fixed_topology_v3 --data_b data_regime_b_contingencies_v3 --normalize pu_zscore --out results_v3\physics
python rank_analysis.py --results_dir results_v3
python recompute_tables.py --results_dir results_v3
```
