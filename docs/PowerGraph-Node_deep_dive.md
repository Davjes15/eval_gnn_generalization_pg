# PowerGraph-Node — Deep Dive

**Purpose:** node-level GNN regression for **Power Flow (PF)** and **Optimal Power Flow (OPF)** on transmission grids (IEEE-24, IEEE-39, IEEE-118, UK; `texas` is stubbed in the loader). Each grid *loading condition* becomes one graph, and the GNN predicts the full per-bus electrical state `[P, Q, V, θ]`.

Part of the **PowerGraph benchmark** (Varbella et al., *"PowerGraph: A power grid benchmark dataset for graph neural networks"*). Sibling repos: `PowerGraph-Graph` (graph-level cascading-failure tasks) and `PowerGraph-XAI` (explainability benchmark).

---

## Table of Contents
1. [End-to-end pipeline](#1-end-to-end-pipeline)
2. [Data generation (MATLAB / MATPOWER)](#2-data-generation-matlab--matpower)
3. [Data format (the `.mat` files)](#3-data-format-the-mat-files)
4. [Dataset loader (`powergrid.py`)](#4-dataset-loader-powergridpy)
5. [Models (`gnn/model.py`)](#5-models-gnnmodelpy)
6. [Training & evaluation (`train_gnn.py`)](#6-training--evaluation-train_gnnpy)
7. [Data loading & splits (`gendata.py`)](#7-data-loading--splits-gendatapy)
8. [Config & defaults (`parser_utils.py`)](#8-config--defaults-parser_utilspy)
9. [Post-processing / plotting](#9-post-processing--plotting)
10. [How it compares to ENGAGE](#10-how-it-compares-to-engage)
11. [⚠️ Gotchas & things to watch](#11-️-gotchas--things-to-watch)

---

## 1. End-to-end pipeline

```
MATLAB (gendataopf.m + MATPOWER)
        │  runs OPF & PF over many demand snapshots
        ▼
.mat files  (X, Y_polar, edge_index, edge_attr, + OPF variants)   ← hosted on figshare (~1.08 GB)
        │
        ▼
PyG InMemoryDataset  (code/dataset/powergrid.py)
        │  normalize, build mask, save processed_node/ or processed_nodeopf/
        ▼
GNN training/eval  (code/train_gnn.py + code/gnn/model.py)
        │  masked MSE loss, R² metric, de-normalize with per-graph maxs
        ▼
Excel summaries (summary*.xlsx)  →  plots (plotnodelevel.py)
```

The repo ships **only** the code and the MATPOWER case files (`13_Power_system/<grid>/System.m`). The actual `.mat` datasets are downloaded from figshare:
```bash
wget -O data.tar.gz "https://figshare.com/ndownloader/files/46619152"
tar -xf data.tar.gz
```

---

## 2. Data generation (MATLAB / MATPOWER)

### `interpDemand.m`
Loads a grid's historical `hourlyDemandBus.mat` and **linearly interpolates** each bus's demand time series to `35040` points (= 15-minute resolution over a year), saving `hourlyDemandBusnew.mat`. This gives many distinct loading conditions per grid.

### `gendataopf.m` (the core generator)
For each grid and each demand snapshot `j`:
1. Sets each bus's demand `PD` from `hourlyDemandBusnew(:,j)`.
2. Splits buses by MATPOWER type: **PQ = type 1, PV = type 2, slack = type 3**.
3. **OPF branch:** `runopf(mpctry)` → records:
   - input `X{}` = `[−P_load, −Q_load, 0, bus_type]` per bus
   - output `Y_polar{}` = `[P, Q, V, θ]` (PQ buses get `[0, 0, V, θ]`; PV/slack get generator `P, Q`).
4. **PF branch:** re-injects the OPF generator setpoints into the case, then `runpf(mpctry)` → records:
   - input `Xpf{}` = `[−P_load(+P_gen for PV), −Q_load, V(for PV/slack), bus_type]`
   - output `Y_polarpf{}` = `[P, Q, V, θ]` with zeros for the "known" quantities per bus type.
5. **Edges (once):** builds the nodal **admittance matrix** `Ybus = makeYbus(baseMVA, bus, branch)`, then:
   ```matlab
   [i,j,s] = find(Ybus);
   edge_index = [i,j];
   edge_attr  = [real(s), imag(s)];   % [G conductance, B susceptance]
   ```
   ⚠️ This uses the **full Ybus sparsity pattern including the diagonal**, so `edge_index` contains **self-loops** whose `[G,B]` carry each bus's self/shunt admittance.
6. Saves (all `-v7.3` MATLAB, i.e. HDF5):
   - OPF: `Xopf.mat`, `Y_polar_opf.mat`, `edge_index_opf.mat`, `edge_attr_opf.mat`
   - PF: `X.mat` (contains `Xpf`), `Y_polar.mat` (contains `Y_polarpf`), `edge_index.mat`, `edge_attr.mat`

> The `System.m` files under `13_Power_system/<grid>/` are the MATPOWER base cases (bus/branch/gen data) each grid is built from.

---

## 3. Data format (the `.mat` files)

**Trust the code/MATLAB over the README** — the README lists node features as `[Pg−Pd, Qg−Qd, V, θ, N_loads, N_gen]`, but the shipped arrays produced by `gendataopf.m` are 4 columns:

| Array | Shape (per graph) | Columns |
|-------|-------------------|---------|
| `Xpf` (PF input) | `(N, 4)` | `[P, Q, V, bus_type]` — 4th col is **bus_type**, not angle |
| `X` (OPF input) | `(N, 4)` | `[P, Q, 0, bus_type]` (V column is all zeros in OPF input) |
| `Y_polarpf` / `Y_polar` (targets) | `(N, 4)` | `[P, Q, V, θ]`, with **0 where the quantity is "known"/not predicted** for that bus type |
| `edge_index` | `(2, E)` (after load) | branch list from `find(Ybus)`, **1-based** in the file |
| `edge_attr` | `(E, 2)` | `[G, B]` = conductance, susceptance |

Per-bus-type zero pattern in the targets (this is what the `mask = (Y != 0)` later keys off):
- **PQ:** `[0, 0, V, θ]` → predict V, θ.
- **PV:** `[P, Q, 0, θ]`-ish (generator P/Q, angle known-varies).
- **slack:** `[P, Q, 0, 0]` → predict generator P, Q.

---

## 4. Dataset loader (`powergrid.py`)

`class PowerGrid(InMemoryDataset)` — selected by the `datatype` string. Relevant node-regression branches:

- **`node`** (Power Flow): reads `X['Xpf']`, `Y['Y_polarpf']`, `edge_index.mat` (via `scipy.io.loadmat`), `edge_attr.mat`. Keeps **all 4 input columns** → `input_dim = 4`.
- **`nodeopf`** (Optimal Power Flow): reads `Xopf` but keeps **only columns `[0, 1, 3]`** = `[P, Q, bus_type]` (drops the all-zero V column) → `input_dim = 3`.

Common processing for both:
```python
edge_index = (edge_index - 1).T.long()          # MATLAB 1-based → 0-based, to (2, E)
edge_attr  = F.normalize(edge_attr, dim=0)        # L2-normalize each column across all edges; edge_dim = 2

# global max-abs normalization over the WHOLE dataset
maxsX = max(|concat all X|, dim=0);  maxsY = max(|concat all Y|, dim=0)
x = X_i / maxsX
y = Y_i / maxsY
mask = (Y_i != 0)                                 # which entries are actually predicted
data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr, maxs=maxsY, mask=mask)
```
- `maxs` is stored per-`Data` so predictions can be **de-normalized** at test time (`pred * maxs`).
- Processed tensors are cached under `processed_node/` or `processed_nodeopf/` (also `processed_b/`, `processed_r/`, `processed_m/` for the graph-level cascade tasks that share this class).
- The `binary`/`regression`/`multiclass` branches instead read `Bf/Ef/blist/of_*/exp` and handle line **contingencies** (edges with all-zero features are dropped) — that's the graph-level cascade task, not node regression.

---

## 5. Models (`gnn/model.py`)

Factory `get_gnnNets(...)` builds one of: `GCN`, `GAT`, `GIN`, `TRANSFORMER`, or the `NNConv`-based `GNN_basic`.

- Shared base `GNN_basic`: `num_layers` message-passing conv layers → for **node tasks** apply a per-node MLP to the embeddings and return them directly (no pooling); for **graph tasks** apply a readout pool (`mean`/`sum`/`max`/`cat_max_sum`) then MLP.
- Layer types:
  - **GCN** — `GCNConv`, ignores edge features (only valid for no/1-D edge features).
  - **GAT** — `GATConv(edge_dim=2)`.
  - **GIN** — `GINEConv` with an internal MLP, `edge_dim=2`.
  - **TRANSFORMER** — `TransformerConv(heads=4, edge_dim=2, concat=False)`.
  - **base** — `NNConv` (edge-conditioned), multiplies `edge_attr * edge_weight`.
- `_argsparse` is a flexible input parser: accepts a `Data`, or `(x, edge_index[, edge_attr[, batch]])`, and fabricates default `edge_attr`/`edge_weight`/`batch` when missing.
- For node PF/OPF: `forward` returns `self.mlps(emb)` (per-node predictions of dim 4).

---

## 6. Training & evaluation (`train_gnn.py`)

`class TrainModel`:
- **Loss** (`__loss__`): `F.mse_loss` for regression. For node tasks the loss is computed **only on masked entries**:
  ```python
  labels[~mask] = 1e-7; logits[~mask] = 1e-7
  loss = mse(logits[data.mask], labels[data.mask])
  ```
- **Optimizer:** Adam + `ReduceLROnPlateau(mode='min', factor=0.1, patience=10)`.
- **Early stopping:** on eval loss (after half the epochs); **best model** selected by highest **R²**.
- **Test:** loads `<name>_best.pth`, de-normalizes (`denpreds = preds * batch.maxs`), reports MSE loss + R² (via `sklearn.metrics.r2_score`), and writes per-bus `[V, θ, Pg, Qg]` targets vs preds to an Excel `summary<...>.xlsx` (one sheet Data, one sheet Metrics).
- **`__main__`** runs a full **benchmark grid-search**: grids × tasks(`node`, `nodeopf`) × models(`gin`, `gcn`, `gat`, `transformer`) × seeds `[0,100,300,700,1000]` × hidden `{8,16,32}` × layers `{1,2,3}`. Model checkpoints named `<grid>_<model>_<task>_<L>l_<H>h_<seed>s`.

---

## 7. Data loading & splits (`gendata.py`)

- `get_dataset(...)` → instantiates `PowerGrid(root, name, datatype)`.
- `get_dataloader(...)` → **random split** (default `data_split_ratio = [0.98, 0.01, 0.01]`, seed-controlled `torch.randperm`) into train/val/test loaders; **test batch_size = 1**. (Optionally uses a supplied `split_indices` if `random_split_flag=False`.)

---

## 8. Config & defaults (`parser_utils.py`)

Key defaults:
- `datatype=nodeopf`, `dataset_name=ieee118`, `model_name=transformer`
- `hidden_dim=20`, `num_layers=3`, `dropout=0.1`, `readout=mean`, `edge_dim=2`
- `lr=1e-3`, `weight_decay=5e-5`, `num_epochs=30`, `batch_size=8`
- split `train/val/test = 0.98/0.01/0.01`

`get_data_args` auto-derives from the loaded dataset:
- `num_node_features = x.size(1)` (4 for PF, 3 for OPF)
- `num_classes = y.size(1)` = **4** for node tasks (the 4 predicted quantities)
- `edge_dim = edge_attr.size(1)` = 2

There is also an **`--unseen`** flag ("split the dataset into data and unseen data and test on unseen data") — a built-in hook toward generalization testing.

Run examples:
```bash
python code/train_gnn.py --model_name transformer --datatype node    --dataset_name ieee118
python code/train_gnn.py --model_name gat         --datatype nodeopf --dataset_name uk
```

---

## 9. Post-processing / plotting

- `plotnodelevel.py` — reads the best-model `summary*.xlsx` files and computes/plots per-quantity MSE (`V`, `θ`, `Pg`, `Qg`) across grids and tasks. **Note:** it hard-codes Windows absolute paths and a `path_best_results` dict — needs editing to run elsewhere.
- `Postprocess.py` / `Postprocess_grid.py` — additional result aggregation.

---

## 10. How it compares to ENGAGE

| | **PowerGraph-Node** | **ENGAGE** |
|---|---|---|
| Grid type | **Transmission** (IEEE-24/39/118, UK) | **Distribution** (SimBench LV/MV) |
| Data source | MATPOWER `.mat` (v7.3) | pandapower/SimBench JSON |
| Node input | `[P, Q, V, bus_type]` (PF) / `[P, Q, bus_type]` (OPF) | `[Slack?, PV?, PQ?, p, q, vm, va]` with NaN masking |
| Unknown handling | `mask = (Y != 0)` + max-abs normalization | per-bus-type **NaN** masking + **per-unit** features |
| Edge attributes | `[G, B]` from Ybus (**includes self-loops**) | `[trafo?, r_pu, x_pu, sc_voltage]` |
| Task(s) | PF and OPF node regression | AC PF node regression |
| Metric | R² (masked, de-normalized) | NRMSE-range |
| Evaluation scope | **within-grid** random split (has `--unseen` hook) | **cross-grid** generalization (Cross-Context + OOD) |
| Headline contribution | benchmark datasets + baseline GNNs (+ explainability in sibling repos) | **generalization score (g-score)** relating MMD graph-distance to error |

**Bottom line for a transmission-generalization study:** PowerGraph-Node provides ready transmission PF/OPF graph data plus baseline GNN architectures, but trains and tests within a single grid. Combining it with ENGAGE's cross-grid evaluation (train on some transmission grids, test on held-out ones, quantify with MMD / g-score) is the natural experiment. The integration work is mainly **reconciling the two conventions**: node-feature layout, target/mask semantics, normalization (max-abs vs per-unit), and edge-feature definitions (`[G,B]` vs `[trafo?, r_pu, x_pu, sc_voltage]`).

---

## 11. ⚠️ Gotchas & things to watch

1. **README vs reality:** node-feature description in the README (`[Pg−Pd, Qg−Qd, V, θ, N_loads, N_gen]`) does **not** match the shipped 4-column arrays (`[P, Q, V, bus_type]`). Trust `gendataopf.m` / `powergrid.py`.
2. **PF vs OPF input dim differs:** PF uses 4 columns, OPF uses 3 (drops the zero V column). Any code assuming a fixed input dim will break when switching tasks.
3. **Self-loops in `edge_index`:** edges come from `find(Ybus)` including the diagonal, so each node has a self-edge carrying self-admittance. Some GNN layers treat self-loops specially — be aware when swapping architectures or computing graph descriptors.
4. **1-based → 0-based:** the loader subtracts 1 from `edge_index`; if you regenerate data, keep the convention consistent.
5. **`mask = (Y != 0)` is heuristic:** it assumes a genuinely-predicted quantity is never exactly zero. A true physical zero would be masked out incorrectly. This differs fundamentally from ENGAGE's explicit NaN-by-bus-type masking.
6. **Global max-abs normalization:** computed over the entire dataset; store/reuse `maxs` for de-normalization. If you mix grids or add new data, the normalization constants change.
7. **`.mat` v7.3 loading:** node arrays use `mat73`; `edge_index`/`edge_attr` use `scipy.io.loadmat`. Both dependencies are required.
8. **Datasets not in the repo:** must be downloaded from figshare; the repo only has code + MATPOWER cases.
9. **Hard-coded Windows paths:** `gendataopf.m` and `plotnodelevel.py` contain absolute `C:\Users\avarbella\...` paths that must be edited to run.
10. **`requirements.txt` is effectively empty** — you must assemble the environment yourself (PyTorch, torch-geometric, mat73, scipy, scikit-learn, pandas, xlsxwriter/openpyxl, matplotlib/seaborn).
11. **Within-grid split only:** default 98/1/1 random split means train and test come from the *same* grid's loading conditions — do **not** mistake the reported R² for cross-grid generalization.
