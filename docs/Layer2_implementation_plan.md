# Layer 2 — Detailed Implementation Plan (the clean, single-pipeline solution)

Status: **IMPLEMENTED (Steps 1–7) and run end-to-end.** Companion to `PowerGraph_to_ENGAGE_design_decisions.md` (the *why*), `Experimental_Design_transmission_GNN_generalization.md` (the *experiment*), and `Pipeline_Report.md` (the *as-built* report with flow diagram + run guide). This file is the *how*: every file we create or modify, what it contains, how it connects to the rest of the pipeline under the **new mental model**, the design decision behind each step, and how it serves the final goal.

> **As-built note (differs from the original plan below).** Rather than *modifying* ENGAGE files in place (`graph_gen.py`, `cross_context_experiment.py`, …), the final implementation is **fully self-contained** (Decision 12): ENGAGE's extractors are **vendored** into `engage_contract.py`; the experiment core is a single new `experiments.py`; MMD is a new `mmd_utils.py`; training/metrics are `training_utils.py`. The step→branch mapping actually shipped is: `step-1-grid-conversion`, `step-2-grid-loader`, `step-3-data-generation`, `step-4-model-zoo`, `step-5-experiments`, `step-6-validation`, `step-7-harvest-contingencies`. `experiments.py` also gained `--save_models` (Decision 13) writing `cc_<model>_<grid>.pt` / `ood_<model>_heldout_<grid>.pt`, and a **two-flavour g-score**: the cross-context `gscore.csv` (+ small-N `gscore_smallN.csv`) and the better-posed **OOD g-score** `gscore_ood.csv` via `compute_ood_gscores` (per model over held-out grids, distance = mean Laplacian-MMD from each held-out grid to its training grids, no trim — Decision 13). See the corrected file map at the bottom.

## The new mental model (drop "map two repos")
We build **one clean pipeline**: *ENGAGE's methodology applied to PowerGraph's transmission grids.* We do **not** interoperate with PowerGraph's generated data format or its training code. We borrow only three things: (1) PowerGraph's **grid models** (`System.m`) and **real demand** (`hourlyDemandBus.mat`), (2) pandapower as the **solver**, and (3) standard `torch_geometric` **layers**. Everything else is ENGAGE's existing machinery, which already implements the generalization study correctly.

**Final goal it serves:** measure how GNN architectures generalize to an **unseen transmission grid** for the AC power-flow node task, quantified by ENGAGE's g-score (NRMSE vs. topological distance via MMD), and benchmarked against PowerGraph's within-grid regime.

### What ENGAGE already gives us (reuse, do not rebuild)
Confirmed by reading the code:
- `graph_gen.py::get_node_features` / `get_edge_features` — produce the exact ENGAGE `Data` contract (`x=(N,7)`, `edge_index=(2,2E)`, `edge_attr=(2E,4)`, `y=(N,4)`, `dc_pf=(N,4)`) with **per-unit** edge features and **bus-type NaN masking**.
- `training_utils.py::evaluate_mmd` — the **correct** MMD: it builds one networkx graph *per sample* and calls ggme's `degree_distribution` / `normalised_laplacian_spectrum` over the **distribution** of graphs. This is what makes the g-score well-posed — and it is *not* the degenerate single-vector MMD from `engage_pg` v2.
- `training_utils.py::train / test / test_dc_pf / weighted_mse_loss / nrmse_range / get_generalization_score` — ENGAGE's training loop, range-NRMSE metric, and g-score.
- `cross_context_experiment.py` / `out_of_distribution_experiment.py` — the CC (pairwise train/test) and OOD (leave-one-out) drivers.
- `graph_utils.py::get_pyg_graphs` / `get_dataset` — load datasets from `data_dir/<grid>/train/dataset.pt`.

**Key architectural insight:** the entire downstream (models, training, MMD, g-score, CC/OOD) is grid-agnostic. It only cares that (a) datasets live at `data_dir/<grid>/train/dataset.pt`, and (b) the grid list comes from a function currently called `get_dist_grid_codes`. So Layer 2 is mostly a **new data-generation front-end** plus **model additions**; the experiment core barely changes.

---

# Implementation steps

## Step 0 — Environment + evaluation submodule
**Files:** `environment.yaml` (existing), `ggme/` (submodule), `requirements.txt` (reuse v2's).
**What:** create the ENGAGE conda env; populate the **`ggme`** submodule (needed by `training_utils.evaluate_mmd`); confirm `pandapower`, `torch`, `torch_geometric`, `omegaconf`, `networkx` import.
**How it connects:** `training_utils.py` appends `ggme/src` to `sys.path` and imports `evaluate_mmd`, `degree_distribution`, `normalised_laplacian_spectrum`. Nothing else works for the g-score without it.
**Design decision:** **Use ENGAGE's `ggme` `evaluate_mmd`, not v2's custom `compute_laplacian_spectrum`/`compute_mmd`.** *Reason:* v2's version collapses each grid to a single descriptor vector and saturates the Gaussian kernel (`sigma_laplacian=1e-2` → every pair = √2). ENGAGE's builds a per-graph descriptor over the whole distribution, which is the statistically valid MMD (Decision 9). `powerdata-gen` is **not** required for Layer 2 (Route B replaces it), which also sidesteps the empty-submodule blocker.
**Serves goal:** a correct, non-degenerate topological-distance measure is the backbone of the g-score.

## Step 1 — One-time grid conversion (`System.m` → `.mat` → pandapower)
**New files:** `transmission/convert_cases.m` (Octave), `transmission/cases/IEEE24.mat`, `IEEE39.mat`, `IEEE118.mat`, `UK.mat` (committed outputs).
**What:** run each PowerGraph `System.m` in Octave and `save('-v7', ...)` the `mpc` struct; commit the resulting `.mat` so downstream needs only pandapower.
```matlab
% transmission/convert_cases.m
cases = {'IEEE24','IEEE39','IEEE118','UK'};
for i = 1:numel(cases)
    addpath(fullfile('PowerGraph-Node','13_Power_system',cases{i}));
    mpc = System();
    save('-v7', fullfile('transmission','cases',[cases{i} '.mat']), 'mpc');
    rmpath(fullfile('PowerGraph-Node','13_Power_system',cases{i}));
end
```
**How it connects:** feeds Step 2's loader; nothing else depends on Octave afterwards.
**Design decisions:** **D4** (convert PowerGraph's *own* `System.m` for all four grids, not pandapower built-ins) and **D5** (Octave, once, commit the `.mat`). *Reason:* fidelity — PowerGraph may have modified the standard cases; Octave uses the real MATLAB semantics; committing `.mat` makes the repo self-contained (no Octave/MATLAB needed later). Verified in-session: Octave 6.4.0 converts IEEE24 in seconds.
**Serves goal:** guarantees the grids are *identical* to PowerGraph's, so cross-grid results are attributable to topology, not to case discrepancies.

## Step 2 — Transmission grid loader (`transmission_grids.py`)
**New file:** `transmission_grids.py`.
**What it includes:**
- `get_transmission_grid_codes() -> ['IEEE24','IEEE39','IEEE118','UK']` — the transmission analogue of `graph_utils.get_dist_grid_codes`.
- `load_case(code) -> pandapower net` via `pandapower.converter.from_mpc('transmission/cases/<code>.mat')`.
- `load_hourly_demand(code) -> np.ndarray (N_bus, ~8760)` from `hourlyDemandBus.mat` (`scipy.io.loadmat`).
- Post-import fixups: map MATPOWER bus types (3/2/1 → ext_grid/gen/load), ensure branches with `tap≠1`/phase shift become **pandapower trafos** (so ENGAGE's `trafo?`/`sc_voltage` are set), carry `baseMVA`/`baseKV`.
**How it connects:** used by Step 3's generator; `get_transmission_grid_codes()` replaces `get_dist_grid_codes()` in the experiment scripts (Step 5).
**Design decisions:** **D1** (raw `System.m` source), **D4**, and the **semantic-mapping contract** (bus-type one-hot, trafo vs line, per-unit base). *Reason:* ENGAGE's `get_node_features`/`get_edge_features` rely on pandapower's element tables (`net.gen`, `net.ext_grid`, `net.line`, `net.trafo`) and `net.sn_mva`; the loader must populate these correctly or the feature contract silently breaks.
**Serves goal:** turns the source grids into the pandapower objects ENGAGE's feature extractors already understand — the single clean representation.

## Step 3 — Data-generation engine (`transmission_graph_gen.py`)  ← the heart of Layer 2
**New file:** `transmission_graph_gen.py` (a transmission sibling of `graph_gen.py`, reusing its `get_node_features`/`get_edge_features`).
**What it includes (per grid):**
1. `net = transmission_grids.load_case(code)`.
2. `demand = transmission_grids.load_hourly_demand(code)` (Route B).
3. **Contingency sampler** `sample_contingency(net, depth)` — pick N-1, then N-2/N-k lines to set `in_service=False`; reject islanding via a connectivity check; retune outage probabilities for meshed grids. *Optionally* draw the outage set from **PowerGraph-Graph** harvested contingencies.
4. **The re-solve loop:**
```python
for t in sampled_hours:
    for c in sampled_contingencies:
        net_i = deepcopy(net)
        apply_contingency(net_i, c)                 # lines out of service
        set_demand(net_i, demand[:, t])             # real hourly demand (Route B)
        to_per_unit_inputs(net_i)                    # P/Q -> per-unit (÷ baseMVA)  [Decision 9]
        try:
            pp.runpp(net_i)                          # AC power flow (Newton-Raphson)
        except LoadflowNotConverged:
            continue                                 # filter
        if violates_limits(net_i): continue          # voltage/loading filter
        X, Y = get_node_features(net_i)              # ENGAGE node contract (bus-type NaN mask)
        A, E = get_edge_features(net_i)              # ENGAGE per-unit edge contract
        dc_pf = rundcpp_features(net_i)              # ENGAGE dc baseline
        dataset.append(Data(x, edge_index, edge_attr, y, dc_pf))
    # save data_dir/<code>/train/dataset.pt + dataset_src.csv
```
**How it connects:**
- **Imports** `get_node_features`, `get_edge_features`, and the `dc_pf` block from `graph_gen.py` (refactor those three into importable helpers so both generators share them — no duplication).
- **Writes** to the exact layout `graph_utils.get_pyg_graphs` expects: `data_dir/<code>/train/dataset.pt`. That single convention is why the rest of the pipeline needs no changes.
- **Replaces** `powerdata_gen.build_datasets` — Route B does its own demand + PF, so the empty `powerdata-gen` submodule is not needed.
**Design decisions:** **D2** (Route B real demand), **D10** (topology distribution via contingency + `pp.runpp` re-solve), **D11** (optional PowerGraph-Graph contingencies), **D6** (ENGAGE masking/`dc_pf` via reused extractors), **D9** (convert node P/Q to per-unit for cross-grid comparability). *Reasons:*
- A topology change invalidates all node values, so each contingency is a **fresh AC PF solve**, not a data edit (that is why we need the re-solvable pandapower model, not PowerGraph's frozen `.mat`).
- Contingencies give each grid a **distribution of topologies** → the MMD/g-score become well-posed (the whole point vs Layer 1).
- Per-unit node inputs remove the units/scaling artifact that would otherwise contaminate cross-grid NRMSE.
**Serves goal:** produces the per-grid *clouds of graphs* that the generalization study consumes, all in one consistent ENGAGE representation.

## Step 4 — Model zoo under ENGAGE's interface (`models.py`)
**Modify:** `models.py` (fix + extend).
**What it includes:**
- Keep `GCN`, `ARMA_GNN` (unchanged, correct reference).
- **Fix** the appended `GIN`/`GAT`/`TransformerGNN` so they (a) actually pass edge features into the conv layers (`GINEConv`, `GATConv(edge_dim=...)`, `TransformerConv(edge_dim=...)`), (b) fix the Transformer hidden-dim mismatch (`heads`×`out_channels` with `concat`), and (c) implement the ENGAGE **`inference()`** per-bus-type re-injection like `GCN.inference`.
- **Add** `NNConv` edge-conditioned model.
- All conform to the ENGAGE contract: constructor `__init__(input_dim=..., ...)`, `forward(data)` returning `(N,4)`, NaN-safe, edge-aware, with `inference()`.
**How it connects:** registered in each experiment's `model_classes` dict (Step 5); instantiated as `model_class(input_dim=...)` exactly as `cross_context_experiment.evaluate_performance` already does (`input_dim = next(iter(loader_train)).x.shape[1]`).
**Design decision:** **D7** (full ENGAGE+PowerGraph zoo: `GCN, ARMA_GNN, GAT, GIN, TRANSFORMER, NNConv`, all under ENGAGE's interface). *Reason:* a fair architecture comparison requires every model under one identical data contract, mask, and metric; edge-awareness must be genuine (v2's versions computed `edge_emb` but never used it) so the ablation "does edge information help on transmission grids?" is valid.
**Serves goal:** the architectures whose generalization we are actually comparing.

## Step 5 — Experiment drivers (`cross_context_experiment.py`, `out_of_distribution_experiment.py`)
**Modify:** both scripts, minimally.
**What changes:**
- Replace `grids_to_compare = get_dist_grid_codes(scenario)` with `get_transmission_grid_codes()`.
- Extend `model_classes` to `{'gcn':GCN, 'arma_gnn':ARMA_GNN, 'gat':GAT, 'gin':GIN, 'transformer':TRANSFORMER, 'nnconv':NNConv}`.
- Leave the pairwise-permutation CC loop, the leave-one-out OOD loop, `evaluate_mmd`, and `get_generalization_score` **unchanged**.
**How it connects:** these call `get_dataloaders → get_dataset → get_pyg_graphs`, which read the datasets Step 3 wrote. `evaluate_mmd` runs over the per-grid graph distributions produced in Step 3.
**Design decision:** **D8** (two-layer plan; this is the well-posed Layer 2 experiment). *Reason:* reusing ENGAGE's CC/OOD + g-score unchanged maximizes correctness and comparability; the only transmission-specific inputs are the grid list and the model set.
**Serves goal:** directly outputs the cross-grid NRMSE transfer matrix, the MMD matrix, and per-architecture g-scores — the study's results.

## Step 6 — Validation gates (before trusting any result)
**New file:** `transmission/validate.py`.
**What it checks:**
1. **Conversion fidelity:** run `pp.runpp` on each converted net at a base demand and compare solved `V/θ` to a known PowerGraph PF solution (tolerance check).
2. **MMD non-degeneracy:** confirm `evaluate_mmd` gives a *spread* of values across grids/contingency depths (not a constant √2); retune `sigma_degree`/`sigma_laplacian` if needed.
3. **Contract sanity:** shapes `x=(N,7)`, `edge_attr=(2E,4)`, `y=(N,4)`; NaN mask matches bus types; per-unit ranges reasonable.
4. **Connectivity:** no islanded samples slipped through.
**Design decisions:** **D5** (validate every conversion), **D9** (MMD must be non-degenerate). *Reason:* the whole Layer 2 argument is that its numbers are *valid*; these gates enforce that before we report anything.
**Serves goal:** guarantees the generalization numbers reflect physics/topology, not bugs or scaling artifacts.

## Step 7 — Run + analysis
**Outputs:** CC results (`results_cc.csv`), MMD (`results_cc_mmd.csv`), g-scores (`results_cc_gen_stats.csv`), OOD equivalents.
**What:** run CC and OOD across all six architectures with multiple seeds; assemble the **NRMSE transfer matrix**, the **g-score ranking**, the **edge-awareness ablation**, and the **within-grid vs unseen-grid** comparison against PowerGraph.
**Serves goal:** answers RQ2/RQ2a–c from the experimental-design doc.

---

# File map (new mental model)
**Corrected AS-BUILT file map** (what actually shipped; supersedes the original plan's "modify ENGAGE in place" rows):

| File | New/Modified | Role | Connects to |
|---|---|---|---|
| `transmission/convert_cases.m` | new | Octave `.m`→`.mat` (once) | `transmission_grids.load_case` |
| `transmission/cases/*.mat` | new (committed) | converted MATPOWER cases | pandapower `from_mpc` |
| `transmission_grids.py` | new | load nets + demand; grid list | generator, experiments |
| `engage_contract.py` | new (**vendored** ENGAGE) | contingency-aware `get_node_features`/`get_edge_features` | imported by the generator |
| `transmission_graph_gen.py` | new | Route B + contingency **re-solve** → `Data` | uses `engage_contract`; writes `data_dir/<grid>/<split>/dataset.pt` |
| `contingency_harvest.py` | new | Step 7: harvest real outages from PowerGraph-Graph | optional source for the generator |
| `models.py` | new | six edge-aware GNNs + shared `inference()` | `MODELS` registry, imported by `experiments.py` |
| `training_utils.py` | new | training loop, per-quantity NRMSE, DC baseline, g-score | `experiments.py` |
| `mmd_utils.py` | new | distribution-based MMD (non-degenerate) | `experiments.py` |
| `experiments.py` | new | single driver: CC + OOD + MMD + DC + g-score (+ `--save_models`) | datasets + `MODELS` |
| `validate.py` | new | conversion/contract/masking/topology/MMD gates | all of the above |
| `requirements.txt` | new | pinned deps (incl. optional `mat73` for Step 7) | environment |

# Dependency / sequencing
Step 0 → Step 1 → Step 2 → Step 3 (needs 1,2) → Step 4 (parallel with 1–3) → Step 5 (needs 3,4) → Step 6 (gates 1–5) → Step 7. The generator (Step 3) and the models (Step 4) are the only substantial new code; everything else is small glue or reuse.

# What we explicitly do NOT do (new mental model)
- No conversion of PowerGraph's generated `X/Y/edge_index/edge_attr` `.mat` arrays (lossy — Decision 1).
- No PowerGraph training code, `mask=(Y!=0)`, max-abs normalization, or R² (Decision 6/9).
- No merging of the two model zoos or the v2 custom MMD script (superseded by ENGAGE's `evaluate_mmd`).
