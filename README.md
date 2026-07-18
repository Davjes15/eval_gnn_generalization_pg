# eval_gnn_generalization_pg

Evaluating the **generalization of GNN architectures for the AC power-flow (PF) node task on transmission grids**.

## Goal
Measure how well graph neural networks trained for node-level AC power flow generalize to **unseen transmission topologies** (and unseen grids), and benchmark this against [PowerGraph](https://github.com/PowerGraph-Datasets), which only trains and tests *within* a single fixed-topology grid. Generalization is quantified with ENGAGE's **g-score** (NRMSE vs. topological distance via MMD).

Because AC power flow is deterministic physics, the value of a learned surrogate is **amortization/speed** across many cases (contingency screening, planning, real-time what-ifs) and **robustness to topology change** — so the primary axis studied is generalization **across contingencies / topological variations**, with transfer between structurally different grids kept as a scientific stress test.

## Approach (one clean pipeline)
ENGAGE's generalization methodology applied to PowerGraph's transmission grids. We reuse:
- **Grid models + real demand** from PowerGraph (`System.m`, `hourlyDemandBus.mat`).
- **pandapower** as the AC power-flow solver.
- **ENGAGE**'s data contract, masking, training loop, MMD and g-score.

Each grid is turned into a **distribution of topologies** by sampling credible N-1/N-k contingencies and re-solving AC power flow, so the MMD/g-score are well-posed.

## Grids
IEEE24, IEEE39, IEEE118, and the UK 29-bus system (PowerGraph's own `System.m` cases).

## Task & data contract
Node-level AC PF state estimation — predict per-bus `[P, Q, V, θ]`.
- `x`: `(N, 7)` = `[Slack?, PV?, PQ?, p_mw, q_mvar, vm_pu, va_degree]` (unknown inputs masked by bus type)
- `edge_index`: `(2, 2E)`
- `edge_attr`: `(2E, 4)` = `[transformer?, r_pu, x_pu, sc_voltage]`
- `y`: `(N, 4)` = `[p_mw, q_mvar, vm_pu, va_degree]`
- `dc_pf`: `(N, 4)` DC power-flow baseline

## Model zoo
`GCN`, `ARMA_GNN` (ENGAGE) plus `GAT`, `GIN`, `TRANSFORMER`, `NNConv` (PowerGraph), all under one ENGAGE-style interface (edge-aware, with per-bus-type known-value re-injection).

## Repository layout
```
eval_gnn_generalization_pg/
├── README.md                     # this file — start here
├── docs/                         # design & experiment documents (the "why")
│   ├── PowerGraph_to_ENGAGE_design_decisions.md
│   ├── Experimental_Design_transmission_GNN_generalization.md
│   ├── Layer2_implementation_plan.md
│   └── PowerGraph-Node_deep_dive.md
└── transmission/                 # grid conversion + data generation
    ├── convert_cases.m           # Step 1: System.m -> .mat (Octave, one-time)
    ├── cases/                     # Step 1 output: IEEE24/IEEE39/IEEE118/UK .mat
    └── README.md                 # per-step instructions for this folder
```

## How to run the experiments (step by step)
This guide is built up **incrementally, one implementation step per branch**. Each
step below is marked with its status so you always know what is runnable today.

> Branches are *stacked*: `step-2` builds on `step-1`, `step-3` on `step-2`, etc.
> To review/run a given step, check out its branch:
> `git fetch origin && git checkout step-1-grid-conversion`

### Prerequisites (all steps)
- **Python 3.10+**
- A checkout of **PowerGraph-Node** (for the raw `System.m` grids and hourly demand):
  https://github.com/PowerGraph-Datasets/PowerGraph-Node
- Python packages (installed per step as they become needed):
  `pandapower`, `torch`, `torch_geometric`, `scipy`, `numpy`, `pandas`,
  `networkx`, `omegaconf`. A pinned `requirements.txt` is added in a later step.

### Step 1 — Convert the grids  ✅ available on `step-1-grid-conversion`
Turns PowerGraph's `System.m` files into committed `.mat` cases. You normally only
run this once (the `.mat` files are committed, so you can skip straight to Step 2).
Needs **Octave** only.
```bash
# install Octave (free, no license):  sudo apt-get install -y octave   # or: brew install octave
export POWERGRAPH_NODE_DIR=/absolute/path/to/PowerGraph-Node/13_Power_system
octave --no-gui --eval "cd transmission; convert_cases"
```
Full details, expected output, and a verification snippet: see
[`transmission/README.md`](transmission/README.md).

### Step 2 — Load grids into pandapower  ⏳ coming on `step-2-grid-loader`
### Step 3 — Generate the datasets (contingencies + AC power-flow re-solve)  ⏳ `step-3-data-generation`
### Step 4 — The model zoo  ⏳ `step-4-model-zoo`
### Step 5 — Run the experiments (cross-context + out-of-distribution)  ⏳ `step-5-experiments`
### Step 6 — Validation gates  ⏳ `step-6-validation`

Each `⏳` section will be filled in with exact commands, expected output, and
troubleshooting as its branch lands.

## Status
Early stage — **Step 1 (grid conversion) implemented**; Steps 2–6 in progress.
See [`docs/Layer2_implementation_plan.md`](docs/Layer2_implementation_plan.md) for
the full plan and the reasoning behind each step.

## License
TBD.
