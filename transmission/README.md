# `transmission/` — grid conversion & data generation

This folder holds everything that turns PowerGraph's raw transmission grids into
ready-to-use data for the experiments.

## Step 1 — one-time grid conversion (`convert_cases.m`)  ← this branch

### What it does
Runs each PowerGraph-Node `System.m` (a MATLAB/MATPOWER case function) in GNU
Octave and saves the returned `mpc` struct as a portable `.mat` file under
`transmission/cases/`. These `.mat` files are **committed**, so downstream steps
need only Python + pandapower — no MATLAB or Octave.

### Why (design decisions)
- **D4** — convert PowerGraph's *own* `System.m` cases (not pandapower's built-in
  IEEE cases) so the four grids are identical to the ones PowerGraph trained on.
- **D5** — Octave (free, no license) runs `System.m` faithfully; convert once and
  commit the `.mat`, keeping the repo self-contained.

See `../docs/PowerGraph_to_ENGAGE_design_decisions.md` and
`../docs/Layer2_implementation_plan.md` (Step 1) for the full rationale.

### Inputs
PowerGraph-Node's case folder, i.e. `.../PowerGraph-Node/13_Power_system/<CODE>/System.m`
for `CODE` in `IEEE24, IEEE39, IEEE118, UK`.

### How to run
You only need Octave for this step (already-committed `.mat` files mean most users
can **skip it**).

```bash
# 1. Install Octave (free, no license):
#      Ubuntu/Debian:  sudo apt-get install -y octave
#      macOS (brew):   brew install octave

# 2. Point to your PowerGraph-Node checkout (folder containing 13_Power_system/):
export POWERGRAPH_NODE_DIR=/absolute/path/to/PowerGraph-Node/13_Power_system

# 3. Run the conversion from the repo root:
octave --no-gui --eval "cd transmission; convert_cases"
```

If `POWERGRAPH_NODE_DIR` is not set, the script looks for a sibling checkout at
`../../PowerGraph-Node-main/13_Power_system` relative to this folder.

### Expected output
```
Converted IEEE24   -> cases/IEEE24.mat   (baseMVA=100, buses=24,  branches=38,  gens=33)
Converted IEEE39   -> cases/IEEE39.mat   (baseMVA=100, buses=39,  branches=46,  gens=10)
Converted IEEE118  -> cases/IEEE118.mat  (baseMVA=100, buses=118, branches=186, gens=54)
Converted UK       -> cases/UK.mat       (baseMVA=100, buses=29,  branches=99,  gens=66)
```
(The UK case prints an Octave warning that the function inside `System.m` is named
`GBreducednetwork`; this is harmless — the struct is still returned and saved.)

### Verify the result
```bash
python3 - <<'PY'
import scipy.io as sio
for c in ['IEEE24','IEEE39','IEEE118','UK']:
    mpc = sio.loadmat(f'transmission/cases/{c}.mat', struct_as_record=False, squeeze_me=True)['mpc']
    print(c, 'baseMVA', mpc.baseMVA, 'bus', mpc.bus.shape, 'branch', mpc.branch.shape, 'gen', mpc.gen.shape)
PY
```

### Outputs (consumed by Step 2)
- `transmission/cases/IEEE24.mat`
- `transmission/cases/IEEE39.mat`
- `transmission/cases/IEEE118.mat`
- `transmission/cases/UK.mat`
