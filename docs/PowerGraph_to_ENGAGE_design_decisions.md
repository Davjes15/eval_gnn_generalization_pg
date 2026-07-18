# Design Decisions — Using PowerGraph Source Data in ENGAGE's Data Generation

Status: **agreed plan; partially probed by `engage_pg` v2.** This document records the decisions made so far and the reasoning behind each, so the approach is explicit before any code is written. See the companion `Experimental_Design_transmission_GNN_generalization.md` for the per-layer research questions, setup, and methodology.

> **Note on `engage_pg` v2 (the uploaded fork):** it implemented the **Level 2** path (evaluate PowerGraph-format, PowerGraph-trained models with ENGAGE's g-score/MMD) that this doc had *rejected* as the primary route — not the chosen Level 1 / Route B. It is a useful first probe (the g-score/MMD harness is reusable) but its cross-grid numbers are not yet valid: it keeps PowerGraph's per-grid max-abs normalization and has one topology per grid, and its Laplacian MMD is degenerate. Decisions 8–11 below re-scope the work into two layers to fix this.

## Goal
Use **PowerGraph's transmission grids as the input to ENGAGE's data-generation pipeline**, so that operating points are produced in **ENGAGE's `Data` format** and can be run through ENGAGE's cross-grid generalization experiments (Cross-Context, Out-of-Distribution, g-score / MMD). We explicitly do **not** want PowerGraph's own graph output format. The ultimate objective is a **study of GNN-architecture generalization to unseen transmission grids**, benchmarked against PowerGraph's within-grid results.

---

## Decision 1 — Source = PowerGraph's raw grids, not their generated `.mat` graphs
**Decision:** Feed ENGAGE the PowerGraph **MATPOWER `System.m` case files** (`13_Power_system/<grid>/System.m`) as the reference grids, and let **ENGAGE's `get_node_features` / `get_edge_features` build the PyG `Data`**. Do not use PowerGraph's `gendataopf.m` output or their figshare `.mat` arrays as the graph objects.

**Why:**
- The user wants everything in ENGAGE format (per-unit features, bus-type one-hot, NaN masking, `dc_pf`). ENGAGE's feature extractors produce exactly that contract.
- PowerGraph's generated `.mat` files carry a *different* contract (`[G, B]` edge features, max-abs normalization, `mask=(Y!=0)`, no per-unit `r/x`, no `trafo?` flag, no `dc_pf`). Converting those arrays would be lossy and would still require reconstructing edge attributes from the grid anyway.
- Working from the raw grids keeps a single, clean data contract (ENGAGE's) end-to-end.

**Three candidate "source levels" that were considered:**
- **Level 1 — raw grids (`System.m`) + real demand (`hourlyDemandBus.mat`), regenerate.** ⭐ chosen.
- **Level 2 — figshare generated `.mat` (`X/Y/edge_index/edge_attr`), convert arrays.** Rejected: lossy, awkward, must rebuild edge attributes regardless.
- **Level 3 — pandapower built-in cases + synthetic sampling.** Rejected as the primary path: least faithful to PowerGraph (see Decision 4).

---

## Decision 2 — Operating points via Route B (real hourly demand), not Route A (synthetic sampling)
**Decision:** Generate operating points by driving each grid with PowerGraph's **measured hourly demand** (`hourlyDemandBus.mat`, shape `(N_bus, ~8760)`), running AC power flow per snapshot, and converting the solved net to ENGAGE `Data`.

```
for t in hours:
    net = load PowerGraph grid (from System.m -> pandapower net)
    set bus loads PD/QD = hourlyDemandBus[:, t]      # real measured demand at hour t
    (optionally apply a chosen contingency / outage)
    run AC power flow (pandapower)                    # solve physics
    Data = ENGAGE.get_node_features(net) + get_edge_features(net)   # ENGAGE format
```

**Why:**
- **Route A (synthetic):** ENGAGE/`powerdata-gen` draws i.i.d. random loads/gen/topology (e.g. total load factor 0.5–1.2, random power factors), then AC-PF-solves. No time axis, no realistic daily/seasonal structure — just a cloud of random valid states.
- **Route B (real demand):** reproduces the *same operating points PowerGraph used* (same grids + same measured demand curves + AC PF) — this is exactly what PowerGraph's `gendataopf.m` does — but re-expressed in ENGAGE format. It carries realistic load patterns and is faithful to PowerGraph's data.
- Route B also **reduces the dependency on the `powerdata-gen` submodule** (still empty/unfetchable here): it needs mainly pandapower + ENGAGE's feature code.

**Note on ENGAGE synthetic data:** it is **not** "one value per bus per hour for a year." That description applies to PowerGraph's demand time series. ENGAGE synthetic data is a fixed count of independent random snapshots (`n_train/n_val/n_test`), with no chronology.

---

## Decision 3 — Grid coverage: all four grids (IEEE24, IEEE39, IEEE118, UK)
**Decision:** Target all four PowerGraph grids so cross-grid generalization experiments have multiple transmission topologies to train/test across.

**Why:**
- ENGAGE's Cross-Context and OOD experiments are inherently multi-grid (train on some, test on held-out). A single grid can't exercise the g-score / MMD machinery.
- The four grids span a useful size range (24 / 39 / 29 / 118 buses).

---

## Decision 4 — Convert PowerGraph's own `System.m` for every grid (do not substitute pandapower built-ins)
**Decision:** Build each pandapower net from PowerGraph's **own** `System.m`, including for IEEE24/39/118 (which pandapower also ships as built-ins). Built-ins are used only as an optional cross-check.

**Why:**
- PowerGraph may have modified the standard IEEE cases (limits, costs, shunts). Using their exact `System.m` guarantees identical topology and parameters, preserving fidelity to their source.
- The UK 29-bus grid is custom and has no built-in equivalent, so it must be converted regardless — converting all four keeps the pipeline uniform.

---

## Decision 5 — Conversion method: Octave (CONFIRMED doable in-session, fast); do the conversion once and commit the `.mat`
**Decision:** Convert `System.m` → `.mat` via **GNU Octave**, then import with **`pandapower.converter.from_mpc`**. Do the conversion **once in this session** and **commit the resulting `.mat` files** to the repo, so no one needs Octave (or MATLAB) later. Validate every converted net by running a power flow and comparing the solved V/θ to a known PowerGraph operating point. A pure-Python `.m` parser remains an unused fallback.

**Verified in this session (2026-07-18):**
- **Octave runs here and is free — no MATLAB/Octave license exists or is needed.** Installed GNU Octave **6.4.0** via `apt` in this session.
- **Conversion is trivial and fast, not time-consuming.** Each PowerGraph `System.m` is a MATLAB *function* that returns the `mpc` struct directly, so we don't even need MATPOWER's `loadcase`/`savecase` — just run the function and save:
  ```matlab
  mpc = System();                     % run the case function -> mpc struct
  save('-v7', 'IEEE24.mat', 'mpc');   % write a MATLAB v7 .mat
  ```
  Tested on IEEE24: produced a valid `.mat` in seconds (baseMVA=100, 24 buses, 38 branches, 33 gens; fields `version/baseMVA/bus/gen/branch/gencost`). ~a few seconds per grid for all four.

**Why:**
- **Octave** interprets the `.m` exactly as MATLAB/MATPOWER would — every field handled correctly, zero interpretation risk. It is the most faithful/robust option, and since it's confirmed working in-session there is no reason to fall back to the more brittle hand-written Python parser (column mapping, per-unit base, tap ratios, service flags are easy to get subtly wrong).
- **Committing the `.mat` outputs** means the repo becomes self-contained: contributors reproduce datasets with only pandapower — no Octave/MATLAB dependency at all.
- **Validation** (re-run PF, compare to a PowerGraph solution) is the real proof of a correct import, regardless of method.

**Answering the license/local-machine question:**
- *In this session:* yes, Octave works — nothing for the user to install or license.
- *On the user's local Mac (only if they ever want to redo it themselves):* install **GNU Octave** for free (`brew install octave`) — **no license required**; MATLAB is *not* needed. But because we commit the converted `.mat` files, the user will **not** need Octave locally at all.

**What is Octave / why UK needs conversion:** Octave is a free, open-source, MATLAB-compatible interpreter. It's used here purely as a format bridge (`.m` → `.mat`) because `from_mpc` reads `.mat`. The IEEE grids have built-in equivalents, but the custom UK grid does not, so its `System.m` must be converted regardless — and Octave handles all four uniformly.

---

## Decision 6 — Masking / training convention: adopt ENGAGE's throughout
**Decision:** Use **ENGAGE's bus-type-based NaN masking + per-unit normalization + norm-weighted MSE** uniformly across all grids and models. Do not mix masking conventions within an experiment.

**Why:**
- ENGAGE masks by **bus type (Slack/PV/PQ)** — physically principled. PowerGraph masks by the heuristic **`Y != 0`**, which wrongly drops genuinely-zero target quantities and conflates "known" with "happens to be zero."
- The mask decides which residuals enter the loss/metric, so absolute loss and R² shift with the convention. Direct comparison to PowerGraph's published numbers requires their mask; a clean generalization study should standardize on ENGAGE's.
- **Consistency is paramount:** never compare models trained under different masking. For the generalization goal, ENGAGE's masking is both consistent and more accurate.
- Trade-off: choosing ENGAGE masking means results are **not directly comparable to PowerGraph's paper** — accepted, because the objective is cross-grid generalization in ENGAGE, not reproducing PowerGraph's benchmark.

---

## Decision 7 — Model zoo: implement ALL GNNs from both ENGAGE and PowerGraph
**Decision:** Provide a unified model set covering **both** frameworks' architectures, all conforming to ENGAGE's model interface so they drop into the Cross-Context / OOD experiments:
- **From ENGAGE:** `GCN`, `ARMA_GNN`.
- **From PowerGraph-Node:** `GAT` (`GATConv`), `GIN` (`GINEConv`), `TRANSFORMER` (`TransformerConv`), and the `NNConv` edge-conditioned base.
- (`GCN` exists in both → keep one unified implementation.)

Unified target set: **`GCN`, `ARMA_GNN`, `GAT`, `GIN`, `TRANSFORMER`, `NNConv`.**

**Why:**
- The research goal is a **fair cross-grid generalization comparison across architectures** on transmission grids; that requires every candidate model available under one consistent pipeline, data contract, mask, and metric.
- PowerGraph's edge-aware layers (GAT/GIN/Transformer/NNConv) exploit edge features, which suits ENGAGE's richer `edge_attr = [trafo?, r_pu, x_pu, sc_voltage]`; ENGAGE's ARMA/GCN give continuity with the original paper.

**Implementation contract for the ported PowerGraph models (must match ENGAGE, not PowerGraph):**
- Constructor signature `__init__(input_dim=..., num_layers=...)` like ENGAGE's models, so experiment scripts can instantiate them uniformly (input dim grows with augmented features).
- Node input dim = **7** (`[Slack?, PV?, PQ?, p, q, vm, va]`), output dim = **4** (`[p, q, vm, va]`), `edge_attr` dim = **4** — not PowerGraph's `[G,B]`/3-or-4-col layouts.
- Implement an `inference()` step that **re-injects the physically-known quantities per bus type** (Slack/PV/PQ), exactly as ENGAGE's `GCN.inference` does — PowerGraph's models have no such step.
- Train/eval with **ENGAGE's masked, norm-weighted MSE + NRMSE-range metric** (per Decision 6), not PowerGraph's `mask=(Y!=0)` + R².

---

## Decision 8 — Two-layer experimental structure
**Decision:** Execute the work in **two layers**: **Layer 1** = correct and sanity-check what `engage_pg` v2 already built (reuse existing PowerGraph-trained models, but harmonize normalization and report a cross-grid NRMSE transfer matrix as the headline; g-score provisional). **Layer 2** = the well-posed generalization study on ENGAGE-format data with a distribution of topologies and the full retrained model zoo.

**Why:**
- Layer 1 de-risks and yields an honest first result cheaply (the models are already trained), while Layer 2 delivers the publishable benchmark.
- The layers are **not cleanly separable**: a pure Layer 1 "runs but is not insightful." Two Layer-2 concerns must be pulled into Layer 1 — **per-unit normalization** (Decision 9) and the awareness that the **g-score needs a distribution of topologies** (Decision 10), which Layer 1 lacks (one topology per grid), so Layer 1's g-score stays provisional.

---

## Decision 9 — Cross-grid comparability: per-unit normalization + fix the MMD defects
**Decision:** For any cross-grid comparison, **normalize on a physically consistent per-unit basis** (`baseMVA`/`baseKV`), not PowerGraph's per-grid max-abs. Also fix the two MMD defects before trusting topological distance: (a) retune kernel sigmas so the Gaussian is not saturated, and (b) compute topology on the **physical one-line graph, not the Ybus sparsity pattern with self-loops**.

**Why:**
- PowerGraph normalizes features/targets by each grid's **own** max-abs. A model trained on grid A then sees grid B in a *different* unit system → cross-grid NRMSE conflates a scaling mismatch with true generalization and is uninterpretable. Per-unit makes grids physically comparable (this is also why Decision 6 chose ENGAGE's per-unit convention).
- The v2 Laplacian MMD is degenerate: with `sigma_laplacian=1e-2` the kernel bandwidth is `1/(2·0.01²)=5000`, so every distinct pair collapses to `MMD=√2≈1.41421` and same-grid to 0 — a saturated 0/1 indicator, not a distance. The MMD math (ggme) is untouched; the defect is in the feature/sigma choices in the new custom scripts.
- PowerGraph's `edge_index = find(Ybus)` includes the diagonal (self-admittance) → degree/Laplacian describe the admittance pattern with self-loops, not the physical network.
- **Also:** the MMD/g-score assume a *distribution of graphs*; with one topology per grid each grid is a single point, so the g-score is fit to 3–4 points and is statistically fragile. Report it in Layer 1 only as provisional; it becomes well-posed in Layer 2. Use `get_generalization_score_raw` (no percentile trim) given the tiny sample.

---

## Decision 10 — Layer 2 generation spec: topology distribution via contingencies + the `runpp` re-solve engine
**Decision:** Build the distribution of topologies the g-score requires by **perturbing each base grid with credible contingencies (N-1, then N-2/N-k line/branch outages, optional generator outages)** and, for **every** perturbed topology, **re-solving AC power flow** to regenerate all node/edge values. A topology change invalidates the stored node values, so each contingency is a fresh solve — not a data edit.

**The re-solve engine (pandapower, in ENGAGE's pipeline — not PowerGraph's MATLAB):**
```python
import pandapower as pp
net = convert_from_systemm(...)          # the re-solvable grid MODEL (impedances, setpoints)
net.line.at[line_idx, "in_service"] = False   # the outage (N-1)
net.load["p_mw"], net.load["q_mvar"] = demand_p, demand_q   # hourly (Route B) or sampled (Route A)
pp.runpp(net)                            # AC power flow (Newton-Raphson)
# fresh solved state -> net.res_bus.vm_pu / va_degree, net.res_gen.p_mw/q_mvar, net.res_line...
# pp.runopp -> generator re-dispatch (more realistic post-contingency); pp.rundcpp -> dc_pf baseline
```
Then filter (drop non-converged / islanded / voltage-violating / overloaded) and convert to ENGAGE `Data`. Each grid becomes a **cloud of graphs with varying topology + loading**.

**Why:**
- Removing a line reroutes power, so `V/θ` at every bus and all branch flows change; keeping the old values would produce physically invalid samples. The stored PowerGraph `.mat` tensors are **solved outputs** with no impedances/setpoints — they cannot be re-solved, which is exactly why the `System.m → pandapower` model (Decisions 1/4/5) is mandatory.
- AC power flow is a standard Newton-Raphson solve provided by pandapower (`runpp`/`runopp`/`rundcpp`); ENGAGE already runs it in `graph_gen.py` + `powerdata-gen`. PowerGraph does the same physics in MATLAB (`gendataopf.m`) but emits its own format, so we reuse ENGAGE's engine.
- N-1/N-k contingency analysis is standard transmission practice, so the perturbed states are physically credible, and removing lines genuinely changes degree/Laplacian descriptors → a real spread of topological distances for the MMD/g-score.
- **Connectivity/tuning:** reject islanding (or handle islands) and retune disconnection probabilities for meshed transmission (islands less easily than radial distribution); use a range of contingency depths so descriptors spread and `mmd_range` is non-degenerate.

---

## Decision 11 — Optionally harvest contingencies from PowerGraph-Graph to inform outages
**Decision:** Use the **PowerGraph-Graph** cascading-failure dataset as an optional source of **credible, grid-specific contingencies** to drive Layer 2 generation, instead of (or alongside) blind random N-k. Harvest only the **topology (which lines are out)** from each sample, then re-solve AC PF (Decision 10) to produce node-level PF targets.

**Why:**
- PowerGraph-Graph encodes real outage states per grid: each sample removes failed lines, `exp.mat` marks the triggering branch(es), and `of_*` labels demand-not-served — i.e. which outages are credible and which are consequential.
- This lets us **stratify** sampling toward consequential contingencies (widening the MMD range) and build a **curriculum** from benign N-1 to severe cascades.

**Caveats:**
- Use only their **topology**, not their graph-level values/labels; re-solve PF ourselves for node targets.
- Drop cascade end-states that are **islanded/blackout** (no converged single-grid PF).
- Mixing two PowerGraph datasets is messier and less controllable than generating N-1/N-k directly — default to generating our own, keep the harvest as a cross-check / realism boost.

---

## Decision 12 — Drop the "two-repo mapping" model: one clean self-contained pipeline
**Decision:** Abandon the "make ENGAGE and PowerGraph interoperate" framing entirely. The repository (`eval_gnn_generalization_pg`) is a **single, self-contained Layer-2 pipeline**: ENGAGE's *methodology* reimplemented directly (not imported), applied to PowerGraph's transmission grids. No `powerdata-gen` submodule, no `ggme` submodule, no ENGAGE package dependency.

**Why:**
- Gluing two incompatible repos (different normalization, masking, edge features, data format, and repo layout) is effort spent reconciling conventions instead of doing science; it was also the source of `engage_pg` v2's degenerate MMD.
- The essential logic is small: ENGAGE's feature extractors are vendored in `engage_contract.py` (with attribution); the re-solve loop is `transmission_graph_gen.py`; MMD/g-score are reimplemented in `mmd_utils.py`/`training_utils.py`. Everything is readable in one repo and runs on its own.
- `ggme`/`powerdata-gen` are distribution-grid (SimBench) oriented and would drag in unused loading code — exactly the mess this decision removes.

**Implication:** the two-layer structure of Decision 8 collapses to "just build Layer 2." Layer 1 (wrapping the pre-trained PowerGraph models) is retained only as an optional cheap sanity check, reported honestly as an NRMSE-vs-graph-distance transfer study, never as a g-score.

---

## Decision 13 — Model checkpointing + small-N g-score reading
**Decision (checkpointing):** `experiments.py` takes an optional `--save_models <dir>` flag. When set, every trained model's `state_dict` is written with a stable naming convention:
- Cross-context: `cc_<model>_<train_grid>.pt` (e.g. `cc_gcn_IEEE118.pt`).
- Leave-one-grid-out OOD: `ood_<model>_heldout_<grid>.pt` (e.g. `ood_gat_heldout_UK.pt`).
A full run therefore yields 24 cross-context + 24 OOD = 48 checkpoints, each reloadable via `MODELS[name](input_dim=7).load_state_dict(torch.load(path))`.

**Why:** reproducibility and reuse — the exact trained GNNs behind the reported numbers can be inspected, fine-tuned, or served without retraining.

**Decision (g-score at small N):** the ENGAGE g-score uses a 2/98 percentile trim (`bounds=2`) that assumes many samples. With only 3 unseen grids per training grid it keeps a single point, forcing `std_nrmse=0` and `mmd_range=0` (degenerate). We therefore additionally report a **small-N g-score** (no percentile trim, all unseen grids) as `gscore_smallN.csv`, and treat the **transfer matrix + MMD** as the headline. This is the concrete manifestation of the earlier caveat that the g-score is statistically under-powered with only ~4 grids.

**Decision (OOD g-score — `compute_ood_gscores`, `gscore_ood.csv`):** the cross-context g-score is *per training grid* and therefore has only the 3 unseen TEST grids as points (the degeneracy above). We additionally compute an **OOD g-score** *per model* over the **held-out grids** of the leave-one-grid-out experiment: one point per held-out grid (up to 4 points), where the topological distance is the **mean Laplacian-MMD from each held-out grid to its TRAINING grids**. No percentile trim (`bounds=0`); NaN cells (e.g. a diverged ARMA split) are dropped. This is the **better-posed** g-score at N=4 (more points, no trim collapse) and is the flavour most aligned with the study's operational question — *generalize to a genuinely new grid after training on several*. ENGAGE itself reports a g-score for both its cross-context and OOD experiments; we had initially reported OOD only as per-grid NRMSE, and this decision closes that gap. Rationale for the distance choice: the g-score's x-axis must be a **grid↔grid** topological distance (a single comparable scalar per pair, built from size-invariant descriptors — see D9/MMD), so the held-out grid's distance to the *set* of training grids is summarized as the mean of its pairwise Laplacian-MMDs.

---

## Semantic mappings that must be preserved (implementation contract)
- **Bus type → one-hot** `[Slack, PV, PQ]` from MATPOWER type (3/2/1).
- **Per-unit base**: carry `baseMVA` / `baseKV` so `r_pu`, `x_pu` are correct (ENGAGE edge attr = `[trafo?, r_pu, x_pu, sc_voltage]`).
- **Transformers vs lines**: MATPOWER branches with `tap ≠ 0` / `ratio ≠ 1` → pandapower trafos (set `trafo?` flag + `sc_voltage`); others → lines.
- **NaN masking of unknowns** per bus type — produced by ENGAGE's `get_node_features`, not PowerGraph's `mask=(Y!=0)`.
- **`dc_pf`** baseline — computed by ENGAGE via `pp.rundcpp` (works for transmission).
- **Transmission tuning** — tighten voltage filters (~0.95–1.05 vs distribution 0.85–1.15) and retune line-disconnection probabilities in `base_gen_config.yaml` (meshed grids island less easily).

---

## Known prerequisites / blockers
- **ENGAGE conda environment** (`environment.yaml`) must be set up. pandapower/torch/torch-geometric/simbench are not currently installed on the VM.
- **ENGAGE submodules** `powerdata-gen` and `ggme` are empty in the uploaded zips (including `engage_pg` v2) and can't be fetched from this VM (firewall). Route B minimizes the `powerdata-gen` dependency; `ggme` is needed for the MMD / g-score evaluation step (v2's cached MMD CSVs were computed on the user's machine where `ggme` existed).
- ~~**Octave** is not installed~~ → **Resolved:** GNU Octave 6.4.0 installed and verified in this session; conversion confirmed trivial (see Decision 5).

---

## Summary of the chosen path
**Two layers.** **Layer 1** corrects `engage_pg` v2's Level-2 probe: harmonize to per-unit normalization, fix the MMD defects, and report a **cross-grid NRMSE transfer matrix** (g-score provisional) using the already-trained models. **Layer 2** is the well-posed study: **Level 1 / Route B, all four grids, Octave-based conversion** (done once in-session and committed as `.mat`, validated against a PowerGraph PF solution), a **distribution of topologies via N-1/N-k contingency re-solves** (`pp.runpp`, optionally informed by PowerGraph-Graph), the full ENGAGE+PowerGraph model zoo (`GCN, ARMA_GNN, GAT, GIN, TRANSFORMER, NNConv`) under ENGAGE's interface, and ENGAGE masking + per-unit + weighted-MSE throughout. This is the most faithful *and* cleanest way to turn PowerGraph's source transmission data into ENGAGE-format datasets and unlock a well-posed cross-grid generalization study across architectures on transmission grids.
