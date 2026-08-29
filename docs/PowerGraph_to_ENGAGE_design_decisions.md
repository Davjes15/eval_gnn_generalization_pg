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
> **Superseded in part by Decision 20 (read that first).** The intent below — that a
> model must not see grid B in a different unit system than grid A — is correct and
> retained. Two claims in it were wrong in fact: (i) ENGAGE's feature extractors do
> **not** deliver per-unit *node* features (only edge impedances are per-unit; node
> quantities are raw MW/Mvar/p.u./degrees, in ENGAGE's code and in ours), and (ii)
> per-unit conversion could not have fixed the cross-grid scale problem here anyway,
> because all four transmission cases share `sn_mva = 100`, so it is a division by
> one common constant. No node-level scaling was implemented until Decision 20.

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

**Decision (g-score at small N):** the ENGAGE g-score uses a 2/98 percentile trim (`bounds=2`) that assumes many samples. With only 3 unseen grids per training grid it keeps a single point, forcing `std_nrmse=0` and `mmd_range=0` (degenerate). We therefore additionally report a **small-N g-score** (no percentile trim, all unseen grids) as `gscore_smallN.csv`, and treat the **transfer matrix + MMD** as the headline.
This is the concrete manifestation of the earlier caveat that the g-score is statistically under-powered with only ~4 grids.
*Superseded in form, not in reasoning:* the current pipeline emits the no-trim
reading as the pooled `gscore_cc_aggregate.csv` (ENGAGE Table-3 format, one row
per model); `gscore_smallN.csv` survives only in the legacy `full_run/results/`.

**Decision (OOD g-score — `compute_ood_gscores`, `gscore_ood.csv`):** the cross-context g-score is *per training grid* and therefore has only the 3 unseen TEST grids as points (the degeneracy above). We additionally compute an **OOD g-score** *per model* over the **held-out grids** of the leave-one-grid-out experiment: one point per held-out grid (up to 4 points). No percentile trim (`bounds=0`); NaN cells (e.g. a diverged ARMA split) are dropped. This is the **better-posed** g-score at N=4 (more points, no trim collapse) and is the flavour most aligned with the study's operational question — *generalize to a genuinely new grid after training on several*. ENGAGE itself reports a g-score for both its cross-context and OOD experiments; we had initially reported OOD only as per-grid NRMSE, and this decision closes that gap. The choice of the distance x-axis is fixed by Decision 14.

---

## Decision 14 — OOD distance is the POOLED MMD to the training mixture (ENGAGE-consistent), not a mean of pairwise MMDs
**Decision:** For the OOD g-score, the topological distance of each held-out grid is the **pooled** Laplacian-MMD between that grid and the **union of its training grids treated as one distribution** — `MMD(held, A∪B∪C)`. Concretely, `ood_distances(data, grids)` concatenates the training grids' graphs into a single sample set and calls `evaluate_mmd(pooled_train, held_test)` once. `ood_distance.csv` now stores `mmd_pooled_degree` and `mmd_pooled_laplacian` per held-out grid (the pooled Laplacian value is the g-score x-axis).

**Why (this fixes a bug):** the initial implementation summarized the held-out grid's distance as the **mean of the pairwise** Laplacian-MMDs to each training grid separately (`mean(MMD(held,A), MMD(held,B), MMD(held,C))`). That is **not** how ENGAGE computes it and is not the same quantity: ENGAGE's `evaluate_cc_mmd` builds `loader_train` by pooling **all** training grid codes (`get_dataset` does `complete_dataset.extend(...)` over every grid) and computes a **single** MMD between that pooled training distribution and the held-out grid. `mean(MMD(held,·))  ≠  MMD(held, ∪·)` in general, and the pooled form is the correct notion of "distance to the mixture the model was actually trained on."

**Impact (distance axis only — no NRMSE / verdict change):**
- Pooled Laplacian OOD distances: IEEE118 0.62, IEEE24 0.65, IEEE39 0.67, **UK 0.97** (vs. old mean-of-pairwise 0.94 / 0.82 / 0.94 / 1.13). UK remains the farthest; IEEE118 becomes the closest to its training mixture.
- OOD g-scores shift only marginally (`mmd_range` 0.305→0.353): transformer 0.154→**0.153** (still best), gat 0.163, gin 0.164→0.163, gcn 0.183→0.182, nnconv 1.982→**1.961** (still disqualified), arma_gnn 0.146→0.147 (still optimistic; UK point dropped).
- The OOD generalizability curve's **rank** correlation improves (Spearman −0.11→**+0.62**) because the pooled distance correctly orders UK as farthest-and-hardest; linear Pearson stays ≈0 (−0.05). The architecture verdict is unchanged.

---

## Decision 15 — A candidate that diverges is a failed candidate, not a candidate with a bad score
**Decision:** `tune_budget.py` **disqualifies** any hyperparameter candidate whose validation loss is non-finite on **any** grid, and requires the surviving winner to **reproduce at a second seed (100)** before it is frozen. If no candidate survives at one learning rate the full grid is rescored at the other; if none survives either, the sweep **fails explicitly** rather than freezing an unstable configuration.

**Why (this fixes a defect the results exposed):** the original rule took the argmin of the mean validation loss with `inf` merely ranked last. ARMA's sweep had recorded `inf` for all three hidden-64 candidates, yet 8×128/lr 1e-3 won because it happened to be finite at the single seed Stage 1 scores (seed 0). That configuration then diverged to NaN in **10 of 20** Regime-A within-grid runs and **49 of 80** cross-context runs — seeds 700 and 1000 everywhere, seed 100 on most grids. A selection procedure that can crown a configuration which only trains at one seed is not a selection procedure.

**Scope of the amendment:** this rule was adopted **after** ARMA's instability surfaced, and is disclosed as such. It is applied identically to all six architectures. For `gcn`, `gat`, `gin`, `transformer` and `nnconv` it is a **no-op** — none of them produced a single non-finite value in any sweep trial or any of the final runs — so their frozen configurations and completed results are unchanged by it. The alternative considered and rejected was to keep the config and report ARMA's divergence rate, which would have rested ARMA's ranking on 31 of 80 runs with means biased toward exactly the seeds that survived.

---

## Decision 16 — ARMA's edge weight is softplus, not leaky ReLU (the actual cause of the divergence)
**Decision:** in `models.py`, `ARMA_GNN`'s scalar edge encoder ends in **softplus** instead of the shared leaky ReLU, so its learned edge weight is **non-negative by construction**. The change is ARMA-scoped: the other five architectures' training is bit-for-bit unchanged and their completed results remain valid.

**Why (mechanism, not a guess):** PyG's `ARMAConv` normalizes the adjacency with `gcn_norm(..., add_self_loops=False)`, whereas `GCNConv` uses `add_self_loops=True`. With a leaky-ReLU edge encoder the learned weight can be **negative**; a bus whose incident weights sum to ≤ 0 makes `deg ** -0.5` infinite inside the normalization, and the **forward pass** produces `inf`/NaN before a gradient exists. GCN is immune on the identical tensors because unit self-loops keep its degrees positive.

**Evidence the diagnosis is right:**
- **Gradient clipping does nothing.** Probed 8×128/lr 1e-3 on IEEE39 and UK at seeds 100 and 700, with and without `clip_grad_norm_(…, 1.0)`: still `inf` with clipping. So ordinary gradient explosion was **not** the cause, and the "clip everywhere and re-run everything" option was never going to work.
- **The data is not the cause.** Five architectures consume the identical tensors with zero non-finite values; ARMA at seed 0 is finite on the same grids; the Regime-A generation gate passed.
- **After the fix**, the seeds that always died are finite and *better* than the old surviving numbers (IEEE24 0.0063 / 0.0083, IEEE39 0.00037 / 0.00035, UK 0.0282 / 0.0164 at seeds 100 / 700), and the previously-`inf` hidden-64 candidates train normally.

**Consequence:** because this changes ARMA's definition, its **entire tuning sweep was redone** under the fixed layer (9 configs × 2 learning rates × 2 seeds × 4 grids → `results_a/arma_v2/`), with **zero divergences**, re-selecting 8 × 128 / lr 1e-3 reproducibly (0.01829 at seed 0, 0.01830 mean over seeds 0 and 100). Hansen et al.'s stack count (5, `shared_weights=False`) is retained; depth/width/lr are selected by the same equal-budget protocol as every other architecture. The **pre-fix ARMA result rows are deliberately kept** in `results_a/within_arma_gnn/`, `results_a/arma_stability/`, `results_a/arma_lowlr/` and `results_tuned/arma_gnn/` as the evidence for the divergence finding, and are **excluded from all analysis inputs**; the corrected arms live in `results_a/within_arma_v2/` and `results_tuned/arma_v2/`.

---

## Decision 17 — NNConv is our own addition, and is replicated at 3 seeds rather than 5
**Decision:** keep `NNConv` in the comparison, run it at seeds `[0, 100, 300]` instead of `[0, 100, 300, 700, 1000]`, and disclose the reduced replication wherever its numbers appear.

**Provenance, stated plainly:** PowerGraph's baselines are exactly four — `GCNConv`, `GATConv`, `GINEConv`, `TransformerConv`. `ARMAConv` comes from ENGAGE (via Hansen et al.). **`NNConv` is neither paper's baseline; it is our addition**, included as the most edge-expressive layer available, which is the interesting case for power flow where edge admittances carry the physics. Dropping it would therefore cost no coverage of either source paper — it would cost the edge-expressive end of the architecture axis.

**Why the reduction:** NNConv's edge network emits a full `hidden × hidden` transform **per edge**. At the frozen 2 × 128 that is a 128×128 matrix per edge, making one IEEE118 training ≈ 3 h and the full 60-run programme (within + cross-context + OOD × 5 seeds) ≈ 1.5–2 days wall clock on 8 cores — the pooled-grid OOD arm being most of it. Three seeds keeps a variance estimate at ~60% of the cost. This is a **compute** decision, explicitly approved, not a methodological one, and it is the only architecture with fewer than 5 seeds.

---

## Decision 18 — Checkpoints for four architectures; ARMA and NNConv reproduce from the recorded seed
**Decision:** `ckpt_a/` and `ckpt_b/` hold trained weights for `gcn`, `gat`, `gin` and `transformer` (one file per grid × seed). **`arma_gnn` and `nnconv` have no checkpoints** and are reproduced by re-running the documented command at the recorded seed; every result row carries `model`, `num_layers`, `hidden`, `learning_rate` and `seed`, and the seeds are fixed.

**Why:** their arms were launched without `--save_models`, so only metric rows were written. The stale ARMA checkpoints that *did* exist were from the pre-fix definition of Decision 16 and **many of their tensors were NaN**, so they were **deleted** (220 MB) rather than kept — NaN weights sitting next to valid ones invite a superseded checkpoint into a result, and they held no information the result CSVs do not. A `PROVENANCE.txt` is left in each emptied directory. **Known consequence:** `--skip_existing` requires `--save_models`, so an interrupted ARMA or NNConv arm restarts from its first grid.

---

## Decision 19 — The inherited-config `full_run/results/` tables are superseded, not deleted
**Decision:** every headline number — rankings, g-score, per-quantity P/Q/V/θ, DC-baseline comparison — is recomputed from the **tuned-configuration** runs (`results_a/`, `results_tuned/`). The earlier `full_run/results/` tables were produced under the **inherited** ENGAGE configuration (before the equal-budget sweep of Decision 15) and are retained only as the historical run, marked in `RUN_METADATA.txt`, and never mixed into a tuned table.

**Why:** the two sets differ in depth, width and learning rate per architecture, so a table that mixed them would compare architectures at unequal budgets — precisely the confound the sweep exists to remove.

---

## Decision 20 — Feature/target scaling is an explicit training option, and the benchmark protocol uses `pu_zscore` (audit item A2)
**Decision:** scaling lives in one module, `normalization.py`, and is selected per run with
`experiments.py --normalize {none,pu,pu_zscore}`.

| mode | meaning | role |
|---|---|---|
| `none` | raw physical units | the protocol every pre-existing artifact was produced with; kept as the **default** so those rows stay bit-identically reproducible, and reported as an ablation |
| `pu` | powers ÷ `sn_mva`, angles → radians | engineering per-unit only; documented, not used for headline results |
| `pu_zscore` | per-unit, then per-quantity z-score with **training-split statistics** | **the benchmark protocol** for all final results |

Statistics are fitted on training data only — per grid for the within-grid arm, on the
source grid for cross-context, on the pooled training grids for leave-one-grid-out — and
applied unchanged to validation and test. Node features (columns 3:7) and targets are
scaled with the *same* statistics, which is what keeps the known-value re-injection in
`models.py::inference` legal. Predictions are de-normalized before any metric is
computed, so every reported number stays in physical units, and the DC baseline is never
scaled at all.

**Why:** the defect is not primarily the cross-grid magnitude gap the audit pointed at — it
is *inside each sample*. In raw units the four target quantities differ by up to ten orders
of magnitude in the loss, so the fraction of the training loss attributable to voltage
magnitude is ≈ 5·10⁻⁸ on IEEE24 and ≈ 1·10⁻¹¹ on UK: voltage is effectively not optimized.
That is exactly what the raw-unit results show — every architecture is worse than the
constant predictor `V ≡ 1.0` p.u. even in-distribution. A 15-epoch control on IEEE24/`gcn`
moves voltage NRMSE from 2.97 to 0.021 with no loss on P, Q or θ.

**Field grounding:** ENGAGE does not scale node features, and gets away with it because its
SimBench LV/MV feeders have MW injections of the same order as `vm_pu ≈ 1`; our
transmission systems carry 10³–10⁴ MW. **PowerGraph-Node's released code does scale** —
per-dimension max-abs on both X and Y, de-normalized for reporting. PowerFlowNet z-scores
X, Y and edge features on train-split statistics; Hansen et al. (the source of our ARMA
setup) divides powers by `baseMVA`. Scaling with de-normalized reporting is the field norm;
ENGAGE is the exception. Details and measurements: `Normalization_assessment.md`.

**What it does *not* fix:** the UK system genuinely moves an order of magnitude more power
than IEEE24, and no unit system removes that. A train-grid-fitted scaler is the honest
choice, but the transfer claim must then read "generalization to an unseen *system*"
(topology **and** scale), not "to an unseen topology".

---

## Decision 21 — Regime B is regenerated with a blocked temporal split (audit item A5)
**Decision:** the final varying-topology datasets (`data_full_v2`) are generated with
`transmission_graph_gen.py --time_split blocked`, which gives train, validation and test
**disjoint contiguous windows of the demand year**, separated by a one-day gap
(`--time_gap 96` steps at 15-minute resolution), and draws each demand snapshot at most
once. Validation gate H (`validate.py --regime b --expect_blocked`) asserts the resulting
properties, and `tests/test_split_hygiene.py` asserts that the gate actually catches a
leaky dataset. `data_full` is retained as provenance for the superseded results, never
mixed into a final table.

**Why:** the original Regime B data drew every split's snapshots uniformly from the whole
year, so splits shared demand snapshots outright (4/3/1/0 shared snapshots across the four
grids). Uniqueness alone is still too weak, because consecutive 15-minute snapshots are
near-duplicates: a test point drawn one step after a training point is not an independent
operating condition. Blocking in time is the standard remedy for temporally correlated
data and is what makes "unseen operating conditions" defensible. Regime A (`data_a`) was
already generated with `--unique_demand` and is unaffected, which is why only Regime B is
regenerated.

**Sequencing consequence (why this came before training):** A5 changes the Regime B data, so
the cross-context and OOD arms had to be trained *after* regeneration, not before —
otherwise the same six architectures would have been trained twice on those two arms.

---

## Decision 22 — The aggregate NRMSE is kept for comparability, but the reported physics is per quantity and over predicted entries only (audit item A3)
**Decision:** `physics_metrics.py` is the reporting layer for every final result, run over saved
checkpoints by `eval_checkpoints.py`. It reports, in **physical units** after de-normalization:
per-quantity NRMSE and MAE; the same restricted to the entries the model genuinely predicts;
predicted-entry counts; p95/p99/max absolute error; and voltage-limit violation rate, missed-violation
("false-secure") rate and false-alarm rate. ENGAGE's pooled `nrmse_range` stays in the tables as the
comparability column, not as the headline.

**Why:** two of the four target columns per bus are ground truth re-injected at inference (slack: V, θ
known; PV: P, V known; PQ: P, Q known — Decision 6), so an aggregate over all four columns is partly
a score of the inputs. And pooling MW with p.u. lets the large-magnitude quantities carry the number:
the measured share of the loss is P 0.83, Q 0.15, θ 0.02, V 5e-8 on IEEE24, which is how every
architecture came to be worse than the constant `V ≡ 1.0` in-distribution while the aggregate looked
excellent. For a screening application the quantity of interest is exactly the one the aggregate
hides, which is why the violation and false-secure rates are reported rather than only the error.

## Decision 23 — Reproducibility is an artifact, not a claim (audit item A4)
**Decision:** `docs/Reproducibility.md` is the single entry point: pinned versions,
generation and training commands, `docs/provenance/*.csv` (all 24 `dataset_src.csv` files),
the realised blocked windows per grid, and `checkpoint_index.py` → `docs/tables/checkpoint_index.csv`
mapping every results row to a weight file by path, size, SHA-256 and parameter count. The tuning
CSVs the configuration tables cite are committed. Two things are stated rather than fixed: the
tensors and weights are too large for git and need a data release, and regeneration is a *fresh draw
from the same protocol*, so it reproduces the protocol, not the numbers.

**Why:** the repository is pushed as an explicit file list, so citations to uncommitted artifacts had
accumulated silently — the selection tables referenced tuning CSVs that were not in the repo, and
`requirements.txt` was unpinned even though the A1 bug was a pandapower version regression. A
dead reference is indistinguishable from a fabricated one to a reader, which is the real cost.

## Decision 24 — The g-score is reported as a risk-averse summary, not as a distance-aware ranking; MMD gains an electrical descriptor (audit items A6, A7)
**Decision:** `μ` and `σ` are the primary transfer numbers and the g-score accompanies them for
ENGAGE comparability, with the degeneracy stated: `Δ_MMD` is a property of the data, so within an arm
it is one constant and `g = μ + 0.806σ` (cross-context) or `μ + 0.857σ` (OOD) — verified against all
committed rows to 1e-4, with rank-by-g ≡ rank-by-mean (τ = 1.0). For the distance itself, `mmd()` is
named as the biased V-statistic (with `unbiased=True` available), the per-pair median bandwidth is
stated, and `mmd_utils.reactance_histogram` adds a `log10(x_pu)` electrical descriptor alongside the
degree and Laplacian ones. Full write-up: `docs/Generalization_score_and_MMD.md`.

**Why:** with a single dataset the MMD term cannot reorder architectures, so presenting the g-score as
"the generalization metric" would imply information it does not carry; it is still worth reporting
because it prefers uniform mediocrity over a catastrophic fold, which is the right preference for
security screening. And a purely topological distance is blind to the ~20× load-scale spread between
our cases — UK → IEEE24 is the closest off-diagonal pair by degree MMD (0.276) and far apart
electrically (0.833), so an electrical complement is needed before any distance is used as a covariate.

---

## Decision 25 — Hyperparameters are selected once on Regime A validation data and frozen; seeds are replication, never a tuned parameter
**Decision:** two separate passes over the data, with the three splits kept to non-overlapping
jobs — train (800) fits weights, validation (100) *chooses* hyperparameters, test (100) is read
only to report. Pass 1 (`tune_budget.py` on `data_a`) scores ~10 candidate configurations per
architecture by mean best **validation** loss across the four grids at one seed plus a tie-break
seed, and freezes the winner in `configs/arch_config.json`. Pass 2 (`experiments.py`) re-trains
that one configuration at seeds 0/100/300/700/1000 (NNConv 0/100/300) and scores on **test**;
those are the reported numbers. The same frozen configuration goes into all three arms —
within-grid, cross-context and OOD — and Regime B is never re-tuned.

**Why:** the selection pass's own scores are optimistically biased, having been selected on the
validation set, and exist at one seed with no spread, so reporting the search's winner as the
result is textbook selection bias. Freezing across arms is what makes the headline claim
falsifiable: if each arm chose its own hyperparameters, a rank change between regimes could be
explained by "different configurations" rather than by generalization, and re-tuning on the
transfer data would select on the very quantity being measured. The cost is disclosed — a
configuration tuned under fixed topology may be suboptimal under varying topology, so absolute
Regime B errors are an upper bound on a per-arm-tuned model.

On seeds: a seed fixes weight initialisation and batch ordering, and serves exactly two purposes
— exact reproducibility of a single row (the seed is carried in every result row and every
checkpoint filename) and a spread over training randomness, so an architecture gap can be judged
against run-to-run noise. There is **no best seed**: selecting one per architecture would let any
desired ranking be manufactured and would not reproduce elsewhere. Five seeds matches PowerGraph
(p. 5; ENGAGE does not report its seed count — see `Paper_verification.md` §5), and is small enough
that fine-grained gaps are not claimed — hence τ per seed and per grid with permutation
p-values instead of one aggregate ranking. It also captures training randomness only, not the
uncertainty of the dataset draw. Full statement: the *Final protocol* section of
`docs/Experimental_Design_transmission_GNN_generalization.md`.

---

## Decision 26 — AC feasibility is scored by replay, and the second audit is answered without retraining
**Decision:** the physical validity of a *prediction* is measured directly, not inferred from the
regression error. `ac_feasibility.py` rebuilds every test network from `dataset_src.csv`, takes the
post-contingency admittance matrix, the branch data and `max_i_ka` from pandapower's own internal
representation, and reports (a) the AC power-balance residual of the predicted state,
`dS = S_spec - (V conj(Y V) - S_shunt)`, in MW/Mvar and as a share of the snapshot's served load,
and (b) the branch loading of the predicted state against the ratings, as an
overload confusion (missed / false alarms). It runs as `eval_checkpoints.py --feasibility` over the
saved checkpoints — **no retraining, no change to any weight**.

Two details are load-bearing and were both found by calibrating against the labels. The shunt term
must be subtracted, because pandapower books shunt consumption in `res_bus` while `Ybus` already
contains the shunt admittance; without it the IEEE24 shunt bus shows a fictitious ~100 Mvar
residual. And the true state is scored through the identical path in every run
(`ac_dp_true_max_mw`, ≤ 2.8e-2 MW), so the reconstruction floor is reported next to the model's
number rather than assumed to be zero. The predicted loading is likewise reported next to
`branch_loading_max_pct_true`, because several source OPF snapshots are not thermally secure
themselves (true loadings up to ~680 % on IEEE118). The screen covers every in-service branch,
lines and two-winding transformers alike (5/38 of IEEE24's branches are transformers, 11/46
IEEE39, 9/184 IEEE118, 4/90 UK), each end against its own current rating; audit 3 (C4) found the
first implementation screened lines only.

**Why the rest of audit 2 was answered without training:** re-selecting hyperparameters under
`pu_zscore` (finding B4) and lifting NNConv from 3 to 5 seeds are the only two items that need
GPU-hours, and both change *how well* the models were trained rather than *what is claimed* about
transfer. They are therefore declined and recorded as accepted limitations L1 and L2 in
`docs/Audit_response.md`, with the concrete consequence of each spelled out, rather than left for a
later reviewer to rediscover. Everything else — the non-finite policy, the g-score trim, the
protocol/grid decomposition, the pooled-vs-cell rank reporting and the run metadata — is a
recomputation over artifacts that already exist.

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
