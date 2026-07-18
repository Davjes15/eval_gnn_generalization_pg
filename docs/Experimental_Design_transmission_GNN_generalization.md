# Experimental Design — Generalization of GNN Architectures for Transmission Grids

Status: **design specification** (no experiments run yet). This document defines, for each of the two layers, the **research question**, the **experimental setup**, and the **methodology**. It is the companion to `PowerGraph_to_ENGAGE_design_decisions.md` (which records *why* each choice was made); this file records *what experiment we actually run*.

## Overarching goal
Study **how well GNN architectures generalize to unseen transmission topologies** for the AC power-flow node task, and benchmark this against PowerGraph, which only ever trains and tests *within* a single fixed-topology grid. Generalization is quantified with ENGAGE's g-score (NRMSE vs. topological distance via MMD).

### Framing (power-systems motivation — read this first)
AC power flow is **deterministic physics**: given a grid's full model (topology + impedances + injections) you can just solve it with Newton-Raphson. So the value of a learned GNN surrogate is **not** "predict a grid you could otherwise solve" — it is **(i) amortization/speed** across huge numbers of cases (contingency screening, planning scenarios, real-time what-ifs) and **(ii) staying accurate as topology changes**. Accordingly:
- **Primary, operationally-motivated axis:** generalization **across contingencies / topological variations** of transmission grids (and to *related* unseen systems). This is the headline claim and is exactly what Layer 2's contingency distribution enables.
- **Secondary, scientific stress test:** transfer between structurally very different grids (e.g. IEEE24 → UK). Interesting as a limit test, but it has weak *operational* motivation, so it is reported as a stress test, not the main result.

## Grids
IEEE24, IEEE39, IEEE118, and the UK 29-bus system (PowerGraph's own `System.m` cases). Task: **node-level AC power-flow (PF) state estimation** — predict per-bus `[P, Q, V, θ]`.

### Metrics & baselines (applies to both layers)
A single aggregate NRMSE **overstates** performance, because the four targets are not equally hard: **V** is tightly bounded (~0.95–1.05 pu) and nearly trivial to predict, while **θ (angles)** and **Q (reactive power)** are the hard, informative quantities. Therefore every result must report:
- **Per-quantity errors — V, θ, P, Q separately** (not just the aggregate `nrmse_range`).
- **The DC-PF baseline** (`training_utils.test_dc_pf`), and ideally a warm-started single Newton step, so "the GNN beats trivial physics" is *demonstrated*, not assumed.
- **Topological distance via MMD** (degree + Laplacian). Because power engineers reason in **electrical distance** (impedance-weighted), optionally complement MMD with an electrical measure (e.g. X/R or short-circuit-ratio distribution distance, or a PTDF-based distance) to strengthen power-systems credibility. MMD stays the primary distance for the g-score; the electrical measure is a corroborating cross-check.

---

# The two-layer approach (and why it is split this way)

| | Layer 1 — Correct & sanity-check what exists | Layer 2 — The well-posed generalization study |
|---|---|---|
| Data | PowerGraph's existing fixed-topology node snapshots | ENGAGE-format data regenerated from `System.m` with a **distribution of topologies** |
| Models | The GCN (and others) already trained in the PowerGraph-Node pipeline | Full model zoo retrained under ENGAGE's contract |
| Normalization | **Must be harmonized to per-unit** (else results are invalid) | Per-unit throughout |
| Primary metric | **Cross-grid NRMSE transfer matrix** | g-score (NRMSE vs MMD) + transfer matrix |
| g-score | Provisional only (ill-posed: 1 topology/grid, few points) | Well-posed (distribution of topologies per grid) |
| Purpose | De-risk, validate software, get a first honest result | The publishable benchmark |

**Why they are not cleanly separable:** a "pure" Layer 1 (just fix bugs, keep PowerGraph's per-grid max-abs normalization and single-topology MMD) will *run but not be insightful*. Two Layer-2 concerns must be pulled forward into Layer 1 to make it meaningful: (1) **per-unit normalization**, and (2) awareness that the **g-score needs a distribution of topologies** — which Layer 1 does not have, so its g-score stays provisional.

---

# LAYER 1 — Corrected zero-shot cross-grid transfer of existing models

## Research question
**RQ1:** When a GNN trained for node-level PF *within one transmission grid* is applied **zero-shot to an unseen transmission grid**, how much does accuracy degrade, and does that degradation grow with the topological distance between grids?

Sub-questions:
- **RQ1a:** How large is the within-grid → unseen-grid accuracy gap (the "generalization gap") relative to PowerGraph's within-grid numbers?
- **RQ1b:** Is the ordering of test grids by error consistent with their topological distance (degree/Laplacian MMD) from the training grid?

## Experimental setup
- **Models under test:** the checkpoints already trained in the PowerGraph-Node pipeline (currently GCN for IEEE118, IEEE24, UK; extend to the other architectures/grids as available).
- **Protocol:** train-on-one-grid, test-on-the-other-three (leave-the-training-grid-out), zero-shot (no fine-tuning on the target grid).
- **Held-out reference:** each model's *own* within-grid test split (PowerGraph's regime) is the baseline the cross-grid numbers are compared against.
- **Fixed factors:** identical preprocessing, mask convention, and metric across every (train, test) pair — no mixing.

## Methodology
1. **Harmonize normalization to per-unit (mandatory).** Replace PowerGraph's per-grid max-abs scaling with a **physically consistent per-unit basis** (`baseMVA`/`baseKV`) so features/targets are comparable across grids. Without this, cross-grid NRMSE conflates a units/scaling mismatch with true generalization and is not interpretable.
2. **Inference across sizes.** GNN message passing is size-agnostic, so a model trained on grid A runs on grid B despite different bus counts — verify shapes (`x` = `(N,·)`) and that the mask/target columns align.
3. **Primary result — the cross-grid NRMSE transfer matrix.** For every (train grid, test grid, architecture), report NRMSE (ENGAGE `nrmse_range`). This directly answers RQ1/RQ1a and is valid once normalization is harmonized.
4. **Topological distance (secondary).** Compute degree- and Laplacian-spectrum MMD between grids, but **fix the two known defects first**:
   - build descriptors so the kernel is not saturated (retune `sigma_degree`, `sigma_laplacian`; the default `sigma_laplacian=1e-2` collapses every pair to √2);
   - compute topology on the **physical one-line graph, not the Ybus pattern with self-loops** (PowerGraph `edge_index = find(Ybus)` includes the diagonal).
5. **g-score = provisional.** Report it, but flag that with **one topology per grid** and only 3–4 grids the g-score is fit to 3–4 points and is statistically fragile; it is *not* the headline. Use `get_generalization_score_raw` (no percentile trim) given the tiny sample.
6. **Validity checklist:** confirm (a) normalization harmonized, (b) MMD non-degenerate, (c) mask identical across pairs, (d) no target leakage from the training grid's scaling.

## Deliverables
- NRMSE transfer matrix per architecture (the headline).
- Within-grid vs unseen-grid gap table (benchmark against PowerGraph).
- Provisional MMD/g-score with explicit caveats.
- A short validity note stating what Layer 1 can and cannot conclude.

## Threats to validity (Layer 1)
- **Normalization mismatch** (fixed by step 1) — the dominant risk.
- **Single topology per grid** → g-score ill-posed (resolved only in Layer 2).
- **MMD on admittance graph with self-loops** → distorts "topological distance."
- **Small number of grids** → weak statistics for any distance-based summary.

---

# LAYER 2 — Well-posed generalization benchmark with a topology distribution

## Research questions
**RQ2 (primary — operational):** Across a **distribution of credible transmission topologies** (a base grid + its N-1/N-k contingencies), which GNN architectures **stay accurate on unseen topologies**, and how does that error scale with topological distance (MMD) from the training distribution? Does the GNN beat the DC-PF baseline, per quantity?

Sub-questions:
- **RQ2a:** Does a physically consistent, ENGAGE-format dataset (per-unit, bus-type NaN masking, `dc_pf` baseline) change the architecture ranking vs Layer 1?
- **RQ2b:** How does each architecture's g-score compare (both the cross-context g-score and the better-posed **OOD g-score** over held-out grids), and does edge-awareness (GAT/GIN/Transformer/NNConv using `edge_attr`) help on transmission grids?
- **RQ2c (secondary — scientific stress test):** Out-of-distribution across *different* grids — leave-one-grid-out (train on three grids, test on the fourth, incl. IEEE24↔UK). Reported as a limit test, not the operational headline.
- **RQ2d:** Per-quantity behaviour — is the apparent accuracy driven by trivially-bounded **V**, and how do the harder **θ** and **Q** generalize?

## Experimental setup
- **Data:** regenerated from PowerGraph's `System.m` into **ENGAGE `Data`** (Decision 1/4/5), with a **distribution of topologies per grid** produced by contingency perturbation (see methodology). Operating points via **Route B** (real hourly demand) and/or Route A.
- **Model zoo (unified, ENGAGE interface):** `GCN`, `ARMA_GNN`, `GAT`, `GIN`, `TRANSFORMER`, `NNConv` — all with input dim 7, output dim 4, `edge_attr` dim 4, ENGAGE masking, and the per-bus-type `inference()` re-injection.
- **Experiments:** ENGAGE's **Cross-Context** (ordered train-grid/test-grid) and **Out-of-Distribution** (leave-one-grid-out) scripts, unchanged.
- **Seeds:** multiple seeds per configuration for error bars.

## Methodology — data generation engine
1. **Grid model:** convert each `System.m` → pandapower net via Octave + `from_mpc` (Decision 5); commit the `.mat`.
2. **Sample a credible topology (contingency):** remove line(s)/branch(es) — N-1, then N-2/N-k — optionally generator outages. Reject islanding (or handle islands); retune disconnection probabilities for meshed transmission.
3. **Set demand:** real hourly profile (`hourlyDemandBus.mat`, Route B) or sampled (Route A).
4. **Re-solve the physics — the re-solve engine.** A topology change invalidates all stored node values, so each sample is a fresh solve:
   ```python
   import pandapower as pp
   net.line.at[idx, "in_service"] = False   # the outage
   net.load["p_mw"], net.load["q_mvar"] = demand_p, demand_q
   pp.runpp(net)                            # AC power flow (Newton-Raphson)
   # net.res_bus.vm_pu / va_degree, net.res_gen.p_mw/q_mvar, net.res_line...
   ```
   Use `pp.runpp` (slack absorbs imbalance) or `pp.runopp` (generator re-dispatch, more realistic post-contingency); `pp.rundcpp` for the `dc_pf` baseline. This runs in **ENGAGE's** pandapower pipeline (`graph_gen.py` + `powerdata-gen`), not PowerGraph's MATLAB `gendataopf.m`.
5. **Filter:** drop non-converged / islanded / voltage-violating / overloaded solutions.
6. **Convert:** `get_node_features` + `get_edge_features` → ENGAGE `Data` (per-unit, bus-type one-hot, NaN masking, `dc_pf`).
7. Repeat → each grid becomes a **cloud of graphs with varying topology + loading** = the distribution the MMD/g-score requires.

### Optional — harvest contingencies from PowerGraph-Graph
PowerGraph-**Graph** is a cascading-failure dataset: each sample is an outage state (removed lines), `exp.mat` marks the triggering branch(es), and `of_*` labels demand-not-served. Use it to make outages **credible and grid-specific**:
- harvest the **outage line-sets** (topologies only) instead of blind random N-k;
- **stratify** sampling toward consequential contingencies (those causing DNS) to widen the MMD range;
- build a **curriculum** benign N-1 → severe cascades.
Caveats: use only their **topology**, then **re-solve AC PF yourself** (step 4) for node targets; drop cascade end-states that are islanded/blackout (no converged single-grid PF).

## Methodology — evaluation
- **Per-unit normalization** and **ENGAGE bus-type NaN masking + norm-weighted MSE** throughout (Decision 6).
- **Metrics:** `nrmse_range` **broken out per quantity (V, θ, P, Q)** as well as aggregate; degree + Laplacian **MMD** on the **physical** topology with tuned sigmas; **g-score** now well-posed because each grid is a distribution of topologies.
- **Baselines:** always report the **DC-PF baseline** (`test_dc_pf`), optionally a warm-started single Newton step, so improvement over trivial physics is explicit.
- **Distance:** MMD is primary; optionally add an **electrical-distance** cross-check (X/R or short-circuit-ratio distribution distance, or PTDF-based) since MMD ignores impedances/loading.
- **Two g-score flavours** (both produced by `experiments.py`):
  - **Cross-context g-score** (`gscore.csv`) — *per training grid* over its unseen TEST grids. At only 3 points/training grid the ENGAGE 2/98 trim collapses it, so a no-trim `gscore_smallN.csv` is the appropriate reading.
  - **OOD g-score** (`gscore_ood.csv`, `compute_ood_gscores`) — *per model* over the held-out grids of the leave-one-grid-out experiment (one point per held-out grid, up to 4), with the topological distance = **mean Laplacian-MMD from each held-out grid to its training grids**, no trim, NaN cells dropped. This is the **better-posed** g-score at N=4 and the one most aligned with the operational question (generalize to a new grid after training on several); mirrors ENGAGE reporting a g-score for both its CC and OOD experiments.
- **Cross-Context matrix** and **OOD leave-one-out** results per architecture, with seeds → error bars.
- **Benchmark vs PowerGraph:** compare within-topology (PowerGraph regime) to unseen-topology (this study) for the shared architectures, reported as **relative degradation** under our own consistent pipeline (numeric values are not directly comparable across the two masking/normalization conventions).

## Deliverables
- ENGAGE-format transmission datasets (four grids, topology distribution) + committed `.mat` cases.
- Full architecture comparison: Cross-Context + OOD g-scores, NRMSE transfer matrices, edge-awareness ablation.
- Reproducible pipeline (Octave conversion committed; pandapower-only downstream).

## Threats to validity (Layer 2)
- **Contingency realism / connectivity** — reject islanding, tune outage depth so descriptors actually spread.
- **PF vs OPF post-contingency** — document the choice (slack absorption vs re-dispatch); it affects targets and post-contingency realism (real systems re-dispatch via AGC/OPF).
- **Metric inflation by V** — aggregate NRMSE can look strong purely from tightly-bounded voltages; per-quantity reporting (RQ2d) guards against this.
- **Topological vs electrical distance** — MMD captures pure structure, not impedance/loading; the optional electrical-distance cross-check mitigates over-interpretation.
- **Sigma/kernel tuning** for MMD — validate against ENGAGE's `ggme` reference.
- **Demand coverage** (Route B) — ensure the hourly profile spans seasonal/daily range.

---

# Summary
- **Layer 1** answers "does a within-grid-trained GNN transfer to an unseen grid, and by how much?" via a **per-unit-normalized cross-grid NRMSE transfer matrix**, reusing existing models. The g-score here is provisional because there is only one topology per grid.
- **Layer 2** builds the **distribution of topologies** (contingency re-solves in ENGAGE's pandapower pipeline, optionally informed by PowerGraph-Graph) so the **g-score/MMD generalization study becomes well-posed**, and compares the full architecture zoo apples-to-apples against PowerGraph's within-grid benchmark.
