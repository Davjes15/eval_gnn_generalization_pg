# 5-Minute Talk — GNN Generalization for AC Power Flow on Transmission Grids

A slide-ready script for a technical audience (~700 words ≈ 5 min). Each **beat**
is roughly one slide; suggested visuals reference figures already in the repo
(`docs/figures/`, `README.md`).

---

## Beat 1 — The hook: problem → gap → question (45s)
> **The problem.** The field is racing toward **Grid Foundation Models** — one large,
> pre-trained graph neural network meant to serve *any* grid and *any* task, from
> power flow to contingency screening (IBM's *GridFM*, Microsoft's *GridSFM* trained
> on 200 grids). The entire promise rests on a single assumption: that a GNN trained
> on some grids **transfers to grids and topologies it has never seen**. Because the
> grid is a graph that *constantly changes* — lines trip, topologies reconfigure — a
> model that only works on its training grid is useless for operations.
>
> **The research gap.** That assumption is usually *asserted*, not *measured*. Papers
> say "GNNs generalize" and scale up — but we lack controlled evidence for **which**
> graph architectures actually transfer across grids, by **how much**, and **how to
> even measure** distance between grids of different sizes. Without that, a Grid
> Foundation Model is scaling an unvalidated core.
>
> **The research question.** So, concretely: *Which GNN architectures generalize
> across unseen transmission topologies and unseen grids — and does training on
> multiple grids actually deliver the cross-grid transfer a foundation model needs?*
> We answer it on a task where the ground truth is exact — **AC power flow** — so any
> failure to generalize is unambiguous, not hidden by label noise.

## Beat 2 — The setup / methodology (45s)
> Why learn a surrogate at all, if power flow is exact? **Amortization** — a solver
> re-runs from scratch for every scenario, but a trained GNN answers instantly, which
> is what makes screening thousands of topologies feasible. Here is the methodology.
> Four transmission grids — IEEE 24, 39, 118-bus, and a UK model. Each grid is not
> one graph: we generate a **distribution** of topologies by sampling demand
> snapshots and N-1/N-2 contingencies, then re-solve AC power flow for ground
> truth. That is ~1,000 graphs per grid, **4,000 total**. Every bus predicts four
> numbers: active power **P**, reactive power **Q**, voltage magnitude **V**, and
> angle **θ**. We benchmark six architectures — GCN, ARMA, GAT, GIN,
> TransformerConv, NNConv — under **one identical training recipe**, so the *only*
> thing that varies is the architecture.

## Beat 3 — How the model works (45s)  ·  *visual: training-method diagram*
> Same skeleton for all six: **encode → process → decode**, per node. An MLP
> encodes each bus, a message-passing block — the only part that differs between
> models — mixes neighbor information over the grid, then an MLP decodes to the four
> outputs. Two physics-aware tricks: we **mask** inputs by bus type — the network is
> told which quantities are known boundary conditions and which it must infer — and
> at inference we **re-inject** the known values so predictions stay physically
> consistent. The loss is the same weighted MSE for every model.

## Beat 4 — Measuring generalization (45s)  ·  *visual: MMD heatmap*
> To ask "how far is a new grid from what I trained on?" we need a distance between
> grids of *different sizes* with no shared bus numbering. We use **MMD** on graph
> fingerprints — degree and Laplacian-spectrum histograms — which is size- and
> labeling-invariant. Then two experiments: **Cross-Context** — train on *one* grid,
> test on all — the pessimistic single-source case; and **Out-of-Distribution** —
> train on *three* grids, deploy on the held-out one — the realistic case.

## Beat 5 — The punchline results (60s)  ·  *visual: performance bar chart*
> Three findings. **One:** every model nails power flow *within* a grid — NRMSE
> under 5%. Easy part. **Two:** single-grid transfer is *fragile* — train on the big
> dense IEEE118, test on a small grid, and GIN's error explodes by **27×**, NNConv
> by **17×**. The most *expressive* models are the *least* robust. **Three — the
> headline:** train on three grids instead of one and that fragility largely
> disappears — out-of-distribution error drops to **~10–15%** and stabilizes. The
> winners are the **attention models, Transformer and GAT**: their softmax
> aggregation is scale-invariant, so they degrade gracefully when the grid changes
> size. Sum-based GIN, edge-matrix NNConv, and recursive ARMA fit beautifully but
> destabilize out of distribution — ARMA literally **diverged to NaN** on UK.

## Beat 6 — The honest caveat (30s)
> Two things a technical audience should hear. First, voltage looks "flat" — near
> 1.0 per-unit — because it *physically is*; it is regulated. That inflates a
> normalized voltage metric, so **read errors per-quantity**, not as one aggregate.
> Second, with only four grids, topological distance does **not** cleanly predict
> error — *which* grid you trained on matters more. So we lead with transfer
> matrices and raw OOD error, and treat the generalization score as supporting
> evidence, not gospel.

## Beat 7 — The close (20s)
> Bottom line: for deployable, cross-grid power-flow surrogates, **train on multiple
> grids and use attention-based GNNs**. Expressiveness wins the training set;
> **scale-invariance wins the real world.** And that is exactly the groundwork a
> **Grid Foundation Model** needs — evidence for *which* graph backbone to scale and
> proof that multi-grid pre-training actually transfers. Thank you.

---

### Suggested slide → figure map
| Beat | Figure |
|---|---|
| 3 (model) | training-method Mermaid diagram (`README.md`) |
| 4 (distance) | `figures/fig_mmd_heatmap.png` |
| 5 (results) | `figures/fig_performance.png` (the three-regime bar chart) |
| 6 (caveat) | `figures/fig_generalizability_curve.png` (weak MMD↔NRMSE correlation) |

Full detail and all figures: [`Findings.md`](Findings.md).
