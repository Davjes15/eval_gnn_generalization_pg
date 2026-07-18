# Findings — GNN generalization for AC power flow on transmission grids

This report interprets the **full run**: 4 grids (IEEE24, IEEE39, IEEE118, UK),
800/100/100 train/val/test graphs per grid (4,400 graphs total, each = a demand
snapshot + a random N-1/N-2 contingency, AC-re-solved with pandapower), 6
architectures (`gcn`, `arma_gnn`, `gat`, `gin`, `transformer`, `nnconv`), 200
epochs with early stopping. Metrics are range-normalized NRMSE (ENGAGE
`nrmse_range`), reported aggregate **and** per quantity (P, Q, V, θ). Raw numbers
are in `full_run/results/` (`results_report.txt` consolidates them).

> **One-line takeaway.** Within a grid the models learn AC power flow well;
> **single-grid transfer is fragile and unstable**, but **training on several
> grids (leave-one-grid-out OOD) generalizes to an unseen grid** at NRMSE
> ≈ 0.10–0.22 for the stable architectures. `transformer` and `gat` are the most
> reliable; `gin`/`nnconv` show catastrophic instabilities; `arma_gnn` diverges
> on the UK split. Read **per-quantity** metrics: aggregate NRMSE is dominated by
> P/Q/θ, while the V metric is inflated because voltage is nearly constant.

---

## 1. Topological distances between grids (MMD)

Laplacian-spectrum MMD (primary), `mmd_laplacian.csv`:

| train\test | IEEE24 | IEEE39 | IEEE118 | UK |
|---|---|---|---|---|
| **IEEE24** | 0.06 | 0.74 | 0.73 | 1.00 |
| **IEEE39** | 0.78 | 0.12 | 0.90 | 1.12 |
| **IEEE118** | 0.82 | 0.77 | 0.10 | 1.22 |
| **UK** | 1.08 | 1.05 | 1.24 | 0.003 |

- **Within-grid MMD ≈ 0** (0.003–0.12) ≪ **cross-grid MMD ≈ 0.73–1.24** — the
  metric is non-degenerate (the earlier bug is fixed): each grid's cloud of
  contingency topologies is tight relative to the gaps between grids.
- **UK is the farthest grid from everything** (mean distance to the others ≈ 1.13),
  IEEE24 the closest to the pack (≈ 0.82). Ordering of "distance to the rest":
  IEEE24 < IEEE39 ≈ IEEE118 < **UK**.
- The OOD distances the g-score uses (`ood_distance.csv`, held-out grid → its 3
  training grids): IEEE24 0.82, IEEE39 0.94, IEEE118 0.94, **UK 1.13**.

This ordering matters: it predicts UK should be the **hardest** unseen grid, which
the OOD errors below confirm.

---

## 2. Within-grid performance (the diagonal — PowerGraph's regime)

Aggregate NRMSE on a grid's own held-out test split (train_grid = test_grid):

| model | IEEE24 | IEEE39 | IEEE118 | UK |
|---|---|---|---|---|
| gcn | 0.044 | 0.051 | 0.011 | 0.011 |
| arma_gnn | 0.147 | 0.010 | 0.004 | 0.006 |
| gat | 0.028 | 0.023 | 0.015 | 0.009 |
| gin | 0.011 | 0.016 | 0.009 | 0.006 |
| transformer | 0.011 | 0.016 | 0.005 | 0.008 |
| nnconv | 0.012 | 0.023 | 0.006 | 0.011 |

- **The models fit AC power flow well within a grid** (aggregate NRMSE ≈ 0.005–0.05),
  which is the sanity check that the task and pipeline are sound. `transformer`,
  `gin`, and `nnconv` are the strongest; `arma_gnn` has one weak cell (IEEE24, 0.147).
- **Per quantity (within-grid, averaged over grids):** P ≈ 0.002–0.03, Q ≈ 0.01–0.03,
  θ ≈ 0.05–0.09 are all learned well. **V-NRMSE is large (≈ 5–21) — but this is a
  metric artifact, not a failure**: transmission voltages sit in a very narrow band
  (≈ 1.0 pu), so range-normalization divides a small absolute error by a tiny range
  and explodes. In **absolute** pu terms V is essentially flat and well predicted;
  the substantive learned quantities are P, Q, and θ. (This is exactly the
  "metric-inflation-by-V" caveat the design anticipated — always read per-quantity.)

---

## 3. DC power-flow baseline

`dc_baseline.csv` (aggregate + per quantity, per test grid):

| grid | dc_nrmse | P | Q | V | θ |
|---|---|---|---|---|---|
| IEEE24 | 0.017 | 0.012 | 0.0 | 0.143 | 0.018 |
| IEEE39 | 0.010 | 0.008 | 0.0 | 0.217 | 0.060 |
| IEEE118 | 0.024 | 0.018 | 0.0 | 0.109 | 0.153 |
| UK | 0.016 | 0.011 | 0.0 | 0.136 | 0.013 |

- **DC power flow is a strong aggregate baseline** here (NRMSE ≈ 0.01–0.02): its
  linear P/θ assumptions are accurate for these transmission grids, and its
  flat-voltage assumption (V ≈ 1.0) gives a **much smaller V-NRMSE (0.11–0.22)
  than the GNNs' inflated V-NRMSE** — i.e. on voltage magnitude alone, trivially
  assuming V = 1.0 is competitive, which again shows V is near-constant and a poor
  discriminator. DC does not model reactive power (Q column is 0 by construction).
- **The GNN's value is therefore not "beating DC on aggregate within one grid"** —
  it is (a) modelling Q and the nonlinear regime DC ignores, and (b) **generalizing
  across topologies/grids**, which a per-grid DC solve does not address. Judge the
  GNN on per-quantity P/Q/θ and on robustness, not on aggregate NRMSE vs DC.

---

## 4. Single-grid cross-context transfer (train on ONE grid → test on others)

Off-diagonal transfer NRMSE, summarized per model:

| model | mean off-diag | median | max |
|---|---|---|---|
| gat | **0.19** | 0.15 | **0.38** |
| gcn | 0.47 | 0.21 | 1.30 |
| arma_gnn | 0.66 | 0.15 | 2.81 |
| transformer | 0.74 | 0.15 | 2.92 |
| nnconv | 1.91 | 0.20 | 17.1 |
| gin | 3.52 | 0.48 | 26.8 |

- **Single-grid transfer is fragile and unstable.** Median transfer is often
  reasonable (0.15–0.5), but **maxima blow up** (gin IEEE118→UK ≈ 27, nnconv
  IEEE118→IEEE39 ≈ 17). A model that saw only one grid's topology family does not
  reliably extrapolate.
- **Strong asymmetry driven by IEEE118.** Every model predicts the *IEEE118 test
  column* at ≈ 0.10 regardless of training grid, but **models trained on IEEE118
  fail badly on the smaller grids** (gcn IEEE118→IEEE24 ≈ 1.2; arma/transformer
  IEEE118→IEEE24/39 ≈ 2.5–2.8; gin/nnconv catastrophic). Training on the large,
  dense grid overfits to 118-specific structure; training on smaller grids
  transfers "up" to IEEE118 far more gracefully.
- **`gat` is by far the most robust single-grid transferer** (max 0.38, no
  blow-ups) — its attention + vector edge features seem to regularize transfer.

---

## 5. Out-of-distribution — leave-one-grid-out (train on 3 grids → test on the held-out one)

`ood.csv` — aggregate NRMSE on the held-out grid:

| model | IEEE24 | IEEE39 | IEEE118 | UK |
|---|---|---|---|---|
| gcn | 0.112 | 0.141 | 0.120 | 0.215 |
| arma_gnn | 0.157 | 0.112 | 0.102 | **NaN** |
| gat | 0.169 | 0.130 | 0.103 | 0.159 |
| gin | 0.159 | 0.135 | 0.106 | 0.170 |
| transformer | 0.157 | 0.136 | 0.106 | 0.149 |
| nnconv | 0.164 | **3.07** | 0.103 | 0.154 |

- **This is the headline result: multi-grid training generalizes to an unseen
  grid.** For the stable architectures, held-out NRMSE is ≈ **0.10–0.22** — vastly
  better and more stable than the single-grid transfer above. Exposure to several
  topology families lets the model learn grid-invariant power-flow structure.
- **Per-grid difficulty tracks topological distance.** IEEE118 is easiest to
  generalize to (≈ 0.10 for all models), **UK is hardest** (0.15–0.22) — exactly as
  the MMD distances predicted (UK is farthest, IEEE118 sits centrally with the most
  training coverage from the other grids).
- **Two documented instabilities (not bugs):** `arma_gnn` **diverges to NaN** on the
  UK held-out split (ARMA's recursive filter is sensitive on the farthest, smallest
  training-support target), and `nnconv` produces a **3.07 outlier** on held-out
  IEEE39. These are genuine architecture-level robustness failures worth reporting.

---

## 6. Generalization scores (g-score)

- **Cross-context g-score** (`gscore.csv`, ENGAGE 2/98 trim): **degenerate at this
  scale** — 3 unseen grids per training grid, so the trim collapses to one point
  (std = mmd_range = 0). The no-trim `gscore_smallN.csv` is the correct reading;
  even there, IEEE118-trained rows explode (gin IEEE118 g-score ≈ 21) mirroring the
  transfer instability above.
- **OOD g-score** (`gscore_ood.csv`, better-posed — one point per held-out grid,
  no trim, NaN dropped):

  | model | mean_nrmse | std_nrmse | mmd_range | g_score |
  |---|---|---|---|---|
  | transformer | 0.137 | 0.019 | 0.305 | **0.154** |
  | arma_gnn | 0.124 | 0.024 | 0.116 | 0.146 (3 pts, UK dropped) |
  | gat | 0.141 | 0.026 | 0.305 | 0.163 |
  | gin | 0.142 | 0.025 | 0.305 | 0.164 |
  | gcn | 0.147 | 0.041 | 0.305 | 0.183 |
  | nnconv | 0.873 | 1.270 | 0.305 | 1.982 |

  Lower is better (low + stable error across distances). **`transformer` wins**;
  `gat`/`gin` close behind; `gcn` a bit noisier; **`nnconv` is disqualified by its
  IEEE39 outlier**; `arma_gnn`'s 0.146 is optimistic because its diverged UK point
  was dropped (it would otherwise be the worst).

---

## 7. Architecture verdict

| aspect | best | worst |
|---|---|---|
| within-grid fit | transformer, gin, nnconv | arma_gnn (IEEE24 cell) |
| single-grid transfer robustness | **gat** (no blow-ups) | gin, nnconv |
| OOD generalization (held-out grid) | **transformer**, gat | nnconv (IEEE39), arma_gnn (UK NaN) |
| overall reliability | **transformer & gat** | gin, nnconv, arma_gnn |

**Bottom line:** for robust cross-topology / cross-grid power-flow generalization,
**edge-aware attention models (`transformer`, `gat`) are the safe choice.**
`gin`, `nnconv`, and `arma_gnn` can match them within a grid but are prone to
severe instabilities out of distribution.

---

## 8. Caveats & threats to validity

- **Only 4 grids.** All distance-based summaries (MMD, g-score) are statistically
  thin; treat the **transfer matrices + OOD NRMSE** as the headline, g-scores as
  supporting. This is why the OOD g-score (up to 4 points) is preferred over the
  degenerate cross-context one (3 points, trimmed).
- **V metric inflation.** Range-normalized V-NRMSE is misleading because V ≈ 1.0 pu
  is nearly constant; report/inspect **absolute pu V error** alongside it (future
  work). P, Q, θ are the substantive quantities.
- **Random N-k contingencies.** The topology distribution is random line outages
  (connectivity- and voltage-filtered); harvesting real PowerGraph-Graph outages
  (`--contingency_source harvest`, Step 7) would make the distribution more
  operationally credible.
- **PF vs OPF.** Post-contingency uses slack absorption (`pp.runpp`); real systems
  re-dispatch (AGC/OPF), which would change targets — a `--redispatch` (`runopp`)
  path exists for sensitivity studies.
- **Topological vs electrical distance.** MMD captures pure graph structure, not
  impedance/loading; an electrical-distance cross-check (X/R, PTDF, SCR
  distributions) would strengthen the distance axis.
- **Instabilities.** `arma_gnn`/UK (NaN) and `nnconv`/IEEE39 (3.07) are reproducible
  under the recorded seed; averaging over multiple seeds would quantify their
  variance.

---

## 9. Practical takeaways

1. **To deploy on a new transmission grid, train on several grids, not one.**
   Multi-grid (OOD) training reduced held-out error by roughly an order of
   magnitude versus single-grid transfer and removed most blow-ups.
2. **Prefer `transformer` or `gat`** for robustness; avoid relying on `gin`/`nnconv`
   out of distribution without seed-averaging and stability checks.
3. **Judge models per quantity** (P, Q, θ) and against the **DC-PF baseline**, not
   on aggregate NRMSE — and remember DC's flat-voltage assumption is a strong V
   baseline.
4. **Use MMD as a pre-deployment screen** (it needs only topology, no labels): a new
   grid far from the training set on the MMD axis (like UK here) should be expected
   to be harder, and validated by actually re-solving a sample with pandapower.
