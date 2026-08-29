# A2 reassessment: do node features and targets need normalization?

Audit item A2 says "node features and targets are never normalized" and calls it
critical. My paper reading then said "neither source paper normalizes node
features", which appeared to contradict it. This document resolves the tension
with measurements on our own tensors, a second reading of both source code bases,
and the released code of four other AC power-flow GNN studies.

**Verdict: A2 is a real problem in this study, but not for the reason the audit
gives, and the audit's proposed fix (per-unit conversion) would not fix it.**

* The audit's *diagnosis* — no scaling of node features/targets — is factually
  correct (verified again, there is no scaler anywhere in the repo).
* The audit's *implied fix*, converting to per unit, is a no-op here: all four
  cases already carry `baseMVA = 100`, so p.u. conversion is a division of P and Q
  by a single constant. It changes no ratio, no learnability, and no cross-grid
  spread.
* The consequence that actually matters is **intra-sample**, not cross-grid:
  in raw units the loss gives voltage magnitude a gradient share of
  `5e-8` (IEEE24) to `1e-11` (UK). Voltage magnitude is effectively not
  optimized at all, which is exactly what our own per-quantity results show.
* My earlier statement "neither paper normalizes" is **wrong for PowerGraph-Node**.
  Its released code max-abs-normalizes X *and* Y and trains in normalized space.
  Correction recorded in §2.2.

---

## 1. What the four concepts are (they keep getting conflated)

| # | Concept | Purpose | Affects gradients? | Affects interpretability? |
|---|---|---|---|---|
| 1 | **Per-unit conversion** (÷ `baseMVA`, angles in rad) | engineering representation, dimensionless quantities | only via a constant | no — p.u. *is* the field's native unit |
| 2 | **Statistical normalization of X/Y** (z-score, max-abs, range) | conditioning: put the four output dimensions on comparable scales so each gets gradient | yes, decisively | no, *if* predictions are de-normalized before scoring |
| 3 | **Metric normalization** (ENGAGE `nrmse_range`) | comparability of reported error across dimensions/grids | no | it is the report |
| 4 | **Cross-grid magnitude alignment** (per-grid base: nominal load, peak demand) | make grids of different size numerically comparable so transfer error is not dominated by size | yes | needs an explicit, physically meaningful base |

We already do 3. We do not do 1, 2 or 4. Only **2** is a defect; **1** is
cosmetic here; **4** is a scientific-claim question.

---

## 2. What the field actually does (code-level evidence)

### 2.1 ENGAGE (the framework we follow)
`engage/graph_gen.py`:

```python
# x: np.array([Slack?, PV?, PQ?, p_mw, q_mvar, vm_pu, va_degree])
# y: np.array([p_mw, q_mvar, vm_pu, va_degree])
```
Node features/targets are raw pandapower `res_bus` values; **edges** are converted
to per unit (`z = vn_kv**2 / net.sn_mva`). No scaler. Confirmed again, and
consistent with the paper.

Two facts explain why this is harmless *for them* and harmful *for us*:

* Their grid population is SimBench **LV and MV distribution feeders**
  (`graph_utils.get_dist_grid_codes` filters `-LV-`/`-MV-`; `dc_pf_data.csv` rows
  are `1-LV-rural1--1-no_sw`, …). Bus injections in an LV feeder are on the order
  of 1e-3…1e-1 MW — i.e. *the same order as* `vm_pu ≈ 1.0`. All four output
  dimensions are naturally within one or two orders of magnitude of each other,
  so no dimension is starved of gradient. (Order-of-magnitude inference from the
  grid class; `simbench` is not installable in this session, so it is not a
  measurement.)
* Their loss already contains a scale correction:
  `weighted_mse_loss` weights each node's MSE by `1/||y||`, with the code comment
  *"To give equal importance to smaller and larger vectors"*. That is
  per-sample magnitude normalization done in the loss instead of in the data. It
  equalizes *buses*, not *dimensions* — which is precisely the gap that bites us.

### 2.2 PowerGraph-Node (correction to my earlier claim)
`PowerGraph-Node-main/code/dataset/powergrid.py`, node-PF and node-OPF branches:

```python
maxsX, _ = torch.max(torch.abs(fullXcat), dim=0)
maxsY, _ = torch.max(torch.abs(fullYcat), dim=0)
...
N_norm = N / maxsX
Y_norm = Y_o / maxsY
data = Data(x=N_norm, ..., y=Y_norm, edge_attr=edge_attr, maxs=maxsY, mask=mask)
```
with `edge_attr = torch.nn.functional.normalize(edge_attr, dim=0)`, training and
MSE computed in normalized space, and de-normalization only for reporting
(`train_gnn.py`: `denpreds.append(batch_preds.squeeze() * batch.maxs)`).

So **PowerGraph-Node does exactly concept 2**: per-dimension max-abs scaling of
features and targets, dataset-wide, with physical-unit reporting. The paper never
mentions it — which is why my paper-only reading got it wrong. This is the
benchmark our grids and demand profiles come from.

### 2.3 Other AC power-flow GNN work (released code, GitHub reachable)

| Study | Features/targets | Evidence |
|---|---|---|
| Hansen et al., *Power Flow Balancing with Decentralized GNNs* (T-PWRS 2023) — the source of our ARMA setup | **per unit**: `bus_data[:, [2,3,4,5]] /= base_mva`, `gen_data[:, 1] /= base_mva`, branch flows /= base_mva; shift angle converted to radians | `generate_data_example/utils.py:160-168` |
| Lin et al., *PowerFlowNet* (Applied Energy 2024) | **z-score** on node features, targets *and* edge attributes; statistics taken from the training split and passed to val/test; de-normalized for evaluation | `datasets/PowerFlowData.py:_normalize_dataset`, `test.py` `pre_loss_fn=partial(denormalize, …)` |
| KIT-IAI, *Augmented Pre-trained GNNs* (pretraining + cold-start on unseen SimBench grids) | **both**: p.u. at generation (`net.res_bus['p_mw'] / net.sn_mva`) *and* z-score with train statistics | `dataset_generator.py:232-235`, `datasets/PowerFlowData.py:149-169` |
| *Power flow analysis via typed GNNs* | per unit throughout (`/ baseMVA` on Pd, Qd, Pg, shunts) | `TGNN_PF/FuncDeltas.py:60-76` |

Standard practice in the field is therefore: **p.u. as the representation, plus a
per-dimension statistical scaler for training, with physical-unit reporting.**
ENGAGE is the exception, and it gets away with it because of §2.1.

---

## 3. Is it technically necessary *here*? Measurements

### 3.1 The four target dimensions differ by 5-6 orders of magnitude
`data_a` train split, per grid (min…max, std):

| grid | P [MW] | Q [Mvar] | V [p.u.] | theta [deg] |
|---|---|---|---|---|
| IEEE24 | -660 … 1 625 (σ 288) | -610 … 128 (σ 92) | 0.936 … 1.050 (σ 0.028) | -10 … 56 (σ 13) |
| IEEE39 | -830 … 2 110 (σ 410) | -1 800 … 184 (σ 234) | 0.801 … 1.064 (σ 0.057) | -12 … 100 (σ 22) |
| IEEE118 | -6 081 … 296 (σ 430) | -3 526 … 102 (σ 202) | 0.830 … 1.052 (σ 0.027) | -180 … 180 (σ 33) |
| UK | -11 331 … 37 239 (σ 5 420) | -13 310 … 766 (σ 1 709) | 0.970 … 1.000 (σ 0.005) | -6 … 168 (σ 36) |

### 3.2 Voltage magnitude receives ~1e-8 of the training gradient
Share of `weighted_mse_loss` contributed by each target dimension, evaluated at
the per-dimension-mean predictor on the training split:

| grid | P | Q | V | theta |
|---|---|---|---|---|
| IEEE24 | 0.831 | 0.148 | **5.3e-08** | 0.0205 |
| UK | 0.681 | 0.319 | **1.2e-11** | 4.4e-04 |

A 0.05 p.u. voltage error and a 100 MW active-power error are both physically
serious; in this loss the second is ~1e6 times more important. Voltage magnitude
is not being trained.

### 3.3 The prediction results match the diagnosis exactly
Per-quantity NRMSE, `results/analysis/per_quantity.csv` (Regime A, in-distribution):

| model | P | Q | V | theta |
|---|---|---|---|---|
| arma_gnn | 0.0002 | 0.0009 | **5.98** | 0.0128 |
| gin | 0.0003 | 0.0049 | **7.69** | 0.0250 |
| nnconv | 0.0005 | 0.0057 | **5.78** | 0.0225 |
| transformer | 0.0017 | 0.0086 | **24.91** | 0.0508 |
| gat | 0.0055 | 0.0127 | **27.27** | 0.0582 |
| gcn | 0.0077 | 0.0177 | **7.84** | 0.0787 |

NRMSE > 1 means the error exceeds the full observed range of the quantity. Every
architecture is off by 6-27 ranges on V while being near-exact on P — and, as the
audit noted, all of them are beaten on voltage by the constant predictor
`V ≡ 1.0` (MAE 0.0245 p.u.). This is not an architecture finding; it is the
consequence measured in §3.2. **This is why normalization is technically
necessary in this study.**

### 3.4 Per-unit conversion would fix none of it
All four cases have `net.sn_mva = 100.0`. Dividing P and Q by 100:
* leaves every ratio in §3.1-§3.3 unchanged;
* changes the V gradient share from 5.3e-08 to 5.3e-04 (IEEE24) — still negligible;
* leaves the cross-grid spread untouched.

Per-unit is worth doing for convention-compliance with Hansen/PowerFlowNet, but it
is not the remedy for A2.

### 3.5 Cross-grid magnitude: real, but a *claim* problem, not a units problem
Nominal total load: IEEE24 2 850 MW, IEEE39 6 254 MW, IEEE118 3 733 MW,
**UK 56 326 MW** — 9-20× larger. Target σ(P) spans 288 → 5 420 (19×). No unit
system removes this: the UK grid genuinely moves an order of magnitude more power.
Consequences:
* Transfer error today mixes **topology shift + operating-point shift + size
  shift**. The sentence "cross-grid degradation measures topology
  generalization" is not supported, exactly as the audit says.
* A scaler fitted on the training grid(s) and applied unchanged at transfer time
  (PowerFlowNet's protocol) is the *honest* option: it keeps the size shift inside
  the OOD shift, where it physically belongs, and does not leak test-grid
  statistics. But then the metric must be described as "generalization to an
  unseen system", not "to unseen topology".
* Isolating topology requires a **per-grid physical base** — e.g. P, Q divided by
  that grid's nominal total load, which is nameplate input data known at
  inference, so no label leakage. This is the only variant that can support a
  topology-generalization claim.
* PowerGraph-Node's own choice (max-abs per dataset) is a per-grid scaler, and so
  is not transferable as-is: fitted on the target grid it leaks, fitted on the
  source grid it is arbitrary.

---

## 4. Would normalizing damage the engineering grounding?

No, provided three rules hold — all three are what the literature does:

1. **Report in physical units.** Scale for optimization, de-normalize predictions
   before computing any reported metric (PowerGraph-Node: `preds * batch.maxs`;
   PowerFlowNet: `pre_loss_fn=denormalize`). Then every number in the results
   tables keeps its MW / Mvar / p.u. / degree meaning, and the DC baseline and
   `nrmse_range` comparisons stay valid and ENGAGE-comparable.
2. **Fit on training data only**, never on the evaluation grid, and carry the
   statistics with the checkpoint.
3. **Prefer a physically defined base over a statistical one where one exists**
   (`baseMVA` for power, radians for angles, nominal load for cross-grid size).
   A statistical scaler is then only correcting the residual conditioning
   problem, not inventing the representation.

The real risk is the opposite of the audit's worry: *not* normalizing has already
damaged the engineering grounding, because our models do not learn bus voltage
magnitude at all, and voltage is the quantity a power engineer cares about most.

---

## 5. Options, cost, and recommendation

| Option | What it fixes | Cost | Verdict |
|---|---|---|---|
| **N0** Do nothing, document | nothing | 0 | Not acceptable: §3.3 is a real defect, and §3.5 invalidates a headline claim. |
| **N1** p.u. representation only (`/baseMVA`, angles in rad) | convention compliance | full retrain | Necessary-but-insufficient; do it as part of N2, not alone. |
| **N2** p.u. + per-dimension scaler fitted on the training split, de-normalized reporting (PowerFlowNet / PowerGraph-Node protocol) | §3.2-§3.3 (voltage learnability) | full retrain of 6 archs × 3 arms | **Recommended primary.** It is the field standard and it is the fix that matches the measured failure. |
| **N3** N2 + per-grid nominal-load base for P/Q | additionally §3.5 (size shift), enables a topology-generalization claim | second full retrain | **Recommended as the OOD-isolating ablation**, not as the primary, because it deviates from both source papers. |
| **N4** Keep raw-unit results as the ENGAGE-replication headline, add N2 as an ablation on a subset (2 archs × 2 seeds, within + cross-context) | quantifies the effect without a full retrain | hours, not days | **Recommended immediate step**, because it produces the evidence needed to decide whether the full retrain is worth it. |

### Recommended sequence
1. **N4 pilot** (cheap): implement N2 behind a flag (`--normalize {none,pu,pu_zscore}`),
   default `none` so all existing artifacts stay bit-identical and reproducible;
   run gcn + arma, seeds 0/100, within-grid + cross-context. Deliverable: does
   per-quantity V NRMSE drop below 1, and does the cross-context ranking change?
2. If the pilot confirms the mechanism (it should, per §3.2), **N2 full retrain**
   becomes the primary protocol and the raw-unit run is reported as an ablation
   showing what unnormalized training does to voltage — a genuine contribution,
   since ENGAGE's LV setting hides this effect.
3. **N3** afterwards, only for the OOD arm, to support or retire the
   topology-generalization wording.
4. Until 1-3 land, the claims in `docs/Regime_comparison_results.md` must say
   "generalization to an unseen system" and must not attribute cross-grid
   degradation to topology alone. Voltage results must be reported as "not
   learned under the raw-unit protocol", not as an architecture comparison.

Nothing in this document has been implemented; no retraining has started.

## 6. Evidence provenance

* Verified from released code, re-read in this session: ENGAGE `graph_gen.py`,
  `training_utils.py`, `models.py`, `graph_utils.py`; PowerGraph-Node
  `code/dataset/powergrid.py`, `code/train_gnn.py`; Hansen
  `generate_data_example/utils.py`; PowerFlowNet `datasets/PowerFlowData.py`,
  `train.py`, `test.py`; KIT-IAI `dataset_generator.py`,
  `datasets/PowerFlowData.py`.
* Measured on our tensors in this session: §3.1, §3.2, §3.4 (`sn_mva`, nominal
  load), §3.5.
* Read from our own artifacts: §3.3 (`results/analysis/per_quantity.csv`).
* **Not** independently verified: the absolute magnitude of SimBench LV/MV bus
  injections (§2.1) — `simbench` cannot be installed here (PyPI is outside the
  session allowlist), so that is an order-of-magnitude argument from the grid
  class, not a measurement. Paper PDFs for the four external studies are also
  unreachable; all four claims above are code-level.
