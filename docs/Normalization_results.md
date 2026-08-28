# A2 remediation: normalized-representation campaign (N2)

Companion to [`Normalization_assessment.md`](Normalization_assessment.md), which
established *why* the raw-unit protocol had to change. This file records what was
implemented, the evidence that triggered the full retrain, and the results.

Status: **campaign complete.** 336 checkpoints, 672 replayed physics rows, all
six architectures in all three arms. Section 4 holds the final normalized
numbers; these supersede the raw-unit tables in `results/`, which are kept as the
documented ablation.

---

## 1. What was implemented

`normalization.py` adds an affine per-quantity scaler with three modes, selected
by `experiments.py --normalize`:

| mode | representation | purpose |
|---|---|---|
| `none` | raw MW / Mvar / p.u. / degrees | the protocol every existing artifact was produced with; **default**, so nothing already published changes |
| `pu` | P, Q on the case's `S_base`, angles in radians | engineering conversion only, isolated so its effect can be separated from the statistical one |
| `pu_zscore` | `pu`, then per-quantity z-score with **training-split statistics** | the A2 remediation, and the field-standard protocol (PowerGraph-Node max-abs, PowerFlowNet z-score, both de-normalizing for reporting) |

Protocol guarantees, each covered by a check in `tests/test_normalization.py`:

1. **No leakage.** The scaler is fitted on the training datasets of the arm that
   uses it — the single grid for within-grid and cross-context, the pooled
   training grids for leave-one-grid-out — and is applied unchanged to the
   evaluation grid. The unseen grid's statistics are never observed.
2. **Physical-unit reporting.** Predictions are de-normalized before any metric,
   and the untransformed targets travel with the sample as `y_raw`, so P, Q, V
   and theta errors stay in MW, Mvar, p.u. and degrees. Metrics are
   representation-invariant to within 1e-4 relative.
3. **Bus-type masking is preserved.** Node-feature columns 3:7 and the targets
   are scaled with the *same* statistics, which is what makes the known-value
   re-injection in `models.py::inference` legal; masked (NaN) entries stay NaN.
4. **DC baseline untouched.** `dc_pf` stays in physical units and `test_dc_pf`
   scores it against `y_raw`, so the baseline is identical across modes and
   remains comparable to the raw-unit campaign and to ENGAGE's Table 3.
5. **Bit-identical default.** Re-running gcn / IEEE24 / seed 0 / 200 epochs with
   `--normalize none` reproduces the published row in
   `results/regime_a/within_grid.csv` exactly (maximum relative difference over
   all 15 metric columns: `0.0`).

## 2. Evidence that triggered the full retrain

A 15-epoch probe on IEEE24 with gcn, all three modes, identical seed:

| mode | aggregate NRMSE | P | Q | V | theta |
|---|---|---|---|---|---|
| `none` | 0.0522 | 0.0304 | 0.0545 | **2.9735** | 0.1142 |
| `pu` | 0.0565 | 0.0341 | 0.0516 | **0.1971** | 0.1386 |
| `pu_zscore` | 0.0486 | 0.0299 | 0.0410 | **0.0208** | 0.0202 |

Voltage magnitude improves by two orders of magnitude while P and Q do not
degrade, which is the mechanism predicted in the assessment: in raw units V
contributes ~5e-8 of the training loss, so it was never optimized.

One correction to the assessment: `pu` is **not** inert for learning. It leaves
the cross-grid magnitude spread unchanged (all four cases carry
`sn_mva = 100`), which was the claim, but dividing P and Q by 100 and converting
angles to radians does rebalance the loss *within* a sample, and on its own
recovers most of the voltage error. The assessment's "per-unit conversion alone
changes nothing" should be read strictly as "changes nothing about the
cross-grid confound".

## 3. Campaign design

* All six architectures, all three arms, `--normalize pu_zscore`, frozen
  `configs/arch_config.json`, 200 epochs, seeds 0/100/300/700/1000 (nnconv
  0/100/300, as in the raw-unit campaign).
* Data: `data_a` for Regime A; **`data_full_v2`** for Regime B — the A5
  remediation (blocked temporal windows, one-day gap) landed before the Regime B
  arms were trained, so Regime B differs from the raw-unit campaign in *both* the
  representation and the split hygiene. Regime A differs only in the
  representation, and is therefore the clean A2 comparison.
* Sharded one job per (arm, architecture) through a parallel pool
  (`launch_normalized.sh`), one Torch thread per process.
* `--save_models` for every shard, into `ckpt_norm/<arm>_<model>/`, so every
  reported row is replayable from a checkpoint instead of a retrain — the gap
  that made the raw-unit ARMA and NNConv rows seed-reproducible only.
* Results land in `results_norm/<arm>_<model>/`; the raw-unit tables in
  `results/` are kept as the documented ablation and are not overwritten.

## 4. Results

All numbers from `results_norm/analysis/` (`rank_analysis.py` +
`recompute_tables.py`), aggregate NRMSE in physical units, mean over seeds.

### 4.1 Ranking per arm

| rank | Regime A (within) | cross-context | OOD (leave-one-grid-out) |
|---|---|---|---|
| 1 | arma_gnn 0.00044 | arma_gnn 0.821 | transformer 0.269 |
| 2 | gin 0.0037 | transformer 0.939 | gat 0.346 |
| 3 | transformer 0.0039 | gin 1.004 | gin 0.387 |
| 4 | gat 0.0040 | gcn 1.058 | arma_gnn 0.798 |
| 5 | nnconv 0.0040 | gat 1.105 | gcn 1.043 |
| 6 | gcn 0.0100 | nnconv 1.985 | nnconv 3.410 |

The in-distribution error is **3 orders of magnitude** below the transfer error,
and the in-distribution ordering does not predict the transfer ordering:
Kendall tau, computed per (grid, seed) and then averaged, is
**0.023 ± 0.369** for A → cross-context and **0.020 ± 0.380** for A → OOD, with
individual cells ranging from -0.73 to +0.87. That is the study's headline: the
within-grid leaderboard everyone reports carries essentially no information
about which architecture survives an unseen grid.

### 4.2 Per quantity (NRMSE, mean over seeds)

| model | A: P / Q / V / θ | cross-context: P / Q / V / θ | OOD: P / Q / V / θ |
|---|---|---|---|
| arma_gnn | 0.0002 / 0.0005 / 0.0010 / 0.0037 | 0.35 / 0.72 / 0.75 / 1.41 | 0.50 / 0.81 / 1.03 / 1.25 |
| gat | 0.0007 / 0.0050 / 0.0109 / 0.0140 | 0.40 / 1.15 / 1.21 / 3.08 | 0.19 / 0.36 / 0.91 / 2.45 |
| gcn | 0.0046 / 0.0107 / 0.0151 / 0.0231 | 0.37 / 1.01 / 2.80 / 2.08 | 0.76 / 0.61 / 1.71 / 1.30 |
| gin | 0.0004 / 0.0047 / 0.0014 / 0.0091 | 0.22 / 1.32 / 11.14 / 6.91 | 0.16 / 0.58 / 11.91 / 3.35 |
| nnconv | 0.0006 / 0.0050 / 0.0112 / 0.0144 | 0.84 / 1.43 / 1.32 / 6.88 | 2.64 / 1.20 / 0.95 / 1.88 |
| transformer | 0.0005 / 0.0049 / 0.0110 / 0.0138 | 0.32 / 1.05 / 1.11 / 2.10 | 0.14 / 0.30 / 0.93 / 1.39 |

A2 is settled by the V column: in the raw-unit campaign in-distribution voltage
NRMSE was 5.8–27, i.e. every architecture lost to the constant `V ≡ 1.0`. Under
`pu_zscore` it is 0.001–0.015 in-distribution. Voltage is now actually learned,
and the transfer degradation in V (0.75–11.9) is a property of transfer, not an
artifact of an unoptimized loss term.

GIN is the instructive case: best or near-best on P out of distribution
(0.16) and simultaneously the worst on V by an order of magnitude (11.9). Sum
aggregation makes it sensitive to the degree distribution of an unseen grid, and
the aggregate NRMSE hides that entirely — the argument for reporting the four
quantities separately.

### 4.3 Against the DC baseline

Ratio GNN / DC on the three quantities DC solves (P, V, θ); < 1 means the GNN
wins:

| model | Regime A | cross-context | OOD |
|---|---|---|---|
| arma_gnn | **0.019** | 12.0 | 13.2 |
| gat | **0.101** | 22.4 | 16.9 |
| gcn | **0.169** | 24.7 | 18.0 |
| gin | **0.043** | 87.1 | 73.5 |
| nnconv | **0.104** | 43.1 | 26.1 |
| transformer | **0.100** | 16.9 | 11.7 |

In distribution every architecture beats DC power flow by 6–50×. On an unseen
grid every architecture is beaten by DC power flow by 12–87×, and DC needs no
training data. Any deployment claim for a trained GNN on a grid it has not seen
has to clear this bar first.

### 4.4 g-scores

Cross-context aggregate (`mean + 0.806 sd`) and OOD (`mean + 0.857 sd`):

| model | cc mean | cc sd | cc g | OOD mean | OOD g |
|---|---|---|---|---|---|
| arma_gnn | 0.821 | 1.490 | 2.020 | 0.798 | 1.311 |
| gcn | 1.019 | 1.818 | 2.482 | 1.043 | 1.976 |
| transformer | 0.939 | 1.937 | 2.497 | 0.269 | 0.412 |
| gin | 1.004 | 2.103 | 2.696 | 0.387 | 0.652 |
| gat | 1.105 | 2.265 | 2.928 | 0.346 | 0.528 |
| nnconv | 1.985 | 3.133 | 4.507 | 3.410 | 6.713 |
| dc_pf | 0.067 | 0.023 | **0.067** | — | — |

As derived for A6, with four grids the MMD term collapses to a constant, so the
g-score is a monotone function of `mean + c·sd` and cannot reorder architectures
by topological distance; it is reported as a variability-penalized error, not as
a distance-aware metric.

### 4.5 Physics-aware replay (A3)

Every one of the 336 checkpoints was replayed with `eval_checkpoints.py`, scoring
**only the entries a model actually predicts** (the known bus-type values are
re-injected, so including them would flatter every model equally).
`docs/tables/physics_summary_norm.csv`, means over runs:

| arm | model | P | Q | V | θ | MAE V [p.u.] | p99 V | V-violation rate true / predicted | false alarms |
|---|---|---|---|---|---|---|---|---|---|
| within | arma_gnn | 0.004 | 0.001 | 0.002 | 0.004 | 0.0001 | 0.001 | 0.115 / 0.115 | 0.001 |
| within | gin | 0.004 | 0.009 | 0.002 | 0.010 | 0.0002 | 0.001 | 0.115 / 0.114 | 0.001 |
| within | transformer | 0.007 | 0.009 | 0.013 | 0.015 | 0.0009 | 0.018 | 0.115 / 0.119 | 0.008 |
| within | gcn | 0.050 | 0.018 | 0.019 | 0.025 | 0.0016 | 0.020 | 0.115 / 0.117 | 0.011 |
| cross | transformer | 4.28 | 1.60 | 2.78 | 2.18 | 0.098 | 0.313 | 0.141 / 0.284 | 0.250 |
| cross | gin | 2.94 | 1.93 | 29.4 | 7.07 | 0.660 | 3.371 | 0.141 / 0.384 | 0.347 |
| ood | transformer | 2.46 | 0.44 | 2.18 | 1.47 | 0.094 | 0.291 | 0.141 / 0.381 | 0.348 |
| ood | nnconv | 44.8 | 1.98 | 2.25 | 2.09 | 0.088 | 0.383 | 0.141 / 0.351 | 0.322 |

The operational reading is in the last two columns. In distribution the models
reproduce the true 11.5 % voltage-violation rate almost exactly and raise false
alarms on ~1 % of buses. On an unseen grid the true rate is 14.1 % but the models
flag 28–53 % of buses, i.e. **25–50 % of all buses are false alarms**. The errors
are conservative rather than dangerous — over-flagging, not missing violations —
but a screening tool with that false-alarm rate is unusable, which is a more
concrete statement of the transfer failure than any NRMSE.

`vm_false_secure` (missed violations) is undefined for 168 of the 672 rows
because those grids have **zero** true violations in the test split; 0/0 is
reported as NaN rather than as a rate of 0, and those rows are excluded from that
column's mean.

## 4bis. One reproducible failure: GCN produces NaN on an unseen grid

Two of the 448 cross-context rows are non-finite — `gcn`, seed 1000, trained on
IEEE39 tested on IEEE118, and trained on IEEE118 tested on IEEE24. They are
listed in `results_norm/analysis/nonfinite_runs.csv` and are excluded from the
means, and the affected (model, train grid, seed) cell is voided in the ranking
rather than averaged over the remaining test grids.

The cause is not numerical noise and not a NaN in the data. The ENGAGE-style GCN
learns a **scalar edge weight** through a leaky-ReLU head, so the weight may be
negative; `GCNConv(normalize=True)` then forms `deg^(-1/2)` over the summed
edge weights. On the training grid the weighted degree stays positive, but on an
unseen grid it need not: on IEEE24 test sample 2 the minimum edge weight is
-0.504 and three nodes end with a negative weighted degree, so the symmetric
normalization takes the square root of a negative number and the whole forward
pass becomes NaN. Checkpoint weights are finite, and 99 of the 100 samples of
that same grid evaluate normally.

This is the same defect that made ARMA diverge in tuning, where it was fixed by
passing the edge weight through a softplus. GCN was left as ENGAGE ships it, so
the failure stands as a **result**: an architecture whose edge weighting is
unconstrained can emit an undefined answer on a grid it has never seen, without
any warning during training. Applying the softplus to GCN as well would remove
it, at the cost of re-tuning and retraining that architecture — deliberately not
done unilaterally, since it changes a reported architecture.

## 5. What this does and does not fix

Fixes:
* Voltage magnitude becomes a learned quantity rather than an unoptimized
  residual, so the per-quantity table means what it says.
* The comparison across architectures is no longer confounded by each model's
  ability to cope with a 1e4 dynamic range in its inputs.

Does not fix:
* The cross-grid **physical** shift (nominal load 2,850 / 6,254 / 3,733 /
  56,326 MW). A training-grid-fitted scaler is the honest, leak-free choice, but
  the unseen-grid result therefore remains "generalization to an unseen
  *system*", not isolated topology generalization. Isolating topology needs a
  per-grid physical base (e.g. each case's own nominal load, which is input
  data) and is a separate ablation.
* Audit item A5 (Regime B demand-snapshot overlap) and the remaining reporting
  items, which are independent of normalization.
