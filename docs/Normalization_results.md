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

Regime A and Regime B differ in **two** ways at once: Regime B uses a harder
protocol (blocked temporal split, N-k contingency topologies) *and* a different
grid. The same-grid **diagonal** of the cross-context table separates the grid
change from the rest — it is Regime B evaluated on the grid the model was trained
on, so A → diagonal is the same-grid step and diagonal → unseen is the grid step.
Both are free from the runs already done, and reporting only A → unseen charges
the whole gap to generalization.

The same-grid step is **not** a clean protocol effect, and audit C6 was right to
say so: Regime B changed the temporal split *and* the topology at the same time,
so `same_grid_factor` bounds the two jointly and attributes it to neither.
Separating them would need a blocked-split-only dataset and a
contingencies-only dataset with a retrained campaign on each — declined with the
other training items (limitation L8).

| rank | Regime A (within, fixed topology) | Regime B diagonal (same grid, N-k) | cross-context (unseen grid) | OOD (leave-one-grid-out) |
|---|---|---|---|---|
| 1 | arma_gnn 0.00044 | arma_gnn 0.0047 | arma_gnn 0.821 | transformer 0.269 |
| 2 | gin 0.0037 | transformer 0.0079 | transformer 0.939 | gat 0.346 |
| 3 | transformer 0.0039 | gat 0.0082 | gin 1.004 | gin 0.387 |
| 4 | gat 0.0040 | gin 0.0086 | gcn 1.058 | arma_gnn 0.798 |
| 5 | nnconv 0.0040 | nnconv 0.0089 | gat 1.105 | gcn 1.043 |
| 6 | gcn 0.0100 | gcn 0.0155 | nnconv 1.985 | nnconv 3.410 |

Decomposed as multiplicative factors (`protocol_decomposition.csv`, aggregate
NRMSE):

| model | same-grid step (A → diagonal) | grid step (diagonal → cross-context) | total |
|---|---:|---:|---:|
| arma_gnn | 10.5× | 177× | 1861× |
| gat | 2.1× | 134× | 279× |
| gcn | 1.5× | 68× | 105× |
| gin | 2.3× | 117× | 273× |
| nnconv | 2.3× | 222× | 501× |
| transformer | 2.0× | 119× | 240× |

Blocked splits plus contingencies together cost a factor of ~2 (ARMA ~10). **The
unseen grid costs two orders of magnitude on top of that**, so the headline gap is a
generalization result, not a split artefact — but it is 68–222×, not the ~1000×
that the raw A → unseen ratio suggests.

**Where the ranking survives and where it breaks.** Kendall tau per (grid, seed)
cell, averaged, with an exact permutation null over all 720 architecture
relabellings (`rank_correlation_summary.csv`, `rank_permutation_test.csv`):

| comparison | mean tau ± sd | permutation p |
|---|---|---:|
| A → Regime B diagonal | **0.663 ± 0.206** | **0.004** |
| A → cross-context | 0.067 ± 0.369 | 0.72 |
| A → OOD | 0.020 ± 0.380 | 1.00 |

The fixed-topology leaderboard **does** survive a harder protocol on the same
grid, and stops predicting anything once the grid changes. That contrast is the
result; the earlier version of this section reported only the second row and
therefore could not distinguish "ranking is meaningless" from "ranking is
grid-specific".

**Pooled ordering vs per-cell ordering.** Ranking the architectures by their
pooled mean error gives tau = 0.60 (A vs cross-context) and 0.20 (A vs OOD)
(`rank_correlation_pooled.csv`) — visibly not zero. There is no contradiction:
the pooled leaderboard is one ordering of six averages, dominated by the extremes
(ARMA good, NNConv bad, and those hold everywhere), while the per-cell tau asks
whether that ordering reproduces in an individual grid × seed environment, and it
does not — cells range from -0.73 to +0.87.

The defensible claim is therefore **rank instability**, not zero correlation. A
permutation p of 1.00 says the data are consistent with random relabelling; it is
not evidence that the true correlation is exactly zero, and with 12 cells and six
architectures this design could not detect a modest true correlation anyway. What
a practitioner can take from it: picking an architecture from a fixed-topology
leaderboard gives no reliable guarantee on an unseen grid, though the extremes of
that leaderboard do tend to stay extreme.

The 12 cells are 4 grids × the 3 seeds NNConv was run at. The earlier raw-unit
reading (A ↔ OOD tau = 0.222, p = 0.21) does not survive the final protocol.

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

Cross-context aggregate (`mean + 0.806 sd`) and OOD (`mean + 0.857 sd`), averaged
over the seeds whose cell is complete:

| model | cc mean | cc sd | cc g | finite rate | OOD mean | OOD g |
|---|---|---|---|---|---|---|
| arma_gnn | 0.821 | 1.490 | 2.020 | 1.00 | 0.798 | 1.311 |
| transformer | 0.939 | 1.937 | 2.497 | 1.00 | 0.269 | 0.412 |
| gin | 1.004 | 2.103 | 2.696 | 1.00 | 0.387 | 0.652 |
| gcn | 1.145 | 2.157 | 2.881 | **0.97** | 1.043 | 1.976 |
| gat | 1.105 | 2.265 | 2.928 | 1.00 | 0.346 | 0.528 |
| nnconv | 1.985 | 3.133 | 4.507 | 1.00 | 3.410 | 6.713 |
| dc_pf | 0.067 | 0.023 | **0.067** | 1.00 | — | — |

**Divergence does not earn a g-score.** GCN at seed 1000 fails on 2 of its 12
unseen pairs (§4bis). Scoring it on the 10 survivors gave it `g = 0.888`, the
best value in the table — the failed pairs are the hard ones, so dropping them is
a reward for diverging. The score for an incomplete cell is therefore voided, and
`finite_rate` in `gscore_cc_aggregate.csv` reports what fraction of the expected
points survived, which is the same void-the-cell rule the rank analysis uses.
GCN's row above is the mean over its four complete seeds.

The per-training-grid table (`gscore.csv`) is also un-trimmed now. ENGAGE's 2/98
percentile trim on three unseen grids keeps a single point, which silently turned
`mean_nrmse` into a median and `std_nrmse` into a column of zeros; with `bounds=0`
the columns mean what they say and agree with the pooled table.

The `dc_pf` row aggregates one point per grid where the model rows aggregate one
per ordered training→test pair (4 vs 12 points, recorded in the `basis` and
`n_expected` columns), so it is a reference bar and not a seventh competitor.

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

### 4.6 Is the predicted state a valid operating point? (audit B1)

Everything above compares a prediction with a label. It cannot say whether the
predicted state is *physically possible*, and that is the question a screening
tool has to answer. Because the node task emits all four quantities at every bus,
the predicted complex voltage and the predicted complex injection are both fully
determined, so the AC power-balance residual on the post-contingency admittance
matrix can be computed directly:

```
dS_i = S_i^spec - ( V_i conj((Y V)_i) - S_i^shunt )
```

`ac_feasibility.py` rebuilds each test network from `dataset_src.csv` (same case,
same demand snapshot, same outage), takes `Y`, the branch data and the branch
ratings from pandapower's own internal representation, and evaluates the residual
and the branch loading of the predicted state. The thermal screen covers **every
in-service branch** — lines against `max_i_ka`, two-winding transformers against
their `sn_mva` rating converted to a current limit at each end, which is a
different rating per end whenever the two voltage levels differ (audit C4; the
first implementation screened lines only and dropped 5/38 of IEEE24's branches,
11/46 IEEE39, 9/184 IEEE118, 4/90 UK). `tests/test_ac_feasibility.py` validates
the result against pandapower's own `res_line.loading_percent` and
`res_trafo.loading_percent` to 1e-12 %, including under a transformer outage.
Nothing is retrained: this is a second pass over the same 336 checkpoints
(`eval_checkpoints.py --feasibility`, 672 rows in
`results_norm/physics/physics_metrics.csv`, summarised in
`docs/tables/ac_feasibility_norm.csv`).

The scale is set by the labels themselves: run through the identical code path,
the **true** states have a residual of at most 2.8e-2 MW, so anything above that
is the model's, not the checker's.

Means over runs; `dP` is the summed absolute bus mismatch as a percentage of the
snapshot's served load, which is the only cross-grid comparable form. Two policy
points from the fourth audit (D1, D2): a checkpoint whose transfer error is
non-finite is **voided here as it is in the ranking** — `feasibility_metrics`
averages over buses with `nanmean` and would otherwise report a finite residual
for a run that produced no answer, so the two diverged GCN cross-context cells
(seed 1000, `IEEE118→IEEE24` and `IEEE39→IEEE118`) no longer enter any mean, and
the count of dropped rows is the `n_voided` column of the CSV. And the
distribution is heavy-tailed, so the CSV also carries a median and a max of the
two residual columns; read the median as typical, not the mean:

| setting | model | dP [% load] | dQ [% load] | max loading pred / true | overloads true / predicted | missed | false alarms |
|---|---|---:|---:|---|---|---:|---:|
| **reconstruction floor** (Regime A labels) | floor | 0.000 | 0.000 | 364 % / 364 % | 0.136 / 0.136 | 0.000 | 0.000 |
| **DC power flow** (Regime A) | dc_pf | 35 | 162 | 292 % / 364 % | 0.136 / 0.126 | 0.112 | 0.006 |
| within (Regime A) | arma_gnn | 42 | 6 | 362 % / 364 % | 0.136 / 0.139 | 0.019 | 0.005 |
| within | gin | 63 | 13 | 391 % / 364 % | 0.136 / 0.141 | 0.032 | 0.012 |
| within | transformer | 166 | 44 | 939 % / 364 % | 0.136 / 0.147 | 0.041 | 0.021 |
| within | gcn | 330 | 83 | 913 % / 364 % | 0.136 / 0.180 | 0.091 | 0.068 |
| **DC power flow** (Regime B) | dc_pf | 34 | 159 | 333 % / 409 % | 0.126 / 0.113 | 0.126 | 0.005 |
| Regime B, same grid | arma_gnn | 120 | 37 | 560 % / 409 % | 0.126 / 0.139 | 0.107 | 0.029 |
| Regime B, same grid | gcn | 354 | 142 | 1,846 % / 392 % | 0.121 / 0.176 | 0.220 | 0.086 |
| unseen grid | arma_gnn | 3,984 | 7,086 | 6,680 % / 409 % | 0.126 / 0.702 | 0.232 | 0.695 |
| unseen grid | gin | 24,412 | 132,972 | 20,156 % / 409 % | 0.126 / 0.791 | 0.164 | 0.787 |
| OOD | transformer | 3,951 | 8,673 | 7,231 % / 409 % | 0.126 / 0.768 | 0.216 | 0.764 |

How much the mean overstates the typical case: on an unseen grid the mean `dP` is
8,174 % of load across all models but the median is 4,323 %, and the gap is
almost entirely GIN (mean 24,412 %, median 4,214 %, worst single run 393,485 %) —
every other architecture's mean is within 14 % of its own median. The
single-worst run in the whole replay is a GIN OOD fold at 303,891 %.

The true column is not a formality: **the source OPF snapshots are not themselves
thermally secure** — 13–14 % of branch-samples exceed their rating in the labels,
with true maxima of 677 % (IEEE118), 442 % (IEEE39), 399 % (UK) and 151 %
(IEEE24) — so a large predicted loading is only an error when it exceeds the true
one, which is why the confusion columns rather than the maxima carry the
operational reading.

A branch counts as overloaded above 100 % of its rating (`max_i_ka × df ×
parallel` for a line, the per-end current equivalent of `sn_mva × df × parallel`
for a transformer); the table is regenerated by `python summarize_feasibility.py`
from the replay CSV plus the reference rows of `dc_feasibility.py`.

The two reference rows are what make the model rows readable (audit C5). The
**floor** row is the labels put through the identical checker: 3e-4 % of load, so
none of the model numbers is an artefact of the reconstruction. The **DC** row is
the same linear baseline the NRMSE tables compare against, scored physically for
the first time — and its active residual (35 % of load) is *better than every
model's except ARMA's* even in distribution, and 100× better than any model on an
unseen grid. Its reactive residual (160 %) is not a measure of the linearisation:
DC fixes |V| = 1 and Q ≡ 0 by construction, so that column is essentially the
snapshot's reactive demand and should not be read as an error. On thermal
screening DC is the mirror image of the models: it under-flags (0.113 predicted
against 0.126 true, 12.6 % of genuine overloads missed, 0.5 % false alarms) where
the surrogates over-flag.

Three readings, in order of how much they should change what the paper claims:

1. **In distribution, a small NRMSE does not mean a feasible state.** ARMA's
   within-grid NRMSE is 4.4e-4 — three orders of magnitude better than DC — and
   its predicted states still violate active power balance by ~40 % of served
   load in aggregate, against a reconstruction floor of 2.8e-2 MW. Its thermal
   screening *is* good (362 % predicted worst loading against 364 % true, 1.9 %
   missed and 0.5 % false), so in distribution the models are usable as
   screening tools while still not being solutions of the power flow: they are
   accurate as regressors of the label, and "the surrogate reproduces AC power
   flow" is not a statement these numbers support. No conclusion of §4 changes.
   The DC reference sharpens this: DC power flow is 6–50× *worse* than the
   surrogates in NRMSE and simultaneously *better* than five of the six in AC
   active-power residual, so the two rankings disagree — label accuracy and
   physical admissibility are not the same axis, and only the second one is what
   a screening tool needs.
2. **The transfer failure is much larger in feasibility terms than in NRMSE
   terms.** Unseen-grid NRMSE is ~200× the in-distribution value; the AC residual
   is ~10²–10³× and the reactive residual worse still, because the residual is
   quadratic in the state error and because a wrong voltage angle profile
   misplaces every branch flow at once.
3. **The screening error flips direction.** In distribution the models
   over-flag mildly (0.5–7 % false alarms) and miss few real overloads. On an
   unseen grid they flag ~70–80 % of all branches as overloaded while missing ~20 %
   of the genuine ones — simultaneously unusable and unsafe.

Caveats on these numbers: the residual is evaluated on the topology the sample
was generated on, so it measures the state, not the model's view of the topology;
the screen covers lines and two-winding transformers, which is every rated branch
here (no case has a three-winding transformer, and the 2 explicit impedance
elements of IEEE118 and 9 of UK enter `Ybus` and therefore the residual, but
pandapower gives them no current rating and no `loading_percent`, so they cannot
be screened); and the DC row's reactive residual is a convention, not an error,
for the reason given above.

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
