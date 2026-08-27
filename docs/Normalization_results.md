# A2 remediation: normalized-representation campaign (N2)

Companion to [`Normalization_assessment.md`](Normalization_assessment.md), which
established *why* the raw-unit protocol had to change. This file records what was
implemented, the evidence that triggered the full retrain, and the results.

Status: **campaign running.** Section 4 is filled in as shards land; anything not
yet measured is marked as such rather than estimated.

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
* Data unchanged: `data_a` for Regime A, `data_full` for Regime B. No
  regeneration, so the comparison against the raw-unit campaign differs only in
  the training representation.
* Sharded one job per (arm, architecture) through a parallel pool
  (`launch_normalized.sh`), one Torch thread per process.
* `--save_models` for every shard, into `ckpt_norm/<arm>_<model>/`, so every
  reported row is replayable from a checkpoint instead of a retrain — the gap
  that made the raw-unit ARMA and NNConv rows seed-reproducible only.
* Results land in `results_norm/<arm>_<model>/`; the raw-unit tables in
  `results/` are kept as the documented ablation and are not overwritten.

## 4. Results

_Pending: filled in as the shards complete._

Early within-grid rows (IEEE24) already show the aggregate NRMSE dropping by
roughly an order of magnitude against the raw-unit campaign (gcn 0.0042 →
0.0008, gat → 0.0002, gin → 0.0003, transformer → 0.0002 at seed 0). The
question the campaign answers is not whether accuracy improves but whether the
**transfer ranking** — the study's actual claim — survives the change of
representation.

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
