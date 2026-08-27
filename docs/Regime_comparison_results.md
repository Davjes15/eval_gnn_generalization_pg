# Does the architecture ranking survive generalization? (final)

> **Status: FINAL — all six architectures.** Every number below is computed from
> the consolidated tuned-configuration runs: `gcn`, `gat`, `gin`, `transformer`
> and `arma_gnn` at 5 seeds (`0, 100, 300, 700, 1000`) and **`nnconv` at 3 seeds
> (`0, 100, 300`)** — a reduced replication, disclosed here and in design
> decision D17. No inherited-configuration result (`full_run/results/`) enters
> any table (D19).

All six architectures carry **one frozen configuration into both regimes**
([`docs/Model_configurations.md`](Model_configurations.md)), so a change of
ordering between regimes cannot be attributed to re-tuning.

| | Regime A | Regime B |
|---|---|---|
| Topology | base case only, `k = 0` | sampled N-k, `k ∈ {0,1,2}` |
| Evaluation | within-grid | cross-context (unseen grid) + leave-one-grid-out OOD |
| Data | `data_a/` | `full_run/data/` (frozen) |

Row counts, all finite, all matching `configs/arch_config.json`: Regime A 112,
cross-context 448, OOD 112.

## 1. The pre-registered decision rule

Fixed before the runs (see `rank_analysis.py`), so that τ ≈ +1 would have to be
reported as a null result rather than reframed:

- **τ ≈ +1, stable over seeds** → the ranking is regime-invariant; the ranking
  claim *fails* (absolute degradation may still be large).
- **τ clearly < +1, stable** → the ranking changes under generalization.
- **τ unstable across seeds** → architecture selection under generalization is
  seed-noise dominated; report distributions, not one ranking.

## 2. Rank correlation, Regime A vs Regime B

Kendall τ-b over the 20 (grid, seed) keys. Six architectures ⇒ **15 pairs** per
key for seeds 0/100/300; seeds 700/1000 have no `nnconv` run, so those keys rank
5 architectures (10 pairs). τ-b is computed per key and then averaged, so the
mixed key sizes do not bias it.

| Comparison | metric | τ mean | τ sd | τ min | τ max | ρ mean |
|---|---|---:|---:|---:|---:|---:|
| A vs cross-context | NRMSE | **−0.03** | 0.19 | −0.40 | 0.47 | −0.05 |
| A vs cross-context | MSE | 0.01 | 0.31 | −0.80 | 0.60 | 0.02 |
| A vs cross-context | MAE | −0.11 | 0.27 | −0.73 | 0.20 | −0.12 |
| A vs OOD | NRMSE | **0.32** | 0.26 | −0.20 | 0.80 | 0.38 |
| A vs OOD | MSE | 0.32 | 0.26 | −0.20 | 0.80 | 0.38 |
| A vs OOD | MAE | 0.39 | 0.34 | −0.20 | 1.00 | 0.44 |

Per seed (mean τ over the four grids, NRMSE):

| Seed | A vs cross-context | A vs OOD |
|---|---:|---:|
| 0 | 0.10 | 0.20 |
| 100 | −0.10 | 0.37 |
| 300 | −0.07 | 0.10 |
| 700 | 0.10 | 0.45 |
| 1000 | −0.20 | 0.50 |

Per grid (mean τ over the seeds, NRMSE):

| Grid | A vs cross-context | A vs OOD |
|---|---:|---:|
| IEEE24 | −0.03 | 0.48 |
| IEEE39 | −0.09 | 0.41 |
| IEEE118 | −0.11 | 0.29 |
| UK | 0.09 | 0.11 |

**Reading against the rule.** Cross-context lands on **τ ≈ 0**: fixed-topology
accuracy carries essentially *no* information about the ordering under transfer
to an unseen grid, and it is stable — every seed's mean is within ±0.2 of zero,
every grid's mean within ±0.11, and all three error metrics agree (−0.11 … 0.01).
OOD lands on **τ ≈ 0.32**: a weak positive association, so Regime A predicts part
of the OOD ordering but not the ordering itself. Both are clearly below +1, so
**the ranking claim holds**: an architecture chosen on a fixed-topology table is
not the architecture that survives a topology or grid change. The τ spread
(−0.4 … 0.8 across keys) additionally keeps the third outcome in play — per-key
orderings are noisy — which is why the tables below report distributions and
overlap flags rather than a single order.

## 3. Ranking per arm (NRMSE, mean over grids × seeds)

| Rank | Regime A | cross-context | OOD |
|---|---|---|---|
| 1 | `arma_gnn` 0.00065 | `gat` 0.345 | `arma_gnn` 0.132 |
| 2 | `gin` 0.0038 | `arma_gnn` 0.579 | `transformer` 0.154 |
| 3 | `nnconv` 0.0043 | `transformer` 1.318 | `gat` 0.158 |
| 4 | `transformer` 0.0067 | `nnconv` 1.595 | `gin` 0.166 |
| 5 | `gat` 0.0128 | `gcn` 2.729 | `gcn` 0.231 |
| 6 | `gcn` 0.0163 | `gin` 4.978 | `nnconv` 1.617 |

Four observations carry the argument:

1. **`gin` is 2nd on fixed topology and last under cross-context** — its mean
   NRMSE rises by three orders of magnitude (0.0038 → 4.98). This is exactly the
   failure mode the study is about: the model chosen from a PowerGraph-style
   fixed-grid table is not the model that survives an unseen grid.
2. **`nnconv` is 3rd on fixed topology and last under OOD** (0.0043 → 1.617),
   and the collapse is concentrated in one fold — see §6. The most
   edge-expressive layer in the set is the least transferable.
3. **`gat` is 5th on fixed topology and 1st by cross-context mean.** The two
   architectures a fixed-grid table would rank 5th and 6th (`gat`, `gcn`) are not
   the two it would discard once generalization is measured.
4. **The absolute degradation is enormous for every architecture.** The best OOD
   error (0.132) is ~200× the best within-grid error (0.00065). Even where the
   ordering partly survives, the *magnitude* does not, so a generalization
   measurement is needed regardless of which way τ falls.

`gcn` — ENGAGE's own architecture — is 6th in Regime A and 5th in both Regime B
arms at its own selected configuration, which is worth stating plainly.

## 4. Rank stability

From `results/analysis/ranking_table.csv`: Regime A is not a knife-edge
(`arma_gnn` takes rank 1 in 11 of 20 keys, `gcn` rank 5 or 6 in 18 of 20), but
cross-context genuinely is. `gat` has the best mean yet a **modal rank of 4**
(frequencies `{1:3, 2:5, 3:3, 4:6, 5:2, 6:1}`), and `gcn` has the 5th-best mean
yet a **modal rank of 1** (`{1:5, 2:3, 3:2, 4:3, 5:4, 6:3}`). Heavy-tailed errors
over train→test pairs do that: the mean is set by whichever pair transfers worst.
`nnconv`'s OOD mean rank (3.75) similarly contradicts its last-place mean, since
three of its four folds are normal and one is catastrophic.

The interpretation therefore rests on the **pair** (mean, modal) — reported for
every arm — not on either alone. Every adjacent pair in every arm except the last
is flagged `overlaps_next` (mean ± sd overlap), so the orderings should be read
as coarse bands, not strict sequences.

## 5. The four physical quantities separately (NRMSE)

Aggregating P, Q, V and θ into one NRMSE hides which physics fails, so the
per-quantity table (`results/analysis/per_quantity.csv`) is the primary error
report:

| Arm | model | P | Q | V | θ |
|---|---|---:|---:|---:|---:|
| Regime A | `arma_gnn` | 0.0002 | 0.0009 | 5.98 | 0.013 |
| | `gin` | 0.0003 | 0.0049 | 7.69 | 0.025 |
| | `nnconv` | 0.0005 | 0.0057 | 5.78 | 0.023 |
| | `transformer` | 0.0017 | 0.0086 | 24.9 | 0.051 |
| | `gat` | 0.0055 | 0.0127 | 27.3 | 0.058 |
| | `gcn` | 0.0077 | 0.0177 | 7.84 | 0.079 |
| cross-context | `gat` | 0.071 | 0.341 | 143.6 | 4.89 |
| | `arma_gnn` | 0.065 | 0.525 | 232.5 | 16.1 |
| | `transformer` | 0.072 | 1.518 | 312.3 | 11.1 |
| | `nnconv` | 0.235 | 1.327 | 299.5 | 19.3 |
| | `gcn` | 0.317 | 2.606 | 551.2 | 34.5 |
| | `gin` | 0.176 | 6.273 | 3584.1 | 210.7 |
| OOD | `arma_gnn` | 0.085 | 0.101 | 25.8 | 1.07 |
| | `transformer` | 0.089 | 0.144 | 33.9 | 1.67 |
| | `gat` | 0.102 | 0.121 | 37.5 | 1.34 |
| | `gin` | 0.088 | 0.169 | 151.8 | 3.56 |
| | `gcn` | 0.143 | 0.176 | 59.3 | 1.58 |
| | `nnconv` | **1.369** | 0.242 | 76.0 | 1.94 |

Two things this exposes that the aggregate does not:

- **Voltage magnitude is the hardest target by orders of magnitude** in every arm
  and for every architecture (its NRMSE normalizer is small because V is tightly
  banded around 1 p.u., so relative error is unforgiving). `gin`'s cross-context
  collapse is almost entirely a V and θ collapse (3584 and 211).
- **`nnconv`'s OOD failure is an active-power failure** (P = 1.37, ~13× the next
  worst), not a voltage one — a different mechanism from `gin`'s.

## 6. `nnconv`'s OOD collapse is one fold, and it is real

Per held-out grid (NRMSE, mean over seeds):

| model | IEEE118 | IEEE24 | IEEE39 | UK |
|---|---:|---:|---:|---:|
| `arma_gnn` | 0.102 | 0.154 | 0.125 | 0.148 |
| `gat` | 0.105 | 0.174 | 0.171 | 0.183 |
| `gcn` | 0.107 | 0.452 | 0.194 | 0.170 |
| `gin` | 0.106 | 0.157 | 0.143 | 0.259 |
| `transformer` | 0.103 | 0.192 | 0.148 | 0.173 |
| `nnconv` | 0.103 | 0.174 | **6.038** | 0.153 |

Three of `nnconv`'s four folds are indistinguishable from the field; held-out
IEEE39 is ~40× worse than any other architecture on the same fold, consistently
across all three seeds (6.599 / 5.278 / 6.237). It is **not** a numerical
failure: every value is finite, training was clean, and `nnconv` reaches 0.013
NRMSE *within* IEEE39 in Regime A. The natural reading is that edge-conditioned
filters — a full `hidden × hidden` transform generated per edge from the edge
attributes — fit the admittance statistics of the training mixture and do not
transfer when the held-out grid's edge distribution differs. It is reported as a
result, not excluded.

## 7. The DC power-flow baseline beats every GNN out of distribution

`results/analysis/dc_comparison.csv`, aggregate NRMSE as a ratio (GNN ÷ DC-PF;
< 1 means the GNN wins):

| model | Regime A | cross-context | OOD |
|---|---:|---:|---:|
| `arma_gnn` | **0.04** | 26.1 | 7.9 |
| `gin` | **0.23** | 223.6 | 9.9 |
| `nnconv` | **0.26** | 71.8 | 96.8 |
| `transformer` | **0.40** | 59.3 | 9.2 |
| `gat` | **0.76** | 15.8 | 9.5 |
| `gcn` | **0.98** | 122.9 | 13.8 |

Every architecture beats DC power flow on its own fixed topology — `arma_gnn` by
25× — and **every architecture loses to it by one to two orders of magnitude on
an unseen grid**. DC-PF is a linearization with no learned parameters, so it has
nothing to transfer and its error is topology-agnostic; that is precisely why it
is the right floor to report. Caveats: DC-PF carries no reactive power (Q ≡ 0, so
per-quantity Q ratios are written as NaN) and its g-score is an artifact
(`mmd_range = 0`), a reference bar rather than a competitor.

## 8. g-score (ENGAGE, used cautiously at N = 4 grids)

Cross-context aggregate g-score and OOD g-score, mean over seeds
(`gscore_cc_aggregate.csv`, `gscore_ood.csv`; no percentile trim, D13):

| model | CC mean NRMSE | CC std | **CC g-score** | OOD mean NRMSE | **OOD g-score** |
|---|---:|---:|---:|---:|---:|
| `gat` | 0.345 | 0.371 | **0.644** | 0.158 | 0.197 |
| `arma_gnn` | 0.579 | 0.833 | **1.251** | 0.132 | **0.150** |
| `transformer` | 1.318 | 2.176 | 3.072 | 0.154 | 0.188 |
| `nnconv` | 1.595 | 3.774 | 4.638 | 1.617 | 3.803 |
| `gcn` | 2.729 | 4.234 | 6.142 | 0.231 | 0.349 |
| `gin` | 4.978 | 12.277 | 14.876 | 0.166 | 0.217 |
| *(dc_pf ref)* | *0.017* | *0.005* | *0.017* | — | — |

The two g-score flavours **disagree at the top** (`gat` best on cross-context,
`arma_gnn` best on OOD), which is the same instability τ reports. With only four
grids the g-score's std and `mmd_range` terms rest on 3–4 points, so it is
reported as a secondary statistic; the transfer matrix, the per-quantity table
and the DC floor are the headline.

## 9. Reproducing every table here

```bash
# 1. consolidate seed shards (one architecture at a time)
python3 gather_results.py --shards 'results_a/within_arma_v2_s*' \
    --file within_grid.csv --out results_a/within_arma_v2 \
    --models arma_gnn --seed_shards
python3 gather_results.py --shards 'results_a/within_nnconv_s*' \
    --file within_grid.csv --out results_a/within_nnconv \
    --models nnconv --seed_shards
python3 gather_results.py --shards 'results_tuned/nnconv_cc_s*' \
    --file cross_context.csv --out results_tuned/nnconv_cc \
    --models nnconv --seed_shards
python3 gather_results.py --shards 'results_tuned/nnconv_ood_s*' \
    --file ood.csv --out results_tuned/nnconv_ood --models nnconv --seed_shards

# 2. consolidate architectures per arm (--seed_shards: nnconv carries 3 seeds)
python3 gather_results.py \
    --shards results_a/within_{gcn,gat,gin,transformer,arma_v2,nnconv} \
    --file within_grid.csv --out results/regime_a \
    --models gcn gat gin transformer arma_gnn nnconv --seed_shards
python3 gather_results.py \
    --shards results_tuned/{gcn,gat,gin,transformer,arma_v2,nnconv_cc} \
    --file cross_context.csv --out results/regime_b \
    --models gcn gat gin transformer arma_gnn nnconv --seed_shards
python3 gather_results.py \
    --shards results_tuned/{gcn,gat,gin,transformer,arma_v2,nnconv_ood} \
    --file ood.csv --out results/regime_b \
    --models gcn gat gin transformer arma_gnn nnconv --seed_shards

# 3. ranking (§2-§4) and the downstream tables (§5-§8)
python3 rank_analysis.py --regime_a results/regime_a/within_grid.csv \
    --cross results/regime_b/cross_context.csv \
    --ood results/regime_b/ood.csv --out results/analysis
python3 recompute_tables.py --within results/regime_a/within_grid.csv \
    --cross results/regime_b/cross_context.csv \
    --ood results/regime_b/ood.csv \
    --topology results_tuned/gcn results_tuned/arma_v2_s0 \
                results_tuned/nnconv_cc_s0 results_tuned/nnconv_ood_s0 \
    --out results/analysis
```

## 10. Limitations of this comparison

- **Four grids.** Every cross-grid statistic — τ, the g-score's `mmd_range`, the
  OOD mean — rests on 4 topologies, 12 train→test pairs and 4 leave-one-out
  folds. The direction of the findings is stable across seeds and metrics; the
  precise values are not to be quoted as population estimates.
- **`nnconv` at 3 seeds.** Its within-arm variance is estimated from 3 draws, and
  keys at seeds 700/1000 rank 5 architectures instead of 6 (D17).
- **One configuration per architecture.** Frozen by an equal-budget sweep
  (9 depth×width candidates + the lower learning rate) so that regime changes are
  not confounded with re-tuning — but a ranking is a ranking *at those
  configurations*. The second-best-configuration robustness check is not run.
- **ARMA's protocol amendment came after seeing results** (D15/D16): the
  divergence-disqualification rule and the softplus edge weight were introduced
  because ARMA diverged. Both are disclosed, applied uniformly, and are no-ops
  for the other five architectures; the pre-fix ARMA rows are archived, not
  deleted.
- **No checkpoints for `arma_gnn` and `nnconv`** — they reproduce from the
  recorded seeds (D18).
- **Regime A and Regime B use different datasets** (`data_a/` vs the frozen
  `full_run/data/`), matched on grids, split sizes and preprocessing but not
  identical draws, so a small part of the A→B change is dataset, not regime.
- **No practitioner arm**, by explicit scope decision: nothing here estimates
  what a practitioner would achieve by re-tuning on the target grid.
