# Does the architecture ranking survive generalization? (interim)

> **Status: INTERIM — five of six architectures.** `nnconv`'s final runs are
> still executing (3 arms × 3 seeds; at hidden 128 one IEEE118 training is ~3 h).
> Every number below is computed from `gcn`, `gat`, `gin`, `transformer` and
> `arma_gnn` at 5 seeds each and **will be recomputed with all six** before any
> of it is quoted as a result. It is recorded now because the conclusion is
> already visible and should be reviewable.

All five architectures carry **one frozen configuration into both regimes**
([`docs/Model_configurations.md`](Model_configurations.md)), so a change of
ordering between regimes cannot be attributed to re-tuning.

| | Regime A | Regime B |
|---|---|---|
| Topology | base case only, `k = 0` | sampled N-k, `k ∈ {0,1,2}` |
| Evaluation | within-grid | cross-context (unseen grid) + leave-one-grid-out OOD |
| Data | `data_a/` | `full_run/data/` (frozen) |

## 1. The pre-registered decision rule

Fixed before the runs (see `rank_analysis.py`), so that τ ≈ +1 would have to be
reported as a null result rather than reframed:

- **τ ≈ +1, stable over seeds** → the ranking is regime-invariant; the ranking
  claim *fails* (absolute degradation may still be large).
- **τ clearly < +1, stable** → the ranking changes under generalization.
- **τ unstable across seeds** → architecture selection under generalization is
  seed-noise dominated; report distributions, not one ranking.

## 2. Rank correlation, Regime A vs Regime B (NRMSE)

Kendall τ-b over the 20 (grid, seed) keys, 5 architectures ⇒ 10 pairs per key:

| Comparison | τ mean | τ sd | τ min | τ max | ρ mean |
|---|---:|---:|---:|---:|---:|
| A vs cross-context | **−0.03** | 0.24 | −0.4 | 0.4 | −0.04 |
| A vs OOD | **0.32** | 0.29 | −0.2 | 0.8 | 0.37 |

Per seed (mean τ over the four grids):

| Seed | A vs cross-context | A vs OOD |
|---|---:|---:|
| 0 | 0.00 | 0.25 |
| 100 | −0.05 | 0.25 |
| 300 | 0.00 | 0.15 |
| 700 | 0.10 | 0.45 |
| 1000 | −0.20 | 0.50 |

Per grid (mean τ over the five seeds):

| Grid | A vs cross-context | A vs OOD |
|---|---:|---:|
| IEEE24 | −0.04 | 0.44 |
| IEEE39 | −0.12 | 0.48 |
| IEEE118 | −0.16 | 0.28 |
| UK | 0.20 | 0.08 |

**Reading against the rule.** Cross-context lands on τ ≈ 0: fixed-topology
accuracy carries *no* information about the ordering under transfer to an unseen
grid — and this is stable, every seed's mean sits within ±0.2 of zero, so it is
not a single unlucky draw. OOD lands on τ ≈ 0.32: a weak positive association,
i.e. Regime A predicts part of the OOD ordering but not the ordering itself.
Both are clearly below +1, so on the interim data the ranking claim **holds**;
the τ spread (−0.4 … 0.8 across keys) additionally puts the third outcome in
play — per-key rankings are noisy, which is why the arm tables below report
distributions and overlap flags rather than a single order.

## 3. Ranking per arm (NRMSE, mean over 4 grids × 5 seeds)

| Rank | Regime A | cross-context | OOD |
|---|---|---|---|
| 1 | `arma_gnn` 0.00065 | `gat` 0.345 | `arma_gnn` 0.132 |
| 2 | `gin` 0.0038 | `arma_gnn` 0.579 | `transformer` 0.154 |
| 3 | `transformer` 0.0067 | `transformer` 1.318 | `gat` 0.158 |
| 4 | `gat` 0.0128 | `gcn` 2.729 | `gin` 0.166 |
| 5 | `gcn` 0.0163 | `gin` 4.978 | `gcn` 0.231 |

Two observations that carry the argument:

1. **`gin` is 2nd on fixed topology and last under cross-context** — its mean
   NRMSE rises by three orders of magnitude (0.0038 → 4.98). This is precisely
   the failure mode the study is about: a model chosen on a PowerGraph-style
   fixed-grid table is not the model that survives an unseen grid.
2. **The absolute degradation is enormous for every architecture** — the best
   OOD error (0.132) is ~200× the best within-grid error (0.00065). Even where
   the ordering is preserved, the *magnitude* is not, so a generalization
   measurement is needed regardless of which way τ falls.

`gcn` is last in all three arms, which is worth stating plainly: the ENGAGE
baseline architecture is the weakest of the five here at its own selected
configuration.

## 4. Rank stability

The modal ranks and rank frequencies (`results/ranking_table.csv`) show the
ordering is not a knife-edge in Regime A — `arma_gnn` takes rank 1 in 12 of 20
keys and `gcn` rank 5 in 13 of 20 — but is genuinely unstable in cross-context,
where `gat` has the best mean yet a modal rank of 4 (its rank frequencies are
`{1: 3, 2: 5, 3: 3, 4: 8, 5: 1}`). A heavy-tailed error distribution over train
grids is doing that: the mean is dominated by which train→test pair happens to
transfer badly. Both the mean-based and the modal ordering are therefore
reported, and the interpretation rests on the pair (mean, modal), not on either
alone.

## 5. Reproducing this table

```bash
# consolidate: seed shards first (one architecture), then architectures
python3 gather_results.py --shards 'results_a/within_arma_v2_s*' \
    --file within_grid.csv --out results_a/within_arma_v2 \
    --models arma_gnn --seed_shards
python3 gather_results.py --shards results_a/within_{gcn,gat,gin,transformer} \
    results_a/within_arma_v2 --file within_grid.csv --out results/regime_a \
    --models gcn gat gin transformer arma_gnn

python3 rank_analysis.py --regime_a results/regime_a/within_grid.csv \
    --cross results/regime_b/cross_context.csv \
    --ood results/regime_b/ood.csv --out results/analysis
```

## 6. What is still missing

- `nnconv`'s three arms at seeds 0, 100, 300 (reduced replication, disclosed).
- Recomputation of every table here with six architectures (15 pairs per key
  instead of 10, so the τ values will move even if no ordering changes).
- The g-score, per-quantity (P/Q/V/θ) and DC-baseline tables under the tuned
  configurations; the inherited-configuration versions in `full_run/results/`
  are superseded and must not be mixed with these.
