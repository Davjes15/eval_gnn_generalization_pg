# Frozen model configurations and why each one was selected

This document is the authoritative record of the six architecture configurations
used in every experiment of the regime comparison, and of the evidence that
selected them. The machine-readable form is
[`configs/arch_config.json`](../configs/arch_config.json); every result row
carries `num_layers`, `hidden` and `learning_rate` so a table can always be
traced back to this file.

## 1. The selection protocol

One configuration per architecture, frozen once and reused in **both** regimes.
The regimes differ only in the evaluation data, so any ranking change between
them is attributable to the architecture rather than to a re-tuning.

| | |
|---|---|
| Search space | `num_layers ∈ {2, 3, 8}` × `hidden ∈ {32, 64, 128}` × `learning_rate ∈ {1e-3, 3e-4}` |
| Budget | 10 candidates per architecture — the 9-point depth×width grid at `lr=1e-3`, then the leader re-scored at `lr=3e-4` |
| Training data | Regime A (`data_a`, fixed topology), 800 train / 100 val / 100 test per grid |
| Score | mean over the four grids of the best **validation** weighted MSE; test splits are never read |
| Stage-1 seed | 0 |
| Tie / confirmation seed | 100 |
| Declared tie | if the leader and runner-up are within 5%, both are re-scored at seed 100 |
| Epochs, batch size | 200, 32 |
| Fixed by inheritance | ENGAGE's skeleton: 7 node features, MLP node encoder, edge MLP → scalar or vector edge feature, known-value re-injection at inference; ARMA's 5 filter stacks come from Hansen et al. (2023) |

**Stability rule (amendment, see §4).** A candidate whose validation loss is
non-finite on *any* grid is **disqualified**, not merely ranked last; and the
leading candidate must reproduce at the confirmation seed before it can be
frozen. If no candidate survives at one learning rate the whole grid is
re-scored at the other, and if nothing survives at either the architecture is
reported as unusable rather than frozen at an unstable setting.

Artifacts per architecture: `tuning.csv` (every trial), `tuning_summary.csv`
(per-candidate aggregate, `stable` and `selected` flags), and
`tuning_per_grid_argmin.csv` (which configuration each grid would have picked on
its own — the free diagnostic for "you should have tuned per grid").

## 2. The frozen configurations

| Architecture | Layers | Hidden | LR | Params | Stage-1 mean val (seed 0) | Confirmed mean (seeds 0+100) | Stable |
|---|---:|---:|---:|---:|---:|---:|---|
| `gcn` | 2 | 128 | 1e-3 | 85,093 | 0.3007 | 0.8260 | yes |
| `gat` | 2 | 128 | 1e-3 | 122,980 | 0.1251 | 11.1164 | yes |
| `gin` | 3 | 128 | 1e-3 | 202,836 | 0.0900 | 0.0873 | yes |
| `transformer` | 2 | 128 | 1e-3 | 188,516 | 0.1423 | 0.1380 | yes |
| `nnconv` | 2 | 128 | 1e-3 | 1,166,660 | 0.1089 | 0.0998 | yes |
| `arma_gnn` | 8 | 128 | 1e-3 | 1,367,909 | 0.0183 | 0.0183 | yes |

Two regularities are worth stating because they are consistent across all six:

- **`hidden = 128` wins everywhere.** Width is the only dimension on which every
  architecture agrees, and no architecture selected 32.
- **`lr = 3e-4` never wins.** At a 200-epoch budget with early stopping the
  lower learning rate is under-trained rather than better-behaved: it is worse
  for all six, by 2× (ARMA, GIN) up to 16× (GCN). This is a statement about the
  budget, not about the optimum of an unbounded run.

Depth is the dimension that separates the architectures, and it separates them
in the way the layer algebra predicts: the message-passing layers that apply
their weights once per hop peak at 2–3 layers and degrade sharply at 8, whereas
ARMA — whose recursive filter is *designed* to reach far without stacking depth
— is the only one that improves monotonically to 8.

## 3. Per-architecture justification

Each subsection gives the leader, the runner-up it had to beat, and the reason
the choice is not an artifact of a single seed. "Gap" is the runner-up's
degradation relative to the leader.

### `gcn` — 2 layers, hidden 128, lr 1e-3 (85,093 params)

| Rank | Config | Mean val (s0) |
|---|---|---:|
| 1 | **2 × 128** | **0.3007** |
| 2 | 3 × 128 | 1.0158 |
| 3 | 2 × 32 | 1.2777 |
| … | 8 × 128 | 13.7093 |

Gap to runner-up **238%** — far outside the 5% tie band, so no second-seed
tie-break was required; the winner was nonetheless confirmed at seed 100
(finite, mean over both seeds 0.8260) as the amended rule demands. `lr=3e-4`
was 16× worse (4.9383) and is rejected.

The result contradicts ENGAGE's inherited 8 layers, and the sweep says why:
GCN at 8 × 128 is 46× worse than at 2 × 128. Repeated symmetric-normalized
averaging over-smooths, and on these small transmission graphs (24–118 buses)
8 hops covers a large fraction of the diameter, so bus embeddings converge
toward each other exactly where the target (a bus-local balance) needs them to
stay distinct. Note this is a *fixed-topology, single-snapshot-family* regime;
ENGAGE's 8 layers were chosen for a different data regime and are not wrong
there.

Per-grid diagnostic: IEEE24 and IEEE39 would each have chosen 8 × 128 on their
own, IEEE118 and UK would have chosen 3 × 128. GCN is therefore the
architecture whose pooled choice agrees with *no* individual grid — recorded
here because it bounds how much of any Regime-A→B ranking change could be
attributed to the frozen-configuration decision.

### `gat` — 2 layers, hidden 128, lr 1e-3 (122,980 params)

| Rank | Config | Mean val (s0) |
|---|---|---:|
| 1 | **2 × 128** | **0.1251** |
| 2 | 3 × 64 | 0.1954 |
| 3 | 3 × 128 | 0.1975 |
| … | 8 × 128 | 22.3043 |

Gap **56%**, no tie. `lr=3e-4` is 142× worse (17.7250) — the steepest
learning-rate penalty of the six, consistent with attention logits needing to
move away from uniform early.

GAT is the one architecture whose confirmation run is *poor* rather than merely
different: at seed 100 the mean is 22.1, driven by UK (87.9). It is finite, so
the rule keeps it, and this is deliberate — the rule disqualifies divergence,
not high variance. **GAT's seed sensitivity is a result to report, not a defect
to tune away**, and it is one of the reasons the final tables report standard
deviations over 5 seeds rather than a single number. A rule that also excluded
high-variance candidates would have been a post-hoc quality filter on the
outcome we are trying to measure.

### `gin` — 3 layers, hidden 128, lr 1e-3 (202,836 params)

| Rank | Config | Mean val (s0) |
|---|---|---:|
| 1 | **3 × 128** | **0.0900** |
| 2 | 3 × 64 | 0.1065 |
| 3 | 2 × 128 | 0.1100 |
| … | 8 × 128 | 3.3399 |

Gap **18%**, outside the tie band. GIN is the most seed-stable of the six: the
confirmation run *improved* the mean (0.0873 over seeds 0+100 versus 0.0900 at
seed 0 alone), i.e. its two seeds agree to within 3%. `lr=3e-4` is 2.3× worse
(0.2023).

GIN is also the only architecture that prefers 3 layers to 2. Its sum
aggregation with a learned MLP is injective on multisets, so an extra hop adds
usable structure instead of smoothing it away — but only up to a point: 8 × 128
is 37× worse, the same depth collapse as everywhere else.

### `transformer` — 2 layers, hidden 128, lr 1e-3 (188,516 params)

| Rank | Config | Mean val (s0) |
|---|---|---:|
| 1 | **2 × 128** | **0.1423** |
| 2 | 2 × 64 | 0.2448 |
| 3 | 3 × 128 | 0.3659 |
| … | 8 × 32 | 16.3059 |

Gap **72%**, no tie; confirmed at seed 100 with a 3% improvement (0.1380).
`lr=3e-4` is 3.6× worse (0.5161). The runner-up is the same depth at half the
width, so this selection is a statement about width, not depth: 128 is worth a
1.7× error reduction at 3.8× the parameters.

### `nnconv` — 2 layers, hidden 128, lr 1e-3 (1,166,660 params)

| Rank | Config | Mean val (s0) | Mean val (s100) | Mean over both |
|---|---|---:|---:|---:|
| 1 | **2 × 128** | **0.1089** | **0.0907** | **0.0998** |
| 2 | 2 × 64 | 0.1141 | 0.1029 | 0.1085 |
| 3 | 3 × 32 | 0.2146 | — | — |
| … | 8 × 128 | 2.3e+07 | — | — |
| stage 2 | 2 × 128, lr 3e-4 | 0.1950 | 0.1738 | 0.1844 |

This is **the only architecture where the declared 5% tie-break fired**: the gap
to 2 × 64 is 4.8%. Both candidates were therefore re-scored at seed 100, and
2 × 128 won again on the two-seed mean (0.0998 vs 0.1085, an 8.7% margin). The
tie-break was declared in advance precisely so that a 4.8% gap could not be
resolved by inspection after the fact.

The per-grid detail is worth recording, because it is the clearest instance of
pooled selection disagreeing with a single grid:

| Grid | 2 × 128 (s0 / s100) | 2 × 64 (s0 / s100) |
|---|---:|---:|
| IEEE24 | 0.0121 / 0.0158 | 0.0130 / 0.0251 |
| IEEE39 | 0.3161 / 0.2621 | **0.2625 / 0.2572** |
| IEEE118 | 0.0808 / 0.0697 | 0.1278 / 0.0942 |
| UK | 0.0266 / 0.0151 | 0.0532 / 0.0351 |

2 × 64 is better on IEEE39 at **both** seeds; 2 × 128 wins the other three and
the pooled mean. The frozen config is the pooled winner by protocol (§1) — a
per-grid choice would leave the pooled-grid OOD arm undefined and would confound
architecture with configuration — and the disagreement is reported here rather
than smoothed over. It is the same effect the per-grid argmin diagnostic
(`results_a/*/tuning_per_grid_argmin.csv`) exists to expose.

The stage-2 row also settles the learning rate for this architecture: at
lr 3e-4 the same shape is 1.8× worse at both seeds, so 1e-3 is not a marginal
preference.

NNConv also shows the depth collapse most violently — 8 × 128 reaches 2.3e+07
mean validation loss — which is expected: its edge network emits a full
`hidden × hidden` transform per edge, so stacked layers compose eight learned
dense operators per message path.

Two disclosures belong with this architecture:

1. **NNConv is not a baseline of either source paper.** PowerGraph-Node reports
   GCNConv, GATConv, GINEConv and TransformerConv; ARMA comes from ENGAGE via
   Hansen et al. NNConv was added by this benchmark as the most
   edge-expressive layer available, because edge admittances are the physical
   carrier of power flow, and a comparison in which no architecture conditions
   its message on a *learned matrix* of the edge features would leave that
   hypothesis untested.
2. **NNConv's final runs use 3 seeds (0, 100, 300), not 5.** At hidden 128 a
   single IEEE118 training is ~3 h on this machine, putting its 60 final runs at
   1.5–2 days of wall clock. The reduced replication is disclosed in the results
   tables and in the limitations, and NNConv's standard deviations are therefore
   not comparable in precision to the other five.

### `arma_gnn` — 8 layers, hidden 128, lr 1e-3 (1,367,909 params)

| Rank | Config | Mean val (s0) | Mean val (s100) |
|---|---|---:|---:|
| 1 | **8 × 128** | **0.0183** | **0.0183** |
| 2 | 8 × 64 | 0.0193 | 0.0223 |
| 3 | 8 × 128 @ 3e-4 | 0.0602 | 0.0723 |
| 4 | 8 × 32 | 0.0750 | 0.0845 |
| 5 | 3 × 128 | 0.0802 | 0.0791 |
| … | 2 × 32 | 0.3660 | 0.3005 |

Gap **5.4%** — just outside the tie band, and the winner is the same at the
confirmation seed with a reproducibility of 0.1% (0.018288 → 0.018313), the
tightest of all six architectures. ARMA is also the **best-scoring architecture
in the sweep by a factor of 5** against the next architecture's leader, and the
only one that wants 8 layers: its recursive filter approximates a rational
rather than a polynomial response, so depth buys frequency resolution instead
of over-smoothing.

`lr=3e-4` is 3.6× worse. Every one of the ten candidates is stable at both
seeds on all four grids — **zero non-finite trials**, which matters because that
was not true before the fix described next.

## 4. The two protocol amendments, and why they were made

Both were made *after* seeing results and are disclosed as such. Neither
changes the five architectures whose runs were already complete.

### 4.1 Divergence is disqualification, not a bad score

The original rule took the argmin of the mean validation loss over a single
seed. ARMA's first sweep recorded `inf` for all three hidden-64 candidates; the
rule ranked them last and moved on, and froze 8 × 128 / lr 1e-3 on the strength
of seed 0. That configuration then diverged to NaN in **10 of 20** Regime A
within-grid runs and **49 of 80** cross-context runs: seeds 700 and 1000 failed
everywhere, seed 100 on most grids, and only seeds 0 and 300 survived. Tuning
had scored seed 0 — one of the two that happened to work.

The defect is the selection rule: *a candidate whose training diverges is a
failed candidate, not a candidate with a bad number*, and a winner that is never
re-run cannot be known to be reproducible. Hence the stability rule in §1. It
is applied identically to all six architectures; for `gcn`, `gat`, `gin`,
`transformer` and `nnconv` it changes nothing (no divergence at any seed), so
their completed results stand unaltered.

### 4.2 ARMA's edge weight is made non-negative

Applying the stability rule to ARMA's original sweep disqualified the *entire*
search space: re-scored at seed 100, every one of the nine depth×width
candidates diverged on IEEE39 and UK, at both learning rates, including
2 × 32. An architecture that cannot be trained at any setting in the space
cannot be frozen — so the cause had to be found rather than tuned around.

It is not the data (the same Regime A tensors train the other five
architectures with zero non-finite values, and ARMA itself is finite at seed 0
on the same grids), and it is not gradient explosion (a direct probe of
8 × 128 / lr 1e-3 on IEEE39 and UK at seeds 100 and 700 produced `inf` with
`clip_grad_norm_(…, 1.0)` exactly as without it).

The mechanism is in the layer. PyG's `ARMAConv` normalizes the adjacency with

```python
gcn_norm(edge_index, edge_weight, ..., add_self_loops=False, ...)
```

whereas `GCNConv` uses `add_self_loops=True`. The shared edge encoder ended in
a leaky ReLU, so the learned scalar edge weight could be **negative**; a bus
whose incident weights sum to zero or below then yields `deg ** -0.5` = inf/NaN
inside the normalization, before any gradient exists. GCN never sees this
because its self-loops of weight 1 keep every degree positive.

The fix is three lines and is **scoped to ARMA**: its scalar edge weight passes
through `softplus` instead of leaky ReLU, which guarantees a non-negative
weight and keeps the normalization defined. The other five architectures'
encoders are untouched and bit-for-bit unchanged, so their completed runs remain
valid. `tests/test_plumbing.py` asserts the ARMA weight is non-negative and
that GCN's is deliberately left as it was.

The effect is not merely "no longer NaN" — it is better training. On the seeds
that always failed:

| Grid | seed 100 | seed 700 |
|---|---:|---:|
| IEEE24 | 0.00629 | 0.00828 |
| IEEE39 | 0.000374 | 0.000349 |
| UK | 0.0282 | 0.0164 |

and the three previously-`inf` hidden-64 candidates became finite, with 8 × 64
now the runner-up. ARMA's sweep was therefore re-run in full under the fixed
layer (`results_a/arma_v2`, 10 candidates × 2 seeds × 4 grids, zero
divergences) and only ARMA's three final arms were re-run. The superseded
sweep is kept for provenance in `results_a/arma_gnn` and
`results_a/arma_stability`.

Gradient clipping is available as an opt-in `grad_clip=None` argument of
`train()` and is **not** used by any reported run: enabling it globally would
have altered the optimization of the five architectures that never needed it and
invalidated every completed result, to fix a problem that turned out not to be
gradient explosion at all.

## 5. What is deliberately *not* done

- **No per-grid tuning.** Per-grid configurations would make the cross-context
  drop larger and un-interpretable, because a Regime-A→B ranking change could
  then be attributed to the re-tuning rather than to the architecture. The
  per-grid argmin is nevertheless recorded (`tuning_per_grid_argmin.csv`) so the
  question can be answered with evidence.
- **No "practitioner arm"** (tune on the training grid, transfer from there).
  Considered and dropped by decision of the study owner; the frozen-config
  design is the whole comparison.
- **No test data in selection.** Every number in this document is a validation
  score.
- **No inherited defaults.** `--arch_config` is mandatory in `experiments.py`;
  falling back silently to ENGAGE's 8 layers / hidden 64 — which were never
  selected under any protocol here — would invalidate the comparison. Only
  `--allow_default_config` (smoke tests) bypasses it.

## 6. Reproducing the selection

```bash
# per-architecture sweep (resumable; trials are cached by their full key)
python3 tune_budget.py --data_dir data_a --epochs 200 --models gcn \
    --out results_a/gcn --config_out results_a/gcn/arch_config.json --skip_existing

# a slow architecture's remaining trials can be spread over processes ...
python3 prefill_trials.py --model nnconv --grids IEEE118 --num_layers 2 \
    --hidden 128 --learning_rates 1e-3 3e-4 --seeds 100 --out results_a/nnconv_bygrid/tie_IEEE118

# ... then unioned, after which the staged selection retrains nothing
python3 gather_trials.py --shards 'results_a/nnconv_bygrid/*' --out results_a/nnconv
python3 tune_budget.py --models nnconv --out results_a/nnconv --skip_existing
```

## 7. Verification of the ARMA re-run

The point of the remediation is only made if the replacement runs are clean, so
they were checked as a gate rather than assumed. ARMA's three arms were re-run
at the corrected layer and the frozen 8 × 128 / lr 1e-3 configuration, one
process per seed (`results_a/within_arma_v2_s<seed>`,
`results_tuned/arma_v2_s<seed>`), then unioned with

```bash
python3 gather_results.py --shards 'results_a/within_arma_v2_s*' \
    --file within_grid.csv --out results_a/within_arma_v2 \
    --models arma_gnn --seed_shards
```

`--seed_shards` is new: consolidation previously required one shard per
architecture, which cannot express "one shard per seed of one architecture".
It relaxes the `seeds` agreement check to a union and enforces uniqueness on
`(model, seed)` instead of `model`, so a re-run seed still cannot silently
appear twice.

**Result: 20 within-grid + 80 cross-context + 20 OOD rows, 5 seeds, zero
non-finite values** — against 10/20 and 49/80 diverged before the fix. Mean
test NRMSE over the 5 seeds:

| Arm | IEEE24 | IEEE39 | IEEE118 | UK |
|---|---:|---:|---:|---:|
| Regime A within-grid | 0.00042 | 0.00030 | 0.00111 | 0.00077 |
| Regime B OOD (grid held out) | 0.1536 | 0.1247 | 0.1023 | 0.1479 |

Seed-to-seed spread in Regime A is now ≤ 0.00017 (std), and the cross-context
diagonal (0.0053 mean) versus off-diagonal (0.5789 mean) shows the transfer
penalty the study is about — a two-orders-of-magnitude gap that the diverged
runs could not have measured at all.

## 8. A caution on sharding

**Two processes must never share an output
directory.** Each loads the shard's `tuning.csv` at start and rewrites it at
end, so the second silently drops the first's rows — this destroyed GAT's
confirmation trials once and they had to be recomputed in their own shard
(`results_a/gat_confirm/<grid>`). Shard by directory, then `gather_trials.py`.

The same hazard applied to the frozen configuration file itself: a per-model
sweep wrote `--config_out` as a whole document, so `--models nnconv` replaced a
file holding all six architectures with one holding only `nnconv`. It was caught
by inspection and the file restored from the pushed copy — no experiment ran on
a wrong configuration, since the swept model's entry was identical and every
final run names its own model. `tune_budget.merge_config` now folds a sweep's
result into whatever is already frozen, replacing only the swept models'
entries and keeping the provenance keys earlier sweeps recorded
(`tests/test_tune_budget.py::test_config_merge_keeps_other_models`).
