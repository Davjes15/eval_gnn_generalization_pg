# V3 campaign results

## Key findings

- Within-grid rank predicts the Regime B diagonal rank (pooled Kendall tau-b 0.867, permutation p = 0.007), but not cross-context (tau-b 0.200, p = 0.635) or OOD (tau-b 0.200, p = 1.000).
- ARMA-GNN ranks first within-grid, on the Regime B diagonal, and in cross-context. Transformer ranks first OOD, with GIN a close second.
- NNConv is worst in both transfer arms. One GCN cross-context result is non-finite.

## Campaign protocol

The published campaign uses `data_regime_a_fixed_topology_v3` for within-grid
training and `data_regime_b_contingencies_v3` for cross-context and OOD
evaluation. It compares GCN, ARMA-GNN, GAT, GIN, Transformer, and NNConv after
200 epochs with `pu_zscore` normalization. Seeds are 0, 100, 300, 700, and 1000,
except NNConv, which uses 0, 100, and 300. MMD uses the multigraph estimator and
the shared reference ruler in `configs/mmd_bandwidth.json`.

## 4. Results

### 4.0 Population and void rule

A non-finite transfer error is not a measurement. The whole
`(arm, model, train_grid, seed)` group containing it is voided so that a model
does not benefit from averaging only its easier surviving test cases. The v3
campaign has one non-finite cross-context row: GCN, seed 1000, trained on IEEE39
and tested on IEEE118. The corresponding GCN seed-1000 cell is voided in
complete-cell summaries. In `gscore_cc_aggregate.csv` its `finite_rate` is
11/12 = 0.9167; the cross-context finite rate aggregated over seeds is 0.98.
`ac_feasibility_norm.csv` records the dropped rows in `n_voided`.

The all-entry NRMSE tables score all target entries, including values re-injected
from known bus types. The physics replay instead scores entries actually
predicted by the network. These are different populations and must not be
compared as if they were the same metric. Rank permutation tests use the 12
complete (grid, seed) cells common to all six architectures; NNConv has only
three seeds.

### 4.1 Ranking by arm

| Arm | Model | Rank | Mean NRMSE | SD | n |
|---|---|---:|---:|---:|---:|
| Within | arma_gnn | 1 | 0.000426 | 0.000263 | 20 |
| Within | gin | 2 | 0.003671 | 0.005544 | 20 |
| Within | transformer | 3 | 0.004774 | 0.005506 | 20 |
| Within | nnconv | 4 | 0.004799 | 0.005515 | 12 |
| Within | gat | 5 | 0.004822 | 0.005563 | 20 |
| Within | gcn | 6 | 0.010560 | 0.013528 | 20 |
| Regime B diagonal | arma_gnn | 1 | 0.004943 | 0.001500 | 20 |
| Regime B diagonal | gin | 2 | 0.008486 | 0.004526 | 20 |
| Regime B diagonal | transformer | 3 | 0.008516 | 0.004664 | 20 |
| Regime B diagonal | gat | 4 | 0.008699 | 0.004646 | 20 |
| Regime B diagonal | nnconv | 5 | 0.009556 | 0.004027 | 12 |
| Regime B diagonal | gcn | 6 | 0.015772 | 0.014968 | 20 |
| Cross-context | arma_gnn | 1 | 0.755250 | 1.030760 | 20 |
| Cross-context | gcn | 2 | 0.881576 | 1.170064 | 20 |
| Cross-context | gin | 3 | 0.931274 | 1.624911 | 20 |
| Cross-context | gat | 4 | 1.060844 | 1.575006 | 20 |
| Cross-context | transformer | 5 | 1.134608 | 1.934364 | 20 |
| Cross-context | nnconv | 6 | 3.115400 | 5.389949 | 12 |
| OOD | transformer | 1 | 0.272616 | 0.201994 | 20 |
| OOD | gin | 2 | 0.294955 | 0.185730 | 20 |
| OOD | gat | 3 | 0.423609 | 0.521425 | 20 |
| OOD | arma_gnn | 4 | 0.977271 | 0.938527 | 20 |
| OOD | gcn | 5 | 1.440131 | 1.986141 | 20 |
| OOD | nnconv | 6 | 4.018067 | 5.141343 | 12 |

Source: `results_v3/analysis/ranking_table.csv`; columns `arm`, `metric` (NRMSE), `model`, `rank_overall`, `mean`, `sd`, and `n`.

### 4.2 Protocol decomposition

| Model | Within | B diagonal | Cross-context | OOD | Same-grid factor | Unseen factor CC | Total factor CC | Unseen factor OOD | Total factor OOD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arma_gnn | 0.000426 | 0.004943 | 0.755250 | 0.977271 | 11.603605 | 152.793362 | 1772.953891 | 197.710150 | 2294.150577 |
| gat | 0.004822 | 0.008699 | 1.060844 | 0.423609 | 1.804059 | 121.955807 | 220.015411 | 48.698519 | 87.854977 |
| gcn | 0.010560 | 0.015772 | 0.881576 | 1.440131 | 1.493601 | 55.894759 | 83.484474 | 91.308921 | 136.379105 |
| gin | 0.003671 | 0.008486 | 0.931274 | 0.294955 | 2.311916 | 109.736811 | 253.702290 | 34.756052 | 80.353073 |
| nnconv | 0.004799 | 0.009556 | 3.115400 | 4.018067 | 1.991340 | 326.013184 | 649.203187 | 420.473414 | 837.305649 |
| transformer | 0.004774 | 0.008516 | 1.134608 | 0.272616 | 1.783855 | 133.239513 | 237.679916 | 32.013915 | 57.108170 |

Source: `results_v3/analysis/protocol_decomposition.csv`; all displayed columns, filtered to `metric=nrmse`.

### 4.3 Per-quantity NRMSE

Values are mean NRMSE in physical units, over all target entries.

| Within model | P | Q | V | theta |
|---|---:|---:|---:|---:|
| arma_gnn | 0.000213 | 0.000496 | 0.000756 | 0.001973 |
| gat | 0.000658 | 0.006407 | 0.010820 | 0.013948 |
| gcn | 0.004561 | 0.011686 | 0.014968 | 0.021663 |
| gin | 0.000380 | 0.004674 | 0.001250 | 0.008904 |
| nnconv | 0.000568 | 0.006392 | 0.010981 | 0.013837 |
| transformer | 0.000466 | 0.006333 | 0.010793 | 0.013663 |

Source: `results_v3/analysis/per_quantity.csv`; columns `arm`, `model`,
`quantity`, and `nrmse_mean`, filtered to `arm=regime_a`.

| Cross-context model | P | Q | V | theta |
|---|---:|---:|---:|---:|
| arma_gnn | 0.337116 | 0.608662 | 0.447395 | 1.340527 |
| gat | 0.418086 | 1.023476 | 1.006813 | 2.373147 |
| gcn | 0.299886 | 0.863640 | 2.891518 | 2.102933 |
| gin | 0.219871 | 1.192718 | 6.259818 | 6.155972 |
| nnconv | 1.385922 | 2.248181 | 1.319018 | 4.478994 |
| transformer | 0.351554 | 1.345145 | 1.060107 | 2.164599 |

Source: `results_v3/analysis/per_quantity.csv`; columns `arm`, `model`,
`quantity`, and `nrmse_mean`, filtered to `arm=cross_context`.

| OOD model | P | Q | V | theta |
|---|---:|---:|---:|---:|
| arma_gnn | 0.587022 | 1.083949 | 1.041046 | 1.349043 |
| gat | 0.249184 | 0.401498 | 0.801389 | 1.373111 |
| gcn | 1.066635 | 0.805109 | 1.399038 | 1.548869 |
| gin | 0.131147 | 0.411694 | 4.191922 | 2.184624 |
| nnconv | 3.141298 | 1.309689 | 0.981614 | 2.588530 |
| transformer | 0.161916 | 0.267620 | 1.000884 | 1.544857 |

Source: `results_v3/analysis/per_quantity.csv`; columns `arm`, `model`,
`quantity`, and `nrmse_mean`, filtered to `arm=ood`.

### 4.4 DC baseline

| Grid | Regime A NRMSE | Regime B NRMSE |
|---|---:|---:|
| IEEE24 | 0.065065 | 0.060854 |
| IEEE39 | 0.097855 | 0.105498 |
| IEEE118 | 0.051496 | 0.051151 |
| UK | 0.066463 | 0.050214 |

Source: `results_v3/dc_baseline_regime_a.csv` and `results_v3/topology/dc_baseline.csv`; columns `grid`, `dc_nrmse`.
| Arm | Model | P ratio | Q ratio | V ratio | theta ratio |
|---|---|---:|---:|---:|---:|
| regime_a | arma_gnn | 0.017998 | 0.004171 | 0.004298 | 0.030410 |
| regime_a | gat | 0.055482 | 0.053898 | 0.061480 | 0.215022 |
| regime_a | gcn | 0.384877 | 0.098303 | 0.085050 | 0.333950 |
| regime_a | gin | 0.032054 | 0.039319 | 0.007102 | 0.137268 |
| regime_a | nnconv | 0.047930 | 0.053771 | 0.062393 | 0.213312 |
| regime_a | transformer | 0.039289 | 0.053274 | 0.061325 | 0.210631 |
| cross_context | arma_gnn | 29.593194 | 5.352670 | 2.909334 | 30.024810 |
| cross_context | gat | 36.700972 | 9.000606 | 6.547133 | 53.153220 |
| cross_context | gcn | 26.324941 | 7.594988 | 18.803042 | 47.101008 |
| cross_context | gin | 19.301046 | 10.488953 | 40.706507 | 137.880074 |
| cross_context | nnconv | 121.660813 | 19.770858 | 8.577344 | 100.319504 |
| cross_context | transformer | 30.860566 | 11.829418 | 6.893692 | 48.482187 |
| ood | arma_gnn | 51.530708 | 9.532417 | 6.769741 | 30.215568 |
| ood | gat | 21.874229 | 3.530841 | 5.211293 | 30.754629 |
| ood | gcn | 93.632733 | 7.080256 | 9.097704 | 34.691218 |
| ood | gin | 11.512526 | 3.620502 | 27.259344 | 48.930713 |
| ood | nnconv | 275.753528 | 11.517610 | 6.383262 | 57.977301 |
| ood | transformer | 14.213530 | 2.353493 | 6.508576 | 34.601349 |

Source: `results_v3/analysis/dc_comparison.csv`; columns `arm`, `model`, `quantity`, and `gnn_over_dc`, reshaped without aggregation.

### 4.5 G-scores

G-scores are means over complete seeds. Incomplete cells have no
g-score: omitting failed test points would reward a model for failing on harder
cases. The `gscore.csv` per-training-grid table is untrimmed; the former small-N
trim would collapse three unseen points and distort the mean and standard
deviation. The `dc_pf` row is a reference with one point per grid, unlike the
GNN's unseen train/test pairs.

| Cross-context model | Complete seeds | Mean NRMSE | SD | g | Finite rate |
|---|---:|---:|---:|---:|---:|
| dc_pf | 5 | 0.0669 | 0.0227 | 0.0669 | 1.00 |
| arma_gnn | 5 | 0.7552 | 1.2989 | 1.9028 | 1.00 |
| gcn | 4 | 0.8303 | 1.4380 | 2.1008 | 0.98 |
| gin | 5 | 0.9313 | 1.7943 | 2.5166 | 1.00 |
| gat | 5 | 1.0608 | 2.0795 | 2.8981 | 1.00 |
| transformer | 5 | 1.1346 | 2.5912 | 3.4240 | 1.00 |
| nnconv | 3 | 3.1154 | 6.3030 | 8.6842 | 1.00 |

Source: `results_v3/analysis/gscore_cc_aggregate.csv`; columns `model`,
`mean_nrmse`, `std_nrmse`, `g_score`, `n_finite`, and `n_expected`. The complete
seed averages and aggregate finite rates are reported as supplied for this release.

| OOD model | Complete seeds | Mean NRMSE | SD | g | Finite rate |
|---|---:|---:|---:|---:|---:|
| transformer | 5 | 0.2726 | 0.1518 | 0.4087 | 1.00 |
| gin | 5 | 0.2950 | 0.1619 | 0.4400 | 1.00 |
| gat | 5 | 0.4236 | 0.4575 | 0.8337 | 1.00 |
| arma_gnn | 5 | 0.9773 | 0.8387 | 1.7290 | 1.00 |
| gcn | 5 | 1.4401 | 1.5721 | 2.8491 | 1.00 |
| nnconv | 3 | 4.0181 | 4.8541 | 8.3687 | 1.00 |

Source: `results_v3/analysis/gscore_ood.csv`; columns `model`, `mean_nrmse`,
`std_nrmse`, `g_score`, `n_finite`, and `n_expected`.

The shared multiplier is `c = log(Δ+1)/Δ`, where Δ is `mmd_range`: cross-context
Δ = 0.274779 gives c = 0.883520; OOD Δ = 0.240053 gives c = 0.896278.
Sources: `results_v3/analysis/gscore_cc_aggregate.csv` and
`results_v3/analysis/gscore_ood.csv`, column `mmd_range`.

### 4.6 MMD distances

Biased multigraph/reference MMD distances between Regime B training-grid distributions:

| Train | Test | Degree | Laplacian | Reactance |
|---|---|---:|---:|---:|
| IEEE24 | IEEE24 | 0.035966 | 0.063792 | 0.003610 |
| IEEE24 | IEEE39 | 0.968586 | 0.748817 | 1.125722 |
| IEEE24 | IEEE118 | 1.025426 | 0.715991 | 0.549960 |
| IEEE24 | UK | 1.323868 | 0.959212 | 1.168827 |
| IEEE39 | IEEE24 | 0.970355 | 0.741215 | 1.125776 |
| IEEE39 | IEEE39 | 0.029603 | 0.054362 | 0.005524 |
| IEEE39 | IEEE118 | 1.088139 | 0.728052 | 1.198464 |
| IEEE39 | UK | 1.363398 | 0.902908 | 0.651650 |
| IEEE118 | IEEE24 | 1.018466 | 0.702208 | 0.548839 |
| IEEE118 | IEEE39 | 1.078138 | 0.726961 | 1.198502 |
| IEEE118 | IEEE118 | 0.004505 | 0.018614 | 0.001785 |
| IEEE118 | UK | 1.366558 | 0.976986 | 1.187427 |
| UK | IEEE24 | 1.329209 | 0.937579 | 1.168821 |
| UK | IEEE39 | 1.363365 | 0.896070 | 0.656305 |
| UK | IEEE118 | 1.366143 | 0.967501 | 1.187463 |
| UK | UK | 0.031399 | 0.051663 | 0.002562 |

Source: `docs/tables/mmd_data_regime_b_contingencies_v3_multigraph_reference.csv`; columns `train_grid`, `test_grid`, `mmd_degree`, `mmd_laplacian`, `mmd_reactance`, filtered to `estimator=biased`.

| Held-out grid | Training mixture | Pooled degree MMD | Pooled Laplacian MMD |
|---|---|---:|---:|
| IEEE24 | IEEE39+IEEE118+UK | 0.838342 | 0.622126 |
| IEEE39 | IEEE24+IEEE118+UK | 0.894918 | 0.606615 |
| IEEE118 | IEEE24+IEEE39+UK | 0.928413 | 0.636078 |
| UK | IEEE24+IEEE39+IEEE118 | 1.213765 | 0.846668 |

Source: `results_v3/topology/ood_distance.csv`; columns `held_out_grid`, `train_grids`, `mmd_pooled_degree`, and `mmd_pooled_laplacian`.

### 4.7 Physics replay and AC feasibility

`results_v3/physics/physics_metrics.csv` contains the 672 per-checkpoint replay
rows. It reports predicted-only quantity errors, voltage-band violations,
AC-balance residuals, and branch loading. These are row-level measurements, not
an aggregate table; no v3 aggregate equivalent to the former physics-summary
table is present, so no new aggregate is computed here.

`results_v3/analysis/ac_feasibility_norm.csv` contains the direct grouped
feasibility summary. Its `n_rows` and `n_voided` fields apply the void rule.
GCN has 19 diagonal rows with 1 void and 57 unseen rows with 3 voided; the other
models' counts are directly recorded in that CSV. The DC and truth reference
rows by grid are in `results_v3/physics/dc_feasibility.csv`.


| Setting | Model | dP % load | dQ % load | Max loading pred % | Max loading true % | n rows | n voided |
|---|---|---:|---:|---:|---:|---:|---:|
| cc_diagonal | arma_gnn | 109.658 | 43.963 | 727.512 | 409.199 | 20 | 0 |
| cc_diagonal | gat | 262.627 | 68.514 | 1161.04 | 409.199 | 20 | 0 |
| cc_diagonal | gcn | 355.418 | 99.076 | 1360.32 | 407.459 | 19 | 1 |
| cc_diagonal | gin | 152.19 | 60.666 | 840.227 | 409.199 | 20 | 0 |
| cc_diagonal | nnconv | 294.704 | 79.115 | 965.649 | 409.199 | 12 | 0 |
| cc_diagonal | transformer | 267.906 | 75.647 | 1104.23 | 409.199 | 20 | 0 |
| cc_unseen | arma_gnn | 4485.72 | 6814.92 | 7391.32 | 409.199 | 60 | 0 |
| cc_unseen | gat | 4115.74 | 7947.81 | 6300.02 | 409.199 | 60 | 0 |
| cc_unseen | gcn | 4597.48 | 8830.4 | 6916.98 | 409.779 | 57 | 3 |
| cc_unseen | gin | 7821.62 | 31504.3 | 9997.71 | 409.199 | 60 | 0 |
| cc_unseen | nnconv | 5385.57 | 9785.74 | 7613.06 | 409.199 | 36 | 0 |
| cc_unseen | transformer | 4655.75 | 9270.76 | 8011.91 | 409.199 | 60 | 0 |
| ood | arma_gnn | 3908.87 | 5628.19 | 6788.65 | 409.199 | 20 | 0 |
| ood | gat | 4140.03 | 7096.77 | 6690.85 | 409.199 | 20 | 0 |
| ood | gcn | 4298.27 | 7090.92 | 5854.81 | 409.199 | 20 | 0 |
| ood | gin | 4456.66 | 11542.2 | 7177.57 | 409.199 | 20 | 0 |
| ood | nnconv | 5257.92 | 8757.89 | 6245.75 | 409.199 | 12 | 0 |
| ood | transformer | 3881.6 | 8835.51 | 7885.15 | 409.199 | 20 | 0 |
| within | arma_gnn | 32.196 | 6.324 | 359.768 | 364.296 | 20 | 0 |
| within | gat | 184.582 | 46.126 | 1152.83 | 364.296 | 20 | 0 |
| within | gcn | 323.516 | 73.476 | 961.482 | 364.296 | 20 | 0 |
| within | gin | 62.294 | 12.092 | 389.054 | 364.296 | 20 | 0 |
| within | nnconv | 196.903 | 46.45 | 1113.79 | 364.296 | 12 | 0 |
| within | transformer | 184.699 | 43.239 | 1131.61 | 364.296 | 20 | 0 |
| regime_a | dc_pf | 34.597 | 161.602 | 292.206 | 364.296 | 4 | 0 |
| regime_a | floor | 0 | 0 | 364.296 | 364.296 | 4 | 0 |
| regime_b | dc_pf | 33.577 | 159.113 | 333.161 | 409.199 | 4 | 0 |
| regime_b | floor | 0 | 0 | 409.199 | 409.199 | 4 | 0 |

Source: `results_v3/analysis/ac_feasibility_norm.csv`; columns `setting`, `model`, `ac_dp_pct_load`, `ac_dq_pct_load`, `branch_loading_max_pct`, `branch_loading_max_pct_true`, `n_rows`, `n_voided`.

| Dataset | Grid | State | n samples | dP % load | dQ % load | Max loading pred % | Max loading true % | Missed overload rate | False overload rate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| data_regime_a_fixed_topology_v3 | IEEE24 | truth | 100 | 0.000111158 | 0.000467143 | 119.597 | 119.597 | 0 | 0 |
| data_regime_a_fixed_topology_v3 | IEEE24 | dc_pf | 100 | 27.7213 | 246.031 | 126.541 | 119.597 | 0 | 0.00951604 |
| data_regime_a_fixed_topology_v3 | IEEE39 | truth | 100 | 0.000312885 | 0.00039517 | 278.855 | 278.855 | 0 | 0 |
| data_regime_a_fixed_topology_v3 | IEEE39 | dc_pf | 100 | 22.0952 | 178.903 | 234.155 | 278.855 | 0.123007 | 0.00188071 |
| data_regime_a_fixed_topology_v3 | IEEE118 | truth | 100 | 0.000738631 | 0.000200705 | 659.481 | 659.481 | 0 | 0 |
| data_regime_a_fixed_topology_v3 | IEEE118 | dc_pf | 100 | 50.3435 | 109.097 | 391.953 | 659.481 | 0.332993 | 0 |
| data_regime_a_fixed_topology_v3 | UK | truth | 100 | 0.000185064 | 0.000227796 | 399.25 | 399.25 | 0 | 0 |
| data_regime_a_fixed_topology_v3 | UK | dc_pf | 100 | 38.2273 | 112.375 | 416.177 | 399.25 | 0.0203252 | 0.0108422 |
| data_regime_b_contingencies_v3 | IEEE24 | truth | 100 | 9.36381e-05 | 0.000480677 | 151.312 | 151.312 | 0 | 0 |
| data_regime_b_contingencies_v3 | IEEE24 | dc_pf | 100 | 27.4154 | 251.502 | 152.205 | 151.312 | 0.0111111 | 0.00970067 |
| data_regime_b_contingencies_v3 | IEEE39 | truth | 100 | 0.000342026 | 0.000323172 | 442.254 | 442.254 | 0 | 0 |
| data_regime_b_contingencies_v3 | IEEE39 | dc_pf | 100 | 22.9663 | 164.488 | 403.583 | 442.254 | 0.121659 | 0.000291715 |
| data_regime_b_contingencies_v3 | IEEE118 | truth | 100 | 0.000759411 | 0.000197367 | 677.228 | 677.228 | 0 | 0 |
| data_regime_b_contingencies_v3 | IEEE118 | dc_pf | 100 | 50.1321 | 105.842 | 393.091 | 677.228 | 0.338257 | 0 |
| data_regime_b_contingencies_v3 | UK | truth | 100 | 0.00014902 | 0.000265631 | 366.002 | 366.002 | 0 | 0 |
| data_regime_b_contingencies_v3 | UK | dc_pf | 100 | 33.795 | 114.619 | 383.767 | 366.002 | 0.0627119 | 0.00812065 |

Source: `results_v3/physics/dc_feasibility.csv`; columns `data_dir`, `grid`, `state`, `n_samples`, `ac_dp_pct_load`, `ac_dq_pct_load`, `branch_loading_max_pct`, `branch_loading_max_pct_true`, `missed_overload_rate`, and `false_overload_rate`.

### 4bis. GCN non-finite prediction

The single non-finite result is GCN, seed 1000, trained on IEEE39 and evaluated
on IEEE118. The leaky-ReLU scalar edge weight can become negative; a negative
weighted degree then makes `deg^-1/2` NaN. In v3 this affects 94 of 100 IEEE118
test samples, or 2,626 node predictions. The softplus correction was
deliberately not applied, preserving the model under study; the affected cell is
voided rather than assigned an error score.

### Limitations

The sample contains four grids, with grid size and topology confounded. NNConv
has three seeds rather than five. Seeds measure training randomness on fixed
generated datasets, not uncertainty across data draws. The within-grid split is
random, so nearby demand snapshots may appear across splits. Cross-context
Regime B changes both topology distribution and temporal split. Finally,
aggregate NRMSE uses all target entries, while physical replay scores only
predicted entries; neither metric alone establishes that a predicted state is a
valid AC operating point.

### Rank transfer

| Comparison | Kendall tau-b | Spearman rho | n cells | Observed mean tau | Permutation p |
|---|---:|---:|---:|---:|---:|
| Within vs Regime B diagonal | 0.866667 | 0.942857 | 12 | 0.544444 | 0.006944 |
| Within vs cross-context | 0.200000 | 0.257143 | 12 | 0.088889 | 0.634722 |
| Within vs OOD | 0.200000 | 0.371429 | 12 | -0.011111 | 1.000000 |

Sources: `results_v3/analysis/rank_correlation_pooled.csv` columns
`comparison`, `metric`, `kendall_tau_b`, and `spearman_rho`; and
`results_v3/analysis/rank_permutation_test.csv`, filtered to `metric=nrmse`,
columns `comparison`, `n_cells`, `observed_mean_tau`, and `p_value`.
