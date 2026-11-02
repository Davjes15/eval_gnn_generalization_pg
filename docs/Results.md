# v3 results

## Protocol

The campaign uses corrected v3 topology data, six architectures (ARMA-GNN,
NNConv, Transformer, GIN, GAT, GCN), 200 epochs, `pu_zscore`, and the within,
cross-context, and leave-one-grid-out OOD arms. Seeds are 0, 100, 300, 700,
1000 for all models except NNConv, which uses 0, 100, 300. MMD and g-scores use
the multigraph descriptor and the shared reference ruler in
`configs/mmd_bandwidth.json`.

Tables below are transcribed from the committed `results_v3` CSVs. NRMSE values
are aggregate physical-unit scores unless the quantity is named explicitly.
The sample count `n` is the contributing run count.

## Rankings by arm

Source: `results_v3/analysis/ranking_table.csv` (`metric=nrmse`).

| Arm | Rank | Model | Mean NRMSE | SD | n |
|---|---:|---|---:|---:|---:|
| Regime A | 1 | arma_gnn | 0.000426 | 0.000263 | 20 |
| Regime A | 2 | gin | 0.003671 | 0.005544 | 20 |
| Regime A | 3 | transformer | 0.004774 | 0.005506 | 20 |
| Regime A | 4 | nnconv | 0.004799 | 0.005515 | 12 |
| Regime A | 5 | gat | 0.004822 | 0.005563 | 20 |
| Regime A | 6 | gcn | 0.010560 | 0.013528 | 20 |
| Regime B diagonal | 1 | arma_gnn | 0.004943 | 0.001500 | 20 |
| Regime B diagonal | 2 | gin | 0.008486 | 0.004526 | 20 |
| Regime B diagonal | 3 | transformer | 0.008516 | 0.004664 | 20 |
| Regime B diagonal | 4 | gat | 0.008699 | 0.004646 | 20 |
| Regime B diagonal | 5 | nnconv | 0.009556 | 0.004027 | 12 |
| Regime B diagonal | 6 | gcn | 0.015772 | 0.014968 | 20 |
| Cross-context | 1 | arma_gnn | 0.755250 | 1.030760 | 20 |
| Cross-context | 2 | gcn | 0.881576 | 1.170060 | 20 |
| Cross-context | 3 | gin | 0.931274 | 1.624910 | 20 |
| Cross-context | 4 | gat | 1.060840 | 1.575010 | 20 |
| Cross-context | 5 | transformer | 1.134610 | 1.934360 | 20 |
| Cross-context | 6 | nnconv | 3.115400 | 5.389950 | 12 |
| OOD | 1 | transformer | 0.272616 | 0.201994 | 20 |
| OOD | 2 | gin | 0.294955 | 0.185730 | 20 |
| OOD | 3 | gat | 0.423609 | 0.521425 | 20 |
| OOD | 4 | arma_gnn | 0.977271 | 0.938527 | 20 |
| OOD | 5 | gcn | 1.440130 | 1.986140 | 20 |
| OOD | 6 | nnconv | 4.018070 | 5.141340 | 12 |

## Protocol decomposition

Source: `results_v3/analysis/protocol_decomposition.csv`.

| Model | Regime A | Regime B diagonal | Cross-context | OOD | Same-grid factor | Unseen factor CC | Total factor CC | Unseen factor OOD | Total factor OOD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arma_gnn | 0.000426 | 0.004943 | 0.755250 | 0.977271 | 11.604 | 152.793 | 1772.954 | 197.710 | 2294.151 |
| gat | 0.004822 | 0.008699 | 1.060844 | 0.423609 | 1.804 | 121.956 | 220.015 | 48.699 | 87.855 |
| gcn | 0.010560 | 0.015772 | 0.881576 | 1.440131 | 1.494 | 55.895 | 83.484 | 91.309 | 136.379 |
| gin | 0.003671 | 0.008486 | 0.931274 | 0.294955 | 2.312 | 109.737 | 253.702 | 34.756 | 80.353 |
| nnconv | 0.004799 | 0.009556 | 3.115400 | 4.018067 | 1.991 | 326.013 | 649.203 | 420.473 | 837.306 |
| transformer | 0.004774 | 0.008516 | 1.134608 | 0.272616 | 1.784 | 133.240 | 237.680 | 32.014 | 57.108 |

## Per-quantity NRMSE

Source: `results_v3/analysis/per_quantity.csv` (`nrmse_mean`, `nrmse_sd`,
`nrmse_n_nonfinite`). The full per-arm/model/quantity rows are in that CSV;
selected examples below show the within-grid and transfer regimes.

| Arm | Model | Quantity | Mean | SD | Non-finite |
|---|---|---|---:|---:|---:|
| Regime A | arma_gnn | P | 0.000213 | 0.000133 | 0 |
| Regime A | arma_gnn | Q | 0.000496 | 0.000298 | 0 |
| Regime A | arma_gnn | V | 0.000756 | 0.000393 | 0 |
| Regime A | arma_gnn | theta | 0.001973 | 0.001965 | 0 |
| Cross-context | arma_gnn | P | 0.337116 | 0.687627 | 0 |
| Cross-context | arma_gnn | Q | 0.608662 | 1.671370 | 0 |
| Cross-context | arma_gnn | V | 0.447395 | 0.672359 | 0 |
| Cross-context | arma_gnn | theta | 1.340530 | 3.017270 | 0 |
| OOD | transformer | P | 0.161916 | 0.117529 | 0 |
| OOD | transformer | Q | 0.267620 | 0.301843 | 0 |
| OOD | transformer | V | 1.000880 | 1.570170 | 0 |
| OOD | transformer | theta | 1.544860 | 1.460710 | 0 |

## DC baseline

Source: `results_v3/dc_baseline_regime_a.csv` (`dc_nrmse`).

| Grid | Aggregate NRMSE | P | Q | V | theta |
|---|---:|---:|---:|---:|---:|
| IEEE24 | 0.065065 | 0.010480 | 0.135647 | 0.171965 | 0.024718 |
| IEEE39 | 0.097855 | 0.007329 | 0.125856 | 0.222257 | 0.065846 |
| IEEE118 | 0.051496 | 0.018194 | 0.068519 | 0.126937 | 0.154253 |
| UK | 0.066463 | 0.011402 | 0.145473 | 0.182817 | 0.014652 |

The model-to-DC aggregate comparison by arm is in
`results_v3/analysis/dc_comparison.csv`.

## G-scores

Sources: `results_v3/analysis/gscore_cc_aggregate.csv` and
`results_v3/analysis/gscore_ood.csv`. Rows use seed 1000, except NNConv, which
uses seed 300 because it was run with three seeds.

| Arm | Seed | Model | Finite / expected | Finite rate | Mean NRMSE | SD | MMD range | G-score |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Cross-context | 1000 | arma_gnn | 12/12 | 1.000 | 0.760464 | 1.341098 | 0.274779 | 1.945352 |
| Cross-context | 1000 | gat | 12/12 | 1.000 | 1.210531 | 2.360619 | 0.274779 | 3.296186 |
| Cross-context | 1000 | gcn | 11/12 | 0.917 | 1.055286 | 1.030207 | 0.274779 | NaN |
| Cross-context | 1000 | gin | 12/12 | 1.000 | 0.658908 | 0.849779 | 0.274779 | 1.409705 |
| Cross-context | 300 | nnconv | 12/12 | 1.000 | 5.290900 | 12.429918 | 0.274779 | 16.272986 |
| Cross-context | 1000 | transformer | 12/12 | 1.000 | 1.709381 | 4.310343 | 0.274779 | 5.517657 |
| OOD | 1000 | arma_gnn | 4/4 | 1.000 | 0.857396 | 0.722582 | 0.240053 | 1.505030 |
| OOD | 1000 | gat | 4/4 | 1.000 | 0.626934 | 0.717290 | 0.240053 | 1.269825 |
| OOD | 1000 | gcn | 4/4 | 1.000 | 0.826697 | 0.611931 | 0.240053 | 1.375157 |
| OOD | 1000 | gin | 4/4 | 1.000 | 0.317409 | 0.181932 | 0.240053 | 0.480471 |
| OOD | 300 | nnconv | 4/4 | 1.000 | 3.493082 | 3.874199 | 0.240053 | 6.965438 |
| OOD | 1000 | transformer | 4/4 | 1.000 | 0.293238 | 0.143154 | 0.240053 | 0.421544 |

`finite_rate` and non-finite counts are reported by the source files rather
than silently treating invalid results as low errors.

## Physics replay and feasibility

The replay source is `results_v3/physics/physics_metrics.csv` (672 rows);
`results_v3/analysis/ac_feasibility_norm.csv` has 28 summary rows.

| Setting | Model | n rows | n voided |
|---|---|---:|---:|
| cc_diagonal | gcn | 19 | 1 |
| cc_unseen | gcn | 57 | 3 |

Other model-setting groups have `n_voided=0`. The DC/truth reference replay is
in `results_v3/physics/dc_feasibility.csv` (16 rows).

## Rank transfer

Source: `results_v3/compare_old_new/headline_rank_transfer.csv`.

| Comparison | n | Old Kendall tau-b | New Kendall tau-b | Mean tau old / new | Permutation p old / new |
|---|---:|---:|---:|---:|---:|
| Regime A vs cross-context | 20 / 20 | 0.600 / 0.200 | 0.010 / 0.010 | 0.067 / 0.089 | 0.717 / 0.635 |
| Regime A vs OOD | 20 / 20 | 0.200 / 0.200 | 0.033 / -0.011 | 0.000 / -0.011 | 1.000 / 1.000 |
| Regime A vs Regime B diagonal | 20 / 20 | 0.733 / 0.867 | 0.663 / 0.567 | 0.622 / 0.544 | 0.004 / 0.007 |

The full arm-level and grid-cell rank-transfer data are in
`results_v3/compare_old_new/arm_kendall_old_vs_new.csv` and
`cell_kendall_old_vs_new.csv`.

## One non-finite GCN transfer

`results_v3/analysis/nonfinite_runs.csv` records one non-finite run:
GCN, seed 1000, IEEE39 → IEEE118 in cross-context transfer. The scalar edge
weight uses leaky ReLU; its negative weighted degree reaches the symmetric
normalization `deg^-1/2`, producing NaNs. In v3 this affects 94 of 100 IEEE118
test samples and 2,626 node predictions. The softplus correction was
deliberately not applied; the run is reported and voided rather than repaired.

## Limitations

NNConv has three seeds while other architectures have five. The seeds measure
training randomness on these fixed datasets, not variability across new demand
or contingency draws. The reported OOD distances cover four held-out grids;
they do not establish performance on arbitrary unseen networks. For full
row-level outputs and provenance, see `results_v3/` and the `dataset_src.csv`
files in each v3 data split.
