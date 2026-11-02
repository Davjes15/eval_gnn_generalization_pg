# Old vs new (v3) results

Old: `/home/ubuntu/repos/eval_gnn_generalization_pg/results_norm`; new: `/home/ubuntu/repos/eval_gnn_generalization_pg/results_v3`. NRMSE is aggregate pu_zscore NRMSE; Kendall tau-b compares the old and new model orderings (6 models; NaN models dropped).

## 1. Per arm and architecture (overall ranking from rank_analysis)

### cross_context (Kendall tau old vs new = 0.47, n=6)

| model | old NRMSE (sd) | old rank | new NRMSE (sd) | new rank | change |
|---|---|---|---|---|---|
| arma_gnn | 0.8215 (1.1815) | 1 | 0.7552 (1.0308) | 1 | +0 |
| gcn | 1.0581 (1.5592) | 4 | 0.8816 (1.1701) | 2 | -2 |
| gin | 1.0045 (1.9378) | 3 | 0.9313 (1.6249) | 3 | +0 |
| gat | 1.1050 (1.6908) | 5 | 1.0608 (1.5750) | 4 | -1 |
| transformer | 0.9386 (1.4538) | 2 | 1.1346 (1.9344) | 5 | +3 |
| nnconv | 1.9854 (2.2918) | 6 | 3.1154 (5.3899) | 6 | +0 |

### ood (Kendall tau old vs new = 0.87, n=6)

| model | old NRMSE (sd) | old rank | new NRMSE (sd) | new rank | change |
|---|---|---|---|---|---|
| transformer | 0.2694 (0.2230) | 1 | 0.2726 (0.2020) | 1 | +0 |
| gin | 0.3869 (0.3719) | 3 | 0.2950 (0.1857) | 2 | -1 |
| gat | 0.3464 (0.2781) | 2 | 0.4236 (0.5214) | 3 | +1 |
| arma_gnn | 0.7977 (0.6321) | 4 | 0.9773 (0.9385) | 4 | +0 |
| gcn | 1.0435 (1.3118) | 5 | 1.4401 (1.9861) | 5 | +0 |
| nnconv | 3.4095 (4.2359) | 6 | 4.0181 (5.1413) | 6 | +0 |

### regime_a (Kendall tau old vs new = 0.87, n=6)

| model | old NRMSE (sd) | old rank | new NRMSE (sd) | new rank | change |
|---|---|---|---|---|---|
| arma_gnn | 0.0004 (0.0003) | 1 | 0.0004 (0.0003) | 1 | +0 |
| gin | 0.0037 (0.0055) | 2 | 0.0037 (0.0055) | 2 | +0 |
| transformer | 0.0039 (0.0057) | 3 | 0.0048 (0.0055) | 3 | +0 |
| nnconv | 0.0040 (0.0057) | 5 | 0.0048 (0.0055) | 4 | -1 |
| gat | 0.0040 (0.0057) | 4 | 0.0048 (0.0056) | 5 | +1 |
| gcn | 0.0100 (0.0137) | 6 | 0.0106 (0.0135) | 6 | +0 |

### regime_b_diagonal (Kendall tau old vs new = 0.73, n=6)

| model | old NRMSE (sd) | old rank | new NRMSE (sd) | new rank | change |
|---|---|---|---|---|---|
| arma_gnn | 0.0047 (0.0012) | 1 | 0.0049 (0.0015) | 1 | +0 |
| gin | 0.0086 (0.0045) | 4 | 0.0085 (0.0045) | 2 | -2 |
| transformer | 0.0079 (0.0051) | 2 | 0.0085 (0.0047) | 3 | +1 |
| gat | 0.0082 (0.0049) | 3 | 0.0087 (0.0046) | 4 | +1 |
| nnconv | 0.0089 (0.0046) | 5 | 0.0096 (0.0040) | 5 | +0 |
| gcn | 0.0155 (0.0151) | 6 | 0.0158 (0.0150) | 6 | +0 |

## 2. Per cell: Kendall tau of old vs new model ordering (seed-mean NRMSE)

| arm | cell | tau | n | best old | best new |
|---|---|---|---|---|---|
| cross_context | IEEE118->IEEE24 | 0.60 | 6 | nnconv | nnconv |
| cross_context | IEEE118->IEEE39 | 0.87 | 6 | transformer | transformer |
| cross_context | IEEE118->UK | 0.33 | 6 | arma_gnn | arma_gnn |
| cross_context | IEEE24->IEEE118 | 1.00 | 6 | transformer | transformer |
| cross_context | IEEE24->IEEE39 | 1.00 | 6 | gcn | gcn |
| cross_context | IEEE24->UK | 0.87 | 6 | arma_gnn | nnconv |
| cross_context | IEEE39->IEEE118 | 1.00 | 6 | arma_gnn | arma_gnn |
| cross_context | IEEE39->IEEE24 | 1.00 | 6 | gat | gat |
| cross_context | IEEE39->UK | 1.00 | 6 | nnconv | nnconv |
| cross_context | UK->IEEE118 | 0.07 | 6 | nnconv | arma_gnn |
| cross_context | UK->IEEE24 | 0.60 | 6 | arma_gnn | arma_gnn |
| cross_context | UK->IEEE39 | 0.87 | 6 | gin | gin |
| ood | heldout_IEEE118 | 0.60 | 6 | gat | arma_gnn |
| ood | heldout_IEEE24 | 0.73 | 6 | transformer | gin |
| ood | heldout_IEEE39 | 0.60 | 6 | transformer | gat |
| ood | heldout_UK | 0.87 | 6 | nnconv | nnconv |
| regime_a | IEEE118 | 0.60 | 6 | gin | arma_gnn |
| regime_a | IEEE24 | 1.00 | 6 | arma_gnn | arma_gnn |
| regime_a | IEEE39 | 1.00 | 6 | arma_gnn | arma_gnn |
| regime_a | UK | 0.73 | 6 | arma_gnn | arma_gnn |
| regime_b_diagonal | IEEE118 | 1.00 | 6 | gin | gin |
| regime_b_diagonal | IEEE24 | 1.00 | 6 | arma_gnn | arma_gnn |
| regime_b_diagonal | IEEE39 | 1.00 | 6 | arma_gnn | arma_gnn |
| regime_b_diagonal | UK | 0.87 | 6 | transformer | gat |

## 3. Headline rank-transfer statistics (Regime A ranking vs other arms)

| comparison | per-cell tau mean old (n) | new (n) | pooled tau old | new | perm p old | new |
|---|---|---|---|---|---|---|
| regime_a_vs_cross_context | 0.02±0.37 (20) | 0.01±0.30 (20) | 0.60 | 0.20 | 0.717 | 0.635 |
| regime_a_vs_ood | 0.02±0.38 (20) | 0.03±0.33 (20) | 0.20 | 0.20 | 1 | 1 |
| regime_a_vs_regime_b_diagonal | 0.66±0.21 (20) | 0.57±0.28 (20) | 0.73 | 0.87 | 0.00417 | 0.00694 |

## 4. g-score per model (mean over seeds)

### ood (Kendall tau old vs new = 0.87, n=6)

| model | g old | rank old | g new | rank new |
|---|---|---|---|---|
| transformer | 0.412 | 1 | 0.409 | 1 |
| gin | 0.652 | 3 | 0.440 | 2 |
| gat | 0.528 | 2 | 0.834 | 3 |
| arma_gnn | 1.311 | 4 | 1.729 | 4 |
| gcn | 1.976 | 5 | 2.849 | 5 |
| nnconv | 6.713 | 6 | 8.369 | 6 |

### cc_aggregate (Kendall tau old vs new = 0.62, n=7)

| model | g old | rank old | g new | rank new |
|---|---|---|---|---|
| dc_pf/one_point_per_grid | 0.067 | 1 | 0.067 | 1 |
| arma_gnn/unseen_pairs | 2.020 | 2 | 1.903 | 2 |
| gcn/unseen_pairs | 2.881 | 5 | 2.101 | 3 |
| gin/unseen_pairs | 2.696 | 4 | 2.517 | 4 |
| gat/unseen_pairs | 2.928 | 6 | 2.898 | 5 |
| transformer/unseen_pairs | 2.497 | 3 | 3.424 | 6 |
| nnconv/unseen_pairs | 4.507 | 7 | 8.684 | 7 |
