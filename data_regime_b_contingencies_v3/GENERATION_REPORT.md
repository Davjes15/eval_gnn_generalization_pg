# Regime B (line contingencies, blocked time split) -- generation report

## Generation

Generated from `data_regime_b_contingencies_v3` by the v3 generation runner
in `/home/ubuntu/v3_logs/gen.sh`, using the pinned environment and
`OMP_NUM_THREADS=1` with `POWERGRAPH_NODE_DIR` set to
`/home/ubuntu/engage_src/PowerGraph-Node/13_Power_system`. `--seed` was not
passed; `transmission_graph_gen.py` defaults to 12 and derives per-grid RNG
seeds as `seed * 100 + grid offset` (IEEE24=1201, IEEE39=1202, IEEE118=1203,
UK=1204).

The four invocations, from `/home/ubuntu/repos/v3_gen`:

```text
/home/ubuntu/venvs/eval_gnn_pinned/bin/python -u transmission_graph_gen.py --grid IEEE118 --max_k 2 --time_split blocked --n_train 800 --n_val 100 --n_test 100 --outage_types line --q_mode base --out_dir data_regime_b_contingencies_v3
/home/ubuntu/venvs/eval_gnn_pinned/bin/python -u transmission_graph_gen.py --grid UK --max_k 2 --time_split blocked --n_train 800 --n_val 100 --n_test 100 --outage_types line --q_mode base --out_dir data_regime_b_contingencies_v3
/home/ubuntu/venvs/eval_gnn_pinned/bin/python -u transmission_graph_gen.py --grid IEEE39 --max_k 2 --time_split blocked --n_train 800 --n_val 100 --n_test 100 --outage_types line --q_mode base --out_dir data_regime_b_contingencies_v3
/home/ubuntu/venvs/eval_gnn_pinned/bin/python -u transmission_graph_gen.py --grid IEEE24 --max_k 2 --time_split blocked --n_train 800 --n_val 100 --n_test 100 --outage_types line --q_mode base --out_dir data_regime_b_contingencies_v3
```

## Validation and splits

`/home/ubuntu/v3_logs/validate_b.log` reports `ALL GATES PASSED`. Each grid
has 800 train, 100 validation, and 100 test samples; the splits use disjoint
blocked time windows, with unique scenarios in each split.

| grid | buses | directed edges (2E) across samples |
|---|---:|---:|
| IEEE24 | 24 | 72, 74, 76 |
| IEEE39 | 39 | 88, 90, 92 |
| IEEE118 | 118 | 368, 370, 372 |
| UK | 29 | 194, 196, 198 |

## Old/new sanity checks

Comparing the old and v3 datasets with `training_utils.load_grid_dataset`
across all splits found `x` and `y` value-identical on all four grids
(including matching non-finite masks). IEEE24 and IEEE39 edge data are
unchanged. IEEE118 reaches 372 directed edges versus 368 in the old data,
with no isolated buses; UK reaches 198 versus 180 directed edges and one
connected component instead of two. Outage IDs remain the same line
contingencies; their metadata label changes from `13` to `line:13`. The
`dataset_src.csv` files add the `q_mode` column, populated as `base`.
