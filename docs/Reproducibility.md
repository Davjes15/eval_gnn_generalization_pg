# Reproducibility

## 1. Environment and data

Use Python 3.10 and install `requirements.txt`. The published campaign used
torch 2.13.0+cu130 on CPU, PyG 2.8.0, pandapower 3.5.4, numpy 2.2.1,
scipy 1.14.1, pandas 2.3.3, networkx 3.4.2, and numba 0.68.1. Set
`POWERGRAPH_NODE_DIR` to PowerGraph-Node's `13_Power_system` directory.

The corrected, committed datasets are `data_regime_a_fixed_topology_v3`,
`data_regime_a_blocked_v3`, and `data_regime_b_contingencies_v3`. Each contains
generation metadata and a `GENERATION_REPORT.md`. See the dataset reports for
the exact generation commands and validation results. Validate with:

```bash
python transmission_grids.py
python validate.py --data_dir data_regime_a_fixed_topology_v3 --regime a
python validate.py --data_dir data_regime_a_blocked_v3 --regime a --expect_blocked
python validate.py --data_dir data_regime_b_contingencies_v3 --regime b --expect_blocked
```

## 2. Training

Frozen model settings are in `configs/arch_config.json`. The published
campaign used six models (`arma_gnn nnconv transformer gin gat gcn`), 200
epochs, `pu_zscore`, and seeds `0 100 300 700 1000` (NNConv uses
`0 100 300`). `launch_campaign.py` schedules within, cross-context, and
leave-one-grid-out OOD jobs with small restartable shards. For a full campaign,
download the three checkpoint archives linked in the README or run the
campaign using the command shown by:

```bash
python launch_campaign.py --data_a data_regime_a_fixed_topology_v3 \
  --data_b data_regime_b_contingencies_v3 --out results_v3 \
  --ckpt ckpt_v3 --logs logs_v3 --dry_run
```

Campaign commands, status markers, and output layout are documented in
`launch_campaign.py --help`.

## 3. Downstream tables

After all jobs finish, merge each arm and regenerate the published tables.
Every MMD or g-score calculation uses `--mmd_graph multi --mmd_bandwidth
reference`; the shared ruler is `configs/mmd_bandwidth.json`.

```bash
python gather_results.py --shards "results_v3/within_*" --file within_grid.csv \
  --out results_v3/all_within --models arma_gnn nnconv transformer gin gat gcn --seed_shards
python gather_results.py --shards "results_v3/cross_*" --file cross_context.csv \
  --out results_v3/all_cross --models arma_gnn nnconv transformer gin gat gcn --seed_shards
python gather_results.py --shards "results_v3/ood_*" --file ood.csv \
  --out results_v3/all_ood --models arma_gnn nnconv transformer gin gat gcn --seed_shards
python mmd_report.py --data_dir data_regime_b_contingencies_v3 \
  --out results_v3/mmd_report --mmd_graph multi --mmd_bandwidth reference
python checkpoint_index.py --ckpt_root ckpt_v3 --out results_v3/checkpoint_index.csv
python eval_checkpoints.py --ckpt_root ckpt_v3 --data_a data_regime_a_fixed_topology_v3 \
  --data_b data_regime_b_contingencies_v3 --normalize pu_zscore --out results_v3/physics
python dc_feasibility.py --data data_regime_a_fixed_topology_v3 data_regime_b_contingencies_v3 \
  --out results_v3/physics/dc_feasibility.csv
python summarize_feasibility.py --physics results_v3/physics/physics_metrics.csv \
  --baselines results_v3/physics/dc_feasibility.csv \
  --out results_v3/analysis/ac_feasibility_norm.csv
python recompute_dc_baseline.py --data_dir data_regime_a_fixed_topology_v3 \
  --out results_v3/dc_baseline_regime_a.csv
python rank_analysis.py --regime_a results_v3/all_within/within_grid.csv \
  --cross results_v3/all_cross/cross_context.csv --ood results_v3/all_ood/ood.csv \
  --out results_v3/analysis
python recompute_tables.py --within results_v3/all_within/within_grid.csv \
  --cross results_v3/all_cross/cross_context.csv --ood results_v3/all_ood/ood.csv \
  --topology results_v3/topology --dc_regime_a results_v3/dc_baseline_regime_a.csv \
  --dc_regime_b results_v3/topology/dc_baseline.csv --out results_v3/analysis
```

`gather_results.py --seed_shards` checks protocol metadata, unique model/seed
rows, and duplicate shards before writing merged files. For Windows, see
[`Reproduce_on_Windows.md`](Reproduce_on_Windows.md).

## 4. Resume and atomic checkpoints

Training writes atomic final checkpoints and, when `--save_models` is enabled,
uses `<checkpoint>.resume` files every five completed epochs. Resume files are
used with `--skip_existing` and removed after the final `.pt` is written.
`launch_campaign.py` shards work by model/seed (and OOD fold), writes `DONE.json`
only for complete outputs, retries failed jobs, appends events to
`campaign_ledger.jsonl`, and supports `--status`. Merge completed shards with
`gather_results.py --seed_shards`, as in §3.

```bash
python launch_campaign.py --data_a data_regime_a_fixed_topology_v3 \
  --data_b data_regime_b_contingencies_v3 --out results_v3 \
  --ckpt ckpt_v3 --logs logs_v3 --pool 7
```
