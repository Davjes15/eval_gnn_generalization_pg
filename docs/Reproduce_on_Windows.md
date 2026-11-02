# Reproduce the v3 campaign on Windows

This guide assumes PowerShell and a Python 3.10 environment. Install the
packages in `requirements.txt`, set `POWERGRAPH_NODE_DIR` to the PowerGraph-Node
`13_Power_system` folder, and clone this repository.

The corrected datasets are committed as `data_regime_a_fixed_topology_v3`,
`data_regime_a_blocked_v3`, and `data_regime_b_contingencies_v3`. Their
generation commands and validation records are in the `GENERATION_REPORT.md`
files inside each folder. The published architecture settings are in
`configs\arch_config.json`; the common MMD ruler is
`configs\mmd_bandwidth.json`.

Validate the source grids and datasets:

```powershell
python transmission_grids.py
python validate.py --data_dir data_regime_a_fixed_topology_v3 --regime a
python validate.py --data_dir data_regime_b_contingencies_v3 --regime b --expect_blocked
```

The v3 campaign uses six architectures, 200 epochs, `pu_zscore`, seeds
`0 100 300 700 1000` (NNConv uses `0 100 300`), and within, cross-context, and
OOD arms. OOD holds out one grid per job. Preview the jobs, then remove
`--dry_run` to launch:

```powershell
python launch_campaign.py --data_a data_regime_a_fixed_topology_v3 `
  --data_b data_regime_b_contingencies_v3 --out results_v3 `
  --ckpt ckpt_v3 --logs logs_v3 --pool 7 --dry_run
```

The launcher records a `DONE.json` marker for completed jobs, retries failed
jobs, and appends events to `campaign_ledger.jsonl`. Check campaign state with
`--status`. Resume checkpoints are written every five epochs and are reused
when launching with `--skip_existing` through the campaign launcher.

For result merging, checkpoint replay, feasibility, MMD, and rankings, follow
the commands in [`Reproducibility.md`](Reproducibility.md). All MMD and
g-score calculations use the multigraph descriptor and shared reference ruler.
