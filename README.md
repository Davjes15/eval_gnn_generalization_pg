# Transmission-grid GNN generalization

Reproducible experiments for comparing six graph neural network architectures
on power-flow prediction across four transmission grids. The committed v3 data
corrects the UK and IEEE118 topology; the published campaign outputs are in
`results_v3/`.

## Pipeline

```text
PowerGraph demand + committed grid cases
            │
            ▼
transmission_graph_gen.py ──► data_regime_*_v3/
            │
            ▼
launch_campaign.py ─────────► results_v3/ + ckpt_v3/
            │
            ▼
gather_results.py ─► MMD / checkpoint replay / feasibility / ranking tables
```

The model zoo is ARMA-GNN, NNConv, Transformer, GIN, GAT, and GCN. A shared
node-level encoder, skip connection, decoder, and four-output readout surround
the message-passing layer; only that layer changes between architectures.
Predictions cover active power, reactive power, voltage magnitude, and voltage
angle. The same frozen architecture configuration is used across arms.

The three released datasets are fixed-topology Regime A
(`data_regime_a_fixed_topology_v3/`), blocked fixed-topology Regime A
(`data_regime_a_blocked_v3/`), and blocked contingency Regime B
(`data_regime_b_contingencies_v3/`). Each dataset directory includes its
`GENERATION_REPORT.md` and per-split `dataset_src.csv`.

## Repository layout

```text
configs/                         frozen architecture config and shared MMD ruler
data_regime_a_fixed_topology_v3/ fixed-topology dataset and generation report
data_regime_a_blocked_v3/        blocked fixed-topology dataset and report
data_regime_b_contingencies_v3/  contingency dataset and generation report
docs/                            reproducibility, design, and results documentation
results_v3/                      merged campaign outputs and analysis tables
tests/                            script-style regression tests
transmission/                     committed grid cases and conversion instructions
*.py                              generation, training, evaluation, and analysis tools
```

See [the v3 results](docs/Results.md), [the pipeline report](docs/Pipeline_Report.md),
and [reproduction instructions](docs/Reproducibility.md).

## Checkpoint downloads

The v3.0.0 release provides one archive per arm:

| Arm | Download | SHA-256 |
|---|---|---|
| Within | [ckpt_v3_within.tar.gz](https://github.com/Davjes15/eval_gnn_generalization_pg/releases/download/v3.0.0/ckpt_v3_within.tar.gz) | `a98370c1569f4bc817a526845a2f2f051a9e6f884f5a86026ba96fdd6dc16845` |
| Cross-context | [ckpt_v3_cross.tar.gz](https://github.com/Davjes15/eval_gnn_generalization_pg/releases/download/v3.0.0/ckpt_v3_cross.tar.gz) | `882b3d93cda065385a69bd2aa863dd37bb75f5eba53efc456b09892fc47c97e7` |
| OOD | [ckpt_v3_ood.tar.gz](https://github.com/Davjes15/eval_gnn_generalization_pg/releases/download/v3.0.0/ckpt_v3_ood.tar.gz) | `49ffb90775e512b36fe952b4b7075c0b6225dc55a654576949fa057dc0324359` |

Download and verify an archive, for example:

```bash
curl -L -o ckpt_v3_within.tar.gz \
  https://github.com/Davjes15/eval_gnn_generalization_pg/releases/download/v3.0.0/ckpt_v3_within.tar.gz
echo "a98370c1569f4bc817a526845a2f2f051a9e6f884f5a86026ba96fdd6dc16845  ckpt_v3_within.tar.gz" | sha256sum -c
tar -xzf ckpt_v3_within.tar.gz
```

## Reproduction at a glance

Use Python 3.10 and install `requirements.txt`. Set `POWERGRAPH_NODE_DIR` to
the PowerGraph-Node `13_Power_system` directory. Validate the committed grids
with `python transmission_grids.py`, then use the saved v3 datasets and
`configs/arch_config.json` to train or replay checkpoints. The published
protocol uses 200 epochs, `pu_zscore`, seeds `0 100 300 700 1000` (NNConv:
`0 100 300`), within, cross-context, and OOD arms, and the shared multigraph
MMD ruler in `configs/mmd_bandwidth.json`. Full commands are in
[docs/Reproducibility.md](docs/Reproducibility.md).

## History

v3 = corrected topology (UK was missing 9/99 branches, IEEE118 2/186 and had edges on wrong buses). Earlier runs (v1 full_run, v2 data_a/data_full_v2 → results/, results_norm/), their docs and the audit are in branch `archive/full-history` and tag `archive-pre-v3`; release v1.0.0 has the pre-fix checkpoints.
