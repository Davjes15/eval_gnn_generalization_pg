# Generalization score and multigraph MMD

The generalization score summarizes prediction error relative to a
distributional distance between training and test grids. For each model, the
pipeline fits the error-versus-distance relation on the available grid
transfers; the slope and intercept are reported as diagnostics, not as a
standalone guarantee of performance on arbitrary networks.

## Graph representation

The MMD descriptor counts each in-service circuit, so parallel lines remain
distinct. This is important because removing one of two parallel circuits
changes the electrical topology even when the bus-pair connectivity is
unchanged. The default graph mode is `multi`; the published results use the
shared reference bandwidth ruler in `configs/mmd_bandwidth.json`.

The report can be regenerated with:

```bash
python mmd_report.py --data_dir data_regime_b_contingencies_v3 \
  --out results_v3/mmd_report --mmd_graph multi --mmd_bandwidth reference
```

## Published outputs

The per-grid descriptor distances are in
`results_v3/mmd_report/mmd_data_regime_b_contingencies_v3_multigraph_reference.csv`.
The topology matrices and pooled distances used downstream are in
`results_v3/topology/mmd_degree.csv`,
`results_v3/topology/mmd_laplacian.csv`, and
`results_v3/topology/ood_distance.csv`. The seed-level scores are
`results_v3/analysis/gscore.csv`,
`results_v3/analysis/gscore_cc_aggregate.csv`, and
`results_v3/analysis/gscore_ood.csv`; summary results are in
[`Results.md`](Results.md).

The shared ruler fixes the descriptor scale across comparisons. A score must
therefore be interpreted together with its finite-row count, error mean and
standard deviation. Those counts are retained in the score CSVs; non-finite
transfers are not silently converted into successful results.
