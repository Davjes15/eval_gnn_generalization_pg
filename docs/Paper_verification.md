# Paper and implementation traceability

This note identifies the sources used to describe the two benchmark pipelines;
it does not substitute their original papers or code releases.

## PowerGraph

PowerGraph-Node supplies the hourly demand data and bus-level load conventions.
The loader and conversion path are documented in
[`PowerGraph-Node_deep_dive.md`](PowerGraph-Node_deep_dive.md). The repository
uses its four transmission systems (IEEE24, IEEE39, IEEE118, and UK) and records
the exact generated sample metadata in each v3 split's `dataset_src.csv`.

## ENGAGE and generalization metrics

ENGAGE provides the graph-level generalization-score formulation and MMD
implementation lineage. This repository adapts it to node-level transmission
power-flow prediction and uses the multigraph descriptor with a shared
reference bandwidth. The implemented calculation and committed outputs are
described in [`Generalization_score_and_MMD.md`](Generalization_score_and_MMD.md)
and [`Results.md`](Results.md).

## Reproducible evidence

The model configurations are frozen in `configs/arch_config.json`; the
published campaign protocol and exact replay commands are in
[`Reproducibility.md`](Reproducibility.md). Numerical claims in this repository
should be traced to the named CSVs under `results_v3/` rather than copied from
paper figures or earlier experiments.
