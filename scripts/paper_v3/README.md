# Paper appendix analyses (v3.0.0)

These scripts produce the camera-ready appendix numbers that are not in the main campaign outputs. They only run predictions with the v3.0.0 release checkpoints. Nothing is trained.

Setup: unpack the v3.0.0 release checkpoints into `ckpt_v3/` and set `POWERGRAPH_NODE_DIR` to the `PowerGraph-Node/13_Power_system` folder (see `docs/Reproducibility.md`). Run from the repo root.

| Script | Output in `results_v3/analysis/paper_v3/` | Paper |
|---|---|---|
| `step3_analyses.py --models <models>` | `overlap_<model>.csv`, `severity_<model>.csv` | Regime B outage overlap, overload severity |
| `step3_analyses.py --dc --models none` | `severity_dc.csv` | DC and label-floor severity rows |
| `aggregate.py` | `overlap_summary.csv`, `severity_summary.csv` | Summaries with the void rule. Also checks the replay against `results_v3/physics/physics_metrics.csv` |
| `worked_example.py IEEE39 1000` | `worked_example_IEEE39_arma_s1000.json` | IEEE39 worked example |
| `mismatch_calibration.py gcn IEEE39 0` and `gcn IEEE118 0` | `mismatch_calibration.csv` | Mismatch calibration table |

The void rule matches the main tables. A (arm, model, training grid, seed) group with any non-finite prediction is dropped. In v3 this is only cross / GCN / IEEE39 / seed 1000.

Outage overlap uses the Regime B diagonal (train and test on the same grid). An outage set counts as seen when the same set of lines occurs in that grid's training split. Severity bins are 100–110, 110–150 and above 150% of rating for missed overloads, and below 90 and 90–100% for false alarms.
