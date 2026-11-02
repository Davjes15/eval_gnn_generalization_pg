# Normalization assessment

The model predicts four physical quantities with different units and scales:
active power, reactive power, voltage magnitude, and voltage angle. A shared
loss on raw values would let the largest-magnitude target dominate training.
The published campaign therefore uses `pu_zscore`: power quantities are
represented consistently in per unit, and target scaling statistics are fit
on training data only. Validation and test splits are transformed with those
training statistics; they do not influence the scaler.

The implementation is in `normalization.py`. It retains physical-unit targets
for reporting and applies the inverse transform before computing evaluation
metrics. The v3 aggregate and per-quantity results are in
`results_v3/analysis/per_quantity.csv`; the model-to-DC comparison is in
`results_v3/analysis/dc_comparison.csv`.

This choice separates optimization scale from scientific reporting units. It
does not remove the need to inspect each target separately: the per-quantity
tables retain P, Q, V, and theta rather than collapsing them into one score.
