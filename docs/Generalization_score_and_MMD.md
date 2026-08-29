# The g-score and the grid-distance it is built on

Audit items **A6** (the g-score is an affine function of mean and spread) and **A7**
(what the MMD estimator and descriptors actually measure). Both are stated here as
protocol-level observations, because both change how the numbers in
`docs/Regime_comparison_results.md` should be read.

Nothing in this document required retraining. The code it describes is
`training_utils.get_generalization_score`, `mmd_utils.py` and `mmd_report.py`; the
table it reports is `docs/tables/mmd_data_full_v2.csv`.

---

## 1. A6 -- with one dataset, the MMD term cannot reorder architectures

ENGAGE's generalization score, as implemented:

```python
score = mean_nrmse + alpha * std_nrmse * log(mmd_range + 1) / mmd_range
```

where `mmd_range` is the spread of the topological distances across the test grids
of an arm, and `mean_nrmse` / `std_nrmse` are that model's mean and standard
deviation of error over the same grids (after trimming the outer 2% of errors).
The distances are Laplacian MMDs under the **biased** estimator throughout, that
being what every committed `mmd_laplacian.csv` / `ood_distance.csv` holds (§2.1).

The decisive point is that **`mmd_range` is a property of the data, not of the
model.** Every architecture in an arm is evaluated on the same set of grids, so
every architecture gets the same distance spread. In our runs it is a single
constant repeated down the column:

Raw-unit campaign, `results/analysis/` (biased Laplacian MMD on `data_full_v2`):

| arm | `mmd_range` | `log(Δ+1)/Δ` | so the score is |
|---|---|---|---|
| cross-context (aggregated over train grids) | 0.516485 | 0.80621 | `g = μ + 0.806 σ` |
| leave-one-grid-out (OOD) | 0.352837 | 0.85651 | `g = μ + 0.857 σ` |
| cross-context, per train grid (`gscore.csv`) | 0.0 | -- (`Δ = 0`) | `g = μ` exactly |

The normalized campaign reported in
[`Normalization_results.md`](Normalization_results.md) §4.4 has slightly
different distances, hence slightly different constants; the structure of the
argument is identical:

| arm | `mmd_range` | `log(Δ+1)/Δ` | so the score is |
|---|---|---|---|
| cross-context (`results_norm/analysis/gscore_cc_aggregate.csv`) | 0.521637 | 0.80475 | `g = μ + 0.805 σ` |
| leave-one-grid-out (`results_norm/analysis/gscore_ood.csv`) | 0.349052 | 0.85776 | `g = μ + 0.858 σ` |

Verified against the committed artifacts rather than asserted: over all 33
cross-context and 28 OOD rows in `results/analysis/`, the largest deviation between
the stored `g_score` and `μ + cσ` with the constants above is **1.0e-4** and
**3.5e-5** respectively -- i.e. the identity holds to rounding. In the normalized
campaign it holds to **1.0e-7** over the 28 cross-context rows with `Δ > 0` and
**1.3e-7** over the 28 OOD rows (`results_norm/analysis/`; the `dc_pf` rows have
`Δ = 0` and are excluded, see below). Ranking
the six architectures by `g_score` and by `mean_nrmse` gives **Kendall τ = 1.0 in
both arms**: the MMD machinery does not move a single position. (Those six
per-model averages are over 5 seeds each except nnconv, which ran 3 -- limitation
L2 -- so nnconv's mean, spread and rank rest on a 40 % smaller sample than every
other architecture's, here and in every pooled table.)

**Consequence.** With a *fixed* dataset the g-score is a monotone re-expression of
"mean plus about 0.85 standard deviations". It is a legitimate risk-averse summary
-- it prefers a model that is uniformly mediocre over one that is excellent on
three grids and catastrophic on the fourth, which is the right preference for a
security-screening application -- but it carries **no topological information about
the architecture**. The MMD term only becomes informative when comparing
*protocols or dataset designs* (where `Δ` differs between the things being
compared), not when comparing architectures on one benchmark.

ENGAGE never claims otherwise. But the natural reading of "use the g-score to
choose an architecture" is that the distance term is doing work, and here it
provably is not. We report the g-score for comparability with ENGAGE, and report
`μ` and `σ` separately as the primary numbers.

Related and already correct in the results doc: the `dc_pf` rows have
`mmd_range = 0`, so their g-score equals their mean and is not comparable with the
GNN rows on the same column.

---

## 2. A7 -- what the distance measures, stated plainly

### 2.1 The estimator is biased, and is now named as one

`mmd_utils.mmd` averages the three Gram matrices in full, diagonals included, so it
is the **biased V-statistic**, not the unbiased U-statistic. The previous docstring
called it "unbiased-ish", which was wrong. `mmd(..., unbiased=True)` now provides
the U-statistic (within-sample diagonals dropped).

The default remains the biased form, deliberately: every committed result CSV was
produced with it, and silently changing published numbers is worse than naming the
estimator correctly. Concretely, **biased** is what
`results_norm/topology/mmd_degree.csv`, `mmd_laplacian.csv` and
`ood_distance.csv` hold, and therefore what the `mmd_range` and every g-score in
`results_norm/analysis/` and in `results/analysis/` are built from; the only
committed table holding both is `docs/tables/mmd_data_full_v2.csv`, which carries
an `estimator` column with a `biased` and an `unbiased` row for each of the 16
grid pairs and all three descriptors:

* different-grid pairs: the two estimators agree to ~0.005 (IEEE24 -> IEEE39
  degree MMD **1.0579 biased vs 1.0558 unbiased**; UK -> IEEE24 degree 0.2763 vs
  0.2762; the mean over different-grid pairs is 0.899 / 0.960 / 1.014 biased and
  0.897 / 0.958 / 1.011 unbiased for degree / Laplacian / reactance). No
  cross-grid statement in either document depends on the choice.
* same-grid pairs: the bias is most or all of the value, i.e. the diagonal is an
  **estimator artefact**. IEEE39 -> IEEE39 degree is 0.0713 biased and
  **0.0000** unbiased; IEEE24 -> IEEE24 is 0.0856 biased against 0.0387 unbiased
  on degree and 0.0671 against **0.0000** on the Laplacian. Averaged over the
  four same-grid pairs the drop is 0.061 -> 0.010 (degree), 0.055 -> 0.0004
  (Laplacian) and 0.067 -> 0.010 (reactance). What is left under the unbiased
  estimator is a train-split vs test-split difference within one grid (the two
  splits carry different N-k topologies), not a distance between grids; the exact
  0.0 cells are the U-statistic going negative, since `mmd_utils.mmd` returns
  `sqrt(max(mmd2, 0))`, i.e. the two splits are not distinguishable at this
  sample size.

So the bias never affects a cross-grid comparison, but the *within*-grid MMD floor
reported anywhere is an artifact of the estimator, not a real distance.

### 2.2 The bandwidth is refit per pair

The Gaussian bandwidth is chosen by the median heuristic **on each pair
separately**. That keeps every pair well-scaled (and fixed the earlier degenerate
`sqrt(2)` saturation, D9), but it means two cells of an MMD matrix are computed
under different kernels. The matrix is a table of pairwise distances, not a set of
values on a single common scale, and differences between cells should be read as
ordinal.

### 2.3 The topological descriptors are blind to the shift that dominates our transfer error

The degree histogram and the normalised-Laplacian spectral histogram are functions
of connectivity only. They are invariant to how much power a system carries -- and
our four cases differ by ~20x in nominal load (2,850 / 6,254 / 3,733 / 56,326 MW,
see `docs/Normalization_assessment.md`). A distance that cannot see the dominant
distribution shift is a weak covariate for a generalization score.

We therefore added an **electrical descriptor**: a density histogram of
`log10(x_pu)` over the in-service branches (`mmd_utils.reactance_histogram`). It is
reported alongside the topological ones, not instead of them, and
`tests/test_mmd_utils.py` pins the complementarity: a 100x impedance change with
identical topology gives degree/Laplacian MMD of 0.0 and reactance MMD of 1.12,
while a ring-vs-star change at identical impedance gives the reverse.

On the real data (`docs/tables/mmd_data_full_v2.csv`, `data_full_v2`, train split
vs test split), with the **biased** estimator that every result CSV uses and the
**unbiased** one alongside it:

| estimator | | degree | Laplacian | reactance |
|---|---|---:|---:|---:|
| biased | mean over same-grid pairs | 0.061 | 0.055 | 0.067 |
| biased | mean over different-grid pairs | 0.899 | 0.960 | 1.014 |
| unbiased | mean over same-grid pairs | 0.010 | 0.0004 | 0.010 |
| unbiased | mean over different-grid pairs | 0.897 | 0.958 | 1.011 |

The gap between the diagonal and the off-diagonal is what makes the arm labels
meaningful, and it is present under all three descriptors -- our "unseen grid" is
genuinely far from the training grids however the distance is measured.

The interesting disagreement is per pair. UK -> IEEE24 has a degree MMD of only
**0.276** (biased; 0.276 unbiased) -- the lowest off-diagonal value in the matrix,
because the dense UK network has a degree profile not unlike the small IEEE24 --
while its reactance MMD is **0.833** (biased; 0.828 unbiased).
A purely topological reading would call that pair "nearly
in-distribution"; electrically it is not. This is the concrete reason the g-score's
distance axis should not be treated as a measure of how hard a transfer is.

---

## 3. Reproduce

```bash
# grid-distance tables under all three descriptors and both estimators
python mmd_report.py --data_dir data_full_v2 --out docs/tables

# the A6 identity and the rank equivalence, from the committed artifacts
python - <<'PY'
import pandas as pd, numpy as np
from scipy.stats import kendalltau
for f, c in (("results/analysis/gscore_cc_aggregate.csv", 0.80621),
             ("results/analysis/gscore_ood.csv", 0.85651)):
    df = pd.read_csv(f)
    df = df[df.model != "dc_pf"]
    print(f, "max |g - (mu + c*sigma)| =",
          float((df.g_score - (df.mean_nrmse + c * df.std_nrmse)).abs().max()))
    m = df.groupby("model")[["mean_nrmse", "g_score"]].mean()
    print("  tau(rank by mean, rank by g) =",
          kendalltau(m.mean_nrmse.rank(), m.g_score.rank()).statistic)
PY

python tests/test_mmd_utils.py
```
