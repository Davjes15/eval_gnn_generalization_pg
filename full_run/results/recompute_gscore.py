"""Post-hoc small-N g-score (no percentile trim).

The default get_generalization_score(bounds=2) trims the 2nd/98th NRMSE
percentiles; with only 3 unseen grids per training grid that keeps a single
point, forcing std_nrmse=0 and mmd_range=0. With only 4 grids the well-posed
choice is bounds=0 (use all unseen grids), as the design docs note. This
recomputes the g-score table over ALL unseen grids from the saved CSVs.
"""
import numpy as np
import pandas as pd

RES = "/home/ubuntu/full_run/results"
cc = pd.read_csv(f"{RES}/cross_context.csv")
lap = pd.read_csv(f"{RES}/mmd_laplacian.csv", index_col=0)

ALPHA = 1.0
EPS = 1e-8
rows = []
for name in cc.model.unique():
    for train_grid in lap.index:
        sub = cc[(cc.model == name) & (cc.train_grid == train_grid) & (cc.unseen)]
        if sub.empty:
            continue
        nrmses = sub["nrmse"].to_numpy(float)
        mmds = np.array([lap.loc[train_grid, tg] for tg in sub["test_grid"]], float)
        mean_n = float(np.nanmean(nrmses))
        std_n = float(np.nanstd(nrmses))
        mmd_rng = float(mmds.max() - mmds.min())
        score = mean_n + ALPHA * std_n * (np.log(mmd_rng + 1) / (mmd_rng + EPS))
        rows.append({"model": name, "train_grid": train_grid,
                     "n_unseen": int(np.isfinite(nrmses).sum()),
                     "mean_nrmse": mean_n, "std_nrmse": std_n,
                     "mmd_range": mmd_rng, "g_score": float(score)})

out = pd.DataFrame(rows)
out.to_csv(f"{RES}/gscore_smallN.csv", index=False)
pd.set_option("display.width", 140)
print(out.to_string(index=False))
