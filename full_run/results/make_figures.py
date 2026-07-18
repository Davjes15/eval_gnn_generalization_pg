"""Generate ENGAGE-style figures from the full-run results."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

RES = "/home/ubuntu/full_run/results"
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

cc = pd.read_csv(f"{RES}/cross_context.csv")
ood = pd.read_csv(f"{RES}/ood.csv")
lap = pd.read_csv(f"{RES}/mmd_laplacian.csv", index_col=0)
deg = pd.read_csv(f"{RES}/mmd_degree.csv", index_col=0)
oodist = pd.read_csv(f"{RES}/ood_distance.csv").set_index("held_out_grid")
gs_ood = pd.read_csv(f"{RES}/gscore_ood.csv")
grids = list(lap.index)
models = list(cc.model.unique())
CMAP = plt.cm.tab10(np.linspace(0, 1, len(models)))
COL = {m: CMAP[i] for i, m in enumerate(models)}


def offdiag_pairs(dfmat):
    vals = []
    for i in grids:
        for j in grids:
            if i != j:
                vals.append(dfmat.loc[i, j])
    return np.array(vals)


# 1. MMD range plot (sorted MMDs across the 12 off-diagonal grid pairs)
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
for ax, mat, name in [(axes[0], deg, "mmd_degree"), (axes[1], lap, "mmd_laplacian")]:
    v = np.sort(offdiag_pairs(mat))
    ax.scatter(range(len(v)), v, label=name, color="tab:blue")
    ax.axhline(offdiag_pairs(mat).min(), ls="--", c="gray", lw=0.8)
    ax.axhline(offdiag_pairs(mat).max(), ls="--", c="gray", lw=0.8)
    ax.set_ylabel(name)
    ax.legend(loc="upper left")
axes[0].set_title("MMD range across the 12 cross-grid pairs (sorted)")
axes[1].set_xlabel("grid-pair id (sorted by MMD)")
fig.tight_layout()
fig.savefig(f"{FIG}/fig_mmd_range.png", dpi=150)
plt.close(fig)

# 2. MMD heatmaps (degree + laplacian)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, mat, name in [(axes[0], deg, "mmd_degree"), (axes[1], lap, "mmd_laplacian")]:
    im = ax.imshow(mat.values, cmap="viridis")
    ax.set_xticks(range(len(grids)))
    ax.set_xticklabels(grids, rotation=45, ha="right")
    ax.set_yticks(range(len(grids)))
    ax.set_yticklabels(grids)
    for i in range(len(grids)):
        for j in range(len(grids)):
            ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center",
                    color="w" if mat.values[i, j] < mat.values.max() * 0.6 else "k",
                    fontsize=8)
    ax.set_title(name)
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Grid-to-grid MMD (rows=train, cols=test)")
fig.tight_layout()
fig.savefig(f"{FIG}/fig_mmd_heatmap.png", dpi=150)
plt.close(fig)

# 3. Performance analysis: mean NRMSE per model (within / CC transfer / OOD) + OOD boxplot
diag = cc[cc.train_grid == cc.test_grid]
offc = cc[cc.train_grid != cc.test_grid]
within = diag.groupby("model")["nrmse"].mean().reindex(models)
cctrans = offc.groupby("model")["nrmse"].mean().reindex(models)
oodm = ood.groupby("model")["nrmse"].mean().reindex(models)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(models))
w = 0.27
axes[0].bar(x - w, within.values, w, label="within-grid (diag)")
axes[0].bar(x, cctrans.values, w, label="CC single-grid transfer (off-diag)")
axes[0].bar(x + w, oodm.values, w, label="OOD (train on 3)")
axes[0].set_yscale("log")
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=30, ha="right")
axes[0].set_ylabel("mean NRMSE (log scale)")
axes[0].set_title("Mean NRMSE by model and regime")
axes[0].legend()
# OOD distribution boxplot per model
data = [ood[ood.model == m]["nrmse"].dropna().values for m in models]
axes[1].boxplot(data, labels=models, showfliers=True)
axes[1].set_yscale("log")
axes[1].set_xticklabels(models, rotation=30, ha="right")
axes[1].set_ylabel("OOD NRMSE per held-out grid (log)")
axes[1].set_title("OOD NRMSE distribution by model")
fig.tight_layout()
fig.savefig(f"{FIG}/fig_performance.png", dpi=150)
plt.close(fig)

# 4. Generalizability curve: MMD vs NRMSE (CC off-diagonal, and OOD)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
# CC
xs, ys = [], []
for m in models:
    sub = offc[offc.model == m]
    px = [lap.loc[r.train_grid, r.test_grid] for _, r in sub.iterrows()]
    py = list(sub.nrmse.values)
    axes[0].scatter(px, py, s=28, color=COL[m], label=m, alpha=0.8)
    xs += px
    ys += py
xs, ys = np.array(xs), np.array(ys)
pr = pearsonr(xs, ys)[0]
sr = spearmanr(xs, ys)[0]
axes[0].set_yscale("log")
axes[0].set_xlabel("MMD (Laplacian) train→test")
axes[0].set_ylabel("NRMSE (log)")
axes[0].set_title(f"CC generalizability curve\nPearson={pr:.2f}  Spearman={sr:.2f}")
axes[0].legend(fontsize=7, ncol=2)
# OOD
xs2, ys2 = [], []
for m in models:
    sub = ood[ood.model == m].dropna(subset=["nrmse"])
    px = [oodist.loc[r.held_out_grid, "mmd_to_train_mean"] for _, r in sub.iterrows()]
    py = list(sub.nrmse.values)
    axes[1].scatter(px, py, s=40, color=COL[m], label=m, alpha=0.85)
    xs2 += px
    ys2 += py
xs2, ys2 = np.array(xs2), np.array(ys2)
pr2 = pearsonr(xs2, ys2)[0]
sr2 = spearmanr(xs2, ys2)[0]
axes[1].set_yscale("log")
axes[1].set_xlabel("MMD (Laplacian) held-out→training grids (mean)")
axes[1].set_ylabel("NRMSE (log)")
axes[1].set_title(f"OOD generalizability curve\nPearson={pr2:.2f}  Spearman={sr2:.2f}")
axes[1].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_generalizability_curve.png", dpi=150)
plt.close(fig)

# 5. OOD g-score bar chart
fig, ax = plt.subplots(figsize=(7, 4.5))
g = gs_ood.set_index("model").reindex([m for m in models if m in gs_ood.model.values])
ax.bar(g.index, g.g_score.values, color=[COL[m] for m in g.index])
for i, (m, r) in enumerate(g.iterrows()):
    ax.text(i, r.g_score, f"{r.g_score:.2f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("OOD g-score (lower = better)")
ax.set_title("OOD g-score by model (no trim; NaN cells dropped)")
ax.set_xticklabels(g.index, rotation=30, ha="right")
fig.tight_layout()
fig.savefig(f"{FIG}/fig_gscore_ood.png", dpi=150)
plt.close(fig)

print("figures written to", FIG)
print("CC corr: pearson=%.3f spearman=%.3f | OOD corr: pearson=%.3f spearman=%.3f" % (pr, sr, pr2, sr2))
for f in sorted(os.listdir(FIG)):
    print(" ", f)
