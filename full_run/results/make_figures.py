"""Generate ENGAGE-style figures from the full-run results.

Styling: minimal, light-background, pastel palette inspired by clean editorial
line charts -- off-white canvas, horizontal-only gridlines, no top/right spines,
boxed value labels, bold markers.
Results directory (where the *.csv live) is resolved in this order:
    1. first command-line argument  ->  python make_figures.py C:\\path\\to\\results
    2. RESULTS_DIR environment variable
    3. the folder this script lives in (the committed full_run/results copy)
Figures are written to <results_dir>/figures.
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

RES = (sys.argv[1] if len(sys.argv) > 1 else
       os.environ.get("RESULTS_DIR") or os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

# ----------------------------------------------------------------------------
# Global pastel / minimal style
# ----------------------------------------------------------------------------
BG = "#f6f4ef"        # off-white canvas
INK = "#2b2b2b"       # near-black text / spines
GRID = "#d7d3cb"      # soft gridline
BOX = dict(boxstyle="square,pad=0.35", fc="white", ec=INK, lw=1.0)

# Muted pastel palette (one per model / series)
PASTEL = ["#8fbfb0", "#efb48c", "#e69a9a", "#a9a4d6", "#e9cb86", "#8fb8de"]
# Soft sequential ramp for heatmaps (white -> pastel teal)
PASTEL_SEQ = LinearSegmentedColormap.from_list(
    "pastel_seq", ["#f6f4ef", "#cfe3db", "#9fc7ba", "#6fa896", "#417c6b"]
)

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "text.color": INK,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})


def style(ax, ygrid=True):
    """Apply the minimal look to an axis: no top/right spines, soft y-grid."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(1.1)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def boxed(ax, x, y, text, dy=0, fs=10, ha="center", va="bottom"):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(0, dy),
                ha=ha, va=va, fontsize=fs, bbox=BOX, zorder=6)


cc = pd.read_csv(f"{RES}/cross_context.csv")
ood = pd.read_csv(f"{RES}/ood.csv")
lap = pd.read_csv(f"{RES}/mmd_laplacian.csv", index_col=0)
deg = pd.read_csv(f"{RES}/mmd_degree.csv", index_col=0)
oodist = pd.read_csv(f"{RES}/ood_distance.csv").set_index("held_out_grid")
gs_ood = pd.read_csv(f"{RES}/gscore_ood.csv")
grids = list(lap.index)
models = list(cc.model.unique())
COL = {m: PASTEL[i % len(PASTEL)] for i, m in enumerate(models)}


def offdiag_pairs(dfmat):
    """All directed off-diagonal MMD values (kept for the correlation stats)."""
    vals = []
    for i in grids:
        for j in grids:
            if i != j:
                vals.append(dfmat.loc[i, j])
    return np.array(vals)


def directed_pairs(dfmat):
    """The 12 directed off-diagonal grid pairs as (label, value). Entry [i,j] is
    MMD(train-split of i, test-split of j) -- directed, hence 'i->j'."""
    out = []
    for i in grids:
        for j in grids:
            if i != j:
                out.append((f"{i}\u2192{j}", dfmat.loc[i, j]))
    return out


# ----------------------------------------------------------------------------
# 1. MMD range plot (12 directed train->test grid pairs, sorted, named on x-axis)
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, sharex=False, figsize=(10.5, 8.4))
for ax, mat, name, c in [(axes[0], deg, "Degree MMD", PASTEL[0]),
                         (axes[1], lap, "Laplacian MMD", PASTEL[5])]:
    pairs = sorted(directed_pairs(mat), key=lambda t: t[1])
    labels = [p[0] for p in pairs]
    v = np.array([p[1] for p in pairs])
    xs = range(len(v))
    ax.plot(xs, v, color=INK, lw=1.6, zorder=2)
    ax.scatter(xs, v, s=95, color=c, edgecolor=INK, lw=1.2, zorder=3)
    boxed(ax, 0, v[0], f"{v[0]:.2f}", dy=12)
    boxed(ax, len(v) - 1, v[-1], f"{v[-1]:.2f}", dy=12)
    ax.set_ylabel(name)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.margins(x=0.04, y=0.20)
    style(ax)
axes[0].set_title("MMD across the 12 directed cross-grid pairs (train\u2192test, sorted)",
                  loc="left")
axes[1].set_xlabel("cross-grid pair (train\u2192test)")
fig.tight_layout()
fig.savefig(f"{FIG}/fig_mmd_range.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 2. MMD heatmaps (degree + laplacian)
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
for ax, mat, name in [(axes[0], deg, "Degree MMD"), (axes[1], lap, "Laplacian MMD")]:
    im = ax.imshow(mat.values, cmap=PASTEL_SEQ)
    ax.set_xticks(range(len(grids)))
    ax.set_xticklabels(grids, rotation=45, ha="right")
    ax.set_yticks(range(len(grids)))
    ax.set_yticklabels(grids)
    thr = mat.values.max() * 0.55
    for i in range(len(grids)):
        for j in range(len(grids)):
            ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat.values[i, j] > thr else INK, fontsize=9)
    ax.set_title(name, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
fig.suptitle("Grid-to-grid MMD  (rows = train, cols = test)", x=0.02, ha="left",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{FIG}/fig_mmd_heatmap.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 3. Performance analysis: mean NRMSE per model (within / CC / OOD) + OOD boxplot
# ----------------------------------------------------------------------------
diag = cc[cc.train_grid == cc.test_grid]
offc = cc[cc.train_grid != cc.test_grid]
within = diag.groupby("model")["nrmse"].mean().reindex(models)
cctrans = offc.groupby("model")["nrmse"].mean().reindex(models)
oodm = ood.groupby("model")["nrmse"].mean().reindex(models)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
x = np.arange(len(models))
w = 0.27
bars = [
    (within.values, -w, PASTEL[0], "within-grid (diag)"),
    (cctrans.values, 0.0, PASTEL[2], "CC single-grid transfer"),
    (oodm.values, w, PASTEL[5], "OOD (train on 3)"),
]
for vals, off, c, lab in bars:
    axes[0].bar(x + off, vals, w, label=lab, color=c, edgecolor=INK, lw=1.0, zorder=3)
axes[0].set_yscale("log")
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=30, ha="right")
axes[0].set_ylabel("mean NRMSE (log scale)")
axes[0].set_title("Mean NRMSE by model and regime", loc="left")
axes[0].legend(frameon=False)
style(axes[0])
# OOD distribution boxplot per model (pastel-filled boxes)
data = [ood[ood.model == m]["nrmse"].dropna().values for m in models]
bp = axes[1].boxplot(data, tick_labels=models, showfliers=True, patch_artist=True,
                     medianprops=dict(color=INK, lw=1.4),
                     whiskerprops=dict(color=INK), capprops=dict(color=INK),
                     flierprops=dict(marker="o", mfc=PASTEL[2], mec=INK, ms=5))
for patch, m in zip(bp["boxes"], models):
    patch.set_facecolor(COL[m])
    patch.set_edgecolor(INK)
axes[1].set_yscale("log")
axes[1].set_xticklabels(models, rotation=30, ha="right")
axes[1].set_ylabel("OOD NRMSE per held-out grid (log)")
axes[1].set_title("OOD NRMSE distribution by model", loc="left")
style(axes[1])
fig.tight_layout()
fig.savefig(f"{FIG}/fig_performance.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 4. Generalizability curve: MMD vs NRMSE (CC off-diagonal, and OOD)
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
# CC
xs, ys = [], []
for m in models:
    sub = offc[offc.model == m]
    px = [lap.loc[r.train_grid, r.test_grid] for _, r in sub.iterrows()]
    py = list(sub.nrmse.values)
    axes[0].scatter(px, py, s=70, color=COL[m], edgecolor=INK, lw=0.9, label=m, zorder=3)
    xs += px
    ys += py
xs, ys = np.array(xs), np.array(ys)
pr = pearsonr(xs, ys)[0]
sr = spearmanr(xs, ys)[0]
axes[0].set_yscale("log")
axes[0].set_xlabel("MMD (Laplacian) train->test")
axes[0].set_ylabel("NRMSE (log)")
axes[0].set_title("CC generalizability curve", loc="left")
style(axes[0])
# OOD
xs2, ys2 = [], []
for m in models:
    sub = ood[ood.model == m].dropna(subset=["nrmse"])
    px = [oodist.loc[r.held_out_grid, "mmd_to_train_mean"] for _, r in sub.iterrows()]
    py = list(sub.nrmse.values)
    axes[1].scatter(px, py, s=90, color=COL[m], edgecolor=INK, lw=0.9, label=m, zorder=3)
    xs2 += px
    ys2 += py
xs2, ys2 = np.array(xs2), np.array(ys2)
pr2 = pearsonr(xs2, ys2)[0]
sr2 = spearmanr(xs2, ys2)[0]
axes[1].set_yscale("log")
axes[1].set_xlabel("MMD (Laplacian) held-out->training grids (mean)")
axes[1].set_ylabel("NRMSE (log)")
axes[1].set_title("OOD generalizability curve", loc="left")
style(axes[1])
# correlation annotation in axes-fraction coords (bottom-right, clear of points)
for ax, txt in [(axes[0], f"Pearson={pr:.2f}   Spearman={sr:.2f}"),
                (axes[1], f"Pearson={pr2:.2f}   Spearman={sr2:.2f}")]:
    ax.text(0.97, 0.04, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, bbox=BOX, zorder=6)
# single shared legend above the panels (out of the data area)
handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="upper center", ncol=len(models), frameon=False,
           bbox_to_anchor=(0.5, 1.02), fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{FIG}/fig_generalizability_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------
# 5. OOD g-score bar chart
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
g = gs_ood.set_index("model").reindex([m for m in models if m in gs_ood.model.values])
ax.bar(g.index, g.g_score.values, color=[COL[m] for m in g.index],
       edgecolor=INK, lw=1.0, zorder=3)
for i, (m, r) in enumerate(g.iterrows()):
    boxed(ax, i, r.g_score, f"{r.g_score:.2f}", dy=8)
ax.set_ylabel("OOD g-score (lower = better)")
ax.set_title("OOD g-score by model  (no trim; NaN cells dropped)", loc="left")
ax.set_xticks(range(len(g.index)))
ax.set_xticklabels(g.index, rotation=30, ha="right")
ax.margins(y=0.15)
style(ax)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_gscore_ood.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 6. Cross-context transfer-matrix heatmaps (one 4x4 train x test per model)
# ----------------------------------------------------------------------------
mats = {m: pd.read_csv(f"{RES}/transfer_matrix_{m}.csv", index_col=0).reindex(
            index=grids, columns=grids) for m in models}
allvals = np.concatenate([m.values[np.isfinite(m.values)] for m in mats.values()])
vmin, vmax = max(allvals.min(), 1e-3), allvals.max()
fig, axes = plt.subplots(2, 3, figsize=(15, 9.6),
                         gridspec_kw={"hspace": 0.55, "wspace": 0.35})
for ax, m in zip(axes.ravel(), models):
    M = mats[m].values
    im = ax.imshow(M, cmap="RdPu", norm=LogNorm(vmin=vmin, vmax=vmax))
    ax.set_xticks(range(len(grids)))
    ax.set_xticklabels(grids, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(grids)))
    ax.set_yticklabels(grids, fontsize=9)
    for i in range(len(grids)):
        for j in range(len(grids)):
            val = M[i, j]
            txt = "NaN" if not np.isfinite(val) else (f"{val:.2f}" if val < 10 else f"{val:.0f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if np.isfinite(val) and val > vmax * 0.15 else INK)
    ax.set_title(m, loc="left")
    ax.set_ylabel("train grid", fontsize=9)
    ax.set_xlabel("test grid", fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
fig.suptitle("Cross-context transfer NRMSE  (diagonal = within-grid; off-diagonal = "
             "unseen grid; log color)", x=0.02, ha="left", fontsize=14, fontweight="bold")
cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
cb.outline.set_visible(False)
cb.set_label("NRMSE (log)")
fig.savefig(f"{FIG}/fig_transfer_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------
# 7. Per-quantity within-grid NRMSE (P, Q, V, theta) per model
# ----------------------------------------------------------------------------
quant = ["nrmse_P", "nrmse_Q", "nrmse_V", "nrmse_theta"]
qlabel = ["P", "Q", "V", r"$\theta$"]
qcol = [PASTEL[0], PASTEL[1], PASTEL[2], PASTEL[5]]
wq = diag.groupby("model")[quant].mean().reindex(models)
fig, ax = plt.subplots(figsize=(11, 5.4))
x = np.arange(len(models))
w = 0.2
for k, (q, lab, c) in enumerate(zip(quant, qlabel, qcol)):
    ax.bar(x + (k - 1.5) * w, wq[q].values, w, label=lab, color=c,
           edgecolor=INK, lw=0.9, zorder=3)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha="right")
ax.set_ylabel("within-grid NRMSE (log)")
ax.set_title("Per-quantity within-grid NRMSE  (V is inflated by its tiny range, "
             "not by poor accuracy)", loc="left")
ax.legend(title="quantity", frameon=False, ncol=4, loc="upper center",
          bbox_to_anchor=(0.5, 1.0))
ax.margins(y=0.18)
style(ax)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_per_quantity.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 8. GNN (best within-grid) vs DC-PF baseline, per quantity, per grid
# ----------------------------------------------------------------------------
dc = pd.read_csv(f"{RES}/dc_baseline.csv").set_index("grid").reindex(grids)
# best GNN per grid/quantity = min within-grid per-quantity NRMSE across models
gnn_best = diag.groupby("test_grid")[quant].min().reindex(grids)
fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), sharey=False)
dc_cols = ["dc_nrmse_P", "dc_nrmse_Q", "dc_nrmse_V", "dc_nrmse_theta"]
for ax, q, dcq, lab in zip(axes, quant, dc_cols, qlabel):
    xg = np.arange(len(grids))
    ax.bar(xg - 0.2, gnn_best[q].values, 0.4, label="best GNN", color=PASTEL[0],
           edgecolor=INK, lw=0.9, zorder=3)
    ax.bar(xg + 0.2, dc[dcq].values, 0.4, label="DC-PF", color=PASTEL[2],
           edgecolor=INK, lw=0.9, zorder=3)
    ax.set_yscale("log")
    if (dc[dcq].values == 0).all():
        ax.text(0.5, 0.02, "DC-PF ≡ 0  (not shown on log axis)",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=9,
        style="italic", color=INK)
    ax.set_xticks(xg)
    ax.set_xticklabels(grids, rotation=45, ha="right", fontsize=9)
    ax.set_title(lab, loc="left")
    style(ax)
axes[0].set_ylabel("NRMSE (log)")
axes[0].legend(frameon=False, loc="upper left", fontsize=9)
fig.suptitle("Best within-grid GNN vs DC-PF baseline, per quantity  "
             "(DC Q-NRMSE=0 is a bookkeeping artifact, not a solved reactive result)",
             x=0.02, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{FIG}/fig_gnn_vs_dc.png", dpi=150)
plt.close(fig)

print("figures written to", FIG)
print("CC corr: pearson=%.3f spearman=%.3f | OOD corr: pearson=%.3f spearman=%.3f"
      % (pr, sr, pr2, sr2))
for f in sorted(os.listdir(FIG)):
    print(" ", f)
