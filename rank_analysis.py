"""rank_analysis.py -- does the architecture ranking survive generalization?

WHY (design decision D14/D15)
    The headline claim of the study is that ranking six architectures by
    fixed-topology accuracy (Regime A, the PowerGraph-like regime) is not
    sufficient for choosing an architecture, because the ordering changes once
    the models are evaluated on unseen grids (Regime B: cross-context and
    leave-one-grid-out OOD). Every architecture carries ONE frozen configuration
    into both regimes, so a reordering cannot be attributed to hyperparameters.

    Kendall tau-b is the primary statistic: with 6 architectures there are 15
    pairs, and tau-b = (concordant - discordant) / 15 (ties handled by the -b
    correction). Spearman rho is reported as a secondary check.

    The decision rule is fixed IN ADVANCE so that tau ~ +1 is reported as a null
    result rather than reframed:
        tau ~ +1, stable over seeds  -> ranking is regime-invariant; the
                                        ranking claim FAILS (absolute transfer
                                        degradation may still be large).
        tau clearly < +1, stable     -> the ranking changes under generalization.
        tau unstable across seeds    -> architecture selection under
                                        generalization is seed-noise dominated;
                                        report distributions, not one ranking.

HOW IT CONNECTS
    results_a/within_grid.csv          (Regime A, Step 5)
    results_tuned/cross_context.csv    (Regime B CC, Step 6)
    results_tuned/ood.csv              (Regime B OOD, Step 6)
        -> results/rank_correlation.csv   (tau-b / rho per grid, seed, metric)
        -> results/ranking_table.csv      (mean +/- sd error, mean rank, modal rank)
        -> results/bump_chart_<metric>.png

HOW TO RUN
    python3 rank_analysis.py --regime_a results_a/within_grid.csv \
        --cross results_tuned/cross_context.csv --ood results_tuned/ood.csv \
        --out results
"""
from __future__ import annotations

import argparse
import itertools
import os
from collections import Counter

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

METRICS = ["nrmse", "mse", "mae"]
ARMS = ["cross_context", "ood"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regime_a", default="results_a/within_grid.csv")
    p.add_argument("--cross", default="results_tuned/cross_context.csv")
    p.add_argument("--ood", default="results_tuned/ood.csv")
    p.add_argument("--out", default="results")
    p.add_argument("--metrics", nargs="+", default=METRICS)
    p.add_argument("--bump_metric", default="nrmse",
                   help="metric used for the bump chart")
    return p.parse_args()


def load_arms(args):
    """Per-arm long tables of (model, grid, seed, metric...) test errors.

    Each arm is reduced to ONE row per (model, grid, seed) so the three arms are
    directly comparable:
      * Regime A  -- within-grid test error on that grid.
      * cross-context -- error on UNSEEN test grids only, averaged over them,
        attributed to the grid the model was TRAINED on (that is the choice a
        practitioner makes: "I have data for grid X").
      * OOD -- error on the held-out grid, attributed to that held-out grid.
    """
    arms = {}

    a = pd.read_csv(args.regime_a)
    arms["regime_a"] = a.rename(columns={"grid": "grid"})

    cc = pd.read_csv(args.cross)
    cc = cc[cc.unseen] if "unseen" in cc.columns else cc
    value_cols = [c for c in cc.columns
                  if c not in ("model", "train_grid", "test_grid", "unseen",
                               "seed", "regime")
                  and pd.api.types.is_numeric_dtype(cc[c])]
    # A transfer that produced a non-finite error is a failure of that
    # (model, train_grid, seed), not a missing observation: pandas would drop it
    # and average the remaining test grids, which reports a model as better than
    # the run where it broke down. The whole group is voided instead and counted.
    cc = cc.copy()
    cc[value_cols] = cc[value_cols].replace([np.inf, -np.inf], np.nan)
    grouped = cc.groupby(["model", "train_grid", "seed"], as_index=False)
    means = grouped[value_cols].mean()
    voided = grouped[value_cols].agg(lambda s: bool(s.isna().any()))
    means[value_cols] = means[value_cols].mask(
        voided[value_cols].to_numpy().astype(bool))
    arms["cross_context"] = means.rename(columns={"train_grid": "grid"})
    lost = int(voided["nrmse"].sum()) if "nrmse" in voided else 0
    if lost:
        print(f"note: {lost} cross-context (model, train grid, seed) group(s) "
              f"voided by a non-finite transfer error; they are excluded from "
              f"the ranking and listed in the arm table as NaN")

    ood = pd.read_csv(args.ood)
    arms["ood"] = ood.rename(columns={"held_out_grid": "grid"})
    return arms


def ranks(frame, metric):
    """model -> rank (1 = lowest error). Ties get the average rank."""
    sub = frame.dropna(subset=[metric])
    return sub.set_index("model")[metric].rank(method="average")


def correlations(arms, metrics):
    """tau-b and rho between Regime A and each Regime B arm, per grid and seed."""
    rows = []
    a = arms["regime_a"]
    for arm in ARMS:
        b = arms[arm]
        keys = sorted(set(map(tuple, a[["grid", "seed"]].values))
                      & set(map(tuple, b[["grid", "seed"]].values)))
        for grid, seed in keys:
            for metric in metrics:
                if metric not in a.columns or metric not in b.columns:
                    continue
                ra = ranks(a[(a.grid == grid) & (a.seed == seed)], metric)
                rb = ranks(b[(b.grid == grid) & (b.seed == seed)], metric)
                shared = sorted(set(ra.index) & set(rb.index))
                if len(shared) < 3:
                    # tau on fewer than 3 architectures is not interpretable.
                    continue
                x, y = ra[shared].to_numpy(), rb[shared].to_numpy()
                tau = stats.kendalltau(x, y, variant="b")
                rho = stats.spearmanr(x, y)
                rows.append({
                    "comparison": f"regime_a_vs_{arm}", "grid": grid,
                    "seed": seed, "metric": metric, "n_models": len(shared),
                    "kendall_tau_b": tau.statistic, "kendall_p": tau.pvalue,
                    "spearman_rho": rho.statistic, "spearman_p": rho.pvalue,
                })
    return pd.DataFrame(rows)


def correlation_summary(corr):
    """Mean/sd of tau-b over grids and seeds -- the stability check."""
    if corr.empty:
        return corr
    return (corr.groupby(["comparison", "metric"])
            .agg(n=("kendall_tau_b", "size"),
                 tau_mean=("kendall_tau_b", "mean"),
                 tau_sd=("kendall_tau_b", "std"),
                 tau_min=("kendall_tau_b", "min"),
                 tau_max=("kendall_tau_b", "max"),
                 rho_mean=("spearman_rho", "mean"),
                 rho_sd=("spearman_rho", "std"))
            .reset_index())


def ranking_table(arms, metrics):
    """Per-arm error and ranking per architecture, with the modal rank.

    `overlaps_next` flags a pair whose mean +/- sd error intervals overlap, i.e.
    an adjacent ordering that the seed spread does not resolve.
    """
    rows = []
    for arm, frame in arms.items():
        for metric in metrics:
            if metric not in frame.columns:
                continue
            per_seed_ranks = {}
            for (grid, seed), sub in frame.groupby(["grid", "seed"]):
                for model, rank in ranks(sub, metric).items():
                    per_seed_ranks.setdefault(model, []).append(rank)
            agg = (frame.groupby("model")[metric]
                   .agg(["mean", "std", "size"]).reset_index())
            agg = agg.sort_values("mean").reset_index(drop=True)
            for pos, r in agg.iterrows():
                rank_list = per_seed_ranks.get(r.model, [])
                counts = Counter(rank_list)
                nxt = agg.iloc[pos + 1] if pos + 1 < len(agg) else None
                overlaps = bool(
                    nxt is not None
                    and pd.notna(r["std"]) and pd.notna(nxt["std"])
                    and r["mean"] + r["std"] >= nxt["mean"] - nxt["std"])
                rows.append({
                    "arm": arm, "metric": metric, "model": r.model,
                    "rank_overall": pos + 1,
                    "mean": r["mean"], "sd": r["std"], "n": int(r["size"]),
                    "mean_rank": (sum(rank_list) / len(rank_list)
                                  if rank_list else float("nan")),
                    "modal_rank": (counts.most_common(1)[0][0]
                                   if counts else float("nan")),
                    "rank_frequencies": dict(sorted(counts.items())),
                    "overlaps_next": overlaps,
                })
    return pd.DataFrame(rows)


def bump_chart(table, metric, path):
    """Regime A -> cross-context -> OOD rank trajectory per architecture."""
    order = [a for a in ("regime_a", "cross_context", "ood")
             if a in set(table.arm)]
    sub = table[(table.metric == metric) & (table.arm.isin(order))]
    if sub.empty:
        return None
    pivot = sub.pivot_table(index="model", columns="arm",
                            values="rank_overall")[order]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, row in pivot.iterrows():
        ax.plot(range(len(order)), row.to_numpy(), marker="o", label=model)
        ax.annotate(model, (len(order) - 1, row.to_numpy()[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["Regime A\n(fixed topology)", "Cross-context\n(unseen grid)",
                        "OOD\n(leave-one-grid-out)"][:len(order)])
    ax.set_ylabel(f"rank by {metric} (1 = best)")
    ax.set_yticks(range(1, len(pivot) + 1))
    ax.invert_yaxis()
    ax.set_title(f"Architecture ranking across evaluation regimes ({metric})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    arms = load_arms(args)

    corr = correlations(arms, args.metrics)
    corr.to_csv(os.path.join(args.out, "rank_correlation.csv"), index=False)
    summary = correlation_summary(corr)
    summary.to_csv(os.path.join(args.out, "rank_correlation_summary.csv"),
                   index=False)

    table = ranking_table(arms, args.metrics)
    table.to_csv(os.path.join(args.out, "ranking_table.csv"), index=False)

    chart = bump_chart(table, args.bump_metric,
                       os.path.join(args.out, f"bump_chart_{args.bump_metric}.png"))

    print("== rank correlation (Regime A vs Regime B) ==")
    print(summary.round(4).to_string(index=False) if not summary.empty
          else "(no overlapping (grid, seed) keys)")
    print("\n== ranking by arm ==")
    for arm, metric in itertools.product(arms, [args.bump_metric]):
        sub = table[(table.arm == arm) & (table.metric == metric)]
        if sub.empty:
            continue
        print(f"\n-- {arm} ({metric})")
        print(sub[["model", "rank_overall", "mean", "sd", "modal_rank",
                   "overlaps_next"]].round(5).to_string(index=False))
    if chart:
        print(f"\nbump chart: {chart}")


if __name__ == "__main__":
    main()
