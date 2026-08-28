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
    correction). Spearman rho is reported as a secondary check. At n = 6 the
    per-cell p-value is uninformative, so the mean tau over cells is tested
    against an exact permutation null over the architecture labels.

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
    results_tuned/cross_context.csv    (Regime B CC + the same-grid diagonal, Step 6)
    results_tuned/ood.csv              (Regime B OOD, Step 6)
        -> results/rank_correlation.csv   (tau-b / rho per grid, seed, metric)
        -> results/rank_correlation_pooled.csv (tau between the pooled leaderboards)
        -> results/protocol_decomposition.csv  (protocol step vs unseen-grid step)
        -> results/rank_permutation_test.csv (exact null for the mean tau)
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
# Ordered by how much they change relative to Regime A: the diagonal arm changes
# only the protocol (blocked split + N-k topologies, SAME grid), the other two
# additionally change the grid. Comparing A directly with the unseen arms alone
# confounds those two effects (audit B2).
ARMS = ["regime_b_diagonal", "cross_context", "ood"]


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
      * regime_b_diagonal -- Regime B evaluated on the grid it was TRAINED on,
        i.e. the same-grid rows of the cross-context table. Same grid as Regime
        A, different protocol (blocked temporal split, N-k topologies), so the
        A -> diagonal step isolates protocol difficulty from unseen-grid
        difficulty.
      * OOD -- error on the held-out grid, attributed to that held-out grid.
    """
    arms = {}

    a = pd.read_csv(args.regime_a)
    arms["regime_a"] = a.rename(columns={"grid": "grid"})

    cc_all = pd.read_csv(args.cross)
    if "unseen" in cc_all.columns:
        diag = cc_all[~cc_all.unseen].rename(columns={"train_grid": "grid"})
        arms["regime_b_diagonal"] = diag.drop(columns=["test_grid"],
                                              errors="ignore")
    cc = cc_all[cc_all.unseen] if "unseen" in cc_all.columns else cc_all
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


def permutation_test(arms, metrics, min_models=6):
    """Is the mean tau over cells distinguishable from a random relabelling?

    Kendall's own p-value is per cell and at n = 6 architectures it cannot resolve
    anything; the statistic that carries the claim is the MEAN tau over the
    (grid, seed) cells, and its null is obtained by permuting the architecture
    labels of the Regime B ranking within every cell. All `min_models`!
    relabellings are enumerated, so the p-value is exact, not sampled.

    Cells with fewer than `min_models` architectures are dropped: a mean taken
    over cells of different width is not a statistic with one null.
    """
    rows = []
    a = arms["regime_a"]
    perms = list(itertools.permutations(range(min_models)))
    for arm in ARMS:
        b = arms[arm]
        for metric in metrics:
            if metric not in a.columns or metric not in b.columns:
                continue
            pairs = []
            keys = sorted(set(map(tuple, a[["grid", "seed"]].values))
                          & set(map(tuple, b[["grid", "seed"]].values)))
            for grid, seed in keys:
                ra = ranks(a[(a.grid == grid) & (a.seed == seed)], metric)
                rb = ranks(b[(b.grid == grid) & (b.seed == seed)], metric)
                shared = sorted(set(ra.index) & set(rb.index))
                if len(shared) != min_models:
                    continue
                pairs.append((ra[shared].to_numpy(), rb[shared].to_numpy()))
            if not pairs:
                continue
            observed = float(np.mean([stats.kendalltau(x, y, variant="b").statistic
                                      for x, y in pairs]))
            null = np.array([
                np.mean([stats.kendalltau(x, y[list(p)], variant="b").statistic
                         for x, y in pairs])
                for p in perms])
            # two-sided: how often is a random relabelling at least this extreme
            p_value = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
            rows.append({"comparison": f"regime_a_vs_{arm}", "metric": metric,
                         "n_cells": len(pairs), "n_models": min_models,
                         "observed_mean_tau": observed,
                         "null_mean": float(null.mean()),
                         "null_sd": float(null.std(ddof=1)),
                         "n_relabellings": len(perms), "p_value": p_value})
    return pd.DataFrame(rows)


def pooled_correlations(arms, metrics):
    """tau between arms after pooling, i.e. between the LEADERBOARDS.

    `correlations` asks whether the ordering holds inside a single (grid, seed)
    cell, which is the question a practitioner faces. This asks the weaker
    question the pooled tables answer: rank the architectures by their mean error
    over everything, then compare those two orderings. The two can disagree --
    a pooled ordering can be reproducible while no individual cell reproduces it
    -- and reporting only the per-cell statistic overstates the instability
    (audit B3), so both are emitted.
    """
    rows = []
    names = ["regime_a"] + [a for a in ARMS if a in arms]
    for metric in metrics:
        pooled = {}
        for name in names:
            frame = arms[name]
            if metric not in frame.columns:
                continue
            pooled[name] = frame.groupby("model")[metric].mean()
        for left, right in itertools.combinations(pooled, 2):
            shared = sorted(set(pooled[left].index) & set(pooled[right].index))
            if len(shared) < 3:
                continue
            x = pooled[left][shared].rank().to_numpy()
            y = pooled[right][shared].rank().to_numpy()
            tau = stats.kendalltau(x, y, variant="b")
            rho = stats.spearmanr(x, y)
            rows.append({"comparison": f"{left}_vs_{right}", "metric": metric,
                         "n_models": len(shared),
                         "kendall_tau_b": tau.statistic,
                         "spearman_rho": rho.statistic,
                         "order_left": " > ".join(pooled[left][shared]
                                                  .sort_values().index),
                         "order_right": " > ".join(pooled[right][shared]
                                                   .sort_values().index)})
    return pd.DataFrame(rows)


def protocol_decomposition(arms, metric="nrmse"):
    """Split the transfer gap into a protocol step and an unseen-grid step.

    Regime A -> regime_b_diagonal changes the evaluation protocol on the SAME
    grid; regime_b_diagonal -> cross_context/ood then changes the grid. Quoting
    a single A -> unseen ratio attributes both to generalization.
    """
    if "regime_a" not in arms:
        return pd.DataFrame()
    base = arms["regime_a"].groupby("model")[metric].mean()
    rows = []
    for model in base.index:
        row = {"model": model, "metric": metric, "regime_a": float(base[model])}
        for name in ARMS:
            frame = arms.get(name)
            if frame is None or metric not in frame.columns:
                continue
            sub = frame[frame.model == model][metric]
            row[name] = float(sub.mean()) if len(sub) else float("nan")
        diag = row.get("regime_b_diagonal", float("nan"))
        row["protocol_factor"] = diag / row["regime_a"]
        for name in ("cross_context", "ood"):
            if name in row:
                row[f"unseen_grid_factor_{name}"] = row[name] / diag
                row[f"total_factor_{name}"] = row[name] / row["regime_a"]
        rows.append(row)
    return pd.DataFrame(rows)


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
    order = [a for a in ("regime_a", "regime_b_diagonal", "cross_context", "ood")
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
    labels = {"regime_a": "Regime A\n(fixed topology)",
              "regime_b_diagonal": "Regime B\n(same grid, N-k)",
              "cross_context": "Cross-context\n(unseen grid)",
              "ood": "OOD\n(leave-one-grid-out)"}
    ax.set_xticklabels([labels[a] for a in order])
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

    perm = permutation_test(arms, args.metrics)
    perm.to_csv(os.path.join(args.out, "rank_permutation_test.csv"), index=False)

    pooled = pooled_correlations(arms, args.metrics)
    pooled.to_csv(os.path.join(args.out, "rank_correlation_pooled.csv"),
                  index=False)

    decomp = protocol_decomposition(arms, args.bump_metric)
    decomp.to_csv(os.path.join(args.out, "protocol_decomposition.csv"),
                  index=False)

    table = ranking_table(arms, args.metrics)
    table.to_csv(os.path.join(args.out, "ranking_table.csv"), index=False)

    chart = bump_chart(table, args.bump_metric,
                       os.path.join(args.out, f"bump_chart_{args.bump_metric}.png"))

    print("== rank correlation (Regime A vs Regime B) ==")
    print(summary.round(4).to_string(index=False) if not summary.empty
          else "(no overlapping (grid, seed) keys)")
    if not perm.empty:
        print("\n== permutation test on the mean tau (exact, all 6! "
              "relabellings per cell) ==")
        print(perm.round(4).to_string(index=False))
    if not pooled.empty:
        print("\n== pooled-leaderboard correlation (ordering of the arm means) ==")
        print(pooled[pooled.metric == args.bump_metric]
              [["comparison", "kendall_tau_b", "spearman_rho"]]
              .round(4).to_string(index=False))
    if not decomp.empty:
        print("\n== protocol step vs unseen-grid step ==")
        print(decomp.round(4).to_string(index=False))

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
