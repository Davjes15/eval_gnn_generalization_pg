"""Focused checks for rank_analysis.py (Step 7).

Synthetic result CSVs with a KNOWN ranking change, so the tau-b value can be
verified by hand rather than trusted.

    python3 tests/test_rank_analysis.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rank_analysis  # noqa: E402
from rank_analysis import (  # noqa: E402
    bump_chart,
    correlation_summary,
    correlations,
    load_arms,
    ranking_table,
    ranks,
)

MODELS = ["gcn", "gat", "gin", "transformer"]
GRIDS = ["IEEE24", "IEEE39"]
SEEDS = [0, 100]


class Args:
    def __init__(self, tmp):
        self.regime_a = os.path.join(tmp, "within_grid.csv")
        self.cross = os.path.join(tmp, "cross_context.csv")
        self.ood = os.path.join(tmp, "ood.csv")
        self.out = tmp
        self.metrics = ["nrmse", "mse", "mae"]
        self.bump_metric = "nrmse"


def write_fixtures(tmp, reversed_b=True):
    """Regime A ranks models best->worst in MODELS order.

    With reversed_b=True Regime B reverses that order exactly, so tau-b must be
    -1.0 and every architecture's rank trajectory must cross.
    """
    a_rows, cc_rows, ood_rows = [], [], []
    for grid in GRIDS:
        for seed in SEEDS:
            for i, m in enumerate(MODELS):
                base = 0.1 * (i + 1)
                a_rows.append({"model": m, "grid": grid, "seed": seed,
                               "regime": "A", "nrmse": base, "mse": base * 10,
                               "mae": base * 2})
                j = (len(MODELS) - 1 - i) if reversed_b else i
                bb = 0.1 * (j + 1) + 1.0
                for test_grid in GRIDS:
                    cc_rows.append({"model": m, "train_grid": grid,
                                    "test_grid": test_grid,
                                    "unseen": test_grid != grid, "seed": seed,
                                    "regime": "B", "nrmse": bb,
                                    "mse": bb * 10, "mae": bb * 2})
                ood_rows.append({"model": m, "held_out_grid": grid, "seed": seed,
                                 "regime": "B", "nrmse": bb, "mse": bb * 10,
                                 "mae": bb * 2})
    pd.DataFrame(a_rows).to_csv(os.path.join(tmp, "within_grid.csv"), index=False)
    pd.DataFrame(cc_rows).to_csv(os.path.join(tmp, "cross_context.csv"), index=False)
    pd.DataFrame(ood_rows).to_csv(os.path.join(tmp, "ood.csv"), index=False)


def check(cond, label):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    if not cond:
        raise AssertionError(label)


def test_cross_context_uses_unseen_only():
    with tempfile.TemporaryDirectory() as tmp:
        write_fixtures(tmp)
        arms = load_arms(Args(tmp))
        cc = arms["cross_context"]
        check(set(cc.columns) >= {"model", "grid", "seed", "nrmse"},
              "cross-context arm is keyed by (model, grid, seed)")
        check(len(cc) == len(MODELS) * len(GRIDS) * len(SEEDS),
              "one cross-context row per (model, train grid, seed)")
        # Same-grid rows carry the WITHIN-grid error; if they leaked in, the
        # mean would not equal the unseen-only value.
        raw = pd.read_csv(os.path.join(tmp, "cross_context.csv"))
        unseen = raw[raw.unseen]
        expect = unseen.groupby(["model", "train_grid", "seed"]).nrmse.mean()
        got = cc.set_index(["model", "grid", "seed"]).nrmse
        check(all(abs(got.loc[k] - v) < 1e-12 for k, v in expect.items()),
              "same-grid (seen) cross-context rows are excluded")


def test_ranks_and_reversal():
    with tempfile.TemporaryDirectory() as tmp:
        write_fixtures(tmp, reversed_b=True)
        arms = load_arms(Args(tmp))
        a = arms["regime_a"]
        r = ranks(a[(a.grid == "IEEE24") & (a.seed == 0)], "nrmse")
        check(list(r.sort_values().index) == MODELS,
              "rank 1 is the lowest error")

        corr = correlations(arms, ["nrmse", "mse", "mae"])
        check(not corr.empty, "correlations produced rows")
        check(set(corr.comparison) == {"regime_a_vs_cross_context",
                                       "regime_a_vs_ood"},
              "both Regime A vs Regime B comparisons are present")
        n_expect = 2 * len(GRIDS) * len(SEEDS) * 3
        check(len(corr) == n_expect,
              f"one row per (comparison, grid, seed, metric) = {n_expect}")
        check(all(abs(t + 1.0) < 1e-12 for t in corr.kendall_tau_b),
              "an exactly reversed ranking gives tau-b = -1")
        check(all(abs(r + 1.0) < 1e-12 for r in corr.spearman_rho),
              "an exactly reversed ranking gives rho = -1")

        summ = correlation_summary(corr)
        check(set(summ.columns) >= {"tau_mean", "tau_sd", "tau_min", "tau_max",
                                    "rho_mean", "n"},
              "summary reports mean/sd/min/max -- the seed-stability check")
        check(all(abs(v + 1.0) < 1e-12 for v in summ.tau_mean),
              "summary tau_mean = -1")


def test_identical_ranking_gives_tau_plus_one():
    """The pre-registered null outcome must come out as exactly +1."""
    with tempfile.TemporaryDirectory() as tmp:
        write_fixtures(tmp, reversed_b=False)
        arms = load_arms(Args(tmp))
        corr = correlations(arms, ["nrmse"])
        check(all(abs(t - 1.0) < 1e-12 for t in corr.kendall_tau_b),
              "a regime-invariant ranking gives tau-b = +1")


def test_ranking_table_and_bump_chart():
    with tempfile.TemporaryDirectory() as tmp:
        write_fixtures(tmp, reversed_b=True)
        args = Args(tmp)
        arms = load_arms(args)
        table = ranking_table(arms, ["nrmse"])
        check(set(table.arm) == {"regime_a", "cross_context", "ood"},
              "ranking table covers all three arms")
        check(len(table) == 3 * len(MODELS), "one row per (arm, model)")

        a = table[table.arm == "regime_a"].sort_values("rank_overall")
        check(list(a.model) == MODELS, "Regime A ordering matches the fixture")
        b = table[table.arm == "ood"].sort_values("rank_overall")
        check(list(b.model) == MODELS[::-1], "OOD ordering is reversed")
        check(all(a.modal_rank == a.rank_overall),
              "modal rank agrees with the overall rank when seeds agree")
        check(all(isinstance(f, dict) for f in table.rank_frequencies),
              "rank frequencies are recorded per architecture")
        check(not any(table.overlaps_next),
              "well-separated errors are not flagged as overlapping")

        path = bump_chart(table, "nrmse", os.path.join(tmp, "bump.png"))
        check(path is not None and os.path.getsize(path) > 1000,
              "bump chart written")


def test_overlap_flag():
    """Adjacent architectures within one sd of each other must be flagged."""
    frame = pd.DataFrame([
        {"model": "a", "grid": "G", "seed": s, "nrmse": 1.0 + 0.5 * s}
        for s in (0, 1)
    ] + [
        {"model": "b", "grid": "G", "seed": s, "nrmse": 1.2 + 0.5 * s}
        for s in (0, 1)
    ])
    table = ranking_table({"regime_a": frame}, ["nrmse"])
    check(bool(table.iloc[0].overlaps_next),
          "overlapping mean +/- sd intervals are flagged")


def test_missing_model_is_dropped_not_fatal():
    """A diverged run (NaN error) must not silently shift everyone's rank."""
    with tempfile.TemporaryDirectory() as tmp:
        write_fixtures(tmp, reversed_b=True)
        path = os.path.join(tmp, "within_grid.csv")
        df = pd.read_csv(path)
        df.loc[(df.model == "gin") & (df.grid == "IEEE24") & (df.seed == 0),
               "nrmse"] = float("nan")
        df.to_csv(path, index=False)
        corr = correlations(load_arms(Args(tmp)), ["nrmse"])
        row = corr[(corr.grid == "IEEE24") & (corr.seed == 0)
                   & (corr.comparison == "regime_a_vs_ood")].iloc[0]
        check(row.n_models == len(MODELS) - 1,
              "a NaN error drops that architecture from the comparison only")
        full = corr[(corr.grid == "IEEE39")].iloc[0]
        check(full.n_models == len(MODELS),
              "other (grid, seed) cells keep all architectures")


def test_metric_absent_is_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        write_fixtures(tmp)
        args = Args(tmp)
        arms = load_arms(args)
        corr = correlations(arms, ["nrmse", "not_a_metric"])
        check(set(corr.metric) == {"nrmse"},
              "an unavailable metric is skipped rather than raising")


if __name__ == "__main__":
    print(f"rank_analysis from {rank_analysis.__file__}")
    for fn in (test_cross_context_uses_unseen_only,
               test_ranks_and_reversal,
               test_identical_ranking_gives_tau_plus_one,
               test_ranking_table_and_bump_chart,
               test_overlap_flag,
               test_missing_model_is_dropped_not_fatal,
               test_metric_absent_is_skipped):
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + "=" * 50 + "\nALL CHECKS PASSED")
