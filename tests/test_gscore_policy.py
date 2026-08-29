"""test_gscore_policy.py -- how the g-score tables treat divergence and trimming.

Run:  python3 tests/test_gscore_policy.py

Two defects motivate these checks. A model that emits a non-finite transfer error
used to be scored on its surviving pairs, i.e. on the pairs it did NOT fail, which
made divergence look like generalization. And the per-training-grid table used the
ENGAGE percentile trim on three points, which keeps one point, so `mean_nrmse` was
a median and `std_nrmse` was identically zero.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from experiments import compute_cc_aggregate_gscores, compute_gscores
from training_utils import gscore_row

FAILURES = []
GRIDS = ["IEEE24", "IEEE39"]


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def _lap(grids):
    n = len(grids)
    return pd.DataFrame(np.full((n, n), 0.5) - np.eye(n) * 0.5,
                        index=grids, columns=grids)


def _cc(model, values, train_grid="IEEE24"):
    """One cross-context record per unseen test grid."""
    return [{"model": model, "train_grid": train_grid, "test_grid": g,
             "unseen": True, "nrmse": v}
            for g, v in zip(["IEEE39", "IEEE118", "UK"], values)]


def test_divergence_voids_the_score_instead_of_improving_it():
    print("\nA non-finite transfer error voids the cell rather than being dropped")
    good = gscore_row([0.0, 0.5, 1.0], [0.9, 1.0, 1.1])
    diverged = gscore_row([0.0, 0.5, 1.0], [0.9, 1.0, float("nan")])
    check("a complete cell is scored",
          math.isfinite(good["g_score"]) and good["finite_rate"] == 1.0,
          str(good["g_score"]))
    check("an incomplete cell has no g-score",
          math.isnan(diverged["g_score"]), str(diverged["g_score"]))
    check("the failure is visible as a finite rate",
          math.isclose(diverged["finite_rate"], 2 / 3)
          and diverged["n_expected"] == 3 and diverged["n_finite"] == 2,
          str(diverged))
    check("the surviving points are still described",
          math.isclose(diverged["mean_nrmse"], 0.95), str(diverged["mean_nrmse"]))
    empty = gscore_row([0.0, 0.5], [float("nan"), float("nan")])
    check("an all-non-finite cell is NaN throughout",
          math.isnan(empty["g_score"]) and math.isnan(empty["mean_nrmse"]),
          str(empty))


def test_per_training_grid_table_is_not_trimmed_to_a_single_point():
    print("\nThe per-training-grid table reports a real mean and spread")
    values = [0.10, 0.20, 0.60]
    rows = compute_gscores(_cc("gcn", values), _lap(GRIDS + ["IEEE118", "UK"]),
                           ["gcn"], ["IEEE24"])
    row = rows[0]
    check("mean_nrmse is the mean of all three unseen grids",
          math.isclose(row["mean_nrmse"], float(np.mean(values))),
          f"{row['mean_nrmse']} vs {np.mean(values)}")
    check("std_nrmse is not identically zero",
          math.isclose(row["std_nrmse"], float(np.std(values))),
          str(row["std_nrmse"]))
    check("completeness travels with the row", row["finite_rate"] == 1.0)


def test_dc_reference_row_declares_its_aggregation_basis():
    print("\nThe DC row is labelled: it pools grids, not ordered pairs")
    records = _cc("gcn", [0.10, 0.20, 0.60])
    dc_rows = [{"dc_nrmse": 0.05}, {"dc_nrmse": 0.06}]
    rows = compute_cc_aggregate_gscores(
        records, _lap(GRIDS + ["IEEE118", "UK"]), dc_rows, ["gcn"], GRIDS)
    by_model = {r["model"]: r for r in rows}
    check("the model row is scored over unseen pairs",
          by_model["gcn"]["basis"] == "unseen_pairs"
          and by_model["gcn"]["n_expected"] == 3, str(by_model["gcn"]))
    check("the DC row declares a different basis",
          by_model["dc_pf"]["basis"] == "one_point_per_grid"
          and by_model["dc_pf"]["n_expected"] == 2, str(by_model["dc_pf"]))


if __name__ == "__main__":
    test_divergence_voids_the_score_instead_of_improving_it()
    test_per_training_grid_table_is_not_trimmed_to_a_single_point()
    test_dc_reference_row_declares_its_aggregation_basis()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all checks passed")
