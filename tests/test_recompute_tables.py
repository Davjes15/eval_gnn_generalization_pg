"""test_recompute_tables.py -- checks for the consolidated downstream tables.

Run:  python3 tests/test_recompute_tables.py

These tables are the ones that get read as results, so the checks are that the
per-quantity table keeps P/Q/V/theta separate (never averaged into one number),
that the DC comparison divides by the DC error of the same quantity, uses a
per-arm DC table and reports the Q-excluded aggregate, and that topology inputs
from disagreeing shards are refused instead of silently merged.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from recompute_tables import QUANTITIES, dc_comparison, per_quantity, topology_inputs

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def _rows(model, base):
    """Two seeds of one model with a distinct value per quantity."""
    out = []
    for i, seed in enumerate((0, 100)):
        row = {"model": model, "grid": "IEEE24", "seed": seed, "regime": "A",
               "nrmse": base, "mse": base, "mae": base}
        for j, q in enumerate(QUANTITIES):
            step = base * (j + 1) + i
            row[f"nrmse_{q}"] = step
            row[f"mse_{q}"] = step * 10
            row[f"mae_{q}"] = step / 10
        out.append(row)
    return out


def _dc(scale=1.0):
    """A DC table with a non-zero Q error: DC predicts Q = 0, it is not exempt."""
    row = {"grid": "IEEE24", "dc_nrmse": 0.02 * scale,
           "dc_nrmse_PVtheta": 0.015 * scale}
    for j, q in enumerate(QUANTITIES):
        row[f"dc_nrmse_{q}"] = 0.01 * (j + 1) * scale
    return pd.DataFrame([row])


def test_per_quantity_keeps_targets_separate():
    print("\nPer-quantity table reports P, Q, V, theta separately")
    df = pd.DataFrame(_rows("gcn", 1.0) + _rows("gat", 2.0))
    out = per_quantity({"regime_a": df})
    check("one row per (model, quantity)", len(out) == 8, str(len(out)))
    check("all four quantities present",
          sorted(out.quantity.unique()) == sorted(QUANTITIES),
          str(sorted(out.quantity.unique())))
    gcn_v = out[(out.model == "gcn") & (out.quantity == "V")]
    # gcn V over seeds 0, 100 is 3.0 and 4.0 -> mean 3.5, sd 1/sqrt(2)
    check("mean is over seeds of that quantity only",
          math.isclose(float(gcn_v.nrmse_mean.iloc[0]), 3.5),
          str(float(gcn_v.nrmse_mean.iloc[0])))
    check("sd is reported alongside the mean",
          math.isclose(float(gcn_v.nrmse_sd.iloc[0]), 2 ** -0.5),
          str(float(gcn_v.nrmse_sd.iloc[0])))
    quantities = out[out.model == "gcn"].set_index("quantity").nrmse_mean
    check("quantities are not collapsed into one value",
          len(set(quantities.round(6))) == 4, str(dict(quantities)))


def test_dc_comparison_matches_quantities():
    print("\nDC comparison divides by the same quantity, per arm")
    df = pd.DataFrame(_rows("gcn", 1.0))
    tables = {"regime_a": _dc(), "ood": _dc(scale=2.0)}
    out = dc_comparison({"regime_a": df, "ood": df}, tables)
    p = out[(out.arm == "regime_a") & (out.quantity == "P")].iloc[0]
    check("P ratio uses dc_nrmse_P", math.isclose(p.gnn_over_dc, p.gnn_nrmse / 0.01),
          f"{p.gnn_nrmse} / 0.01 -> {p.gnn_over_dc}")
    q = out[(out.arm == "regime_a") & (out.quantity == "Q")].iloc[0]
    check("Q has a real DC error under the Q = 0 convention",
          math.isclose(q.dc_nrmse, 0.02) and math.isclose(
              q.gnn_over_dc, q.gnn_nrmse / 0.02),
          f"{q.dc_nrmse} -> {q.gnn_over_dc}")
    agg = out[(out.arm == "regime_a") & (out.quantity == "aggregate")].iloc[0]
    check("the aggregate row uses the aggregate NRMSE columns",
          math.isclose(agg.gnn_nrmse, 1.0) and math.isclose(agg.dc_nrmse, 0.02),
          f"{agg.gnn_nrmse} vs {agg.dc_nrmse}")
    pvt = out[(out.arm == "regime_a") & (out.quantity == "PVtheta")].iloc[0]
    # gcn P/V/theta over seeds 0, 100: (1,3,4) and (2,4,5) -> 19/6; DC 0.01,
    # 0.03, 0.04 -> 0.02666. Both sides average the three per-quantity NRMSEs.
    check("the Q-excluded aggregate averages P, V and theta only",
          math.isclose(pvt.gnn_nrmse, 19 / 6)
          and math.isclose(pvt.dc_nrmse, 0.08 / 3),
          f"{pvt.gnn_nrmse} vs {pvt.dc_nrmse}")
    ood_p = out[(out.arm == "ood") & (out.quantity == "P")].iloc[0]
    check("each arm is scored against its own DC table",
          math.isclose(ood_p.dc_nrmse, 0.02), str(ood_p.dc_nrmse))


def _topo_dir(tmp, name, lap_value, dc_value=0.02, with_ood=True):
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame([[0.1, lap_value], [lap_value, 0.1]],
                 index=["IEEE24", "IEEE39"],
                 columns=["IEEE24", "IEEE39"]).to_csv(
        os.path.join(d, "mmd_laplacian.csv"))
    dc = _dc()
    dc["dc_nrmse"] = dc_value
    dc.to_csv(os.path.join(d, "dc_baseline.csv"), index=False)
    if with_ood:
        pd.DataFrame([{"held_out_grid": "IEEE24", "train_grids": "IEEE39",
                       "mmd_pooled_degree": 0.6,
                       "mmd_pooled_laplacian": 0.65}]).to_csv(
            os.path.join(d, "ood_distance.csv"), index=False)
    return d


def _expect_exit(fn, label):
    try:
        fn()
    except SystemExit as exc:
        check(label, True, str(exc).splitlines()[0][:70])
        return
    check(label, False, "no SystemExit raised")


def test_topology_inputs_must_agree():
    print("\nTopology inputs from disagreeing shards are refused")
    with tempfile.TemporaryDirectory() as tmp:
        a = _topo_dir(tmp, "a", 0.7)
        b = _topo_dir(tmp, "b", 0.7, with_ood=False)
        lap, pooled, dc = topology_inputs([a, b])
        check("agreeing shards merge", math.isclose(lap.loc["IEEE24", "IEEE39"], 0.7))
        check("pooled Laplacian distance is keyed by held-out grid",
              math.isclose(pooled["IEEE24"], 0.65), str(pooled))
        check("the shards' DC table is read for the agreement check",
              len(dc) == 1, str(len(dc)))
        c = _topo_dir(tmp, "c", 0.9)
        _expect_exit(lambda: topology_inputs([a, c]),
                     "a differing MMD matrix is refused")
        d = _topo_dir(tmp, "d", 0.7, dc_value=0.5)
        _expect_exit(lambda: topology_inputs([a, d]),
                     "a differing DC baseline is refused")
        e = _topo_dir(tmp, "e", 0.7, with_ood=False)
        _expect_exit(lambda: topology_inputs([e]),
                     "a missing ood_distance.csv is refused")


if __name__ == "__main__":
    test_per_quantity_keeps_targets_separate()
    test_dc_comparison_matches_quantities()
    test_topology_inputs_must_agree()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all checks passed")
