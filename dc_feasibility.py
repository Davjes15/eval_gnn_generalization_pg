"""dc_feasibility.py -- the two reference rows of the AC feasibility table (C5).

WHY
    `eval_checkpoints.py --feasibility` reports how far a predicted state is from
    satisfying the AC power flow, in MW and as a share of served load. A number
    like "40 % of served load" is only readable against two references, and the
    third audit was right that the table had neither:

    * the RECONSTRUCTION FLOOR -- the same residual evaluated on the labels. It
      is what this pipeline cannot do better than, and it comes out at ~3e-2 MW,
      so it is not what the model numbers are made of.
    * the DC POWER FLOW -- the baseline the whole paper compares NRMSE against.
      Its state is stored per sample (`dc_pf`), so scoring it through the same
      checker costs one pass over the test splits and no training.

    Together they bracket the model rows: a surrogate that is worse than DC on
    physical feasibility is not a screening tool, whatever its NRMSE says.

READING THE DC ROW
    DC power flow assumes |V| = 1 pu everywhere and Q = 0 (the convention
    `training_utils.apply_dc_convention` enforces, audit A1), so its REACTIVE
    residual is large by construction -- it is essentially the reactive demand of
    the snapshot, and it says nothing about the approximation's quality. The
    comparable columns are the active-power residual and the thermal screening.

USAGE
    POWERGRAPH_NODE_DIR=... python dc_feasibility.py \
        --data data_a data_full_v2 --out results_norm/physics/dc_feasibility.csv
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import torch

from ac_feasibility import build_cases, feasibility_metrics
from training_utils import apply_dc_convention, load_grid_dataset

GRIDS = ("IEEE24", "IEEE39", "IEEE118", "UK")


def _states(dataset):
    """(true, dc) physical state matrices stacked over the split's samples.

    The dataset is read straight from disk and never scaled here, so `y` is in
    physical units (MW/Mvar/pu/rad) -- the `y_raw` copy that `normalization.py`
    adds only exists inside a training run.
    """
    true = torch.cat([d.y for d in dataset])
    dc = apply_dc_convention(torch.cat([d.dc_pf for d in dataset]))
    return true.numpy(), dc.numpy()


def score(data_dir: str, grid: str, cases_dir: str | None = None):
    """The floor row and the DC row for one grid of one dataset."""
    dataset = load_grid_dataset(data_dir, grid, "test")
    cases = build_cases(grid, os.path.join(data_dir, grid, "test"), cases_dir)
    true, dc = _states(dataset)
    n_bus = true.shape[0] // len(cases)
    rows = []
    for state, name in ((true, "truth"), (dc, "dc_pf")):
        row = {"data_dir": data_dir, "grid": grid, "state": name,
               "n_samples": len(cases), "n_bus": n_bus}
        row.update(feasibility_metrics(true, state, cases, n_bus))
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", default=["data_a", "data_full_v2"])
    p.add_argument("--grids", nargs="+", default=list(GRIDS))
    p.add_argument("--cases_dir", default=None)
    p.add_argument("--out", default="results_norm/physics/dc_feasibility.csv")
    args = p.parse_args()

    rows = []
    for data_dir in args.data:
        for grid in args.grids:
            for row in score(data_dir, grid, args.cases_dir):
                rows.append(row)
                print(f"  [{data_dir}/{grid}/{row['state']}] "
                      f"dP={row['ac_dp_mean_mw']:.4g} MW "
                      f"({row['ac_dp_pct_load']:.3g} % of load) "
                      f"load={row['branch_loading_max_pct']:.4g} %", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nwrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
