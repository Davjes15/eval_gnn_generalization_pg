"""recompute_dc_baseline.py -- rebuild the DC power-flow baseline table under the
stated reactive-power convention.

WHY
    `pp.rundcpp` never writes `res_bus.q_mvar` -- DC power flow has no reactive
    power. ENGAGE read the column anyway and zeroed the NaNs it got back
    (`graph_gen.py`, pandapower 2.14). Under pandapower >= 3 the column instead
    keeps whatever an earlier AC solve left in it, so datasets generated here
    stored the AC ground-truth Q as the "DC prediction" and every dc_nrmse_Q
    came out as exactly 0 -- the baseline was scored against its own labels.

    `training_utils.apply_dc_convention` now forces Q = 0 at scoring time, which
    reproduces ENGAGE's intent on datasets already written (P, V and theta are
    unaffected: a DC solve on a fresh net and on a copy of an AC-solved net give
    identical values for those three columns). This script re-scores the stored
    baseline so the tables match.

    The table also carries `dc_nrmse_PVtheta`, the aggregate over the three
    quantities DC actually solves, so a GNN-vs-DC comparison can be stated
    either way: charging DC for Q = 0 (ENGAGE's convention, comparable to their
    published numbers) or excluding Q entirely.

HOW TO RUN
    python3 recompute_dc_baseline.py --data_dir data_a \
        --out results/analysis/dc_baseline_regime_a.csv
"""
import argparse

import pandas as pd

from experiments import dc_baseline
from training_utils import load_grid_dataset
from transmission_grids import get_transmission_grid_codes


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", required=True,
                   help="dataset root holding <GRID>/test/dataset.pt")
    p.add_argument("--grids", nargs="+", default=None,
                   help="default: every transmission grid code")
    p.add_argument("--out", required=True, help="output CSV path")
    return p.parse_args()


def main():
    args = parse_args()
    grids = args.grids or get_transmission_grid_codes()
    data = {g: {"test": load_grid_dataset(args.data_dir, g, "test")}
            for g in grids}
    table = pd.DataFrame(dc_baseline(data, grids))
    table.to_csv(args.out, index=False)
    cols = ["grid", "dc_nrmse", "dc_nrmse_PVtheta", "dc_nrmse_P",
            "dc_nrmse_Q", "dc_nrmse_V", "dc_nrmse_theta"]
    print(table[cols].round(4).to_string(index=False))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
