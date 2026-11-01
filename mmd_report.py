"""mmd_report.py -- the grid-distance tables behind the g-score (audit item A7).

Writes, for one dataset directory, the pairwise MMD between every training grid
and every test grid under three descriptors:

    degree        topological (node-degree histogram)
    laplacian     topological (normalised-Laplacian spectral histogram)
    reactance     ELECTRICAL (log10 branch reactance histogram)

plus each pair under the biased (published) and the unbiased estimator, so the
estimator choice is visible rather than asserted.

Why the third descriptor: the two topological descriptors are invariant to the
electrical size of a system, so they cannot register the ~20x power-scale spread
between our four cases -- plausibly the shift that dominates transfer error. A
distance blind to the dominant shift is a weak covariate for a generalization
score; this table shows how much of the apparent "topological distance" between
our grids is actually electrical.

    python mmd_report.py --data_dir data_regime_b_contingencies --out docs/tables
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from mmd_utils import evaluate_mmd, evaluate_mmd_electrical, resolve_mmd_bandwidth
from training_utils import load_grid_dataset

GRIDS = ("IEEE24", "IEEE39", "IEEE118", "UK")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="data_regime_b_contingencies")
    p.add_argument("--grids", nargs="+", default=list(GRIDS))
    p.add_argument("--out", default="docs/tables")
    p.add_argument("--mmd_graph", choices=["multi", "simple"], default="multi")
    p.add_argument("--mmd_bandwidth", choices=["reference", "pair"], default="reference")
    p.add_argument("--mmd_bandwidth_file", default="configs/mmd_bandwidth.json")
    args = p.parse_args()

    bandwidth = resolve_mmd_bandwidth(args.mmd_bandwidth, args.mmd_bandwidth_file,
                                      [args.data_dir], args.mmd_graph)
    legacy = args.mmd_graph == "simple" and args.mmd_bandwidth == "pair"
    train = {g: load_grid_dataset(args.data_dir, g, "train") for g in args.grids}
    test = {g: load_grid_dataset(args.data_dir, g, "test") for g in args.grids}

    rows = []
    for a in args.grids:
        for b in args.grids:
            for unbiased in (False, True):
                deg, lap = evaluate_mmd(train[a], test[b], unbiased, args.mmd_graph,
                                        bandwidth)
                electrical = evaluate_mmd_electrical(train[a], test[b], unbiased, bandwidth)
                rows.append({
                    "train_grid": a, "test_grid": b,
                    "same_grid": a == b,
                    "estimator": "unbiased" if unbiased else "biased",
                    "mmd_degree": deg,
                    "mmd_laplacian": lap,
                    "mmd_reactance": electrical,
                    **({"mmd_graph": args.mmd_graph} if args.mmd_graph == "multi" else {}),
                    "mmd2_degree": deg ** 2, "mmd2_laplacian": lap ** 2,
                    "mmd2_reactance": electrical ** 2,
                    "bandwidth_mode": args.mmd_bandwidth,
                })
            print(f"  {a} -> {b}: " + ", ".join(
                f"{r['estimator']} deg={r['mmd_degree']:.4f} "
                f"lap={r['mmd_laplacian']:.4f} react={r['mmd_reactance']:.4f}"
                for r in rows[-2:]), flush=True)

    os.makedirs(args.out, exist_ok=True)
    df = pd.DataFrame(rows)
    suffix = "_multigraph" if args.mmd_graph == "multi" else ""
    if args.mmd_bandwidth == "reference":
        suffix += "_reference"
    path = os.path.join(args.out, f"mmd_{os.path.basename(args.data_dir)}{suffix}.csv")
    if os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite {path}; choose a new --out directory")
    if args.mmd_graph == "multi":
        print("PROVISIONAL: current data predates P8; final numbers require P9.")
    if legacy:
        details_path = path.removesuffix(".csv") + "_details.csv"
        if os.path.exists(details_path):
            raise FileExistsError(f"Refusing to overwrite {details_path}")
        df.to_csv(details_path, index=False)
        legacy_columns = [c for c in df if not c.startswith("mmd2_") and c != "bandwidth_mode"]
        df[legacy_columns].to_csv(path, index=False)
        print(f"wrote detailed MMD/MMD² report -> {details_path}")
    else:
        df.to_csv(path, index=False)
    print(f"\nwrote {len(df)} rows -> {path}")

    biased = df[df.estimator == "biased"]
    print("\nmean MMD, same-grid vs different-grid (biased estimator):")
    print(biased.groupby("same_grid")[
        ["mmd_degree", "mmd_laplacian", "mmd_reactance"]].mean().to_string())


if __name__ == "__main__":
    main()
