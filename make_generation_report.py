"""make_generation_report.py -- dataset provenance report for a generated regime.

PURPOSE
    Summarise what a `transmission_graph_gen.py` run actually produced, so a
    dataset directory carries its own evidence: split sizes, distinct demand
    snapshots, the topology distribution, the outage distribution, and the
    range of each target quantity. Optionally puts a reference regime's target
    ranges next to it, which is how we check that the fixed-topology control arm
    (`data_a/`) covers comparable physics to the topology-varying arm
    (`full_run/data/`) rather than a much narrower slice of it.

HOW TO RUN
    python3 make_generation_report.py --data_dir data_a \
        --reference_dir full_run/data --out data_a/GENERATION_REPORT.md
"""
from __future__ import annotations

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from training_utils import load_grid_dataset
from transmission_grids import get_transmission_grid_codes

SPLITS = ("train", "val", "test")
QUANTITIES = ("p_mw", "q_mvar", "vm_pu", "va_degree")


def _meta(data_dir: str, grid: str, split: str) -> pd.DataFrame | None:
    path = os.path.join(data_dir, grid, split, "dataset_src.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def _targets(data_dir: str, grid: str) -> np.ndarray | None:
    """Stack y over every split of a grid -> (n_samples * n_buses, 4)."""
    blocks = []
    for split in SPLITS:
        try:
            blocks += [d.y for d in load_grid_dataset(data_dir, grid, split)]
        except FileNotFoundError:
            continue
    return torch.cat(blocks).numpy() if blocks else None


def _stats_rows(y: np.ndarray, label: str) -> list[dict]:
    return [
        {"regime": label, "quantity": q, "min": y[:, i].min(), "max": y[:, i].max(),
         "mean": y[:, i].mean(), "std": y[:, i].std()}
        for i, q in enumerate(QUANTITIES)
    ]


def _cell(v) -> str:
    return f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)


def _fmt(df: pd.DataFrame) -> str:
    """Minimal markdown table (avoids a `tabulate` dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_cell(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def grid_section(data_dir: str, reference_dir: str | None, grid: str) -> str:
    out = [f"\n## {grid}\n"]

    rows = []
    for split in SPLITS:
        meta = _meta(data_dir, grid, split)
        if meta is None:
            continue
        rows.append({
            "split": split,
            "samples": len(meta),
            "distinct demand snapshots": meta["t_idx"].nunique(),
            "k values": sorted(meta["k"].unique().tolist()),
        })
    if not rows:
        return "".join(out + ["_no data found_\n"])
    out += ["### Splits\n", _fmt(pd.DataFrame(rows)), "\n"]

    # Overlap of demand snapshots between splits -- must be empty for a
    # fixed-topology regime, where a shared snapshot is a duplicated sample.
    sets = {s: set(_meta(data_dir, grid, s)["t_idx"]) for s in SPLITS
            if _meta(data_dir, grid, s) is not None}
    overlaps = {f"{a} n {b}": len(sets[a] & sets[b])
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
                if a in sets and b in sets}
    out += [f"\nDemand-snapshot overlap between splits: `{overlaps}`\n"]

    ds = [d for s in SPLITS for d in load_grid_dataset(data_dir, grid, s)]
    edge_counts = pd.Series([int(d.edge_index.shape[1]) for d in ds]).value_counts()
    out += [f"\nBuses: {int(ds[0].x.shape[0])}. Directed edges (2E) observed: "
            f"`{edge_counts.sort_index().to_dict()}`\n"]

    stats = _stats_rows(_targets(data_dir, grid), os.path.basename(data_dir.rstrip("/")))
    if reference_dir:
        ref = _targets(reference_dir, grid)
        if ref is not None:
            stats += _stats_rows(ref, os.path.basename(reference_dir.rstrip("/")))
    out += ["\n### Target ranges\n",
            _fmt(pd.DataFrame(stats).sort_values(["quantity", "regime"])), "\n"]
    return "".join(out)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--reference_dir", default=None,
                   help="second dataset dir whose target ranges are shown alongside")
    p.add_argument("--grids", nargs="+", default=None)
    p.add_argument("--out", default=None, help="markdown path (default: stdout)")
    p.add_argument("--title", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    grids = args.grids or get_transmission_grid_codes()
    title = args.title or f"Generation report -- `{args.data_dir}`"
    body = [f"# {title}\n",
            f"\nGenerated from `{os.path.abspath(args.data_dir)}`"]
    if args.reference_dir:
        body.append(f", compared against `{os.path.abspath(args.reference_dir)}`")
    body.append(".\n")
    for grid in grids:
        body.append(grid_section(args.data_dir, args.reference_dir, grid))
    text = "".join(body)

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
