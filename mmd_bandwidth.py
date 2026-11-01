"""Compute the shared Gaussian ruler after data generation, before experiments.

    python mmd_bandwidth.py --data_dirs data_regime_a_fixed_topology \
        data_regime_b_contingencies --out configs/mmd_bandwidth.json --provisional

All train/test rows contribute with their multiplicity. Validation is excluded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from mmd_utils import (_descriptors, _reference_bandwidth_stats, dataset_fingerprint,
                       pyg_to_networkx, reactance_histogram)
from training_utils import load_grid_dataset

GRIDS = ("IEEE24", "IEEE39", "IEEE118", "UK")
SPLITS = ("train", "test")


def build_reference(data_dirs):
    """Build both graph-mode rulers from the same immutable dataset snapshots."""
    reference = {"multi": {}, "simple": {}, "data_dirs": list(map(str, data_dirs)),
                 "splits": list(SPLITS), "n_descriptors": {}, "fingerprint": {},
                 "zero_fraction": {"multi": {}, "simple": {}}}
    descriptors = {mode: {kind: [] for kind in ("degree", "laplacian")}
                   for mode in ("multi", "simple")}
    electrical = []
    for data_dir in data_dirs:
        for grid in GRIDS:
            for split in SPLITS:
                key = f"{data_dir}/{grid}/{split}"
                path = Path(data_dir) / grid / split / "dataset.pt"
                fingerprint = dataset_fingerprint(path)
                data = load_grid_dataset(data_dir, grid, split)
                reference["fingerprint"][key] = fingerprint
                reference["n_descriptors"][key] = len(data)
                for mode in descriptors:
                    graphs = [pyg_to_networkx(d, mode) for d in data]
                    for kind in descriptors[mode]:
                        descriptors[mode][kind].append(_descriptors(graphs, kind))
                electrical.append(np.vstack([reactance_histogram(d) for d in data]))
                if dataset_fingerprint(path) != fingerprint:
                    raise ValueError(f"Dataset changed while computing reference: {path}")
                print(f"  {key}: {len(data)} descriptors", flush=True)
    for mode in descriptors:
        for kind, sets in descriptors[mode].items():
            width, fraction = _reference_bandwidth_stats(sets)
            reference[mode][kind] = width
            reference["zero_fraction"][mode][kind] = fraction
    width, fraction = _reference_bandwidth_stats(electrical)
    reference["reactance"] = width
    reference["zero_fraction"]["reactance"] = fraction
    reference["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    return reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--out", default="configs/mmd_bandwidth.json")
    p.add_argument("--provisional", action="store_true",
                   help="mark a current-data smoke ruler, not the final P9 ruler")
    args = p.parse_args()
    if len(set(map(lambda d: Path(d).resolve(), args.data_dirs))) != len(args.data_dirs):
        p.error("--data_dirs must not repeat a folder")
    reference = build_reference(args.data_dirs)
    reference["provisional"] = args.provisional
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reference, indent=2) + "\n")
    for mode in ("multi", "simple"):
        print(f"{mode}: degree={reference[mode]['degree']:.17g}, "
              f"laplacian={reference[mode]['laplacian']:.17g}, "
              f"reactance={reference['reactance']:.17g}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
