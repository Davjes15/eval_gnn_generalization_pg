"""gather_results.py -- combine per-architecture experiment shards into the one
results file the analysis scripts read.

WHY
    An architecture's runs are launched as their own process (one core each, and
    NNConv is far slower than the attention models), so each writes its own
    <shard>/within_grid.csv | cross_context.csv | ood.csv. The rank analysis
    needs a single file per arm containing every architecture.

WHAT IS CHECKED
    Merging is refused unless the shards are consistent with a single frozen
    protocol, because a silently mismatched merge is exactly the failure that
    would invalidate the comparison:
      * no architecture may appear in two shards (duplicated runs)
      * every requested architecture must be present (no partial ranking)
      * all shards must agree on seeds, epochs, data_dir and batch size
      * each architecture must carry ONE (num_layers, hidden, learning_rate)

HOW TO RUN
    python3 gather_results.py --shards 'results_a/within_*' --file within_grid.csv \
        --out results_a
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

from models import MODELS

CONFIG_COLS = ["num_layers", "hidden", "learning_rate"]
SHARED_KEYS = ["seeds", "epochs", "data_dir", "batch_size", "batch_size_ood"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", nargs="+", required=True,
                   help="shard directories or globs")
    p.add_argument("--file", required=True,
                   help="file name inside each shard, e.g. within_grid.csv")
    p.add_argument("--out", required=True, help="destination directory")
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                   help="architectures that must all be present")
    return p.parse_args()


def shard_dirs(patterns):
    dirs = []
    for pattern in patterns:
        for match in sorted(glob.glob(pattern)) or [pattern]:
            if os.path.isdir(match):
                dirs.append(match)
    return dirs


def check_protocol(dirs):
    """Each shard's summary.json must describe the same run protocol."""
    reference, ref_dir = None, None
    for d in dirs:
        path = os.path.join(d, "summary.json")
        if not os.path.exists(path):
            raise SystemExit(f"{d} has no summary.json; refusing to merge a shard "
                             "whose protocol cannot be verified")
        with open(path) as fh:
            summary = json.load(fh)
        keys = {k: summary.get(k) for k in SHARED_KEYS}
        if reference is None:
            reference, ref_dir = keys, d
        elif keys != reference:
            differing = {k for k in SHARED_KEYS if keys[k] != reference[k]}
            raise SystemExit(f"{d} disagrees with {ref_dir} on {differing}: "
                             "the shards were not run under one protocol")
    return reference


def gather(dirs, fname, models):
    frames = []
    for d in dirs:
        path = os.path.join(d, fname)
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit(f"no {fname} found in {dirs}")
    df = pd.concat(frames, ignore_index=True)

    owners = {}
    for frame, d in zip(frames, [d for d in dirs
                                 if os.path.exists(os.path.join(d, fname))]):
        for model in frame.model.unique():
            if model in owners:
                raise SystemExit(f"{model} appears in both {owners[model]} and "
                                 f"{d}: duplicated runs must not be merged")
            owners[model] = d

    missing = [m for m in models if m not in owners]
    if missing:
        raise SystemExit(f"no rows for {missing}: a partial set of architectures "
                         "cannot be ranked")

    per_model_cfg = df.groupby("model")[CONFIG_COLS].nunique()
    varying = per_model_cfg[(per_model_cfg > 1).any(axis=1)]
    if len(varying):
        raise SystemExit("more than one configuration per architecture:\n"
                         f"{varying.to_string()}")
    return df, owners


def main():
    args = parse_args()
    dirs = shard_dirs(args.shards)
    protocol = check_protocol(dirs)
    df, owners = gather(dirs, args.file, args.models)

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, args.file)
    df.to_csv(out_csv, index=False)
    print(f"protocol: {protocol}")
    for model, d in sorted(owners.items()):
        print(f"  {model:12s} <- {d}")
    print(f"{len(df)} row(s) from {len(dirs)} shard(s) -> {out_csv}")


if __name__ == "__main__":
    main()
