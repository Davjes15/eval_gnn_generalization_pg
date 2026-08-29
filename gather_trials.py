"""gather_trials.py -- collect sharded tuning trials into one architecture's
`tuning.csv` so the staged selection can be finished without retraining.

WHY
    NNConv's per-edge weight generator makes its wide/deep candidates ~an order
    of magnitude slower than the attention models, so its Stage-1 grid was
    sharded over several processes (by grid and by width). Each shard holds a
    SUBSET of the protocol's trials and therefore selected from a subset, which
    is not the protocol. Merging every shard's trial rows into one
    `<out>/tuning.csv` lets the unnarrowed selection run be resumed:

        python3 gather_trials.py --shards 'results_a/nnconv_bygrid/*' \
            --out results_a/nnconv
        python3 tune_budget.py --data_dir data_a --epochs 200 --models nnconv \
            --out results_a/nnconv --skip_existing

    The second command re-scores nothing that is already present: trials are
    cached by their full key, so it only trains what is genuinely missing and
    then applies the pooled staged budget to the complete set.

Duplicate trial keys across shards must agree on the recorded validation loss
(the same key is the same deterministic run); the first occurrence is kept and
any disagreement is reported rather than silently averaged.
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

from tune_budget import TRIAL_KEY


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", nargs="+", required=True,
                   help="shard directories or globs, each holding a tuning.csv")
    p.add_argument("--out", required=True,
                   help="directory whose tuning.csv is written (created if needed)")
    p.add_argument("--rtol", type=float, default=1e-6,
                   help="tolerance when duplicate trial keys disagree")
    return p.parse_args()


def shard_paths(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) or [pattern]
        for match in matches:
            csv = os.path.join(match, "tuning.csv")
            if os.path.exists(csv):
                paths.append(csv)
    return paths


def gather(paths, rtol=1e-6):
    """Concatenated, de-duplicated trials plus the list of disagreeing keys."""
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    spread = df.groupby(TRIAL_KEY)["val_loss"].agg(["min", "max"])
    conflict = spread[(spread["max"] - spread["min"]).abs()
                      > rtol * spread["min"].abs()]
    return df.drop_duplicates(TRIAL_KEY).reset_index(drop=True), conflict


def main():
    args = parse_args()
    paths = shard_paths(args.shards)
    if not paths:
        raise SystemExit(f"no tuning.csv found under {args.shards}")

    df, conflict = gather(paths, args.rtol)
    if len(conflict):
        print(f"[warn] {len(conflict)} trial key(s) recorded different losses in "
              "different shards; keeping the first occurrence:")
        print(conflict.to_string())

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, "tuning.csv")
    df.to_csv(out_csv, index=False)
    print(f"{len(df)} unique trial(s) from {len(paths)} shard(s) -> {out_csv}")
    print(df.groupby(["model", "num_layers", "hidden", "learning_rate"])
            .grid.count().to_string())


if __name__ == "__main__":
    main()
