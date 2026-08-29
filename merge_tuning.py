"""merge_tuning.py -- combine per-architecture tuning shards into one sweep.

WHY
    tune_budget.py is run one architecture per process so the sweep can use all
    cores (and so NNConv, which is ~10x slower than the attention models, does
    not serialise everything behind it). Each shard writes its own
    results_a/<model>/{tuning,tuning_summary,tuning_per_grid_argmin}.csv plus its
    own arch_config.json; this merges them into the single published sweep and
    the ONE frozen configuration file the experiments consume.

HOW TO RUN
    python3 merge_tuning.py --shards results_a --out results_a \
        --config_out configs/arch_config.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

from models import MODELS

FILES = ("tuning.csv", "tuning_summary.csv", "tuning_per_grid_argmin.csv")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", default="results_a",
                   help="directory containing one subdirectory per architecture")
    p.add_argument("--out", default="results_a")
    p.add_argument("--config_out", default="configs/arch_config.json")
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    missing = [m for m in args.models
               if not os.path.exists(os.path.join(args.shards, m, "tuning.csv"))]
    if missing:
        raise SystemExit(f"no tuning.csv for {missing} under {args.shards}/ -- "
                         "the sweep is incomplete; a partially-tuned comparison "
                         "must not be merged")

    for fname in FILES:
        parts = [pd.read_csv(p) for p in sorted(
            glob.glob(os.path.join(args.shards, "*", fname)))]
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(os.path.join(args.out, fname), index=False)
        print(f"{fname}: {len(df)} rows from {len(parts)} shard(s)")

    configs, protocol = {}, None
    for model in args.models:
        with open(os.path.join(args.shards, model, "arch_config.json")) as fh:
            payload = json.load(fh)
        configs.update(payload["configs"])
        shard_protocol = payload["protocol"]
        if protocol is None:
            protocol = shard_protocol
        elif protocol != shard_protocol:
            # Every architecture must have been searched under the SAME protocol,
            # otherwise "equal budget" is not what was actually run.
            differing = {k for k in protocol
                         if protocol[k] != shard_protocol.get(k)}
            raise SystemExit(f"shard {model} used a different protocol: {differing}")

    os.makedirs(os.path.dirname(args.config_out) or ".", exist_ok=True)
    with open(args.config_out, "w") as fh:
        json.dump({"protocol": protocol, "configs": configs}, fh, indent=2)

    print("\n== frozen configuration ==")
    summary = pd.read_csv(os.path.join(args.out, "tuning_summary.csv"))
    sel = summary[summary.selected].set_index("model")
    for model in args.models:
        cfg = configs[model]
        row = sel.loc[model]
        print(f"  {model:12s} layers={cfg['num_layers']} hidden={cfg['hidden']:4d} "
              f"lr={cfg['learning_rate']:.0e} params={int(row.n_params):>9,} "
              f"val={row.mean_val_loss:.5g}")
    print(f"\nwritten: {args.config_out}")


if __name__ == "__main__":
    main()
