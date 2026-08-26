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

SEED SHARDS
    One architecture's seeds are sometimes split across processes too (one seed
    each), which is the only way an expensive architecture fits the wall clock
    available. `--seed_shards` accepts that layout: `seeds` may then differ
    between shards and an architecture may appear in several of them, but every
    (model, seed) pair must still occur exactly once, so a duplicated or
    silently overwritten run is still refused.

    The merged directory carries a `summary.json` of the protocol it was verified
    against, so it can itself be a shard of a later merge: an architecture whose
    seeds were sharded is consolidated first, then merged with the others.

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
    p.add_argument("--seed_shards", action="store_true",
                   help="shards carry different seeds of the same architecture; "
                        "uniqueness is then enforced per (model, seed)")
    return p.parse_args()


def shard_dirs(patterns):
    dirs = []
    for pattern in patterns:
        for match in sorted(glob.glob(pattern)) or [pattern]:
            if os.path.isdir(match):
                dirs.append(match)
    return dirs


def check_protocol(dirs, seed_shards=False):
    """Each shard's summary.json must describe the same run protocol.

    With `seed_shards` the seed list is expected to differ, so it is collected
    into a union instead of being compared; everything else still has to match.
    """
    shared = [k for k in SHARED_KEYS if not (seed_shards and k == "seeds")]
    seeds: list = []
    reference, ref_dir = None, None
    for d in dirs:
        path = os.path.join(d, "summary.json")
        if not os.path.exists(path):
            raise SystemExit(f"{d} has no summary.json; refusing to merge a shard "
                             "whose protocol cannot be verified")
        with open(path) as fh:
            summary = json.load(fh)
        keys = {k: summary.get(k) for k in shared}
        for seed in summary.get("seeds") or []:
            if seed not in seeds:
                seeds.append(seed)
        if reference is None:
            reference, ref_dir = keys, d
        elif keys != reference:
            differing = {k for k in shared if keys[k] != reference[k]}
            raise SystemExit(f"{d} disagrees with {ref_dir} on {differing}: "
                             "the shards were not run under one protocol")
    if seed_shards and reference is not None:
        reference = {**reference, "seeds": sorted(seeds)}
    return reference


def gather(dirs, fname, models, seed_shards=False):
    frames = []
    for d in dirs:
        path = os.path.join(d, fname)
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit(f"no {fname} found in {dirs}")
    df = pd.concat(frames, ignore_index=True)

    owners: dict = {}
    claimed: dict = {}
    for frame, d in zip(frames, [d for d in dirs
                                 if os.path.exists(os.path.join(d, fname))]):
        for model in frame.model.unique():
            if seed_shards:
                for seed in frame.loc[frame.model == model, "seed"].unique():
                    key = (model, int(seed))
                    if key in claimed:
                        raise SystemExit(
                            f"{model} seed {seed} appears in both {claimed[key]} "
                            f"and {d}: duplicated runs must not be merged")
                    claimed[key] = d
                owners.setdefault(model, d)
                continue
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


def write_summary(protocol, dirs, fname, out):
    """Record the verified protocol next to the merged file.

    The merged directory must be usable as a shard of a later merge (seeds are
    gathered first, then architectures), which requires the protocol it was
    checked against to travel with it. Only the checked keys are written: the
    per-run fields of one shard's summary do not describe a union. Merging a
    second arm into the same directory adds to the record rather than replacing
    it.
    """
    path = os.path.join(out, "summary.json")
    summary = {**(protocol or {}), "merged_from": dirs, "merged_file": fname}
    if os.path.exists(path):
        with open(path) as fh:
            summary = {**json.load(fh), **summary}
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main():
    args = parse_args()
    dirs = shard_dirs(args.shards)
    protocol = check_protocol(dirs, args.seed_shards)
    df, owners = gather(dirs, args.file, args.models, args.seed_shards)

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, args.file)
    df.to_csv(out_csv, index=False)
    write_summary(protocol, dirs, args.file, args.out)
    print(f"protocol: {protocol}")
    for model, d in sorted(owners.items()):
        print(f"  {model:12s} <- {d}")
    print(f"{len(df)} row(s) from {len(dirs)} shard(s) -> {out_csv}")


if __name__ == "__main__":
    main()
