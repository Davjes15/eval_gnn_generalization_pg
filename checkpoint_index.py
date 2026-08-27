#!/usr/bin/env python3
"""Index a checkpoint tree so a saved model can be located and verified (A4).

Walks a `--ckpt_root` written by `experiments.py --save_models`, parses the arm,
architecture, grid and seed out of each filename, and records the file size,
SHA-256 and parameter count. The result is the lookup table a replicator needs in
order to answer "which file reproduces row X of the results table, and is the copy
I have the same one" without loading the whole tree.

Filename conventions produced by experiments.py:
    within_<model>_<grid>_s<seed>.pt          within-grid arm
    cc_<model>_<train_grid>_s<seed>.pt        cross-context arm (train grid)
    ood_<model>_heldout_<grid>_s<seed>.pt     leave-one-grid-out arm (held-out grid)

Usage:
    python checkpoint_index.py --ckpt_root ckpt_norm --out docs/tables/checkpoint_index.csv
"""
import argparse
import hashlib
import os

import pandas as pd
import torch

ARMS = {"within": "within", "cc": "cross", "ood": "ood"}


def parse_name(name: str):
    """(arm, model, grid, seed) from a checkpoint filename, or None if unknown."""
    stem = name[:-3] if name.endswith(".pt") else name
    parts = stem.split("_")
    if len(parts) < 3 or parts[0] not in ARMS or not parts[-1].startswith("s"):
        return None
    arm, seed = ARMS[parts[0]], parts[-1][1:]
    if not seed.isdigit():
        return None
    middle = parts[1:-1]
    if arm == "ood":
        if "heldout" not in middle:
            return None
        cut = middle.index("heldout")
        model, grid = "_".join(middle[:cut]), "_".join(middle[cut + 1:])
    else:
        model, grid = "_".join(middle[:-1]), middle[-1]
    if not model or not grid:
        return None
    return arm, model, grid, int(seed)


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def n_params(path: str) -> int:
    state = torch.load(path, map_location="cpu", weights_only=True)
    return int(sum(t.numel() for t in state.values() if hasattr(t, "numel")))


def build(ckpt_root: str, with_params: bool = True) -> pd.DataFrame:
    rows, skipped = [], []
    for dirpath, _, files in os.walk(ckpt_root):
        for f in sorted(files):
            if not f.endswith(".pt"):
                continue
            parsed = parse_name(f)
            path = os.path.join(dirpath, f)
            if parsed is None:
                skipped.append(os.path.relpath(path, ckpt_root))
                continue
            arm, model, grid, seed = parsed
            rows.append({
                "arm": arm,
                "model": model,
                # within: the grid trained and tested on; cross: the training
                # grid; ood: the held-out grid (trained on the other three).
                "grid": grid,
                "seed": seed,
                "path": os.path.relpath(path, ckpt_root),
                "bytes": os.path.getsize(path),
                "n_params": n_params(path) if with_params else -1,
                "sha256": sha256(path),
            })
    for s in skipped:
        print(f"  skipped (unparsable name): {s}")
    return pd.DataFrame(rows).sort_values(["arm", "model", "grid", "seed"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_root", default="ckpt_norm")
    p.add_argument("--out", default="docs/tables/checkpoint_index.csv")
    p.add_argument("--no_params", action="store_true",
                   help="skip loading each file (faster, leaves n_params = -1)")
    a = p.parse_args()

    df = build(a.ckpt_root, with_params=not a.no_params)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\n{len(df)} checkpoints -> {a.out}")
    if not df.empty:
        print(df.groupby(["arm", "model"]).size().to_string())
        print(f"\ntotal size: {df.bytes.sum() / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
