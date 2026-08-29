"""prefill_trials.py -- compute an explicitly listed set of tuning trials so a
slow architecture's remaining budget can be spread over several processes.

WHY
    `tune_budget.py` decides WHICH trial to run next from the protocol's stages,
    so its only sharding handle is the Stage-1 depth/width grid at the base
    learning rate. The trials that the later stages need -- the second seed of a
    declared near-tie, and the winner at the second learning rate -- therefore
    could only be produced serially inside one process. For NNConv, whose
    per-edge weight generator makes a single IEEE118 trial ~3 h, that is a day
    of wall clock for work that is embarrassingly parallel.

    This script computes exactly the (config, seed, grid) trials named on the
    command line and writes them in `tuning.csv` form, using the same
    `run_trial` as the sweep itself, so the rows are indistinguishable from ones
    the sweep would have produced. `gather_trials.py` then unions the shards and
    `tune_budget.py --skip_existing` applies the staged protocol to the complete
    set without retraining anything.

    It selects nothing: the protocol's aggregation and tie-break stay in
    `tune_budget.py`. Trials that a stage turns out not to need are simply
    unused, and remain in the published sweep.

HOW TO RUN
    python3 prefill_trials.py --model nnconv --grids IEEE118 \
        --num_layers 2 --hidden 128 --learning_rates 1e-3 3e-4 --seeds 100 \
        --data_dir data_a --epochs 200 --out results_a/nnconv_bygrid/pf_118_128
"""
from __future__ import annotations

import argparse
import itertools
import os

import pandas as pd

from experiments import _load_all
from models import MODELS
from training_utils import get_device
from transmission_grids import get_transmission_grid_codes
from tune_budget import TRIAL_KEY, _load_previous, run_trial


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=list(MODELS.keys()))
    p.add_argument("--out", required=True,
                   help="shard directory whose tuning.csv is appended to")
    p.add_argument("--data_dir", default="data_a")
    p.add_argument("--grids", nargs="+", default=None)
    p.add_argument("--num_layers", type=int, nargs="+", required=True)
    p.add_argument("--hidden", type=int, nargs="+", required=True)
    p.add_argument("--learning_rates", type=float, nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def requested(args, grids):
    """Every (config, seed, grid) named on the command line, in run order."""
    out = []
    for num_layers, hidden, lr in itertools.product(
            args.num_layers, args.hidden, args.learning_rates):
        cfg = {"num_layers": num_layers, "hidden": hidden, "learning_rate": lr}
        for seed in args.seeds:
            for grid in grids:
                out.append((cfg, seed, grid))
    return out


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    grids = args.grids or get_transmission_grid_codes()
    csv_path = os.path.join(args.out, "tuning.csv")
    done, rows = _load_previous(csv_path)
    device = get_device()
    data = _load_all(args.data_dir, grids)
    print(f"device={device} model={args.model} grids={grids} "
          f"resuming from {len(done)} trial(s) in {csv_path}")

    todo = requested(args, grids)
    for i, (cfg, seed, grid) in enumerate(todo, 1):
        key = (args.model, cfg["num_layers"], cfg["hidden"],
               cfg["learning_rate"], seed, grid)
        if key in done:
            print(f"[{i}/{len(todo)}] skip (cached) {key}")
            continue
        rec = run_trial(args.model, cfg, seed, grid, data, device, args.epochs,
                        args.batch_size)
        done[key] = rec
        rows.append(rec)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[{i}/{len(todo)}] {args.model} L{cfg['num_layers']} "
              f"h{cfg['hidden']} lr{cfg['learning_rate']:g} s{seed} {grid}: "
              f"val={rec['val_loss']:.5g} ({rec['seconds']}s)", flush=True)

    print(f"{len(rows)} trial(s) in {csv_path}; "
          f"keys are {'/'.join(TRIAL_KEY)}")


if __name__ == "__main__":
    main()
