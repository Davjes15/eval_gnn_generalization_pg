"""eval_checkpoints.py -- physics-aware evaluation replayed from saved models (A3).

WHAT IT IS FOR
    The training campaign writes one checkpoint per (arm, model, grid, seed).
    This script re-loads those weights and re-scores them with the reporting the
    aggregate NRMSE hides (physics_metrics.py): per-quantity error restricted to
    the entries the model genuinely predicts, the p95/p99/max error tails in
    physical units, and the voltage-band violation rates including the missed
    violations. With `--feasibility` it also asks whether the predicted state is
    a valid operating point at all (ac_feasibility.py): the AC P/Q residual on
    the post-contingency admittance matrix and the thermal loading against the
    line ratings. Nothing is trained, so the numbers are by construction the same
    models the published tables describe.

CHECKPOINT NAMING (written by experiments.py)
    within_<model>_<grid>_s<seed>.pt          tested on that grid
    cc_<model>_<train_grid>_s<seed>.pt        tested on EVERY grid
    ood_<model>_heldout_<grid>_s<seed>.pt     tested on the held-out grid

SCALER
    The scaler is re-fitted here exactly as the training run fitted it -- on the
    training split(s) of that arm, never on the evaluation grid -- and the
    statistics are deterministic given the data, so the replay reproduces the
    training-time representation. Predictions are inverted to physical units
    before any metric, and the physical targets are read from `y_raw`.

USAGE
    python eval_checkpoints.py --ckpt_root ckpt_norm --data_a data_a \
        --data_b data_full_v2 --normalize pu_zscore --out results_norm/physics \
        --feasibility
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd
import torch

from ac_feasibility import build_cases, feasibility_metrics
from experiments import _build_model, _load_all, load_arch_config
from models import MODELS
from normalization import Scaler
from physics_metrics import physics_metrics
from training_utils import get_device, nrmse_range

GRIDS = ("IEEE24", "IEEE39", "IEEE118", "UK")
ARMS = ("within", "cross", "ood")

# One pattern per arm; the group is the grid the checkpoint was TRAINED on (for
# ood, the grid held out of training).
PATTERNS = {
    "within": re.compile(r"^within_(?P<model>.+)_(?P<grid>[^_]+)_s(?P<seed>\d+)\.pt$"),
    "cross": re.compile(r"^cc_(?P<model>.+)_(?P<grid>[^_]+)_s(?P<seed>\d+)\.pt$"),
    "ood": re.compile(r"^ood_(?P<model>.+)_heldout_(?P<grid>[^_]+)_s(?P<seed>\d+)\.pt$"),
}


def _parse(fname: str):
    """Return (arm, model, grid, seed) for a checkpoint name, or None."""
    for arm, pattern in PATTERNS.items():
        m = pattern.match(fname)
        if not m:
            continue
        model = m.group("model")
        if model not in MODELS:      # e.g. a model name containing an underscore
            continue                 # that a looser pattern mis-split
        return arm, model, m.group("grid"), int(m.group("seed"))
    return None


def find_checkpoints(root: str):
    """Every parseable checkpoint under `root`, recursively."""
    found = []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            parsed = _parse(f)
            if parsed:
                found.append((os.path.join(dirpath, f), *parsed))
    return found


def _scaler_for(arm, data, grid, normalize):
    """The scaler the training run for this arm would have fitted (train-only)."""
    if arm == "ood":
        train_grids = [g for g in data if g != grid]
        return Scaler.fit([data[g]["train"] for g in train_grids], normalize)
    return Scaler.fit([data[grid]["train"]], normalize)


def _test_sets(arm, data, grid, scaler):
    """{test_grid: scaled test dataset} for this checkpoint's evaluation."""
    if arm == "within":
        targets = [grid]
    elif arm == "ood":
        targets = [grid]              # the held-out grid is the test grid
    else:
        targets = list(data)          # cross-context tests on every grid
    return {g: scaler.transform(data[g]["test"]) for g in targets}


@torch.no_grad()
def _predict(model, device, dataset, batch_size, scaler):
    """(y_true_physical, y_pred_physical, x_physical) concatenated over a dataset."""
    from torch_geometric.loader import DataLoader
    model.eval()
    preds, truths, xs = [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        pred = model(batch).cpu()
        if not scaler.identity:
            pred = scaler.inverse_targets(pred)
            truth = batch.y_raw.cpu()
            x = batch.x.cpu().clone()
            x[:, 3:7] = scaler.inverse_targets(x[:, 3:7])
        else:
            truth, x = batch.y.cpu(), batch.x.cpu()
        preds.append(pred)
        truths.append(truth)
        xs.append(x)
    return torch.cat(truths), torch.cat(preds), torch.cat(xs)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_root", default="ckpt_norm")
    p.add_argument("--data_a", default="data_a", help="data for the within arm")
    p.add_argument("--data_b", default="data_full_v2",
                   help="data for the cross-context and OOD arms")
    p.add_argument("--normalize", default="pu_zscore",
                   help="the mode the checkpoints were trained with")
    p.add_argument("--arch_config", default="configs/arch_config.json")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--grids", nargs="+", default=list(GRIDS))
    p.add_argument("--out", default="results_norm/physics",
                   help="output directory for physics_metrics.csv")
    p.add_argument("--feasibility", action="store_true",
                   help="also score the AC power-balance residual and the line "
                        "loading of the predicted state (audit B1); needs the "
                        "source cases and POWERGRAPH_NODE_DIR to rebuild the "
                        "post-contingency networks")
    p.add_argument("--cases_dir", default=None,
                   help="converted MATPOWER cases (defaults to Step 1's output)")
    args = p.parse_args()

    device = get_device()
    ckpts = [c for c in find_checkpoints(args.ckpt_root) if c[1] in args.arms]
    if not ckpts:
        raise SystemExit(f"no checkpoints for arms {args.arms} under {args.ckpt_root}")
    models_needed = sorted({c[2] for c in ckpts})
    arch_cfg = load_arch_config(args.arch_config, models_needed)

    # Load each dataset directory once; the within arm and the Regime B arms use
    # different data (fixed topology vs N-k contingencies).
    cache = {}
    for arm in args.arms:
        d = args.data_a if arm == "within" else args.data_b
        if d not in cache:
            cache[d] = _load_all(d, args.grids)

    # The post-contingency networks are rebuilt once per (data dir, grid) and
    # shared by every checkpoint evaluated on them; each one costs a power flow
    # per distinct outage set, which would otherwise be paid 336 times over.
    case_cache: dict[tuple[str, str], list] = {}

    def cases_for(data_dir, test_grid):
        key = (data_dir, test_grid)
        if key not in case_cache:
            case_cache[key] = build_cases(
                test_grid, os.path.join(data_dir, test_grid, "test"),
                args.cases_dir)
        return case_cache[key]

    rows = []
    for path, arm, model_name, grid, seed in ckpts:
        data = cache[args.data_a if arm == "within" else args.data_b]
        scaler = _scaler_for(arm, data, grid, args.normalize)
        model = _build_model(model_name, arch_cfg[model_name], device)
        model.load_state_dict(torch.load(path, map_location=device))
        for test_grid, dataset in _test_sets(arm, data, grid, scaler).items():
            y_true, y_pred, x = _predict(model, device, dataset,
                                         args.batch_size, scaler)
            row = {"arm": arm, "model": model_name, "train_grid": grid,
                   "test_grid": test_grid, "seed": seed,
                   "unseen": test_grid != grid if arm != "ood" else True,
                   "normalize": args.normalize,
                   "nrmse": nrmse_range(y_true, y_pred),
                   "checkpoint": os.path.relpath(path)}
            row.update(physics_metrics(y_true, y_pred, x))
            if args.feasibility:
                data_dir = args.data_a if arm == "within" else args.data_b
                cases = cases_for(data_dir, test_grid)
                n_bus = y_true.shape[0] // len(cases)
                row.update(feasibility_metrics(y_true.numpy(), y_pred.numpy(),
                                               cases, n_bus))
            rows.append(row)
            print(f"  [{arm}/{model_name}] train={grid} test={test_grid} s{seed} "
                  f"nrmse={row['nrmse']:.4g} "
                  f"predV={row['pred_nrmse_V']:.4g} maxV={row['max_V']:.4g}"
                  + (f" dP={row['ac_dp_mean_mw']:.4g}MW "
                     f"load={row['line_loading_max_pct']:.4g}%"
                     if args.feasibility else ""),
                  flush=True)

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "physics_metrics.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
