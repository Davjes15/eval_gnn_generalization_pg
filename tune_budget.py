"""tune_budget.py -- Step 4: select ONE configuration per architecture under an
equal tuning budget.

PURPOSE
    Produce `configs/arch_config.json`: one frozen {num_layers, hidden,
    learning_rate} per architecture, reused unchanged on every grid, in both
    regimes, and for every final seed.

WHY
    The inherited defaults are a mixture of transplanted and pragmatic values --
    GCN/ARMA depth 8 from ENGAGE's single undocumented tuning sentence,
    GAT/GIN/Transformer depth 3 from PowerGraph's search at a DIFFERENT hidden
    width, NNConv depth 2 chosen for runtime. No single procedure produced them,
    so a reviewer has nothing to check. Here every architecture gets the SAME
    search space, the SAME number of candidates, the SAME training procedure and
    the SAME aggregation rule, which is what makes "best vs best" defensible
    (Errica et al., ICLR 2020: equal budget over a shared space).

PROTOCOL
    Search space, identical for all architectures:
        num_layers    in {2, 3, 8}
        hidden        in {32, 64, 128}
        learning_rate in {1e-3, 3e-4}
    Staged budget of 10 candidates each:
        Stage 1: the 9-point num_layers x hidden grid at lr = 1e-3
        Stage 2: the Stage-1 winner re-scored at lr = 3e-4
    Scoring:
        Every candidate is trained on ALL FOUR grids under REGIME A (fixed
        topology), and scored by the MEAN best VALIDATION weighted-MSE across
        grids. Test splits are never touched. Tuning under Regime B would select
        each architecture FOR generalization, which is the quantity under test.
    Tie-break, declared in advance:
        if the top two Stage-1 candidates are within `--tie_pct` (default 5%),
        both are re-run at a second seed and the two-seed mean decides.

OUTPUTS
    results_a/tuning.csv                 every trial (per grid, per seed)
    results_a/tuning_summary.csv         per-candidate score + best/2nd gap
    results_a/tuning_per_grid_argmin.csv per-grid winners (diagnostic only --
                                         they do NOT change the frozen config;
                                         they answer "should you have tuned per
                                         grid?" with evidence)
    configs/arch_config.json             the frozen configuration per model

HOW TO RUN
    python3 tune_budget.py --data_dir data_a --epochs 200 --out results_a
    # resumable: re-running skips trials already present in tuning.csv
    python3 tune_budget.py --data_dir data_a --skip_existing
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import pandas as pd
import torch

from experiments import _build_model, _load_all
from models import MODELS
from training_utils import get_device, make_loaders, train
from transmission_grids import get_transmission_grid_codes

NUM_LAYERS = [2, 3, 8]
HIDDENS = [32, 64, 128]
LEARNING_RATES = [1e-3, 3e-4]
TRIAL_KEY = ["model", "num_layers", "hidden", "learning_rate", "seed", "grid"]


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def _load_previous(path):
    """Previously-completed trials, keyed for resumability."""
    if not os.path.exists(path):
        return {}, []
    df = pd.read_csv(path)
    rows = df.to_dict("records")
    done = {tuple(r[k] for k in TRIAL_KEY): r for r in rows}
    return done, rows


def run_trial(name, cfg, seed, grid, data, device, epochs, batch_size):
    """Train one (architecture, config, seed, grid) and return its VALIDATION loss."""
    torch.manual_seed(seed)
    model = _build_model(name, cfg, device)
    tl, vl = make_loaders(data[grid]["train"], data[grid]["val"],
                          batch_size=batch_size)
    t0 = time.time()
    val_loss = train(model, device, tl, vl, epochs=epochs,
                     learning_rate=cfg["learning_rate"])
    return {"model": name, **cfg, "seed": seed, "grid": grid,
            "val_loss": float(val_loss), "n_params": n_params(model),
            "seconds": round(time.time() - t0, 1)}


def score_candidate(name, cfg, seeds, grids, data, device, epochs, batch_size,
                    done, rows, csv_path):
    """Mean validation loss over all grids (and seeds), training what is missing.

    Every trial is flushed to `csv_path` as soon as it finishes so a long sweep
    can be interrupted and resumed without losing work.
    """
    losses = []
    for seed in seeds:
        for grid in grids:
            key = (name, cfg["num_layers"], cfg["hidden"], cfg["learning_rate"],
                   seed, grid)
            if key in done:
                losses.append(done[key]["val_loss"])
                continue
            rec = run_trial(name, cfg, seed, grid, data, device, epochs, batch_size)
            done[key] = rec
            rows.append(rec)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            losses.append(rec["val_loss"])
            print(f"    {name} L{cfg['num_layers']} h{cfg['hidden']} "
                  f"lr{cfg['learning_rate']:g} s{seed} {grid}: "
                  f"val={rec['val_loss']:.5g} ({rec['seconds']}s)")
    return sum(losses) / len(losses)


def tune_model(name, grids, data, device, args, done, rows, csv_path):
    """Run the staged budget for one architecture. Returns (config, summary rows)."""
    print(f"\n== {name} ==")
    summary = []

    # -- Stage 1: the 9-point depth x width grid at the base learning rate ----
    stage1 = []
    for num_layers, hidden in itertools.product(args.num_layers, args.hidden):
        cfg = {"num_layers": num_layers, "hidden": hidden,
               "learning_rate": LEARNING_RATES[0]}
        score = score_candidate(name, cfg, [args.seed], grids, data, device,
                                args.epochs, args.batch_size, done, rows, csv_path)
        stage1.append({**cfg, "stage": 1, "seeds": str([args.seed]),
                       "mean_val_loss": score})
        print(f"  stage1 L{num_layers} h{hidden}: mean_val={score:.5g}")

    stage1.sort(key=lambda r: r["mean_val_loss"])
    summary += stage1
    best = stage1[0]
    second = stage1[1] if len(stage1) > 1 else None
    seeds = [args.seed]

    # -- declared tie-break: a near-tie is resolved with a second seed --------
    gap = (float("inf") if second is None else
           (second["mean_val_loss"] - best["mean_val_loss"]) / best["mean_val_loss"])
    print(f"  best/2nd gap = {gap:.3%} (tie threshold {args.tie_pct:.1%})")
    if second is not None and gap < args.tie_pct:
        seeds = [args.seed, args.tie_seed]
        print(f"  -> near-tie: re-scoring the top two at seeds {seeds}")
        rescored = []
        for cand in (best, second):
            cfg = {k: cand[k] for k in ("num_layers", "hidden", "learning_rate")}
            score = score_candidate(name, cfg, seeds, grids, data, device,
                                    args.epochs, args.batch_size, done, rows,
                                    csv_path)
            rescored.append({**cfg, "stage": "1-tiebreak", "seeds": str(seeds),
                             "mean_val_loss": score})
            print(f"  tiebreak L{cfg['num_layers']} h{cfg['hidden']}: "
                  f"mean_val={score:.5g}")
        rescored.sort(key=lambda r: r["mean_val_loss"])
        summary += rescored
        best = rescored[0]

    # -- Stage 2: the winner re-scored at the second learning rate -----------
    cfg2 = {"num_layers": best["num_layers"], "hidden": best["hidden"],
            "learning_rate": LEARNING_RATES[1]}
    score2 = score_candidate(name, cfg2, seeds, grids, data, device, args.epochs,
                             args.batch_size, done, rows, csv_path)
    summary.append({**cfg2, "stage": 2, "seeds": str(seeds),
                    "mean_val_loss": score2})
    print(f"  stage2 lr={LEARNING_RATES[1]:g}: mean_val={score2:.5g}")

    # The Stage-1 winner's score must be on the SAME seed set as Stage 2's to be
    # comparable; a tie-break already re-scored it, otherwise it is single-seed.
    winner = best if best["mean_val_loss"] <= score2 else summary[-1]
    chosen = {k: winner[k] for k in ("num_layers", "hidden", "learning_rate")}
    print(f"  SELECTED {name}: {chosen} (mean_val={winner['mean_val_loss']:.5g})")

    for row in summary:
        row.update(model=name, selected=(row is winner))
    return chosen, summary


def per_grid_argmin(rows):
    """Each grid's own best candidate, per architecture (diagnostic).

    Reported to show whether pooled selection agrees with grid-specific
    selection. It does NOT feed the frozen configuration: per-grid configs would
    leave the pooled-grid OOD arm with no defined config, and would confound
    architecture with configuration in the ranking comparison.
    """
    df = pd.DataFrame(rows)
    df = df[df.seed == df.seed.min()]  # one seed, so all candidates are present
    out = []
    for (model, grid), sub in df.groupby(["model", "grid"]):
        best = sub.loc[sub.val_loss.idxmin()]
        out.append({"model": model, "grid": grid,
                    "num_layers": int(best.num_layers), "hidden": int(best.hidden),
                    "learning_rate": float(best.learning_rate),
                    "val_loss": float(best.val_loss)})
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default="data_a",
                   help="Regime A (fixed-topology) data directory")
    p.add_argument("--out", default="results_a")
    p.add_argument("--config_out", default="configs/arch_config.json")
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    p.add_argument("--grids", nargs="+", default=None)
    # The Stage-1 grid can be narrowed to shard one architecture's sweep over
    # several processes (NNConv's per-edge weight generator makes its wide/deep
    # candidates ~an order of magnitude slower than the attention models). Every
    # trial is cached by its full key, so a final unnarrowed run with
    # --skip_existing does the selection from the shards without retraining.
    p.add_argument("--num_layers", type=int, nargs="+", default=NUM_LAYERS,
                   help=f"Stage-1 depths to score (protocol: {NUM_LAYERS})")
    p.add_argument("--hidden", type=int, nargs="+", default=HIDDENS,
                   help=f"Stage-1 widths to score (protocol: {HIDDENS})")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0, help="Stage-1 seed")
    p.add_argument("--tie_seed", type=int, default=100,
                   help="second seed used only to break a near-tie")
    p.add_argument("--tie_pct", type=float, default=0.05,
                   help="relative gap below which the top two are re-scored")
    p.add_argument("--skip_existing", action="store_true",
                   help="reuse trials already recorded in <out>/tuning.csv")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    grids = args.grids or get_transmission_grid_codes()
    device = get_device()
    csv_path = os.path.join(args.out, "tuning.csv")

    done, rows = ({}, [])
    if args.skip_existing:
        done, rows = _load_previous(csv_path)
        print(f"resuming: {len(done)} trial(s) already in {csv_path}")

    print(f"device={device} grids={grids} models={args.models} "
          f"epochs={args.epochs} data_dir={args.data_dir}")
    data = _load_all(args.data_dir, grids)

    configs, summaries = {}, []
    for name in args.models:
        cfg, summary = tune_model(name, grids, data, device, args, done, rows,
                                  csv_path)
        configs[name] = cfg
        summaries += summary

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    summary_df = pd.DataFrame(summaries)
    # Parameter count of every candidate, so an architecture that wins with far
    # more capacity than another is visible rather than hidden.
    counts = pd.DataFrame(rows).groupby(
        ["model", "num_layers", "hidden"])["n_params"].first().reset_index()
    summary_df = summary_df.merge(counts, on=["model", "num_layers", "hidden"],
                                  how="left")
    summary_df.to_csv(os.path.join(args.out, "tuning_summary.csv"), index=False)
    pd.DataFrame(per_grid_argmin(rows)).to_csv(
        os.path.join(args.out, "tuning_per_grid_argmin.csv"), index=False)

    os.makedirs(os.path.dirname(args.config_out) or ".", exist_ok=True)
    payload = {
        "protocol": {
            "regime": "A (fixed topology)", "data_dir": args.data_dir,
            "search_space": {"num_layers": args.num_layers,
                             "hidden": args.hidden,
                             "learning_rate": LEARNING_RATES},
            "candidates_per_model": len(args.num_layers) * len(args.hidden) + 1,
            "selection": "mean best validation weighted-MSE across grids",
            "grids": grids, "epochs": args.epochs,
            "batch_size": args.batch_size,
            "stage1_seed": args.seed, "tie_seed": args.tie_seed,
            "tie_pct": args.tie_pct,
        },
        "configs": configs,
    }
    with open(args.config_out, "w") as fh:
        json.dump(payload, fh, indent=2)

    print("\n== frozen configuration ==")
    print(json.dumps(configs, indent=2))
    print(f"\nwritten: {args.config_out}, {csv_path}, "
          f"{args.out}/tuning_summary.csv, {args.out}/tuning_per_grid_argmin.csv")


if __name__ == "__main__":
    main()
