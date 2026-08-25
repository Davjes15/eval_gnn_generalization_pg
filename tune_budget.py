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
    Stability requirement (amendment, see below):
        A candidate whose training DIVERGES (non-finite validation loss on any
        grid) is DISQUALIFIED, not merely ranked last; and the leading candidate
        must reproduce at a second seed before it can be frozen. If no candidate
        at lr = 1e-3 survives, the depth x width grid is re-scored at lr = 3e-4
        (the same 18-point space, reached in a different order).
    Scoring:
        Every candidate is trained on ALL FOUR grids under REGIME A (fixed
        topology), and scored by the MEAN best VALIDATION weighted-MSE across
        grids. Test splits are never touched. Tuning under Regime B would select
        each architecture FOR generalization, which is the quantity under test.
    Tie-break, declared in advance:
        if the top two Stage-1 candidates are within `--tie_pct` (default 5%),
        both are re-run at a second seed and the two-seed mean decides.

WHY THE STABILITY REQUIREMENT
    The original rule took the argmin of the mean validation loss at a single
    seed. That crowned ARMA at 8 layers x hidden 128 / lr 1e-3, which trains at
    seed 0 and diverges to NaN at seeds 100, 700 and 1000 -- 4 of 5 final seeds,
    in Regime A as well as in both Regime B arms. The sweep had in fact already
    recorded `inf` for all three hidden-64 ARMA candidates, and the rule scored
    that as "a very bad number" instead of "this candidate failed". A
    configuration that only trains at the seed it was selected on is not a
    configuration; it is a coincidence. The rule is applied identically to all
    six architectures -- it is a no-op for the five that never diverge.

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
import math
import os
import time

import numpy as np
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
    """Score one candidate over all grids (and seeds), training what is missing.

    Returns (mean validation loss, stable), where `stable` is False if training
    diverged on ANY grid or seed. A diverged candidate is disqualified rather
    than ranked last, so its mean is reported as inf regardless of the grids it
    did survive -- averaging a finite grid with a diverged one would otherwise
    let a partially-broken candidate outscore a sound one.

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
    stable = all(math.isfinite(x) for x in losses)
    mean = sum(losses) / len(losses) if stable else float("inf")
    return mean, stable


def _cfg_of(row):
    return {k: row[k] for k in ("num_layers", "hidden", "learning_rate")}


def _stage1_grid(name, lr, stage, grids, data, device, args, done, rows,
                 csv_path, summary):
    """Score the whole depth x width grid at one learning rate, best first."""
    scored = []
    for num_layers, hidden in itertools.product(args.num_layers, args.hidden):
        cfg = {"num_layers": num_layers, "hidden": hidden, "learning_rate": lr}
        score, stable = score_candidate(name, cfg, [args.seed], grids, data,
                                        device, args.epochs, args.batch_size,
                                        done, rows, csv_path)
        scored.append({**cfg, "stage": stage, "seeds": str([args.seed]),
                       "mean_val_loss": score, "stable": stable})
        print(f"  {stage} L{num_layers} h{hidden} lr{lr:g}: mean_val={score:.5g}"
              + ("" if stable else "  DISQUALIFIED (diverged)"))
    scored.sort(key=lambda r: r["mean_val_loss"])
    summary += scored
    return scored


def _confirm(name, scored, stage, grids, data, device, args, done, rows,
             csv_path, summary):
    """Walk the ranked candidates until one reproduces at the second seed.

    A candidate that diverged at the Stage-1 seed is skipped outright; one that
    survived Stage 1 but diverges at the confirmation seed is disqualified here.
    This is what the original rule lacked: it froze whichever config won on a
    single seed, without ever asking whether that win was reproducible.
    """
    seeds = [args.seed, args.tie_seed]
    for cand in scored:
        if not cand["stable"]:
            continue
        cfg = _cfg_of(cand)
        score, stable = score_candidate(name, cfg, seeds, grids, data, device,
                                        args.epochs, args.batch_size, done, rows,
                                        csv_path)
        summary.append({**cfg, "stage": stage, "seeds": str(seeds),
                        "mean_val_loss": score, "stable": stable})
        print(f"  {stage} L{cfg['num_layers']} h{cfg['hidden']} "
              f"lr{cfg['learning_rate']:g}: mean_val={score:.5g}"
              + ("" if stable else "  DISQUALIFIED (diverged at "
                                   f"seed {args.tie_seed})"))
        if stable:
            return summary[-1], seeds
    return None, seeds


def tune_model(name, grids, data, device, args, done, rows, csv_path):
    """Run the staged budget for one architecture. Returns (config, summary rows)."""
    print(f"\n== {name} ==")
    summary = []

    # -- Stage 1: the depth x width grid at the base learning rate, then the
    # confirmation seed. If nothing survives, the same grid is re-scored at the
    # lower learning rate -- the search space is unchanged, only the order in
    # which it is explored.
    best = None
    for lr, stage in zip(LEARNING_RATES, ("1", "1b")):
        scored = _stage1_grid(name, lr, stage, grids, data, device, args, done,
                              rows, csv_path, summary)
        best, seeds = _confirm(name, scored, f"{stage}-confirm", grids, data,
                               device, args, done, rows, csv_path, summary)
        if best is not None:
            break
        print(f"  no candidate at lr={lr:g} survived; falling back to the "
              f"next learning rate")
    if best is None:
        raise SystemExit(
            f"{name}: every candidate in the search space diverged. This is a "
            "finding about the architecture, not a bug -- report it rather than "
            "freezing an unstable configuration."
        )

    # -- declared tie-break: a near-tie is resolved on the confirmation seeds --
    stage1 = [r for r in summary if r["stage"] in ("1", "1b")
              and r["learning_rate"] == best["learning_rate"] and r["stable"]]
    runner_up = next((r for r in stage1 if _cfg_of(r) != _cfg_of(best)), None)
    if runner_up is not None:
        ref = next(r for r in stage1 if _cfg_of(r) == _cfg_of(best))
        gap = ((runner_up["mean_val_loss"] - ref["mean_val_loss"])
               / ref["mean_val_loss"])
        print(f"  best/2nd gap = {gap:.3%} (tie threshold {args.tie_pct:.1%})")
        if gap < args.tie_pct:
            print(f"  -> near-tie: re-scoring the runner-up at seeds {seeds}")
            cfg = _cfg_of(runner_up)
            score, stable = score_candidate(name, cfg, seeds, grids, data, device,
                                            args.epochs, args.batch_size, done,
                                            rows, csv_path)
            summary.append({**cfg, "stage": "1-tiebreak", "seeds": str(seeds),
                            "mean_val_loss": score, "stable": stable})
            print(f"  tiebreak L{cfg['num_layers']} h{cfg['hidden']}: "
                  f"mean_val={score:.5g}")
            if stable and score < best["mean_val_loss"]:
                best = summary[-1]

    # -- Stage 2: the winner re-scored at the other learning rate ------------
    # Skipped when the fallback already searched that learning rate: its whole
    # grid was scored, so there is nothing left to compare against.
    winner = best
    other = [lr for lr in LEARNING_RATES if lr != best["learning_rate"]]
    if other and not any(r["stage"] == "1b" for r in summary):
        cfg2 = {"num_layers": best["num_layers"], "hidden": best["hidden"],
                "learning_rate": other[0]}
        score2, stable2 = score_candidate(name, cfg2, seeds, grids, data, device,
                                          args.epochs, args.batch_size, done,
                                          rows, csv_path)
        summary.append({**cfg2, "stage": 2, "seeds": str(seeds),
                        "mean_val_loss": score2, "stable": stable2})
        print(f"  stage2 lr={other[0]:g}: mean_val={score2:.5g}")
        if stable2 and score2 < best["mean_val_loss"]:
            winner = summary[-1]

    chosen = _cfg_of(winner)
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
    df = df[np.isfinite(df.val_loss)]  # diverged candidates are disqualified
    out = []
    for (model, grid), sub in df.groupby(["model", "grid"]):
        if sub.empty:
            continue
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
            "stability_rule": (
                "a candidate with a non-finite validation loss on any grid is "
                "disqualified; the leading candidate must also reproduce at the "
                "tie seed before it can be frozen; if no candidate survives at "
                "one learning rate the grid is re-scored at the other"
            ),
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
