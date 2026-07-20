"""experiments.py -- Step 5: run the generalization experiments.

PURPOSE
    Answer the study's research questions by producing:
      * CROSS-CONTEXT transfer matrix -- train each architecture on one grid,
        test on every grid (the headline "does it transfer to an UNSEEN grid?").
      * OUT-OF-DISTRIBUTION (leave-one-grid-out) -- train on the other grids,
        test on the held-out one.
      * the g-SCORE per (model, train grid) -- NRMSE vs topological distance (MMD).
      * per-quantity errors (P, Q, V, theta) and the DC-PF baseline for every cell.

WHY (design decisions D8 + reporting corrections)
    PowerGraph only ever tests WITHIN a grid. The novel, operationally-meaningful
    result is the degradation from within-grid to unseen-grid/topology, quantified
    consistently and broken out per quantity so voltage magnitude does not flatter
    the numbers.

HOW IT CONNECTS
    data/<CODE>/<split>/dataset.pt (Step 3)  +  MODELS (Step 4)
        -> train (training_utils)  -> evaluate (per-quantity + DC baseline)
        -> evaluate_mmd (mmd_utils)  -> g-score (training_utils)
        -> results/*.csv, results/summary.json

HOW TO RUN
    # quick smoke test (few epochs, two models):
    python3 experiments.py --experiment both --models gcn gat --epochs 20 \
        --data_dir data --out results
    # full run:
    python3 experiments.py --experiment both --data_dir data --out results
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from models import MODELS
from mmd_utils import evaluate_mmd
from training_utils import (
    TARGET_NAMES,
    evaluate,
    get_device,
    get_generalization_score,
    load_grid_dataset,
    make_loaders,
    test_dc_pf,
    train,
)
from transmission_grids import get_transmission_grid_codes


def _load_all(data_dir, grids):
    """Return {grid: {'train':..., 'val':..., 'test':...}} of PyG datasets."""
    out = {}
    for g in grids:
        out[g] = {s: load_grid_dataset(data_dir, g, s) for s in ("train", "val", "test")}
    return out


def _mmd_matrix(data, grids):
    """MMD (degree, laplacian) between every train grid and every test grid."""
    deg = pd.DataFrame(index=grids, columns=grids, dtype=float)
    lap = pd.DataFrame(index=grids, columns=grids, dtype=float)
    for a in grids:
        for b in grids:
            md, ml = evaluate_mmd(data[a]["train"], data[b]["test"])
            deg.loc[a, b], lap.loc[a, b] = md, ml
    return deg, lap


def run_cross_context(data, grids, model_names, device, epochs, seed, save_dir=None):
    """Train on each grid, test on every grid. Returns records + trained matrices.

    If save_dir is given, each trained model's state_dict is written to
    save_dir/cc_<model>_<train_grid>.pt so the exact trained GNNs are reusable.
    """
    records = []
    for name in model_names:
        for train_grid in grids:
            torch.manual_seed(seed)
            model = MODELS[name](input_dim=7).to(device)
            tl, vl = make_loaders(data[train_grid]["train"], data[train_grid]["val"])
            train(model, device, tl, vl, epochs=epochs)
            if save_dir is not None:
                torch.save(model.state_dict(),
                           os.path.join(save_dir, f"cc_{name}_{train_grid}.pt"))
            for test_grid in grids:
                nrmse, per_q = evaluate(model, device, data[test_grid]["test"])
                rec = {
                    "model": name, "train_grid": train_grid, "test_grid": test_grid,
                    "unseen": train_grid != test_grid, "nrmse": nrmse,
                    **{f"nrmse_{q}": per_q[q] for q in TARGET_NAMES},
                }
                records.append(rec)
                print(f"  [{name}] train={train_grid} test={test_grid} "
                      f"nrmse={nrmse:.4f} unseen={train_grid != test_grid}")
    return records


def run_ood(data, grids, model_names, device, epochs, seed, save_dir=None):
    """Leave-one-grid-out: train on the other grids, test on the held-out grid.

    If save_dir is given, each trained model's state_dict is written to
    save_dir/ood_<model>_heldout_<held>.pt.
    """
    records = []
    for name in model_names:
        for held in grids:
            train_grids = [g for g in grids if g != held]
            train_ds = [d for g in train_grids for d in data[g]["train"]]
            val_ds = [d for g in train_grids for d in data[g]["val"]]
            torch.manual_seed(seed)
            model = MODELS[name](input_dim=7).to(device)
            tl, vl = make_loaders(train_ds, val_ds)
            train(model, device, tl, vl, epochs=epochs)
            if save_dir is not None:
                torch.save(model.state_dict(),
                           os.path.join(save_dir, f"ood_{name}_heldout_{held}.pt"))
            nrmse, per_q = evaluate(model, device, data[held]["test"])
            records.append({
                "model": name, "held_out_grid": held, "nrmse": nrmse,
                **{f"nrmse_{q}": per_q[q] for q in TARGET_NAMES},
            })
            print(f"  [{name}] held_out={held} nrmse={nrmse:.4f}")
    return records


def compute_gscores(cc_records, lap_mmd, model_names, grids):
    """g-score per (model, train grid) over the UNSEEN test grids."""
    df = pd.DataFrame(cc_records)
    rows = []
    for name in model_names:
        for train_grid in grids:
            sub = df[(df.model == name) & (df.train_grid == train_grid) & (df.unseen)]
            if sub.empty:
                continue
            nrmses = sub["nrmse"].values
            mmds = np.array([lap_mmd.loc[train_grid, tg] for tg in sub["test_grid"]])
            mean_n, std_n, mmd_rng, score = get_generalization_score(mmds, nrmses)
            rows.append({"model": name, "train_grid": train_grid,
                         "mean_nrmse": mean_n, "std_nrmse": std_n,
                         "mmd_range": mmd_rng, "g_score": score})
    return rows


def compute_cc_aggregate_gscores(cc_records, lap_mmd, dc_rows, model_names, grids):
    """Cross-context g-score in ENGAGE's Table-3 format: ONE aggregated row per model.

    Unlike `compute_gscores` (per training grid), ENGAGE pools ALL train->test pairs
    into a single g-score per model -- `get_generalization_score(mmd, nrmse)` over
    every cross-context (unseen) pair, with the default 2/98 trim. Reproduced here for
    paper-comparability; the per-training-grid table is kept for the source-grid
    mechanism. A DC-PF reference row is appended with mmd=0 (so its distance term
    vanishes); note DC-PF's g-score is an artifact (Dmmd=0 + the Q==0 bookkeeping),
    a reference bar rather than a competitor.
    """
    df = pd.DataFrame(cc_records)
    rows = []
    for name in model_names:
        sub = df[(df.model == name) & (df.unseen)].dropna(subset=["nrmse"])
        if sub.empty:
            continue
        nrmses = sub["nrmse"].values
        mmds = np.array([lap_mmd.loc[r.train_grid, r.test_grid]
                         for _, r in sub.iterrows()])
        mean_n, std_n, mmd_rng, score = get_generalization_score(mmds, nrmses)
        rows.append({"model": name, "n_pairs": len(nrmses),
                     "mean_nrmse": mean_n, "std_nrmse": std_n,
                     "mmd_range": mmd_rng, "g_score": score})
    dc = np.array([r["dc_nrmse"] for r in dc_rows])
    mean_n, std_n, mmd_rng, score = get_generalization_score(np.zeros(len(dc)), dc)
    rows.append({"model": "dc_pf", "n_pairs": len(dc),
                 "mean_nrmse": mean_n, "std_nrmse": std_n,
                 "mmd_range": mmd_rng, "g_score": score})
    return rows


def ood_distances(data, grids):
    """Per held-out grid, its POOLED topological distance to the training grids.

    This is the exact distance the OOD g-score uses. Following ENGAGE's OOD MMD
    (`evaluate_cc_mmd`), the leave-one-grid-out training grids are POOLED into a
    single distribution of graphs and ONE MMD is computed between that pooled
    training distribution and the held-out grid's test split -- i.e.
    MMD(held, A u B u C), NOT a mean of the pairwise MMDs MMD(held, A/B/C).
    Pooling reflects the mixture distribution the model is actually trained on.
    Model-independent (topology only). Returns (rows, pooled_lap) where
    pooled_lap maps held-out grid -> pooled Laplacian-MMD (the g-score x-axis).
    """
    rows = []
    pooled_lap = {}
    for held in grids:
        train_grids = [g for g in grids if g != held]
        pooled_train = [d for g in train_grids for d in data[g]["train"]]
        md, ml = evaluate_mmd(pooled_train, data[held]["test"])
        pooled_lap[held] = float(ml)
        rows.append({"held_out_grid": held,
                     "train_grids": "+".join(train_grids),
                     "mmd_pooled_degree": float(md),
                     "mmd_pooled_laplacian": float(ml)})
    return rows, pooled_lap


def compute_ood_gscores(ood_records, pooled_lap, model_names, grids):
    """OOD g-score per model over the held-out grids.

    Unlike the cross-context g-score (which is per TRAINING grid and has only the
    unseen TEST grids as points), the OOD g-score has ONE point per held-out grid
    -- i.e. as many points as grids -- so it is better-posed at small N. For each
    held-out grid the topological distance is the POOLED Laplacian-MMD from that
    grid to the mixture of its TRAINING grids (ENGAGE-consistent: MMD(held,
    A u B u C), supplied via `pooled_lap`), NOT a mean of pairwise MMDs.

    No percentile trim is used (bounds=0): with only a handful of grids the
    ENGAGE default trim collapses the statistics (see design decision D13).
    NaN NRMSE cells (e.g. a diverged model) are dropped before scoring.
    """
    df = pd.DataFrame(ood_records)
    rows = []
    for name in model_names:
        sub = df[df.model == name]
        nrmses, mmds = [], []
        for _, r in sub.iterrows():
            if not np.isfinite(r["nrmse"]):
                continue
            mmds.append(float(pooled_lap[r["held_out_grid"]]))
            nrmses.append(float(r["nrmse"]))
        if len(nrmses) < 2:
            continue  # need >=2 points for std / mmd_range
        mean_n, std_n, mmd_rng, score = get_generalization_score(
            np.array(mmds), np.array(nrmses), bounds=0)
        rows.append({"model": name, "n_points": len(nrmses),
                     "mean_nrmse": mean_n, "std_nrmse": std_n,
                     "mmd_range": mmd_rng, "g_score": score})
    return rows


def dc_baseline(data, grids):
    rows = []
    for g in grids:
        nrmse, per_q = test_dc_pf(data[g]["test"])
        rows.append({"grid": g, "dc_nrmse": nrmse,
                     **{f"dc_nrmse_{q}": per_q[q] for q in TARGET_NAMES}})
    return rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=["cross", "ood", "both"], default="both")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    p.add_argument("--grids", nargs="+", default=None,
                   help="default: all available transmission grids")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--save_models", default=None,
                   help="directory to write trained model state_dicts (.pt)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    grids = args.grids or get_transmission_grid_codes()
    device = get_device()
    print(f"device={device} grids={grids} models={args.models} epochs={args.epochs}")

    save_dir = args.save_models
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    data = _load_all(args.data_dir, grids)
    summary = {}

    print("\n== MMD (topological distance) ==")
    deg_mmd, lap_mmd = _mmd_matrix(data, grids)
    deg_mmd.to_csv(os.path.join(args.out, "mmd_degree.csv"))
    lap_mmd.to_csv(os.path.join(args.out, "mmd_laplacian.csv"))
    print(lap_mmd.round(4).to_string())

    print("\n== DC-PF baseline (per test grid) ==")
    dc_rows = dc_baseline(data, grids)
    pd.DataFrame(dc_rows).to_csv(os.path.join(args.out, "dc_baseline.csv"), index=False)
    print(pd.DataFrame(dc_rows).round(4).to_string(index=False))

    if args.experiment in ("cross", "both"):
        print("\n== Cross-context transfer ==")
        cc = run_cross_context(data, grids, args.models, device, args.epochs,
                                args.seed, save_dir=save_dir)
        cc_df = pd.DataFrame(cc)
        cc_df.to_csv(os.path.join(args.out, "cross_context.csv"), index=False)
        # headline NRMSE transfer matrix (first model shown; all in the CSV)
        for name in args.models:
            mat = cc_df[cc_df.model == name].pivot(
                index="train_grid", columns="test_grid", values="nrmse")
            mat.to_csv(os.path.join(args.out, f"transfer_matrix_{name}.csv"))
        gs = compute_gscores(cc, lap_mmd, args.models, grids)
        pd.DataFrame(gs).to_csv(os.path.join(args.out, "gscore.csv"), index=False)
        print("\n-- g-scores (over unseen grids) --")
        print(pd.DataFrame(gs).round(4).to_string(index=False))
        cc_agg = compute_cc_aggregate_gscores(cc, lap_mmd, dc_rows, args.models, grids)
        pd.DataFrame(cc_agg).to_csv(
            os.path.join(args.out, "gscore_cc_aggregate.csv"), index=False)
        print("\n-- CC g-score (ENGAGE Table-3 format, aggregated per model) --")
        print(pd.DataFrame(cc_agg).round(4).to_string(index=False))
        summary["cross_context_rows"] = len(cc)

    if args.experiment in ("ood", "both"):
        print("\n== Out-of-distribution (leave-one-grid-out) ==")
        ood = run_ood(data, grids, args.models, device, args.epochs, args.seed,
                      save_dir=save_dir)
        pd.DataFrame(ood).to_csv(os.path.join(args.out, "ood.csv"), index=False)
        print(pd.DataFrame(ood).round(4).to_string(index=False))
        ood_dist, pooled_lap = ood_distances(data, grids)
        pd.DataFrame(ood_dist).to_csv(os.path.join(args.out, "ood_distance.csv"), index=False)
        print("\n-- OOD topological distance (held-out grid → POOLED training grids) --")
        print(pd.DataFrame(ood_dist).round(4).to_string(index=False))
        ood_gs = compute_ood_gscores(ood, pooled_lap, args.models, grids)
        pd.DataFrame(ood_gs).to_csv(os.path.join(args.out, "gscore_ood.csv"), index=False)
        print("\n-- OOD g-scores (over held-out grids, no trim) --")
        print(pd.DataFrame(ood_gs).round(4).to_string(index=False))
        summary["ood_rows"] = len(ood)

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {args.out}/")


if __name__ == "__main__":
    main()
