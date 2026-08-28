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
        --data_dir data --out results --allow_default_config
    # fixed-topology control arm (Regime A), 5 seeds, tuned configs:
    python3 experiments.py --experiment within --data_dir data_a \
        --arch_config configs/arch_config.json --seeds 0 100 300 700 1000 \
        --regime_tag A --out results_tuned/regime_a
    # generalization arms (Regime B), same frozen configs:
    python3 experiments.py --experiment both --data_dir data_full \
        --arch_config configs/arch_config.json --seeds 0 100 300 700 1000 \
        --regime_tag B --out results_tuned/regime_b

ARCHITECTURE CONFIGURATION
    `--arch_config` is a JSON file {model: {num_layers, hidden, learning_rate}}
    produced by tune_budget.py. It is REQUIRED: the inherited ENGAGE/PowerGraph
    defaults were never selected under any protocol, so falling back to them
    silently would invalidate the architecture comparison. Pass
    `--allow_default_config` to opt into them explicitly (smoke tests only).

BATCH SIZE (ENGAGE section 3.3)
    Batch size is scaled with the training-set size so the arms get comparable
    numbers of optimizer steps per epoch: within-grid and cross-context train on
    one grid (~800 samples, batch 32), OOD trains on three pooled grids (~2400
    samples, batch 96). Without this, OOD would take ~3x more updates per epoch
    and the CC-vs-OOD comparison would be confounded.
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
from normalization import MODES, Scaler
from training_utils import (
    evaluate,
    get_device,
    get_generalization_score,
    load_grid_dataset,
    make_loaders,
    test_dc_pf,
    train,
)
from transmission_grids import get_transmission_grid_codes


def load_arch_config(path, model_names, allow_default=False):
    """Load the frozen per-architecture configuration, or the inherited defaults.

    Returns {model: {num_layers, hidden, learning_rate}}. Missing entries are a
    hard error: a partially-tuned comparison is worse than no comparison.
    """
    if path is None:
        if not allow_default:
            raise SystemExit(
                "--arch_config is required. The inherited ENGAGE/PowerGraph "
                "defaults were not selected under any protocol, so using them "
                "silently would invalidate the architecture comparison. Run "
                "tune_budget.py, or pass --allow_default_config for a smoke test."
            )
        print("[warn] no --arch_config: using the INHERITED (untuned) defaults")
        return {name: {} for name in model_names}

    with open(path) as fh:
        cfg = json.load(fh)
    cfg = cfg.get("configs", cfg)  # accept tune_budget.py's wrapper or a bare map
    missing = [n for n in model_names if n not in cfg]
    if missing:
        raise SystemExit(f"--arch_config {path} has no entry for {missing}")
    known = {"num_layers", "hidden", "learning_rate"}
    out = {}
    for name in model_names:
        unknown = set(cfg[name]) - known
        if unknown:
            raise SystemExit(f"--arch_config {path}: {name} has unknown keys {unknown}")
        out[name] = {k: cfg[name][k] for k in cfg[name]}
    return out


def _build_model(name, cfg, device):
    """Instantiate `name` with its selected depth/width (learning rate is
    consumed by the training loop, not the constructor)."""
    kwargs = {k: v for k, v in cfg.items() if k in ("num_layers", "hidden")}
    return MODELS[name](input_dim=7, **kwargs).to(device)


def _config_columns(name, cfg, scaler=None):
    """Configuration provenance, carried on every result row."""
    out = {"num_layers": cfg.get("num_layers"), "hidden": cfg.get("hidden"),
           "learning_rate": cfg.get("learning_rate", 1e-3)}
    if scaler is not None:
        out["normalize"] = scaler.mode
    return out


def _fit(model, device, train_ds, val_ds, epochs, cfg, batch_size,
         ckpt=None, skip_existing=False):
    """Train, or reuse an existing checkpoint when resuming a sharded run."""
    if skip_existing and ckpt is not None and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"    reusing {ckpt}")
        return model
    tl, vl = make_loaders(train_ds, val_ds, batch_size=batch_size)
    train(model, device, tl, vl, epochs=epochs,
          learning_rate=cfg.get("learning_rate", 1e-3))
    if ckpt is not None:
        torch.save(model.state_dict(), ckpt)
    return model


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


def run_within(data, grids, model_names, device, epochs, seeds, arch_cfg,
               batch_size=32, save_dir=None, skip_existing=False, regime_tag="",
               normalize="none"):
    """Fixed-topology control arm: train and test on the SAME grid.

    This is the PowerGraph-like regime -- no unseen grid, no unseen topology --
    and the reference ranking the generalization arms are compared against.

    The scaler is fitted per grid on that grid's training split (`normalize`,
    see normalization.py); metrics are always reported in physical units.
    """
    records = []
    for name in model_names:
        cfg = arch_cfg[name]
        for grid in grids:
            scaler = Scaler.fit([data[grid]["train"]], normalize)
            splits = {s: scaler.transform(data[grid][s])
                      for s in ("train", "val", "test")}
            for seed in seeds:
                torch.manual_seed(seed)
                model = _build_model(name, cfg, device)
                ckpt = (os.path.join(save_dir, f"within_{name}_{grid}_s{seed}.pt")
                        if save_dir else None)
                _fit(model, device, splits["train"], splits["val"],
                     epochs, cfg, batch_size, ckpt, skip_existing)
                nrmse, _, metrics = evaluate(model, device, splits["test"],
                                             full=True, scaler=scaler)
                records.append({
                    "model": name, "grid": grid, "seed": seed,
                    "regime": regime_tag,
                    **_config_columns(name, cfg, scaler), **metrics,
                })
                print(f"  [{name}] grid={grid} seed={seed} nrmse={nrmse:.4f} "
                      f"mse={metrics['mse']:.4g}")
    return records


def run_cross_context(data, grids, model_names, device, epochs, seeds, arch_cfg,
                      batch_size=32, save_dir=None, skip_existing=False,
                      regime_tag="", normalize="none"):
    """Train on each grid, test on every grid. Returns records + trained matrices.

    If save_dir is given, each trained model's state_dict is written to
    save_dir/cc_<model>_<train_grid>_s<seed>.pt so the exact trained GNNs are
    reusable. The seed is part of the filename so seed replicates cannot
    overwrite one another.

    The scaler is fitted on the TRAINING grid only and applied unchanged to the
    unseen grids -- the deployment-realistic protocol, and the one that does not
    leak the target grid's statistics.
    """
    records = []
    for name in model_names:
        cfg = arch_cfg[name]
        for train_grid in grids:
            scaler = Scaler.fit([data[train_grid]["train"]], normalize)
            train_ds = scaler.transform(data[train_grid]["train"])
            val_ds = scaler.transform(data[train_grid]["val"])
            test_ds = {g: scaler.transform(data[g]["test"]) for g in grids}
            for seed in seeds:
                torch.manual_seed(seed)
                model = _build_model(name, cfg, device)
                ckpt = (os.path.join(save_dir, f"cc_{name}_{train_grid}_s{seed}.pt")
                        if save_dir else None)
                _fit(model, device, train_ds, val_ds, epochs, cfg, batch_size,
                     ckpt, skip_existing)
                for test_grid in grids:
                    nrmse, _, metrics = evaluate(model, device,
                                                 test_ds[test_grid], full=True,
                                                 scaler=scaler)
                    records.append({
                        "model": name, "train_grid": train_grid,
                        "test_grid": test_grid, "unseen": train_grid != test_grid,
                        "seed": seed, "regime": regime_tag,
                        **_config_columns(name, cfg, scaler), **metrics,
                    })
                    print(f"  [{name}] train={train_grid} test={test_grid} "
                          f"seed={seed} nrmse={nrmse:.4f} "
                          f"unseen={train_grid != test_grid}")
    return records


def run_ood(data, grids, model_names, device, epochs, seeds, arch_cfg,
            batch_size=96, save_dir=None, skip_existing=False, regime_tag="",
            normalize="none", held_out=None):
    """Leave-one-grid-out: train on the other grids, test on the held-out grid.

    If save_dir is given, each trained model's state_dict is written to
    save_dir/ood_<model>_heldout_<held>_s<seed>.pt. The default batch size is
    larger than the cross-context one because three grids are pooled here (see
    the module docstring).

    held_out restricts which folds this process runs, for sharding one arm
    across processes; the pooled training set of a fold is unaffected, so a
    fold's result does not depend on how the arm was sharded.
    """
    folds = grids if held_out is None else [g for g in grids if g in held_out]
    records = []
    for name in model_names:
        cfg = arch_cfg[name]
        for held in folds:
            train_grids = [g for g in grids if g != held]
            scaler = Scaler.fit([data[g]["train"] for g in train_grids], normalize)
            train_ds = [d for g in train_grids
                        for d in scaler.transform(data[g]["train"])]
            val_ds = [d for g in train_grids
                      for d in scaler.transform(data[g]["val"])]
            held_ds = scaler.transform(data[held]["test"])
            for seed in seeds:
                torch.manual_seed(seed)
                model = _build_model(name, cfg, device)
                ckpt = (os.path.join(save_dir,
                                     f"ood_{name}_heldout_{held}_s{seed}.pt")
                        if save_dir else None)
                _fit(model, device, train_ds, val_ds, epochs, cfg, batch_size,
                     ckpt, skip_existing)
                nrmse, _, metrics = evaluate(model, device, held_ds, full=True,
                                             scaler=scaler)
                records.append({
                    "model": name, "held_out_grid": held, "seed": seed,
                    "regime": regime_tag,
                    **_config_columns(name, cfg, scaler), **metrics,
                })
                print(f"  [{name}] held_out={held} seed={seed} nrmse={nrmse:.4f}")
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
    every cross-context (unseen) pair. Reproduced here for paper-comparability; the
    per-training-grid table is kept for the source-grid mechanism. NO percentile trim
    is used (bounds=0) so this table is consistent with the OOD g-score table, which
    is also un-trimmed (with only 4 grids the ENGAGE default trim is degenerate; see
    design decision D13). A DC-PF reference row is appended with mmd=0 (so its distance
    term vanishes); note DC-PF's g-score is an artifact (Dmmd=0 + the Q==0 bookkeeping),
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
        mean_n, std_n, mmd_rng, score = get_generalization_score(mmds, nrmses, bounds=0)
        rows.append({"model": name, "n_pairs": len(nrmses),
                     "mean_nrmse": mean_n, "std_nrmse": std_n,
                     "mmd_range": mmd_rng, "g_score": score})
    dc = np.array([r["dc_nrmse"] for r in dc_rows])
    mean_n, std_n, mmd_rng, score = get_generalization_score(
        np.zeros(len(dc)), dc, bounds=0)
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


def per_seed(fn, records, seeds, *args):
    """Apply a g-score function to each seed's records separately.

    Pooling seeds would fold seed variance into the g-score's std term, which is
    meant to capture variation ACROSS TEST GRIDS. So each seed gets its own
    g-score and a `seed` column; aggregation over seeds happens downstream.
    """
    rows = []
    for seed in seeds:
        sub = [r for r in records if r.get("seed") == seed]
        if not sub:
            continue
        rows += [{"seed": seed, **row} for row in fn(sub, *args)]
    return rows


def dc_baseline(data, grids):
    rows = []
    for g in grids:
        _, _, metrics = test_dc_pf(data[g]["test"], full=True)
        rows.append({"grid": g,
                     **{f"dc_{k}": v for k, v in metrics.items()}})
    return rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment",
                   choices=["within", "cross", "ood", "both"], default="both",
                   help="'within' is the fixed-topology control arm (Regime A); "
                        "'both' means cross-context + OOD (Regime B)")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    p.add_argument("--grids", nargs="+", default=None,
                   help="default: all available transmission grids")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=12,
                   help="single-seed shorthand; superseded by --seeds")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="training seeds to replicate over, e.g. 0 100 300 700 1000")
    p.add_argument("--arch_config", default=None,
                   help="JSON of per-architecture {num_layers, hidden, "
                        "learning_rate} from tune_budget.py (required)")
    p.add_argument("--allow_default_config", action="store_true",
                   help="explicitly opt into the inherited untuned defaults")
    p.add_argument("--batch_size", type=int, default=32,
                   help="within-grid and cross-context batch size")
    p.add_argument("--batch_size_ood", type=int, default=96,
                   help="OOD batch size (3 grids pooled -- see module docstring)")
    p.add_argument("--normalize", choices=list(MODES), default="none",
                   help="feature/target scaling (normalization.py). 'none' is "
                        "the raw-unit protocol every existing artifact was "
                        "produced with; 'pu_zscore' is the A2 remediation")
    p.add_argument("--regime_tag", default="",
                   help="label carried on every result row, e.g. A or B")
    p.add_argument("--skip_existing", action="store_true",
                   help="reuse a run's checkpoint instead of retraining it "
                        "(requires --save_models); makes sharded runs resumable")
    p.add_argument("--skip_mmd", action="store_true",
                   help="skip MMD and the g-scores; implied by --experiment within, "
                        "where one topology per grid makes them degenerate")
    p.add_argument("--save_models", default=None,
                   help="directory to write trained model state_dicts (.pt)")
    p.add_argument("--held_out", nargs="+", default=None,
                   help="restrict the OOD arm to these held-out grids, to shard "
                        "one arm across processes; each fold still pools all "
                        "other grids for training, so a fold's numbers do not "
                        "depend on the sharding. Merge the shards with "
                        "gather_results.py")
    p.add_argument("--only_topology", action="store_true",
                   help="write the model-independent tables (MMD matrices, "
                        "pooled OOD distances, DC baseline) and exit without "
                        "training; use it when the training shards ran with "
                        "--skip_mmd, so the analysis still has its topology "
                        "inputs and every shard shares one copy of them")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    grids = args.grids or get_transmission_grid_codes()
    device = get_device()
    seeds = args.seeds if args.seeds else [args.seed]
    arch_cfg = load_arch_config(args.arch_config, args.models,
                                allow_default=args.allow_default_config)
    if args.skip_existing and args.save_models is None:
        raise SystemExit("--skip_existing requires --save_models")
    unknown_folds = set(args.held_out or []) - set(grids)
    if unknown_folds:
        raise SystemExit(f"--held_out names no such grid: {sorted(unknown_folds)}; "
                         f"available: {grids}")
    # With one topology per grid the within-grid MMD is 0 by construction, so
    # the MMD matrix and every g-score built on it are degenerate.
    skip_mmd = args.skip_mmd or args.experiment == "within"
    print(f"device={device} grids={grids} models={args.models} "
          f"epochs={args.epochs} seeds={seeds} regime={args.regime_tag or '-'} "
          f"normalize={args.normalize}")
    print(f"arch_config: {json.dumps(arch_cfg)}")

    save_dir = args.save_models
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    data = _load_all(args.data_dir, grids)
    summary = {"regime_tag": args.regime_tag, "seeds": seeds,
               "experiment": args.experiment, "data_dir": args.data_dir,
               "epochs": args.epochs, "arch_config_path": args.arch_config,
               "arch_config": arch_cfg,
               "batch_size": args.batch_size,
               "batch_size_ood": args.batch_size_ood,
               "normalize": args.normalize,
               "held_out": args.held_out}

    lap_mmd = None
    if skip_mmd:
        print("\n== MMD skipped (degenerate for a fixed-topology regime) ==")
    else:
        print("\n== MMD (topological distance) ==")
        deg_mmd, lap_mmd = _mmd_matrix(data, grids)
        deg_mmd.to_csv(os.path.join(args.out, "mmd_degree.csv"))
        lap_mmd.to_csv(os.path.join(args.out, "mmd_laplacian.csv"))
        print(lap_mmd.round(4).to_string())

    print("\n== DC-PF baseline (per test grid) ==")
    dc_rows = dc_baseline(data, grids)
    pd.DataFrame(dc_rows).to_csv(os.path.join(args.out, "dc_baseline.csv"), index=False)
    print(pd.DataFrame(dc_rows).round(4).to_string(index=False))

    if args.only_topology:
        if lap_mmd is None:
            raise SystemExit("--only_topology needs the MMD matrix; drop "
                             "--skip_mmd and do not use --experiment within")
        ood_dist, _ = ood_distances(data, grids)
        pd.DataFrame(ood_dist).to_csv(
            os.path.join(args.out, "ood_distance.csv"), index=False)
        print("\n-- OOD topological distance (held-out grid -> POOLED "
              "training grids) --")
        print(pd.DataFrame(ood_dist).round(4).to_string(index=False))
        summary["only_topology"] = True
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nTopology tables written to {args.out}/")
        return

    if args.experiment == "within":
        print("\n== Within-grid (fixed-topology control arm) ==")
        wi = run_within(data, grids, args.models, device, args.epochs, seeds,
                        arch_cfg, batch_size=args.batch_size, save_dir=save_dir,
                        skip_existing=args.skip_existing,
                        regime_tag=args.regime_tag, normalize=args.normalize)
        wi_df = pd.DataFrame(wi)
        wi_df.to_csv(os.path.join(args.out, "within_grid.csv"), index=False)
        print(wi_df.round(4).to_string(index=False))
        summary["within_rows"] = len(wi)

    if args.experiment in ("cross", "both"):
        print("\n== Cross-context transfer ==")
        cc = run_cross_context(data, grids, args.models, device, args.epochs,
                               seeds, arch_cfg, batch_size=args.batch_size,
                               save_dir=save_dir,
                               skip_existing=args.skip_existing,
                               regime_tag=args.regime_tag,
                               normalize=args.normalize)
        cc_df = pd.DataFrame(cc)
        cc_df.to_csv(os.path.join(args.out, "cross_context.csv"), index=False)
        summary["cross_context_rows"] = len(cc)
        # Headline NRMSE transfer matrix per model, averaged over seeds.
        for name in args.models:
            mat = cc_df[cc_df.model == name].pivot_table(
                index="train_grid", columns="test_grid", values="nrmse",
                aggfunc="mean")
            mat.to_csv(os.path.join(args.out, f"transfer_matrix_{name}.csv"))

        if lap_mmd is None:
            print("(g-scores skipped: no MMD matrix)")
        else:
            gs = per_seed(compute_gscores, cc, seeds, lap_mmd, args.models, grids)
            pd.DataFrame(gs).to_csv(os.path.join(args.out, "gscore.csv"), index=False)
            print("\n-- g-scores (over unseen grids) --")
            print(pd.DataFrame(gs).round(4).to_string(index=False))
            cc_agg = per_seed(compute_cc_aggregate_gscores, cc, seeds,
                              lap_mmd, dc_rows, args.models, grids)
            pd.DataFrame(cc_agg).to_csv(
                os.path.join(args.out, "gscore_cc_aggregate.csv"), index=False)
            print("\n-- CC g-score (ENGAGE Table-3 format, aggregated per model) --")
            print(pd.DataFrame(cc_agg).round(4).to_string(index=False))

    if args.experiment in ("ood", "both"):
        print("\n== Out-of-distribution (leave-one-grid-out) ==")
        ood = run_ood(data, grids, args.models, device, args.epochs, seeds,
                      arch_cfg, batch_size=args.batch_size_ood,
                      save_dir=save_dir, skip_existing=args.skip_existing,
                      regime_tag=args.regime_tag, normalize=args.normalize,
                      held_out=args.held_out)
        pd.DataFrame(ood).to_csv(os.path.join(args.out, "ood.csv"), index=False)
        print(pd.DataFrame(ood).round(4).to_string(index=False))
        summary["ood_rows"] = len(ood)

        if lap_mmd is None:
            print("(OOD g-scores skipped: no MMD matrix)")
        else:
            ood_dist, pooled_lap = ood_distances(data, grids)
            pd.DataFrame(ood_dist).to_csv(
                os.path.join(args.out, "ood_distance.csv"), index=False)
            print("\n-- OOD topological distance (held-out grid → POOLED training "
                  "grids) --")
            print(pd.DataFrame(ood_dist).round(4).to_string(index=False))
            ood_gs = per_seed(compute_ood_gscores, ood, seeds, pooled_lap,
                              args.models, grids)
            pd.DataFrame(ood_gs).to_csv(
                os.path.join(args.out, "gscore_ood.csv"), index=False)
            print("\n-- OOD g-scores (over held-out grids, no trim) --")
            print(pd.DataFrame(ood_gs).round(4).to_string(index=False))

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {args.out}/")


if __name__ == "__main__":
    main()
