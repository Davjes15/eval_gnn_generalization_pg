"""recompute_tables.py -- rebuild every downstream table from the CONSOLIDATED
six-architecture result CSVs.

WHY
Each architecture's arm ran as its own process, so `experiments.py` wrote the
g-score / DC tables of that shard only -- over the models it happened to
contain. Once the shards are merged (`gather_results.py`), those per-shard
tables are stale: the cross-context aggregate g-score and the OOD g-score are
per-model, but the tables that RANK models must be computed once over all six.
Recomputing here also guarantees every headline number comes from the tuned
configurations (design decision D19); the inherited-config tables in
`full_run/results/` are never an input.

WHAT IT COMPUTES
  gscore.csv               per (seed, model, train grid), unseen test grids
  gscore_cc_aggregate.csv  per (seed, model), all unseen train->test pairs
                           + the DC-PF reference row
  gscore_ood.csv           per (seed, model), one point per held-out grid,
                           pooled Laplacian-MMD as the distance
  per_quantity.csv         mean/sd of nrmse|mse|mae for P, Q, V, theta, per
                           (arm, model) -- the four physical targets reported
                           separately, never as one aggregate
  dc_comparison.csv        per (arm, model, quantity) the GNN error, the DC-PF
                           error on the same grids, and their ratio. The DC
                           table is per ARM, because Regime A and Regime B were
                           generated separately (`data_a` vs `data_full`).

Topology-only inputs (`mmd_laplacian.csv`, `ood_distance.csv`) are
model-independent, so any tuned shard's copy is valid; the script checks the
shards agree before using one. The DC baseline is passed in explicitly instead,
because the per-shard `dc_baseline.csv` files predate the reactive-power fix
(see recompute_dc_baseline.py) and are wrong.

HOW TO RUN
  python3 recompute_tables.py --cross results/regime_b/cross_context.csv \
      --ood results/regime_b/ood.csv --within results/regime_a/within_grid.csv \
      --topology results_tuned/gcn \
      --dc_regime_a results/analysis/dc_baseline_regime_a.csv \
      --dc_regime_b results/analysis/dc_baseline_regime_b.csv \
      --out results/analysis
"""
import argparse
import os

import numpy as np
import pandas as pd

from experiments import (compute_cc_aggregate_gscores, compute_gscores,
                         compute_ood_gscores, per_seed)

QUANTITIES = ["P", "Q", "V", "theta"]
METRICS = ["nrmse", "mse", "mae"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--within", required=True, help="consolidated within_grid.csv")
    p.add_argument("--cross", required=True, help="consolidated cross_context.csv")
    p.add_argument("--ood", required=True, help="consolidated ood.csv")
    p.add_argument("--topology", nargs="+", required=True,
                   help="tuned shard dirs holding the model-independent "
                        "mmd_laplacian.csv / ood_distance.csv / dc_baseline.csv")
    p.add_argument("--dc_regime_a", required=True,
                   help="DC baseline table for the Regime A data (data_a), "
                        "from recompute_dc_baseline.py")
    p.add_argument("--dc_regime_b", required=True,
                   help="DC baseline table for the Regime B data (data_full)")
    p.add_argument("--out", required=True)
    return p.parse_args()


def topology_inputs(dirs):
    """Read the model-independent tables, refusing shards that disagree.

    They are recomputed identically by every arm, so a mismatch means the shards
    came from different data and must not be merged into one table. The shards'
    `dc_baseline.csv` is still read as part of that agreement check, but its
    values are NOT returned for use: they carry the contaminated reactive power
    and are superseded by `--dc_regime_a` / `--dc_regime_b`.
    """
    lap = ood_dist = dc = None
    for d in dirs:
        this_lap = pd.read_csv(os.path.join(d, "mmd_laplacian.csv"), index_col=0)
        this_dc = pd.read_csv(os.path.join(d, "dc_baseline.csv"))
        this_ood = None
        path = os.path.join(d, "ood_distance.csv")
        if os.path.exists(path):
            this_ood = pd.read_csv(path)
        if lap is None:
            lap, dc, ood_dist = this_lap, this_dc, this_ood
            continue
        if not np.allclose(lap.values, this_lap.values, atol=1e-9):
            raise SystemExit(f"mmd_laplacian.csv differs in {d}")
        if not np.allclose(dc["dc_nrmse"], this_dc["dc_nrmse"], atol=1e-9):
            raise SystemExit(f"dc_baseline.csv differs in {d}")
        if this_ood is not None and ood_dist is None:
            ood_dist = this_ood
    if ood_dist is None:
        raise SystemExit("no ood_distance.csv among the topology shards")
    pooled = dict(zip(ood_dist.held_out_grid, ood_dist.mmd_pooled_laplacian))
    return lap, pooled, dc


def per_quantity(frames):
    """Mean and sd of every per-quantity metric, per (arm, model).

    P, Q, V and theta are reported separately because the aggregate NRMSE is a
    weighted mixture of quantities with different units and different masking:
    an architecture can be good at active power and poor at voltage angle, and
    the aggregate hides exactly that.
    """
    rows = []
    for arm, df in frames.items():
        for model, sub in df.groupby("model"):
            for q in QUANTITIES:
                row = {"arm": arm, "model": model, "quantity": q,
                       "n": int(len(sub))}
                for m in METRICS:
                    col = f"{m}_{q}"
                    row[f"{m}_mean"] = float(sub[col].mean())
                    row[f"{m}_sd"] = float(sub[col].std(ddof=1))
                rows.append(row)
    return pd.DataFrame(rows)


def dc_comparison(frames, dc_tables):
    """GNN error vs DC power flow on the same grids, per quantity.

    The DC baseline is per grid, so each arm is compared against the mean DC
    error over the grids that arm evaluates on: the within-grid and OOD arms are
    keyed by one grid per row, the cross-context arm spans all of them. Regime A
    and Regime B use different datasets, so `dc_tables` maps arm -> DC table.

    DC power flow predicts no reactive power, so its Q column is 0 by convention
    (ENGAGE's) and its Q error is the full spread of the AC reactive power. That
    is a real number, not a missing one, and the ratio is reported -- but read it
    as "the GNN against a model that does not attempt Q". The `PVtheta` row is
    the aggregate over the three quantities DC does solve, which is the fairer
    comparison and the one to quote out of distribution. That row is the mean of
    the three per-quantity NRMSEs on BOTH sides: the DC table also carries a
    pooled three-column NRMSE (`dc_nrmse_PVtheta`), but the stored GNN rows have
    no pooled counterpart, so using it here would compare two estimators.
    """
    rows = []
    for arm, df in frames.items():
        dc = dc_tables[arm]
        dc_mean = {q: float(dc[f"dc_nrmse_{q}"].mean()) for q in QUANTITIES}
        dc_mean["aggregate"] = float(dc["dc_nrmse"].mean())
        dc_mean["PVtheta"] = float(
            np.mean([dc_mean[c] for c in ("P", "V", "theta")]))
        for model, sub in df.groupby("model"):
            for q in QUANTITIES + ["aggregate", "PVtheta"]:
                if q == "aggregate":
                    gnn = float(sub["nrmse"].mean())
                elif q == "PVtheta":
                    gnn = float(sub[[f"nrmse_{c}" for c in ("P", "V", "theta")]]
                                .mean(axis=1).mean())
                else:
                    gnn = float(sub[f"nrmse_{q}"].mean())
                ref = dc_mean[q]
                ratio = float("nan") if ref == 0 else gnn / ref
                rows.append({"arm": arm, "model": model, "quantity": q,
                             "gnn_nrmse": gnn, "dc_nrmse": ref,
                             "gnn_over_dc": ratio})
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    within = pd.read_csv(args.within)
    cross = pd.read_csv(args.cross)
    ood = pd.read_csv(args.ood)
    lap, pooled, _ = topology_inputs(args.topology)
    dc_a = pd.read_csv(args.dc_regime_a)
    dc_b = pd.read_csv(args.dc_regime_b)

    models = sorted(set(cross.model) | set(ood.model) | set(within.model))
    grids = list(lap.index)
    print(f"models: {', '.join(models)}")
    print(f"grids:  {', '.join(grids)}")

    cc_records = cross.to_dict("records")
    ood_records = ood.to_dict("records")
    cc_seeds = sorted(cross.seed.unique())
    ood_seeds = sorted(ood.seed.unique())

    gs = per_seed(compute_gscores, cc_records, cc_seeds, lap, models, grids)
    pd.DataFrame(gs).to_csv(os.path.join(args.out, "gscore.csv"), index=False)

    dc_rows = dc_b.to_dict("records")
    agg = per_seed(compute_cc_aggregate_gscores, cc_records, cc_seeds,
                   lap, dc_rows, models, grids)
    agg = pd.DataFrame(agg)
    agg.to_csv(os.path.join(args.out, "gscore_cc_aggregate.csv"), index=False)

    ood_gs = per_seed(compute_ood_gscores, ood_records, ood_seeds,
                      pooled, models, grids)
    ood_gs = pd.DataFrame(ood_gs)
    ood_gs.to_csv(os.path.join(args.out, "gscore_ood.csv"), index=False)

    frames = {"regime_a": within, "cross_context": cross, "ood": ood}
    per_quantity(frames).to_csv(
        os.path.join(args.out, "per_quantity.csv"), index=False)
    dc_tables = {"regime_a": dc_a, "cross_context": dc_b, "ood": dc_b}
    dc_comparison(frames, dc_tables).to_csv(
        os.path.join(args.out, "dc_comparison.csv"), index=False)

    pd.set_option("display.width", 150)
    print("\ncross-context aggregate g-score (mean over seeds)")
    print(agg.groupby("model")[["mean_nrmse", "std_nrmse", "g_score"]]
          .mean().sort_values("g_score").to_string())
    print("\nOOD g-score (mean over seeds)")
    print(ood_gs.groupby("model")[["n_points", "mean_nrmse", "g_score"]]
          .mean().sort_values("g_score").to_string())
    print(f"\ntables -> {args.out}")


if __name__ == "__main__":
    main()
