"""Aggregate the AC-feasibility replay into the table cited by the docs.

The replay (`eval_checkpoints.py --feasibility`) writes one row per checkpoint;
this collapses it to one row per (setting, model), where `setting` separates the
three things that are otherwise conflated: the fixed-topology Regime A arm, the
Regime B *same-grid* diagonal, the Regime B unseen-grid transfer, and the
leave-one-grid-out folds (audit B2).

The two reference rows come from `dc_feasibility.py` and are appended as their
own settings, because a residual expressed as a share of served load is only
readable against them (audit C5): `floor` is the same residual on the labels,
`dc_pf` is the DC baseline the NRMSE tables already compare against. DC assumes
|V| = 1 and Q = 0, so read its ACTIVE residual, not its reactive one.

    python summarize_feasibility.py \
        --physics results_norm/physics/physics_metrics.csv \
        --baselines results_norm/physics/dc_feasibility.csv \
        --out docs/tables/ac_feasibility_norm.csv
"""
import argparse
import os

import pandas as pd

COLS = ["ac_dp_pct_load", "ac_dq_pct_load", "ac_dp_mean_mw", "ac_dq_mean_mvar",
        "branch_loading_max_pct", "branch_loading_max_pct_true",
        "overload_rate_true", "overload_rate_pred",
        "missed_overload_rate", "false_overload_rate"]

# The baseline states do not depend on a training grid, so they are reported per
# data regime: Regime A test splits (fixed topology) and Regime B (N-k).
REGIME = {"data_a": "regime_a", "data_full_v2": "regime_b"}
STATE = {"truth": "floor", "dc_pf": "dc_pf"}


def setting_of(row: pd.Series) -> str:
    if row["arm"] == "within":
        return "within"
    if row["arm"] == "ood":
        return "ood"
    return "cc_diagonal" if row["train_grid"] == row["test_grid"] \
        else "cc_unseen"


def baseline_rows(path: str) -> pd.DataFrame:
    """The floor and DC rows, one per (regime, state), averaged over grids."""
    df = pd.read_csv(path)
    df["setting"] = df["data_dir"].map(REGIME).fillna(df["data_dir"])
    df["model"] = df["state"].map(STATE).fillna(df["state"])
    return df.groupby(["setting", "model"])[COLS].mean().reset_index()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--physics", default="results_norm/physics/"
                                        "physics_metrics.csv")
    p.add_argument("--baselines", default="results_norm/physics/"
                                          "dc_feasibility.csv",
                   help="dc_feasibility.py output; skipped if absent")
    p.add_argument("--out", default="docs/tables/ac_feasibility_norm.csv")
    args = p.parse_args()

    df = pd.read_csv(args.physics)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"{args.physics} lacks {missing}: it was written "
                         "without --feasibility")
    df["setting"] = df.apply(setting_of, axis=1)
    out = df.groupby(["setting", "model"])[COLS].mean().reset_index()
    if os.path.exists(args.baselines):
        out = pd.concat([out, baseline_rows(args.baselines)],
                        ignore_index=True)
    else:
        print(f"note: {args.baselines} absent, table has no floor/DC rows")
    out[COLS] = out[COLS].round(3)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} rows -> {args.out}")
    print("true-state residual floor: "
          f"{df['ac_dp_true_max_mw'].max():.4g} MW")


if __name__ == "__main__":
    main()
