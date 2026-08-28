"""Aggregate the AC-feasibility replay into the table cited by the docs.

The replay (`eval_checkpoints.py --feasibility`) writes one row per checkpoint;
this collapses it to one row per (setting, model), where `setting` separates the
three things that are otherwise conflated: the fixed-topology Regime A arm, the
Regime B *same-grid* diagonal, the Regime B unseen-grid transfer, and the
leave-one-grid-out folds (audit B2).

    python summarize_feasibility.py \
        --physics results_norm/physics/physics_metrics.csv \
        --out docs/tables/ac_feasibility_norm.csv
"""
import argparse

import pandas as pd

COLS = ["ac_dp_pct_load", "ac_dq_pct_load", "ac_dp_mean_mw", "ac_dq_mean_mvar",
        "line_loading_max_pct", "line_loading_max_pct_true",
        "overload_rate_true", "overload_rate_pred",
        "missed_overload_rate", "false_overload_rate"]


def setting_of(row: pd.Series) -> str:
    if row["arm"] == "within":
        return "within"
    if row["arm"] == "ood":
        return "ood"
    return "cc_diagonal" if row["train_grid"] == row["test_grid"] \
        else "cc_unseen"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--physics", default="results_norm/physics/"
                                        "physics_metrics.csv")
    p.add_argument("--out", default="docs/tables/ac_feasibility_norm.csv")
    args = p.parse_args()

    df = pd.read_csv(args.physics)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"{args.physics} lacks {missing}: it was written "
                         "without --feasibility")
    df["setting"] = df.apply(setting_of, axis=1)
    out = (df.groupby(["setting", "model"])[COLS].mean().round(3)
             .reset_index())
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} rows -> {args.out}")
    print("true-state residual floor: "
          f"{df['ac_dp_true_max_mw'].max():.4g} MW")


if __name__ == "__main__":
    main()
