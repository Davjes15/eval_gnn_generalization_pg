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

Two policies come from the fourth audit. A checkpoint whose transfer error is
non-finite produced no answer, so its residuals are not a measurement either:
`feasibility_metrics` averages over buses with `nanmean` and would otherwise
still report a finite number for it. Those rows are dropped under the same
void-the-cell rule the ranking uses (`rank_analysis.py`, audit B5) -- the whole
(arm, model, train grid, seed) group -- and counted in `n_voided`, so the failure
stays visible instead of being averaged in. And because the residual
distribution is heavy-tailed (a single architecture can move an unseen-grid mean
by 2x), the two residuals the text quotes also carry a median and a max; the
median is what should be read as typical.

    python summarize_feasibility.py \
        --physics results_norm/physics/physics_metrics.csv \
        --baselines results_norm/physics/dc_feasibility.csv \
        --out docs/tables/ac_feasibility_norm.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

COLS = ["ac_dp_pct_load", "ac_dq_pct_load", "ac_dp_mean_mw", "ac_dq_mean_mvar",
        "branch_loading_max_pct", "branch_loading_max_pct_true",
        "overload_rate_true", "overload_rate_pred",
        "missed_overload_rate", "false_overload_rate"]

# The baseline states do not depend on a training grid, so they are reported per
# data regime: Regime A test splits (fixed topology) and Regime B (N-k).
REGIME = {"data_a": "regime_a", "data_full_v2": "regime_b"}
STATE = {"truth": "floor", "dc_pf": "dc_pf"}

# Distribution columns for the two residuals the text quotes.
SPREAD = ["ac_dp_pct_load", "ac_dq_pct_load"]
# A voided group is the ranking's unit of failure: (arm, model, train grid, seed).
VOID_KEY = ["arm", "model", "train_grid", "seed"]


def valid_mask(df: pd.DataFrame) -> pd.Series:
    """False for rows in a group where any checkpoint gave a non-finite error."""
    failed = df["nrmse"].replace([np.inf, -np.inf], np.nan).isna()
    return ~failed.groupby([df[k] for k in VOID_KEY]).transform("any")


def aggregate(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Means of every metric, plus median/max of the quoted residuals."""
    grouped = df.groupby(keys)
    out = grouped[COLS].mean()
    for c in SPREAD:
        out[f"{c}_median"] = grouped[c].median()
        out[f"{c}_max"] = grouped[c].max()
    out["n_rows"] = grouped.size()
    return out.reset_index()


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
    out = aggregate(df, ["setting", "model"])
    out["n_voided"] = 0
    return out


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
    valid = valid_mask(df)
    voided = df[~valid].groupby(["setting", "model"]).size().rename("n_voided")
    if len(voided):
        print(f"note: {int(voided.sum())} row(s) voided by a non-finite "
              "transfer error, excluded from the means")
    out = aggregate(df[valid], ["setting", "model"])
    # An outer join keeps a (setting, model) whose every row was voided: it is
    # reported as NaN with its count, the way the ranking lists a voided cell.
    out = out.merge(voided.reset_index(), on=["setting", "model"], how="outer")
    out["n_voided"] = out["n_voided"].fillna(0).astype(int)
    out["n_rows"] = out["n_rows"].fillna(0).astype(int)
    if os.path.exists(args.baselines):
        out = pd.concat([out, baseline_rows(args.baselines)],
                        ignore_index=True)
    else:
        print(f"note: {args.baselines} absent, table has no floor/DC rows")
    value_cols = COLS + [f"{c}_{s}" for c in SPREAD for s in ("median", "max")]
    out[value_cols] = out[value_cols].round(3)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} rows -> {args.out}")
    print("true-state residual floor: "
          f"{df['ac_dp_true_max_mw'].max():.4g} MW")


if __name__ == "__main__":
    main()
