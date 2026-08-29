"""test_summarize_feasibility.py -- the AC-feasibility table (audits C2, C4, C5).

Run:  python3 tests/test_summarize_feasibility.py    (no pytest, no data needed)

Three things can silently break the table a reader is pointed at, and each is
checked here: the summary must refuse a replay CSV written without
`--feasibility` instead of emitting a table of empty columns (C2); the settings
must keep the Regime B same-grid diagonal apart from the unseen-grid cells (B2);
and the floor/DC reference rows must be appended per data regime, since a
residual quoted as a share of served load means nothing without them (C5).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from summarize_feasibility import COLS, baseline_rows, setting_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILURES = []


def check(ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {detail}")
    if not ok:
        FAILURES.append(detail)


def physics_frame():
    """One row per arm, with the diagonal and an unseen cell of cross-context."""
    rows = [
        {"arm": "within", "model": "gcn", "train_grid": "IEEE24",
         "test_grid": "IEEE24"},
        {"arm": "cross", "model": "gcn", "train_grid": "IEEE24",
         "test_grid": "IEEE24"},
        {"arm": "cross", "model": "gcn", "train_grid": "IEEE24",
         "test_grid": "UK"},
        {"arm": "ood", "model": "gcn", "train_grid": "pooled",
         "test_grid": "UK"},
    ]
    for i, row in enumerate(rows):
        row.update({c: float(i + 1) for c in COLS})
        row["ac_dp_true_max_mw"] = 0.01
    return pd.DataFrame(rows)


def baseline_frame():
    rows = []
    for data_dir in ("data_a", "data_full_v2"):
        for grid, scale in (("IEEE24", 1.0), ("UK", 3.0)):
            for state in ("truth", "dc_pf"):
                row = {"data_dir": data_dir, "grid": grid, "state": state}
                row.update({c: scale for c in COLS})
                rows.append(row)
    return pd.DataFrame(rows)


def test_settings_separate_the_diagonal():
    df = physics_frame()
    got = list(df.apply(setting_of, axis=1))
    check(got == ["within", "cc_diagonal", "cc_unseen", "ood"],
          f"the Regime B diagonal is its own setting, got {got}")


def test_baseline_rows_are_per_regime_and_state():
    out = baseline_rows_from(baseline_frame())
    keys = sorted(zip(out.setting, out.model))
    want = [("regime_a", "dc_pf"), ("regime_a", "floor"),
            ("regime_b", "dc_pf"), ("regime_b", "floor")]
    check(keys == want, f"four reference rows, one per (regime, state): {keys}")
    check(abs(float(out[out.model == "floor"].iloc[0].ac_dp_pct_load) - 2.0)
          < 1e-9, "the reference row averages over grids (mean of 1 and 3)")


def baseline_rows_from(df):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "dc.csv")
        df.to_csv(path, index=False)
        return baseline_rows(path)


def run_cli(physics, baselines, out):
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "summarize_feasibility.py"),
         "--physics", physics, "--baselines", baselines, "--out", out],
        capture_output=True, text=True)


def test_cli_joins_models_and_baselines():
    with tempfile.TemporaryDirectory() as tmp:
        phys = os.path.join(tmp, "physics.csv")
        base = os.path.join(tmp, "dc.csv")
        out = os.path.join(tmp, "table.csv")
        physics_frame().to_csv(phys, index=False)
        baseline_frame().to_csv(base, index=False)
        proc = run_cli(phys, base, out)
        check(proc.returncode == 0, f"the summary runs: {proc.stderr.strip()}")
        table = pd.read_csv(out)
        check(len(table) == 8,
              f"four model settings plus four reference rows, got {len(table)}")
        check(set(table.model) >= {"floor", "dc_pf"},
              "the floor and DC rows are in the committed table")


def test_cli_refuses_a_replay_without_the_ac_columns():
    with tempfile.TemporaryDirectory() as tmp:
        phys = os.path.join(tmp, "physics.csv")
        out = os.path.join(tmp, "table.csv")
        physics_frame().drop(columns=["branch_loading_max_pct"]).to_csv(
            phys, index=False)
        proc = run_cli(phys, os.path.join(tmp, "absent.csv"), out)
        check(proc.returncode != 0 and not os.path.exists(out),
              "a replay CSV written without --feasibility is rejected")
        check("branch_loading_max_pct" in proc.stdout + proc.stderr,
              "the error names the missing column")


def test_cli_without_baselines_still_writes_the_model_rows():
    with tempfile.TemporaryDirectory() as tmp:
        phys = os.path.join(tmp, "physics.csv")
        out = os.path.join(tmp, "table.csv")
        physics_frame().to_csv(phys, index=False)
        proc = run_cli(phys, os.path.join(tmp, "absent.csv"), out)
        table = pd.read_csv(out)
        check(proc.returncode == 0 and len(table) == 4,
              "absent baselines are a note, not a failure")
        check("absent" in proc.stdout, "and the note is printed")


def main():
    for fn in (test_settings_separate_the_diagonal,
               test_baseline_rows_are_per_regime_and_state,
               test_cli_joins_models_and_baselines,
               test_cli_refuses_a_replay_without_the_ac_columns,
               test_cli_without_baselines_still_writes_the_model_rows):
        print(f"{fn.__name__}:")
        fn()
    print("\nALL CHECKS PASSED" if not FAILURES
          else f"\nFAILURES: {FAILURES}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
