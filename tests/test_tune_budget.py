"""test_tune_budget.py -- checks for the equal-budget tuning sweep.

Run:  python3 tests/test_tune_budget.py

The sweep is long-running and its output freezes the configuration used by every
downstream experiment, so what is checked here is that the budget is equal
across architectures, that selection never touches a test split, that the
declared tie-break fires as specified, and that an interrupted sweep resumes
without retraining.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from torch_geometric.data import Data

from models import MODELS
from tune_budget import (HIDDENS, LEARNING_RATES, NUM_LAYERS, _load_previous,
                         per_grid_argmin, tune_model)

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def _toy_data(grids, n=4, n_bus=5):
    torch.manual_seed(0)
    out = {}
    for g in grids:
        out[g] = {}
        for split in ("train", "val", "test"):
            ds = []
            for _ in range(n):
                ei = torch.tensor([[0, 1, 2, 3, 1, 2, 3, 4],
                                   [1, 2, 3, 4, 0, 1, 2, 3]])
                ds.append(Data(x=torch.randn(n_bus, 7), edge_index=ei,
                               edge_attr=torch.randn(ei.shape[1], 4),
                               y=torch.randn(n_bus, 4)))
            out[g][split] = ds
    return out


def _args(tmp, **over):
    base = dict(epochs=1, batch_size=2, seed=0, tie_seed=100, tie_pct=0.05,
                out=tmp, num_layers=NUM_LAYERS, hidden=HIDDENS)
    base.update(over)
    return argparse.Namespace(**base)


def test_search_space():
    print("\nSearch space and budget")
    check("space is 3 depths x 3 widths x 2 learning rates",
          (NUM_LAYERS, HIDDENS) == ([2, 3, 8], [32, 64, 128])
          and LEARNING_RATES == [1e-3, 3e-4])
    check("budget is 10 candidates per architecture",
          len(NUM_LAYERS) * len(HIDDENS) + 1 == 10)


def test_sweep_and_selection():
    print("\nOne architecture's sweep")
    grids = ["G1", "G2"]
    data = _toy_data(grids)
    # tie_pct=0 disables the tie-break so the plain budget is countable.
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "tuning.csv")
        done, rows = {}, []
        cfg, summary = tune_model("gcn", grids, data, "cpu", _args(tmp, tie_pct=0.0),
                                  done, rows, csv)
        check("config has exactly the three searched hyperparameters",
              set(cfg) == {"num_layers", "hidden", "learning_rate"}, str(cfg))
        check("selected config is inside the search space",
              cfg["num_layers"] in NUM_LAYERS and cfg["hidden"] in HIDDENS
              and cfg["learning_rate"] in LEARNING_RATES)
        check("10 candidates scored", len(summary) == 10, str(len(summary)))
        check("exactly one candidate flagged selected",
              sum(bool(r["selected"]) for r in summary) == 1)
        check("selected candidate has the lowest score on its own seed set",
              min(r["mean_val_loss"] for r in summary)
              == [r for r in summary if r["selected"]][0]["mean_val_loss"])
        check("one trial per candidate x grid", len(rows) == 10 * len(grids),
              str(len(rows)))
        df = pd.read_csv(csv)
        check("every trial is flushed to tuning.csv", len(df) == len(rows))
        check("trial rows record the parameter count",
              (df.n_params > 0).all())
        check("wider candidates have more parameters",
              df[df.hidden == 128].n_params.min() > df[df.hidden == 32].n_params.max())

        argmin = per_grid_argmin(rows)
        check("per-grid argmin has one row per grid", len(argmin) == len(grids),
              str(argmin))

        # Resuming must reuse every recorded trial rather than retrain.
        reloaded, rrows = _load_previous(csv)
        check("resume reads back every trial", len(reloaded) == len(rows))
        n_before = len(rrows)
        tune_model("gcn", grids, data, "cpu", _args(tmp, tie_pct=0.0),
                   reloaded, rrows, csv)
        check("resume adds no new trials", len(rrows) == n_before, str(len(rrows)))


def test_tiebreak():
    print("\nDeclared tie-break")
    grids = ["G1"]
    data = _toy_data(grids)
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "tuning.csv")
        rows = []
        # tie_pct=1.0 makes any gap a "near-tie", so the second seed must appear.
        _, summary = tune_model("gcn", grids, data, "cpu",
                                _args(tmp, tie_pct=1.0), {}, rows, csv)
        stages = [r["stage"] for r in summary]
        check("tie-break rows are recorded", "1-tiebreak" in stages, str(set(stages)))
        check("the tie-break used the second seed",
              any(r["seed"] == 100 for r in rows))
        check("the tie-break re-scored exactly the top two candidates",
              stages.count("1-tiebreak") == 2)
        check("stage 2 is scored on the same seed set as the winner",
              [r for r in summary if r["stage"] == 2][0]["seeds"] == "[0, 100]")


def test_no_test_split_used():
    print("\nSelection never touches a test split")
    grids = ["G1"]
    data = _toy_data(grids)
    # A tripwire: any read of data['G1']['test'] during tuning is a leak.
    class Tripwire(list):
        touched = False

        def __iter__(self):
            Tripwire.touched = True
            return super().__iter__()

        def __len__(self):
            Tripwire.touched = True
            return super().__len__()

    data["G1"]["test"] = Tripwire(data["G1"]["test"])
    with tempfile.TemporaryDirectory() as tmp:
        tune_model("gcn", grids, data, "cpu", _args(tmp, tie_pct=0.0), {}, [],
                   os.path.join(tmp, "tuning.csv"))
    check("the test split was never read", not Tripwire.touched)


def test_equal_budget_across_architectures():
    print("\nEqual budget across architectures")
    grids = ["G1"]
    data = _toy_data(grids)
    # The full 3x3 grid over 6 architectures is too slow for a unit test, so the
    # depth/width lists are narrowed -- the property under test is that every
    # architecture gets the SAME number of candidates, not their values.
    narrowed = dict(num_layers=[2], hidden=[32], tie_pct=0.0)
    counts = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in MODELS:
            rows = []
            _, summary = tune_model(name, grids, data, "cpu",
                                    _args(tmp, **narrowed), {}, rows,
                                    os.path.join(tmp, f"{name}.csv"))
            counts[name] = len(summary)
    check("every architecture scores the same number of candidates",
          len(set(counts.values())) == 1, str(counts))


def main():
    test_search_space()
    test_sweep_and_selection()
    test_tiebreak()
    test_no_test_split_used()
    test_equal_budget_across_architectures()
    print("\n" + "=" * 50)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
