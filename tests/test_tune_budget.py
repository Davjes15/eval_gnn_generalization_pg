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
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from torch_geometric.data import Data

import tune_budget
from models import MODELS
from tune_budget import (HIDDENS, LEARNING_RATES, NUM_LAYERS, _load_previous,
                         merge_config, per_grid_argmin, tune_model)

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


def _distinct_configs(summary):
    """The candidate configurations scored -- re-scoring one at another seed is
    a confirmation, not an extra candidate, so the budget counts configs."""
    return {(r["num_layers"], r["hidden"], r["learning_rate"]) for r in summary}


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
        check("10 candidate configurations scored",
              len(_distinct_configs(summary)) == 10,
              str(len(_distinct_configs(summary))))
        check("exactly one candidate flagged selected",
              sum(bool(r["selected"]) for r in summary) == 1)
        sel = [r for r in summary if r["selected"]][0]
        check("the selected candidate is stable", bool(sel["stable"]))
        check("the selected candidate was confirmed at a second seed",
              sel["seeds"] == "[0, 100]", str(sel["seeds"]))
        check("selected candidate has the lowest score among rows on its seed set",
              sel["mean_val_loss"]
              == min(r["mean_val_loss"] for r in summary
                     if r["seeds"] == sel["seeds"]))
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
        # tie_pct=1.0 makes any gap a "near-tie", so the runner-up must be
        # re-scored on the confirmation seeds too.
        _, summary = tune_model("gcn", grids, data, "cpu",
                                _args(tmp, tie_pct=1.0), {}, rows, csv)
        stages = [r["stage"] for r in summary]
        check("tie-break rows are recorded", "1-tiebreak" in stages, str(set(stages)))
        check("the tie-break used the second seed",
              any(r["seed"] == 100 for r in rows))
        check("the tie-break re-scored one runner-up",
              stages.count("1-tiebreak") == 1)
        check("stage 2 is scored on the confirmation seed set",
              [r for r in summary if r["stage"] == 2][0]["seeds"] == "[0, 100]")


REAL_RUN_TRIAL = tune_budget.run_trial


def _diverging(*keys):
    """Patch run_trial so the named (hidden, seed) trials return a non-finite
    validation loss, imitating ARMA's exploding-gradient behaviour."""
    def fake(name, cfg, seed, grid, data, device, epochs, batch_size):
        rec = REAL_RUN_TRIAL(name, cfg, seed, grid, data, device, epochs,
                             batch_size)
        if (cfg["hidden"], seed) in keys:
            rec["val_loss"] = float("inf")
        return rec

    return fake


def test_divergence_disqualifies():
    print("\nStability requirement")
    grids = ["G1"]
    data = _toy_data(grids)
    narrowed = dict(num_layers=[2], hidden=[32, 64], tie_pct=0.0)

    # (a) a candidate that diverges at the Stage-1 seed can never be frozen.
    tune_budget.run_trial = _diverging((32, 0), (32, 100))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cfg, summary = tune_model("gcn", grids, data, "cpu", _args(tmp, **narrowed),
                                      {}, [], os.path.join(tmp, "t.csv"))
        check("a candidate diverging at the stage-1 seed is not selected",
              cfg["hidden"] == 64, str(cfg))
        check("the diverging candidate is recorded as unstable",
              all(not r["stable"] for r in summary if r["hidden"] == 32))
        check("a diverged candidate scores inf, not a finite mean",
              all(r["mean_val_loss"] == float("inf")
                  for r in summary if r["hidden"] == 32))

        # (b) the ARMA case: finest at the stage-1 seed, diverging at the
        # confirmation seed. The old rule froze this; it must now be rejected.
        tune_budget.run_trial = _diverging((32, 100))
        with tempfile.TemporaryDirectory() as tmp:
            cfg, summary = tune_model("gcn", grids, data, "cpu", _args(tmp, **narrowed),
                                      {}, [], os.path.join(tmp, "t.csv"))
        check("a candidate that only trains at the stage-1 seed is rejected",
              cfg["hidden"] == 64, str(cfg))
        check("its disqualification is recorded at the confirmation stage",
              any(r["stage"] == "1-confirm" and r["hidden"] == 32
                  and not r["stable"] for r in summary))

        # (c) nothing survives at lr 1e-3 -> the grid is re-scored at 3e-4.
        def only_base_lr(name, c, seed, grid, data_, device, epochs, bs):
            rec = REAL_RUN_TRIAL(name, c, seed, grid, data_, device, epochs, bs)
            if c["learning_rate"] == LEARNING_RATES[0]:
                rec["val_loss"] = float("inf")
            return rec

        tune_budget.run_trial = only_base_lr
        with tempfile.TemporaryDirectory() as tmp:
            cfg, summary = tune_model("gcn", grids, data, "cpu", _args(tmp, **narrowed),
                                      {}, [], os.path.join(tmp, "t.csv"))
        check("the fallback learning rate is searched when nothing survives",
              cfg["learning_rate"] == LEARNING_RATES[1], str(cfg))
        check("the fallback grid is recorded as stage 1b",
              any(r["stage"] == "1b" for r in summary), str(set(
                  r["stage"] for r in summary)))
        check("the fallback does not also run stage 2",
              not any(r["stage"] == 2 for r in summary))

        # (d) everything diverges -> refuse to freeze anything.
        tune_budget.run_trial = lambda *a, **k: dict(
            model=a[0], **a[1], seed=a[2], grid=a[3],
            val_loss=float("inf"), n_params=1, seconds=0.0)
        raised = False
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tune_model("gcn", grids, data, "cpu", _args(tmp, **narrowed), {}, [],
                           os.path.join(tmp, "t.csv"))
        except SystemExit:
            raised = True
        check("a fully diverging architecture raises instead of freezing a config",
              raised)
    finally:
        tune_budget.run_trial = REAL_RUN_TRIAL


def test_per_grid_argmin_skips_diverged():
    print("\nPer-grid argmin excludes diverged trials")
    rows = [
        dict(model="m", num_layers=2, hidden=32, learning_rate=1e-3, seed=0,
             grid="G1", val_loss=float("inf"), n_params=1, seconds=0.0),
        dict(model="m", num_layers=8, hidden=64, learning_rate=1e-3, seed=0,
             grid="G1", val_loss=0.5, n_params=2, seconds=0.0),
    ]
    out = per_grid_argmin(rows)
    check("the finite candidate wins even when an inf row is present",
          len(out) == 1 and out[0]["hidden"] == 64, str(out))


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

    # On four random graphs at one epoch some architectures legitimately
    # diverge, which would trigger the fallback search and change the count.
    # The property under test is the SIZE of the budget, so losses are made
    # finite here; divergence handling is covered by its own test.
    def finite(name, cfg, seed, grid, data_, device, epochs, bs):
        rec = REAL_RUN_TRIAL(name, cfg, seed, grid, data_, device, epochs, bs)
        if not math.isfinite(rec["val_loss"]):
            rec["val_loss"] = 1.0
        return rec

    counts = {}
    tune_budget.run_trial = finite
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for name in MODELS:
                rows = []
                _, summary = tune_model(name, grids, data, "cpu",
                                        _args(tmp, **narrowed), {}, rows,
                                        os.path.join(tmp, f"{name}.csv"))
                counts[name] = len(_distinct_configs(summary))
    finally:
        tune_budget.run_trial = REAL_RUN_TRIAL
    check("every architecture scores the same number of candidates",
          len(set(counts.values())) == 1, str(counts))


def test_config_merge_keeps_other_models():
    print("\nOne model's sweep does not drop the others")
    frozen = {
        "protocol": {"epochs": 200, "notes": "arma re-swept"},
        "configs": {"gcn": {"num_layers": 2, "hidden": 128},
                    "nnconv": {"num_layers": 3, "hidden": 32}},
    }
    fresh = {"protocol": {"epochs": 200},
             "configs": {"nnconv": {"num_layers": 2, "hidden": 128}}}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "arch_config.json")
        check("a missing file is written as-is",
              merge_config(fresh, path) == fresh)
        with open(path, "w") as fh:
            json.dump(frozen, fh)
        merged = merge_config(fresh, path)
    check("the untouched model survives",
          merged["configs"]["gcn"] == {"num_layers": 2, "hidden": 128},
          str(merged["configs"]))
    check("the swept model is replaced",
          merged["configs"]["nnconv"] == {"num_layers": 2, "hidden": 128},
          str(merged["configs"]["nnconv"]))
    check("provenance recorded by earlier sweeps survives",
          merged["protocol"]["notes"] == "arma re-swept",
          str(merged["protocol"]))


def main():
    test_search_space()
    test_sweep_and_selection()
    test_tiebreak()
    test_divergence_disqualifies()
    test_per_grid_argmin_skips_diverged()
    test_no_test_split_used()
    test_equal_budget_across_architectures()
    test_config_merge_keeps_other_models()
    print("\n" + "=" * 50)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
