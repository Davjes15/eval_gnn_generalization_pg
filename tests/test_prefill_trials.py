"""test_prefill_trials.py -- checks for the explicit-trial shard runner.

Run:  python3 tests/test_prefill_trials.py

Prefilled rows are fed straight into the protocol's selection, so what matters
is that the script computes EXACTLY the trials named on the command line, keys
them the way tune_budget does, re-uses cached rows instead of retraining, and
writes a file that tune_budget's resume path and gather_trials both accept.
Training itself is not exercised: `run_trial` is the sweep's own function and is
replaced here with a deterministic stub so the checks stay fast.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import prefill_trials
from gather_trials import gather, shard_paths
from tune_budget import TRIAL_KEY, _load_previous

FAILURES = []
TRAINED = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def _fake_run_trial(name, cfg, seed, grid, data, device, epochs, batch_size):
    """Stand-in for the sweep's trial: records the call, returns its row."""
    TRAINED.append((name, cfg["num_layers"], cfg["hidden"], cfg["learning_rate"],
                    seed, grid))
    return {"model": name, **cfg, "seed": seed, "grid": grid,
            "val_loss": 0.5, "n_params": 123, "seconds": 1.0}


def _run(out, grids, num_layers, hidden, lrs, seeds):
    argv = ["prefill_trials.py", "--model", "nnconv", "--out", out,
            "--grids", *grids,
            "--num_layers", *[str(x) for x in num_layers],
            "--hidden", *[str(x) for x in hidden],
            "--learning_rates", *[str(x) for x in lrs],
            "--seeds", *[str(x) for x in seeds]]
    old_argv = sys.argv
    sys.argv = argv
    try:
        prefill_trials.main()
    finally:
        sys.argv = old_argv


def test_requested_set():
    print("\n== the requested trial set is the cartesian product ==")
    global TRAINED
    TRAINED = []
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "shard")
        _run(out, ["IEEE24", "UK"], [2], [64, 128], [1e-3, 3e-4], [100])
        expected = {("nnconv", 2, h, lr, 100, g)
                    for h in (64, 128) for lr in (1e-3, 3e-4)
                    for g in ("IEEE24", "UK")}
        check("exactly the named trials were trained",
              set(TRAINED) == expected and len(TRAINED) == 8,
              f"{len(TRAINED)} trained")

        df = pd.read_csv(os.path.join(out, "tuning.csv"))
        check("one row per trial", len(df) == 8, str(len(df)))
        check("rows carry the full trial key",
              all(k in df.columns for k in TRIAL_KEY))
        check("no extra configuration crept in",
              set(df.hidden) == {64, 128} and set(df.num_layers) == {2}
              and set(df.seed) == {100})


def test_cached_rows_are_not_retrained():
    print("\n== cached trials are skipped ==")
    global TRAINED
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "shard")
        TRAINED = []
        _run(out, ["IEEE24"], [2], [128], [1e-3], [100])
        check("first pass trains", len(TRAINED) == 1, str(len(TRAINED)))

        TRAINED = []
        _run(out, ["IEEE24", "UK"], [2], [128], [1e-3], [100])
        check("second pass trains only what is missing",
              TRAINED == [("nnconv", 2, 128, 1e-3, 100, "UK")], str(TRAINED))
        df = pd.read_csv(os.path.join(out, "tuning.csv"))
        check("the cached row survives", len(df) == 2, str(len(df)))


def test_output_is_consumable():
    print("\n== output feeds gather_trials and tune_budget's resume ==")
    global TRAINED
    TRAINED = []
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "pf_shard")
        _run(out, ["IEEE24"], [2], [64], [1e-3], [0, 100])
        paths = shard_paths([os.path.join(tmp, "*")])
        check("the shard is discoverable", len(paths) == 1, str(paths))
        df, conflict = gather(paths)
        check("gathered without conflict", len(df) == 2 and len(conflict) == 0)
        done, rows = _load_previous(os.path.join(out, "tuning.csv"))
        check("tune_budget can resume from it",
              len(done) == 2
              and ("nnconv", 2, 64, 1e-3, 0, "IEEE24") in done,
              str(sorted(done)))


def main():
    prefill_trials.run_trial = _fake_run_trial
    prefill_trials._load_all = lambda data_dir, grids: {g: {} for g in grids}
    test_requested_set()
    test_cached_rows_are_not_retrained()
    test_output_is_consumable()
    print("\n" + ("FAILURES: " + ", ".join(FAILURES) if FAILURES else "all checks passed"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
