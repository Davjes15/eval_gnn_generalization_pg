"""test_gather_trials.py -- checks for sharded-trial consolidation.

Run:  python3 tests/test_gather_trials.py

Consolidation feeds the resumed selection run, so a lost or duplicated trial row
would silently change the frozen configuration: what is checked here is that the
union of shards is preserved exactly, that duplicate keys collapse to one row,
that disagreeing duplicates are reported, and that the output is readable by
tune_budget's resume path.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from gather_trials import gather, shard_paths
from tune_budget import TRIAL_KEY, _load_previous

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def _row(grid, hidden, val_loss, num_layers=2, lr=1e-3):
    return {"model": "nnconv", "num_layers": num_layers, "hidden": hidden,
            "learning_rate": lr, "seed": 0, "grid": grid,
            "val_loss": val_loss, "n_params": 100, "seconds": 1.0}


def _write_shards(tmp, shards):
    for name, rows in shards.items():
        os.makedirs(os.path.join(tmp, name), exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(tmp, name, "tuning.csv"),
                                  index=False)


def test_union_and_dedup():
    print("\n== union / de-duplication ==")
    shards = {
        "IEEE24_narrow": [_row("IEEE24", 32, 0.1), _row("IEEE24", 64, 0.2)],
        "IEEE24_wide": [_row("IEEE24", 128, 0.3)],
        # the same trial appearing in two shards must collapse to one row
        "UK_narrow": [_row("UK", 32, 0.4)],
        "UK_wide": [_row("UK", 32, 0.4), _row("UK", 128, 0.5)],
    }
    with tempfile.TemporaryDirectory() as tmp:
        _write_shards(tmp, shards)
        paths = shard_paths([os.path.join(tmp, "*")])
        check("every shard is found", len(paths) == 4, str(len(paths)))

        df, conflict = gather(paths)
        check("duplicate trial keys collapse to one row", len(df) == 5,
              f"{len(df)} rows")
        check("no key appears twice", not df.duplicated(TRIAL_KEY).any())
        check("no conflict reported for identical duplicates",
              len(conflict) == 0, str(len(conflict)))
        expected = {(r["grid"], r["hidden"])
                    for rows in shards.values() for r in rows}
        got = set(zip(df.grid, df.hidden))
        check("union of shards is preserved", got == expected,
              str(sorted(expected - got)))


def test_conflicting_duplicate_is_reported():
    print("\n== disagreeing duplicates ==")
    with tempfile.TemporaryDirectory() as tmp:
        _write_shards(tmp, {"a": [_row("UK", 32, 0.4)],
                            "b": [_row("UK", 32, 0.9)]})
        df, conflict = gather(shard_paths([os.path.join(tmp, "*")]))
        check("disagreeing duplicate is flagged", len(conflict) == 1,
              str(len(conflict)))
        check("one row is still emitted", len(df) == 1, str(len(df)))


def test_output_is_resumable():
    print("\n== resume compatibility ==")
    with tempfile.TemporaryDirectory() as tmp:
        _write_shards(tmp, {"s1": [_row("IEEE24", 32, 0.1)],
                            "s2": [_row("UK", 32, 0.2, lr=3e-4)]})
        df, _ = gather(shard_paths([os.path.join(tmp, "*")]))
        out = os.path.join(tmp, "merged.csv")
        df.to_csv(out, index=False)
        done, rows = _load_previous(out)
        check("tune_budget can resume from the gathered file",
              len(done) == 2 and len(rows) == 2, f"{len(done)} keys")
        key = ("nnconv", 2, 32, 1e-3, 0, "IEEE24")
        check("trial keys round-trip", key in done, str(list(done)[:1]))


def test_missing_shard_is_skipped():
    print("\n== shards without a tuning.csv ==")
    with tempfile.TemporaryDirectory() as tmp:
        _write_shards(tmp, {"s1": [_row("IEEE24", 32, 0.1)]})
        os.makedirs(os.path.join(tmp, "empty"))
        paths = shard_paths([os.path.join(tmp, "*")])
        check("a shard with no trials is ignored", len(paths) == 1, str(paths))


def main():
    test_union_and_dedup()
    test_conflicting_duplicate_is_reported()
    test_output_is_resumable()
    test_missing_shard_is_skipped()
    print("\n" + "=" * 50)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
