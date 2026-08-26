"""test_gather_results.py -- checks for per-architecture result consolidation.

Run:  python3 tests/test_gather_results.py

The merged file is what the ranking is computed from, so the checks here are that
a clean merge preserves every row, and that each way of producing a misleading
ranking -- a duplicated architecture, a missing architecture, shards run under
different protocols, or two configurations for one architecture -- is refused.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from gather_results import check_protocol, gather, shard_dirs

FAILURES = []
MODELS_2 = ["gcn", "gat"]
SUMMARY = {"seeds": [0, 100], "epochs": 200, "data_dir": "data_a",
           "batch_size": 32, "batch_size_ood": 96}


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def _rows(model, hidden=128, seeds=(0, 100)):
    return [{"model": model, "grid": "IEEE24", "seed": s, "regime": "A",
             "num_layers": 2, "hidden": hidden, "learning_rate": 0.001,
             "nrmse": 0.01 + s / 1e4, "mse": 1.0, "mae": 0.5} for s in seeds]


def _write(tmp, name, rows, summary=None):
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(d, "within_grid.csv"), index=False)
    with open(os.path.join(d, "summary.json"), "w") as fh:
        json.dump({**SUMMARY, **(summary or {})}, fh)
    return d


def _expect_exit(fn, label):
    try:
        fn()
    except SystemExit as exc:
        check(label, True, str(exc).splitlines()[0][:70])
        return
    check(label, False, "no SystemExit raised")


def test_clean_merge():
    print("\n== clean merge ==")
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "within_gcn", _rows("gcn"))
        _write(tmp, "within_gat", _rows("gat"))
        dirs = shard_dirs([os.path.join(tmp, "within_*")])
        check("both shards found", len(dirs) == 2, str(len(dirs)))
        protocol = check_protocol(dirs)
        check("protocol is reported", protocol["epochs"] == 200, str(protocol))
        df, owners = gather(dirs, "within_grid.csv", MODELS_2)
        check("every row is kept", len(df) == 4, str(len(df)))
        check("each architecture is attributed to its shard",
              set(owners) == set(MODELS_2), str(owners))


def test_duplicate_model_refused():
    print("\n== duplicated architecture ==")
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "a", _rows("gcn"))
        _write(tmp, "b", _rows("gcn"))
        dirs = shard_dirs([os.path.join(tmp, "*")])
        _expect_exit(lambda: gather(dirs, "within_grid.csv", ["gcn"]),
                     "a duplicated architecture is refused")


def test_missing_model_refused():
    print("\n== missing architecture ==")
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "within_gcn", _rows("gcn"))
        dirs = shard_dirs([os.path.join(tmp, "within_*")])
        _expect_exit(lambda: gather(dirs, "within_grid.csv", MODELS_2),
                     "a partial architecture set is refused")


def test_protocol_mismatch_refused():
    print("\n== mismatched protocol ==")
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "a", _rows("gcn"))
        _write(tmp, "b", _rows("gat"), summary={"epochs": 50})
        dirs = shard_dirs([os.path.join(tmp, "*")])
        _expect_exit(lambda: check_protocol(dirs),
                     "shards run under different protocols are refused")

    with tempfile.TemporaryDirectory() as tmp:
        d = _write(tmp, "a", _rows("gcn"))
        os.remove(os.path.join(d, "summary.json"))
        _expect_exit(lambda: check_protocol([d]),
                     "a shard with no summary.json is refused")


def test_two_configs_for_one_model_refused():
    print("\n== two configurations for one architecture ==")
    with tempfile.TemporaryDirectory() as tmp:
        rows = _rows("gcn", hidden=128, seeds=(0,)) + \
            _rows("gcn", hidden=64, seeds=(100,))
        _write(tmp, "a", rows)
        dirs = shard_dirs([os.path.join(tmp, "*")])
        _expect_exit(lambda: gather(dirs, "within_grid.csv", ["gcn"]),
                     "an architecture with two configurations is refused")


def test_seed_shards():
    print("\n== seed shards ==")
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "s0", _rows("gcn", seeds=(0,)), summary={"seeds": [0]})
        _write(tmp, "s100", _rows("gcn", seeds=(100,)), summary={"seeds": [100]})
        dirs = shard_dirs([os.path.join(tmp, "s*")])
        _expect_exit(lambda: check_protocol(dirs),
                     "differing seeds are refused without --seed_shards")
        protocol = check_protocol(dirs, seed_shards=True)
        check("the seed union is reported", protocol["seeds"] == [0, 100],
              str(protocol["seeds"]))
        df, owners = gather(dirs, "within_grid.csv", ["gcn"], seed_shards=True)
        check("both seeds are kept", sorted(df.seed) == [0, 100], str(list(df.seed)))
        check("the architecture is attributed", set(owners) == {"gcn"}, str(owners))

    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "s0", _rows("gcn", seeds=(0,)), summary={"seeds": [0]})
        _write(tmp, "s0b", _rows("gcn", seeds=(0,)), summary={"seeds": [0]})
        dirs = shard_dirs([os.path.join(tmp, "s*")])
        _expect_exit(lambda: gather(dirs, "within_grid.csv", ["gcn"],
                                    seed_shards=True),
                     "a repeated (model, seed) pair is still refused")


def main():
    test_clean_merge()
    test_duplicate_model_refused()
    test_missing_model_refused()
    test_protocol_mismatch_refused()
    test_two_configs_for_one_model_refused()
    test_seed_shards()
    print("\n" + "=" * 50)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
