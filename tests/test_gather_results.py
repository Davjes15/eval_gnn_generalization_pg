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

from gather_results import (check_protocol, gather, shard_dirs,
                            write_summary)

FAILURES = []
MODELS_2 = ["gcn", "gat"]
SUMMARY = {"seeds": [0, 100], "epochs": 200, "data_dir": "data_a",
           "batch_size": 32, "batch_size_ood": 96, "normalize": "pu_zscore"}


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


def test_normalization_mismatch_refused():
    """Two objectives must not be merged into one leaderboard (audit B8)."""
    print("\n== mismatched normalization ==")
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "a", _rows("gcn"))
        _write(tmp, "b", _rows("gat"), summary={"normalize": "none"})
        dirs = shard_dirs([os.path.join(tmp, "*")])
        _expect_exit(lambda: check_protocol(dirs),
                     "shards with different --normalize are refused")


def test_merged_summary_records_provenance():
    """The merged summary must state the objective, the models and their configs."""
    print("\n== merged provenance ==")
    cfg_gcn = {"num_layers": 2, "hidden": 128, "learning_rate": 0.001}
    cfg_gat = {"num_layers": 3, "hidden": 128, "learning_rate": 0.001}
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "within_gcn", _rows("gcn"),
               summary={"models": ["gcn"], "arch_config": {"gcn": cfg_gcn}})
        _write(tmp, "within_gat", _rows("gat"),
               summary={"models": ["gat"], "arch_config": {"gat": cfg_gat}})
        dirs = shard_dirs([os.path.join(tmp, "within_*")])
        protocol = check_protocol(dirs)
        check("the objective is recorded",
              protocol["normalize"] == "pu_zscore", str(protocol.get("normalize")))
        check("the merged architectures are recorded",
              protocol["models"] == ["gat", "gcn"], str(protocol.get("models")))
        check("each architecture's configuration travels with it",
              protocol["arch_config"] == {"gcn": cfg_gcn, "gat": cfg_gat},
              str(protocol.get("arch_config")))
        out = os.path.join(tmp, "merged")
        os.makedirs(out)
        written = write_summary(protocol, dirs, "within_grid.csv", out)
        check("it is written to the merged summary",
              written["normalize"] == "pu_zscore"
              and written["arch_config"]["gat"] == cfg_gat, str(written))

    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "s0", _rows("gcn", seeds=(0,)),
               summary={"seeds": [0], "arch_config": {"gcn": cfg_gcn}})
        _write(tmp, "s100", _rows("gcn", seeds=(100,)),
               summary={"seeds": [100], "arch_config": {"gcn": cfg_gat}})
        dirs = shard_dirs([os.path.join(tmp, "s*")])
        _expect_exit(lambda: check_protocol(dirs, seed_shards=True),
                     "one architecture with two configurations is refused")


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


def _ood_rows(model, folds, seeds):
    return [{"model": model, "held_out_grid": f, "seed": s, "regime": "B",
             "num_layers": 2, "hidden": 128, "learning_rate": 0.001,
             "nrmse": 0.2, "mse": 1.0, "mae": 0.5}
            for f in folds for s in seeds]


def _write_ood(tmp, name, rows, summary=None):
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(d, "ood.csv"), index=False)
    with open(os.path.join(d, "summary.json"), "w") as fh:
        json.dump({**SUMMARY, **(summary or {})}, fh)
    return d


def test_fold_shards():
    """The OOD arm of an expensive architecture is split by held-out grid as
    well as by seed, so the same seed legitimately appears in several shards --
    but only ever once per fold."""
    print("\n== OOD fold shards ==")
    folds = ["IEEE24", "IEEE39", "IEEE118", "UK"]
    with tempfile.TemporaryDirectory() as tmp:
        dirs = [_write_ood(tmp, f"s0_{f}", _ood_rows("nnconv", [f], [0]),
                           summary={"seeds": [0]}) for f in folds]
        dirs += [_write_ood(tmp, f"s100_{f}", _ood_rows("nnconv", [f], [100]),
                            summary={"seeds": [100]}) for f in folds]
        df, owners = gather(dirs, "ood.csv", ["nnconv"], seed_shards=True)
        check("every fold x seed row survives the merge", len(df) == 8,
              str(len(df)))
        check("one seed spread over four fold shards is accepted",
              sorted(df.loc[df.seed == 0, "held_out_grid"]) == sorted(folds))
        check("the architecture is attributed", set(owners) == {"nnconv"})

    with tempfile.TemporaryDirectory() as tmp:
        a = _write_ood(tmp, "a", _ood_rows("nnconv", ["UK"], [0]),
                       summary={"seeds": [0]})
        b = _write_ood(tmp, "b", _ood_rows("nnconv", ["UK"], [0]),
                       summary={"seeds": [0]})
        _expect_exit(lambda: gather([a, b], "ood.csv", ["nnconv"],
                                    seed_shards=True),
                     "the same fold trained twice is still refused")


def test_merged_dir_is_a_valid_shard():
    print("\n== a merged directory can be merged again ==")
    with tempfile.TemporaryDirectory() as tmp:
        s0 = _write(tmp, "s0", _rows("gcn", seeds=(0,)), summary={"seeds": [0]})
        s100 = _write(tmp, "s100", _rows("gcn", seeds=(100,)),
                      summary={"seeds": [100]})
        merged = os.path.join(tmp, "gcn_all")
        os.makedirs(merged)
        df, _ = gather([s0, s100], "within_grid.csv", ["gcn"], seed_shards=True)
        df.to_csv(os.path.join(merged, "within_grid.csv"), index=False)
        protocol = check_protocol([s0, s100], seed_shards=True)
        summary = write_summary(protocol, [s0, s100], "within_grid.csv", merged)
        check("the merged summary carries the seed union",
              summary["seeds"] == [0, 100], str(summary["seeds"]))
        check("the merged summary records its provenance",
              summary["merged_from"] == [s0, s100])

        other = _write(tmp, "gat", _rows("gat"))
        again = check_protocol([merged, other])
        check("the merged directory passes the protocol check as a shard",
              again["seeds"] == [0, 100] and again["epochs"] == 200, str(again))
        df2, owners = gather([merged, other], "within_grid.csv", MODELS_2)
        check("the second-stage merge keeps every row", len(df2) == 4,
              str(len(df2)))
        check("both architectures are attributed", set(owners) == set(MODELS_2),
              str(owners))


def test_seed_shards_across_architectures():
    """NNConv runs three seeds where the others run five, so the final merge of
    the six per-architecture shards is a seed-shard merge too. Two architectures
    sharing a (seed, grid) is the normal case there, not a duplicated run."""
    print("\n== seed shards across architectures ==")
    with tempfile.TemporaryDirectory() as tmp:
        gcn = _write(tmp, "within_gcn", _rows("gcn", seeds=(0, 100)))
        gat = _write(tmp, "within_gat", _rows("gat", seeds=(0,)),
                     summary={"seeds": [0]})
        df, owners = gather([gcn, gat], "within_grid.csv", MODELS_2,
                            seed_shards=True)
        check("architectures sharing a (seed, grid) are not a duplicate",
              len(df) == 3, str(len(df)))
        check("both architectures are attributed", set(owners) == set(MODELS_2),
              str(owners))
        _expect_exit(
            lambda: gather([gcn, gcn], "within_grid.csv", ["gcn"],
                           seed_shards=True),
            "a genuinely repeated run is still refused")


def main():
    test_clean_merge()
    test_duplicate_model_refused()
    test_missing_model_refused()
    test_protocol_mismatch_refused()
    test_normalization_mismatch_refused()
    test_merged_summary_records_provenance()
    test_two_configs_for_one_model_refused()
    test_seed_shards()
    test_fold_shards()
    test_seed_shards_across_architectures()
    test_merged_dir_is_a_valid_shard()
    print("\n" + "=" * 50)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
