"""test_split_hygiene.py -- checks for the A5 blocked temporal split.

Run:  python3 tests/test_split_hygiene.py

Two things are verified without touching pandapower (so the test is fast):
  * `blocked_time_ranges` returns contiguous, gap-separated, mutually exclusive
    windows wide enough for their split, and refuses impossible requests;
  * `validate.gate_split_hygiene` fails on a dataset whose splits share a demand
    snapshot or repeat a scenario, and passes on a clean blocked one -- i.e. the
    gate would have caught the original Regime B data.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from transmission_graph_gen import blocked_time_ranges
from validate import Gate, gate_split_hygiene

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def _write_dataset(root, code, rows):
    """rows: {split: [(t_idx, out_lines), ...]} -> dataset_src.csv files."""
    for split, pairs in rows.items():
        d = os.path.join(root, code, split)
        os.makedirs(d, exist_ok=True)
        pd.DataFrame([{"grid": code, "t_idx": t, "k": len(o), "out_lines": str(o),
                       "source": "random"} for t, o in pairs]).to_csv(
            os.path.join(d, "dataset_src.csv"), index=False)


def test_ranges():
    print("blocked_time_ranges")
    counts = {"train": 800, "val": 100, "test": 100}
    w = blocked_time_ranges(35040, counts, gap=96)
    check("one window per split", set(w) == set(counts))
    check("windows are wide enough for their split",
          all(w[s][1] - w[s][0] >= counts[s] for s in counts),
          str({s: w[s][1] - w[s][0] for s in counts}))
    ordered = sorted(w.values())
    check("windows do not overlap",
          all(ordered[i][1] <= ordered[i + 1][0] for i in range(len(ordered) - 1)))
    check("windows are separated by the gap",
          all(ordered[i + 1][0] - ordered[i][1] == 96
              for i in range(len(ordered) - 1)))
    check("windows stay inside the time axis", ordered[-1][1] <= 35040)
    check("proportional to sample counts",
          (w["train"][1] - w["train"][0]) > 4 * (w["test"][1] - w["test"][0]))

    too_small = False
    try:
        blocked_time_ranges(500, counts, gap=96)
    except ValueError:
        too_small = True
    check("refuses a demand axis that cannot supply distinct snapshots", too_small)

    single = blocked_time_ranges(1000, {"train": 10, "val": 0, "test": 5}, gap=10)
    check("empty splits are skipped", set(single) == {"train", "test"})


def test_gate(tmp="/tmp/_hygiene_test"):
    print("\ngate_split_hygiene")
    os.makedirs(tmp, exist_ok=True)

    clean = os.path.join(tmp, "clean")
    _write_dataset(clean, "IEEE24", {
        "train": [(t, [1]) for t in range(0, 10)],
        "val": [(t, [2]) for t in range(200, 205)],
        "test": [(t, [3]) for t in range(400, 405)],
    })
    g = Gate()
    gate_split_hygiene(g, clean, ["IEEE24"], expect_blocked=True)
    check("clean blocked dataset passes", not g.failures, str(g.failures))

    shared = os.path.join(tmp, "shared")
    _write_dataset(shared, "IEEE24", {
        "train": [(t, [1]) for t in range(0, 10)],
        "val": [(t, [2]) for t in range(200, 205)],
        "test": [(t, [3]) for t in range(5, 10)],
    })
    g = Gate()
    gate_split_hygiene(g, shared, ["IEEE24"], expect_blocked=True)
    check("a shared demand snapshot is caught",
          any("share no demand snapshot" in f for f in g.failures), str(g.failures))
    check("non-blocked windows are caught",
          any("time windows are disjoint" in f for f in g.failures))

    dup = os.path.join(tmp, "dup")
    _write_dataset(dup, "IEEE24", {
        "train": [(0, [1]), (0, [1]), (1, [2])],
        "val": [(200, [2])],
        "test": [(400, [3])],
    })
    g = Gate()
    gate_split_hygiene(g, dup, ["IEEE24"], expect_blocked=False)
    check("a repeated scenario inside a split is caught",
          any("no repeated scenario" in f for f in g.failures), str(g.failures))

    # A repeated snapshot with a *different* outage set is legal when topology
    # varies -- it is a different operating point, not a duplicate sample.
    varied = os.path.join(tmp, "varied")
    _write_dataset(varied, "IEEE24", {
        "train": [(0, [1]), (0, [2]), (1, [3])],
        "val": [(200, [2])],
        "test": [(400, [3])],
    })
    g = Gate()
    gate_split_hygiene(g, varied, ["IEEE24"], expect_blocked=False)
    check("same snapshot with a different outage is not flagged",
          not g.failures, str(g.failures))


def main():
    print("=" * 50)
    test_ranges()
    test_gate()
    print("=" * 50)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
