"""Test q modes and the optional blocked Regime A pipeline."""
from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import os
import sys
import tempfile
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pandapower as pp

import transmission_graph_gen as graph_gen
import validate
from transmission_grids import load_case

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def _test_q_modes():
    print("q modes")
    net = pp.networks.case14()
    zero_idx = pp.create_load(net, bus=int(net.bus.index[0]), p_mw=0.0, q_mvar=5.0)
    case_loads = net.load.copy(deep=True)
    demand = np.random.default_rng(31).uniform(30.0, 100.0, len(net.bus))

    zero_count = graph_gen._apply_demand(net, demand, q_mode="constant_pf")
    p_is_demand = all(
        net.load.at[idx, "p_mw"] == float(demand[bus])
        for idx, bus in case_loads["bus"].items()
    )
    ratios_match = all(
        np.isclose(
            net.load.at[idx, "q_mvar"] / net.load.at[idx, "p_mw"],
            case_loads.at[idx, "q_mvar"] / case_loads.at[idx, "p_mw"],
            rtol=1e-12,
            atol=0.0,
        )
        for idx in case_loads.index
        if case_loads.at[idx, "p_mw"] != 0
    )
    zero_q_stays = net.load.at[zero_idx, "q_mvar"] == 5.0
    check("constant_pf sets each load P from demand", p_is_demand)
    check("constant_pf preserves each nonzero load's Q/P ratio", ratios_match)
    check("constant_pf keeps Q for zero case P", zero_q_stays)
    check("constant_pf returns the number of zero-P loads", zero_count == 1,
          str(zero_count))

    base = pp.networks.case14()
    base_before = base.load.copy(deep=True)
    base_net = deepcopy(base)
    reference = deepcopy(base)
    for idx, bus in reference.load["bus"].items():
        reference.load.at[idx, "p_mw"] = float(demand[bus])
    base_q = base_net.load["q_mvar"].to_numpy(copy=True)
    base_zero_count = graph_gen._apply_demand(base_net, demand)
    check("base returns zero zero-P loads", base_zero_count == 0, str(base_zero_count))
    check("base leaves Q exactly unchanged",
          np.array_equal(base_net.load["q_mvar"].to_numpy(), base_q))
    check("base writes demand P exactly",
          all(base_net.load.at[idx, "p_mw"] == float(demand[bus])
              for idx, bus in base_before["bus"].items()))
    check("base matches the previous P-only loop exactly",
          base_net.load.equals(reference.load))

    unknown_raises = False
    try:
        graph_gen._apply_demand(deepcopy(base), demand, q_mode="not_a_mode")
    except ValueError:
        unknown_raises = True
    check("unknown q_mode raises ValueError", unknown_raises)


def _synthetic_demand():
    net = load_case("IEEE24")
    p_by_bus = np.zeros(len(net.bus), dtype=float)
    np.add.at(
        p_by_bus,
        net.load["bus"].to_numpy(dtype=int),
        net.load["p_mw"].to_numpy(dtype=float),
    )
    t = np.arange(600)
    factor = 0.7 + 0.3 * np.sin(2 * np.pi * t / 144.0)
    return p_by_bus[:, None] * factor[None, :]


def _generate(out_dir, demand, q_mode, random_split=False):
    argv = [
        "transmission_graph_gen.py",
        "--grid", "IEEE24",
        "--max_k", "0",
        "--time_split", "random" if random_split else "blocked",
        "--time_gap", "20",
        "--n_train", "20",
        "--n_val", "5",
        "--n_test", "5",
        "--q_mode", q_mode,
        "--out_dir", out_dir,
    ]
    if random_split:
        argv.append("--unique_demand")
    with mock.patch.object(sys, "argv", argv), mock.patch.object(
        graph_gen, "load_hourly_demand", return_value=demand
    ):
        graph_gen.main()


def _validate(out_dir, expect_blocked):
    argv = ["validate.py", "--data_dir", out_dir, "--grids", "IEEE24", "--regime", "a"]
    if expect_blocked:
        argv.append("--expect_blocked")
    output = io.StringIO()
    exit_code = None
    with mock.patch.object(sys, "argv", argv), redirect_stdout(output):
        try:
            validate.main()
        except SystemExit as exc:
            exit_code = exc.code
    return exit_code, output.getvalue()


def _check_blocked_dataset(out_dir, q_mode):
    counts = {"train": 20, "val": 5, "test": 5}
    windows = graph_gen.blocked_time_ranges(600, counts, 20)
    spans = {}
    for split, expected_count in counts.items():
        meta_path = os.path.join(out_dir, "IEEE24", split, "dataset_src.csv")
        meta = pd.read_csv(meta_path)
        check(f"{q_mode}/{split} has the requested sample count",
              len(meta) == expected_count, str(len(meta)))
        check(f"{q_mode}/{split} has only k == 0",
              bool((meta["k"] == 0).all()), str(sorted(meta["k"].unique())))
        check(f"{q_mode}/{split} has no outaged lines",
              bool(meta["out_lines"].astype(str).eq("[]").all()))
        check(f"{q_mode}/{split} records q_mode",
              bool(meta["q_mode"].eq(q_mode).all()))
        lo, hi = windows[split]
        in_window = bool(meta["t_idx"].between(lo, hi - 1).all())
        check(f"{q_mode}/{split} stays in its blocked window",
              in_window, f"expected [{lo}, {hi})")
        spans[split] = (int(meta["t_idx"].min()), int(meta["t_idx"].max()))
    ordered = [spans[s] for s in ("train", "val", "test")]
    check(f"{q_mode} split windows are disjoint",
          all(ordered[i][1] < ordered[i + 1][0]
              for i in range(len(ordered) - 1)),
          str(spans))


def _test_blocked_generation():
    print("\nend-to-end blocked Regime A")
    demand = _synthetic_demand()
    with tempfile.TemporaryDirectory(prefix="q-mode-blocked-") as root:
        for q_mode in ("base", "constant_pf"):
            out_dir = os.path.join(root, q_mode)
            _generate(out_dir, demand, q_mode)
            _check_blocked_dataset(out_dir, q_mode)
            exit_code, output = _validate(out_dir, expect_blocked=True)
            check(f"{q_mode} blocked Regime A passes validation",
                  exit_code is None and "ALL GATES PASSED" in output,
                  output[-400:])

        random_dir = os.path.join(root, "random")
        _generate(random_dir, demand, "base", random_split=True)
        blocked_exit, blocked_output = _validate(random_dir, expect_blocked=True)
        check("blocked gate rejects the randomly split Regime A",
              blocked_exit not in (None, 0)
              and "[FAIL] IEEE24 split time windows are disjoint" in blocked_output,
              blocked_output[-400:])
        normal_exit, normal_output = _validate(random_dir, expect_blocked=False)
        check("random Regime A still passes without --expect_blocked",
              normal_exit is None and "ALL GATES PASSED" in normal_output,
              normal_output[-400:])


def main():
    print("=" * 50)
    _test_q_modes()
    _test_blocked_generation()
    print("=" * 50)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
