"""Checks for ac_feasibility.py (audit B1).

The whole module is only worth anything if the residual of the TRUE state is
numerically zero and the loading agrees with pandapower's own `res_line`, so
that is what is asserted here -- against a network pandapower solved itself,
not against a fixture of our own numbers.

    POWERGRAPH_NODE_DIR=... python3 tests/test_ac_feasibility.py
"""
from __future__ import annotations

import copy
import os
import sys

import numpy as np
import pandapower as pp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ac_feasibility import build_case, feasibility_metrics  # noqa: E402
from transmission_grids import load_case, load_hourly_demand  # noqa: E402

GRID = "IEEE24"
OUT_LINES = [7, 24]


def check(cond, label):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    if not cond:
        raise AssertionError(label)


def solved_reference(out_lines):
    """A post-contingency network solved by pandapower, plus its state."""
    from transmission_graph_gen import _apply_demand
    from ac_feasibility import _apply_outage

    base = load_case(GRID)
    demand = load_hourly_demand(GRID)[:, 1000]
    net = copy.deepcopy(base)
    _apply_demand(net, demand)
    _apply_outage(net, out_lines)
    pp.runpp(net)
    state = net.res_bus[["p_mw", "q_mvar", "vm_pu", "va_degree"]].values
    return base, demand, net, state


def test_true_state_has_no_residual():
    base, demand, net, state = solved_reference(OUT_LINES)
    case = build_case(base, demand, OUT_LINES)
    dp, dq = case.mismatch(*state.T)
    check(np.abs(dp).max() < 1e-2,
          f"solved state satisfies P balance, max |dP| = {np.abs(dp).max():.2e} MW")
    check(np.abs(dq).max() < 1e-2,
          f"solved state satisfies Q balance, max |dQ| = {np.abs(dq).max():.2e} Mvar")


def test_shunt_is_not_double_counted():
    """Without the shunt correction the true state misses by the shunt's rating.

    `res_bus` books a shunt as consumption at its bus and the ppc Ybus carries
    the same shunt in its diagonal; the residual of a perfectly solved state
    would otherwise be ~100 Mvar at that bus, which is the failure this guards.
    """
    base, demand, net, state = solved_reference(OUT_LINES)
    check(len(net.shunt) > 0, "the test grid actually has a shunt to double count")
    case = build_case(base, demand, OUT_LINES)
    v = case.voltage(state[:, 2], state[:, 3])
    uncorrected = (-(state[:, 0] + 1j * state[:, 1]) / case.sn_mva
                   - (v * np.conj(case.ybus @ v))[case.order]) * case.sn_mva
    corrected = case.mismatch(*state.T)[1]
    check(np.abs(uncorrected.imag).max() > 10.0,
          "the uncorrected residual is large at the shunt bus (the bug)")
    check(np.abs(corrected).max() < 1e-2, "the corrected residual is zero")


def test_loading_matches_pandapower():
    base, demand, net, state = solved_reference(OUT_LINES)
    case = build_case(base, demand, OUT_LINES)
    got = case.loading_percent(state[:, 2], state[:, 3])
    want = net.res_line.loading_percent[net.line.in_service].values
    check(len(got) == len(want), "one loading per in-service line")
    check(np.abs(got - want).max() < 1e-3,
          f"loading matches res_line, max diff = {np.abs(got - want).max():.2e} %")


def test_perturbed_state_is_penalised():
    """A wrong state must produce a residual; a metric that cannot fail is useless."""
    base, demand, net, state = solved_reference(OUT_LINES)
    case = build_case(base, demand, OUT_LINES)
    bad = state.copy()
    bad[:, 2] += 0.05                      # 5% voltage error everywhere
    dp, dq = case.mismatch(*bad.T)
    check(np.abs(dq).max() > 1.0,
          f"a 0.05 pu voltage error shows up as {np.abs(dq).max():.1f} Mvar")


def test_metrics_shape_and_true_floor():
    base, demand, net, state = solved_reference(OUT_LINES)
    case = build_case(base, demand, OUT_LINES)
    n_bus = len(net.bus)
    truth = np.tile(state, (2, 1))
    pred = truth.copy()
    pred[:, 2] += 0.02
    out = feasibility_metrics(truth, pred, [case, case], n_bus)
    check(out["ac_dp_true_max_mw"] < 1e-2,
          "the label state is reported as the numerical floor")
    check(out["ac_dq_max_mvar"] > out["ac_dp_true_max_mw"],
          "the perturbed prediction is worse than the floor")
    check(0.0 <= out["overload_rate_pred"] <= 1.0, "overload rate is a rate")
    check(set(out) >= {"line_loading_max_pct", "missed_overload_rate",
                       "false_overload_rate"},
          "the thermal screening columns are present")


def test_overload_confusion_directions():
    """Missed vs false overload must be counted against the right denominators."""
    base, demand, net, state = solved_reference(OUT_LINES)
    case = build_case(base, demand, OUT_LINES)
    n_bus = len(net.bus)
    # Put every line just below its rating in the true state, so the direction
    # of a voltage error decides the direction of the screening mistake.
    true_load = case.loading_percent(state[:, 2], state[:, 3])
    case.i_limit_ka = case.i_limit_ka * true_load / 99.0

    # Branch current follows the voltage DIFFERENCE, so scaling every magnitude
    # up scales every current up with it.
    high = state.copy()
    high[:, 2] *= 1.05
    out = feasibility_metrics(state, high, [case], n_bus)
    check(out["overload_rate_true"] == 0.0, "the true state is secure")
    check(out["false_overload_rate"] > 0,
          "over-flagged lines are counted as false alarms")

    # And the operationally dangerous direction: the true state is overloaded
    # and the prediction says it is not.
    case.i_limit_ka = case.i_limit_ka * 99.0 / 101.0
    low = state.copy()
    low[:, 2] *= 0.95
    out = feasibility_metrics(state, low, [case], n_bus)
    check(out["overload_rate_true"] == 1.0, "every line is overloaded in truth")
    check(out["missed_overload_rate"] > 0,
          "unflagged overloads are counted as misses")


if __name__ == "__main__":
    for fn in (test_true_state_has_no_residual,
               test_shunt_is_not_double_counted,
               test_loading_matches_pandapower,
               test_perturbed_state_is_penalised,
               test_metrics_shape_and_true_floor,
               test_overload_confusion_directions):
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + "=" * 50 + "\nALL CHECKS PASSED")
