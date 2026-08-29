"""ac_feasibility.py -- does the predicted state satisfy the AC power flow? (audit B1)

WHY THIS MODULE EXISTS
    Every metric in `physics_metrics.py` compares a prediction with the label.
    None of them asks the question an operator asks: *is the predicted state a
    physically consistent operating point of the post-contingency network?* A
    surrogate can have a small NRMSE and still return a state that violates
    Kirchhoff's laws at every bus, or one that loads a line past its thermal
    rating -- and a screening tool that misses an overload is worse than useless.

    The node task makes this checkable without any extra modelling: the model
    emits all four quantities at every bus (two predicted, two re-injected known
    values), so the complex voltage and the complex injection are both fully
    determined. Given the network admittance matrix of the topology the sample
    was actually solved on, the AC residual

        dS_i = S_i^spec - ( V_i * conj( (Y V)_i ) - S_i^shunt )

    is exactly zero for the true state and is the honest error measure for a
    predicted one. Branch currents follow from the same state, so the thermal
    check against the ratings costs nothing extra.

WHICH BRANCHES THE THERMAL CHECK COVERS
    Lines and two-winding transformers, i.e. every branch of these four grids.
    The two have different ratings: a line carries one `max_i_ka` at both ends,
    a transformer is rated in MVA and pandapower's `loading_percent` is the
    larger of the two end currents relative to that end's rated current
    (`sn_mva / (sqrt(3) vn_side)`), so the check keeps a rating per branch END.
    Transformers are 5/38, 11/46, 9/184 and 4/90 branches of IEEE24, IEEE39,
    IEEE118 and UK, and they are the branches an outage most often removes in
    the harvested contingencies, so excluding them would silently drop part of
    every overload rate (audit C4).

CONVENTIONS AND WHY THE SHUNT TERM IS THERE
    * Targets are pandapower `res_bus` rows, i.e. NET CONSUMPTION at the bus
      (load positive, generation negative), so the injection is `-(P + jQ)`.
    * `res_bus` counts a shunt element as consumption at its bus, while the ppc
      `Ybus` also carries that shunt in its diagonal. Subtracting
      `S^shunt = conj(Ysh) |V|^2` removes the double count; without it the
      residual of the TRUE state is ~100 Mvar at every shunt bus, which is the
      calibration this module is validated against (`tests/test_ac_feasibility.py`
      requires the true state to come out at ~1e-3 MW).
    * The admittance matrix, branch data and ratings come from pandapower's own
      internal ppc for the reconstructed post-contingency network, not from the
      graph's `edge_attr` (which carries series r/x only, so it would miss line
      charging, tap ratios and phase shifts and give the true state a spurious
      residual).

WHAT IT DOES NOT DO
    It does not re-solve the power flow, and it does not "correct" a prediction
    onto the feasible manifold. It reports the violation of the physics, per
    sample, in MW / Mvar / percent of rating.
"""
from __future__ import annotations

import ast
import copy
import os
from dataclasses import dataclass

import numpy as np
import pandapower as pp
import pandas as pd

from transmission_grids import load_case, load_hourly_demand

# ppc branch columns (MATPOWER ordering).
F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT = 0, 1, 2, 3, 4, 8, 9
# ppc bus columns for the fixed shunt, in MW / Mvar at V = 1.0 pu.
GS, BS = 4, 5

OVERLOAD_PCT = 100.0


@dataclass
class TopologyCase:
    """Everything needed to score a state on one post-contingency topology.

    Built once per distinct outage set and reused across every demand snapshot
    and every checkpoint, since the admittance matrix depends on the topology
    alone.
    """
    ybus: np.ndarray          # (n, n) complex, ppc bus ordering
    ysh: np.ndarray           # (n,) complex shunt admittance, ppc ordering
    order: np.ndarray         # (n_bus,) dataset row -> ppc bus index
    sn_mva: float
    f_bus: np.ndarray         # per in-service branch, ppc bus index
    t_bus: np.ndarray
    y_series: np.ndarray      # 1 / (r + jx) per branch, pu
    b_half: np.ndarray        # charging susceptance / 2 per branch, pu
    ratio: np.ndarray         # complex tap ratio per branch
    i_base_from: np.ndarray   # pu current -> kA at the from end
    i_base_to: np.ndarray
    i_limit_from_ka: np.ndarray   # rated current at the from/hv end
    i_limit_to_ka: np.ndarray     # rated current at the to/lv end
    is_line: np.ndarray       # branch is a line rather than a transformer

    def voltage(self, vm: np.ndarray, va_degree: np.ndarray) -> np.ndarray:
        v = np.zeros(self.ybus.shape[0], dtype=complex)
        v[self.order] = vm * np.exp(1j * np.deg2rad(va_degree))
        return v

    def mismatch(self, p_mw, q_mvar, vm, va_degree):
        """(dP, dQ) per bus in MW / Mvar, in dataset row order.

        `p_mw`/`q_mvar` follow the `res_bus` consumption convention.
        """
        v = self.voltage(vm, va_degree)
        spec = np.zeros(self.ybus.shape[0], dtype=complex)
        spec[self.order] = -(np.asarray(p_mw) + 1j * np.asarray(q_mvar)) / self.sn_mva
        network = v * np.conj(self.ybus @ v) - np.conj(self.ysh) * np.abs(v) ** 2
        residual = (spec - network)[self.order] * self.sn_mva
        return residual.real, residual.imag

    def loading_percent(self, vm, va_degree):
        """Per in-service branch (lines then transformers), loading in percent.

        Same definition pandapower uses for `res_line.loading_percent` and
        `res_trafo.loading_percent` under its default current-based convention,
        so it can be validated against both on a solved network.
        """
        v = self.voltage(vm, va_degree)
        vf, vt = v[self.f_bus], v[self.t_bus]
        i_from = ((self.y_series + 1j * self.b_half) / np.abs(self.ratio) ** 2 * vf
                  - self.y_series / np.conj(self.ratio) * vt)
        i_to = (-self.y_series / self.ratio * vf
                + (self.y_series + 1j * self.b_half) * vt)
        return np.maximum(
            np.abs(i_from) * self.i_base_from / self.i_limit_from_ka,
            np.abs(i_to) * self.i_base_to / self.i_limit_to_ka) * 100.0


def _apply_outage(net, out_lines) -> None:
    """Take the recorded outage out of service.

    `out_lines` is either positional line indices (random N-k) or harvested
    ``('line'|'trafo', index)`` pairs, matching the two generators in
    `transmission_graph_gen.py`.
    """
    if not out_lines:
        return
    if isinstance(out_lines[0], (list, tuple)):
        for etype, idx in out_lines:
            net[etype].at[idx, "in_service"] = False
        return
    if isinstance(out_lines[0], str):        # a stringified ('line', 3) pair
        for item in out_lines:
            etype, idx = ast.literal_eval(item)
            net[etype].at[idx, "in_service"] = False
        return
    net.line.loc[net.line.index[list(out_lines)], "in_service"] = False


def build_case(base_net, demand_col, out_lines) -> TopologyCase:
    """Reconstruct the post-contingency network and extract its ppc quantities.

    A power flow is solved once here only to populate pandapower's internal ppc;
    the demand used is irrelevant to everything the case stores.
    """
    from transmission_graph_gen import _apply_demand

    net = copy.deepcopy(base_net)
    _apply_demand(net, demand_col)
    _apply_outage(net, out_lines)
    pp.runpp(net)

    ppc = net._ppc["internal"]
    ybus = ppc["Ybus"].toarray()
    bus = np.real(ppc["bus"])
    branch = ppc["branch"]
    order = net._pd2ppc_lookups["bus"][net.bus.index.values]

    # The internal ppc drops out-of-service branches and keeps pandapower's
    # element order, so the first rows are the in-service lines and the next the
    # in-service two-winding transformers.
    lines = net.line[net.line.in_service]
    trafos = net.trafo[net.trafo.in_service]
    n_branch = len(lines) + len(trafos)
    rows = branch[:n_branch]
    f_bus = np.real(rows[:, F_BUS]).astype(int)
    t_bus = np.real(rows[:, T_BUS]).astype(int)
    r = np.real(rows[:, BR_R])
    x = np.real(rows[:, BR_X])
    b = np.real(rows[:, BR_B])
    tap = np.real(rows[:, TAP]).copy()
    tap[tap == 0] = 1.0
    ratio = tap * np.exp(1j * np.deg2rad(np.real(rows[:, SHIFT])))

    # ppc bus index -> nominal voltage, for the pu-current to kA conversion.
    vn_kv = np.empty(ybus.shape[0])
    vn_kv[order] = net.bus.vn_kv.values
    i_base = net.sn_mva / (np.sqrt(3) * vn_kv)

    line_ka = lines.max_i_ka.values * lines.df.values * lines.parallel.values
    s_rated = (trafos.sn_mva.values * trafos.df.values
               * trafos.parallel.values)
    i_limit_from = np.concatenate(
        [line_ka, s_rated / (np.sqrt(3) * trafos.vn_hv_kv.values)])
    i_limit_to = np.concatenate(
        [line_ka, s_rated / (np.sqrt(3) * trafos.vn_lv_kv.values)])

    return TopologyCase(
        ybus=ybus,
        ysh=(bus[:, GS] + 1j * bus[:, BS]) / net.sn_mva,
        order=order,
        sn_mva=float(net.sn_mva),
        f_bus=f_bus, t_bus=t_bus,
        y_series=1.0 / (r + 1j * x),
        b_half=b / 2.0,
        ratio=ratio,
        i_base_from=i_base[f_bus],
        i_base_to=i_base[t_bus],
        i_limit_from_ka=i_limit_from,
        i_limit_to_ka=i_limit_to,
        is_line=np.arange(n_branch) < len(lines),
    )


def build_cases(grid: str, split_dir: str, cases_dir: str | None = None):
    """One `TopologyCase` per sample of a split, aligned with `dataset.pt`.

    `dataset_src.csv` is written row-for-row with the samples, so row i is the
    provenance of sample i. Cases are cached by outage set, which is what makes
    this affordable: the admittance matrix does not depend on the demand.
    """
    src = pd.read_csv(os.path.join(split_dir, "dataset_src.csv"))
    base = load_case(grid, cases_dir)
    demand = load_hourly_demand(grid)
    cache, cases = {}, []
    for _, row in src.iterrows():
        out_lines = ast.literal_eval(str(row["out_lines"]))
        key = tuple(map(str, out_lines))
        if key not in cache:
            cache[key] = build_case(base, demand[:, int(row["t_idx"])], out_lines)
        cases.append(cache[key])
    return cases


def _rates(count, total):
    return float(count) / total if total else float("nan")


def feasibility_metrics(y_true, y_pred, cases, n_bus: int) -> dict[str, float]:
    """AC residual and thermal metrics for one (checkpoint, test grid) pair.

    `y_true`/`y_pred` are (n_samples * n_bus, 4) in PHYSICAL units and in the
    order the loader produced them, so they reshape onto the per-sample cases.
    The true state is scored too: its residual is the numerical floor of this
    check and belongs next to the model's number rather than being assumed.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(len(cases), n_bus, 4)
    y_pred = np.asarray(y_pred, dtype=float).reshape(len(cases), n_bus, 4)

    dp, dq, load_max, load_max_true = [], [], [], []
    dp_true, over_true, over_pred = [], [], []
    dp_share, dq_share = [], []
    for i, case in enumerate(cases):
        p, q, vm, va = y_pred[i].T
        rp, rq = case.mismatch(p, q, vm, va)
        dp.append(np.abs(rp))
        dq.append(np.abs(rq))
        # MW means nothing across grids of different size (IEEE24 carries ~2 GW,
        # UK far more), so the residual is also reported as a share of the
        # snapshot's own served load.
        load_mw = np.abs(y_true[i, :, 0].clip(min=0)).sum()
        load_mvar = np.abs(y_true[i, :, 1]).sum()
        dp_share.append(np.abs(rp).sum() / load_mw if load_mw else np.nan)
        dq_share.append(np.abs(rq).sum() / load_mvar if load_mvar else np.nan)

        tp, tq, tvm, tva = y_true[i].T
        dp_true.append(np.abs(case.mismatch(tp, tq, tvm, tva)[0]))

        pred_load = case.loading_percent(vm, va)
        true_load = case.loading_percent(tvm, tva)
        load_max.append(np.nanmax(pred_load))
        load_max_true.append(np.nanmax(true_load))
        over_pred.append(pred_load > OVERLOAD_PCT)
        over_true.append(true_load > OVERLOAD_PCT)

    dp, dq = np.concatenate(dp), np.concatenate(dq)
    over_true = np.concatenate(over_true)
    over_pred = np.concatenate(over_pred)
    n_over_true = int(over_true.sum())
    n_secure_true = int((~over_true).sum())
    return {
        "ac_dp_mean_mw": float(np.nanmean(dp)),
        "ac_dp_max_mw": float(np.nanmax(dp)),
        "ac_dp_p95_mw": float(np.nanpercentile(dp, 95)),
        "ac_dq_mean_mvar": float(np.nanmean(dq)),
        "ac_dq_max_mvar": float(np.nanmax(dq)),
        "ac_dq_p95_mvar": float(np.nanpercentile(dq, 95)),
        "ac_dp_pct_load": float(np.nanmean(dp_share) * 100.0),
        "ac_dq_pct_load": float(np.nanmean(dq_share) * 100.0),
        # The same residual on the labels: anything the reconstruction itself
        # cannot represent shows up here, so the model number is only meaningful
        # against it.
        "ac_dp_true_max_mw": float(np.nanmax(np.concatenate(dp_true))),
        "branch_loading_max_pct": float(np.nanmax(load_max)),
        # The source OPF snapshots are not themselves thermally secure -- several
        # cases run branches past their rating -- so the predicted loading is
        # only interpretable next to the true one.
        "branch_loading_max_pct_true": float(np.nanmax(load_max_true)),
        "overload_rate_true": _rates(n_over_true, over_true.size),
        "overload_rate_pred": _rates(int(over_pred.sum()), over_pred.size),
        # Screening errors, the operational reading of the thermal check.
        "missed_overload_rate": _rates(int((over_true & ~over_pred).sum()),
                                       n_over_true),
        "false_overload_rate": _rates(int((~over_true & over_pred).sum()),
                                      n_secure_true),
    }
