"""engage_contract.py -- the ENGAGE graph data contract, vendored.

PURPOSE
    Turn a *solved* pandapower network into the ENGAGE-format tensors used by the
    model zoo and training loop:
        x         : (N, 7)  = [Slack?, PV?, PQ?, p_mw, q_mvar, vm_pu, va_degree]
        edge_index: (2, 2E)
        edge_attr : (2E, 4) = [trafo?, r_pu, x_pu, sc_voltage]
        y         : (N, 4)  = [p_mw, q_mvar, vm_pu, va_degree]

WHY VENDORED (design decisions D6 + D8)
    We deliberately dropped the "make two repos interoperate" model in favour of
    ONE clean, self-contained pipeline. Rather than importing ENGAGE as a
    dependency, we vendor its two feature extractors here (attributed below) so
    this repo runs on its own. The logic mirrors ENGAGE's `graph_gen.py` almost
    verbatim, with ONE deliberate change for our study:

      * CONTINGENCY SUPPORT: ENGAGE's original extractors read *all* rows of
        `net.line` / `net.trafo`. For N-1/N-k contingencies we take branches out
        of service (`in_service = False`), so here we only include IN-SERVICE
        branches when building `edge_index`/`edge_attr`. This is what makes the
        graph topology actually change with each contingency (the whole point of
        the g-score study).

MASKING (D6)
    Inputs unknown at inference are set to NaN according to bus type, exactly as
    ENGAGE does:
      * Slack (ref): p_mw, q_mvar unknown  -> NaN
      * PV  (gen)  : q_mvar, va_degree unknown -> NaN
      * PQ  (load) : vm_pu, va_degree unknown  -> NaN
    The targets `y` always contain the full solved state.

SOURCE / ATTRIBUTION
    Adapted from ENGAGE (energy-management-technologies-public/engage),
    `graph_gen.py: get_node_features / get_edge_features`.
    ENGAGE paper: https://doi.org/10.1145/3679240.3734610
"""
from __future__ import annotations

import numpy as np


def get_node_features(net):
    """Return (x, y) node feature/label arrays from a *solved* net.

    Requires `net.res_bus` to be populated (i.e. call pp.runpp/runopp first).
    """
    node_features_x, node_features_y = [], []
    for bus_id in net.bus.index:
        # Default bus type is PQ (load) -> one-hot [Slack?, PV?, PQ?].
        bus_type = (0, 0, 1)

        gens = net.gen.loc[net.gen["bus"] == bus_id]
        if len(gens) > 0:
            bus_type = (0, 1, 0)  # PV bus (voltage-controlled generator)

        slack = net.ext_grid.loc[net.ext_grid["bus"] == bus_id, ["vm_pu", "va_degree"]]
        if len(slack) > 0:
            assert len(gens) == 0, (
                "PV and Swing generators cannot be placed on the same bus."
            )
            bus_type = (1, 0, 0)  # slack / reference bus

        # net.res_bus already aggregates every component at the bus.
        features = net.res_bus.loc[bus_id, ["p_mw", "q_mvar", "vm_pu", "va_degree"]]
        masked_features = features.copy()
        if bus_type[0]:            # slack: injections unknown
            masked_features["p_mw"] = np.nan
            masked_features["q_mvar"] = np.nan
        elif bus_type[1]:          # PV: reactive + angle unknown
            masked_features["q_mvar"] = np.nan
            masked_features["va_degree"] = np.nan
        else:                      # PQ: voltage magnitude + angle unknown
            masked_features["vm_pu"] = np.nan
            masked_features["va_degree"] = np.nan

        node_features_x.append(np.append(bus_type, masked_features.values))
        node_features_y.append(features.values)

    return np.array(node_features_x), np.array(node_features_y)


def get_edge_features(net):
    """Return (edge_index, edge_attr) for the *in-service* branches of net.

    edge_index : (2, 2E) COO for an undirected graph (both directions).
    edge_attr  : (2E, 4) = [trafo?, r_pu, x_pu, sc_voltage].
    """

    def get_line_features(net):
        lines = net.line[net.line["in_service"]]  # <-- contingency-aware
        edge_index = lines.loc[:, ["from_bus", "to_bus", "to_bus", "from_bus"]].values
        edge_index = edge_index.reshape(-1, 2).T

        r = lines["r_ohm_per_km"].values * lines["length_km"].values
        x = lines["x_ohm_per_km"].values * lines["length_km"].values

        # Convert r, x to per-unit using the base impedance z = vn_kv**2 / sn_mva.
        vn_kv = net.bus.loc[lines["to_bus"], ["vn_kv"]].values.reshape(-1)
        z = np.square(vn_kv) / net.sn_mva
        r_pu = (r / z).repeat(2)
        x_pu = (x / z).repeat(2)

        e = edge_index.shape[1]
        edge_features = np.vstack(
            [np.zeros(e), r_pu, x_pu, np.nan * np.ones(e)]  # trafo?, r_pu, x_pu, sc_voltage
        ).T
        return edge_index, edge_features

    def get_trafo_features(net):
        trafos = net.trafo[net.trafo["in_service"]]  # <-- contingency-aware
        edge_index = trafos.loc[:, ["hv_bus", "lv_bus", "lv_bus", "hv_bus"]].values
        edge_index = edge_index.reshape(-1, 2).T

        # Impedance per pandapower trafo docs (vk_percent = short-circuit voltage).
        z_pu = (trafos["vk_percent"].values / 100) * (net.sn_mva / trafos["sn_mva"].values)
        r_pu = (trafos["vkr_percent"].values / 100) * (net.sn_mva / trafos["sn_mva"].values)
        x_pu = np.sqrt(np.square(z_pu) - np.square(r_pu))
        sc_voltage = trafos["vk_percent"].values

        e = edge_index.shape[1]
        edge_features = np.vstack(
            [np.ones(e), r_pu.repeat(2), x_pu.repeat(2), sc_voltage.repeat(2)]
        ).T
        return edge_index, edge_features

    A_line, E_line = get_line_features(net)
    A_trafo, E_trafo = get_trafo_features(net)

    A = np.hstack([A_line, A_trafo])
    E = np.vstack([E_line, E_trafo])

    # Remap (possibly sparse) bus ids to a dense 0..N-1 range for edge_index.
    # NOTE: this remaps only nodes that appear in the edge list; for a connected
    # in-service graph that is every node.
    unique_nodes = set(A[0])
    remapping = dict(zip(sorted(unique_nodes), range(len(unique_nodes))))
    applyall = np.vectorize(lambda v: remapping[v])
    A = applyall(A)

    return A, E
