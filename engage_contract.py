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
    verbatim, with explicit series-branch and contingency support:

      * Impedance elements use the line encoding [0, r_pu, x_pu, NaN],
        preserving the four-column edge contract. Bus positions follow the
        full net.bus.index order, including buses without line/trafo elements.

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

from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
import pandapower.topology as top


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


def branch_table(net) -> pd.DataFrame:
    """Return all series elements on the system base, in table/index order.

    Impedance shunts (gf/bf/gt/bt) are ignored, as are line shunts.
    Asymmetric impedances cannot be represented by the undirected contract.
    """
    supported = {"line", "trafo", "impedance"}
    unsupported = {"trafo3w", "dcline", "tcsc", "line_dc", "vsc", "ssc"}
    for name, table in net.items():
        if not isinstance(table, pd.DataFrame) or not len(table):
            continue
        branch_like = ({"from_bus", "to_bus"} <= set(table.columns)
                       or {"hv_bus", "lv_bus"} <= set(table.columns))
        if name in unsupported or (branch_like and name not in supported):
            raise NotImplementedError(f"Unsupported branch table {name}: {len(table)} rows")
    switches = net.switch[(net.switch.et == "b") & net.switch.closed]
    if len(switches):
        raise NotImplementedError(f"Unsupported closed bus-bus switch table: {len(switches)} rows")

    rows = []
    for etype in ("line", "trafo", "impedance"):
        for element, row in net[etype].iterrows():
            f, t = (row.hv_bus, row.lv_bus) if etype == "trafo" else (row.from_bus, row.to_bus)
            if f not in net.bus.index or t not in net.bus.index:
                raise ValueError(f"{etype}:{element} endpoint not in net.bus")
            sc = np.nan
            if etype == "line":
                z = net.bus.at[t, "vn_kv"] ** 2 / net.sn_mva
                r = row.r_ohm_per_km * row.length_km / z
                x = row.x_ohm_per_km * row.length_km / z
            elif etype == "trafo":
                z = row.vk_percent / 100 * net.sn_mva / row.sn_mva
                r = row.vkr_percent / 100 * net.sn_mva / row.sn_mva
                x = np.sqrt(z ** 2 - r ** 2)
                sc = row.vk_percent
            else:
                if row.rft_pu != row.rtf_pu or row.xft_pu != row.xtf_pu:
                    raise ValueError(f"Asymmetric impedance:{element} cannot use undirected contract")
                r = row.rft_pu * net.sn_mva / row.sn_mva
                x = row.xft_pu * net.sn_mva / row.sn_mva
            rows.append((etype, element, f, t, bool(row.in_service),
                         int(etype == "trafo"), r, x, sc))
    return pd.DataFrame(rows, columns=[
        "etype", "element", "from_bus", "to_bus", "in_service",
        "is_trafo", "r_pu", "x_pu", "sc_voltage"])


def get_edge_features(net):
    """Return both directions of every in-service branch in net.bus row order."""
    branches = branch_table(net)
    branches = branches.loc[branches.in_service.astype(bool)]
    pos = {b: i for i, b in enumerate(net.bus.index)}
    pairs = [(pos[r.from_bus], pos[r.to_bus]) for r in branches.itertuples()]
    edges = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    edge_index = np.stack((edges, edges[:, ::-1]), axis=1).reshape(-1, 2).T
    attrs = branches[["is_trafo", "r_pu", "x_pu", "sc_voltage"]].to_numpy(dtype=float)
    return edge_index, attrs.repeat(2, axis=0)


def check_graph_matches_net(net, edge_index, n_nodes):
    """Raise ValueError unless the bidirectional graph matches pandapower topology."""
    edges = np.asarray(edge_index)
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] % 2:
        raise ValueError("edge_index must have shape (2, 2E)")
    if n_nodes != len(net.bus) or (edges.size and (
            edges.min() < 0 or edges.max() >= n_nodes)):
        raise ValueError("Graph node count or edge endpoint outside net.bus rows")
    if not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("edge_index endpoints must be integer row positions")
    if not np.array_equal(edges[:, ::2], edges[::-1, 1::2]):
        raise ValueError("Graph branches must have paired reverse directions")
    pos = {b: i for i, b in enumerate(net.bus.index)}
    reference = top.create_nxgraph(net, respect_switches=True, multi=True,
                                   include_out_of_service=False)
    wanted = Counter(tuple(sorted((pos[u], pos[v]))) for u, v in reference.edges())
    actual = Counter(tuple(sorted(e)) for e in edges[:, ::2].T.tolist())
    if actual != wanted:
        raise ValueError(f"Graph branches differ from pandapower: missing={wanted - actual}, extra={actual - wanted}")
    graph = nx.MultiGraph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(edges[:, ::2].T.tolist())
    if nx.number_connected_components(graph) != nx.number_connected_components(reference):
        raise ValueError("Graph connected components differ from pandapower")
