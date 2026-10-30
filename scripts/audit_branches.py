"""Audit every MATPOWER case against its extracted graph (no dataset writes)."""
from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pandapower as pp
from pandapower.converter.matpower.from_mpc import from_mpc
from pandapower.topology import create_nxgraph
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engage_contract import get_edge_features


def audit(path):
    mpc = loadmat(path, squeeze_me=True, struct_as_record=False)["mpc"]
    expected = int(np.count_nonzero(np.atleast_2d(mpc.branch)[:, 10]))
    net = from_mpc(str(path))
    tables = ("line", "trafo", "impedance", "trafo3w", "dcline", "tcsc",
              "line_dc", "vsc", "ssc")
    counts = ",".join(f"{t}={len(net[t])}" for t in tables if t in net)
    print(f"{path.stem}: buses={len(net.bus)} MATPOWER={expected} {counts}")
    other = {t: len(v) for t, v in net.items()
             if hasattr(v, "columns") and len(v)
             and t not in tables and not t.startswith("res_")}
    print(f"  other non-empty tables: {other}")
    pp.runpp(net)
    edges, _ = get_edge_features(net)
    graph = nx.MultiGraph()
    graph.add_nodes_from(range(len(net.bus)))
    graph.add_edges_from(edges[:, ::2].T.tolist())
    pos = {b: i for i, b in enumerate(net.bus.index)}
    reference = create_nxgraph(net, respect_switches=True, multi=True,
                               include_out_of_service=False)
    from collections import Counter
    actual_pairs = Counter(tuple(sorted(e)) for e in edges[:, ::2].T.tolist())
    wanted_pairs = Counter(tuple(sorted((pos[u], pos[v]))) for u, v in reference.edges())
    isolated = list(nx.isolates(graph))
    components = nx.number_connected_components(graph)
    ok = (edges.shape[1] == 2 * expected and actual_pairs == wanted_pairs
          and not isolated and components == nx.number_connected_components(reference))
    print(f"  graph={edges.shape[1] // 2} isolated={isolated} components={components} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "transmission/cases"
    cases = sorted(folder.glob("*.mat"))
    if not cases:
        print(f"FAIL: no .mat cases in {folder}")
        return 1
    passed = True
    for path in cases:
        try:
            passed = audit(path) and passed
        except Exception as exc:
            print(f"  FAIL: {path.name}: {type(exc).__name__}: {exc}")
            passed = False
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
