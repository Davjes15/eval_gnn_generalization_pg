"""Series-branch contract checks; core tests require pandapower, not torch."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandapower as pp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engage_contract as contract
from transmission_grids import load_case


def synthetic():
    net = pp.create_empty_network(sn_mva=100)
    for b in (10, 30, 20, 40):
        pp.create_bus(net, 110, index=b)
    pp.create_line_from_parameters(net, 10, 20, 1, 0.1, 0.2, 0, 1)
    pp.create_line_from_parameters(net, 20, 40, 1, 0.1, 0.2, 0, 1)
    pp.create_impedance(net, 10, 30, 0.02, 0.04, sn_mva=50)
    return net


class BranchContract(unittest.TestCase):
    def test_middle_bus_and_scaling(self):
        net = synthetic()
        edges, attrs = contract.get_edge_features(net)
        self.assertEqual(set(edges.ravel()), set(range(4)))
        self.assertIn(1, edges[:, -2:])
        np.testing.assert_allclose(attrs[-1, :3], [0, 0.04, 0.08])
        self.assertTrue(np.isnan(attrs[-1, 3]))
        contract.check_graph_matches_net(net, edges, 4)

    def test_missing_impedance(self):
        net = synthetic()
        edges, attrs = contract.get_edge_features(net)
        with patch.object(contract, "get_edge_features", return_value=(edges[:, :-2], attrs[:-2])):
            with self.assertRaises(ValueError):
                contract.check_graph_matches_net(net, contract.get_edge_features(net)[0], 4)

    def test_unsupported(self):
        net = synthetic()
        pp.create_transformer3w_from_parameters(net, 10, 20, 40, 10, 10, 10,
            110, 110, 110, 10, 10, 10, 1, 1, 1, 0, 0)
        with self.assertRaisesRegex(NotImplementedError, "trafo3w.*1"):
            contract.branch_table(net)

    def test_asymmetric(self):
        net = synthetic()
        net.impedance.at[0, "rtf_pu"] = 0.03
        with self.assertRaises(ValueError):
            contract.branch_table(net)

    def test_outage(self):
        net = synthetic()
        net.impedance.at[0, "in_service"] = False
        edges, _ = contract.get_edge_features(net)
        self.assertEqual(edges.shape, (2, 4))
        contract.check_graph_matches_net(net, edges, 4)

    def test_empty_and_invalid(self):
        net = pp.create_empty_network()
        edges, attrs = contract.get_edge_features(net)
        self.assertEqual(edges.shape, (2, 0))
        self.assertEqual(attrs.shape, (0, 4))
        contract.check_graph_matches_net(net, edges, 0)
        with self.assertRaises(ValueError):
            contract.check_graph_matches_net(net, edges, 1)
        net = synthetic()
        net.line.at[0, "to_bus"] = 99
        with self.assertRaises(ValueError):
            contract.get_edge_features(net)

    def test_real_cases(self):
        for code, expected in (("IEEE24", 38), ("IEEE39", 46), ("IEEE118", 186), ("UK", 99)):
            with self.subTest(code=code):
                net = load_case(code)
                pp.runpp(net)
                edges, _ = contract.get_edge_features(net)
                self.assertEqual(edges.shape[1], expected * 2)
                graph = nx.MultiGraph()
                graph.add_nodes_from(range(len(net.bus)))
                graph.add_edges_from(edges[:, ::2].T.tolist())
                self.assertEqual(list(nx.isolates(graph)), [])
                self.assertEqual(nx.number_connected_components(graph), 1)
                contract.check_graph_matches_net(net, edges, len(net.bus))

    def test_original_sampler_without_torch(self):
        # Load only these dependency-free definitions, not the torch generator.
        import ast
        import argparse
        source = (Path(__file__).resolve().parents[1] / "transmission_graph_gen.py").read_text()
        module = ast.parse(source)
        names = {"_sample_contingency", "_outage_types"}
        subset = ast.Module(body=[n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
        scope = {"argparse": argparse}
        exec(compile(subset, "sampler", "exec"), scope)
        def original(rng, n_lines, k):
            if k <= 0:
                return []
            k = min(k, n_lines)
            return sorted(rng.choice(n_lines, size=k, replace=False).tolist())
        for code in ("IEEE24", "IEEE39", "IEEE118", "UK"):
            net = load_case(code)
            branches = contract.branch_table(net)
            a, b = np.random.default_rng(1203), np.random.default_rng(1203)
            for k in [0, 1, 2, 5, 999] * 10:
                expected = [("line", int(net.line.index[i])) for i in original(a, len(net.line), k)]
                self.assertEqual(scope["_sample_contingency"](b, branches, k, scope["_outage_types"]("line")), expected)
            self.assertEqual(a.bit_generator.state, b.bit_generator.state)
        mixed = scope["_sample_contingency"](a, contract.branch_table(synthetic()), 99, ("impedance",))
        self.assertEqual(mixed, [("impedance", 0)])
        with self.assertRaises(argparse.ArgumentTypeError):
            scope["_outage_types"]("invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
