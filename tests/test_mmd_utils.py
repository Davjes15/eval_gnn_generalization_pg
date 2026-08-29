"""test_mmd_utils.py -- checks for the graph-distance layer (audit item A7).

Run:  python3 tests/test_mmd_utils.py     (no pytest dependency needed)

Three things must hold. The DEFAULT estimator has to stay bit-identical to what
produced the committed result CSVs, because naming the estimator honestly is a
documentation fix and must not silently move published numbers. The `unbiased`
option has to be the U-statistic, i.e. drop the diagonals from the within-sample
terms only. And the new electrical descriptor has to separate two systems that
differ in impedance but not in topology -- the case the topological descriptors
are provably blind to, which is the substance of the A7 finding.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch_geometric.data import Data

from mmd_utils import (_gaussian_gram, _median_bandwidth, degree_histogram,
                       evaluate_mmd, evaluate_mmd_electrical, mmd,
                       pyg_to_networkx, reactance_histogram)

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def ring(n, x_pu):
    """A ring of n buses; every branch carries reactance `x_pu`."""
    src = list(range(n))
    dst = [(i + 1) % n for i in range(n)]
    ei = torch.tensor([src + dst, dst + src], dtype=torch.int64)
    e = ei.shape[1]
    attr = torch.zeros(e, 4)
    attr[:, 1] = 0.1 * x_pu       # r_pu
    attr[:, 2] = x_pu             # x_pu
    return Data(x=torch.zeros(n, 7), edge_index=ei, edge_attr=attr,
                y=torch.zeros(n, 4))


def star(n, x_pu):
    """A star: bus 0 connected to every other bus. Same size, different topology."""
    src = [0] * (n - 1)
    dst = list(range(1, n))
    ei = torch.tensor([src + dst, dst + src], dtype=torch.int64)
    e = ei.shape[1]
    attr = torch.zeros(e, 4)
    attr[:, 2] = x_pu
    return Data(x=torch.zeros(n, 7), edge_index=ei, edge_attr=attr,
                y=torch.zeros(n, 4))


print("A. the default estimator is unchanged (published CSVs stay reproducible)")
rng = np.random.default_rng(0)
a = rng.normal(size=(30, 5))
b = rng.normal(loc=0.7, size=(25, 5))

gamma = 1.0 / _median_bandwidth(a, b)
expected = np.sqrt(max(
    _gaussian_gram(a, a, gamma).mean()
    + _gaussian_gram(b, b, gamma).mean()
    - 2.0 * _gaussian_gram(a, b, gamma).mean(), 0.0))
check("mmd() default equals the biased V-statistic formula",
      abs(mmd(a, b) - float(expected)) < 1e-12,
      f"{mmd(a, b):.12f} vs {float(expected):.12f}")

print("\nB. unbiased=True is the U-statistic")
kxx, kyy = _gaussian_gram(a, a, gamma), _gaussian_gram(b, b, gamma)
n_a, n_b = len(a), len(b)
off_a = (kxx.sum() - np.trace(kxx)) / (n_a * (n_a - 1))
off_b = (kyy.sum() - np.trace(kyy)) / (n_b * (n_b - 1))
expected_u = np.sqrt(max(off_a + off_b
                         - 2.0 * _gaussian_gram(a, b, gamma).mean(), 0.0))
check("mmd(unbiased=True) drops only the within-sample diagonals",
      abs(mmd(a, b, unbiased=True) - float(expected_u)) < 1e-12,
      f"{mmd(a, b, unbiased=True):.12f} vs {float(expected_u):.12f}")
check("the biased estimate is the larger of the two (diagonals are 1)",
      mmd(a, b) > mmd(a, b, unbiased=True),
      f"biased={mmd(a, b):.6f} unbiased={mmd(a, b, unbiased=True):.6f}")

print("\nC. identical distributions give (numerically) zero distance")
same = [ring(10, 0.05) for _ in range(8)]
md, ml = evaluate_mmd(same, same)
check("degree MMD of a distribution with itself is ~0", md < 1e-9, f"{md:.3e}")
check("laplacian MMD of a distribution with itself is ~0", ml < 1e-9, f"{ml:.3e}")
check("electrical MMD of a distribution with itself is ~0",
      evaluate_mmd_electrical(same, same) < 1e-9)

print("\nD. the topological descriptors are blind to impedance; the electrical one is not")
# Same graphs, reactances 100x apart: this is the scale shift A2/A7 are about.
low = [ring(10, 0.01) for _ in range(8)]
high = [ring(10, 1.0) for _ in range(8)]
md_scale, ml_scale = evaluate_mmd(low, high)
el_scale = evaluate_mmd_electrical(low, high)
check("degree MMD cannot see a 100x impedance shift", md_scale < 1e-9,
      f"{md_scale:.3e}")
check("laplacian MMD cannot see a 100x impedance shift", ml_scale < 1e-9,
      f"{ml_scale:.3e}")
check("electrical MMD does see it", el_scale > 0.1, f"{el_scale:.4f}")

# Converse: same reactance, different topology -> topological distance, no
# electrical distance. The two descriptor families are complementary.
star_ds = [star(10, 0.01) for _ in range(8)]
md_topo, _ = evaluate_mmd(low, star_ds)
el_topo = evaluate_mmd_electrical(low, star_ds)
check("degree MMD sees a ring-vs-star change", md_topo > 0.1, f"{md_topo:.4f}")
check("electrical MMD ignores it (same reactances)", el_topo < 1e-9,
      f"{el_topo:.3e}")

print("\nE. descriptor shapes are fixed, so unequal grids stay comparable")
h24 = degree_histogram(pyg_to_networkx(ring(24, 0.05)))
h118 = degree_histogram(pyg_to_networkx(ring(118, 0.05)))
check("degree histograms have equal length for 24 and 118 buses",
      h24.shape == h118.shape, f"{h24.shape} vs {h118.shape}")
r24 = reactance_histogram(ring(24, 0.05))
r118 = reactance_histogram(ring(118, 0.05))
check("reactance histograms have equal length for 24 and 118 buses",
      r24.shape == r118.shape, f"{r24.shape} vs {r118.shape}")
check("reactance histogram of an edgeless graph is zeros, not NaN",
      bool(np.all(reactance_histogram(
          Data(x=torch.zeros(2, 7), edge_index=torch.zeros(2, 0, dtype=torch.int64),
               edge_attr=torch.zeros(0, 4), y=torch.zeros(2, 4))) == 0)))

print("\n" + ("ALL CHECKS PASSED" if not FAILURES
              else f"{len(FAILURES)} FAILURE(S): {FAILURES}"))
sys.exit(1 if FAILURES else 0)
