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
import contextlib
import io
import json
from pathlib import Path
import tempfile
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

from mmd_utils import (_gaussian_gram, _median_bandwidth, degree_histogram,
                       evaluate_mmd, evaluate_mmd_electrical, mmd,
                       pyg_to_networkx, reactance_histogram,
                       laplacian_spectrum_histogram, reference_bandwidth,
                       _reference_bandwidth_stats, dataset_fingerprint,
                       load_reference_bandwidth, resolve_mmd_bandwidth)

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

print("\nF. circuit-counting graph mode")


def circuit_data(edges, n):
    directed = edges + [(v, u) for u, v in edges]
    return Data(x=torch.zeros(n, 7),
                edge_index=torch.tensor(directed, dtype=torch.long).T.reshape(2, -1))


parallel = circuit_data([(0, 1), (0, 1), (1, 2)], 3)
simple = pyg_to_networkx(parallel, "simple")
multi = pyg_to_networkx(parallel)
check("simple degrees", list(dict(simple.degree()).values()) == [1, 2, 1])
check("multi degrees", list(dict(multi.degree()).values()) == [2, 3, 1])
hand = np.array([[1, -2 / np.sqrt(6), 0],
                 [-2 / np.sqrt(6), 1, -1 / np.sqrt(3)],
                 [0, -1 / np.sqrt(3), 1]])
check("weighted normalized Laplacian equals hand computation",
      np.allclose(nx.normalized_laplacian_matrix(multi).toarray(), hand))
check("weighted spectrum equals hand computation",
      np.allclose(nx.normalized_laplacian_spectrum(multi), np.linalg.eigvalsh(hand)))
# A three-node weighted path always has spectrum {0, 1, 2}; use four nodes
# to test that the spectral descriptor can register a parallel-line outage.
before = circuit_data([(0, 1), (0, 1), (1, 2), (2, 3)], 4)
after = circuit_data([(0, 1), (1, 2), (2, 3)], 4)
for mode in ("simple", "multi"):
    gb, ga = pyg_to_networkx(before, mode), pyg_to_networkx(after, mode)
    check(f"{mode} remains connected after one parallel outage", nx.is_connected(ga))
    for fn in (degree_histogram, laplacian_spectrum_histogram):
        equal = np.array_equal(fn(gb), fn(ga))
        check(f"{mode} outage {fn.__name__}", equal == (mode == "simple"))
for fn in (degree_histogram, laplacian_spectrum_histogram):
    check(f"no parallel circuits: {fn.__name__}", np.array_equal(
        fn(pyg_to_networkx(after)), fn(pyg_to_networkx(after, "simple"))))
check("explicit simple mode retains legacy MMD exactly",
      evaluate_mmd(low, star_ds, graph_mode="simple") ==
      evaluate_mmd(low, star_ds))
empty = circuit_data([], 0)
check("empty Laplacian histogram is zero",
      np.array_equal(laplacian_spectrum_histogram(pyg_to_networkx(empty)), np.zeros(40)))
bad = Data(x=torch.zeros(3, 7), edge_index=torch.tensor([[0, 1], [1, 2]]))
for name, operation in [
        ("unpaired directions", lambda: pyg_to_networkx(bad)),
        ("self-loop", lambda: pyg_to_networkx(circuit_data([(0, 0)], 1))),
        ("invalid mode", lambda: pyg_to_networkx(parallel, "invalid")),
        ("degree overflow", lambda: degree_histogram(multi, max_degree=2))]:
    try:
        operation()
    except ValueError:
        check(name + " raises ValueError", True)
    else:
        check(name + " raises ValueError", False)

print("\nG. shared reference width uses the exact nonzero-distance median")


def expect_error(error, operation, label):
    try:
        operation()
    except error:
        check(label, True)
    else:
        check(label, False)


tiny = np.array([[0.0], [0.0], [2.0], [5.0]])
check("hand-computed nonzero median is 9 (zeros excluded)",
      reference_bandwidth([tiny[:2], tiny[2:]]) == 9.0)
expect_error(ValueError, lambda: reference_bandwidth([np.ones((5, 3))]),
             "all identical reference rows raise")
random_rows = rng.normal(size=(301, 5))
random_rows[-3:] = random_rows[0]
sq = np.sum((random_rows[:, None] - random_rows[None, :]) ** 2, axis=-1)
naive = sq[np.triu_indices(len(random_rows), 1)]
naive_median = np.median(naive[naive > 0])
for chunk in (1, 7, 256):
    width, fraction = _reference_bandwidth_stats([random_rows], chunk_size=chunk)
    check(f"chunk={chunk} equals naive median",
          np.isclose(width, naive_median, rtol=1e-14, atol=0))
    check(f"chunk={chunk} counts only off-diagonal zero distances",
          np.isclose(fraction, np.mean(naive == 0)))

print("\nH. fixed-width kernel, legacy fallback, validation and complementarity")
width = 3.25
for left, right in ((a, b), (a[:7], b[:4]), (b, a)):
    hand_mmd = np.sqrt(max(
        _gaussian_gram(left, left, 1 / width).mean()
        + _gaussian_gram(right, right, 1 / width).mean()
        - 2 * _gaussian_gram(left, right, 1 / width).mean(), 0))
    check("each pair uses the same gamma=1/m",
          mmd(left, right, bandwidth=width) == float(hand_mmd))
check("explicit pair is bit-identical to section A",
      mmd(a, b, bandwidth="pair") == mmd(a, b) == float(expected))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    check("legacy zero-median fallback still returns 1.0",
          _median_bandwidth(np.zeros((3, 1)), np.zeros((2, 1))) == 1.0)
check("legacy fallback emits a warning",
      any("fallback 1.0" in str(w.message) for w in caught))
for invalid in (0, -1, float("nan"), float("inf"), "reference", "bad", None, True):
    expect_error(ValueError, lambda v=invalid: mmd(a, b, bandwidth=v),
                 f"invalid bandwidth {invalid!r} raises")
expect_error(KeyError, lambda: evaluate_mmd(low, high, bandwidth={"degree": 1.0}),
             "missing Laplacian width raises KeyError")
expect_error(KeyError, lambda: evaluate_mmd_electrical(low, high, bandwidth={}),
             "missing reactance width raises KeyError")
fixed = {"degree": width, "laplacian": width, "reactance": width}
md, ml = evaluate_mmd(same, same, bandwidth=fixed)
check("fixed-width self-distance is ~0 for degree and Laplacian", md < 1e-9 and ml < 1e-9)
check("fixed-width electrical self-distance is ~0",
      evaluate_mmd_electrical(same, same, bandwidth=fixed) < 1e-9)
md, ml = evaluate_mmd(low, high, bandwidth=fixed)
check("fixed-width topology is still blind to a 100x impedance shift",
      md < 1e-9 and ml < 1e-9)
check("fixed-width reactance still sees the impedance shift",
      evaluate_mmd_electrical(low, high, bandwidth=fixed) > 0.1)
md, _ = evaluate_mmd(low, star_ds, bandwidth=fixed)
check("fixed-width degree sees topology; reactance ignores it",
      md > 0.1 and evaluate_mmd_electrical(low, star_ds, bandwidth=fixed) < 1e-9)

print("\nI. reference loading is fail-loud, with intentional extra-dataset reuse")
with tempfile.TemporaryDirectory() as tmp:
    folder = Path(tmp) / "reference_data"
    path = Path(tmp) / "ruler.json"
    fingerprints = {}
    for grid in ("IEEE24", "IEEE39", "IEEE118", "UK"):
        for split in ("train", "test"):
            dataset = folder / grid / split / "dataset.pt"
            dataset.parent.mkdir(parents=True)
            dataset.write_bytes(f"{grid}/{split}".encode())
            fingerprints[f"{folder}/{grid}/{split}"] = dataset_fingerprint(dataset)
    payload = {"multi": {"degree": 1.0, "laplacian": 2.0},
               "simple": {"degree": 3.0, "laplacian": 4.0}, "reactance": 5.0,
               "data_dirs": [str(folder)], "splits": ["train", "test"],
               "fingerprint": fingerprints}
    expect_error(FileNotFoundError, lambda: load_reference_bandwidth(path, [folder]),
                 "missing ruler raises (no pair fallback)")
    path.write_text(json.dumps(payload))
    check("matching reference fingerprints accepted",
          load_reference_bandwidth(path, [folder]) == payload)
    check("mode-specific widths resolved with common reactance",
          resolve_mmd_bandwidth("reference", path, [folder], "simple") ==
          {"degree": 3.0, "laplacian": 4.0, "reactance": 5.0})
    # Pair reproduction must not need the reference JSON.
    check("pair opt-in ignores a nonexistent reference file",
          resolve_mmd_bandwidth("pair", path.with_name("missing.json"), [folder]) == "pair")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        extra = load_reference_bandwidth(path, [Path(tmp) / "optional_blocked"])
    check("unlisted folder reuses unchanged ruler and logs one line",
          extra == payload and len(output.getvalue().splitlines()) == 1
          and "reusing" in output.getvalue())
    dataset = folder / "UK" / "train" / "dataset.pt"
    dataset.write_bytes(b"regenerated")
    expect_error(ValueError, lambda: load_reference_bandwidth(path, [folder]),
                 "changed reference data raises fingerprint mismatch")
    del payload["multi"]
    path.write_text(json.dumps(payload))
    expect_error(KeyError, lambda: load_reference_bandwidth(path, [folder]),
                 "missing graph-mode key raises")
    payload["multi"] = {"degree": 1.0}
    path.write_text(json.dumps(payload))
    expect_error(KeyError, lambda: load_reference_bandwidth(path, [folder]),
                 "missing descriptor key raises")

print("\n" + ("ALL CHECKS PASSED" if not FAILURES
              else f"{len(FAILURES)} FAILURE(S): {FAILURES}"))
sys.exit(1 if FAILURES else 0)
