"""validate.py -- Step 6: validation gates.

PURPOSE
    Cheap, automatic checks that catch the failure modes that would make the
    study INVALID rather than merely buggy. Run this before trusting any results.

WHY (design decisions D5 + D9)
    Two classes of silent invalidity motivated these gates:
      * conversion / contract errors (a grid that does not solve, tensors with the
        wrong shape, masking applied to the wrong columns) -- Step 1/3 fidelity.
      * a DEGENERATE MMD (the exact bug in the earlier engage_pg v2, where the
        Laplacian MMD was a constant for every different-grid pair). We assert the
        topology actually varies within a grid and that MMD is non-degenerate and
        orders within-grid < cross-grid.

WHAT IT CHECKS
    A. Conversion fidelity  -- every grid loads and its base AC power flow solves.
    B. Data contract        -- x (N,7), edge_index (2,2E), edge_attr (2E,4),
                               y (N,4), dc_pf (N,4); NaNs only in inputs.
    C. Masking correctness  -- masked input columns are NaN per bus type.
    D. Topology variation   -- contingencies actually change the edge count.
    E. MMD non-degeneracy   -- within-grid MMD < cross-grid MMD, not constant.

HOW TO RUN
    python3 validate.py                       # conversion checks only
    python3 validate.py --data_dir data       # + data/contract/topology/MMD checks
Exit code is non-zero if any gate fails (usable in CI).
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

import pandapower as pp

from transmission_grids import get_transmission_grid_codes, load_case
from training_utils import load_grid_dataset

# Expected (buses, in-service branches after conversion). Recorded from Step 1/2.
EXPECTED = {
    "IEEE24": (24, 38),
    "IEEE39": (39, 46),
    "IEEE118": (118, 184),
    "UK": (29, 90),
}


class Gate:
    def __init__(self):
        self.failures = []

    def check(self, name, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
        if not ok:
            self.failures.append(name)
        return ok


def gate_conversion(g: Gate, grids):
    print("\nA. Conversion fidelity (each grid loads + base AC power flow solves)")
    for code in grids:
        net = load_case(code)
        n_bus = len(net.bus)
        n_branch = int(net.line.in_service.sum() + net.trafo.in_service.sum())
        try:
            pp.runpp(net)
            converged = bool(net.converged)
        except Exception as e:  # noqa: BLE001
            converged = False
        exp = EXPECTED.get(code)
        detail = f"buses={n_bus} branches={n_branch} converged={converged}"
        ok = converged and (exp is None or (n_bus == exp[0]))
        g.check(f"{code} converts & solves", ok, detail)


def gate_contract(g: Gate, data_dir, grids):
    print("\nB. Data contract (tensor shapes + NaN placement)")
    for code in grids:
        try:
            ds = load_grid_dataset(data_dir, code, "train")
        except FileNotFoundError:
            g.check(f"{code} train dataset present", False, "run Step 3 first")
            continue
        d = ds[0]
        N = d.x.shape[0]
        E2 = d.edge_index.shape[1]
        ok_shapes = (
            d.x.shape == (N, 7)
            and d.edge_attr.shape == (E2, 4)
            and d.y.shape == (N, 4)
            and d.dc_pf.shape == (N, 4)
            and d.edge_index.shape == (2, E2)
        )
        g.check(f"{code} tensor shapes", ok_shapes,
                f"x={tuple(d.x.shape)} ei={tuple(d.edge_index.shape)} "
                f"ea={tuple(d.edge_attr.shape)} y={tuple(d.y.shape)}")
        # y must be fully known (no NaNs); x must carry masked NaNs.
        g.check(f"{code} y has no NaN", not bool(torch.isnan(d.y).any()))
        g.check(f"{code} x carries masked NaN inputs", bool(torch.isnan(d.x).any()))


def gate_masking(g: Gate, data_dir, grids):
    print("\nC. Masking correctness (NaN columns match bus type)")
    for code in grids:
        try:
            ds = load_grid_dataset(data_dir, code, "train")
        except FileNotFoundError:
            continue
        d = ds[0]
        x = d.x
        ok = True
        for row in x:
            slack, pv, pq = bool(row[0]), bool(row[1]), bool(row[2])
            p, q, vm, va = [bool(torch.isnan(row[i])) for i in (3, 4, 5, 6)]
            if slack:            # p_mw, q_mvar unknown
                ok &= p and q
            elif pv:             # q_mvar, va unknown
                ok &= q and va
            elif pq:             # vm, va unknown
                ok &= vm and va
        g.check(f"{code} masking matches bus type", ok)


def gate_topology_variation(g: Gate, data_dir, grids):
    print("\nD. Topology variation (contingencies change the edge count)")
    for code in grids:
        try:
            ds = load_grid_dataset(data_dir, code, "train")
        except FileNotFoundError:
            continue
        edge_counts = sorted({int(d.edge_index.shape[1]) for d in ds})
        g.check(f"{code} topology varies across samples", len(edge_counts) > 1,
                f"distinct 2E = {edge_counts}")


def gate_mmd(g: Gate, data_dir, grids):
    print("\nE. MMD non-degeneracy (within-grid < cross-grid, not constant)")
    from mmd_utils import evaluate_mmd

    if len(grids) < 2:
        g.check("MMD needs >=2 grids", False, "provide at least two grids' data")
        return
    data = {}
    for code in grids:
        try:
            data[code] = {
                "train": load_grid_dataset(data_dir, code, "train"),
                "test": load_grid_dataset(data_dir, code, "test"),
            }
        except FileNotFoundError:
            g.check(f"{code} data present for MMD", False)
            return

    within, cross = [], []
    for a in grids:
        for b in grids:
            _, ml = evaluate_mmd(data[a]["train"], data[b]["test"])
            (within if a == b else cross).append(ml)
    within_mean, cross_mean = float(np.mean(within)), float(np.mean(cross))
    g.check("within-grid MMD < cross-grid MMD", within_mean < cross_mean,
            f"within={within_mean:.4f} cross={cross_mean:.4f}")
    # Non-degenerate = cross-grid values are not all (nearly) identical.
    g.check("cross-grid MMD not constant (non-degenerate)",
            float(np.std(cross)) > 1e-6, f"std(cross)={np.std(cross):.4g}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default=None,
                   help="if given, also run data/contract/topology/MMD gates")
    p.add_argument("--grids", nargs="+", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    grids = args.grids or get_transmission_grid_codes()
    g = Gate()

    gate_conversion(g, grids)
    if args.data_dir:
        gate_contract(g, args.data_dir, grids)
        gate_masking(g, args.data_dir, grids)
        gate_topology_variation(g, args.data_dir, grids)
        gate_mmd(g, args.data_dir, grids)
    else:
        print("\n(Skipping data-dependent gates B-E; pass --data_dir to enable.)")

    print("\n" + ("=" * 50))
    if g.failures:
        print(f"VALIDATION FAILED: {len(g.failures)} gate(s) failed: {g.failures}")
        sys.exit(1)
    print("ALL GATES PASSED")


if __name__ == "__main__":
    main()
