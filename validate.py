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
    H. Split hygiene        -- no demand snapshot shared between splits, no
                               repeated (snapshot, outage) scenario, and with
                               --expect_blocked, disjoint contiguous time
                               windows per split (audit item A5).

FIXED-TOPOLOGY REGIME (`--regime a`)
    The fixed-topology control arm inverts gate D: every sample in a grid must
    share ONE topology. Gates D/E are replaced by
    D'. Topology invariance   -- identical edge_index and edge_attr everywhere.
    F.  Contingency metadata  -- every row has k == 0 and no outaged branch.
    G.  Split disjointness    -- no demand snapshot shared between two splits
                                 (with one topology, a repeat is an exact
                                 duplicate sample, i.e. test leakage).

HOW TO RUN
    python3 validate.py                       # conversion checks only
    python3 validate.py --data_dir data       # + data/contract/topology/MMD checks
    python3 validate.py --data_dir data_a --regime a   # fixed-topology gates
Exit code is non-zero if any gate fails (usable in CI).
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import pandas as pd

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


def gate_topology_invariance(g: Gate, data_dir, grids):
    print("\nD'. Topology invariance (fixed-topology regime: one topology per grid)")
    for code in grids:
        try:
            ds = [d for split in ("train", "val", "test")
                  for d in load_grid_dataset(data_dir, code, split)]
        except FileNotFoundError:
            g.check(f"{code} dataset present", False, "generate the regime first")
            continue
        ref = ds[0]

        def _same(a, b):
            # NaN-aware: edge_attr's sc_voltage column is NaN on non-transformer
            # branches, and NaN != NaN under plain equality.
            return (a.shape == b.shape
                    and bool(((a == b) | (a.isnan() & b.isnan())).all()))

        same_index = all(torch.equal(d.edge_index, ref.edge_index) for d in ds)
        same_attr = all(_same(d.edge_attr, ref.edge_attr) for d in ds)
        g.check(f"{code} identical edge_index across all samples", same_index,
                f"n={len(ds)} 2E={int(ref.edge_index.shape[1])}")
        g.check(f"{code} identical edge_attr across all samples", same_attr)


def _load_meta(data_dir, code, split):
    path = os.path.join(data_dir, code, split, "dataset_src.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def gate_contingency_metadata(g: Gate, data_dir, grids):
    print("\nF. Contingency metadata (every sample is the base topology)")
    for code in grids:
        frames = [m for m in (_load_meta(data_dir, code, s)
                              for s in ("train", "val", "test")) if m is not None]
        if not frames:
            g.check(f"{code} metadata present", False)
            continue
        meta = pd.concat(frames, ignore_index=True)
        ks = sorted(meta["k"].unique().tolist())
        # `out_lines` round-trips through CSV as the string "[]" when empty.
        no_outage = meta["out_lines"].astype(str).str.strip().isin(["[]", ""]).all()
        g.check(f"{code} all samples k == 0", ks == [0], f"observed k = {ks}")
        g.check(f"{code} no branch outaged", bool(no_outage))


def gate_split_disjointness(g: Gate, data_dir, grids):
    print("\nG. Split disjointness (no demand snapshot shared between splits)")
    for code in grids:
        sets = {}
        for split in ("train", "val", "test"):
            meta = _load_meta(data_dir, code, split)
            if meta is not None:
                sets[split] = set(meta["t_idx"].tolist())
        if len(sets) < 2:
            g.check(f"{code} metadata present for all splits", False)
            continue
        overlaps = {f"{a}|{b}": len(sets[a] & sets[b])
                    for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
                    if a in sets and b in sets}
        g.check(f"{code} splits share no demand snapshot",
                all(v == 0 for v in overlaps.values()), str(overlaps))
        # Duplicates *within* a split are exact duplicate samples here too.
        for split, s in sets.items():
            meta = _load_meta(data_dir, code, split)
            g.check(f"{code}/{split} no duplicate demand snapshot",
                    len(s) == len(meta), f"{len(s)} distinct of {len(meta)}")


def gate_split_hygiene(g: Gate, data_dir, grids, expect_blocked: bool):
    """H. Split hygiene for the varying-topology regime (audit item A5).

    With topology varying, a repeated demand snapshot is no longer an exact
    duplicate, so gate G's criterion is too weak on its own. What must hold is:
      * no demand snapshot is shared between two splits (a shared snapshot means
        the test set re-uses an operating point the model was fitted on);
      * no exact (snapshot, outage-set) scenario is repeated inside a split;
      * with --expect_blocked, each split's snapshots occupy a contiguous window
        disjoint from the others', so a test snapshot is not the 15-minute
        neighbour of a training one either.
    """
    print("\nH. Split hygiene (demand snapshots and scenarios across splits)")
    for code in grids:
        metas = {s: _load_meta(data_dir, code, s)
                 for s in ("train", "val", "test")}
        if any(m is None for m in metas.values()):
            g.check(f"{code} metadata present for all splits", False)
            continue
        sets = {s: set(m["t_idx"].tolist()) for s, m in metas.items()}
        overlaps = {f"{a}|{b}": len(sets[a] & sets[b])
                    for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
        g.check(f"{code} splits share no demand snapshot",
                all(v == 0 for v in overlaps.values()), str(overlaps))
        for split, meta in metas.items():
            pairs = list(zip(meta["t_idx"].tolist(),
                             meta["out_lines"].astype(str).tolist()))
            g.check(f"{code}/{split} no repeated scenario",
                    len(set(pairs)) == len(pairs),
                    f"{len(set(pairs))} distinct of {len(pairs)}")
        if expect_blocked:
            spans = {s: (min(v), max(v)) for s, v in sets.items()}
            ordered = sorted(spans.items(), key=lambda kv: kv[1][0])
            disjoint = all(ordered[i][1][1] < ordered[i + 1][1][0]
                           for i in range(len(ordered) - 1))
            g.check(f"{code} split time windows are disjoint", disjoint, str(spans))


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
    p.add_argument("--regime", choices=["a", "b"], default="b",
                   help="'b' (default): topology varies -- gates B-E, H. "
                        "'a': fixed topology -- gates B, C, D', F, G.")
    p.add_argument("--expect_blocked", action="store_true",
                   help="additionally require each split's demand snapshots to "
                        "occupy a contiguous window disjoint from the other "
                        "splits' (datasets built with --time_split blocked)")
    return p.parse_args()


def main():
    args = parse_args()
    grids = args.grids or get_transmission_grid_codes()
    g = Gate()

    gate_conversion(g, grids)
    if args.data_dir:
        gate_contract(g, args.data_dir, grids)
        gate_masking(g, args.data_dir, grids)
        if args.regime == "a":
            gate_topology_invariance(g, args.data_dir, grids)
            gate_contingency_metadata(g, args.data_dir, grids)
            gate_split_disjointness(g, args.data_dir, grids)
            print("\n(Gate E skipped: with one topology per grid the within-grid "
                  "MMD is 0 by construction and the g-score is undefined.)")
        else:
            gate_topology_variation(g, args.data_dir, grids)
            gate_split_hygiene(g, args.data_dir, grids, args.expect_blocked)
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
