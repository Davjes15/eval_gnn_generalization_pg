"""contingency_harvest.py -- Step 7 (D11): harvest real contingencies from
PowerGraph-Graph and map them onto our pandapower networks.

PURPOSE
    Instead of (or in addition to) random N-1/N-k outages, drive Step 3's
    data generation with the *actual* cascading-failure outage sets that
    PowerGraph-Graph ships. Those are credible, grid-specific contingencies
    (they are what PowerGraph simulated), so the topology distribution becomes
    more realistic and stratified toward consequential events.

WHY THIS IS DESIGN DECISION D11
    We reuse only PowerGraph-Graph's *topology* (which lines are out) and then
    RE-SOLVE the AC power flow ourselves (Step 3). We deliberately do NOT reuse
    their node/edge values or graph-level labels -- those are for a different
    (graph-level, cascading-failure) task. So this is a contingency *source*,
    not a data import.

HOW POWERGRAPH-GRAPH ENCODES AN OUTAGE  (mirrors code/dataset/powergrid.py)
    - `blist.mat` -> `bList`  : (n_branch, 2) from/to bus per branch, 1-indexed,
      in the same order as the grid's MATPOWER branch list. This is the branch
      ordering.
    - `Ef.mat`    -> `E_f_post`: one edge-feature matrix per sample. A branch that
      is part of the contingency has an ALL-ZERO feature row. PowerGraph detects
      outages exactly this way:  `cont = [j for j in ... if all(Ef[i][j] == 0)]`.
    So the outaged branch set for a sample is the set of all-zero rows, and each
    such row j maps via `bList[j]` to a (from_bus, to_bus) pair.

HOW WE MAP A PowerGraph BRANCH ONTO OUR pandapower NET
    from_mpc keeps MATPOWER bus order, so MATPOWER bus i (1-indexed) == pandapower
    bus i-1. We match a branch's (from_bus, to_bus) pair to a pandapower `line`
    (or `trafo`) with the same unordered bus pair. Parallel branches on the same
    bus pair are consumed one-by-one so an N-k set that hits two parallel lines
    maps to two distinct elements. If any branch cannot be matched the whole
    contingency is skipped (returns None) rather than silently mismatched.

DATA LOCATION (not committed -- ~2.7 GB figshare download; see README)
    <pg_graph_raw>/<name>/raw/{blist.mat, Ef.mat, ...}   with
    name in {uk, ieee24, ieee39, ieee118}.
        wget -O data.tar.gz "https://figshare.com/ndownloader/files/46619158"
        tar -xf data.tar.gz

NOTE ON DEPENDENCIES
    PowerGraph's .mat files are MATLAB v7.3 (HDF5), which scipy.io cannot read;
    they need `mat73`. That import is done lazily (only when actually harvesting)
    so the rest of the pipeline does not require mat73.
"""
from __future__ import annotations

import os

import numpy as np

# PowerGraph-Graph raw folder names (lower-case) keyed by our grid codes.
_CODE_TO_PG_NAME = {
    "IEEE24": "ieee24",
    "IEEE39": "ieee39",
    "IEEE118": "ieee118",
    "UK": "uk",
}


# --- pure logic (unit-testable without any files) --------------------------

def extract_contingencies(ef_samples) -> list[list[int]]:
    """Given an iterable of per-sample edge-feature matrices (each (n_branch, F)),
    return, for each sample, the list of branch indices whose feature row is
    all zeros (i.e. the outaged branches). Mirrors powergrid.py's `cont`."""
    out = []
    for f in ef_samples:
        f = np.asarray(f, dtype=float)
        cont = [j for j in range(f.shape[0]) if np.all(f[j] == 0)]
        out.append(cont)
    return out


def build_buspair_pool(net) -> dict:
    """Map frozenset({from_bus, to_bus}) -> list of (etype, element_index) for
    every branch (lines + transformers). Lists allow parallel branches."""
    pool: dict = {}
    for i, row in net.line.iterrows():
        key = frozenset((int(row["from_bus"]), int(row["to_bus"])))
        pool.setdefault(key, []).append(("line", int(i)))
    for i, row in net.trafo.iterrows():
        key = frozenset((int(row["hv_bus"]), int(row["lv_bus"])))
        pool.setdefault(key, []).append(("trafo", int(i)))
    return pool


def map_contingency(net, blist0: np.ndarray, branch_idxs: list[int]):
    """Map a set of PowerGraph branch indices to pandapower elements.

    blist0 : (n_branch, 2) array of 0-indexed (from_bus, to_bus) pairs.
    Returns list of (etype, index), or None if any branch is unmatched.
    """
    pool = {k: list(v) for k, v in build_buspair_pool(net).items()}
    mapped = []
    for j in branch_idxs:
        fb, tb = int(blist0[j][0]), int(blist0[j][1])
        key = frozenset((fb, tb))
        if pool.get(key):
            mapped.append(pool[key].pop())
        else:
            return None
    return mapped


def map_all_contingencies(net, blist0, contingencies, max_k=None, drop_empty=True):
    """Map many contingencies; drop unmatched ones (and, optionally, empties)."""
    out = []
    for c in contingencies:
        if drop_empty and len(c) == 0:
            continue
        if max_k is not None and len(c) > max_k:
            continue
        m = map_contingency(net, blist0, c)
        if m is not None:
            out.append(m)
    return out


# --- file I/O wrappers (need mat73 + the figshare download) -----------------

def _load_mat73(path):
    try:
        import mat73  # optional dependency, only for harvesting
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "harvesting PowerGraph-Graph contingencies needs `mat73` "
            "(pip install mat73) to read MATLAB v7.3 .mat files."
        ) from e
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download PowerGraph-Graph raw data (~2.7 GB) and "
            f"point --pg_graph_raw at its root (see contingency_harvest.py header)."
        )
    return mat73.loadmat(path)


def _grid_raw_dir(pg_graph_raw: str, code: str) -> str:
    name = _CODE_TO_PG_NAME[code]
    return os.path.join(pg_graph_raw, name, "raw")


def load_branch_list(pg_graph_raw: str, code: str) -> np.ndarray:
    """0-indexed (n_branch, 2) from/to bus pairs from blist.mat."""
    d = _load_mat73(os.path.join(_grid_raw_dir(pg_graph_raw, code), "blist.mat"))
    return np.asarray(d["bList"], dtype=int) - 1


def load_edge_feature_samples(pg_graph_raw: str, code: str):
    """Per-sample edge-feature matrices from Ef.mat (E_f_post[i][0])."""
    d = _load_mat73(os.path.join(_grid_raw_dir(pg_graph_raw, code), "Ef.mat"))
    ef = d["E_f_post"]
    return [np.asarray(ef[i][0], dtype=float) for i in range(len(ef))]


def harvest_contingencies(net, pg_graph_raw: str, code: str, max_k=None):
    """End-to-end: read PowerGraph-Graph raw files and return a list of
    contingencies as pandapower element lists [(etype, idx), ...] for `code`."""
    blist0 = load_branch_list(pg_graph_raw, code)
    ef = load_edge_feature_samples(pg_graph_raw, code)
    conts = extract_contingencies(ef)
    return map_all_contingencies(net, blist0, conts, max_k=max_k)
