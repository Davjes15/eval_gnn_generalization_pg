"""transmission_graph_gen.py -- Step 3: generate the datasets (the heart).

PURPOSE
    Turn each base transmission grid into a *distribution of topologies* by
    sampling credible N-1/N-k line contingencies, applying real hourly demand,
    RE-SOLVING the AC power flow, and emitting ENGAGE-format `Data` objects. This
    is what makes ENGAGE's MMD / g-score well-posed (a cloud of graphs per grid,
    not a single fixed topology).

WHY THIS STEP EXISTS (design decisions D2, D6, D9, D10, D11)
    - D10 (re-solve engine): an outage changes the physics at *every* bus, so we
      cannot edit stored tensors -- we must RE-SOLVE. For each sample we take
      lines out of service, set demand, and call `pp.runpp` (AC power flow,
      Newton-Raphson). The fresh `res_bus`/`res_line` are the new labels.
    - D2 (real demand, "Route B"): active demand per bus comes from PowerGraph's
      `hourlyDemandBusnew` (reactive demand kept at base, mirroring PowerGraph's
      gendataopf.m).
    - D6 (masking): node inputs are masked by bus type via the vendored ENGAGE
      contract (engage_contract.py).
    - D9 (comparability): features are ENGAGE per-unit; downstream normalization
      is handled in training, not here.
    - D11 (harvest contingencies): the contingency sampler is pluggable so a
      future option can inject outage sets harvested from PowerGraph-Graph
      instead of random N-k.

HOW IT CONNECTS
    transmission_grids.load_case / load_hourly_demand   (Step 2)
        -> [sample demand + contingency -> pp.runpp -> filter]
        -> engage_contract.get_node_features / get_edge_features
        -> torch_geometric.data.Data
        -> data/<CODE>/<split>/dataset.pt        (ENGAGE's expected layout)
    The datasets are consumed by the experiment drivers (Step 5), which reuse
    ENGAGE's training loop, MMD and g-score unchanged.

MODELING CHOICE (documented, see design doc "PF vs OPF")
    Default post-contingency solve is PF-with-slack (`pp.runpp`): generator
    setpoints are held and the slack bus absorbs the imbalance. Pass
    `--redispatch` to instead run AC OPF (`pp.runopp`), which re-optimizes
    generation (more realistic, heavier, needs cost data).

HOW TO RUN (quick smoke test)
    python3 transmission_graph_gen.py --grid IEEE24 --n_train 20 --n_val 5 \
        --n_test 5 --max_k 2 --out_dir data
Full grids: run once per code in {IEEE24, IEEE39, IEEE118, UK} (or --grid all).
"""
from __future__ import annotations

import argparse
import os
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandapower as pp
import pandapower.topology as top
import networkx as nx
import torch
from torch_geometric.data import Data

from engage_contract import get_node_features, get_edge_features
from transmission_grids import (
    get_transmission_grid_codes,
    load_case,
    load_hourly_demand,
)


def _apply_demand(net, demand_col: np.ndarray) -> None:
    """Set each load's ACTIVE power from a per-bus demand column (MW).

    Reactive demand is left at the base-case value, mirroring PowerGraph's
    gendataopf.m (which only varies PD). `demand_col` is indexed by bus id
    (row i -> bus i), which is valid because from_mpc yields 0..N-1 bus ids in
    original order.
    """
    for load_idx, bus in net.load["bus"].items():
        net.load.at[load_idx, "p_mw"] = float(demand_col[bus])


def _sample_contingency(rng: np.random.Generator, n_lines: int, k: int) -> list[int]:
    """Return k distinct line positions to take out of service (N-k)."""
    if k <= 0:
        return []
    k = min(k, n_lines)
    return sorted(rng.choice(n_lines, size=k, replace=False).tolist())


def _is_connected(net) -> bool:
    """True if the in-service network is a single connected component."""
    g = top.create_nxgraph(net, respect_switches=False, include_out_of_service=False)
    if g.number_of_nodes() == 0:
        return False
    return nx.is_connected(g)


def _build_sample(net):
    """Extract one ENGAGE `Data` object from a solved net (+ its DC baseline)."""
    X_i, Y_i = get_node_features(net)
    A_i, E_i = get_edge_features(net)

    # DC power-flow baseline (stored once so evaluation need not recompute it).
    dc_net = deepcopy(net)
    pp.rundcpp(dc_net)
    np_dc = dc_net.res_bus[["p_mw", "q_mvar", "vm_pu", "va_degree"]].values
    dc_pf = torch.nan_to_num(torch.tensor(np_dc, dtype=torch.float32), nan=0.0)

    return Data(
        x=torch.tensor(X_i, dtype=torch.float32),
        edge_index=torch.tensor(A_i, dtype=torch.int64),
        edge_attr=torch.tensor(E_i, dtype=torch.float32),
        y=torch.tensor(Y_i, dtype=torch.float32),
        dc_pf=dc_pf,
    )


def generate_dataset(
    code: str,
    n_samples: int,
    rng: np.random.Generator,
    max_k: int = 2,
    k_probs: list[float] | None = None,
    redispatch: bool = False,
    vm_min: float = 0.8,
    vm_max: float = 1.2,
    max_tries_factor: int = 50,
):
    """Generate `n_samples` valid (demand, contingency) operating points for `code`.

    Returns (list[Data], list[dict]) -- samples and their metadata.
    Each sample: random demand snapshot + an N-k line outage (k drawn from
    {0..max_k}), re-solved with AC power flow, filtered for convergence,
    connectivity and voltage sanity.
    """
    base = load_case(code)
    demand = load_hourly_demand(code)
    n_lines = len(base.line)
    n_time = demand.shape[1]
    # k=0 anchors the base topology; higher k spreads the topology distribution.
    ks = np.arange(0, max_k + 1)
    if k_probs is None:
        k_probs = np.ones(len(ks)) / len(ks)

    samples, metas = [], []
    tries = 0
    max_tries = n_samples * max_tries_factor
    while len(samples) < n_samples and tries < max_tries:
        tries += 1
        net = deepcopy(base)

        # 1) demand snapshot
        t = int(rng.integers(0, n_time))
        _apply_demand(net, demand[:, t])

        # 2) contingency
        k = int(rng.choice(ks, p=k_probs))
        out_lines = _sample_contingency(rng, n_lines, k)
        if out_lines:
            net.line.loc[net.line.index[out_lines], "in_service"] = False

        # 3) reject islanding before solving
        if not _is_connected(net):
            continue

        # 4) re-solve AC power flow (or OPF redispatch)
        try:
            if redispatch:
                pp.runopp(net)
            else:
                pp.runpp(net)
        except Exception:
            continue
        if not net.converged:
            continue

        # 5) physical sanity filter on voltage magnitude
        vm = net.res_bus["vm_pu"].values
        if np.any(vm < vm_min) or np.any(vm > vm_max) or np.any(~np.isfinite(vm)):
            continue

        samples.append(_build_sample(net))
        metas.append({"grid": code, "t_idx": t, "k": k, "out_lines": out_lines})

    if len(samples) < n_samples:
        print(
            f"  [warn] {code}: only {len(samples)}/{n_samples} samples after "
            f"{tries} tries (try raising --max_tries_factor or relaxing filters)."
        )
    return samples, metas


def _save_split(out_dir: str, code: str, split: str, samples, metas):
    split_dir = os.path.join(out_dir, code, split)
    os.makedirs(split_dir, exist_ok=True)
    torch.save(samples, os.path.join(split_dir, "dataset.pt"))
    pd.DataFrame(metas).to_csv(os.path.join(split_dir, "dataset_src.csv"), index=False)
    print(f"  saved {len(samples):4d} -> {os.path.join(split_dir, 'dataset.pt')}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", default="all",
                   help="grid code or 'all' (IEEE24/IEEE39/IEEE118/UK)")
    p.add_argument("--n_train", type=int, default=800)
    p.add_argument("--n_val", type=int, default=100)
    p.add_argument("--n_test", type=int, default=100)
    p.add_argument("--max_k", type=int, default=2,
                   help="maximum number of simultaneous line outages (N-k)")
    p.add_argument("--redispatch", action="store_true",
                   help="use AC OPF (runopp) instead of PF-with-slack (runpp)")
    p.add_argument("--out_dir", default="data")
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--max_tries_factor", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    codes = get_transmission_grid_codes() if args.grid == "all" else [args.grid]
    for code in codes:
        print(f"[{code}] generating "
              f"(train={args.n_train}, val={args.n_val}, test={args.n_test}, "
              f"max_k={args.max_k}, redispatch={args.redispatch})")
        # One RNG per grid, seeded deterministically, so runs are reproducible.
        rng = np.random.default_rng(args.seed + hash(code) % 10_000)
        for split, n in [("train", args.n_train), ("val", args.n_val), ("test", args.n_test)]:
            if n <= 0:
                continue
            samples, metas = generate_dataset(
                code, n, rng, max_k=args.max_k, redispatch=args.redispatch,
                max_tries_factor=args.max_tries_factor,
            )
            _save_split(args.out_dir, code, split, samples, metas)


if __name__ == "__main__":
    main()
