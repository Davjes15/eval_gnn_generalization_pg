"""transmission_grids.py -- Step 2 of the Layer 2 pipeline: load the grids.

PURPOSE
    Provide a single, well-documented entry point for loading the four
    transmission grids (as pandapower networks) and their hourly demand
    profiles, so that Step 3 (data generation) never has to know anything about
    MATPOWER/`.mat` file formats.

WHY THIS STEP EXISTS (design decisions D1 + D4; see
    docs/PowerGraph_to_ENGAGE_design_decisions.md)
    - D1: the source of truth is PowerGraph's raw `System.m` grids, converted
      once to `.mat` in Step 1 (transmission/cases/<CODE>.mat).
    - D4: we load PowerGraph's OWN cases (not pandapower's built-in IEEE cases)
      so the grids are identical to the ones PowerGraph trained on.
    pandapower's MATPOWER converter (`from_mpc`) does the semantic mapping we
    rely on downstream:
      * bus type 3 (ref)  -> net.ext_grid   (the slack bus)
      * bus type 2 (PV)   -> net.gen        (voltage-controlled generators)
      * bus Pd/Qd != 0    -> net.load
      * branch with tap   -> net.trafo, otherwise net.line
    ENGAGE's feature extractors (Step 3) read exactly these tables, so getting
    the mapping right here is what makes the ENGAGE data contract work.

HOW IT CONNECTS
    transmission/cases/<CODE>.mat  (Step 1)
        -> load_case(CODE)                 -> pandapower net (re-solvable model)
    PowerGraph-Node/.../hourlyDemandBusnew.mat
        -> load_hourly_demand(CODE)        -> ndarray (N_bus, T) of MW
    Both are consumed by transmission_graph_gen.py (Step 3).

DEMAND SEMANTICS (mirrors PowerGraph's gendataopf.m)
    PowerGraph sets each bus's ACTIVE demand `PD` from `hourlyDemandBusnew` per
    time step and leaves REACTIVE demand `QD` at its base-case value. We follow
    the same convention in Step 3. `hourlyDemandBusnew` is (N_bus, 35040), i.e.
    15-minute resolution over a year, in MW; this is the exact file PowerGraph
    used (there is also a coarser `hourlyDemandBus.mat`).

REFERENCES
    pandapower from_mpc:
      https://pandapower.readthedocs.io/en/latest/converter/matpower.html
    PowerGraph-Node gendataopf.m (demand handling): lines ~31-46.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import scipy.io as sio

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# pandapower 3.x moved the MATPOWER importer to this submodule.
from pandapower.converter.matpower.from_mpc import from_mpc

# The four transmission grids studied (see design doc, "Grids").
TRANSMISSION_GRID_CODES = ["IEEE24", "IEEE39", "IEEE118", "UK"]

# Default location of the committed .mat cases produced by Step 1, resolved
# relative to this file so the module works regardless of CWD.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CASES_DIR = os.path.join(_THIS_DIR, "transmission", "cases")

# Default location of PowerGraph-Node's per-grid demand files. Override with the
# POWERGRAPH_NODE_DIR environment variable (folder containing 13_Power_system).
DEFAULT_POWERGRAPH_NODE_DIR = os.environ.get(
    "POWERGRAPH_NODE_DIR",
    os.path.abspath(os.path.join(_THIS_DIR, "..", "PowerGraph-Node-main", "13_Power_system")),
)


def get_transmission_grid_codes() -> list[str]:
    """Return the list of grid codes -- the transmission analogue of ENGAGE's
    ``graph_utils.get_dist_grid_codes``. The experiment drivers (Step 5) iterate
    over this list."""
    return list(TRANSMISSION_GRID_CODES)


def load_case(code: str, cases_dir: str | None = None):
    """Load a converted MATPOWER case as a re-solvable pandapower network.

    Parameters
    ----------
    code : one of TRANSMISSION_GRID_CODES.
    cases_dir : folder containing ``<CODE>.mat`` (defaults to Step 1's output).

    Returns
    -------
    pandapower net with bus/load/gen/ext_grid/line/trafo tables populated, ready
    for ``pp.runpp`` / ``pp.runopp`` after applying demand and contingencies.
    """
    if code not in TRANSMISSION_GRID_CODES:
        raise ValueError(f"Unknown grid code {code!r}; expected one of {TRANSMISSION_GRID_CODES}")
    cases_dir = cases_dir or DEFAULT_CASES_DIR
    mat_path = os.path.join(cases_dir, f"{code}.mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(
            f"Converted case not found: {mat_path}. Run Step 1 (transmission/convert_cases.m) first."
        )
    # casename_mpc_file defaults to 'mpc', which is the variable name Step 1 saved.
    net = from_mpc(mat_path)
    return net


def load_hourly_demand(
    code: str,
    node_dir: str | None = None,
    variant: str = "new",
) -> np.ndarray:
    """Load PowerGraph's per-bus hourly ACTIVE demand profile (MW).

    Parameters
    ----------
    code : one of TRANSMISSION_GRID_CODES.
    node_dir : PowerGraph-Node ``13_Power_system`` folder (defaults to
        POWERGRAPH_NODE_DIR or a sibling checkout).
    variant : ``"new"`` -> ``hourlyDemandBusnew.mat`` (15-min, the file PowerGraph
        used; shape (N_bus, 35040)); ``"base"`` -> ``hourlyDemandBus.mat``
        (coarser).

    Returns
    -------
    ndarray of shape (N_bus, T): active demand per bus per time step, in MW.
    Row order matches the case's bus order (bus i -> row i).
    """
    node_dir = node_dir or DEFAULT_POWERGRAPH_NODE_DIR
    fname, var = {
        "new": ("hourlyDemandBusnew.mat", "hourlyDemandBusnew"),
        "base": ("hourlyDemandBus.mat", "hourlyDemandBus"),
    }[variant]
    path = os.path.join(node_dir, code, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Demand file not found: {path}. Set POWERGRAPH_NODE_DIR to your "
            f"PowerGraph-Node/13_Power_system folder."
        )
    demand = np.asarray(sio.loadmat(path)[var], dtype=float)
    return demand


if __name__ == "__main__":
    # Self-test: load every grid, run a base AC power flow, and report a summary.
    # Run with:  python3 transmission_grids.py
    import pandapower as pp

    for code in get_transmission_grid_codes():
        net = load_case(code)
        pp.runpp(net)
        try:
            demand = load_hourly_demand(code)
            dshape = demand.shape
        except FileNotFoundError:
            dshape = "(demand file not found -- set POWERGRAPH_NODE_DIR)"
        print(
            f"{code:8s} buses={len(net.bus):4d} loads={len(net.load):4d} "
            f"gens={len(net.gen):3d} ext_grid={len(net.ext_grid)} "
            f"lines={len(net.line):3d} trafos={len(net.trafo):3d} "
            f"converged={net.converged} demand={dshape}"
        )
