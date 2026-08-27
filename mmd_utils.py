"""mmd_utils.py -- Step 5 support: topological distance via MMD (done correctly).

PURPOSE
    Measure the topological distance between two *distributions* of graphs (e.g.
    all IEEE24 samples vs all UK samples) with Maximum Mean Discrepancy (MMD)
    over graph descriptors. This is the x-axis of ENGAGE's g-score.

WHY IT IS WRITTEN THIS WAY (design decision D9 -- fixing the v2 degeneracy)
    The earlier engage_pg v2 produced a DEGENERATE Laplacian MMD (a constant
    sqrt(2) for every different-grid pair) for two reasons, both fixed here:
      1. It used ONE descriptor per grid (a single point), so "MMD" was just a
         saturated kernel indicator. Here we use a DISTRIBUTION of descriptors
         (one per graph sample) -- exactly what MMD needs.
      2. It used a tiny fixed bandwidth (sigma=1e-2) that saturated the kernel.
         Here the Gaussian bandwidth is set by the MEDIAN HEURISTIC on the
         pooled pairwise distances, so the kernel is well-scaled automatically.

    Descriptors are FIXED-LENGTH HISTOGRAMS so they are comparable across grids
    of different sizes (raw Laplacian spectra have length N and cannot be
    compared directly between a 24-bus and a 118-bus grid):
      * degree distribution  -> density histogram over degree bins
      * normalised Laplacian spectrum -> density histogram over [0, 2]
      * branch reactance     -> density histogram of log10(x_pu)  [electrical]

ESTIMATOR AND BANDWIDTH (audit item A7 -- stated, not hidden)
    * `mmd()` is the BIASED V-statistic: the three Gram-matrix means include the
      diagonal self-similarities. With 100-800 descriptors per distribution the
      bias is O(1/n) and changes no ordering here, but it is a biased estimator
      and is now named as one. `mmd(..., unbiased=True)` gives the standard
      U-statistic (diagonals excluded); the default stays biased because the
      committed result CSVs were produced with it, and silently changing
      published numbers is worse than naming the estimator correctly.
    * The bandwidth is refit per PAIR by the median heuristic, so two cells of an
      MMD matrix are computed under different kernels. That keeps every pair
      well-scaled -- it also means the matrix is a table of pairwise distances,
      not a set of values on one common scale.

WHAT THESE DESCRIPTORS CAN AND CANNOT SEE
    Degree and Laplacian histograms are purely TOPOLOGICAL and invariant to the
    electrical size of a system, so they cannot see that the UK case moves ~20x
    more power than IEEE24 (docs/Normalization_assessment.md) -- plausibly the
    shift that dominates transfer error. A distance blind to the dominant shift
    is a weak covariate for a generalization score, which is why
    `reactance_histogram` is provided as an electrical descriptor alongside them.

REFERENCES
    Gretton et al., "A Kernel Two-Sample Test", JMLR 2012 (MMD + median heuristic).
    O'Bray et al. (ggme), "Evaluating Graph Generative Models with ... MMD".
"""
from __future__ import annotations

import numpy as np
import networkx as nx


def pyg_to_networkx(data) -> nx.Graph:
    """Undirected simple graph from a PyG Data object's edge_index."""
    g = nx.Graph()
    n = int(data.x.shape[0])
    g.add_nodes_from(range(n))
    ei = data.edge_index.cpu().numpy()
    g.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    return g


def degree_histogram(g: nx.Graph, n_bins: int = 20, max_degree: int = 20) -> np.ndarray:
    """Density histogram of node degrees over a fixed [0, max_degree] range."""
    degs = np.array([d for _, d in g.degree()], dtype=float)
    hist, _ = np.histogram(degs, bins=n_bins, range=(0, max_degree), density=True)
    return hist


def laplacian_spectrum_histogram(g: nx.Graph, n_bins: int = 40) -> np.ndarray:
    """Density histogram of the normalised-Laplacian eigenvalues over [0, 2]
    (the spectrum of the normalised Laplacian always lies in [0, 2])."""
    if g.number_of_nodes() == 0:
        return np.zeros(n_bins)
    ev = nx.normalized_laplacian_spectrum(g)
    hist, _ = np.histogram(ev, bins=n_bins, range=(0.0, 2.0), density=True)
    return hist


def reactance_histogram(data, n_bins: int = 20,
                        lo: float = -4.0, hi: float = 1.0) -> np.ndarray:
    """Density histogram of log10 branch reactance, from a PyG Data object.

    The electrical counterpart of the two topological descriptors: it is computed
    from `edge_attr[:, 2]` (x_pu, see engage_contract.get_edge_features) rather
    than from the connectivity, so it does register the impedance differences
    between systems that degree and Laplacian histograms are invariant to.
    Reactances span orders of magnitude, hence the log axis; the fixed [1e-4, 10]
    p.u. range keeps the bins comparable across grids of different sizes.
    """
    x_pu = data.edge_attr[:, 2].cpu().numpy().astype(float)
    x_pu = x_pu[np.isfinite(x_pu) & (x_pu > 0)]
    if x_pu.size == 0:
        return np.zeros(n_bins)
    hist, _ = np.histogram(np.log10(x_pu), bins=n_bins, range=(lo, hi), density=True)
    return hist


def _descriptors(graphs, kind: str) -> np.ndarray:
    fn = {"degree": degree_histogram, "laplacian": laplacian_spectrum_histogram}[kind]
    return np.vstack([fn(g) for g in graphs])


def _median_bandwidth(a: np.ndarray, b: np.ndarray) -> float:
    """Median-heuristic Gaussian bandwidth from pooled pairwise squared distances."""
    pooled = np.vstack([a, b])
    # pairwise squared euclidean distances
    sq = np.sum((pooled[:, None, :] - pooled[None, :, :]) ** 2, axis=-1)
    iu = np.triu_indices_from(sq, k=1)
    med = np.median(sq[iu])
    return float(med) if med > 0 else 1.0


def _gaussian_gram(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    sq = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    return np.exp(-gamma * sq)


def _offdiag_mean(k: np.ndarray) -> float:
    """Mean of a square Gram matrix with the diagonal excluded."""
    n = k.shape[0]
    if n < 2:
        return 0.0
    return float((k.sum() - np.trace(k)) / (n * (n - 1)))


def mmd(descr_a: np.ndarray, descr_b: np.ndarray, unbiased: bool = False) -> float:
    """Gaussian-kernel MMD between two descriptor matrices.

    `unbiased=False` (default) is the biased V-statistic, whose Gram means
    include the diagonal self-similarities; it is what every committed result CSV
    was computed with. `unbiased=True` is the standard U-statistic, which drops
    the diagonals from the two within-sample terms. See the module docstring.
    """
    med_sq = _median_bandwidth(descr_a, descr_b)
    gamma = 1.0 / med_sq
    kxx = _gaussian_gram(descr_a, descr_a, gamma)
    kyy = _gaussian_gram(descr_b, descr_b, gamma)
    kxy = _gaussian_gram(descr_a, descr_b, gamma)
    if unbiased:
        mmd2 = _offdiag_mean(kxx) + _offdiag_mean(kyy) - 2.0 * kxy.mean()
    else:
        mmd2 = kxx.mean() + kyy.mean() - 2.0 * kxy.mean()
    return float(np.sqrt(max(mmd2, 0.0)))


def evaluate_mmd(dataset_a, dataset_b, unbiased: bool = False):
    """Return (mmd_degree, mmd_laplacian) between two PyG datasets (distributions
    of graphs). Each dataset is a list of PyG Data objects."""
    ga = [pyg_to_networkx(d) for d in dataset_a]
    gb = [pyg_to_networkx(d) for d in dataset_b]
    md = mmd(_descriptors(ga, "degree"), _descriptors(gb, "degree"), unbiased)
    ml = mmd(_descriptors(ga, "laplacian"), _descriptors(gb, "laplacian"), unbiased)
    return md, ml


def evaluate_mmd_electrical(dataset_a, dataset_b, unbiased: bool = False) -> float:
    """MMD between two datasets under the electrical (reactance) descriptor.

    Reported alongside the topological distances so a reader can see which shift
    a given grid pair actually represents: two systems can be topologically
    similar and electrically far apart, and the transfer error tracks the latter.
    """
    da = np.vstack([reactance_histogram(d) for d in dataset_a])
    db = np.vstack([reactance_histogram(d) for d in dataset_b])
    return mmd(da, db, unbiased)
