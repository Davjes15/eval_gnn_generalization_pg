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


def mmd(descr_a: np.ndarray, descr_b: np.ndarray) -> float:
    """Unbiased-ish Gaussian-kernel MMD between two descriptor matrices."""
    med_sq = _median_bandwidth(descr_a, descr_b)
    gamma = 1.0 / med_sq
    kxx = _gaussian_gram(descr_a, descr_a, gamma)
    kyy = _gaussian_gram(descr_b, descr_b, gamma)
    kxy = _gaussian_gram(descr_a, descr_b, gamma)
    mmd2 = kxx.mean() + kyy.mean() - 2.0 * kxy.mean()
    return float(np.sqrt(max(mmd2, 0.0)))


def evaluate_mmd(dataset_a, dataset_b):
    """Return (mmd_degree, mmd_laplacian) between two PyG datasets (distributions
    of graphs). Each dataset is a list of PyG Data objects."""
    ga = [pyg_to_networkx(d) for d in dataset_a]
    gb = [pyg_to_networkx(d) for d in dataset_b]
    md = mmd(_descriptors(ga, "degree"), _descriptors(gb, "degree"))
    ml = mmd(_descriptors(ga, "laplacian"), _descriptors(gb, "laplacian"))
    return md, ml
