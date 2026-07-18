"""training_utils.py -- Step 5 support: dataset loading, training loop, metrics.

PURPOSE
    Provide the training / evaluation machinery the experiment drivers use,
    adapted from ENGAGE's `training_utils.py` but self-contained (no ggme
    submodule, no SimBench). Two study-specific additions over ENGAGE:
      * PER-QUANTITY NRMSE (V, theta, P, Q separately) -- because the aggregate
        NRMSE is inflated by the trivially-bounded voltage magnitude (see the
        design doc "Metrics & baselines").
      * a DC power-flow baseline evaluator (`test_dc_pf`).

WHY (design decisions D8 + the power-systems reporting corrections)
    A credible power-systems study must show where the error actually lives
    (angles/reactive power are the hard, informative quantities) and must beat
    the trivial DC-PF baseline. These metrics make that explicit.

TARGET COLUMN ORDER (y): [p_mw, q_mvar, vm_pu, va_degree]  ==  [P, Q, V, theta]

ATTRIBUTION
    train / weighted_mse_loss / nrmse_range adapted from ENGAGE training_utils.py.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

TARGET_NAMES = ["P", "Q", "V", "theta"]  # order of columns in y
TRAIN_VAL_SPLIT = [0.8, 0.2]


def get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_grid_dataset(data_dir: str, grid: str, split: str):
    """Load one split of one grid: data_dir/<grid>/<split>/dataset.pt (Step 3)."""
    path = os.path.join(data_dir, grid, split, "dataset.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Generate it with transmission_graph_gen.py (Step 3)."
        )
    return torch.load(path, weights_only=False)


def weighted_mse_loss(pred, target, eps=1e-8):
    """MSE weighted by the inverse of each target vector's norm (ENGAGE)."""
    target_norm = torch.norm(target, dim=-1, keepdim=True) + eps
    weights = 1.0 / target_norm
    mse = nn.functional.mse_loss(pred, target, reduction="none")
    return (weights * mse).mean()


def nrmse_range(y_true, y_pred):
    """Aggregate NRMSE normalised by the average per-dimension range (ENGAGE)."""
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    range_per_dim = y_true.max(dim=0).values - y_true.min(dim=0).values
    avg_range = torch.mean(range_per_dim)
    return (rmse / avg_range).item()


def nrmse_per_quantity(y_true, y_pred, eps=1e-8):
    """Per-column NRMSE (each normalised by its own range). Returns dict
    {P, Q, V, theta} -- exposes where the error really is."""
    out = {}
    rng = y_true.max(dim=0).values - y_true.min(dim=0).values
    for j, name in enumerate(TARGET_NAMES):
        rmse = torch.sqrt(torch.mean((y_true[:, j] - y_pred[:, j]) ** 2))
        out[name] = (rmse / (rng[j] + eps)).item()
    return out


def train(model, device, loader_train, loader_val, epochs=200, learning_rate=1e-3,
          patience=50, log_every=0):
    """Train with early stopping on validation loss; restore best weights."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val, best_weights, wait = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        for batch in loader_train:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = weighted_mse_loss(model(batch), batch.y)
            loss.backward()
            optimizer.step()

        model.eval()
        val = 0.0
        with torch.no_grad():
            for batch in loader_val:
                batch = batch.to(device)
                val += weighted_mse_loss(model(batch), batch.y).item() * batch.num_graphs
        val /= max(len(loader_val.dataset), 1)

        if val < best_val:
            best_val, best_weights, wait = val, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
        if log_every and (epoch + 1) % log_every == 0:
            print(f"    epoch {epoch+1:4d}  val {val:.6f}")

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return best_val


@torch.no_grad()
def evaluate(model, device, dataset, batch_size=32):
    """Evaluate a trained model on a dataset. Returns (aggregate_nrmse, per_quantity_dict)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        preds.append(model(batch).cpu())
        ys.append(batch.y.cpu())
    y_pred, y_true = torch.cat(preds), torch.cat(ys)
    return nrmse_range(y_true, y_pred), nrmse_per_quantity(y_true, y_pred)


@torch.no_grad()
def test_dc_pf(dataset, batch_size=32):
    """DC power-flow baseline: NRMSE of the stored DC solution vs the AC truth."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dc, ys = [], []
    for batch in loader:
        dc.append(batch.dc_pf.cpu())
        ys.append(batch.y.cpu())
    dc_pf, y_true = torch.cat(dc), torch.cat(ys)
    return nrmse_range(y_true, dc_pf), nrmse_per_quantity(y_true, dc_pf)


def make_loaders(train_ds, val_ds, batch_size=32, shuffle=True):
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def get_generalization_score(mmds, nrmses, alpha=1.0, bounds=2):
    """ENGAGE g-score: mean NRMSE + alpha * std(NRMSE) * log-scaled MMD range.

    mmds, nrmses : 1-D arrays aligned by test grid.
    Returns (mean_nrmse, std_nrmse, mmd_range, score).
    """
    mmds, nrmses = np.asarray(mmds, float), np.asarray(nrmses, float)
    eps = 1e-8
    p_min, p_max = np.percentile(nrmses, bounds), np.percentile(nrmses, 100 - bounds)
    keep = (nrmses <= p_max) & (nrmses >= p_min)
    mmd_range = mmds[keep].max() - mmds[keep].min() if keep.any() else 0.0
    mean_nrmse = nrmses[keep].mean() if keep.any() else nrmses.mean()
    std_nrmse = nrmses[keep].std() if keep.any() else nrmses.std()
    score = mean_nrmse + alpha * std_nrmse * (np.log(mmd_range + 1) / (mmd_range + eps))
    return float(mean_nrmse), float(std_nrmse), float(mmd_range), float(score)
