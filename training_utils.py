"""training_utils.py -- Step 5 support: dataset loading, training loop, metrics.

PURPOSE
    Provide the training / evaluation machinery the experiment drivers use,
    adapted from ENGAGE's `training_utils.py` but self-contained (no ggme
    submodule, no SimBench). Two study-specific additions over ENGAGE:
      * PER-QUANTITY NRMSE (V, theta, P, Q separately) -- because the aggregate
        NRMSE is inflated by the trivially-bounded voltage magnitude (see the
        design doc "Metrics & baselines").
      * a DC power-flow baseline evaluator (`test_dc_pf`).
      * PLAIN (unnormalised) MSE and MAE, aggregate and per quantity -- the
        metric PowerGraph-Node reports, so the fixed-topology control arm can be
        compared against its published numbers. It is dominated by P and Q,
        whose numeric scale is orders of magnitude above V and theta, which is
        exactly why the normalised metrics are kept alongside it.

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

import numpy as np
import torch
import torch.nn as nn
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


def mse_plain(y_true, y_pred):
    """Unnormalised MSE over all targets (PowerGraph-Node's reported metric).

    In the raw physical units of y, so P/Q dominate V/theta by construction.
    """
    return torch.mean((y_true - y_pred) ** 2).item()


def mae_plain(y_true, y_pred):
    """Unnormalised MAE over all targets (in the raw physical units of y)."""
    return torch.mean(torch.abs(y_true - y_pred)).item()


def mse_per_quantity(y_true, y_pred):
    """Per-column unnormalised MSE. Returns dict {P, Q, V, theta}."""
    return {name: torch.mean((y_true[:, j] - y_pred[:, j]) ** 2).item()
            for j, name in enumerate(TARGET_NAMES)}


def mae_per_quantity(y_true, y_pred):
    """Per-column unnormalised MAE. Returns dict {P, Q, V, theta}."""
    return {name: torch.mean(torch.abs(y_true[:, j] - y_pred[:, j])).item()
            for j, name in enumerate(TARGET_NAMES)}


def all_metrics(y_true, y_pred):
    """Every metric for one (truth, prediction) pair, as a flat dict.

    Keys: nrmse / mse / mae, plus <metric>_<quantity> for each of P, Q, V, theta.
    Flat so a result row can be built with `row.update(all_metrics(...))`.
    """
    out = {
        "nrmse": nrmse_range(y_true, y_pred),
        "mse": mse_plain(y_true, y_pred),
        "mae": mae_plain(y_true, y_pred),
    }
    for metric, fn in (("nrmse", nrmse_per_quantity),
                       ("mse", mse_per_quantity),
                       ("mae", mae_per_quantity)):
        for name, value in fn(y_true, y_pred).items():
            out[f"{metric}_{name}"] = value
    return out


def train(model, device, loader_train, loader_val, epochs=200, learning_rate=1e-3,
          patience=50, log_every=0, grad_clip=None):
    """Train with early stopping on validation loss; restore best weights.

    ``grad_clip`` caps the global gradient norm before each optimizer step.
    ``None`` leaves the update untouched, so runs without it are unaffected.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val, best_weights, wait = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        for batch in loader_train:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = weighted_mse_loss(model(batch), batch.y)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        model.eval()
        val = 0.0
        with torch.no_grad():
            for batch in loader_val:
                batch = batch.to(device)
                val += weighted_mse_loss(model(batch), batch.y).item() * batch.num_graphs
        val /= max(len(loader_val.dataset), 1)

        if val < best_val:
            best_val, wait = val, 0
            best_weights = {k: v.detach().clone()
                            for k, v in model.state_dict().items()}
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
def evaluate(model, device, dataset, batch_size=32, full=False, scaler=None):
    """Evaluate a trained model on a dataset.

    Returns (aggregate_nrmse, per_quantity_nrmse_dict), and additionally the
    flat `all_metrics` dict when `full=True` (opt-in, so existing two-value
    unpacking keeps working).

    ``scaler`` un-scales the prediction before any metric is computed, so every
    reported number is in physical units whatever the training representation
    was. The dataset then carries the physical targets as `y_raw`.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).cpu()
        truth = batch.y.cpu()
        if scaler is not None and not scaler.identity:
            pred = scaler.inverse_targets(pred)
            truth = batch.y_raw.cpu()
        preds.append(pred)
        ys.append(truth)
    y_pred, y_true = torch.cat(preds), torch.cat(ys)
    nrmse, per_q = nrmse_range(y_true, y_pred), nrmse_per_quantity(y_true, y_pred)
    if full:
        return nrmse, per_q, all_metrics(y_true, y_pred)
    return nrmse, per_q


def apply_dc_convention(dc_pf):
    """Force the DC baseline's reactive power to ENGAGE's convention: Q == 0.

    DC power flow has no reactive power, so `pp.rundcpp` never writes
    res_bus.q_mvar. ENGAGE read the column anyway and zeroed the resulting NaNs;
    under pandapower >= 3 the column instead retains whatever an earlier AC solve
    left there, which for our generator is the ground truth itself. Datasets
    written before that was noticed therefore carry the AC reactive power in
    `dc_pf`, and applying the convention here corrects them at scoring time --
    P, V and theta are untouched (verified identical between a DC solve on a
    fresh net and on a copy of an AC-solved one).
    """
    out = dc_pf.clone()
    out[:, 1] = 0.0
    return out


def nrmse_range_subset(y_true, y_pred, columns):
    """`nrmse_range` restricted to a subset of target columns.

    Used to score the DC baseline on the quantities DC actually solves (P, V,
    theta), alongside the four-column number that charges it for Q == 0.
    """
    idx = [TARGET_NAMES.index(c) for c in columns]
    return nrmse_range(y_true[:, idx], y_pred[:, idx])


@torch.no_grad()
def test_dc_pf(dataset, batch_size=32, full=False):
    """DC power-flow baseline: error of the stored DC solution vs the AC truth.

    The stored solution is put through `apply_dc_convention` first. Same return
    contract as `evaluate`, plus `nrmse_PVtheta` (the Q-excluded aggregate) in
    the `full` metrics dict.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dc, ys = [], []
    for batch in loader:
        dc.append(batch.dc_pf.cpu())
        # `y_raw` is present when the dataset was scaled for training; DC is an
        # analytical solution in physical units, so it is scored against those.
        truth = batch.y_raw if "y_raw" in batch else batch.y
        ys.append(truth.cpu())
    dc_pf, y_true = apply_dc_convention(torch.cat(dc)), torch.cat(ys)
    nrmse, per_q = nrmse_range(y_true, dc_pf), nrmse_per_quantity(y_true, dc_pf)
    if full:
        metrics = all_metrics(y_true, dc_pf)
        metrics["nrmse_PVtheta"] = nrmse_range_subset(
            y_true, dc_pf, ["P", "V", "theta"])
        return nrmse, per_q, metrics
    return nrmse, per_q


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


def gscore_row(mmds, nrmses, alpha=1.0, bounds=0):
    """g-score over the finite points, with the completeness of the cell attached.

    A model that emits a non-finite error on some transfer pair would otherwise be
    scored on its surviving pairs only -- i.e. rewarded for the divergence, because
    the dropped pairs are the hard ones. The score is therefore VOIDED (NaN) unless
    every expected point is finite, and `finite_rate` records what fraction survived
    so the failure is visible in the table rather than absorbed into it. This is the
    same void-the-cell policy the ranking analysis applies.

    Returns a dict with the descriptive statistics over the finite points plus
    `n_expected`, `n_finite`, `finite_rate` and the (possibly voided) `g_score`.
    """
    mmds, nrmses = np.asarray(mmds, float), np.asarray(nrmses, float)
    finite = np.isfinite(nrmses)
    n_expected = int(nrmses.size)
    n_finite = int(finite.sum())
    out = {
        "n_expected": n_expected,
        "n_finite": n_finite,
        "finite_rate": float(n_finite / n_expected) if n_expected else float("nan"),
    }
    if n_finite == 0:
        out.update(mean_nrmse=float("nan"), std_nrmse=float("nan"),
                   mmd_range=float("nan"), g_score=float("nan"))
        return out
    mean_n, std_n, mmd_rng, score = get_generalization_score(
        mmds[finite], nrmses[finite], alpha=alpha, bounds=bounds)
    out.update(mean_nrmse=mean_n, std_nrmse=std_n, mmd_range=mmd_rng,
               g_score=score if n_finite == n_expected else float("nan"))
    return out
