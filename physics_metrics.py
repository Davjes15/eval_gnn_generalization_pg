"""physics_metrics.py -- reporting that shows the physics (audit item A3).

WHY THIS MODULE EXISTS
    The headline metric of both source papers is an aggregate NRMSE over all four
    target columns, computed *after* the known inputs have been written back into
    the prediction (`models.py::inference`). Two consequences were invisible in
    our tables:

      1. Exactly HALF of every reported number is ground truth. Per bus type, two
         of the four quantities are known inputs and are re-injected, so an
         aggregate over all (bus, quantity) entries scores the model on entries it
         never predicted. Aggregating those in dilutes the error by construction.
      2. The aggregate pools quantities with incomparable ranges. Voltage
         magnitude spans ~0.1 p.u. while active power spans thousands of MW, so
         the pooled RMSE is a power-flow metric in name and an active-power metric
         in fact -- voltage can be arbitrarily wrong without moving it.

    This module reports what the aggregate hides: metrics restricted to the
    entries a model actually predicted, per quantity, plus the error tails and the
    operational violation rates that decide whether a prediction is usable in a
    security screening context.

WHAT IT DOES NOT DO
    It does not change training, and it does not replace `nrmse_range` -- the
    ENGAGE-comparable aggregate is still reported so our numbers stay readable
    next to the published ones. It adds the breakdown alongside it.

CONVENTIONS
    Target/prediction columns are [p_mw, q_mvar, vm_pu, va_degree].
    Node feature columns are [slack?, PV?, PQ?, p_mw, q_mvar, vm_pu, va_degree].
    Everything here expects PHYSICAL units (de-normalize first, see
    normalization.py).
"""
from __future__ import annotations

import torch

TARGET_NAMES = ("P", "Q", "V", "theta")

# Per bus type, which target columns the model genuinely predicts. The complement
# is overwritten with the known input by `models.py::inference`:
#   slack: V and theta are set points        -> predicts P, Q
#   PV:    P injection and V are known       -> predicts Q, theta
#   PQ:    P and Q are known                 -> predicts V, theta
PREDICTED_COLUMNS = {"slack": (0, 1), "pv": (1, 3), "pq": (2, 3)}

# Operational limits for the violation rates. The voltage band is the usual
# transmission steady-state security band; it is also the band the data generator
# filters on (it rejects samples outside 0.8-1.2, a looser sanity filter).
VM_MIN, VM_MAX = 0.95, 1.05


def predicted_mask(x: torch.Tensor) -> torch.Tensor:
    """Boolean (N, 4) mask: True where the model's own prediction survives.

    Derived from the bus-type one-hot in `x[:, :3]`, i.e. from the same rule
    `models.py::inference` applies, so the mask cannot drift from the model.
    """
    slack, pv = x[:, 0].bool(), x[:, 1].bool()
    pq = ~(slack | pv)
    mask = torch.zeros(x.shape[0], 4, dtype=torch.bool)
    for bus_type, rows in (("slack", slack), ("pv", pv), ("pq", pq)):
        for col in PREDICTED_COLUMNS[bus_type]:
            mask[rows, col] = True
    return mask


def _range(values: torch.Tensor) -> torch.Tensor:
    r = values.max() - values.min()
    return torch.clamp(r, min=1e-8)


def predicted_only_metrics(y_true, y_pred, x) -> dict[str, float]:
    """Per-quantity error over the genuinely predicted entries only.

    Returns, for each quantity: `pred_nrmse_<q>` (RMSE over that quantity's own
    range, so it is dimensionless and comparable across quantities),
    `pred_mae_<q>`, and `pred_n_<q>` (how many entries the average is over --
    reported because for P it is only the slack buses, a handful per grid).
    """
    mask = predicted_mask(x)
    out: dict[str, float] = {}
    for j, name in enumerate(TARGET_NAMES):
        sel = mask[:, j]
        n = int(sel.sum())
        out[f"pred_n_{name}"] = n
        if n == 0:
            out[f"pred_nrmse_{name}"] = float("nan")
            out[f"pred_mae_{name}"] = float("nan")
            continue
        truth, pred = y_true[sel, j], y_pred[sel, j]
        err = pred - truth
        out[f"pred_nrmse_{name}"] = float(
            torch.sqrt(torch.mean(err ** 2)) / _range(truth))
        out[f"pred_mae_{name}"] = float(torch.mean(torch.abs(err)))
    return out


def error_tails(y_true, y_pred, x, quantiles=(0.95, 0.99)) -> dict[str, float]:
    """Tail of the absolute error per quantity, over predicted entries only.

    A mean hides the case that matters operationally: the worst bus in the worst
    snapshot. Reported in physical units (MW, Mvar, p.u., degrees).
    """
    mask = predicted_mask(x)
    out: dict[str, float] = {}
    for j, name in enumerate(TARGET_NAMES):
        sel = mask[:, j]
        if not bool(sel.any()):
            for q in quantiles:
                out[f"p{int(q * 100)}_{name}"] = float("nan")
            out[f"max_{name}"] = float("nan")
            continue
        err = torch.abs(y_pred[sel, j] - y_true[sel, j])
        for q in quantiles:
            out[f"p{int(q * 100)}_{name}"] = float(torch.quantile(err, q))
        out[f"max_{name}"] = float(err.max())
    return out


def violation_rates(y_true, y_pred, x,
                    vm_min: float = VM_MIN, vm_max: float = VM_MAX) -> dict[str, float]:
    """Voltage-band agreement between prediction and AC ground truth.

    Three rates over the buses whose voltage the model predicts (the PQ buses):
      * `vm_viol_rate_true` / `vm_viol_rate_pred` -- fraction outside the band in
        the ground truth and in the prediction; a model can be accurate on
        average and still report the wrong number of violations;
      * `vm_false_secure` -- fraction of truly-violating buses the model calls
        secure. This is the asymmetric error that matters for screening: a missed
        violation is operationally worse than a false alarm;
      * `vm_false_alarm` -- the converse.
    """
    mask = predicted_mask(x)[:, 2]
    out: dict[str, float] = {}
    n = int(mask.sum())
    out["vm_n"] = n
    if n == 0:
        for k in ("vm_viol_rate_true", "vm_viol_rate_pred",
                  "vm_false_secure", "vm_false_alarm"):
            out[k] = float("nan")
        return out
    t, p = y_true[mask, 2], y_pred[mask, 2]
    viol_t = (t < vm_min) | (t > vm_max)
    viol_p = (p < vm_min) | (p > vm_max)
    out["vm_viol_rate_true"] = float(viol_t.float().mean())
    out["vm_viol_rate_pred"] = float(viol_p.float().mean())
    out["vm_false_secure"] = (float((viol_t & ~viol_p).float().sum() / viol_t.sum())
                              if bool(viol_t.any()) else float("nan"))
    out["vm_false_alarm"] = (float((viol_p & ~viol_t).float().sum() / (~viol_t).sum())
                             if bool((~viol_t).any()) else float("nan"))
    return out


def physics_metrics(y_true, y_pred, x) -> dict[str, float]:
    """All A3 reporting for one (truth, prediction, features) triple."""
    out = predicted_only_metrics(y_true, y_pred, x)
    out.update(error_tails(y_true, y_pred, x))
    out.update(violation_rates(y_true, y_pred, x))
    return out
