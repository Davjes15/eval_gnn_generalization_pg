"""test_metrics.py -- checks for the metric suite in training_utils.

Run:  python3 tests/test_metrics.py     (no pytest dependency needed)

Covers the two things that could silently invalidate results: the new plain
MSE/MAE must be numerically right, and the existing two-value return contract of
`evaluate` / `test_dc_pf` must be unchanged.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch_geometric.data import Data

from training_utils import (TARGET_NAMES, all_metrics, evaluate, mae_per_quantity,
                            mae_plain, mse_per_quantity, mse_plain,
                            nrmse_per_quantity, nrmse_range, test_dc_pf)

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def test_plain_metrics():
    print("\nPlain MSE / MAE against numpy")
    rng = np.random.default_rng(0)
    yt = rng.normal(size=(50, 4)) * np.array([100.0, 50.0, 1.0, 10.0])
    yp = yt + rng.normal(size=(50, 4)) * 0.5
    t_true, t_pred = torch.tensor(yt), torch.tensor(yp)

    check("mse_plain", close(mse_plain(t_true, t_pred), ((yt - yp) ** 2).mean()))
    check("mae_plain", close(mae_plain(t_true, t_pred), np.abs(yt - yp).mean()))

    per_mse = mse_per_quantity(t_true, t_pred)
    per_mae = mae_per_quantity(t_true, t_pred)
    check("mse_per_quantity keys", list(per_mse) == TARGET_NAMES, str(list(per_mse)))
    ok_mse = all(close(per_mse[n], ((yt[:, j] - yp[:, j]) ** 2).mean())
                 for j, n in enumerate(TARGET_NAMES))
    ok_mae = all(close(per_mae[n], np.abs(yt[:, j] - yp[:, j]).mean())
                 for j, n in enumerate(TARGET_NAMES))
    check("mse_per_quantity values", ok_mse)
    check("mae_per_quantity values", ok_mae)
    # The per-quantity MSEs must average to the aggregate MSE (equal column counts).
    check("per-quantity MSE averages to plain MSE",
          close(np.mean(list(per_mse.values())), mse_plain(t_true, t_pred)))

    # A perfect prediction gives exactly zero on every metric.
    zeros = all_metrics(t_true, t_true)
    check("perfect prediction -> all metrics 0",
          all(v == 0.0 for v in zeros.values()), str(zeros))


def test_all_metrics_schema():
    print("\nall_metrics schema and consistency with the existing metrics")
    rng = np.random.default_rng(1)
    t_true = torch.tensor(rng.normal(size=(30, 4)))
    t_pred = t_true + torch.tensor(rng.normal(size=(30, 4))) * 0.1

    m = all_metrics(t_true, t_pred)
    expected = ({"nrmse", "mse", "mae"}
                | {f"{k}_{q}" for k in ("nrmse", "mse", "mae") for q in TARGET_NAMES})
    check("keys", set(m) == expected, str(sorted(set(m) ^ expected)))
    check("nrmse matches nrmse_range", close(m["nrmse"], nrmse_range(t_true, t_pred)))
    per_q = nrmse_per_quantity(t_true, t_pred)
    check("nrmse_<q> matches nrmse_per_quantity",
          all(close(m[f"nrmse_{q}"], per_q[q]) for q in TARGET_NAMES))


def _toy_dataset(n=6, n_bus=5):
    """Minimal graphs with a y and a dc_pf field -- enough for the evaluators."""
    torch.manual_seed(0)
    out = []
    for _ in range(n):
        ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        out.append(Data(x=torch.randn(n_bus, 7), edge_index=ei,
                        edge_attr=torch.randn(ei.shape[1], 4),
                        y=torch.randn(n_bus, 4), dc_pf=torch.randn(n_bus, 4)))
    return out


class _ConstModel(torch.nn.Module):
    """Predicts zeros -- deterministic, so the metrics are checkable by hand."""

    def forward(self, data):
        return torch.zeros_like(data.y)

    def eval(self):
        return self


def test_return_contracts():
    print("\nBackward-compatible return contracts of evaluate / test_dc_pf")
    ds = _toy_dataset()
    model = _ConstModel()

    nrmse, per_q = evaluate(model, "cpu", ds)
    check("evaluate returns 2 values", isinstance(nrmse, float) and isinstance(per_q, dict))
    nrmse_f, per_q_f, metrics = evaluate(model, "cpu", ds, full=True)
    check("evaluate(full=True) adds a metrics dict", isinstance(metrics, dict))
    check("evaluate full=True agrees with default",
          close(nrmse, nrmse_f) and per_q == per_q_f)
    y_true = torch.cat([d.y for d in ds])
    check("evaluate mse matches a zero prediction",
          close(metrics["mse"], float((y_true ** 2).mean())))

    dc_nrmse, dc_per_q = test_dc_pf(ds)
    check("test_dc_pf returns 2 values",
          isinstance(dc_nrmse, float) and isinstance(dc_per_q, dict))
    dc_nrmse_f, dc_per_q_f, dc_metrics = test_dc_pf(ds, full=True)
    check("test_dc_pf(full=True) agrees with default",
          close(dc_nrmse, dc_nrmse_f) and dc_per_q == dc_per_q_f)
    dc = torch.cat([d.dc_pf for d in ds])
    check("test_dc_pf mae matches the stored DC solution",
          close(dc_metrics["mae"], float((y_true - dc).abs().mean())))


def main():
    test_plain_metrics()
    test_all_metrics_schema()
    test_return_contracts()
    print("\n" + "=" * 50)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
