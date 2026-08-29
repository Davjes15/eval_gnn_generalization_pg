"""test_physics_metrics.py -- checks for the A3 reporting in physics_metrics.py.

Run:  python3 tests/test_physics_metrics.py     (no pytest dependency needed)

The point of that module is to score a model only where it actually predicts, so
the checks that matter are: the predicted-entry mask must agree with the
re-injection rule in `models.py::inference` (if the two ever disagree we would be
scoring ground truth again, which is the defect A3 reports), an error placed only
on re-injected entries must be invisible to the predicted-only metrics while an
error on predicted entries must show up, and the violation rates must count the
asymmetric screening error (a missed violation) correctly.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models import BasePFGNN
from physics_metrics import (PREDICTED_COLUMNS, error_tails, physics_metrics,
                             predicted_mask, predicted_only_metrics,
                             violation_rates)

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def make_x():
    """One bus of each type. Columns: [slack, PV, PQ, p_mw, q_mvar, vm_pu, va_deg]."""
    return torch.tensor([
        [1., 0., 0., 10., 20., 1.01, 0.0],    # slack
        [0., 1., 0., 30., 40., 1.02, 1.0],    # PV
        [0., 0., 1., 50., 60., 1.03, 2.0],    # PQ
    ])


print("A. mask agrees with the re-injection rule in models.inference")
x = make_x()
mask = predicted_mask(x)

# Feed a sentinel prediction through inference: whatever no longer holds the
# sentinel was overwritten with a known input, i.e. is NOT predicted. That is the
# ground truth the mask has to reproduce -- read off the model itself rather than
# from a second copy of the bus-type table. The sentinel must be a value the
# features cannot contain (a bus can legitimately have va_degree = 0).
SENTINEL = -12345.0
pred = torch.full((3, 4), SENTINEL)
overwritten = BasePFGNN.inference(None, x, pred.clone()) != SENTINEL
check("mask is the complement of the overwritten entries",
      bool(torch.equal(mask, ~overwritten)),
      f"mask={mask.int().tolist()}")
check("exactly two predicted quantities per bus",
      bool(torch.equal(mask.sum(1), torch.full((3,), 2))),
      f"per-bus counts={mask.sum(1).tolist()}")
for bus, row in (("slack", 0), ("pv", 1), ("pq", 2)):
    cols = tuple(int(c) for c in mask[row].nonzero().flatten())
    check(f"{bus} predicts {PREDICTED_COLUMNS[bus]}", cols == PREDICTED_COLUMNS[bus],
          f"got {cols}")

print("\nB. error on re-injected entries is invisible; on predicted entries it is not")
y_true = torch.tensor([
    [10., 20., 1.01, 0.0],
    [30., 40., 1.02, 1.0],
    [50., 60., 1.03, 2.0],
])
# Corrupt only the known/re-injected entries.
y_reinject = y_true.clone()
y_reinject[~mask] += 100.0
m_reinject = predicted_only_metrics(y_true, y_reinject, x)
check("zero predicted-only MAE when only known entries are wrong",
      all(m_reinject[f"pred_mae_{q}"] == 0.0 for q in ("P", "Q", "V", "theta")),
      str({k: v for k, v in m_reinject.items() if k.startswith("pred_mae")}))

# Corrupt only the genuinely predicted entries.
y_pred = y_true.clone()
y_pred[mask] += 1.0
m_pred = predicted_only_metrics(y_true, y_pred, x)
check("non-zero predicted-only MAE when predicted entries are wrong",
      all(m_pred[f"pred_mae_{q}"] == 1.0 for q in ("P", "Q", "V", "theta")),
      str({k: round(v, 4) for k, v in m_pred.items() if k.startswith("pred_mae")}))
check("entry counts reported per quantity",
      (m_pred["pred_n_P"], m_pred["pred_n_Q"], m_pred["pred_n_V"],
       m_pred["pred_n_theta"]) == (1, 2, 1, 2),
      "P is predicted at the slack bus only")

print("\nC. an absent quantity yields NaN rather than a fabricated zero")
x_pq = torch.tensor([[0., 0., 1., 5., 6., 1.0, 0.]])
m_pq = predicted_only_metrics(torch.zeros(1, 4), torch.zeros(1, 4), x_pq)
check("PQ-only grid reports NaN for P", m_pq["pred_nrmse_P"] != m_pq["pred_nrmse_P"],
      f"pred_nrmse_P={m_pq['pred_nrmse_P']}")
check("PQ-only grid still reports V", m_pq["pred_n_V"] == 1)

print("\nD. error tails are the tails, in physical units")
n = 100
x_tail = torch.tensor([[0., 0., 1., 0., 0., 1.0, 0.]]).repeat(n, 1)
truth = torch.zeros(n, 4)
guess = torch.zeros(n, 4)
guess[:, 2] = torch.linspace(0.0, 0.99, n)     # V error 0 .. 0.99
tails = error_tails(truth, guess, x_tail)
check("max equals the largest absolute error", abs(tails["max_V"] - 0.99) < 1e-6,
      f"max_V={tails['max_V']:.4f}")
check("p95 below the max and above the median",
      0.49 < tails["p95_V"] < tails["max_V"],
      f"p95_V={tails['p95_V']:.4f} p99_V={tails['p99_V']:.4f}")

print("\nE. violation rates count the missed violations")
# Four PQ buses; truth has two outside the 0.95-1.05 band, the prediction calls
# one of them secure (a miss) and invents one violation elsewhere (a false alarm).
x_v = torch.tensor([[0., 0., 1., 0., 0., 1.0, 0.]]).repeat(4, 1)
t_v = torch.zeros(4, 4)
p_v = torch.zeros(4, 4)
t_v[:, 2] = torch.tensor([1.00, 1.00, 1.10, 0.90])
p_v[:, 2] = torch.tensor([1.00, 1.20, 1.10, 1.00])
v = violation_rates(t_v, p_v, x_v)
check("true violation rate", abs(v["vm_viol_rate_true"] - 0.5) < 1e-6,
      f"{v['vm_viol_rate_true']}")
check("predicted violation rate", abs(v["vm_viol_rate_pred"] - 0.5) < 1e-6,
      f"{v['vm_viol_rate_pred']}")
check("one of two true violations missed", abs(v["vm_false_secure"] - 0.5) < 1e-6,
      f"vm_false_secure={v['vm_false_secure']}")
check("one of two secure buses false-alarmed",
      abs(v["vm_false_alarm"] - 0.5) < 1e-6, f"vm_false_alarm={v['vm_false_alarm']}")

print("\nF. the aggregate wrapper returns every family of keys")
allm = physics_metrics(y_true, y_pred, x)
for prefix in ("pred_nrmse_", "pred_mae_", "pred_n_", "p95_", "p99_", "max_"):
    check(f"{prefix}* present",
          sum(k.startswith(prefix) for k in allm) == 4,
          f"{sum(k.startswith(prefix) for k in allm)} keys")
check("violation keys present", "vm_false_secure" in allm and "vm_n" in allm)

print("\n" + ("ALL CHECKS PASSED" if not FAILURES
              else f"{len(FAILURES)} FAILURE(S): {FAILURES}"))
sys.exit(1 if FAILURES else 0)
