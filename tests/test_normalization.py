"""test_normalization.py -- checks for the A2 feature/target scaler.

Run:  python3 tests/test_normalization.py     (no pytest dependency needed)

The properties that matter for the study's validity:
  * mode "none" is a strict identity, so every existing artifact stays
    reproducible;
  * scaling is invertible, so reported metrics are unaffected by the training
    representation (a perfect prediction in scaled space must score 0 in
    physical units, and any prediction must score identically to the raw-unit
    pipeline);
  * x and y are scaled with the SAME statistics, which is what makes the
    known-value re-injection in `models.py::inference` legal;
  * z-score statistics come from the training data only, and the scaled targets
    actually have unit-ish scale per quantity -- the point of the exercise;
  * masked (NaN) feature entries survive;
  * the DC baseline is untouched.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import torch
from torch_geometric.data import Data

from normalization import FEATURE_SLICE, S_BASE_MVA, Scaler
from training_utils import all_metrics, test_dc_pf

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def make_dataset(n=6, buses=5, seed=0):
    """Synthetic graphs with transmission-like magnitudes and masked entries."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        y = torch.stack([
            torch.randn(buses, generator=g) * 800.0 + 200.0,   # P  [MW]
            torch.randn(buses, generator=g) * 300.0 - 100.0,   # Q  [Mvar]
            torch.randn(buses, generator=g) * 0.03 + 1.0,      # V  [p.u.]
            torch.randn(buses, generator=g) * 20.0,            # th [deg]
        ], dim=1)
        x = torch.zeros(buses, 7)
        x[:, 2] = 1.0                       # all PQ buses
        x[:, FEATURE_SLICE] = y.clone()
        x[0, 5] = float("nan")              # masked V on one bus
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        out.append(Data(x=x, y=y, edge_index=edge_index,
                        edge_attr=torch.rand(4, 4, generator=g),
                        dc_pf=y + 5.0))
    return out


def main():
    print("== normalization.Scaler ==")
    ds = make_dataset()
    y_all = torch.cat([d.y for d in ds])

    # 1. mode "none" is a strict identity
    none = Scaler.fit([ds], "none")
    check("mode none is flagged identity", none.identity)
    check("mode none returns the same objects", none.transform(ds) is ds)

    # 2. pu is exactly the engineering conversion, nothing else
    pu = Scaler.fit([ds], "pu")
    tp = pu.transform(ds)
    check("pu divides P by S_base",
          torch.allclose(tp[0].y[:, 0], ds[0].y[:, 0] / S_BASE_MVA))
    check("pu leaves V alone", torch.allclose(tp[0].y[:, 2], ds[0].y[:, 2]))
    check("pu converts theta to radians",
          torch.allclose(tp[0].y[:, 3], ds[0].y[:, 3] * math.pi / 180.0))

    # 3. pu_zscore: train statistics, unit-ish scale per quantity
    z = Scaler.fit([ds], "pu_zscore")
    tz = z.transform(ds)
    yz = torch.cat([d.y for d in tz])
    check("z-scored targets are centred", yz.mean(dim=0).abs().max().item() < 1e-4,
          f"max|mean| = {yz.mean(dim=0).abs().max().item():.2e}")
    check("z-scored targets have unit scale",
          (yz.std(dim=0) - 1.0).abs().max().item() < 1e-4)
    spread = yz.std(dim=0).max().item() / yz.std(dim=0).min().item()
    check("all four quantities are on one scale (the point of A2)", spread < 1.01,
          f"max/min std = {spread:.4f}")
    raw_spread = (y_all.std(dim=0).max() / y_all.std(dim=0).min()).item()
    check("raw units span >1e3 in scale", raw_spread > 1e3,
          f"max/min std = {raw_spread:.3g}")

    # statistics must come from the fitted data only
    other = make_dataset(seed=99)
    z_other = Scaler.fit([other], "pu_zscore")
    check("statistics depend on the fitted split",
          not torch.allclose(z.scale, z_other.scale))

    # 4. invertibility, and identical metrics through either representation
    check("inverse_targets round-trips",
          torch.allclose(z.inverse_targets(tz[0].y), ds[0].y, atol=1e-2))
    pred_phys = ds[0].y + torch.tensor([10.0, 5.0, 0.01, 0.5])
    pred_scaled = z.transform_targets(pred_phys)
    m_raw = all_metrics(ds[0].y, pred_phys)
    m_via = all_metrics(tz[0].y_raw, z.inverse_targets(pred_scaled))
    worst = max(abs(m_raw[k] - m_via[k]) / max(1e-12, abs(m_raw[k]))
                for k in m_raw)
    check("metrics are representation-invariant", worst < 1e-4,
          f"worst relative difference = {worst:.2e}")

    # 5. x and y share the scaler, so known-value re-injection stays legal
    ok = True
    for orig, new in zip(ds, tz):
        got = new.x[:, FEATURE_SLICE]
        want = new.y
        mask = ~torch.isnan(got)
        ok &= torch.allclose(got[mask], want[mask], atol=1e-5)
    check("x features and y targets are scaled identically", ok)

    # 6. masked entries and the DC baseline are untouched
    check("masked NaN survives scaling", bool(torch.isnan(tz[0].x[0, 5])))
    check("physical targets kept as y_raw", torch.allclose(tz[0].y_raw, ds[0].y))
    dc_raw, _ = test_dc_pf(ds)
    dc_scaled, _ = test_dc_pf(tz)
    check("DC baseline is unaffected by the training representation",
          abs(dc_raw - dc_scaled) < 1e-9, f"{dc_raw:.6f} vs {dc_scaled:.6f}")
    check("the original dataset was not mutated",
          torch.allclose(torch.cat([d.y for d in ds]), y_all))

    print("=" * 50)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
