"""normalization.py -- feature/target scaling for the A2 remediation.

PURPOSE
    Put the four physical quantities [P, Q, V, theta] on comparable numeric
    scales for training, and undo the scaling before any metric is computed, so
    every reported number stays in MW / Mvar / p.u. / degrees.

WHY (audit item A2, see docs/Normalization_assessment.md)
    In raw units the training loss is dominated by active and reactive power:
    voltage magnitude contributes 5e-8 (IEEE24) to 1e-11 (UK) of the gradient,
    and correspondingly no architecture learns it -- in-distribution V NRMSE is
    5.8-27, i.e. worse than the constant V == 1.0. Per-unit conversion alone
    cannot fix this here, because all four cases carry sn_mva = 100, so it is a
    single constant divisor.

MODES
    none       today's behaviour, raw physical units. Default everywhere, so
               existing artifacts remain reproducible bit-for-bit.
    pu         engineering representation only: P, Q in per unit on the case's
               own S_base, angles in radians, V already in p.u. Matches Hansen
               et al. (the ARMA power-flow reference) and PowerFlowNet's
               generator. Kept as a separate mode to show that it is NOT the
               remedy.
    pu_zscore  `pu`, then per-quantity z-score with statistics fitted on the
               TRAINING split only. This is the field-standard protocol:
               PowerGraph-Node max-abs-scales X and Y per dimension,
               PowerFlowNet z-scores X, Y and edges with train statistics, and
               both de-normalize predictions for reporting.

CONTRACT
    * The scaler is fitted on the training datasets of the arm that uses it
      (one grid for within/cross-context, the pooled grids for OOD) and never
      sees the evaluation grid's statistics.
    * `transform` scales node-feature columns 3:7 and the targets with the SAME
      statistics, which is required because models re-inject known values from x
      into their prediction of y (`models.py::inference`).
    * Masked (NaN) feature entries stay NaN.
    * `dc_pf` is left in physical units: the DC baseline is analytical and is
      scored directly against the physical targets.
    * The untransformed targets are carried as `y_raw` so metrics use the exact
      physical truth rather than a round-tripped copy.
"""
from __future__ import annotations

import math

import torch

MODES = ("none", "pu", "pu_zscore")
S_BASE_MVA = 100.0
FEATURE_SLICE = slice(3, 7)  # x = [slack?, PV?, PQ?, p_mw, q_mvar, vm_pu, va_degree]


def _pu_scale():
    """Divisors taking [MW, Mvar, p.u., deg] to [p.u., p.u., p.u., rad]."""
    return torch.tensor([S_BASE_MVA, S_BASE_MVA, 1.0, 180.0 / math.pi])


class Scaler:
    """Affine per-quantity scaler: y_scaled = (y - center) / scale."""

    def __init__(self, mode: str, center: torch.Tensor, scale: torch.Tensor):
        if mode not in MODES:
            raise ValueError(f"unknown normalization mode {mode!r}, expected {MODES}")
        self.mode = mode
        self.center = center.to(torch.float32)
        self.scale = scale.to(torch.float32)

    @classmethod
    def fit(cls, datasets, mode: str) -> "Scaler":
        """Fit on one or more training datasets (lists of PyG `Data`)."""
        if mode not in MODES:
            raise ValueError(f"unknown normalization mode {mode!r}, expected {MODES}")
        zeros, ones = torch.zeros(4), torch.ones(4)
        if mode == "none":
            return cls(mode, zeros, ones)
        pu = _pu_scale()
        if mode == "pu":
            return cls(mode, zeros, pu)

        if not isinstance(datasets, (list, tuple)) or len(datasets) == 0:
            raise ValueError("pu_zscore needs at least one non-empty training dataset")
        first = datasets[0]
        if hasattr(first, "y"):  # a single dataset was passed, not a list of them
            datasets = [datasets]
        y = torch.cat([d.y for ds in datasets for d in ds]) / pu
        mean, std = y.mean(dim=0), y.std(dim=0)
        std = torch.where(std > 1e-12, std, torch.ones_like(std))
        return cls(mode, mean * pu, std * pu)

    @property
    def identity(self) -> bool:
        return self.mode == "none"

    def transform_targets(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.center) / self.scale

    def inverse_targets(self, y: torch.Tensor) -> torch.Tensor:
        """Back to physical units -- applied to predictions before scoring."""
        return y * self.scale + self.center

    def transform(self, dataset):
        """Return a scaled copy of `dataset`; the original is left untouched."""
        if self.identity:
            return dataset
        out = []
        for data in dataset:
            new = data.clone()
            new.y_raw = data.y.clone()
            new.y = self.transform_targets(data.y)
            new.x = data.x.clone()
            new.x[:, FEATURE_SLICE] = self.transform_targets(data.x[:, FEATURE_SLICE])
            out.append(new)
        return out

    def state(self) -> dict:
        """Provenance for the result rows and checkpoints."""
        return {"normalize": self.mode,
                "norm_center": [round(v, 6) for v in self.center.tolist()],
                "norm_scale": [round(v, 6) for v in self.scale.tolist()]}
