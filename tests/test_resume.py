"""Tests for epoch-level training resume and atomic checkpoint saves."""
from __future__ import annotations

import contextlib
import io
import os
import random
import sys
import tempfile
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checkpoint_index import build
from eval_checkpoints import find_checkpoints
from experiments import _fit
from training_utils import atomic_torch_save, train


FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


class TinyGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(2, 8)
        self.conv2 = GCNConv(8, 2)
        self.dropout = nn.Dropout(0.35)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index).relu()
        x = self.dropout(x)
        return self.conv2(x, data.edge_index)


def _dataset():
    generator = torch.Generator().manual_seed(91)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0],
        [1, 2, 3, 4, 5, 0, 0, 1, 2, 3, 4, 5],
    ])
    return [
        Data(
            x=torch.randn(6, 2, generator=generator),
            edge_index=edge_index,
            y=torch.randn(6, 2, generator=generator),
        )
        for _ in range(8)
    ]


def _loaders(dataset):
    return (
        DataLoader(dataset, batch_size=2, shuffle=True),
        DataLoader(dataset, batch_size=2, shuffle=False),
    )


def _seed_all(seed=23):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _same_state(left, right):
    return all(
        torch.equal(left.state_dict()[name], right.state_dict()[name])
        for name in left.state_dict()
    )


def test_resume_matches_uninterrupted():
    print("\nEpoch-boundary resume matches uninterrupted training")
    dataset = _dataset()
    with tempfile.TemporaryDirectory() as tmp:
        direct_path = os.path.join(tmp, "direct.resume")
        _seed_all()
        direct = TinyGCN()
        direct_train, direct_val = _loaders(dataset)
        direct_best = train(
            direct, "cpu", direct_train, direct_val, epochs=6,
            learning_rate=1e-3, resume_path=direct_path, save_every=3,
        )

        split_path = os.path.join(tmp, "split.resume")
        _seed_all()
        first = TinyGCN()
        first_train, first_val = _loaders(dataset)
        train(
            first, "cpu", first_train, first_val, epochs=3,
            learning_rate=1e-3, resume_path=split_path, save_every=3,
        )
        _seed_all()
        resumed = TinyGCN()
        resumed_train, resumed_val = _loaders(dataset)
        resumed_best = train(
            resumed, "cpu", resumed_train, resumed_val, epochs=6,
            learning_rate=1e-3, resume_path=split_path, save_every=3,
        )

        check("resumed state_dict equals uninterrupted", _same_state(direct, resumed))
        check("resumed best validation loss equals uninterrupted",
              direct_best == resumed_best, f"{direct_best} vs {resumed_best}")


def test_resume_matches_with_early_stopping():
    print("\nResume preserves early-stopping state")
    dataset = _dataset()
    with tempfile.TemporaryDirectory() as tmp:
        direct_path = os.path.join(tmp, "direct.resume")
        _seed_all()
        direct = TinyGCN()
        direct_train, direct_val = _loaders(dataset)
        direct_best = train(
            direct, "cpu", direct_train, direct_val, epochs=12,
            learning_rate=10.0, patience=1, resume_path=direct_path,
            save_every=1,
        )
        direct_epoch = torch.load(direct_path, weights_only=False)["epoch"]

        split_path = os.path.join(tmp, "split.resume")
        _seed_all()
        first = TinyGCN()
        first_train, first_val = _loaders(dataset)
        train(
            first, "cpu", first_train, first_val, epochs=3,
            learning_rate=10.0, patience=1, resume_path=split_path,
            save_every=1,
        )
        _seed_all()
        resumed = TinyGCN()
        resumed_train, resumed_val = _loaders(dataset)
        resumed_best = train(
            resumed, "cpu", resumed_train, resumed_val, epochs=12,
            learning_rate=10.0, patience=1, resume_path=split_path,
            save_every=1,
        )
        resumed_epoch = torch.load(split_path, weights_only=False)["epoch"]

        check("the direct run stopped before its epoch limit", direct_epoch < 12,
              f"saved boundary={direct_epoch}")
        check("resumed state_dict equals early-stopped run",
              _same_state(direct, resumed))
        check("resumed best validation loss equals early-stopped run",
              direct_best == resumed_best, f"{direct_best} vs {resumed_best}")
        check("resumed run stopped at the same boundary",
              direct_epoch == resumed_epoch,
              f"{direct_epoch} vs {resumed_epoch}")


def test_resume_metadata_mismatch():
    print("\nResume metadata must match the current call")
    dataset = _dataset()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.resume")
        _seed_all()
        model = TinyGCN()
        train_loader, val_loader = _loaders(dataset)
        train(model, "cpu", train_loader, val_loader, epochs=1,
              learning_rate=1e-3, resume_path=path, save_every=1)
        fresh = TinyGCN()
        train_loader, val_loader = _loaders(dataset)
        try:
            train(fresh, "cpu", train_loader, val_loader, epochs=2,
                  learning_rate=2e-3, resume_path=path, save_every=1)
            error = None
        except ValueError as exc:
            error = str(exc)
        check("learning-rate mismatch raises ValueError",
              error is not None and "resume metadata mismatch" in error,
              str(error))


def test_atomic_save_cleans_temporary_file():
    print("\nAtomic save preserves target and cleans failed temporary files")
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "checkpoint.pt")
        with open(target, "wb") as fh:
            fh.write(b"previous checkpoint")

        def failing_save(obj, destination):
            destination.write(b"partial")
            raise OSError("simulated write failure")

        try:
            with mock.patch("training_utils.torch.save", failing_save):
                atomic_torch_save({"weight": torch.ones(1)}, target)
            error = None
        except OSError as exc:
            error = str(exc)
        temporary_files = [name for name in os.listdir(tmp) if ".tmp." in name]
        with open(target, "rb") as fh:
            target_contents = fh.read()
        check("write error is re-raised", error == "simulated write failure",
              str(error))
        check("existing target remains unchanged",
              target_contents == b"previous checkpoint")
        check("failed temporary file is removed", not temporary_files,
              str(temporary_files))


def test_fit_discards_stale_resume_on_fresh_run():
    print("\n_fit's fresh run discards stale resume data")
    dataset = _dataset()
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = os.path.join(tmp, "within_gcn_G1_s0.pt")
        resume_path = f"{checkpoint}.resume"
        with open(resume_path, "wb") as fh:
            fh.write(b"stale resume file")
        model = TinyGCN()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _fit(model, "cpu", dataset, dataset, epochs=1,
                 cfg={"learning_rate": 1e-3}, batch_size=2,
                 ckpt=checkpoint, skip_existing=False)
        check("fresh run does not print a resume message",
              "resuming " not in output.getvalue(), output.getvalue())
        check("final checkpoint exists", os.path.isfile(checkpoint))
        check("final checkpoint is loadable",
              isinstance(torch.load(checkpoint, weights_only=True), dict))
        check("stale resume file is removed", not os.path.exists(resume_path))
        with open(resume_path, "wb") as fh:
            fh.write(b"orphaned resume file")
        reused = TinyGCN()
        with contextlib.redirect_stdout(output := io.StringIO()):
            _fit(reused, "cpu", dataset, dataset, epochs=1,
                 cfg={"learning_rate": 1e-3}, batch_size=2,
                 ckpt=checkpoint, skip_existing=True)
        check("reusing a final checkpoint removes an orphaned resume file",
              "reusing " in output.getvalue()
              and not os.path.exists(resume_path))


def test_checkpoint_discovery_ignores_resume_and_temp_files():
    print("\nCheckpoint discovery ignores resume and temporary suffixes")
    with tempfile.TemporaryDirectory() as root:
        valid_name = "within_gcn_IEEE24_s0.pt"
        torch.save({"weight": torch.zeros(2)}, os.path.join(root, valid_name))
        for suffix in (".resume", ".tmp.123"):
            with open(os.path.join(root, valid_name + suffix), "wb") as fh:
                fh.write(b"not a checkpoint")
        indexed = build(root, with_params=False)
        discovered = find_checkpoints(root)
        check("checkpoint_index indexes only the final .pt",
              len(indexed) == 1 and indexed.iloc[0].path == valid_name,
              str(indexed.path.tolist()))
        check("eval_checkpoints discovers only the final .pt",
              len(discovered) == 1
              and os.path.basename(discovered[0][0]) == valid_name,
              str(discovered))


def main():
    test_resume_matches_uninterrupted()
    test_resume_matches_with_early_stopping()
    test_resume_metadata_mismatch()
    test_atomic_save_cleans_temporary_file()
    test_fit_discards_stale_resume_on_fresh_run()
    test_checkpoint_discovery_ignores_resume_and_temp_files()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S):")
        for failure in FAILURES:
            print(f" - {failure}")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
