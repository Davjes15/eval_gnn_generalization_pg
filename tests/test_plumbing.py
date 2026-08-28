"""test_plumbing.py -- checks for the hidden-width and experiment plumbing.

Run:  python3 tests/test_plumbing.py

What could silently invalidate results, and is therefore checked here:
  * the default configuration must be BIT-FOR-BIT what it was before `hidden`
    became a constructor argument, so the earlier runs stay comparable;
  * every architecture must actually honour `hidden` and `num_layers`;
  * a missing --arch_config must be a hard error, not a silent fallback to the
    untuned defaults;
  * checkpoint names must carry the seed, or seed replicates overwrite each other;
  * ARMA's scalar edge weight must stay non-negative -- `ARMAConv` normalizes
    without self-loops, so a non-positive degree makes the forward pass NaN;
  * `train(grad_clip=...)` must be an opt-in that leaves the default untouched.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.data import Data

from experiments import (_build_model, _config_columns, load_arch_config,
                         run_cross_context, run_ood, run_within)
from models import HIDDEN, MODELS
from training_utils import make_loaders, train

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def test_default_is_unchanged():
    """Same seed + default arguments must give identical initial weights."""
    print("\nDefault configuration is bit-for-bit unchanged")
    for name, cls in MODELS.items():
        torch.manual_seed(0)
        a = cls(input_dim=7)
        torch.manual_seed(0)
        b = cls(input_dim=7, hidden=HIDDEN)   # explicit default
        same = all(torch.equal(pa, pb)
                   for pa, pb in zip(a.state_dict().values(), b.state_dict().values()))
        check(f"{name}: hidden=64 explicit == default", same,
              f"params={n_params(a):,}")


def test_hidden_and_depth_are_honoured():
    print("\nhidden and num_layers change the model as expected")
    for name, cls in MODELS.items():
        small, big = cls(input_dim=7, hidden=32), cls(input_dim=7, hidden=128)
        check(f"{name}: hidden widens the model", n_params(small) < n_params(big),
              f"32 -> {n_params(small):,} | 128 -> {n_params(big):,}")
        check(f"{name}: hidden recorded on the module",
              small.hidden == 32 and big.hidden == 128)
        shallow = cls(input_dim=7, num_layers=2, hidden=32)
        deep = cls(input_dim=7, num_layers=8, hidden=32)
        check(f"{name}: num_layers deepens the model",
              n_params(shallow) < n_params(deep),
              f"2 -> {n_params(shallow):,} | 8 -> {n_params(deep):,}")

    # Attention models must reject a width that is not divisible by the heads.
    for name in ("gat", "transformer"):
        try:
            MODELS[name](input_dim=7, hidden=30, heads=4)
            ok = False
        except AssertionError:
            ok = True
        check(f"{name}: rejects hidden not divisible by heads", ok)


def test_forward_shapes():
    print("\nForward pass shapes hold at every searched width")
    d = _toy_dataset(1)[0]
    for name, cls in MODELS.items():
        for hidden in (32, 64, 128):
            m = cls(input_dim=7, num_layers=2, hidden=hidden)
            m.train()
            out = m(d)
            check(f"{name} hidden={hidden} -> (N, 4)", tuple(out.shape) == (5, 4),
                  str(tuple(out.shape)))


def test_arch_config_loading():
    print("\narch_config loading")
    try:
        load_arch_config(None, ["gcn"])
        ok = False
    except SystemExit:
        ok = True
    check("missing --arch_config is a hard error", ok)
    check("explicit opt-in returns empty (default) configs",
          load_arch_config(None, ["gcn"], allow_default=True) == {"gcn": {}})

    with tempfile.TemporaryDirectory() as tmp:
        good = os.path.join(tmp, "good.json")
        with open(good, "w") as fh:
            json.dump({"configs": {"gcn": {"num_layers": 3, "hidden": 32,
                                           "learning_rate": 3e-4}}}, fh)
        cfg = load_arch_config(good, ["gcn"])
        check("reads the tune_budget.py 'configs' wrapper",
              cfg["gcn"] == {"num_layers": 3, "hidden": 32, "learning_rate": 3e-4})
        check("config drives the constructor",
              _build_model("gcn", cfg["gcn"], "cpu").hidden == 32)
        check("config columns are carried on result rows",
              _config_columns("gcn", cfg["gcn"])
              == {"num_layers": 3, "hidden": 32, "learning_rate": 3e-4})

        try:
            load_arch_config(good, ["gcn", "gat"])
            ok = False
        except SystemExit:
            ok = True
        check("a model missing from the config is a hard error", ok)

        bad = os.path.join(tmp, "bad.json")
        with open(bad, "w") as fh:
            json.dump({"gcn": {"num_layers": 2, "dropout": 0.5}}, fh)
        try:
            load_arch_config(bad, ["gcn"])
            ok = False
        except SystemExit:
            ok = True
        check("an unknown config key is a hard error", ok)


def _toy_dataset(n=4, n_bus=5):
    torch.manual_seed(0)
    out = []
    for _ in range(n):
        ei = torch.tensor([[0, 1, 2, 3, 1, 2, 3, 4], [1, 2, 3, 4, 0, 1, 2, 3]])
        out.append(Data(x=torch.randn(n_bus, 7), edge_index=ei,
                        edge_attr=torch.randn(ei.shape[1], 4),
                        y=torch.randn(n_bus, 4), dc_pf=torch.randn(n_bus, 4)))
    return out


def _toy_data(grids):
    return {g: {s: _toy_dataset() for s in ("train", "val", "test")} for g in grids}


def test_arma_edge_weight_is_non_negative():
    print("\nARMA edge weight keeps the ARMAConv normalization defined")
    torch.manual_seed(0)
    arma = MODELS["arma_gnn"](input_dim=7, num_layers=2, hidden=32)
    edge_attr = torch.randn(64, 4) * 50          # extreme, both signs
    w = arma._scalar_edge(edge_attr)
    check("arma: edge weight is non-negative", bool((w >= 0).all()),
          f"min={float(w.min()):.4g}")
    check("arma: edge weight is finite", bool(torch.isfinite(w).all()))

    gcn = MODELS["gcn"](input_dim=7, num_layers=2, hidden=32)
    check("gcn: edge encoder left as it was (may be negative)",
          bool((gcn._scalar_edge(edge_attr) < 0).any()))

    d = _toy_dataset(1)[0]
    arma.train()
    check("arma: forward is finite on a toy graph",
          bool(torch.isfinite(arma(d)).all()))


def test_grad_clip_is_opt_in():
    print("\nGradient clipping is an opt-in argument")
    data = _toy_dataset(4)
    losses = []
    for clip in (None, 1.0):
        torch.manual_seed(0)
        model = MODELS["gcn"](input_dim=7, num_layers=1, hidden=32)
        tl, vl = make_loaders(data, data, batch_size=2)
        losses.append(float(train(model, "cpu", tl, vl, epochs=1,
                                  grad_clip=clip)))
    check("default (None) and explicit clipping both train",
          all(v == v for v in losses), f"val={losses}")


def test_runners_seeds_and_checkpoints():
    print("\nRunners: seed columns, seeded checkpoint names, resumability")
    grids = ["G1", "G2"]
    data = _toy_data(grids)
    cfg = {"gcn": {"num_layers": 1, "hidden": 32, "learning_rate": 1e-3}}
    seeds = [0, 100]

    with tempfile.TemporaryDirectory() as tmp:
        wi = run_within(data, grids, ["gcn"], "cpu", 1, seeds, cfg,
                        batch_size=2, save_dir=tmp, regime_tag="A")
        check("within: one row per grid x seed", len(wi) == len(grids) * len(seeds),
              str(len(wi)))
        check("within: seed and regime on every row",
              all(r["seed"] in seeds and r["regime"] == "A" for r in wi))
        check("within: config provenance on every row",
              all(r["hidden"] == 32 and r["num_layers"] == 1 for r in wi))
        check("within: plain MSE present", all("mse" in r for r in wi))
        expected = {f"within_gcn_{g}_s{s}.pt" for g in grids for s in seeds}
        check("within: checkpoints carry the seed",
              expected <= set(os.listdir(tmp)), str(sorted(os.listdir(tmp))))

        # Resuming must reuse the checkpoints rather than retrain: same weights.
        before = torch.load(os.path.join(tmp, "within_gcn_G1_s0.pt"))
        run_within(data, grids, ["gcn"], "cpu", 1, seeds, cfg, batch_size=2,
                   save_dir=tmp, skip_existing=True, regime_tag="A")
        after = torch.load(os.path.join(tmp, "within_gcn_G1_s0.pt"))
        check("skip_existing leaves the checkpoint untouched",
              all(torch.equal(before[k], after[k]) for k in before))

    with tempfile.TemporaryDirectory() as tmp:
        cc = run_cross_context(data, grids, ["gcn"], "cpu", 1, seeds, cfg,
                               batch_size=2, save_dir=tmp, regime_tag="B")
        check("cross: one row per train x test x seed",
              len(cc) == len(grids) ** 2 * len(seeds), str(len(cc)))
        check("cross: checkpoints carry the seed",
              {f"cc_gcn_{g}_s{s}.pt" for g in grids for s in seeds}
              <= set(os.listdir(tmp)))
        check("cross: unseen flag still present",
              all("unseen" in r for r in cc))

    with tempfile.TemporaryDirectory() as tmp:
        ood = run_ood(data, grids, ["gcn"], "cpu", 1, seeds, cfg, batch_size=4,
                      save_dir=tmp, regime_tag="B")
        check("ood: one row per held-out grid x seed",
              len(ood) == len(grids) * len(seeds), str(len(ood)))
        check("ood: checkpoints carry the seed",
              {f"ood_gcn_heldout_{g}_s{s}.pt" for g in grids for s in seeds}
              <= set(os.listdir(tmp)))

    # Different seeds must actually produce different models.
    torch.manual_seed(0)
    a = _build_model("gcn", cfg["gcn"], "cpu").state_dict()
    torch.manual_seed(100)
    b = _build_model("gcn", cfg["gcn"], "cpu").state_dict()
    check("different seeds -> different initial weights",
          any(not torch.equal(a[k], b[k]) for k in a))


def test_ood_fold_sharding_is_equivalent():
    """--held_out only splits the work: a fold's rows and weights must not
    depend on whether it ran alone or alongside the other folds. Otherwise the
    sharded campaign is not the experiment the unsharded command describes."""
    print("\nOOD fold sharding changes scheduling, not results")
    grids = ["G1", "G2", "G3"]
    cfg = {"gcn": {"num_layers": 1, "hidden": 32, "learning_rate": 1e-3}}
    seeds = [0]

    def run(held_out, tmp):
        torch.manual_seed(0)
        return run_ood(_toy_data(grids), grids, ["gcn"], "cpu", 1, seeds, cfg,
                       batch_size=4, save_dir=tmp, regime_tag="B",
                       held_out=held_out)

    with tempfile.TemporaryDirectory() as whole, \
            tempfile.TemporaryDirectory() as shard:
        full = run(None, whole)
        shards = [r for g in grids for r in run([g], shard)]

        check("unsharded run covers every fold", len(full) == len(grids),
              str(len(full)))
        check("one shard per fold reproduces the same row count",
              len(shards) == len(full), str(len(shards)))
        by_fold = {r["held_out_grid"]: r for r in shards}
        check("shards cover exactly the same folds",
              set(by_fold) == {r["held_out_grid"] for r in full})
        same = all(by_fold[r["held_out_grid"]]["nrmse"] == r["nrmse"]
                   for r in full)
        check("per-fold nrmse is identical when sharded", same)
        check("checkpoint names are the same either way",
              set(os.listdir(whole)) == set(os.listdir(shard)))

    with tempfile.TemporaryDirectory() as tmp:
        only = run_ood(_toy_data(grids), grids, ["gcn"], "cpu", 1, seeds, cfg,
                       batch_size=4, save_dir=tmp, regime_tag="B",
                       held_out=["G2"])
        check("a single-fold shard trains only that fold",
              [r["held_out_grid"] for r in only] == ["G2"],
              str([r["held_out_grid"] for r in only]))
        check("and writes only that fold's checkpoint",
              os.listdir(tmp) == ["ood_gcn_heldout_G2_s0.pt"],
              str(os.listdir(tmp)))


def main():
    test_default_is_unchanged()
    test_hidden_and_depth_are_honoured()
    test_forward_shapes()
    test_arch_config_loading()
    test_arma_edge_weight_is_non_negative()
    test_grad_clip_is_opt_in()
    test_runners_seeds_and_checkpoints()
    test_ood_fold_sharding_is_equivalent()
    print("\n" + "=" * 50)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
