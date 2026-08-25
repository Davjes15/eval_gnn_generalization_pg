"""Probe: does gradient clipping stabilise ARMA on the seeds that diverge?"""
import argparse
import time

import torch

from experiments import _build_model, _load_all
from training_utils import get_device, make_loaders, train

p = argparse.ArgumentParser()
p.add_argument("--grid", required=True)
p.add_argument("--seeds", type=int, nargs="+", default=[100, 700])
p.add_argument("--num_layers", type=int, default=8)
p.add_argument("--hidden", type=int, default=128)
p.add_argument("--learning_rate", type=float, default=1e-3)
p.add_argument("--grad_clip", type=float, default=1.0)
p.add_argument("--epochs", type=int, default=200)
p.add_argument("--data_dir", default="data_a")
a = p.parse_args()

device = get_device()
data = _load_all(a.data_dir, [a.grid])
cfg = {"num_layers": a.num_layers, "hidden": a.hidden,
       "learning_rate": a.learning_rate}
for seed in a.seeds:
    for clip in ([a.grad_clip, None] if a.grad_clip > 0 else [None]):
        torch.manual_seed(seed)
        model = _build_model("arma_gnn", cfg, device)
        tl, vl = make_loaders(data[a.grid]["train"], data[a.grid]["val"],
                              batch_size=32)
        t0 = time.time()
        val = train(model, device, tl, vl, epochs=a.epochs,
                    learning_rate=a.learning_rate, grad_clip=clip)
        print(f"{a.grid} s{seed} clip={clip}: val={float(val):.6g} "
              f"({round(time.time() - t0, 1)}s)", flush=True)
