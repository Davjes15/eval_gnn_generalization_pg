"""test_checkpoint_index.py -- checks for the A4 checkpoint index.

Run:  python3 tests/test_checkpoint_index.py

The index is the lookup table that maps a results row to the file that reproduces
it, so a filename parsed into the wrong arm or grid is worse than no index at all.
Verified here:
  * every naming convention `experiments.py --save_models` emits round-trips to
    the right (arm, model, grid, seed), including multi-word model names and the
    `heldout` infix of the OOD arm;
  * malformed names are rejected rather than guessed at;
  * `build` walks a tree, reports the parameter count and a content hash, and
    skips unparsable files instead of failing.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from checkpoint_index import build, parse_name, sha256

FAILURES = []


def check(label: str, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {label}: {got!r}")
    if not ok:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


print("-- filename conventions --")
check("within_gcn_IEEE24_s0.pt",
      parse_name("within_gcn_IEEE24_s0.pt"), ("within", "gcn", "IEEE24", 0))
check("multi-word model, within",
      parse_name("within_arma_gnn_IEEE118_s1000.pt"),
      ("within", "arma_gnn", "IEEE118", 1000))
check("cross-context names the TRAIN grid",
      parse_name("cc_transformer_UK_s300.pt"), ("cross", "transformer", "UK", 300))
check("ood names the HELD-OUT grid",
      parse_name("ood_gin_heldout_IEEE39_s700.pt"), ("ood", "gin", "IEEE39", 700))
check("multi-word model, ood",
      parse_name("ood_arma_gnn_heldout_UK_s100.pt"),
      ("ood", "arma_gnn", "UK", 100))

print("\n-- malformed names are rejected, not guessed --")
for bad in ("random_file.pt", "within_gcn.pt", "ood_gcn_IEEE24_s0.pt",
            "within_gcn_IEEE24_seed0.pt", "within_gcn_IEEE24_sX.pt"):
    check(f"reject {bad}", parse_name(bad), None)

print("\n-- build() over a tree --")
with tempfile.TemporaryDirectory() as root:
    os.makedirs(os.path.join(root, "within_gcn"))
    state = {"lin.weight": torch.zeros(3, 4), "lin.bias": torch.zeros(3)}
    for name in ("within_gcn_IEEE24_s0.pt", "within_gcn_UK_s100.pt"):
        torch.save(state, os.path.join(root, "within_gcn", name))
    with open(os.path.join(root, "within_gcn", "PROVENANCE.txt"), "w") as fh:
        fh.write("not a checkpoint\n")
    torch.save(state, os.path.join(root, "junk_name.pt"))

    df = build(root)
    check("rows = the two parsable checkpoints", len(df), 2)
    check("non-.pt files ignored", "PROVENANCE.txt" in "".join(df.path), False)
    check("parameter count = 3*4 + 3", sorted(set(df.n_params)), [15])
    check("grids recovered", sorted(df.grid), ["IEEE24", "UK"])
    # The hash is of the file, not of the tensors: torch.save writes a zip
    # whose metadata differs run to run, so two saves of the same weights hash
    # differently. It answers "is this the same file I published", which is what
    # a replicator needs; it is not a weight fingerprint.
    check("hash is a sha-256 digest",
          all(len(h) == 64 for h in df.sha256), True)
    check("hash matches the file on disk",
          df.sha256.iloc[0] == sha256(os.path.join(root, df.path.iloc[0])), True)
    check("paths are relative to the root",
          all(not p.startswith("/") for p in df.path), True)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(" -", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
