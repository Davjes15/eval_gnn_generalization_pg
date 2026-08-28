# Reproducibility: exact commands, versions, data provenance, checkpoints

Audit item **A4**. The finding was that the results could not be reproduced from
the repository: the tuning artifacts the configuration tables cite were not
committed, the environment was not pinned, the datasets carried no provenance, and
no trained weights were released. This document is the single place that closes
those gaps and points at the artifacts.

Two independent reproduction paths are supported, and they answer different
questions:

| path | cost | what it establishes |
|---|---|---|
| **replay** a saved checkpoint (§4) | minutes | the reported number is the number this weight file produces |
| **retrain** from the recorded seed (§3) | ~24-36 h on 8 cores | the whole pipeline, including training, reproduces the number |

---

## 1. Environment

`requirements.txt` carries the pins. The versions the final campaign actually ran
under:

| package | version | note |
|---|---|---|
| Python | 3.10.12 | |
| torch | 2.13.0+cu130 | CPU execution throughout; no GPU was used |
| torch_geometric | 2.8.0 | |
| **pandapower** | **3.5.4** | hard pin -- the DC baseline is version-sensitive (A1) |
| numpy | 2.2.1 | |
| scipy | 1.14.1 | |
| pandas | 2.3.3 | |
| networkx | 3.4.2 | |

OS: Linux 5.15, x86_64, 8 cores. `pandapower` is the only pin that affects
*correctness* rather than convenience: `rundcpp` does not write `res_bus.q_mvar`,
2.x left the column NaN and 3.x leaves the previous AC result there, which is what
produced the A1 leak. The generator now zeroes it explicitly, so the pin is for
byte-reproducibility of the generated datasets.

Threading matters for wall-clock reproduction, not for values: every job runs with
`OMP_NUM_THREADS=1` and the campaign is parallelised across jobs instead. Running
multi-threaded jobs in parallel oversubscribes the cores and was measurably slower.

---

## 2. Data provenance

Three dataset directories exist and they are not interchangeable.

| directory | regime | protocol | status |
|---|---|---|---|
| `data_a` | A (within-grid) | k = 0, no contingencies, `--unique_demand` | **final** |
| `data_full` | B (transfer) | k <= 2, random time sampling | **superseded** -- shared demand snapshots across splits (A5) |
| `data_full_v2` | B (transfer) | k <= 2, `--time_split blocked` | **final** |

Regeneration commands:

```bash
# Regime A -- fixed topology, one demand snapshot per sample
python transmission_graph_gen.py --grid all --max_k 0 --unique_demand \
    --n_train 800 --n_val 100 --n_test 100 --out_dir data_a

# Regime B -- N-1/N-2 contingencies, blocked temporal split (A5)
python transmission_graph_gen.py --grid all --max_k 2 \
    --time_split blocked \
    --n_train 800 --n_val 100 --n_test 100 --out_dir data_full_v2
```

Every split directory carries a `dataset_src.csv` with one row per sample:
`grid, t_idx, k, out_lines, source`. That file *is* the provenance -- it lets
anyone re-derive which demand snapshot and which outaged lines produced each
graph, and it is what makes the split-hygiene claim checkable rather than asserted.
The **tensors are not in the repository** (40 MB + 39 MB of `.pt`; `data_full` is a
symlink to a working directory), but all 24 provenance files are, under
`docs/provenance/<dataset>_<grid>_<split>.csv`, so the sampling can be audited
without the archive. Releasing the tensors and the checkpoints needs a data
release (e.g. Zenodo), not a git commit; until then the reproduction path is
regenerate-then-verify, and `docs/provenance/` is what a verification is against.

For `data_full_v2` the realised demand-time windows are:

| grid | train `t_idx` | val `t_idx` | test `t_idx` |
|---|---|---|---|
| IEEE24 | 62 - 27,871 | 27,975 - 31,441 | 31,595 - 34,997 |
| IEEE39 | 10 - 27,871 | 28,009 - 31,446 | 31,582 - 34,950 |
| IEEE118 | 5 - 27,869 | 27,992 - 31,418 | 31,558 - 34,773 |
| UK | 13 - 27,845 | 28,022 - 31,433 | 31,597 - 35,015 |

Disjoint, in chronological order, with at least a one-day (96-step) gap between
windows. The windows are requested as `(0, 27878) / (27974, 31458) / (31554,
35038)` out of the 35,040 15-minute steps of the demand year; the realised ranges
sit inside them because samples whose AC solve failed or whose voltages left
[0.8, 1.2] p.u. are rejected and resampled.

Verify a dataset directory before using it:

```bash
python validate.py --data_dir data_full_v2 --expect_blocked   # gate H = split hygiene
python tests/test_split_hygiene.py
```

Regeneration is not bit-exact: the two commands above resample demand snapshots
and contingencies, so a regenerated `data_full_v2` is a *different draw from the
same protocol*, not the same file. What must be reproduced is the protocol
(counts, disjoint blocked windows, `k` distribution, convergence) -- which is
exactly what `validate.py --expect_blocked` checks, and why the realised windows
and per-sample provenance are recorded here rather than left implicit. Results
reproduced from a fresh draw should be compared as distributions, not as identical
numbers; use the saved checkpoints (§4) for the exact numbers.

---

## 3. Retraining from seed

Configurations are frozen in `configs/arch_config.json` (selection evidence:
`results_a/*/tuning.csv`, `tuning_summary.csv`, `tuning_per_grid_argmin.csv`,
rationale in `docs/Model_configurations.md`). Seeds are `0 100 300 700 1000` for
every architecture except NNConv, which has `0 100 300` (documented asymmetry;
A8).

The final normalized campaign is one script:

```bash
bash launch_normalized.sh 7 within cross ood     # 7 = size of the process pool
```

which expands to 18 independent jobs of the form

```bash
OMP_NUM_THREADS=1 python -u experiments.py \
    --experiment within --data_dir data_a --batch_size 32 \
    --models gcn --seeds 0 100 300 700 1000 --epochs 200 \
    --arch_config configs/arch_config.json \
    --normalize pu_zscore --regime_tag A \
    --out results_norm/within_gcn --save_models ckpt_norm/within_gcn \
    --skip_existing
```

with `--experiment cross` / `--experiment ood` on `data_full_v2` (and
`--batch_size_ood 96`) for the transfer arms. `--skip_existing` makes a job
resumable: an interrupted arm restarts at the first missing checkpoint rather than
from the beginning.

Two representations are supported and both are reported: `--normalize pu_zscore`
is the final protocol (A2), `--normalize none` is the raw-unit ablation that
reproduces every pre-A2 artifact bit-identically. All metrics are computed after
de-normalization, in physical units, in both cases.

---

## 4. Replaying a saved checkpoint

`experiments.py --save_models <dir>` writes one file per (arm, model, grid, seed):

| arm | filename | the grid in the name is |
|---|---|---|
| within-grid | `within_<model>_<grid>_s<seed>.pt` | the grid trained *and* tested on |
| cross-context | `cc_<model>_<train_grid>_s<seed>.pt` | the **training** grid (tested on all four) |
| leave-one-out | `ood_<model>_heldout_<grid>_s<seed>.pt` | the **held-out** grid (trained on the other three) |

The weights live outside git for the same size reason as the datasets (~50 MB and
growing), so what the repository carries is the **index**: one row per file with
its relative path, byte size, SHA-256 and parameter count. That is enough to name
the file behind any results row and to verify a copy received out of band. Build
and check it with:

```bash
python checkpoint_index.py --ckpt_root ckpt_norm --out docs/tables/checkpoint_index.csv
python tests/test_checkpoint_index.py
```

and the physics-aware evaluation replays the whole tree without training anything:

```bash
python eval_checkpoints.py --ckpt_root ckpt_norm \
    --data_a data_a --data_b data_full_v2 \
    --normalize pu_zscore --out results_norm/physics
```

The replay re-fits the scaler exactly as training did -- train split of that grid
for the within-grid and cross-context arms, pooled train splits of the three
retained grids for an OOD fold -- because a scaler fitted on anything else would
leak, and a scaler fitted differently would not reproduce the reported number.

**Not every result has a checkpoint.** The pre-A2 raw-unit campaign
(`results/`) was run before checkpointing was made a requirement, and its ARMA
checkpoints were deleted because they predated the softplus fix and contained NaN
tensors (`ckpt_a/PROVENANCE.txt`, `ckpt_b/PROVENANCE.txt`). Those rows reproduce
from seed only. Everything in the final normalized campaign has weights on disk
and an index entry (336 checkpoints, all replayed).

**The result tables themselves are in the repository**, so a reviewer can check
every number without the weights or the tensors: the merged per-run rows
(`results_norm/all_within/within_grid.csv`, `all_cross/cross_context.csv`,
`all_ood/ood.csv`, each with the `summary.json` that records the protocol the
shards were merged under), the 672-row physics replay
(`results_norm/physics/physics_metrics.csv`), the topology and DC tables
(`results_norm/topology/`, `results_norm/dc_baseline_regime_a.csv`) and everything
derived from them (`results_norm/analysis/`, including `nonfinite_runs.csv` and
the exact permutation test `rank_permutation_test.csv`). Re-running the two
analysis commands in §5 on those CSVs
must reproduce the tables in `docs/Normalization_results.md` §4 exactly; that is
the cheapest end-to-end check of this repository.

---

## 5. Downstream analysis, from the committed artifacts

The campaign is sharded one process per architecture, so each arm is merged
first. The two heavy architectures (ARMA, NNConv) are sharded further inside
their arm -- by seed (`--seeds`) and, in the OOD arm, by held-out grid
(`--held_out`) -- because a single process walks folds and seeds sequentially and
would have left cores idle for tens of hours. Sharding is a scheduling change
only: a seed shard trains the same seeds it would have trained in sequence, and
an OOD fold shard still pools *all* other grids for training, so a fold's number
does not depend on how the arm was split (asserted by
`test_ood_fold_sharding_is_equivalent` in `tests/test_plumbing.py`, which runs
one arm whole and one fold-per-process and requires identical per-fold NRMSE and
identical checkpoint names). Each shard writes its own results directory and all
shards of an arm share one checkpoint directory -- checkpoint names carry model,
arm, grid/fold and seed, so they cannot collide -- and the glob in step 1 below
picks up the extra directories without change. `gather_results.py` refuses a
merge that is not consistent with one frozen protocol (an architecture in two shards, a missing architecture,
disagreeing seeds/epochs/data_dir/batch size, two configurations for one
architecture), which is the failure that would silently invalidate the ranking.

The merge is two-stage, because a seed/fold-sharded architecture is consolidated
into a single directory first and that directory is then merged with the other
five as an ordinary shard. `--seed_shards` is required at *both* stages here:
NNConv carries three seeds where the other five architectures carry five, so
even the final cross-architecture merge is a merge of shards with different seed
lists. Under `--seed_shards` a run is identified by model, seed and the arm's
grid columns, so two architectures sharing a `(seed, grid)` merge cleanly while
one shard silently redoing another's run is still refused. `MODELS` below is the
six architectures.

```bash
MODELS="gcn arma_gnn gat gin transformer nnconv"

# 1a. consolidate the architectures whose runs were split by seed / held-out grid
python gather_results.py --shards "results_norm/ood_arma_gnn_s*" --file ood.csv \
    --out results_norm/ood_arma_gnn_merged --models arma_gnn --seed_shards
python gather_results.py --shards "results_norm/ood_nnconv_s*" --file ood.csv \
    --out results_norm/ood_nnconv_merged --models nnconv --seed_shards

# 1b. merge the six architectures into one directory per arm
python gather_results.py --shards "results_norm/within_*" --file within_grid.csv \
    --out results_norm/all_within --models $MODELS --seed_shards
python gather_results.py --shards "results_norm/cross_*" --file cross_context.csv \
    --out results_norm/all_cross --models $MODELS --seed_shards
python gather_results.py --shards "results_norm/ood_*merged" "results_norm/ood_g*" \
    "results_norm/ood_transformer" --file ood.csv \
    --out results_norm/all_ood --models $MODELS --seed_shards

# 2. the model-independent tables (see below) -- one shared copy
python experiments.py --only_topology --experiment ood --data_dir data_full_v2 \
    --out results_norm/topology --regime_tag B --models gcn \
    --arch_config configs/arch_config.json

# 3. the DC baseline on the Regime A data (the topology shard covers Regime B)
python recompute_dc_baseline.py --data_dir data_a \
    --out results_norm/dc_baseline_regime_a.csv

# 4. ranking and downstream tables
python rank_analysis.py --regime_a results_norm/all_within/within_grid.csv \
    --cross results_norm/all_cross/cross_context.csv \
    --ood results_norm/all_ood/ood.csv \
    --out results_norm/analysis
python recompute_tables.py --within results_norm/all_within/within_grid.csv \
    --cross results_norm/all_cross/cross_context.csv \
    --ood results_norm/all_ood/ood.csv \
    --topology results_norm/topology \
    --dc_regime_a results_norm/dc_baseline_regime_a.csv \
    --dc_regime_b results_norm/topology/dc_baseline.csv \
    --out results_norm/analysis
python mmd_report.py --data_dir data_full_v2 --out docs/tables   # A7 distances
```

**Why step 2 exists.** The MMD matrix, the pooled OOD distances and the DC
baseline depend only on the data, not on the architecture, so recomputing them in
all 18 training shards wastes the cores the training needs -- the campaign runs
with `--skip_mmd`. `--only_topology` writes exactly those tables and exits
without training, from the same code path the training shards would have used, so
the analysis input is identical to the one an unsharded run would have produced.
`recompute_tables.py` still cross-checks the shards it is given and refuses to
merge tables that disagree.

Whole test suite, then the style check (`setup.cfg` holds the settings, so the
bare command is the check; it exits 0 on a clean tree):

```bash
for t in tests/test_*.py; do python "$t" || echo "FAILED: $t"; done
python -m flake8 .
```

---

## 6. What still limits reproducibility

Stated because it cannot be fixed by tooling:

* **One data-generation realization.** Seeds vary training initialization only.
  There is no second dataset draw, so every error bar understates total
  uncertainty -- the contingency and demand sampling contribute variance that is
  nowhere measured.
* **Four grids.** n = 4 with size, density and load scale confounded. This bounds
  what the rank correlations and the g-score can support (A6, A8).
* **NNConv has three seeds, not five**, while being the highest-variance model.
* **CPU determinism only.** Runs are reproducible on the same platform with the
  pinned versions; torch does not promise bit-identical results across versions or
  hardware, so replay-from-checkpoint is the stronger of the two paths.
* **Tensors and weights are not in git.** `docs/provenance/` and
  `docs/tables/checkpoint_index.csv` describe them exactly, but reproducing the
  published numbers to the digit needs the archive itself, which needs a data
  release. Regenerating from the commands above reproduces the *protocol*, not the
  draw.
