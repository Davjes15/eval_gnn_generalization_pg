#!/usr/bin/env bash
# launch_ood_shards.sh -- run one architecture's OOD arm as one process per
# (seed, held-out grid) instead of one process per architecture.
#
# WHY THIS EXISTS
#   launch_normalized.sh gives each architecture one process, which walks the
#   folds and seeds sequentially. For the two heavy architectures that is the
#   wall-clock bottleneck: an NNConv OOD fold trains on three pooled grids and
#   takes ~13 h, so 4 folds x 3 seeds in one process is ~7 days while cores sit
#   idle. Splitting the same 12 runs into 12 processes turns that into the ~2
#   waves the machine can actually hold.
#
#   The split is scheduling only, not a protocol change: --held_out selects
#   which folds this process trains, and each fold still pools every other grid
#   for training, so a fold's numbers are the same as in an unsharded run
#   (asserted by test_ood_fold_sharding_is_equivalent in tests/test_plumbing.py).
#   Shards share one checkpoint directory -- names carry model, fold and seed --
#   and write separate result directories, merged afterwards with
#   `gather_results.py --seed_shards` (see docs/Reproducibility.md section 5).
#
# USAGE
#   bash launch_ood_shards.sh <model> "<seeds>" ["<folds>"] [target_procs]
#
#   bash launch_ood_shards.sh nnconv "0 100 300"
#   bash launch_ood_shards.sh arma_gnn "0 100" "UK IEEE118" 8
#
#   target_procs is the total number of experiments.py processes wanted on the
#   box, counting jobs launched by anything else: the queue is dispatched only
#   while fewer than that are running, so this backfills cores as other shards
#   of the campaign finish instead of oversubscribing them.
#
# --skip_existing means re-running this script is safe and free for the runs
# that already produced a checkpoint.
set -u
cd "$(dirname "$0")"

MODEL=${1:?usage: launch_ood_shards.sh <model> "<seeds>" ["<folds>"] [target_procs]}
SEEDS=${2:?seeds, e.g. "0 100 300"}
FOLDS=${3:-"UK IEEE118 IEEE39 IEEE24"}
TARGET=${4:-8}

OUT=results_norm
CKPT=ckpt_norm/ood_${MODEL}
CFG=configs/arch_config.json
DATA_B=data_full_v2

mkdir -p "$OUT" "$CKPT" logs

# Count the training processes themselves. The pattern is anchored because
# launch_normalized.sh wraps each job in `bash -c "... python -u experiments.py
# ..."`: an unanchored match counts that wrapper as well, the pool then looks
# twice as busy as it is, and nothing is ever dispatched.
running() { pgrep -fc "^python -u experiments.py" || true; }

for seed in $SEEDS; do
  for fold in $FOLDS; do
    tag=${MODEL}_s${seed}_${fold}
    while [ "$(running)" -ge "$TARGET" ]; do sleep 60; done
    nohup env OMP_NUM_THREADS=1 nice -n 5 python -u experiments.py \
      --experiment ood --batch_size_ood 96 --data_dir "$DATA_B" \
      --models "$MODEL" --seeds "$seed" --held_out "$fold" --epochs 200 \
      --arch_config "$CFG" --regime_tag B --normalize pu_zscore --skip_mmd \
      --save_models "$CKPT" --skip_existing \
      --out "$OUT/ood_${tag}" > "logs/norm_ood_${tag}.log" 2>&1 &
    echo "launched ood ${tag} (pid $!)"
    sleep 5
  done
done
echo "all shards launched; $(running) experiments.py process(es) running"
