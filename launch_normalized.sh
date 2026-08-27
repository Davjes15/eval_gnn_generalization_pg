#!/usr/bin/env bash
# launch_normalized.sh -- N2 campaign: the full benchmark under --normalize pu_zscore.
#
# Rerun of every architecture and every arm with the A2 remediation in place
# (per-quantity scaling fitted on training data only, metrics reported in
# physical units). The raw-unit results in results/ are kept as the ablation.
#
# One job per (arm, model) so result CSVs and checkpoints never collide, run
# through a fixed-width pool. --save_models makes every row replayable from its
# checkpoint and --skip_existing makes an interrupted shard resumable.
#
#   bash launch_normalized.sh [parallel_jobs]
set -u
cd "$(dirname "$0")"

POOL=${1:-7}
OUT=results_norm
CKPT=ckpt_norm
CFG=configs/arch_config.json
SEEDS_STD="0 100 300 700 1000"
SEEDS_NNCONV="0 100 300"   # as agreed for the raw-unit campaign, kept identical
MODELS="gcn gat gin transformer arma_gnn nnconv"

mkdir -p "$OUT" "$CKPT" logs

jobs_file=$(mktemp)
for arm in within cross ood; do
  for m in $MODELS; do
    seeds=$SEEDS_STD
    [ "$m" = nnconv ] && seeds=$SEEDS_NNCONV
    case $arm in
      within) data=data_a;    tag=A; extra="--experiment within --batch_size 32" ;;
      cross)  data=data_full; tag=B; extra="--experiment cross --batch_size 32" ;;
      ood)    data=data_full; tag=B; extra="--experiment ood --batch_size_ood 96" ;;
    esac
    echo "OMP_NUM_THREADS=1 python -u experiments.py $extra --data_dir $data \
--models $m --seeds $seeds --epochs 200 --arch_config $CFG --regime_tag $tag \
--normalize pu_zscore --skip_mmd --save_models $CKPT/${arm}_${m} \
--skip_existing --out $OUT/${arm}_${m} > logs/norm_${arm}_${m}.log 2>&1" >> "$jobs_file"
  done
done

echo "$(wc -l < "$jobs_file") jobs, pool of $POOL"
xargs -a "$jobs_file" -I CMD -P "$POOL" bash -c CMD
rm -f "$jobs_file"
echo "done"
