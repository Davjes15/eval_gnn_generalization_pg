#!/usr/bin/env bash
# launch_normalized.sh -- the final campaign: every architecture under
# --normalize pu_zscore (audit item A2), on the split-hygienic data (A5).
#
# One job per (arm, model) so result CSVs and checkpoints never collide, run
# through a fixed-width pool. --save_models makes every row replayable from its
# checkpoint and --skip_existing makes an interrupted or re-launched shard
# resume instead of retraining, so calling this script twice is safe.
#
#   bash launch_normalized.sh [parallel_jobs] [arms...]
#
# Examples:
#   bash launch_normalized.sh 7                # everything
#   bash launch_normalized.sh 3 cross ood      # Regime B only, smaller pool
#
# Arms and their data:
#   within        -> data_a         (Regime A, fixed topology, unique demand)
#   cross | ood   -> data_full_v2  (Regime B, N-k topologies, blocked temporal
#                                   split; the pre-A5 data_full is superseded)
set -u
cd "$(dirname "$0")"

POOL=${1:-7}
shift || true
ARMS=${*:-"within cross ood"}
OUT=results_norm
CKPT=ckpt_norm
CFG=configs/arch_config.json
DATA_A=data_a
DATA_B=data_full_v2
SEEDS_STD="0 100 300 700 1000"
SEEDS_NNCONV="0 100 300"   # as agreed for the raw-unit campaign, kept identical
# slowest first: a pool empties fastest when the long jobs start earliest
MODELS="arma_gnn nnconv transformer gin gat gcn"

mkdir -p "$OUT" "$CKPT" logs

jobs_file=$(mktemp)
for arm in $ARMS; do
  for m in $MODELS; do
    seeds=$SEEDS_STD
    [ "$m" = nnconv ] && seeds=$SEEDS_NNCONV
    case $arm in
      within) data=$DATA_A; tag=A; extra="--experiment within --batch_size 32" ;;
      cross)  data=$DATA_B; tag=B; extra="--experiment cross --batch_size 32" ;;
      ood)    data=$DATA_B; tag=B; extra="--experiment ood --batch_size_ood 96" ;;
      *) echo "unknown arm: $arm" >&2; exit 2 ;;
    esac
    echo "OMP_NUM_THREADS=1 python -u experiments.py $extra --data_dir $data \
--models $m --seeds $seeds --epochs 200 --arch_config $CFG --regime_tag $tag \
--normalize pu_zscore --skip_mmd --save_models $CKPT/${arm}_${m} \
--skip_existing --out $OUT/${arm}_${m} > logs/norm_${arm}_${m}.log 2>&1" >> "$jobs_file"
  done
done

echo "$(wc -l < "$jobs_file") jobs over arms [$ARMS], pool of $POOL"
xargs -a "$jobs_file" -I CMD -P "$POOL" bash -c CMD
rm -f "$jobs_file"
echo "done"
