#!/bin/bash
# Recipe v2 dense retraining (300 ep, AdamW, CutMix 0.5, EMA 0.999) of the evolved winners, sharded over 3 GPUs.
# usage: run_retrain_v2.sh orig   -> venom_cifar100 bp427/430/499
#        run_retrain_v2.sh res    -> venom_cifar100_res top-6 (residual + GAP search)
P=$(cd "$(dirname "$0")/.." && pwd)
source $P/scripts/env.sh
cd $P/vnm; mkdir -p $P/logs
case $1 in
  orig) ARGS="--run venom_cifar100 --n 3 --bps 427,430,499" ;;
  res)  ARGS="--run venom_cifar100_res --n 6" ;;
  *) echo "usage: $0 orig|res"; exit 1 ;;
esac
for G in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$G python train_candidates.py $ARGS --dataset cifar100 --epochs 300 --cutmix 0.5 --ema 0.999 \
      --shard $G/3 --out retrained_v2 > $P/logs/retrain_v2_$1_gpu$G.log 2>&1 &
done
wait
