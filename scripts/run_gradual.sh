#!/bin/bash
# Gradual OBS (VENOM/oBERT style): column steps M/2 -> ... -> 4 (5 ep each), then 2:4 inside kept columns (30 ep).
P=$(cd "$(dirname "$0")/.." && pwd)
source $P/scripts/env.sh
cd $P/vnm; mkdir -p $P/logs
for G in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$G python prune_gradual.py --run venom_cifar100_res --dataset cifar100 --top 3 \
      --vs 8,32,128 --ms 8,16,32 --shard $G/3 > $P/logs/gridres_gradual_gpu$G.log 2>&1 &
done
wait
