#!/bin/bash
# One-shot OBS (SparseGPT-style, V:N:M-aware) + 30-epoch KD fine-tune on the residual+GAP winners.
P=$(cd "$(dirname "$0")/.." && pwd)
source $P/scripts/env.sh
cd $P/vnm; mkdir -p $P/logs
for G in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$G python prune_grid2.py --run venom_cifar100_res --dataset cifar100 --top 3 --src retrained_v2 \
      --out vnm_gridres_obs --method obs --vs 8,32,128 --ms 8,16,32 --ft-epochs 30 --layers-for-v "" \
      --shard $G/3 > $P/logs/gridres_obs_gpu$G.log 2>&1 &
done
wait
