#!/bin/bash
# CIFAR-10, strong Ada search (ada_cifar10_strong; winner bp1485 at feature multiplier 3, 11.1M params).
#  1. teammate's recipe: magnitude V=8, M in {4,8,16}, 8-epoch Adam recover        -> runs/ada_cifar10_strong/vnm_repro
#  2. our recipe: magnitude V in {8,32}, M in {4..32}, 30-epoch KD fine-tune        -> runs/ada_cifar10_strong/vnm_grid
#  2b. control: the dense model fine-tuned with the same 30-epoch recipe, no pruning (v=0, m=1 in vnm_grid)
#  3. gradual OBS V in {8,32}, M in {16,32}                                         -> runs/ada_cifar10_strong/vnm_gradual
# The search + full train itself: scripts/run_search.sh ada_cifar10_strong (80 generations, about 2 days on 1 GPU).
P=$(cd "$(dirname "$0")/.." && pwd)
source $P/scripts/env.sh
R=ada_cifar10_strong
(cd $P/CoDeepNEAT && CUDA_VISIBLE_DEVICES=0 python -W ignore ../vnm/prune_eval.py --config $R --run $R \
    --model runs/$R/fully_trained_models/bp-1485_fm-3-best-3.model --vs 8 --ms 4,8,16 --ft-epochs 8 \
    --out vnm_repro --layers-for-v "" > $P/logs/${R}_repro.log 2>&1) &
cd $P/vnm
for S in 0 1; do
  CUDA_VISIBLE_DEVICES=$((S+1)) python -W ignore prune_grid2.py --run $R --dataset cifar10 --top 1 --src dense_fm3 \
      --out vnm_grid --vs 8,32 --ms 4,8,16,32 --ft-epochs 30 --layers-for-v 32 --shard $S/2 > $P/logs/${R}_grid_$S.log 2>&1 &
done
CUDA_VISIBLE_DEVICES=0 python -W ignore prune_grid2.py --run $R --dataset cifar10 --top 1 --src dense_fm3 --out vnm_grid \
    --vs 8 --ms 4 --ft-epochs 30 --layers-for-v "" --dense-control > $P/logs/${R}_densectl.log 2>&1 &
wait
for S in 0 1; do
  CUDA_VISIBLE_DEVICES=$((S+1)) python -W ignore prune_gradual.py --run $R --dataset cifar10 --top 1 --src dense_fm3 \
      --out vnm_gradual --vs 8,32 --ms 16,32 --shard $S/2 > $P/logs/${R}_gradual_$S.log 2>&1 &
done
wait
