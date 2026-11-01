#!/bin/bash
# V x M grid on the CIFAR-100 v2 models (300 ep, CutMix+EMA): 3 shards, Spatha V=32/64/128 timing, summary.
P=$(cd "$(dirname "$0")/.." && pwd)
cd $P
source scripts/env.sh
R=venom_cifar100_res
echo "=== GRIDRES START $(date -Is)" >> logs/gridres.log
for g in 0 1 2; do
  (cd vnm && CUDA_VISIBLE_DEVICES=$g python prune_grid2.py --run $R --dataset cifar100 --top 3 --ft-epochs 30 --src retrained_v2 --out vnm_gridres --layers-for-v 32,64,128 --shard $g/3 >> $P/logs/${R}_gridres_gpu$g.log 2>&1) &
done
wait
echo "=== GRIDRES DONE $(date -Is)" >> logs/gridres.log
( source scripts/venom_env.sh
  for V in 128 64 32; do
    L=""; [ $V != 128 ] && L=$VENOM_BUILD/spatha_v${V}_lib
    SPATHA_LIB=$L CUDA_VISIBLE_DEVICES=0 python -W ignore vnm/spatha_bench.py --run $R --dir vnm_gridres --pattern "*_v${V}_*_layers.pt" >> logs/${R}_gridres_spatha.log 2>&1
  done )
# tables: see results/grids/
echo "=== GRIDRES ALL DONE $(date -Is)" >> logs/gridres.log
