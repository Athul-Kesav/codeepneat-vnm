#!/bin/bash
# V x M grid on the ResNet baselines, Spatha timing V=32/64/128, summary.
P=$(cd "$(dirname "$0")/.." && pwd)
cd $P
source scripts/env.sh
echo "=== GRIDRN START $(date -Is)" >> logs/gridrn.log
for g in 0 1 2; do
  (cd vnm && CUDA_VISIBLE_DEVICES=$g python prune_resnet_grid.py --shard $g/3 >> $P/logs/resnet_grid_gpu$g.log 2>&1) &
done
wait
echo "=== GRIDRN DONE $(date -Is)" >> logs/gridrn.log
( source scripts/venom_env.sh
  for V in 128 64 32; do
    L=""; [ $V != 128 ] && L=$VENOM_BUILD/spatha_v${V}_lib
    SPATHA_LIB=$L CUDA_VISIBLE_DEVICES=0 python -W ignore vnm/spatha_bench.py --run venom_cifar100 --dir vnm_grid_resnet --pattern "*_v${V}_*_layers.pt" >> logs/resnet_grid_spatha.log 2>&1
  done )
# tables: see results/grids/
echo "=== GRIDRN ALL DONE $(date -Is)" >> logs/gridrn.log
