#!/bin/bash
# After run_retrain.sh: V x M grid on the retrained top-3 (3 GPU shards per dataset), Spatha timing, summary.
P=$(cd "$(dirname "$0")/.." && pwd)
cd $P
source scripts/env.sh
for spec in "venom_cifar100 cifar100 30" "imagenet100 imagenet100 30"; do
  set -- $spec
  R=$1; D=$2; E=$3
  echo "=== GRID2 $R START $(date -Is)" >> logs/grid2.log
  for g in 0 1 2; do
    (cd vnm && CUDA_VISIBLE_DEVICES=$g python prune_grid2.py --run $R --dataset $D --top 3 --ft-epochs $E --shard $g/3 >> $P/logs/${R}_grid2_gpu$g.log 2>&1) &
  done
  wait
  echo "=== GRID2 $R DONE $(date -Is)" >> logs/grid2.log
  ( source scripts/venom_env.sh
    CUDA_VISIBLE_DEVICES=0 python -W ignore vnm/spatha_bench.py --run $R --dir vnm_grid2 --pattern "*_v128_*_layers.pt" >> logs/${R}_grid2_spatha.log 2>&1
    SPATHA_LIB=$VENOM_BUILD/spatha_v64_lib CUDA_VISIBLE_DEVICES=0 python -W ignore vnm/spatha_bench.py --run $R --dir vnm_grid2 --pattern "*_v64_*_layers.pt" >> logs/${R}_grid2_spatha.log 2>&1 )
  echo "=== SPATHA2 $R DONE $(date -Is)" >> logs/grid2.log
done
# tables: see results/grids/
echo "=== GRID2 ALL DONE $(date -Is)" >> logs/grid2.log
