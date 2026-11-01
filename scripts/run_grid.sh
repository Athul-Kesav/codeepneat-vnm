#!/bin/bash
# usage: run_grid.sh <run> <gpu> <ft_epochs>
# Full V x M grid for the top-3 models of one run -> runs/<run>/vnm_grid, then Spatha timing for V=128 and V=64.
P=$(cd "$(dirname "$0")/.." && pwd)
C=$1; GPU=$2; FT=$3
LOG=$P/logs/${C}_grid.log
source $P/scripts/env.sh
cd $P
echo "=== GRID START $(date -Is)" >> $LOG
CUDA_VISIBLE_DEVICES=$GPU python vnm/prune_eval.py --config $C --run $C --vs 8,16,32,64,128 --ms 4,8,16,32 \
    --ft-epochs $FT --top-k 3 --out vnm_grid >> $LOG 2>&1
echo "=== SPATHA START $(date -Is)" >> $LOG
( source $P/scripts/venom_env.sh
  CUDA_VISIBLE_DEVICES=$GPU python -W ignore vnm/spatha_bench.py --run $C --dir vnm_grid --pattern "*_v128_*_layers.pt" >> $LOG 2>&1
  SPATHA_LIB=$VENOM_BUILD/spatha_v64_lib CUDA_VISIBLE_DEVICES=$GPU python -W ignore vnm/spatha_bench.py --run $C --dir vnm_grid --pattern "*_v64_*_layers.pt" >> $LOG 2>&1 )
echo "=== GRID DONE $(date -Is)" >> $LOG
