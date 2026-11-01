#!/bin/bash
# usage: run_vnm.sh <config/run name> <gpu> <ft_epochs>   V:N:M prune + fine-tune top-3, then Spatha timing
P=$(cd "$(dirname "$0")/.." && pwd)
C=$1; GPU=$2; FT=${3:-8}
LOG=$P/logs/${C}_vnm.log
source $P/scripts/env.sh
cd $P
echo "=== VNM START $(date -Is)" >> $LOG
CUDA_VISIBLE_DEVICES=$GPU python vnm/prune_eval.py --config $C --run $C --v 128 --ms 4,8,16 --ft-epochs $FT --top-k 3 >> $LOG 2>&1
echo "=== SPATHA START $(date -Is)" >> $LOG
( source $P/scripts/venom_env.sh; CUDA_VISIBLE_DEVICES=$GPU python -W ignore vnm/spatha_bench.py --run $C --pattern "*_v128_*_layers.pt" >> $LOG 2>&1 )
echo "=== VNM DONE $(date -Is)" >> $LOG
