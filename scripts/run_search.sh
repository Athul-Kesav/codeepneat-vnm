#!/bin/bash
# usage: run_search.sh <config_name>   (evolve -> fully train top-3)
P=$(cd "$(dirname "$0")/.." && pwd)
source $P/scripts/env.sh
cd $P/CoDeepNEAT
C=$1
mkdir -p $P/logs
echo "=== EVO START $(date -Is)" >> $P/logs/$C.log
python src/main/evo.py -c $C -g 3 >> $P/logs/$C.log 2>&1
echo "=== FT START $(date -Is)" >> $P/logs/$C.log
python src/main/ft.py -c $C -g 3 >> $P/logs/$C.log 2>&1
echo "=== SEARCH DONE $(date -Is)" >> $P/logs/$C.log
