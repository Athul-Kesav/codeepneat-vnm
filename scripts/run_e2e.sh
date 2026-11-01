#!/bin/bash
# End-to-end Spatha inference benchmark: one V per GPU (V=32 / 64 / 128 builds), all models, M in {4,8,16,32}, batch 128 and 1.
P=$(cd "$(dirname "$0")/.." && pwd)
source $P/scripts/venom_env.sh
cd $P/vnm
G=$1; V=$2
if [ "$V" != 128 ]; then export SPATHA_LIB=$VENOM_BUILD/spatha_v${V}_lib; fi
CUDA_VISIBLE_DEVICES=$G python e2e_bench.py --v $V --rounds 5 --out ../results/end_to_end/e2e_v$V.jsonl ${EXTRA}
echo "E2E V=$V DONE"
