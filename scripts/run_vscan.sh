#!/bin/bash
# Spatha SpMM vs dense fp16 GEMM on synthetic conv-GEMM shapes, for one V build.  usage: run_vscan.sh <gpu> <V>
P=$(cd "$(dirname "$0")/.." && pwd)
source $P/scripts/venom_env.sh
cd $P/vnm
G=$1; V=$2
if [ "$V" != 128 ]; then export SPATHA_LIB=$VENOM_BUILD/spatha_v${V}_lib; fi
CUDA_VISIBLE_DEVICES=$G python spatha_vscan.py --v $V
