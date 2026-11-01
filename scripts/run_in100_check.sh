#!/bin/bash
# Re-check Spatha vs cuDNN agreement on 32 ImageNet-100 test images for every model / M at one V.
P=$(cd "$(dirname "$0")/.." && pwd)
source $P/scripts/venom_env.sh
cd $P/vnm
G=$1; V=$2
if [ "$V" != 128 ]; then export SPATHA_LIB=$VENOM_BUILD/spatha_v${V}_lib; fi
for mdl in bp207 bp182 bp230; do for m in 4 8 16 32; do
  CUDA_VISIBLE_DEVICES=$G python e2e_check.py $mdl $V $m 2>&1 | grep "agree"
done; done
echo "CHECK V=$V DONE"
