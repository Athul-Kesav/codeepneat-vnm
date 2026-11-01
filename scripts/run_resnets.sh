#!/bin/bash
# CIFAR-100 ResNet baselines trained from scratch with the same recipe v2 (one per GPU).
P=$(cd "$(dirname "$0")/.." && pwd)
source $P/scripts/env.sh
cd $P/vnm; mkdir -p $P/logs
i=0
for n in r20w58 r56w32 r18; do
  CUDA_VISIBLE_DEVICES=$i python train_resnet.py --name $n > $P/logs/resnet_$n.log 2>&1 &
  i=$((i+1))
done
wait
