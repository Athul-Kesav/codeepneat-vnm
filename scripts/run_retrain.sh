#!/bin/bash
# usage: run_retrain.sh <gpu>   (3 shards: gpu i trains candidates i, i+3, ...)
P=$(cd "$(dirname "$0")/.." && pwd)
G=$1
source $P/scripts/env.sh
cd $P/vnm
echo "=== RETRAIN cifar100 START $(date -Is)" >> $P/logs/retrain_gpu$G.log
CUDA_VISIBLE_DEVICES=$G python train_candidates.py --run venom_cifar100 --dataset cifar100 --n 10 --epochs 200 --shard $G/3 >> $P/logs/retrain_gpu$G.log 2>&1
echo "=== RETRAIN imagenet100 START $(date -Is)" >> $P/logs/retrain_gpu$G.log
CUDA_VISIBLE_DEVICES=$G python train_candidates.py --run imagenet100 --dataset imagenet100 --n 10 --epochs 120 --shard $G/3 >> $P/logs/retrain_gpu$G.log 2>&1
echo "=== RETRAIN DONE $(date -Is)" >> $P/logs/retrain_gpu$G.log
