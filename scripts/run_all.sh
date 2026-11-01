#!/bin/bash
# The whole project, in order. Each stage only needs the stages before it.
# usage: scripts/run_all.sh [setup|search|train|prune|speed|figures|all]...   (default: all)
# Assumes 3 GPUs (0,1,2). Rough times on 3x RTX 6000 Ada: search ~1-2 days (ada_cifar10_strong ~2 days alone),
# train ~1 day, prune ~1 day, speed ~3 h.
set -e
P=$(cd "$(dirname "$0")/.." && pwd); S=$P/scripts
stages=${*:-all}
want() { [[ " $stages " == *" $1 "* || " $stages " == *" all "* ]]; }

if want setup; then
  $S/setup_main_env.sh
  $S/setup_venom.sh
  $S/setup_data.sh all
fi
if want search; then   # CoDeepNEAT evolution + its own full train of the top-3
  for c in venom_cifar10 venom_cifar100 imagenet100 venom_cifar100_res ada_cifar10_strong; do $S/run_search.sh $c; done
fi
if want train; then    # dense retraining with our recipes + ResNet references
  for g in 0 1 2; do $S/run_retrain.sh $g & done; wait   # recipe v1: CIFAR-100 (200 ep), ImageNet-100 (120 ep)
  $S/run_retrain_v2.sh orig
  $S/run_retrain_v2.sh res
  $S/run_resnets.sh
fi
if want prune; then    # V:N:M grids, criteria, CIFAR-10 strong winner
  $S/run_grid.sh venom_cifar10 0 8 & $S/run_grid.sh venom_cifar100 1 8 & $S/run_grid.sh imagenet100 2 3 & wait
  $S/run_grid2.sh
  $S/run_grid3.sh
  $S/run_gridres.sh
  $S/run_grid_resnet.sh
  $S/run_obs.sh
  $S/run_gradual.sh
  $S/run_cifar10_ada.sh
fi
if want speed; then    # Spatha: GEMM V-scan, end-to-end inference, ImageNet-100 correctness at 32 images
  for V in 32 64 128 256; do $S/run_vscan.sh 0 $V; done
  for g_v in "0 32" "1 64" "2 128"; do
    set -- $g_v
    (EXTRA="--models bp493,bp482,bp741,bp430,bp427,bp499,r18,r20w58,r56w32,bp1485" $S/run_e2e.sh $1 $2
     EXTRA="--models bp207,bp182,bp230 --batches 32,1" $S/run_e2e.sh $1 $2
     $S/run_in100_check.sh $1 $2) &
  done
  wait
fi
if want figures; then  # CoDeepNEAT architecture graphs, then the report
  source $S/env.sh
  (cd $P/vnm && CUDA_VISIBLE_DEVICES=0 python -W ignore plot_architectures.py)
  (cd $P/report && pdflatex -interaction=nonstopmode report.tex >/dev/null && pdflatex -interaction=nonstopmode report.tex >/dev/null)
fi
