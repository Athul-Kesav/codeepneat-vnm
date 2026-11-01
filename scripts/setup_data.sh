#!/bin/bash
# Datasets under CoDeepNEAT/data/Datasets/.
#   CIFAR-10 / CIFAR-100: downloaded by torchvision into CoDeepNEAT/data/Datasets/ (not shipped in the repo).
#   ImageNet-100 (clane9/imagenet-100 on Hugging Face, CMC 100-class split, about 8 GB):
#     download -> export to ImageFolder (CoDeepNEAT search) -> 144 px uint8 cache in $VENOM_BUILD/cache/in100 (training).
# usage: setup_data.sh [cifar|imagenet|all]   (default: all)
set -e
source "$(dirname "$0")/env.sh"
D=$PROJECT_ROOT/CoDeepNEAT/data/Datasets; mkdir -p $D
what=${1:-all}
if [ $what = cifar ] || [ $what = all ]; then
  python -c "
from torchvision.datasets import CIFAR10, CIFAR100
for ds in (CIFAR10, CIFAR100):
    for train in (True, False): ds('$D', train=train, download=True)
print('cifar ok')"
fi
if [ $what = imagenet ] || [ $what = all ]; then
  pip install -q huggingface_hub datasets
  [ -d $D/imagenet100_hf/data ] || huggingface-cli download clane9/imagenet-100 --repo-type dataset --local-dir $D/imagenet100_hf
  [ -d $D/imagenet100/test ] || python $PROJECT_ROOT/scripts/export_imagenet100.py
  [ -f $VENOM_BUILD/cache/in100/test_y.npy ] || python $PROJECT_ROOT/vnm/build_cache.py
fi
