#!/bin/bash
# Main environment (search, training, pruning): Python 3.13 + PyTorch 2.6 (CUDA 12.4 wheels) in external/env_main.
# Needs conda on PATH. Afterwards scripts/env.sh puts external/env_main/bin first on PATH automatically.
set -e
source "$(dirname "$0")/env.sh"
E=$PROJECT_ROOT/external/env_main
[ -x $E/bin/python ] || conda create -y -p $E python=3.13
$E/bin/pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
$E/bin/pip install -r $PROJECT_ROOT/requirements.txt
conda install -y -p $E -c conda-forge graphviz   # `dot`, for vnm/plot_architectures.py
$E/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
