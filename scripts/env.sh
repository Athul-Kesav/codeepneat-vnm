# Common environment for every script. Sourced, never run.
# PROJECT_ROOT = this folder. Machine-local, large things live under external/ (created by the setup scripts):
#   external/env_main     main Python env (search, training, pruning)
#   external/venom_build  VENOM/Spatha env, Spatha builds, ImageNet-100 cache, caches/temp files
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VENOM_BUILD="${VENOM_BUILD:-$PROJECT_ROOT/external/venom_build}"
if [ -x "$PROJECT_ROOT/external/env_main/bin/python" ]; then export PATH="$PROJECT_ROOT/external/env_main/bin:$PATH"; fi
mkdir -p "$VENOM_BUILD/tmp" "$VENOM_BUILD/cache" "$PROJECT_ROOT/logs"
export TMPDIR=$VENOM_BUILD/tmp
export XDG_CACHE_HOME=$VENOM_BUILD/cache
export TORCH_HOME=$VENOM_BUILD/cache/torch
export HF_HOME=$VENOM_BUILD/cache/huggingface
export PIP_CACHE_DIR=$VENOM_BUILD/cache/pip
export CONDA_PKGS_DIRS=$VENOM_BUILD/cache/conda_pkgs
export WANDB_MODE=disabled WANDB_DIR=$VENOM_BUILD/tmp
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
