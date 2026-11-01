# Environment for the VENOM/Spatha scripts (torch 1.13 + CUDA 11.7 env in $VENOM_BUILD/env). Sourced.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export PATH=$VENOM_BUILD/env/bin:$PATH
export CUDA_HOME=$VENOM_BUILD/env
unset PYTORCH_CUDA_ALLOC_CONF
