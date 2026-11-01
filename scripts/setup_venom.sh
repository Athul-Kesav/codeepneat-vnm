#!/bin/bash
# VENOM / Spatha environment in $VENOM_BUILD (default external/venom_build):
#   env/               conda env: Python 3.10, PyTorch 1.13.1, CUDA toolkit 11.7.1, cmake
#   venom/             UDC-GAC/venom at commit 8ddaf38
#   default build      V=128, installed into env/ (import spatha)
#   spatha_v{32,64,256}_lib   extra builds, selected at run time with SPATHA_LIB=<dir>
# Built for sm_86 (runs on Ampere and Ada). Needs conda, git and an NVIDIA driver for CUDA 11.7+.
set -e
source "$(dirname "$0")/env.sh"
P=$PROJECT_ROOT; B=$VENOM_BUILD
[ -x $B/env/bin/python ] || conda create -y -p $B/env -c pytorch -c nvidia/label/cuda-11.7.1 -c conda-forge \
    python=3.10 pytorch=1.13.1 pytorch-cuda=11.7 cuda-toolkit=11.7.1 cmake
source $P/scripts/venom_env.sh
pip install pybind11 numpy==1.26.4
if [ ! -d $B/venom ]; then
  git clone --recurse-submodules https://github.com/UDC-GAC/venom $B/venom
  git -C $B/venom checkout 8ddaf38
fi
cd $B/venom/end2end
python -W ignore setup.py install                                   # V=128 (default tile)
python -W ignore setup_v64.py build_ext --build-lib $B/spatha_v64_lib # V=64 (upstream option)
for V in 32 256; do                                                   # V=32 / 256: our V_CUSTOM patch
  rm -rf $B/v${V}_src && cp -r $B/venom/end2end $B/v${V}_src && cd $B/v${V}_src
  rm -rf build dist spatha.egg-info
  patch -p1 < $P/scripts/patches/spatha_v_custom.patch
  sed "s/'-DV_64'/'-DV_CUSTOM=$V'/" setup_v64.py > setup_v$V.py
  python -W ignore setup_v$V.py build_ext --build-lib $B/spatha_v${V}_lib
done
python -c "import torch, spatha; print('spatha ok')"
