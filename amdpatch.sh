#!/bin/sh

ROCM_PATH=$(hipconfig --rocmpath)

sudo patch $ROCM_PATH/lib/llvm/lib/clang/18/include/__clang_hip_cmath.h ./patches/amd.patch