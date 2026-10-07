#!/usr/bin/env bash
# build_raptor.sh -- build bubble_raptor: the Deforming Bubble mini-app with
# src/advect.cpp recompiled under RAPTOR (-DBUBBLE_RAPTOR) so its WENO kernel
# can be run under emulated floating-point precision at runtime via
# BUBBLE_FP_MODE=tf32|profile (see src/advect.cpp). Everything else is built
# and linked exactly as the normal `make` build.
set -euo pipefail
cd "$(dirname "$0")/.."

RAPTOR_INSTALL_DIR="${RAPTOR_INSTALL_DIR:-$HOME/raptor-install}"
LLVM_VER="${LLVM_VER:-20}"
RAPTOR_CLANGXX="${RAPTOR_CLANGXX:-clang++-$LLVM_VER}"
PASS_PLUGIN="$RAPTOR_INSTALL_DIR/lib/LLVMRaptor-$LLVM_VER.so"

if [ ! -f "$PASS_PLUGIN" ]; then
    echo "build_raptor.sh: missing $PASS_PLUGIN (set RAPTOR_INSTALL_DIR / LLVM_VER)" >&2
    exit 1
fi

make

mkdir -p build_raptor
"$RAPTOR_CLANGXX" -std=c++17 -O2 -DBUBBLE_RAPTOR=1 -Isrc \
    -I"$RAPTOR_INSTALL_DIR/include" $(mpicxx -showme:compile) \
    -fpass-plugin="$PASS_PLUGIN" \
    -c src/advect.cpp -o build_raptor/advect.o

LIBOBJS=$(ls build/*.o | grep -v '/advect\.o$')

mpicxx -std=c++17 -O2 build_raptor/advect.o $LIBOBJS -o bubble_raptor \
    -L"$RAPTOR_INSTALL_DIR/lib" -Wl,-rpath,"$RAPTOR_INSTALL_DIR/lib" \
    -lRaptor-RT-"$LLVM_VER" -lmpfr -lstdc++

echo "build_raptor.sh: wrote ./bubble_raptor"
