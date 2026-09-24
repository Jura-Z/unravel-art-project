#!/bin/sh
# Builds every C/WASM variant. Flags that matter for bit-exactness: -ffp-contract=off, no -ffast-math.
set -e
cd "$(dirname "$0")"
mkdir -p ../../build
COMMON="-O3 -ffp-contract=off -Wall -Wno-unused-function"
WASM="--target=wasm32 -nostdlib -Wl,--no-entry -Wl,--import-memory -Wl,--export=__heap_base -Wl,--export-dynamic"
clang $COMMON $WASM -o ../../build/scalar.wasm kernel.c
clang $COMMON $WASM -msimd128 -o ../../build/simd.wasm kernel.c
clang $COMMON $WASM -msimd128 -matomics -mbulk-memory -Wl,--shared-memory -Wl,--max-memory=2147483648 \
  -Wl,--export=__stack_pointer -o ../../build/simd-mt.wasm kernel.c
clang $COMMON -march=native -o ../../build/native kernel.c main.c -lm
ls -la ../../build
