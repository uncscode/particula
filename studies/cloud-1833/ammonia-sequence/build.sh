#!/usr/bin/env bash
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$1"
nvcc -O3 "-arch=${CLOUD_CUDA_ARCH:-sm_120}" --shared -Xcompiler -fPIC "$here/growth.cu" -o "$1/growth.so"
g++ -O3 -std=c++17 -fopenmp -shared -fPIC "$here/../charge-evaporation/population.cpp" -o "$1/particles.so"
