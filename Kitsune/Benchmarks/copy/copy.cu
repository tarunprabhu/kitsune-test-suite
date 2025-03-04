// Straightforward memory copy

#include <cuda_runtime.h>
#include <iostream>

#include "timing.h"

using ElementType = float;

#define IS_CUDA
#include "copy.inc"

__global__ void copy(ElementType *dst, const ElementType *src, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = src[i];
}

int main(int argc, char *argv[]) {
  size_t n = 0;
  unsigned iterations = 0;
  ElementType *dst = nullptr;
  ElementType *src = nullptr;
  TimerGroup tg("copy");
  Timer &timer = tg.add("copy");

  // This is loosely for consistency with the launch parameters from kitsune.
  unsigned threadsPerBlock = 256;

  parseCommandLineInto(argc, argv, n, iterations, &threadsPerBlock);
  header("cuda", dst, src, n);

  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  for (unsigned t = 0; t < iterations; ++t) {
    timer.start();
    copy<<<blocksPerGrid, threadsPerBlock>>>(dst, src, n);
    cudaDeviceSynchronize();
    uint64_t us = timer.stop();
    std::cout << "\t" << t << ". iteration time: " << Timer::secs(us) << "\n";
  }

  size_t errors = footer(tg, dst, src, n);
  return errors;
}
