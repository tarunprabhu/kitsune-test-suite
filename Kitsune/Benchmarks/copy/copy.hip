// Straightforward memory copy

#include <hip/hip_runtime.h>
#include <iostream>

#include "timing.h"

using ElementType = float;

#define IS_HIP
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

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);

  parseCommandLineInto(argc, argv, n, iterations, &threadsPerBlock);
  header("hip", dst, src, n);

  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  for (unsigned t = 0; t < iterations; ++t) {
    timer.start();
    hipLaunchKernelGGL(copy, blocksPerGrid, threadsPerBlock, 0, 0, dst, src, n);
    hipDeviceSynchronize();
    uint64_t us = timer.stop();
    std::cout << "\t" << t << ". iteration time: " << Timer::secs(us) << "\n";
  }

  size_t errors = footer(tg, dst, src, n);
  return errors;
}
