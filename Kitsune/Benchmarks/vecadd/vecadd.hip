// Simple vector addition benchmark

#include <hip/hip_runtime.h>
#include <iostream>
#include <timing.h>

using ElementType = float;

#define IS_HIP
#include "vecadd.inc"

__global__ void vecadd(const ElementType *a, const ElementType *b,
                       ElementType *c, size_t n) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] + b[i];
}

int main(int argc, char *argv[]) {
  size_t n;
  unsigned iterations;
  unsigned threadsPerBlock;
  ElementType *a = nullptr;
  ElementType *b = nullptr;
  ElementType *c = nullptr;

  TimerGroup tg("vecadd");
  Timer &timer = tg.add("vecadd");

  parseCommandLineInto(argc, argv, n, iterations, &threadsPerBlock);
  header("hip", a, b, c, n);

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  for (unsigned t = 0; t < iterations; ++t) {
    timer.start();
    hipLaunchKernelGGL(vecadd, blocksPerGrid, threadsPerBlock, 0, 0, a, b, c,
                       n);
    hipDeviceSynchronize();
    uint64_t us = timer.stop();
    std::cout << "\t" << t << ". iteration time: " << Timer::secs(us) << "\n";
  }

  size_t errors = footer(tg, a, b, c, n);
  return errors;
}
