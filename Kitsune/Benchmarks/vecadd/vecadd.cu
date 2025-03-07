// Simple vector addition benchmark

#include <cuda_runtime.h>
#include <iostream>

#include "fpcmp.h"
#include "timing.h"

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
  ElementType *a = nullptr;
  ElementType *b = nullptr;
  ElementType *c = nullptr;
  unsigned threadsPerBlock;

  TimerGroup tg("vecadd");
  Timer &total = tg.add("total", "Total");

  parseCommandLineInto(argc, argv, n, iterations, &threadsPerBlock);
  header("cuda", a, b, c, n);

  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  for (unsigned t = 0; t < iterations; ++t) {
    total.start();
    vecadd<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, n);
    cudaDeviceSynchronize();
    uint64_t us = total.stop();
    std::cout << "\t" << t << ". iteration time: " << Timer::secs(us) << "\n";
  }

  size_t errors = footer(tg, a, b, c, n);
  return errors;
}
