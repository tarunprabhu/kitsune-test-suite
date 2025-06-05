// Simple vector scale benchmark

#include <cuda_runtime.h>
#include <iostream>

#include "fpcmp.h"
#include "timing.h"

#include "vecscale.inc"

__global__ void vecscale(const ElementType *a, const ElementType *b, size_t n) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
    b[i] = a[i] * 65;
}

int main(int argc, char *argv[]) {
  size_t n;
  unsigned iterations;
  ElementType *a = nullptr;
  ElementType *b = nullptr;
  unsigned threadsPerBlock;

  TimerGroup tg("vecscale");
  Timer &total = tg.add("total", "Total");

  parseCommandLineInto(argc, argv, n, iterations, &threadsPerBlock);
  header("cuda", a, b, n);

  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  for (unsigned t = 0; t < iterations; ++t) {
    total.start();
    vecscale<<<blocksPerGrid, threadsPerBlock>>>(a, b, n);
    cudaDeviceSynchronize();
    uint64_t us = total.stop();
    std::cout << "\t" << t << ". iteration time: " << Timer::secs(us) << "\n";
  }

  size_t errors = footer(tg, a, b, n);
  return errors;
}
