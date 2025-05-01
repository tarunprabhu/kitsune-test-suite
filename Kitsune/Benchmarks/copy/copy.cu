// Straightforward memory copy

#include <cuda_runtime.h>
#include <iostream>

#include "timing.h"

#include "copy.inc"

__global__ void copy(ElementType *dst, const ElementType *src, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = src[i];
}

int main(int argc, char *argv[]) {
  size_t n;
  unsigned iterations;
  ElementType *dst = nullptr;
  ElementType *src = nullptr;
  unsigned threadsPerBlock;

  TimerGroup tg("copy");
  Timer &total = tg.add("total", "Total");

  parseCommandLineInto(argc, argv, n, iterations, &threadsPerBlock);
  header("cuda", dst, src, n);

  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  for (unsigned t = 0; t < iterations; ++t) {
    total.start();
    copy<<<blocksPerGrid, threadsPerBlock>>>(dst, src, n);
    cudaDeviceSynchronize();
    uint64_t us = total.stop();
    std::cout << "\t" << t << ". iteration time: " << Timer::secs(us) << "\n";
  }

  size_t errors = footer(tg, dst, src, n);
  return errors;
}
