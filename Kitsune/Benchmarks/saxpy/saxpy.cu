// Simple saxpy benchmark

#include <cuda_runtime.h>
#include <iostream>

#include "fpcmp.h"
#include "timing.h"

#include "saxpy.inc"

__global__ void saxpy(ElementType a, const ElementType *x, const ElementType *y,
                      ElementType *r, size_t n) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
    r[i] = a * x[i] + y[i];
}

int main(int argc, char *argv[]) {
  size_t n;
  unsigned iterations;
  unsigned threadsPerBlock;
  ElementType *x = nullptr;
  ElementType *y = nullptr;
  ElementType *r = nullptr;

  TimerGroup tg("saxpy");
  Timer &total = tg.add("total", "Total");

  parseCommandLineInto(argc, argv, n, iterations, &threadsPerBlock);
  header("cuda", x, y, r, n);

  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  for (unsigned t = 0; t < iterations; ++t) {
    total.start();
    saxpy<<<blocksPerGrid, threadsPerBlock>>>(A, x, y, r, n);
    cudaDeviceSynchronize();
    uint64_t us = total.stop();
    std::cout << "\t" << t << ". iteration time: " << Timer::secs(us) << "\n";
  }

  size_t errors = footer(tg, x, y, r, n);
  return errors;
}
