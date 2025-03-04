// Simple saxpy benchmark

#include <hip/hip_runtime.h>
#include <iostream>

#include "timing.h"

using ElementType = float;

#define IS_HIP
#include "saxpy.inc"

const ElementType DEFAULT_X_VALUE = rand() % 1000000;
const ElementType DEFAULT_Y_VALUE = rand() % 1000000;
const ElementType DEFAULT_A_VALUE = rand() % 1000000;

__global__ void doInit(ElementType xc, ElementType yc, ElementType *x,
                       ElementType *y, size_t n) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    x[i] = xc;
    y[i] = yc;
  }
}

__global__ void doSaxpy(ElementType a, const ElementType *x,
                        const ElementType *y, ElementType *r, size_t n) {
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
  ElementType *z = nullptr;

  TimerGroup tg("saxpy");
  Timer &init = tg.add("init");
  Timer &saxpy = tg.add("saxpy");

  parseCommandLineInto(argc, argv, n, iterations, &threadsPerBlock);
  header("hip", DEFAULT_A_VALUE, x, y, r, n);

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  for (unsigned t = 0; t < iterations; ++t) {
    init.start();
    hipLaunchKernelGGL(doInit, blocksPerGrid, threadsPerBlock, 0, 0,
                       DEFAULT_X_VALUE, DEFAULT_Y_VALUE, x, y, n);
    hipDeviceSynchronize();
    uint64_t usInit = init.stop();

    saxpy.start();
    hipLaunchKernelGGL(doSaxpy, blocksPerGrid, threadsPerBlock, 0, 0,
                       DEFAULT_A_VALUE, x, y, r, n);
    hipDeviceSynchronize();
    uint64_t us = saxpy.stop();
    std::cout << "\t" << t
              << ". iteration time: " << Timer::secs(usInit + usSaxpy) << "\n";
  }

  size_t errors = footer(tg, DEFAULT_A_VALUE, x, y, r, n);
  return errors;
}
