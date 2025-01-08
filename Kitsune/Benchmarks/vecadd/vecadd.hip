// Simple vector addition benchmark

#include <iostream>
#include <timing.h>

#include "hip/hip_runtime.h"

#define HIPCHECK(error)                                                        \
  if (error != hipSuccess) {                                                   \
    printf("error: '%s' (%d) at %s:%d\n", hipGetErrorString(error), error,     \
           __FILE__, __LINE__);                                                \
    exit(1);                                                                   \
  }

template <typename T>
static void random_fill(T *data, size_t n) {
  for (size_t i = 0; i < n; ++i)
    data[i] = rand() / (T)RAND_MAX;
}

template <typename T>
static size_t check(const T *a, const T *b, const T *c, size_t n) {
  size_t errors = 0;
  for (size_t i = 0; i < n; i++) {
    float sum = a[i] + b[i];
    if (c[i] != sum)
      errors++;
  }
  return errors;
}

__global__ static void vecadd(float *a, float *b, float *c, size_t n) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] + b[i];
}

int main(int argc, char *argv[]) {
  size_t size = 1024 * 1024 * 256;
  unsigned int iterations = 10;
  if (argc > 1)
    size = atol(argv[1]);
  if (argc > 2)
    iterations = atoi(argv[2]);
  Timer timer("vecadd");

  std::cout << "\n";
  std::cout << "---- vector addition benchmark (hip) ----\n"
            << "  Vector size: " << size << " elements.\n\n";
  std::cout << "  Allocating arrays and filling with random values..."
            << std::flush;

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);

  float *a, *b, *c;
  HIPCHECK(hipMallocManaged(&a, size * sizeof(float)));
  HIPCHECK(hipMallocManaged(&b, size * sizeof(float)));
  HIPCHECK(hipMallocManaged(&c, size * sizeof(float)));

  random_fill(a, size);
  random_fill(b, size);

  // This is loosely for consistency with the launch parameters from kitsune.
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  for (size_t t = 0; t < iterations; ++t) {
    timer.start();
    hipLaunchKernelGGL(vecadd, blocksPerGrid, threadsPerBlock, 0, 0, a, b, c,
                       size);
    HIPCHECK(hipDeviceSynchronize());
    uint64_t us = timer.stop();
    std::cout << "\t" << t << ". iteration time: " << us << " us\n";
  }

  std::cout << "\n  Checking final result..." << std::flush;
  size_t errors = check(a, b, c, size);
  if (errors)
    std::cout << "  FAIL! (" << errors << " errors found)\n\n";
  else
    std::cout << "  pass\n\n";

  json(std::cout, {timer});

  return errors;
}
