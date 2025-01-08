// Straightforward memory copy

#include <iostream>
#include <timing.h>

#include "hip/hip_runtime.h"

#define HIPCHECK(error)                                                        \
  if (error != hipSuccess) {                                                   \
    printf("error: '%s' (%d) at %s:%d\n", hipGetErrorString(error), error,     \
           __FILE__, __LINE__);                                                \
    exit(1);                                                                   \
  }

static void random_fill(float *data, size_t n) {
  for (size_t i = 0; i < n; ++i)
    data[i] = rand() / (float)RAND_MAX;
}

__global__ static void copy(float *dst, const float *src, size_t N) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    dst[i] = src[i];
}

static size_t check(const float *dst, const float *src, size_t n) {
  size_t errors;
  for (size_t i = 0; i < n; ++i) {
    if (src[i] != dst[i])
      errors += 1;
  }
  return errors;
}

int main(int argc, char *argv[]) {
  size_t size = 1024 * 1024 * 256;
  unsigned int iterations = 10;
  if (argc > 1)
    size = atol(argv[1]);
  if (argc > 2)
    iterations = atoi(argv[2]);
  Timer timer("copy");

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);

  std::cout << "\n";
  std::cout << "---- Simple copy benchmark (hip) ----\n"
            << "  Array size: " << size << "\n"
            << "  Iterations: " << iterations << "\n\n";
  std::cout << "Allocating arrays and filling with random values..."
            << std::flush;

  float *dst, *src;
  HIPCHECK(hipMallocManaged(&dst, size * sizeof(float)));
  HIPCHECK(hipMallocManaged(&src, size * sizeof(float)));
  random_fill(dst, size);
  random_fill(src, size);
  std::cout << std::endl;

  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  for (int t = 0; t < iterations; t++) {
    timer.start();
    hipLaunchKernelGGL(copy, blocksPerGrid, threadsPerBlock, 0, 0, dst, src,
                       size);
    HIPCHECK(hipDeviceSynchronize());
    uint64_t us = timer.stop();
    std::cout << "\t" << t << ". iteration time: " << us << " us\n";
  }

  std::cout << "\n  Checking final result..." << std::flush;
  size_t errors = check(dst, src, size);
  if (errors)
    std::cout << "  FAIL! (" << errors << " errors found)\n\n";
  else
    std::cout << "  pass\n\n";

  json(std::cout, {timer});

  return errors;
}
