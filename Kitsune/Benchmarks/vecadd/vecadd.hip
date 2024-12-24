#include <iostream>
#include <kitsune.h>
#include <timing.h>

#include "hip/hip_runtime.h"

#define HIPCHECK(error)                                                        \
  if (error != hipSuccess) {                                                   \
    printf("error: '%s' (%d) at %s:%d\n", hipGetErrorString(error), error,     \
           __FILE__, __LINE__);                                                \
    exit(1);                                                                   \
  }

const size_t VEC_SIZE = 1024 * 1024 * 256;

static void random_fill(float *data, size_t N) {
  for (size_t i = 0; i < N; ++i)
    data[i] = rand() / (float)RAND_MAX;
}

template <typename T>
static size_t check(const T *a, const T *b, const T *c, size_t n) {
  uint64_t errors = 0;
  for (size_t i = 0; i < n; i++) {
    float sum = a[i] + b[i];
    if (c[i] != sum)
      errors++;
  }
  return errors;
}

__global__ void VectorAdd(float *a, float *b, float *c, size_t n) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    c[i] = a[i] + b[i];
}

int main(int argc, char *argv[]) {
  size_t size = VEC_SIZE;
  if (argc > 1)
    size = atol(argv[1]);
  Timer timer("vecadd");
  fprintf(stdout, "problem size: %ld\n", size);

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);

  hipError_t err = hipSuccess;
  float *a, *b, *c;
  HIPCHECK(hipMallocManaged(&a, size * sizeof(float)));
  HIPCHECK(hipMallocManaged(&b, size * sizeof(float)));
  HIPCHECK(hipMallocManaged(&c, size * sizeof(float)));

  random_fill(A, size);
  random_fill(B, size);

  // This is loosely for consistency with the launch parameters
  // from kitsune.
  for (size_t i = 0; i < 10; ++i) {
    timer.start();
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    hipMemPrefetchAsync(a, size * sizeof(float), 0, 0);
    hipMemPrefetchAsync(b, size * sizeof(float), 0, 0);
    hipMemPrefetchAsync(c, size * sizeof(float), 0, 0);
    hipLaunchKernelGGL(VectorAdd, blocksPerGrid, threadsPerBlock, 0, 0, a, b, c,
                       size);
    HIPCHECK(hipDeviceSynchronize());
    uint64_t ms = timer.stop();

    fprintf(stdout, "kernel: %7ld ms\n", ms);
  }

  std::cout << "\n  Checking final result..." << std::flush;
  size_t errors = check(a, b, c, size);
  if (errors)
    std::cout << "  FAIL! (" << errors << " errors found)\n\n";
  else
    std::cout << "  pass\n\n";

  json(std::cout, "vecadd", {timer});

  return errors;
}
