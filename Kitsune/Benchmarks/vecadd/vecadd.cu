// Straightforward vector addition

#include <iostream>
#include <timing.h>

#include <cuda_runtime.h>

template <typename T>
static void random_fill(T *data, size_t N) {
  for (size_t i = 0; i < N; ++i)
    data[i] = rand() / (T)RAND_MAX;
}

template <typename T>
static size_t check(const T *a, const T *b, const T *c, size_t n) {
  size_t errors = 0;
  for (size_t i = 0; i < n; i++)
    if (c[i] != a[i] + b[i])
      errors++;
  return errors;
}

__global__ static void vecadd(const float *a, const float *b, float *c,
                              size_t n) {
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
  std::cout << "---- vector addition benchmark (cuda) ----\n"
            << "  Vector size: " << size << " elements.\n\n";
  std::cout << "  Allocating arrays and filling with random values..."
            << std::flush;

  float *a, *b, *c;
  cudaMallocManaged(&a, size * sizeof(float));
  cudaMallocManaged(&b, size * sizeof(float));
  cudaMallocManaged(&c, size * sizeof(float));

  random_fill(a, size);
  random_fill(b, size);

  // This is loosely for consistency with the launch parameters from kitsune.
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  for (size_t t = 0; t < iterations; ++t) {
    timer.start();
    vecadd<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, size);
    cudaDeviceSynchronize();
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
