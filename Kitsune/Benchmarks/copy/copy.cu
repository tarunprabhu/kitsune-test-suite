// Straightforward memory copy

#include <iostream>
#include <timing.h>

#include <cuda_runtime.h>

static void random_fill(float *data, size_t n) {
  float base = rand() / (float)RAND_MAX;
  for (size_t i = 0; i < n; ++i)
    data[i] = base + i;
}

static size_t check(const float *dst, const float *src, size_t n) {
  const float epsilon = 1e-15;
  size_t errors = 0;
  for (size_t i = 0; i < n; i++)
    if (fabs(dst[i] - src[i]) > epsilon)
      errors++;
  return errors;
}

__global__ void copy(float* dst, const float* src, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = src[i];
}

int main(int argc, char *argv[]) {
  size_t size = 1024 * 1024 * 256;
  unsigned int iterations = 10;
  if (argc > 1)
    size = atol(argv[1]);
  if (argc > 2)
    iterations = atoi(argv[2]);
  Timer timer("copy");

  std::cout << "\n";
  std::cout << "---- Simple copy benchmark (cuda) ----\n"
            << "  Array size: " << size << "\n"
            << "  Iterations: " << iterations << "\n\n";
  std::cout << "Allocating arrays and filling with random values ... "
            << std::flush;

  float *dst = nullptr, *src = nullptr;
  cudaMallocManaged(&dst, size * sizeof(float));
  cudaMallocManaged(&src, size * sizeof(float));

  // random_fill(dst, size);
  random_fill(src, size);
  std::cout << "done" << std::endl;

  // This is loosely for consistency with the launch parameters from kitsune.
  int threadsPerBlock = 32;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  for (size_t t = 0; t < iterations; ++t) {
    timer.start();
    copy<<<blocksPerGrid, threadsPerBlock>>>(dst, src, size);
    cudaDeviceSynchronize();
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
