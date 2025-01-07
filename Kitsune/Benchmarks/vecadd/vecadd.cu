// Straightforward vector addition

#include <iostream>
#include <timing.h>

#include <cuda_runtime.h>

void random_fill(float *data, size_t N) {
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

__global__ void VectorAdd(float *A, float *B, float *C, size_t N) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}

int main(int argc, char *argv[]) {
  size_t size = 1024 * 1024 * 256;
  if (argc > 1)
    size = atol(argv[1]);
  Timer timer("vecadd");

  fprintf(stdout, "problem size: %ld\n", size);

  // This is loosely for consistency with the launch parameters
  // from kitsune.
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  cudaEvent_t start;
  cudaEventCreate(&start);
  cudaEventRecord(start);
  cudaError_t err = cudaSuccess;
  float *a, *b, *c;
  err = cudaMallocManaged(&a, size * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory for A!\n");
    return 1;
  }
  err = cudaMallocManaged(&b, size * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory for B!\n");
    return 1;
  }
  err = cudaMallocManaged(&c, size * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory for C!\n");
    return 1;
  }

  random_fill(a, size);
  random_fill(b, size);
  for (size_t i = 0; i < 10; ++i) {
    cudaEvent_t kstart, kstop;

    timer.start();
    cudaEventCreate(&kstart);
    cudaEventCreate(&kstop);
    cudaEventRecord(kstart);
    VectorAdd<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, size);
    cudaEventRecord(kstop);
    cudaEventSynchronize(kstop);
    timer.stop();

    float msecs = 0;
    cudaEventElapsedTime(&msecs, kstart, kstop);
    printf("kernel time: %7lg ms\n", msecs);
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
