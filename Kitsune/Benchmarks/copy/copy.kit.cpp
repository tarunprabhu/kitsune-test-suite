// Straightforward memory copy

#include <iostream>
#include <kitsune.h>
#include <timing.h>

using namespace kitsune;

template <typename T> static void random_fill(mobile_ptr<T> data, size_t n) {
  // clang-format off
  for (size_t i = 0; i < n; ++i) {
    data[i] = rand() / (T)RAND_MAX;
  }
  // clang-format on
}

template <typename T>
static size_t check(const mobile_ptr<T> src, const mobile_ptr<T> dst,
                    size_t n) {
  size_t errors;
  for (size_t i = 0; i < n; ++i) {
    if (src[i] != dst[i])
      errors += 1;
  }
  return errors;
}

int main(int argc, char **argv) {
  size_t size = 1024 * 1024 * 256;
  unsigned int iterations = 10;
  if (argc > 1)
    size = atol(argv[1]);
  if (argc > 2)
    iterations = atoi(argv[2]);
  Timer timer("copy");

  std::cout << "\n";
  std::cout << "---- Simple copy benchmark (forall) ----\n"
            << "  Array size: " << size << "\n"
            << "  Iterations: " << iterations << "\n\n";
  std::cout << "Allocating arrays and filling with random values..."
            << std::flush;
  mobile_ptr<float> src(size);
  mobile_ptr<float> dst(size);
  random_fill(src, size);
  std::cout << "  done.\n\n";

  std::cout << "Starting benchmark...\n";
  unsigned int mb_size = (sizeof(float) * size) / (1024 * 1024);

  for (int i = 0; i < iterations; i++) {
    timer.start();
    // clang-format off
    forall(int i = 0; i < size; i++) {
      dst[i] = src[i];
    }
    uint64_t ms = timer.stop();
    std::cout << "\t" << i << ". copy time: " << ms << " ms\n";
  }

  std::cout << "\n  Checking final result..." << std::flush;
  size_t errors = check(src, dst, size);
  if (errors)
    std::cout << "  FAIL! (" << errors << " errors found)\n\n";
  else
    std::cout << "  pass\n\n";

  json(std::cout, {timer});

  src.free();
  dst.free();

  return errors;
}
