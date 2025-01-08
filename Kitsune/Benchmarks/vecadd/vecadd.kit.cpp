// Straightforward vector addition

#include <iostream>
#include <kitsune.h>
#include <timing.h>

using namespace kitsune;

template <typename T> static void random_fill(mobile_ptr<T> arr, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    arr[i] = rand() / (T)RAND_MAX;
  }
}

template <typename T>
static size_t check(const mobile_ptr<T> a, const mobile_ptr<T> b,
                    const mobile_ptr<T> c, size_t n) {
  size_t errors = 0;
  for (size_t i = 0; i < n; i++) {
    float sum = a[i] + b[i];
    if (c[i] != sum)
      errors++;
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
  Timer timer("vecadd");

  std::cout << "\n";
  std::cout << "---- vector addition benchmark (forall) ----\n"
            << "  Vector size: " << size << " elements.\n\n";
  std::cout << "  Allocating arrays and filling with random values..."
            << std::flush;

  mobile_ptr<float> a(size);
  mobile_ptr<float> b(size);
  mobile_ptr<float> c(size);
  random_fill(a, size);
  random_fill(b, size);
  std::cout << "  done.\n\n";

  for (unsigned t = 0; t < iterations; t++) {
    timer.start();
    // clang-format off
    forall(int i = 0; i < size; i++) {
      c[i] = a[i] + b[i];
    }
    // clang-format on
    uint64_t us = timer.stop();
    std::cout << "\t" << t << ". iteration time: " << us << "us\n";
  }

  std::cout << "\n  Checking final result..." << std::flush;
  size_t errors = check(a, b, c, size);
  if (errors)
    std::cout << "  FAIL! (" << errors << " errors found)\n\n";
  else
    std::cout << "  pass\n\n";

  json(std::cout, {timer});

  a.free();
  b.free();
  c.free();

  return errors;
}
