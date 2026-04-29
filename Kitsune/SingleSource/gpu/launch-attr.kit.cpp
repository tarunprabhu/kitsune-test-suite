// Check that adding explicit launch parameters does not affect the final
// result. Only relevant for GPU kernels.

#include <iostream>
#include <stdint.h>

#include <kitsune.h>

using namespace kitsune;

template <typename T> static void random_fill(mobile_ptr<T> arr, size_t n) {
  T base_value = rand() / (T)RAND_MAX;
  // clang-format off
  forall(size_t i = 0; i < n; ++i) {
    arr[i] = base_value + i;
  }
  // clang-format on
}

template <typename T>
static size_t check(const mobile_ptr<T> a, const mobile_ptr<T> b,
                    const mobile_ptr<T> c, size_t n) {
  uint64_t errors = 0;
  for (size_t i = 0; i < n; i++) {
    float sum = a[i] + b[i];
    if (c[i] != sum)
      errors++;
  }
  return errors;
}

int main(int argc, char *argv[]) {
  size_t size = 1024 * 1024;
  unsigned int iterations = 10;
  if (argc > 1)
    size = atol(argv[1]);

  mobile_ptr<float> a(size);
  mobile_ptr<float> b(size);
  mobile_ptr<float> c(size);
  random_fill(a, size);
  random_fill(b, size);

  // clang-format off
  [[kitsune::launch(64)]]
  forall(int i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
  // clang-format on

  std::cout << "\n  Checking final result..." << std::flush;
  size_t errors = check(a, b, c, size);
  if (errors)
    std::cout << "  FAIL! (" << errors << " errors found)\n\n";
  else
    std::cout << "  pass\n\n";

  a.free();
  b.free();
  c.free();

  return errors;
}
