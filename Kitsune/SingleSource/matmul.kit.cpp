// Just a simple matrix multiplication. At some point, when Kitsune can do more
// sophisticated optimizations, we may want to make this is a full-blown
// benchmark instead of just a test.

#include <iostream>
#include <kitsune.h>

using namespace kitsune;

static void matrix_multiplication(mobile_ptr<double> a, mobile_ptr<double> b,
                                  mobile_ptr<double> c, size_t m, size_t n,
                                  size_t k) {
  // clang-format off
  forall(size_t tid = 0; tid < m * n; tid++) {
    size_t m = tid / n;
    size_t n = tid % n;
    double sum = 0.0;
    for (size_t l = 0; l < k; l++) {
      sum += a[m * k + l] * b[n * k + l];
    }
    c[tid] = sum;
  }
  // clang-format on
}

static size_t check(mobile_ptr<double> a, mobile_ptr<double> b,
                    mobile_ptr<double> c, size_t m, size_t n, size_t k) {
  size_t errors = 0;
  for (size_t tid = 0; tid < m * n; tid++) {
    size_t m = tid / n;
    size_t n = tid % n;
    double sum = 0.0;
    for (size_t l = 0; l < k; l++) {
      sum += a[m * k + l] * b[n * k + l];
    }
    if (c[tid] != sum)
      errors += 1;
  }
  return errors;
}

int main(int argc, char *argv[]) {
  size_t m = 128, n = 128, k = 32;

  if (argc > 4) {
    std::cerr << "Usage: " << argv[0] << " M N K [iteration]\n\n";
    return 1;
  }
  if (argc > 3)
    k = atoi(argv[3]);
  if (argc > 2)
    n = atoi(argv[2]);
  if (argc > 1)
    m = atoi(argv[1]);

  mobile_ptr<double> a(m * k);
  mobile_ptr<double> b(n * k);
  mobile_ptr<double> c(m * n);

  matrix_multiplication(a, b, c, m, n, k);

  std::cout << "\n  Checking final result..." << std::flush;
  size_t errors = check(a, b, c, m, n, k);
  if (errors)
    std::cout << "  FAIL! (" << errors << " errors found)\n\n";
  else
    std::cout << "  pass\n\n";

  a.free();
  b.free();
  c.free();

  return errors;
}
